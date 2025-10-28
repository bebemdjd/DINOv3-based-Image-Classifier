import os
import json
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple, Dict, Any
import torch
import torch.nn.functional as F
from torch.multiprocessing.spawn import spawn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from data_sampler import DistributedWeightedSampler # Import balanced sampler
from classifier_dataset import ClassifierDataset
from LinearClassifier import LinearClassifier, ModelWithIntermediateLayers, create_linear_input
from dinov3.hub.backbones import dinov3_vitb16
from Classification_Metrics import accuracy
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_recall
from functools import partial



class CombinedLoss(nn.Module):
    """Combined loss function: Cross-entropy + Supervised Contrastive Loss (SDC)"""
    
    def __init__(self, ce_weight=1.0, sdc_weight=0.1, label_smoothing=0.0, temperature=0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.sdc_weight = sdc_weight
        self.temperature = temperature
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def supervised_contrastive_loss(self, features, labels):
        """
        Supervised Contrastive Loss (SDC)
        Encourage features of same class samples to be closer, features of different class samples to be farther
        """
        # features: [batch_size, feature_dim]
        # labels: [batch_size]
        
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label mask
        labels = labels.unsqueeze(1)  # [batch_size, 1]
        mask = torch.eq(labels, labels.T).float()  # [batch_size, batch_size]
        
        # Remove self-similarity
        mask = mask - torch.eye(batch_size, device=device)
        
        # Calculate number of positive and negative samples
        pos_count = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        neg_count = batch_size - 1 - pos_count.squeeze(1)  # [batch_size]
        
        # Avoid division by zero
        pos_count = torch.clamp(pos_count, min=1e-8)
        
        # Compute log probabilities
        # For positive samples: log(exp(sim)/sum(exp(sim)))
        # For negative samples: -log(1 - exp(sim)/sum(exp(sim)))
        
        # Numerically stable computation
        exp_sim = torch.exp(similarity_matrix)  # [batch_size, batch_size]
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Positive sample loss: -log(positive similarity / total similarity)
        pos_loss = -torch.log(exp_sim / exp_sim_sum) * mask  # [batch_size, batch_size]
        pos_loss = pos_loss.sum(dim=1) / pos_count.squeeze(1)  # [batch_size]
        
        # Negative sample loss: -log(1 - negative similarity / total similarity)
        neg_mask = 1 - mask - torch.eye(batch_size, device=device)  # Negative sample mask
        neg_loss = -torch.log(1 - exp_sim / exp_sim_sum + 1e-8) * neg_mask
        neg_loss = neg_loss.sum(dim=1) / torch.clamp(neg_count, min=1e-8)
        
        # Total loss
        total_loss = (pos_loss + neg_loss).mean()
        
        return total_loss
    
    def forward(self, outputs, labels, features=None):
        """
        Args:
            outputs: Model output logits [batch_size, num_classes]
            labels: True labels [batch_size]
            features: Feature vectors [batch_size, feature_dim] (optional, for SDC)
        """
        loss = 0.0
        
        # Cross-entropy loss
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(outputs, labels)
            loss += self.ce_weight * ce_loss
        
        # SDC loss
        if self.sdc_weight > 0 and features is not None:
            sdc_loss = self.supervised_contrastive_loss(features, labels)
            loss += self.sdc_weight * sdc_loss
        
        return loss



class EM_SupervisedContrastiveLoss(nn.Module):
    """
    EM version of Supervised Contrastive Loss (Expectation-Maximization Supervised Contrastive Loss)

    Drawing inspiration from the EM algorithm:
    - E-step: Calculate "responsibility/soft assignment" (posterior probability) of samples to each class
    - M-step: Update class centers to make true class features closer to corresponding centers, false class features farther away
    """

    def __init__(self, num_classes, feature_dim, temperature=0.1, ce_weight=1.0, sdc_weight=0.1,
                 similarity_type='dot_product', update_centers=True):
        """
        Args:
            num_classes: Number of classes
            feature_dim: Feature dimension
            temperature: Temperature parameter
            ce_weight: Cross-entropy loss weight
            sdc_weight: SDC loss weight
            similarity_type: Similarity type ('dot_product', 'cosine', 'euclidean')
            update_centers: Whether to update class centers in M-step
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.sdc_weight = sdc_weight
        self.similarity_type = similarity_type
        self.update_centers = update_centers

        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss()

        # Class center parameters (learnable) - use better initialization
        # Use small random values instead of standard normal distribution
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.01)

        # Normalization parameters (for cosine similarity)
        if similarity_type == 'cosine':
            self.feature_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
            self.center_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)

    def compute_similarity(self, features, centers):
        """Compute similarity between features and class centers"""
        if self.similarity_type == 'dot_product':
            # Dot product similarity
            return features @ centers.T
        elif self.similarity_type == 'cosine':
            # Cosine similarity
            features_norm = self.feature_norm(features)
            centers_norm = self.center_norm(centers)
            return features_norm @ centers_norm.T
        elif self.similarity_type == 'euclidean':
            # Negative Euclidean distance (smaller distance means higher similarity)
            features_expanded = features.unsqueeze(1)  # [B, 1, D]
            centers_expanded = centers.unsqueeze(0)    # [1, K, D]
            distances = torch.sqrt(torch.sum((features_expanded - centers_expanded) ** 2, dim=2))  # [B, K]
            return -distances  # Negative distance as similarity
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")

    def expectation_step(self, features):
        """
        E-step: Calculate responsibility (soft assignment/posterior probability) of each sample to each class
        """
        # Compute similarity matrix
        similarities = self.compute_similarity(features, self.centers)  # [B, K]

        # Compute responsibilities (posterior probabilities)
        responsibilities = torch.softmax(similarities / self.temperature, dim=1)  # [B, K]

        return similarities, responsibilities

    def maximization_step(self, features, labels, responsibilities):
        """
        M-step: Update class centers (optional)
        Use combination of soft responsibilities and true labels to update class centers
        """
        if not self.update_centers:
            return

        with torch.no_grad():
            for k in range(self.num_classes):
                # Get soft responsibilities of samples belonging to class k
                resp_k = responsibilities[:, k]  # [B]

                # Supervision constraint: Only samples with true label k contribute to center update of class k
                mask = (labels == k).float()  # [B]

                # Combined weights: soft responsibilities × supervision constraint
                weights = resp_k * mask  # [B]

                # Weighted average update class center
                if weights.sum() > 0:
                    # Normalize weights
                    weights = weights / (weights.sum() + 1e-8)

                    # Weighted feature average
                    center_update = torch.sum(weights.unsqueeze(1) * features, dim=0)  # [D]

                    # Momentum update
                    momentum = 0.9
                    self.centers[k] = momentum * self.centers[k] + (1 - momentum) * center_update

    def forward(self, outputs, labels, features):
        """
        Forward propagation: Calculate EM version of SDC loss

        Args:
            outputs: Model classification output logits [B, num_classes]
            labels: True labels [B]
            features: Feature vectors [B, feature_dim]

        Returns:
            loss: Total loss (CE + SDC)
        """
        loss = 0.0

        # Cross-entropy loss
        if self.ce_weight > 0:
            loss_ce = self.ce_loss(outputs, labels)
            loss += self.ce_weight * loss_ce

        # EM version of SDC loss
        if self.sdc_weight > 0:
            # E-step: Calculate responsibilities
            similarities, responsibilities = self.expectation_step(features)

            # Calculate SDC loss: Minimize true class responsibilities (equivalent to maximizing true class similarities)
            true_class_probs = responsibilities[torch.arange(len(labels)), labels]  # [B]
            loss_sdc = -torch.log(true_class_probs + 1e-8).mean()

            loss += self.sdc_weight * loss_sdc

            # M-step: Update class centers (optional)
            if self.training and self.update_centers:
                self.maximization_step(features, labels, responsibilities)

        return loss

    def get_centers(self):
        """Get current class centers"""
        return self.centers.detach().clone()



class Trainer:
    def __init__(self, config):
        self.config = config
        
        # DDP initialization
        self.use_ddp = config.get('DDP', False)
        if self.use_ddp:
            self.local_rank = config.get('local_rank', 0)
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.is_main_process = (self.rank == 0)
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.device = torch.device(config['device'])
            self.is_main_process = True
        
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Create output directory (only in main process)
        self.output_dir = Path(config['output_dir'])
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Synchronize all processes
        if self.use_ddp:
            dist.barrier()
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Save config (only in main process)
        if self.is_main_process:
            with open(self.output_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False, default=str)
            self.logger.info(f"Config saved to {self.output_dir / 'config.json'}")
        
        # Build datasets and data loaders
        self.train_loader, self.val_loader = self._build_dataloaders()

        self.model, self.criterion = self._build_model(config)

        self.criterion.to(self.device)

        # Build optimizer and learning rate scheduler
        self.optimizer, self.lr_scheduler = self._build_optimizer()
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0  # Add best accuracy tracking
        self.global_step = 0
        
        # Resume training (if checkpoint exists)
        if config.get('resume'):
            self._load_checkpoint(config['resume'])


      
    def _build_optimizer(self):
        """Build optimizer and learning rate scheduler"""
        # Only optimize linear classifier parameters (backbone is frozen)
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Select optimizer based on configuration
        if self.config.get('optimizer', 'SGD') == 'SGD':
            optimizer = torch.optim.SGD(
                params,
                lr=self.config['lr'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config['weight_decay']
            )
            if self.is_main_process:
                print(f"Using SGD optimizer (lr={self.config['lr']}, momentum={self.config.get('momentum', 0.9)})")
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999)
            )
            if self.is_main_process:
                print(f"Using AdamW optimizer (lr={self.config['lr']})")
        
        # Learning rate scheduler
        if self.config['lr_scheduler'] == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config['epochs'],
                eta_min=self.config['lr'] * 0.01
            )
        elif self.config['lr_scheduler'] == 'step':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(self.config['epochs'] * 0.6), int(self.config['epochs'] * 0.8)],
                gamma=0.1
            )
        else:
            lr_scheduler = None
        
        return optimizer, lr_scheduler
    
    def _load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
        if self.is_main_process:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        # In DDP mode, map_location needs to map to current GPU
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.use_ddp else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # In DDP mode, need to load to module
        if self.use_ddp:
            # Type hint: DDP wrapped model has module attribute
            assert isinstance(self.model, DDP), "Model should be wrapped with DDP"
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)  # Load best accuracy
        self.global_step = checkpoint.get('global_step', 0)
        
        if self.is_main_process:
            print(f"Resumed from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint (only save in main process)"""
        # Only save in main process
        if not self.is_main_process:
            return
        
        # In DDP mode, need to get original model's state_dict
        if self.use_ddp and isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,  # Save best accuracy
            'global_step': self.global_step,
            'config': self.config
        }
        if self.lr_scheduler:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save latest model
        last_path = self.output_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save epoch model periodically
        if (epoch + 1) % self.config.get('save_interval', 10) == 0:
            epoch_path = self.output_dir / f'epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)

    def _setup_logger(self):
        """Configure logger"""
        logger = logging.getLogger('DINOv3_Detector')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create log file handler
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create log formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logger initialized. Log file: {log_file}")
        return logger


    def _build_dataloaders(self):
        """
        构建数据加载器,支持DDP的DistributedSampler，
        训练集加权平均采样，验证集正常采样
        """
        if self.is_main_process:
            print("Building dataloaders...")
        
        train_set = ClassifierDataset(
            self.config['train_images_dir'], 
            img_size=self.config['img_size'],
            is_argument=True
        )
        val_set = ClassifierDataset(
            self.config['val_images_dir'], 
            img_size=self.config['img_size'],
            is_argument=False
        )
        
        # Use DistributedSampler in DDP mode
        if self.use_ddp:
            # Perform weighted sampling for functions
            train_sampler = DistributedWeightedSampler(
                train_set._compute_sample_weights(),
                num_replicas=self.world_size,
                rank=self.rank,
            )
            # Note: Do not use WeightedSampler here because validation set doesn't need balanced sampling
            val_sampler = DistributedSampler( 
                val_set,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config['batch_size'], 
            sampler=train_sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.config.get('prefetch_factor', 2),
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config['batch_size'], 
            sampler=val_sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            prefetch_factor=self.config.get('prefetch_factor', 2),
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        # Save samplers for setting epoch during training
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        
        if self.is_main_process:
            print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
            if self.use_ddp:
                print(f"Using DDP with {self.world_size} GPUs")
        
        return train_loader, val_loader


    def _build_model(self, config):
        """Build model with DDP support - using official DINOv3 linear classifier"""
        if self.is_main_process:
            print("Building model...")
        
        # 1. Load DINOv3 backbone
        backbone = dinov3_vitb16(pretrained=False)
        
        # Load pretrained weights
        if config['backbone_checkpoint'] and os.path.exists(config['backbone_checkpoint']):
            if self.is_main_process:
                print(f"Loading backbone checkpoint from {config['backbone_checkpoint']}")
            checkpoint = torch.load(config['backbone_checkpoint'], map_location='cpu')
            backbone.load_state_dict(checkpoint, strict=True)
        
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        
        # 2. Create feature extractor (with intermediate layer outputs)
        autocast_dtype = torch.float16 if config['use_amp'] else torch.float32
        autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=config['use_amp'], dtype=autocast_dtype)
        
        feature_model = ModelWithIntermediateLayers(
            feature_model=backbone,
            n=config['n_last_blocks'],  # Use last N layers
            autocast_ctx=autocast_ctx,
            reshape=False,
            return_class_token=True
        ).to(self.device)
        
        # 3. Calculate linear layer input dimension
        embed_dim = 768  # ViT-B/16 embedding dimension
        if config['use_avgpool']:
            # If using avgpool: (n_last_blocks + 1) × embed_dim
            linear_input_dim = (config['n_last_blocks'] + 1) * embed_dim
        else:
            # Not using avgpool: n_last_blocks × embed_dim
            linear_input_dim = config['n_last_blocks'] * embed_dim
        
        # 4. Create linear classifier
        linear_classifier = LinearClassifier(
            out_dim=linear_input_dim,
            use_n_blocks=config['n_last_blocks'],
            use_avgpool=config['use_avgpool'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        # 5. Combine models
        class CombinedModel(nn.Module):
            def __init__(self, feature_model, linear_classifier):
                super().__init__()
                self.feature_model = feature_model
                self.linear_classifier = linear_classifier
            
            def forward(self, images, return_features=False):
                # Extract features
                features = self.feature_model(images)
                # Classify
                logits = self.linear_classifier(features)
                
                if return_features:
                    # Return logits and feature tensor for contrastive loss
                    features_flat = create_linear_input(
                        features,
                        self.linear_classifier.use_n_blocks,
                        self.linear_classifier.use_avgpool,
                    )
                    return logits, features_flat
                return logits
        
        model = CombinedModel(feature_model, linear_classifier)
        
        # 6. Create loss function (supports traditional SDC and EM version SDC)
        sdc_type = config.get('sdc_type', 'original')  # 'original' or 'em'
        
        if sdc_type == 'em':
            # EM version of SDC
            criterion = EM_SupervisedContrastiveLoss(
                num_classes=config['num_classes'],
                feature_dim=linear_input_dim,
                temperature=config.get('sdc_temperature', 0.1),
                ce_weight=config.get('ce_weight', 1.0),
                sdc_weight=config.get('sdc_weight', 0.1),
                similarity_type=config.get('sdc_similarity_type', 'dot_product'),
                update_centers=config.get('sdc_update_centers', True)
            )
            if self.is_main_process:
                print(f"Using EM-SDC loss with {config.get('sdc_similarity_type', 'dot_product')} similarity")
        else:
            # Traditional SDC
            criterion = CombinedLoss(
                ce_weight=config.get('ce_weight', 1.0),
                sdc_weight=config.get('sdc_weight', 0.1),
                label_smoothing=config.get('label_smoothing', 0.0),
                temperature=config.get('sdc_temperature', 0.1)
            )
            if self.is_main_process:
                print(f"Using original SDC loss")
        
        # Count parameters (only print in main process)
        if self.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("=" * 100)
            print(f"Model Architecture:")
            print(f"  Backbone: DINOv3 ViT-B/16 (frozen)")
            print(f"  Feature layers: last {config['n_last_blocks']} blocks")
            print(f"  Use avgpool: {config['use_avgpool']}")
            print(f"  Linear input dim: {linear_input_dim}")
            print(f"  Num classes: {config['num_classes']}")
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_params:,}")
            print("=" * 100)
        
        # If using DDP, wrap model
        if self.use_ddp:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Linear classifier doesn't need it
                broadcast_buffers=True
            )
            if self.is_main_process:
                print(f"Model wrapped with DDP on GPU {self.local_rank}")
        else:
            print("Using single GPU")
        
        return model, criterion


    def train_epoch(self, epoch) -> Tuple[float, Dict[str, float]]:

        self.model.train()
        
        # Set sampler epoch in DDP mode to ensure different data shuffle per epoch
        if self.use_ddp and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        # Show progress bar only in main process
        pbar: Union[tqdm, DataLoader]  # Type hint
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        else:
            pbar = self.train_loader
        
        epoch_loss = 0.0
        epoch_losses = {}
        
        # Gradient accumulation settings
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)

        for batch_idx, batch in enumerate(pbar):
            images = batch['image']
            labels = batch['label']
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # Model forward propagation
            with autocast(device_type='cuda', enabled=self.config['use_amp']):
                # Check if feature vectors are needed (for EM-SDC or losses with SDC weight)
                need_features = (self.config.get('sdc_weight', 0.0) > 0 or 
                               isinstance(self.criterion, EM_SupervisedContrastiveLoss))
                
                if need_features:
                    # If using SDC, need to return both logits and features
                    outputs, features = self.model(images, return_features=True)
                    loss = self.criterion(outputs, labels, features=features)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            
            # Scale loss to support gradient accumulation
            scaled_loss = loss / accumulation_steps

            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()


            
            # Update parameters only when accumulation steps are reached
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler:
                    # Gradient clipping
                    if self.config.get('grad_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.get('grad_clip', 0) > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
            
            # Record loss
            epoch_loss += loss.item()

            
            # Update progress bar (only in main process)
            if self.is_main_process and isinstance(pbar, tqdm):
                # Show actual loss (not scaled)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    'acc_step': f"{(batch_idx % accumulation_steps) + 1}/{accumulation_steps}"
                })
            
            # Logging (only in main process)
            if self.is_main_process and batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                # Also print to console and flush
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                import sys
                sys.stdout.flush()
            
            self.global_step += 1
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_loader)
        avg_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}
        
        # In DDP mode, need to synchronize losses across all processes
        if self.use_ddp:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / self.world_size).item()
            
            for k in avg_losses:
                loss_tensor = torch.tensor(avg_losses[k], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_losses[k] = (loss_tensor / self.world_size).item()
        
        return avg_loss, avg_losses



    @torch.no_grad()
    def validate(self, epoch) -> Tuple[float, Dict[str, Any]]:
        """验证
        
        与 train_epoch 保持一致的数据流和目标格式准备，
        并计算准确率、F1分数、召回率等评价指标
        
        Args:
            epoch: 当前训练的epoch数
            
        Returns:
            avg_loss: Average loss
            metrics_results: Dictionary containing all evaluation metrics
        """
        # Check if validation is needed in current epoch
        val_epoch_interval = self.config.get('val_epoch_interval', 1)
        if (epoch + 1) % val_epoch_interval != 0:
            # No validation needed, return empty results
            return float('inf'), {}
        
        self.model.eval()
        
        # Show progress bar only in main process
        pbar: Union[tqdm, DataLoader]  # Type hint
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc="Validation")
        else:
            pbar = self.val_loader
        
        epoch_loss = 0.0
        
        # For calculating Top-K accuracy
        all_outputs = []
        all_labels = []
        
        for batch in pbar:

            images = batch['image']
            labels = batch['label']
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Model forward propagation
            with autocast(device_type='cuda', enabled=self.config['use_amp']):
                # Check if feature vectors are needed (for EM-SDC or losses with SDC weight)
                need_features = (self.config.get('sdc_weight', 0.0) > 0 or 
                               isinstance(self.criterion, EM_SupervisedContrastiveLoss))
                
                if need_features:
                    outputs, features = self.model(images, return_features=True)
                    loss = self.criterion(outputs, labels, features=features)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            
            # Record loss
            epoch_loss += loss.item()
            
            # Collect predictions for metric calculation
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            # Update progress bar only in main process
            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})


        avg_loss = epoch_loss / len(self.val_loader)
        
        # In DDP mode, need to synchronize losses and predictions across all processes
        if self.use_ddp:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / self.world_size).item()

            # Merge predictions from all processes (for calculating global metrics)
            if len(all_outputs) > 0:
                all_outputs_local = torch.cat(all_outputs, dim=0).to(self.device)
                all_labels_local = torch.cat(all_labels, dim=0).to(self.device)
            else:
                all_outputs_local = torch.empty(0, self.config['num_classes'], device=self.device)
                all_labels_local = torch.empty(0, dtype=torch.long, device=self.device)

            local_count = torch.tensor([all_outputs_local.size(0)], device=self.device, dtype=torch.long)
            all_counts = [torch.zeros_like(local_count) for _ in range(self.world_size)]
            dist.all_gather(all_counts, local_count)
            max_count = int(torch.stack(all_counts).max().item())

            # If different processes have different sample counts, need padding for all_gather
            if all_outputs_local.size(0) < max_count:
                pad_size = max_count - all_outputs_local.size(0)
                if all_outputs_local.dim() == 2:
                    pad_outputs = torch.zeros((pad_size, all_outputs_local.size(1)), device=self.device, dtype=all_outputs_local.dtype)
                else:
                    pad_outputs = torch.zeros((pad_size,), device=self.device, dtype=all_outputs_local.dtype)
                all_outputs_local = torch.cat([all_outputs_local, pad_outputs], dim=0)

                pad_labels = torch.zeros((pad_size,), device=self.device, dtype=all_labels_local.dtype)
                all_labels_local = torch.cat([all_labels_local, pad_labels], dim=0)

            gathered_outputs = [torch.zeros_like(all_outputs_local) for _ in range(self.world_size)]
            gathered_labels = [torch.zeros_like(all_labels_local) for _ in range(self.world_size)]

            dist.all_gather(gathered_outputs, all_outputs_local)
            dist.all_gather(gathered_labels, all_labels_local)

            if self.is_main_process:
                # Trim padding based on actual sample counts
                trimmed_outputs = []
                trimmed_labels = []
                for out_tensor, label_tensor, count_tensor in zip(gathered_outputs, gathered_labels, all_counts):
                    count = int(count_tensor.item())
                    trimmed_outputs.append(out_tensor[:count].cpu())
                    trimmed_labels.append(label_tensor[:count].cpu())
                all_outputs = trimmed_outputs
                all_labels = trimmed_labels
        else:
            # Non-DDP mode, use collected data directly
            pass
        
        metrics_results: Dict[str, Any] = {}
        if self.is_main_process and len(all_outputs) > 0:
            all_outputs_tensor = torch.cat(all_outputs, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)

            # Calculate Top-K accuracy (in percentage form)
            top1_acc = accuracy(all_outputs_tensor, all_labels_tensor, topk=(1,))[0]
            metrics_results['top1_accuracy'] = float(top1_acc)

            if self.config['num_classes'] >= 3:
                top3_acc = accuracy(all_outputs_tensor, all_labels_tensor, topk=(3,))[0]
                metrics_results['top3_accuracy'] = float(top3_acc)

            # Calculate micro-average metrics (consistent with original configuration)
            num_classes = self.config['num_classes']
            micro_recall = multiclass_recall(
                all_outputs_tensor,
                all_labels_tensor,
                num_classes=num_classes,
                average='micro'
            )
            micro_f1 = multiclass_f1_score(
                all_outputs_tensor,
                all_labels_tensor,
                num_classes=num_classes,
                average='micro'
            )

            metrics_results['recall'] = float(micro_recall)
            metrics_results['f1'] = float(micro_f1)
            metrics_results['accuracy'] = float(top1_acc / 100.0)
        
        # Logging (only in main process)
        if self.is_main_process:
            self.logger.info(f"Epoch {epoch+1} Validation - Loss: {avg_loss:.4f}")
            
            # Record evaluation metrics
            self.logger.info("Validation Metrics:")
            for metric_name, metric_value in metrics_results.items():
                if metric_name.startswith('top'):
                    self.logger.info(f"  {metric_name}: {metric_value:.2f}%")
                else:
                    self.logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Also print key metrics to console
            print(f"\nValidation Metrics:")
            if not metrics_results:
                print("  (no validation data gathered)")
            else:
                for metric_name, metric_value in metrics_results.items():
                    if metric_name.startswith('top'):
                        print(f"  {metric_name.replace('_', ' ').title()}: {metric_value:.2f}%")
                    else:
                        print(f"  {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
            import sys
            sys.stdout.flush()
        
        return avg_loss, metrics_results  # Return metrics dict instead of avg_losses



    def train(self):
        """Main training loop"""
        if self.is_main_process:
            print("Starting training...")
            print(f"Output directory: {self.output_dir}")
            val_epoch_interval = self.config.get('val_epoch_interval', 1)
            print(f"Validation interval: every {val_epoch_interval} epochs")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            print(f"\n=== Starting Epoch {epoch+1}/{self.config['epochs']} ===")
            import sys
            sys.stdout.flush()
            # Training
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Validate every val_epoch_interval epochs
            val_loss, val_metrics = self.validate(epoch)
            
            # Check and save best model (only save if actually validated)
            is_best = False
            if self.is_main_process and val_metrics:  # val_metrics not empty means validation was performed
                current_acc = val_metrics.get('top1_accuracy', val_metrics.get('acc', 0.0))
                
                if current_acc > self.best_val_acc:
                    self.best_val_acc = current_acc
                    is_best = True
                    print(f"  🎉 New best accuracy: {self.best_val_acc:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                
                self._save_checkpoint(epoch, is_best)
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Print information (only in main process)
            if self.is_main_process:
                print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                for k in train_losses:
                    print(f"    {k}: {train_losses[k]:.4f}")
                print(f"  Current Best Accuracy: {self.best_val_acc:.4f}")
            
            # DDP synchronization
            if self.use_ddp:
                dist.barrier()
            
            # Regularly clear GPU cache
            if self.config.get('empty_cache_interval', 0) > 0:
                if (epoch + 1) % self.config['empty_cache_interval'] == 0:
                    torch.cuda.empty_cache()
                    if self.is_main_process:
                        print(f"  GPU cache cleared")
            
            # Clear GPU memory after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.is_main_process:
                    print(f"  GPU cache cleared after epoch {epoch+1}")
        
        if self.is_main_process:
            print(f"\nTraining completed!")
            print(f"  Best validation accuracy: {self.best_val_acc:.4f}")
            print(f"  Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Training completed! Best val acc: {self.best_val_acc:.4f}, Best val loss: {self.best_val_loss:.4f}")
            
            # Final cleanup
            torch.cuda.empty_cache()




def setup_ddp(rank, world_size):
    """Initialize DDP environment (manual mode)"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Use a fixed port

    # Assign GPU device for each rank
    torch.cuda.set_device(rank)

    # Initialize process group, explicitly specify device_id
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f'cuda:{rank}')
    )
    return True, rank

def cleanup_ddp():
    """Clean up DDP environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def ddp_worker(rank, world_size, config):
    """Entry function for DDP worker process"""
    # Set up DDP environment in each process
    use_ddp, local_rank = setup_ddp(rank, world_size)
    config['DDP'] = use_ddp
    config['local_rank'] = local_rank
    
    # Ensure only main process prints information
    if local_rank != 0:
        # Disable printing for non-main processes
        config['log_interval'] = float('inf')

    try:
        # Create trainer and start training
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"\nTraining failed with error: {e}")
            traceback.print_exc()
    finally:
        # Clean up DDP environment
        cleanup_ddp()

def get_default_config():
    """Get default configuration - single classifier version"""
    return {
        # ============ Data Configuration ============
        'train_images_dir': None,  # Placeholder for training images directory
        'val_images_dir': None,    # Placeholder for validation images directory
        'num_classes': 3,
        'img_size': 768,
        
        # ============ Model Configuration ============
        'backbone_checkpoint': None,  # Placeholder for backbone checkpoint path
        
        # ============ Single Classifier Configuration (Alternative to Multi-Classifier Search) ============
        # These parameters correspond to DINOv3 official linear classifier settings
        'n_last_blocks': 4,           # Use last N layers for feature concatenation (1, 2, 3, 4, ...)
        'use_avgpool': False,          # Whether to use average pooling of patch tokens
        # Feature dimension explanation:
        #   - If use_avgpool=False: feature_dim = n_last_blocks × embed_dim (e.g., 4×768=3072)
        #   - If use_avgpool=True:  feature_dim = (n_last_blocks+1) × embed_dim (e.g., 5×768=3840)
        
        # ============ Training Configuration ============
        'batch_size': 96,
        'epochs': 30,
        'lr': 0.01,                   # Recommended learning rate for linear classifier (0.001, 0.01, 0.1)
        'weight_decay': 0.0,          # Linear classifiers usually don't use weight decay
        'lr_scheduler': 'cosine',     # 'cosine', 'step', or None
        'grad_clip': 1.0,
        'use_amp': False,             # Temporarily disable mixed precision training
        
        # ============ Loss Function Configuration ============
        'label_smoothing': 0.0,       # Label smoothing coefficient (0.0-0.3)
        
        # ============ SDC (Supervised Contrastive Loss) Configuration ============
        'ce_weight': None,             # Cross-entropy loss weight (placeholder)
        'sdc_weight': None,            # SDC loss weight (0.0 means not used) (placeholder)
        'sdc_temperature': None,       # SDC temperature parameter (usually 0.07-0.2) (placeholder)
        'sdc_type': 'em',            # SDC type: 'original' or 'em'
        'sdc_similarity_type': 'cosine',  # EM-SDC similarity type: 'dot_product', 'cosine', 'euclidean'
        'sdc_update_centers': False,     # Whether EM-SDC updates class centers (recommended to set to False first)
        'sdc_temperature': None,         # SDC temperature parameter (reduce slightly) (placeholder)
        'sdc_weight': None,              # SDC loss weight (reduce slightly) (placeholder)
        
        # ============ Optimizer Configuration ============
        # Optional: 'SGD' or 'AdamW'
        'optimizer': 'SGD',           # Recommended SGD for linear classifier
        'momentum': 0.9,              # SGD momentum (only effective when optimizer='SGD')
        
        # ============ Memory Optimization Configuration ============
        'gradient_accumulation_steps': 1,  # Gradient accumulation steps
        'empty_cache_interval': 10,       # Clear cache every N epochs
        
        # ============ Data Loading Configuration ============
        'num_workers': 4,
        'prefetch_factor': 2,
        
        # ============ Logging and Saving Configuration ============
        'output_dir': f'runs/linear_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 100,          # Log every N iterations
        'save_interval': 20,          # Save model every N epochs
        'val_epoch_interval': 1,      # Validate every N epochs
        'resume': None,               # Path to resume training checkpoint
        
        # ============ Device Configuration ============
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # ============ Parallel Computing Configuration ============
        'DDP': True,                  # Whether to use Distributed Data Parallel
        'nproc_per_node': 2,          # Number of processes per node
    }


def main():
    """Main function: Initialize config, set up DDP environment, start training"""
    
    # Get default configuration
    config = get_default_config()
    
    # Check if to use DDP
    use_ddp = config['DDP']
    
    if use_ddp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        print(f"Spawning {world_size} DDP processes...")
        
        # Use spawn to start DDP processes
        spawn(
            ddp_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Run single GPU or CPU mode
        print("Running in single-process mode (DDP disabled).")
        config['DDP'] = False
        try:
            trainer = Trainer(config)
            trainer.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("\nTraining finished!")

if __name__ == '__main__':
    # Set multiprocessing start method, required for CUDA in spawn mode
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()