from torch.utils.data import Sampler
import torch
import math


# Create weighted sampler to avoid imbalance between class numbers, using replacement sampling
class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_samples=None, num_replicas=None, rank=None, replacement=True, seed=42):
        """
        Fix sampler issues
        replacement=True for replacement sampling, avoid sampling count exceeding weight array length
        """
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        
        self.weights = weights.float()
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.weights)
        self.epoch = 0
        self.seed = seed
        self.replacement = replacement  # Changed to True
        
        # Fix: Ensure each process gets a reasonable number of samples
        if num_samples is None:
            # Round up to ensure all samples are sampled
            self.num_samples = math.ceil(len(self.weights) / self.num_replicas)
        else:
            self.num_samples = num_samples

    def __iter__(self):
        # Use epoch-specific seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        probs = self.weights / self.weights.sum()
        
        # Fix: When using non-replacement sampling, ensure sampling count does not exceed weight array length
        total_samples_needed = self.num_samples * self.num_replicas
        
        if not self.replacement and total_samples_needed > len(self.weights):
            # If sampling count exceeds weight array length, switch to replacement sampling
            print(f"Rank {self.rank}: Sampling count ({total_samples_needed}) exceeds weight array length ({len(self.weights)}), switching to replacement sampling")
            self.replacement = True
        
        sampled_indices = torch.multinomial(
            probs, 
            total_samples_needed, 
            self.replacement, 
            generator=g
        ).tolist()
        
        # Allocate indices for current process
        indices = sampled_indices[self.rank:total_samples_needed:self.num_replicas]
        
        # Ensure correct number of indices are returned
        if len(indices) < self.num_samples:
            # If not enough, resample to supplement
            additional_needed = self.num_samples - len(indices)
            
            # Fix: Ensure supplementary sampling also does not exceed weight array length
            if not self.replacement and additional_needed > len(self.weights):
                additional_needed = len(self.weights)
            
            additional_indices = torch.multinomial(
                probs, additional_needed, self.replacement, generator=g
            ).tolist()
            indices.extend(additional_indices)
        
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch






"""
Usage example
# Create distributed weighted sampler
train_sampler = DistributedWeightedSampler(
    train_dataset.sample_weights,
    num_replicas=world_size,
    rank=rank)
"""


    































