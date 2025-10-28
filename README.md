#  DINOv3-based Image Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

</div>

A PyTorch implementation of an image classification system based on the **DINOv3** (self-DIstillation with NO labels) vision transformer. This project provides a complete training pipeline with distributed data parallel (DDP) support, advanced data augmentation, and multiple loss functions including supervised contrastive learning.

---

##  Features

-  **DINOv3 Backbone**: Leverages pre-trained DINOv3 ViT-B/16 for powerful feature extraction
-  **Distributed Training**: Full DDP support for multi-GPU training
-  **Advanced Loss Functions**: 
  - **Combined Cross-Entropy + Supervised Contrastive Loss (SDC)
  - **EM-based Supervised Contrastive Loss with learnable class centers
-  **Smart Data Sampling**: Weighted sampling for handling class imbalance
-  **Medical Image Augmentation**: Specialized augmentation pipeline for medical imaging
-  **Comprehensive Metrics**: Top-K accuracy, F1 score, recall, and more
-  **Gradient Accumulation**: Memory-efficient training for large models

##  Project Structure

```
  argument.py                 # Data augmentation configurations
  Classification_Metrics.py   # Evaluation metrics
  classifier_dataset.py       # Dataset class with weighted sampling
  data_sampler.py            # Distributed weighted sampler
  dense_features_PCA.py      # Feature extraction and PCA visualization
  LinearClassifier.py        # Linear classifier implementation
  train_linear.py            # Main training script
  dinov3/                    # DINOv3 model implementation
  image/                     # Dataset directory
     train/                 # Training images
     val/                   # Validation images
     test/                  # Test images
  pre_weight/                # Pre-trained model weights
```

##  Requirements

```bash
python >= 3.8
torch >= 2.0.0
torchvision >= 0.15.0
torchmetrics
Pillow
tqdm
numpy
matplotlib
scikit-learn
```

 Install dependencies:

```bash
pip install torch torchvision torchmetrics pillow tqdm numpy matplotlib scikit-learn
```

##  Quick Start

### 1 Prepare Your Data

Organize your images in the following structure:

```
image/
 train/
    class1/
    class2/
    class3/
 val/
    class1/
    class2/
    class3/
 test/
     class1/
     class2/
     class3/
```

### 2 Download Pre-trained Weights

Download the DINOv3 ViT-B/16 pre-trained weights and place them in the `pre_weight/` directory.

### 3 Configure Training Parameters

Edit the `get_default_config()` function in `train_linear.py` to set your hyperparameters.

### 4 Train the Model

####  Single GPU Training

```bash
python train_linear.py
```

####  Multi-GPU Training (DDP)

The script automatically detects available GPUs and uses DDP:

```bash
python train_linear.py
```

### 5 Feature Visualization

Extract and visualize features using PCA:

```bash
python dense_features_PCA.py
```

##  Model Architecture

The classifier consists of:

1. **Frozen DINOv3 Backbone**: Pre-trained ViT-B/16 (embedding dim: 768)
2. **Feature Aggregation**: Concatenates features from the last N transformer blocks
3. **Linear Classifier**: Single linear layer for classification

 **Feature dimension calculation:**
- Without avgpool: `n_last_blocks  768` (e.g., 4  768 = 3072)
- With avgpool: `(n_last_blocks + 1)  768` (e.g., 5  768 = 3840)

##  Loss Functions

### 1. Combined Loss (Default)

```python
Loss = α  CrossEntropy + β  SupervisedContrastive
```

- ** Cross-Entropy**: Standard classification loss
- ** Supervised Contrastive**: Encourages same-class features to be closer, different-class features to be farther

### 2. EM-Supervised Contrastive Loss

An expectation-maximization variant with learnable class centers:

- **E-step**: Calculate responsibility (soft assignment) of samples to classes
- **M-step**: Update class centers based on responsibilities
- **Supports multiple similarity metrics**: dot product, cosine, euclidean

##  Data Augmentation

Specialized augmentation for medical images:

-  Random rotation (30)
-  Random horizontal/vertical flip
-  Color jitter (brightness, contrast, saturation)
-  Random affine transformations
-  Gaussian blur
-  Normalization with ImageNet statistics

##  Training Features

###  Class Imbalance Handling

Automatic weighted sampling based on class distribution

###  Gradient Accumulation

For limited GPU memory

###  Mixed Precision Training

Enable automatic mixed precision for faster training

###  Learning Rate Scheduling

Supports cosine annealing and step decay

##  Evaluation Metrics

The training script automatically computes:

-  **Top-1 Accuracy**: Percentage of correct top predictions
-  **Top-3 Accuracy**: Percentage when true class is in top 3 predictions
-  **F1 Score**: Harmonic mean of precision and recall (micro-average)
-  **Recall**: True positive rate (micro-average)

##  Checkpointing

Models are automatically saved:

-  `last.pth`: Latest model checkpoint
-  `best.pth`: Best model based on validation accuracy
-  `epoch_N.pth`: Periodic snapshots every N epochs

##  Performance Tips

1. Batch Size: Start with 96 and adjust based on GPU memory
2. Learning Rate: 0.01 works well for linear classifiers with SGD
3. Feature Layers: Using 4 last blocks (`n_last_blocks=4`) is a good balance
4. Gradient Clipping: Set to 1.0 to prevent gradient explosion
5. Validation Interval: Validate every epoch for small datasets

##  Common Issues

###  Out of Memory (OOM)

-  Reduce `batch_size`
-  Enable gradient accumulation
-  Reduce `img_size`
-  Use fewer feature blocks (`n_last_blocks`)

###  Slow Training

-  Enable DDP for multi-GPU training
-  Increase `num_workers` for data loading
-  Enable `use_amp` for mixed precision

###  Poor Convergence

-  Adjust learning rate
-  Try different optimizers (SGD vs AdamW)
-  Tune SDC loss weight
-  Check data augmentation strength

##  License

This project is released under the **MIT License**.

##  Acknowledgments

-  DINOv3 model from [Meta AI Research](https://github.com/facebookresearch/dinov3)
-  PyTorch team for the excellent deep learning framework
-  The open-source community for various tools and libraries

##  Contact

For questions and feedback, please open an issue on GitHub.

---

<div align="center">

** If you find this project helpful, please consider giving it a star! **

Made with  by the community

</div>
