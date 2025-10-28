import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.v2 as v2
from argument import get_medical_augmentation_config , get_image_augmentor




class ClassifierDataset(Dataset):
    def __init__(self, images_dir, img_size=768,is_argument=False) -> None:
        """
        Args:
            images_dir (str): Image folder path
            img_size (int, optional): Image predetermined size
            is_argument (bool, optional): Whether to apply data augmentation.
        """
        super().__init__()
        self.images_dir = images_dir
        self.img_size = img_size
        self.is_argument = is_argument
        self.images_dir = images_dir

        # Default data augmentation function, this config needs to be defined in data_argument.py
        augmentation_config = get_medical_augmentation_config(img_size=img_size)
        self.argument_image = get_image_augmentor(augmentation_config)
        self.is_argument = is_argument
        
        # Class to index mapping, a dictionary to map class names (subfolder names) to integer indices. E.g.: {'gastric_cancer': 0, 'polyp': 1, 'ulcer': 2}.
        self.class_to_idx = {}
        self.samples = [] # List to store all samples
        
        # Scan subfolders as categories
        for class_name in os.listdir(images_dir):
            class_dir = os.path.join(images_dir, class_name)
            if os.path.isdir(class_dir):
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                # Collect image files (assuming .jpg and .png formats)
                for img_path in glob.glob(os.path.join(class_dir, '*.jpg')) + glob.glob(os.path.join(class_dir, '*.png')):
                    self.samples.append({'img_path': img_path, 'label': self.class_to_idx[class_name]})
        # Shuffle samples to prevent overfitting
        random.shuffle(self.samples)

        # Add attributes required for Mosaic augmentation
        self.cache = "ram"  # Default not to use buffer
        self.buffer = []    # Buffer is empty

        # Function for weighted sampling
        self._compute_sample_weights()
        print(f"Dataset initialized with {len(self.samples)} valid samples")
        

    def _compute_sample_weights(self):
        """
        Compute sampling weights for each sample (inverse frequency weighting based on class frequency)
        Used to solve class imbalance problems
        """
        print("Computing sample weights for class balancing...")
        
        # Count the class of each sample
        self.sample_classes = [sample['label'] for sample in self.samples]
        
        # Count the number of samples for each class
        unique_classes, class_counts = np.unique(self.sample_classes, return_counts=True)
        class_count_map = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
        
        print(f"Class distribution: {class_count_map}")
        
        # Compute weights for each sample (inverse frequency weighting: weight = total samples / samples of this class)
        # This way, minority class samples have higher weights to help balance the dataset
        N = len(self.samples)
        self.sample_weights = torch.DoubleTensor([
            N / class_count_map[cls] for cls in self.sample_classes
        ])
        
        print(f"Sample weights computed: min={self.sample_weights.min():.4f}, max={self.sample_weights.max():.4f}, mean={self.sample_weights.mean():.4f}")
        
        return self.sample_weights

    def resize_with_padding(self , image: Image.Image, img_size: int) -> Image.Image:
        """
            Resize the image proportionally to img_size, maintain aspect ratio, and fill blank with black.
        Image pasted at top-left corner.
        Args:
            image (PIL.Image): Input image (RGB)
            img_size (int): Target output size (width and height)
        Returns:
            Force-resized image
        """
        original_w, original_h = image.size
        
        new_image = image.resize((img_size, img_size) , Image.BILINEAR) # Force resize image, no longer fill with 0
        
        return new_image

    def _make_transform(self):
        """
        Create image preprocessing pipeline (ImageNet normalization)
        Note: For simplicity, we do not do data augmentation in transform
        
        Args:
        
        """
        transforms_list = []
        
        # Basic conversion
        transforms_list.append(v2.ToImage())
        
        # Normalization
        transforms_list.extend([
            v2.ToDtype(torch.float32, scale=True),  # [0,255] -> [0,1]
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
        return v2.Compose(transforms_list)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        label = sample['label']
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        
        if self.is_argument:
            img = self.argument_image(img)
       
        resize_image = self.resize_with_padding(img, self.img_size)
        # Apply preprocessing pipeline
        transform = self._make_transform()
        processed_image = transform(resize_image)
        
        return {'image': processed_image, 'label': label}



















