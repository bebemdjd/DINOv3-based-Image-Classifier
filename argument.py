import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F



def get_medical_augmentation_config(
    img_size: int = 768,
    mixup: float = 0.0, # Classification tasks usually do not recommend using MixUp, especially for medical images
    hsv_h: float = 0.005, # Slight hue perturbation
    hsv_s: float = 0.3, # Moderate saturation perturbation
    hsv_v: float = 0.2, # Moderate brightness perturbation
    degrees: float = 5.0, # Small angle rotation
    translate: float = 0.05, # Slight translation
    scale: float = 0.1, # Small range scaling
    shear: float = 2.0, # Slight shearing
    perspective: float = 0.0, # Avoid perspective transformation for classification tasks
    flipud: float = 0.0, # Up-down flip disabled
    fliplr: float = 0.5, # Keep left-right flip
    crop_fraction: float = 1.0, # Disable random cropping (keep entire image)
    erasing: float = 0.0 # Disable erasing by default (unless strong regularization)

) -> dict:
    """
    Get YOLO data augmentation configuration optimized for medical images.

    This function encapsulates the data augmentation parameter configuration for medical images such as gastroscopy,
    considering the characteristics of medical images (such as color stability, geometric consistency, etc.).

    Args:
        close_mosaic (float): Close Mosaic in the last N epochs to improve convergence. Default 10
        mosaic (float): Mosaic augmentation probability. Default 1.0
        mixup (float): MixUp augmentation probability. Default 0.15 (moderate use recommended for medical images)
        hsv_h (float): Hue jitter range (0.0-1.0). Default 0.01 (smaller recommended for medical images)
        hsv_s (float): Saturation jitter range (0.0-1.0). Default 0.5 (moderate enhancement)
        hsv_v (float): Brightness jitter range (0.0-1.0). Default 0.3 (moderate enhancement)
        degrees (float): Rotation angle range (+/- deg). Default 10.0
        translate (float): Translation range (fraction of image size). Default 0.1
        scale (float): Scale gain (0.5=0.5-1.5x). Default 0.5
        shear (float): Shear angle (+/- deg). Default 5.0
        perspective (float): Perspective transformation strength (0.0-0.001). Default 0.0005
        flipud (float): Up-down flip probability. Default 0.0 (usually not recommended for medical images)
        fliplr (float): Left-right flip probability. Default 0.5
        crop_fraction (float): Image cropping fraction (0.1-1.0). Default 1.0
        erasing (float): Random erasing probability. Default 0.4
        **kwargs: Other additional parameters

    Returns:
        Dict[str, Any]: Configuration dictionary containing all data augmentation parameters

    Examples:
        >>> # Use default configuration
        >>> config = get_medical_augmentation_config()
        >>>
        >>> # Custom configuration
        >>> config = get_medical_augmentation_config(
        ...     mosaic=0.8,
        ...     mixup=0.1,
        ...     degrees=5.0,
        ...     hsv_h=0.005
        ... )
        >>>
        >>> # For YOLO training
        >>> model.train(data="dataset.yaml", **config)
    """
    return {
        # Mosaic and MixUp augmentation
        "mixup": mixup,
        'img_size':img_size,
        # Color augmentation (HSV space)
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,

        # Geometric transformation augmentation
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "flipud": flipud,
        "fliplr": fliplr,

        # Cropping and occlusion augmentation
        "crop_fraction": crop_fraction,
        "erasing": erasing,
    }






def get_image_augmentor(config: dict):
    """
    Return an augmentation function for medical images based on the augmentation configuration, suitable for classification tasks.

    Args:
        config (dict): Configuration dictionary from get_medical_augmentation_config, should contain 'img_size'

    Returns:
        callable: Input PIL image, output augmented PIL image
    """

    transform_list = []

    # HSV jitter (color augmentation)
    if config['hsv_h'] > 0 or config['hsv_s'] > 0 or config['hsv_v'] > 0:
        transform_list.append(T.ColorJitter(
            brightness=config['hsv_v'],
            saturation=config['hsv_s'],
            hue=config['hsv_h']
        ))

    # Random geometric transformation (affine)
    if config['degrees'] > 0 or config['translate'] > 0 or config['scale'] > 0 or config['shear'] > 0:
        transform_list.append(T.RandomAffine(
            degrees=config['degrees'],
            translate=(config['translate'], config['translate']),
            scale=(1 - config['scale'], 1 + config['scale']),
            shear=config['shear']
        ))

    # Perspective transformation
    if config['perspective'] > 0:
        transform_list.append(T.RandomPerspective(
            distortion_scale=config['perspective'],
            p=0.5
        ))

    # Flipping
    if config['fliplr'] > 0:
        transform_list.append(T.RandomHorizontalFlip(p=config['fliplr']))
    if config['flipud'] > 0:
        transform_list.append(T.RandomVerticalFlip(p=config['flipud']))

    # Convert to Tensor before random erasing
    transform_list.append(T.ToTensor())
    if config['erasing'] > 0:
        transform_list.append(T.RandomErasing(
            p=config['erasing'],
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0
        ))
    transform_list.append(T.ToPILImage())

    return T.Compose(transform_list)







