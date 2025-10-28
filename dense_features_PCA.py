"""
DINOv3 Dense Feature Extraction with PCA Visualization
Extract dense features using DINOv3 and reduce dimensions to RGB visualization via PCA
"""

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import v2
from dinov3.hub.backbones import dinov3_vitb16
import os


def make_transform(resize_size: int = 224):
    """
    Create image preprocessing pipeline (ImageNet normalization)
    
    Args:
        resize_size: Target size, DINOv3 default 224
    """
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)  # [0,255] -> [0,1]
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def get_img_tensor(image_path, resize_size=224):
    """
    Read image and convert to model input tensor
    
    Args:
        image_path: Image path
        resize_size: Target size
    
    Returns:
        image_tensor: [1, 3, H, W]
        Original size: (H_orig, W_orig)
    """
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size  # (H, W)
    transform = make_transform(resize_size=resize_size)
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), orig_size  # Add batch dimension


def extract_intermediate_features(model, image_tensor, num_layers=4):
    """Extract intermediate features from last num_layers layers using hook"""
    intermediate_features = []
    
    def hook_fn(module, input, output):
        # output may be list or tensor, unify to tensor first
        tensor_output = output[0] if isinstance(output, list) else output
        
        # tensor_output shape is [B, N, D], where B=1, N = num_patches + 1
        # We only need patch tokens, remove CLS token and batch dimension
        intermediate_features.append(tensor_output[0, 5:].clone())  # [num_patches, D]
    
    # Register hook to last 4 transformer blocks
    total_blocks = len(model.blocks)
    hooks = []
    for i in range(total_blocks - num_layers, total_blocks):
        hook = model.blocks[i].register_forward_hook(hook_fn)
        hooks.append(hook)
    
    with torch.no_grad():
        _ = model.forward_features(image_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return intermediate_features  # [x_l0, x_l1, x_l2, x_l3] each is [num_patches, D]


def extract_dense_features_with_pca(
    model,
    image_tensor,
    orig_size,
    layer_index=-1,
    img_size=800,  # Input image size
    output_path="dense_features.png",
    use_multi_layer=False  # Whether to use multi-layer average
):
    """
    Extract DINOv3 dense features and visualize via PCA
    
    Args:
        model: DINOv3 model
        image_tensor: Input image tensor [1, 3, H, W]
        orig_size: Original image size (W, H)
        layer_index: Which layer's features to extract (-1=last layer, can also be intermediate layers like 3, 6, 9)
        output_path: Output image path
        use_multi_layer: Whether to use average of last 4 layers
    
    Returns:
        rgb: Visualized RGB image [H, W, 3]
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # 1) Forward propagation to get features
    with torch.no_grad():
        if use_multi_layer:
            # Extract last 4 layers and average
            features = extract_intermediate_features(model, image_tensor, num_layers=4)
            x = torch.stack(features, dim=0).mean(dim=0)  # [num_layers, N, D] -> [N, D]
            print(f"Using average of last {len(features)} layers")
        else:
            # DINOv3 returns dictionary, need to extract x_norm_patchtokens
            outputs = model.forward_features(image_tensor)
            
            print(f"outputs['x_norm_patchtokens'].shape: {outputs['x_norm_patchtokens'].shape}")
            print(f"outputs keys: {list(outputs.keys())}")
            
            # If last layer, directly take patch tokens
            if layer_index == -1:
                # DINOv3 output: outputs['x_norm_patchtokens'] is [B, N, D]
                # where N = (H/patch_size) * (W/patch_size)
                x = outputs['x_norm_patchtokens'][0]  # [N, D] remove batch dimension
                print(f"Feature map spatial size: {int(np.sqrt(x.shape[0]))}x{int(np.sqrt(x.shape[0]))}")
            else:
                # If extracting intermediate layer, need to get during model forward
                # Simplified here, only use last layer
                raise NotImplementedError("Intermediate layer extraction requires modifying model's forward method")
    
    # 2) Restore to grid and normalize
    N, D = x.shape
    print(f"Feature map shape: {N}, {D}")
    patch_size = 16  # DINOv3 ViT-B/16 patch size
    side = img_size // patch_size  # 50x50 grid

    assert side*side == N, f"Token count is not square grid: N={N}"
    
    feat = x.reshape(side, side, D).permute(2, 0, 1).contiguous()  # [D, H', W']
    feat = F.normalize(feat, dim=0)  # L2 normalize each position, more stable
    
    # 3) PCA reduce to 3 channels (RGB)
    X = feat.flatten(1).T.cpu().numpy()  # [H'*W', D]
    X = X - X.mean(0, keepdims=True)     # Remove mean
    
    # SVD decomposition (more stable than PCA)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = U[:, :3] @ np.diag(S[:3])        # Take first 3 principal components [H'*W', 3]
    Y = Y.reshape(side, side, 3)
    
    # Map each channel independently to [0, 1]
    Y_min = Y.min(axis=(0, 1), keepdims=True)
    Y_max = Y.max(axis=(0, 1), keepdims=True)
    Y = (Y - Y_min) / (Y_max - Y_min + 1e-6)
    rgb_small = (Y * 255).astype(np.uint8)  # [H', W', 3]
    
    # 4) Upsample to original image size
    W_orig, H_orig = orig_size
    rgb_pil = Image.fromarray(rgb_small).resize((W_orig, H_orig), resample=Image.Resampling.BICUBIC)
    
    # 5) Save result
    rgb_pil.save(output_path)
    print(f"Saved dense feature visualization to: {output_path}")
    
    return np.array(rgb_pil)


# ============ Main Program ============
if __name__ == "__main__":
    # Configure paths
    checkpoint_path = r"F:\胃结直肠内窥镜公开训练数据集\Med_Image_Train_Datasets_Detection_baesd_on_DINOv3\结果1\training_23999\teacher_checkpoint.pth"
    image_path = r"F:\胃结直肠内窥镜公开训练数据集\Med_Image_Train_Datasets_Detection_baesd_on_DINOv3\images\WCEBleedGen\img- (115).png"
    output_dir = "dense_features_comparison"
    resize_size = 1000  # Input image size
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1) Load model
    print("Loading model...")
    model = dinov3_vitb16(pretrained=False)
    model.to(device)
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract teacher weights (DINOv3 checkpoint format)
    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove possible prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' or 'backbone.' prefixes
        k = k.replace('module.', '').replace('backbone.', '')
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded successfully!")
    
    # 2) Read and preprocess image
    print(f"Loading image: {image_path}")
    image_tensor, orig_size = get_img_tensor(image_path, resize_size=resize_size)
    print(f"Original image size: {orig_size}")
    print(f"Input tensor shape: {image_tensor.shape}")
    
    # 3) Save original image
    original_img = Image.open(image_path).convert('RGB')
    original_img.save(os.path.join(output_dir, "1_original.png"))
    print(f"Saved original image")
    
    # 4) Extract last layer features and visualize
    print("\n=== Extracting last layer features ===")
    rgb_last = extract_dense_features_with_pca(
        model=model,
        image_tensor=image_tensor,
        orig_size=orig_size,
        layer_index=-1,
        output_path=os.path.join(output_dir, "2_last_layer.png"),
        use_multi_layer=False,
        img_size=resize_size
    )
    
    # 5) Extract last 4 layers average features and visualize
    print("\n=== Extracting last 4 layers average ===")
    rgb_multi = extract_dense_features_with_pca(
        model=model,
        image_tensor=image_tensor,
        orig_size=orig_size,
        layer_index=-1,
        output_path=os.path.join(output_dir, "3_last_4_layers_avg.png"),
        use_multi_layer=True,
        img_size=resize_size
    )
    
    # 6) Overlay to original image
    print("\n=== Creating overlay ===")
    # Overlay last 4 layers average features to original image
    W_orig, H_orig = orig_size
    original_resized = original_img.resize((W_orig, H_orig))
    original_np = np.array(original_resized)
    
    # Mix: 50% original + 50% feature map
    overlay = (original_np * 0.3 + rgb_multi * 0.7).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(os.path.join(output_dir, "4_overlay.png"))
    print(f"Saved overlay image")
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")
    print("  1_original.png - Original image")
    print("  2_last_layer.png - Last layer feature PCA")
    print("  3_last_4_layers_avg.png - Last 4 layers average feature PCA")
    print("  4_overlay.png - Feature overlay to original image")






