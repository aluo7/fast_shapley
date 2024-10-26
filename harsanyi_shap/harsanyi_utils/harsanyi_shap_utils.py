# harsanyi_shap_utils.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def compute_shap_from_harsanyi_vit(model, data):
    device = next(model.parameters()).device
    data = data.to(device)
    
    shap_values = []
    
    with torch.no_grad():
        patch_embeddings = model.vit.patch_embed(data)
        transformer_output = model.vit.blocks(patch_embeddings)
        
        num_patches_in_output = transformer_output.size(1)
        if model.patch_contributions.size(0) != num_patches_in_output:
            model.patch_contributions = nn.Parameter(
                torch.randn(num_patches_in_output, model.hidden_dim, device=device),
                requires_grad=True
            )

        contributions = torch.einsum('ph,bph->bp', model.patch_contributions, transformer_output)
        
        for i in range(model.num_patches):  # compute shapley values based on contributions
            contribution = contributions[:, i]
            shap_values.append(contribution.cpu().detach())
    
    expected_patches = model.num_patches if model.num_patches else 196
    if len(shap_values) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, but got {len(shap_values)}")

    return torch.stack(shap_values, dim=1)

def map_harsanyi_shap_to_image(shap_values, img_size=224, patch_size=16):
    num_patches = img_size // patch_size
    
    shap_grid = np.abs(shap_values.cpu().numpy().reshape(num_patches, num_patches))
    
    scale_factor = img_size / num_patches
    upscaled_shap = scipy.ndimage.zoom(shap_grid, zoom=(scale_factor, scale_factor), order=1)
    
    return upscaled_shap

def visualize_harsanyi_shap(shap_values, original_input, patch_size=16, smooth=False, color_map='jet'):
    upscaled_shap = map_harsanyi_shap_to_image(shap_values, img_size=224, patch_size=patch_size)
    print(f"SHAP value range before normalization: min={upscaled_shap.min()}, max={upscaled_shap.max()}")
    
    normalized_shap = (upscaled_shap - upscaled_shap.min()) / (upscaled_shap.max() - upscaled_shap.min())
    print(f"Normalized SHAP value range: min={normalized_shap.min()}, max={normalized_shap.max()}")
    
    if smooth:
        normalized_shap = scipy.ndimage.gaussian_filter(normalized_shap, sigma=3)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_input)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(original_input)
    im = ax[1].imshow(normalized_shap, cmap=color_map, alpha=0.5)
    ax[1].set_title('SHAP Overlay')
    ax[1].axis('off')
    
    fig.colorbar(im, ax=ax[1], orientation='vertical')
    plt.tight_layout()
    plt.show()
