# vit_utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

def visualize_attention_map(image, attention_maps, image_size=(224, 224), head_idx=0, avg_heads=True, patch_idx=98):    
    attn_map = attention_maps[-1]
    
    if len(attn_map.shape) == 4:
        attn_map_single = attn_map[0]
        
        if avg_heads:
            attn_map_single = attn_map_single.mean(axis=0)
        else:
            attn_map_single = attn_map_single[head_idx]
        
        cls_attention = attn_map_single[1:, 1:]
        
        if cls_attention.shape == (196, 196):
            attention_for_patch = cls_attention[patch_idx]
        else:
            print(f"Unexpected shape for cls_attention: {cls_attention.shape}")
            return
        
        try:
            attention_for_patch = torch.tensor(attention_for_patch).reshape((14, 14))
        except ValueError as e:
            print(f"Error reshaping attention map: {e}")
            print(f"Current shape of attention data: {attention_for_patch.shape}")
            return
    
        attention_resized = resize(attention_for_patch.unsqueeze(0), image_size).squeeze(0).numpy()
        attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min()) # normalize attn
    
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
        original_image = image.permute(1, 2, 0).cpu().numpy() # original img
        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(original_image) # attn overlay
        ax[1].imshow(attention_resized, cmap='jet', alpha=0.5)
        ax[1].set_title('Attention Overlay')
        ax[1].axis('off')
    
        plt.tight_layout()
        plt.show()
    else:
        print("Unexpected attention map shape. Unable to process.")