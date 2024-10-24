# shap_utils.py
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def compute_shap_values(model, inputs, background=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if background is None:
        background = inputs[:5]
    
    background = background.to(device)
    
    explainer = shap.GradientExplainer(model, background)
    
    inputs = inputs.to(device)
    shap_values = explainer.shap_values(inputs)
    
    return shap_values

def visualize_shap(shap_values, original_image, idx, smooth=True, sigma=1):
    aggregated_shap = np.abs(shap_values[idx].mean(axis=-1))
    
    if smooth:
        aggregated_shap = scipy.ndimage.gaussian_filter(aggregated_shap, sigma=sigma)
    
    normalized_shap = (aggregated_shap - aggregated_shap.min()) / (aggregated_shap.max() - aggregated_shap.min())
    
    try:
        shap_img = normalized_shap.reshape((3, 224, 224)).transpose((1, 2, 0))
    except ValueError as e:
        print(f"Error reshaping normalized SHAP values: {e}")
        return
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(original_image)
    ax[1].imshow(shap_img, cmap='hot', alpha=0.6)
    ax[1].set_title('Smoothed SHAP Overlay')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
