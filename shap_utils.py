# shap_utils.py
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

def compute_shap_values(model, inputs, background=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if background is None:
        background = inputs[:5]
    
    background = background.to(device)
    
    explainer = shap.GradientExplainer(model, background)
    
    inputs = inputs.to(device)
    shap_values = explainer.shap_values(inputs)
    
    return shap_values

def visualize_shap(shap_values, original_image):
    shap_img = shap_values[0].reshape((224, 224, 3))
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(original_image)
    ax[1].imshow(shap_img, cmap='jet', alpha=0.5)
    ax[1].set_title('SHAP Overlay')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
