# vit.py
import torch
import timm

def get_vit_model(model_name='vit_base_patch16_224', pretrained=True, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    model.eval()
    
    return model
