# data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_imagenette_dataloader(batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(
        root='./data/imagenette2-320/val',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
