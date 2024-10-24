# data.py
import os
import requests
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def download_and_extract_imagenette(data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    dataset_path = os.path.join(data_dir, 'imagenette2-320.tgz')

    if not os.path.exists(os.path.join(data_dir, 'imagenette2-320')):
        response = requests.get(dataset_url, stream=True)
        with open(dataset_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        
        with tarfile.open(dataset_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("download complete")
    else:
        print("imagenette already exists, skipping download")

def get_imagenette_dataloader(batch_size=32, img_size=224, data_dir='./data'):
    download_and_extract_imagenette(data_dir=data_dir)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'imagenette2-320/val'),
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
