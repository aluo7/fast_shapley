# harsanyi.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
from models.vit import get_vit_model

class HarsanyiViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_classes=10):
        super(HarsanyiViT, self).__init__()
        
        self.vit = get_vit_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.hidden_dim = self.vit.patch_embed.proj.out_channels
        self.num_patches = None
        
        self.output_layer = nn.Linear(self.hidden_dim, num_classes)
    
    def forward(self, x):
        device = x.device

        patch_embeddings = self.vit.patch_embed(x)

        batch_size = patch_embeddings.size(0)
        cls_tokens = self.vit.cls_token.expand(batch_size, -1, -1)
        patch_embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)

        transformer_output = self.vit.blocks(patch_embeddings)

        if self.num_patches is None:
            self.num_patches = transformer_output.size(1) - 1
            self.patch_contributions = nn.Parameter(torch.randn(self.num_patches, self.hidden_dim, device=device), requires_grad=True)

        cls_token = transformer_output[:, 0, :]

        contributions = torch.einsum('ph,bph->bh', self.patch_contributions, transformer_output[:, 1:, :])
        
        aggregated_output = cls_token + contributions

        prediction = self.output_layer(aggregated_output)

        return prediction


def train_harsanyi_vit(model, train_loader, epochs=20, warmup_epochs=1, lr=0.00025, min_lr=1e-6):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = torch.nn.CrossEntropyLoss()

    total_epochs = epochs
    lr_func = lambda epoch: max(min_lr, min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / total_epochs * math.pi) + 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    device = next(model.parameters()).device
    model.train()

    epochs = epochs
    for epoch in range(epochs):
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target) + 0.000005 * torch.sum(torch.abs(model.patch_contributions))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

