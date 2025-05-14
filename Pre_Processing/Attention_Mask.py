import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Custom 3D Vision Transformer Model (ViT 3D) - Voxel-wise Segmentation (4 Classes)
class ViT3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=384, num_heads=4, depth=6, num_classes=4):
        super(ViT3D, self).__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.position_embedding = nn.Parameter(torch.randn(1, 4096, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=depth
        )
        self.conv_segmentation = nn.Conv3d(embed_dim, num_classes, kernel_size=1)  # Voxel-wise segmentation

    def forward(self, x):
        x = self.patch_embed(x)  # (B, embed_dim, D/4, H/4, W/4)
        x = x.flatten(2).permute(0, 2, 1)  # (B, D*H*W, embed_dim)
        x += self.position_embedding[:, :x.size(1), :]  # Positional encoding
        x = self.transformer(x)  # (B, D*H*W, embed_dim)
        x = x.permute(0, 2, 1).contiguous().reshape(x.size(0), -1, 8, 8, 8)
        x = self.conv_segmentation(x)  # (B, num_classes, D/4, H/4, W/4)
        return x

# Custom Dataset for Loading 3D MRI Scans
class MRIDataset3D(Dataset):
    def __init__(self, img_paths, mask_paths=None, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx]).get_fdata()  # (D, H, W)
        img = np.expand_dims(img, axis=0)  # (1, D, H, W)

        if self.transform:
            img = self.transform(torch.tensor(img, dtype=torch.float32))

        if self.mask_paths:
            mask = nib.load(self.mask_paths[idx]).get_fdata().astype(np.int64)  # (D, H, W)
            return img, torch.tensor(mask, dtype=torch.long)
        else:
            return img

# Data Preparation
def prepare_data_3d(img_paths, mask_paths=None):
    transform = T.Compose([
        T.Lambda(lambda x: torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(32, 64, 64), mode='trilinear', align_corners=False).squeeze(0)
        ),
        T.Normalize(mean=0.0, std=1.0)
    ])
    dataset = MRIDataset3D(img_paths, mask_paths=mask_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
    return loader

# Training Loop
def train_vit3d(model, train_loader, device, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)  # (B, 4, D/4, H/4, W/4)
            masks = torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()

            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation Function
def evaluate_vit3d(model, test_loader, device):
    model.eval()
    correct_voxels = 0
    total_voxels = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            correct_voxels += (predicted == masks).sum().item()
            total_voxels += masks.numel()

    accuracy = correct_voxels / total_voxels
    print(f"Voxel-wise Segmentation Accuracy: {accuracy:.4f}")
