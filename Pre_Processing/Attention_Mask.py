import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Custom 3D Vision Transformer Model (ViT 3D)
class ViT3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=768, num_heads=8, depth=12, num_classes=2):
        super(ViT3D, self).__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=depth)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input x is (B, C, D, H, W)
        x = self.patch_embed(x)  # (B, embed_dim, D/4, H/4, W/4)
        x = x.flatten(2).permute(2, 0, 1)  # (D*H*W, B, embed_dim)
        x = self.transformer(x, x)  # (D*H*W, B, embed_dim)
        x = x.mean(dim=0)  # (B, embed_dim)
        x = self.classifier(x)  # (B, num_classes)
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
            mask = nib.load(self.mask_paths[idx]).get_fdata()
            mask = np.any(mask > 0).astype(np.float32)  # Binary classification (Tumor/No Tumor)
            return img, torch.tensor(mask, dtype=torch.long)
        else:
            return img

# Prepare DataLoader for 3D ViT
def prepare_data_3d(img_paths, mask_paths=None):
    transform = T.Compose([
        T.Resize((64, 128, 128)),  # Resize to (D, H, W)
        T.Normalize(mean=0.5, std=0.5)
    ])
    dataset = MRIDataset3D(img_paths, mask_paths=mask_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=4)
    return loader

# Example Usage
if __name__ == "__main__":
    # Example Image Paths
    img_paths = ["/path/to/your/image1.nii.gz", "/path/to/your/image2.nii.gz"]
    mask_paths = ["/path/to/your/mask1.nii.gz", "/path/to/your/mask2.nii.gz"]

    # Initialize 3D Vision Transformer
    model = ViT3D().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    loader = prepare_data_3d(img_paths, mask_paths)

    # Example Forward Pass
    for imgs, masks in loader:
        imgs = imgs.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(imgs)
        print(outputs.shape)  # (B, num_classes)
