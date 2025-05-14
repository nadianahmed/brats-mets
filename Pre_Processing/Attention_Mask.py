import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vit_b_16
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Vision Transformer Setup (Using Pretrained ViT)
class ViTWithAttention(nn.Module):
    def __init__(self, use_attention_mask=True):
        super(ViTWithAttention, self).__init__()
        self.vit = vit_b_16(weights='IMAGENET1K_V1')  # Use the updated weights method
        self.use_attention_mask = use_attention_mask

        # Replace the final classification head for binary classification
        in_features = self.vit.heads[0].in_features  # Access the in_features of the final layer
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features, 2)  # Binary Classification (Tumor / Non-Tumor)
        )

    def forward(self, x, attention_mask=None):
        if self.use_attention_mask and attention_mask is not None:
            x = x * attention_mask  # Apply attention mask
        x = self.vit(x)
        return x

# Custom Dataset for Loading MRI Scans (Efficient Thresholded Mask)
class MRIDataset(Dataset):
    def __init__(self, img_paths, mask_paths=None, transform=None, threshold=0.2, max_slices=20):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.threshold = threshold
        self.max_slices = max_slices

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load MRI image (3D Volume)
        img = nib.load(self.img_paths[idx]).get_fdata()  # (D, H, W)
        num_slices = img.shape[0]

        # Randomly sample slices (up to max_slices)
        selected_slices = np.random.choice(num_slices, min(self.max_slices, num_slices), replace=False)
        slices = [img[i, :, :] for i in selected_slices]

        # Generate Attention Mask using Thresholding
        attention_masks = [(s > np.percentile(s, 80)).astype(np.float32) for s in slices]

        # Load Ground Truth Mask (3D) only for selected slices
        if self.mask_paths:
            mask = nib.load(self.mask_paths[idx]).get_fdata()
            mask_slices = [mask[i, :, :] for i in selected_slices]
        else:
            mask_slices = [np.zeros_like(s) for s in slices]

        # Convert slices to tensors
        slices = np.stack(slices, axis=0)
        mask_slices = np.stack(mask_slices, axis=0)
        attention_masks = np.stack(attention_masks, axis=0)

        if self.transform:
            slices = self.transform(slices)
            mask_slices = self.transform(mask_slices)
            attention_masks = self.transform(attention_masks)

        return torch.tensor(slices, dtype=torch.float32), torch.tensor(mask_slices, dtype=torch.float32), torch.tensor(attention_masks, dtype=torch.float32)

# Prepare DataLoader Function
def prepare_data(img_paths, mask_paths=None, max_slices=20):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),  # Required for ViT
    ])
    dataset = MRIDataset(img_paths, mask_paths=mask_paths, transform=transform, max_slices=max_slices)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
    return loader
