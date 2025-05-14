import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vit_b_16
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

# Custom Dataset for Loading MRI Scans (2D Slices as Batch)
class MRIDataset(Dataset):
    def __init__(self, img_paths, mask_paths=None, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load MRI image (3D Volume)
        img = nib.load(self.img_paths[idx]).get_fdata()  # (D, H, W)

        # Use all slices (2D slices) as a batch
        slices = [img[i, :, :] for i in range(img.shape[0])]  # List of 2D slices

        # Load Ground Truth Mask (3D)
        if self.mask_paths:
            mask = nib.load(self.mask_paths[idx]).get_fdata()  # (D, H, W)
            mask_slices = [mask[i, :, :] for i in range(mask.shape[0])]  # List of 2D mask slices
        else:
            mask_slices = [np.zeros_like(s) for s in slices]

        # Convert slices to a batch of 2D slices
        slices = np.stack(slices, axis=0)  # (D, H, W)
        mask_slices = np.stack(mask_slices, axis=0)  # (D, H, W)

        if self.transform:
            slices = self.transform(slices)
            mask_slices = self.transform(mask_slices)

        return torch.tensor(slices, dtype=torch.float32), torch.tensor(mask_slices, dtype=torch.float32)

# Prepare DataLoader Function
def prepare_data(img_paths, mask_paths=None):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),  # Required for ViT
    ])
    dataset = MRIDataset(img_paths, mask_paths=mask_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return loader

# Train and Evaluate Function
def train_and_evaluate_vit(model, train_loader, eval_loader, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            # Flatten batch of 2D slices for ViT
            B, D, H, W = imgs.shape
            imgs = imgs.view(B * D, 1, H, W).repeat(1, 3, 1, 1)
            masks = masks.view(B * D, H, W)

            outputs = model(imgs)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        for imgs, masks in eval_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            B, D, H, W = imgs.shape
            imgs = imgs.view(B * D, 1, H, W).repeat(1, 3, 1, 1)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            # Display samples
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.imshow(imgs[0].cpu().squeeze(), cmap='gray')
            plt.title("Original MRI Image")

            plt.subplot(1,3,2)
            plt.imshow(masks[0].cpu().squeeze(), cmap='gray')
            plt.title("Ground Truth Mask")

            plt.subplot(1,3,3)
            plt.imshow(preds[0].cpu().squeeze(), cmap='gray')
            plt.title("Model Prediction")
            plt.show()
    
    return model
