import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Pre_Processing.data_preparation import extract_data
from Pre_Processing.image_analysis import load_image, apply_threshold_contrast, save_image
import Pre_Processing.constants as constants
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16
import torchvision.transforms as T

# Vision Transformer Setup (Using Pretrained ViT)
class ViTWithAttention(nn.Module):
    def __init__(self, use_attention_mask=True):
        super(ViTWithAttention, self).__init__()
        self.vit = vit_b_16(weights='IMAGENET1K_V1')  # Use the updated weights method
        self.use_attention_mask = use_attention_mask

        # Replace the final classification head for binary classification
        in_features = self.vit.heads[0].in_features
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features, 2)  # Binary Classification (Tumor / Non-Tumor)
        )

    def forward(self, x, attention_mask=None):
        if self.use_attention_mask and attention_mask is not None:
            x = x * attention_mask
        x = self.vit(x)
        return x

# Custom Dataset with Thresholded Mask
class MRIDataset(Dataset):
    def __init__(self, img_paths, mask_paths=None, scan_type=None, transform=None, max_slices=20):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.scan_type = scan_type
        self.transform = transform
        self.max_slices = max_slices

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx])
        thresholded_img = apply_threshold_contrast(img, self.scan_type)
        volume = thresholded_img.get_fdata()

        # Sample Slices (Efficient Memory Use)
        num_slices = min(self.max_slices, volume.shape[0])
        selected_slices = np.linspace(0, volume.shape[0] - 1, num_slices).astype(int)
        slices = [volume[i, :, :] for i in selected_slices]

        # Convert slices to tensor
        slices = np.stack(slices, axis=0)

        if self.transform:
            slices = self.transform(slices)

        return torch.tensor(slices, dtype=torch.float32)

# Prepare DataLoader
def prepare_data(img_paths, mask_paths=None, scan_type=None, max_slices=20):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224))
    ])
    dataset = MRIDataset(img_paths, mask_paths=mask_paths, scan_type=scan_type, transform=transform, max_slices=max_slices)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    return loader

# Load Dataset Paths
data = extract_data()
t1c_paths = data['t1c_path'].tolist()
mask_paths = data['label_path'].tolist()

# Prepare DataLoader with Thresholded Mask
train_loader = prepare_data(t1c_paths, mask_paths, scan_type=constants.T1C_SCAN_TYPE)

# Initialize Model
model = ViTWithAttention(use_attention_mask=True).to('cuda' if torch.cuda.is_available() else 'cpu')

# Train and Evaluate Function
def train_and_evaluate_vit(model, train_loader, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for imgs in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            # Generate attention mask dynamically
            attn_mask = (imgs > 0).float()  # Thresholded attention mask
            outputs = model(imgs, attention_mask=attn_mask)

            loss = criterion(outputs, torch.zeros(outputs.size(0), dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

# Train and Save Model
print("Training Vision Transformer with Thresholded Attention Mask...")
trained_model = train_and_evaluate_vit(model, train_loader, epochs=10, lr=1e-4)
torch.save(trained_model.state_dict(), "trained_vit_thresholded.pth")
print("Model saved as 'trained_vit_thresholded.pth'")
