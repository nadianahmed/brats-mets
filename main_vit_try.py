import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import pandas as pd

class T1cMRI_Dataset(Dataset):
    def __init__(self, dataframe, transform=None, use_thresholding=True):
        self.dataframe = dataframe
        self.transform = transform
        self.use_thresholding = use_thresholding

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        t1c_path = self.dataframe.iloc[idx]['t1c_path']
        label_path = self.dataframe.iloc[idx]['label_path']

        # Load NIfTI images
        t1c_image = nib.load(t1c_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Apply Thresholding if enabled
        if self.use_thresholding:
            t1c_image = self.apply_thresholding(t1c_image)

        # Convert to PyTorch tensor (2 channels: MRI + Thresholded Image)
        t1c_image = torch.tensor(t1c_image, dtype=torch.float32)  # (2, D, H, W)
        label = torch.tensor(label, dtype=torch.long)

        # Ensure labels are in range [0, 1, 2, 3]
        label = torch.clamp(label, min=0, max=3)

        return t1c_image, label

    def apply_thresholding(self, image):
        normalized_img = (image - np.mean(image)) / np.std(image)
        threshold = np.percentile(normalized_img, 99)
        thresholded = (normalized_img > threshold).astype(np.float32)
        return np.stack([image, thresholded])  # (2, D, H, W)

class VisionTransformer3D(nn.Module):
    def __init__(self, img_size=(240, 240, 155), patch_size=(16, 16, 16), in_channels=2, num_classes=4, embed_dim=768, num_heads=12, depth=12):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.pos_embed = nn.Parameter(torch.randn(num_patches, embed_dim))

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=depth, batch_first=True)
        self.fc = nn.Conv3d(embed_dim, 4, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.patch_embed(x)  # (B, Embed_Dim, D/16, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)

        # Transformer
        x = self.transformer(x, x)
        x = x.transpose(1, 2).view(B, -1, D // 16, H // 16, W // 16)

        # Voxel-wise Classification
        x = self.fc(x)  # (B, num_classes=4, D/16, H/16, W/16)
        return x

# Load Data
df = extract_data()
dataset = T1cMRI_Dataset(df[['t1c_path', 'label_path']])
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer3D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode='trilinear', align_corners=False)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/10], Loss: {epoch_loss / len(dataloader):.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Visualization Function
def visualize_predictions(images, labels, predictions, slice_idx=80):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display Input Image
    axes[0].imshow(images[0, 0, :, :, slice_idx].cpu(), cmap='gray')
    axes[0].set_title("Input MRI Image")

    # Display Ground Truth Mask
    axes[1].imshow(labels[0, :, :, slice_idx].cpu(), cmap='viridis')
    axes[1].set_title("Ground Truth Mask")

    # Display Predicted Mask
    axes[2].imshow(predictions[0, :, :, slice_idx].cpu(), cmap='viridis')
    axes[2].set_title("Predicted Mask")

    plt.show()

# Evaluation Scheme
model.eval()
all_dice_scores = []
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode='trilinear', align_corners=False)
        predicted = torch.argmax(outputs, dim=1)

        # Dice Score Calculation
        for cls in range(4):
            intersection = (predicted == cls) * (labels == cls)
            dice_score = (2 * intersection.sum()) / (predicted.eq(cls).sum() + labels.eq(cls).sum() + 1e-5)
            all_dice_scores.append(dice_score.item())

        # Visualization
        visualize_predictions(images, labels, predicted)

# Average Dice Score per Class
dice_per_class = np.mean(np.array(all_dice_scores).reshape(-1, 4), axis=0)
for cls, dice in enumerate(dice_per_class):
    print(f"Dice Score for Class {cls}: {dice:.4f}")

