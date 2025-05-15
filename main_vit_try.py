import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from Pre_Processing.data_preparation import extract_data

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

        t1c_image = nib.load(t1c_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        if self.use_thresholding:
            t1c_image, attention_mask = self.apply_threshold_contrast(t1c_image)

        t1c_image = torch.tensor(t1c_image, dtype=torch.float32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        label = torch.clamp(label, min=0, max=3)

        return t1c_image.unsqueeze(0), attention_mask.unsqueeze(0), label

    def apply_threshold_contrast(self, volume, threshold_percentile=0.01, scale=1.0):
        std = np.std(volume)
        if std == 0:
            std = 1e-8
        normalized_img = (volume - np.mean(volume)) / std

        enhanced = np.copy(volume)
        threshold = np.percentile(normalized_img, (1 - threshold_percentile) * 100)
        enhanced[normalized_img <= threshold] = 0
        enhanced *= scale

        attention_mask = (enhanced > 0).astype(np.float32)
        return enhanced, attention_mask

class AttentionGatedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[16, 32, 64, 128]):
        super(AttentionGatedUNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(features[0], features[1], kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.attention_gate = nn.Sequential(
            nn.Conv3d(features[1], features[1], kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(features[1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, attention_mask):
        x = self.encoder(x)

        # Apply attention mask as spatial gate
        gated_x = x * attention_mask
        gated_x = self.attention_gate(gated_x)

        x = self.decoder(gated_x)
        return x

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# Load Data
df = extract_data()
train_dataset = T1cMRI_Dataset(df[['t1c_path', 'label_path']], use_thresholding=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionGatedUNet3D(in_channels=1, out_channels=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler = torch.cuda.amp.GradScaler()

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, masks, labels in train_dataloader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, masks)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice_score = 0
    total_samples = 0

    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)

            preds = torch.argmax(outputs, dim=1)
            intersection = (preds * labels).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))
            dice_score = (2. * intersection) / (union + 1e-6)

            total_dice_score += dice_score.mean().item()
            total_samples += 1

    avg_dice = total_dice_score / total_samples
    print(f"Average Dice Score: {avg_dice:.4f}")

# Example Usage
evaluate_model(model, train_dataloader, device)
