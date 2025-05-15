import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

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

class VisionTransformer3D(nn.Module):
    def __init__(self, img_size=(240, 240, 155), patch_size=(16, 16, 16), in_channels=1, num_classes=4, embed_dim=768):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Conv3d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x, attention_mask):
        B, C, D, H, W = x.shape
        x = self.patch_embed(x)
        D_new, H_new, W_new = x.shape[2], x.shape[3], x.shape[4]
        x = x.view(B, 768, D_new, H_new, W_new)

        # Apply attention (Masked)
        x = self.masked_attention(x, x, x, attention_mask)
        x = self.fc(x)

        return x

    def masked_attention(self, Q, K, V, attention_mask, eps=1e-8):
        scale = Q.size(-1) ** 0.5
        attn_scores = (Q @ K.transpose(-2, -1)) / (scale + eps)
        B, _, D, H, W = attention_mask.shape

        # Flatten the attention mask to match the number of patches
        attention_mask = attention_mask.view(B, -1)

        # Ensure it matches the shape of the attention scores
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attn_scores.size(1), 1)

        # Adjust shape to align with attention scores
        attn_scores = attn_scores * attention_mask


        attn_scores = torch.clamp(attn_scores, min=-50, max=50)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = attn_probs @ V
        return output


# Gradient Clipping Function
def clip_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

# Evaluation and Visualization
def visualize_results(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)

            plt.figure(figsize=(24, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(images[0, 0, :, :, images.shape[-1] // 2].cpu(), cmap='gray')
            plt.title("Input MRI Image")

            plt.subplot(1, 4, 2)
            plt.imshow(masks[0, 0, :, :, images.shape[-1] // 2].cpu(), cmap='gray')
            plt.title("Threshold Mask (Attention)")

            plt.subplot(1, 4, 3)
            plt.imshow(labels[0, :, :, images.shape[-1] // 2].cpu(), cmap='inferno')
            plt.title("Ground Truth Mask")

            plt.subplot(1, 4, 4)
            plt.imshow(outputs[0, 0, :, :, images.shape[-1] // 2].cpu().argmax(dim=0), cmap='inferno')
            plt.title("Predicted Mask")

            plt.show()
            break

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# Load Data
train_dataset = T1cMRI_Dataset(df[['t1c_path', 'label_path']], use_thresholding=True)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer3D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, masks, labels in train_dataloader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, masks)

        # Resizing output to match label size
        outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode='trilinear', align_corners=False)

        loss = criterion(outputs, labels)
        loss.backward()
        clip_gradients(model)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}")

# Evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice_score = 0
    total_samples = 0

    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)

            # Resize outputs
            outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode='trilinear', align_corners=False)
            preds = torch.argmax(outputs, dim=1)

            # Calculate Dice Score
            intersection = (preds * labels).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))
            dice_score = (2. * intersection) / (union + 1e-6)

            total_dice_score += dice_score.mean().item()
            total_samples += 1

    avg_dice = total_dice_score / total_samples
    print(f"Average Dice Score: {avg_dice:.4f}")
