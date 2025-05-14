import os
import torch
import matplotlib.pyplot as plt
from Pre_Processing.Attention_Mask import ViT3D, prepare_data_3d
from Pre_Processing.data_preparation import extract_data
import numpy as np

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Paths
data = extract_data()  # This should be your DataFrame with scan paths

# Example: Using T1C scans for training
print(data.columns)
t1c_paths = data['t1c_path'].tolist()
mask_paths = data['label_path'].tolist()  # Ground truth segmentation masks

# Prepare DataLoader for 3D ViT (Optimized)
train_loader = prepare_data_3d(t1c_paths, mask_paths)
eval_loader = prepare_data_3d(t1c_paths, mask_paths)

# Initialize 3D Vision Transformer
model = ViT3D().to(device)

# Optimized Training with AMP
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            masks = torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/10] Completed - Average Loss: {avg_loss:.4f}")

# Save the Trained Model
model_save_path = "trained_3d_vit.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
