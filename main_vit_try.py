import os
import torch
import matplotlib.pyplot as plt
from Pre_Processing.Attention_Mask import ViTWithAttention, prepare_data, train_and_evaluate_vit
import Pre_Processing.constants as constants
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
scan_type = constants.T1C_SCAN_TYPE

# Prepare DataLoader with Thresholded Mask
train_loader = prepare_data(t1c_paths, mask_paths, scan_type=scan_type, max_slices=20)
eval_loader = prepare_data(t1c_paths, mask_paths, scan_type=scan_type, max_slices=20)

# Initialize Vision Transformer with Attention Mask
model = ViTWithAttention(use_attention_mask=True).to(device)

# Train and Evaluate the Model
print("Training and Evaluating Vision Transformer with Thresholded Attention Mask...")
trained_model = train_and_evaluate_vit(model, train_loader, eval_loader, epochs=10, lr=1e-4)

# Save the Trained Model
model_save_path = "trained_vit_thresholded_attention.pth"
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
