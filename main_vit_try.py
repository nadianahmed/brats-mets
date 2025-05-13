import os
import torch
import matplotlib.pyplot as plt
from Vision_Transformer_with_Attention_Mask import ViTWithAttention, prepare_data, train_vit
import Pre_Processing.constants as constants
from Pre_Processing.data_preparation import extract_data
from sklearn.metrics import confusion_matrix
import numpy as np

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Paths
data = extract_data()  

# Example: Using T1C scans for training
t1c_paths = data['t1c_path'].tolist()
mask_paths = data['seg_path'].tolist()  # Ground truth segmentation masks
scan_type = constants.T1C_SCAN_TYPE

# Prepare DataLoader with Automatic Thresholding
train_loader = prepare_data(t1c_paths, scan_type)

# Initialize Vision Transformer with Attention Mask
model = ViTWithAttention(use_attention_mask=True).to(device)

# Train the Model
print("Training Vision Transformer with Attention Mask...")
trained_model = train_vit(model, train_loader, epochs=10, lr=1e-4)

# Save the Trained Model
model_save_path = "trained_vit_attention.pth"
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluation Function (With Metrics and Visualization)
def evaluate_vit(model, dataloader, mask_paths):
    model.eval()
    dice_scores, sensitivities, specificities, precisions = [], [], [], []

    with torch.no_grad():
        for idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = nib.load(mask_paths[idx]).get_fdata()  # Load ground truth mask
            masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            # Display a few samples
            if idx < 3:
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

            # Calculate Metrics
            pred = preds[0].cpu().numpy().flatten()
            target = masks[0].cpu().numpy().flatten()

            # Dice Similarity Coefficient
            intersection = np.sum(pred * target)
            dice = (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-5)
            dice_scores.append(dice)

            # Precision, Sensitivity, Specificity
            tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-5)
            specificity = tn / (tn + fp + 1e-5)
            precision = tp / (tp + fp + 1e-5)

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)

    print(f"Dice Similarity Coefficient (DSC): {np.mean(dice_scores):.4f}")
    print(f"Sensitivity: {np.mean(sensitivities):.4f}")
    print(f"Specificity: {np.mean(specificities):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")

# Load Evaluation Data
# Example: Using the same T1C for evaluation (split this properly in practice)
eval_loader = prepare_data(t1c_paths, scan_type)
evaluate_vit(trained_model, eval_loader, mask_paths)
