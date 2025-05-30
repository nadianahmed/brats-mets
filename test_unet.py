# -*- coding: utf-8 -*-
"""brats-mets_UNet_v2.py

This script handles pre-processing and training a 3D Attention U-Net on the BraTS-MET dataset.
"""

import os
import pandas as pd
import nibabel as nib
import numpy as np
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Paths
EXTRACTED_FOLDER_NAME = 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'
DATASET_FOLDER = '/project/def-sreeram/hsheikh1/brats-mets/Datasets/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'
LABEL_NAME = 'seg'
T1C_SCAN_TYPE = 't1c'
T1N_SCAN_TYPE = 't1n'
T2F_SCAN_TYPE = 't2f'
T2W_SCAN_TYPE = 't2w'
PRE_PROCESSED_IMAGE_SUFFIX = 'processed'
PROJECT_NAME_PREFIX = 'BraTS'
APPLY_PRE_PROCESSING = False

# Thresholding
T1C_THRESHOLD_PERCENTILE = 0.01
T1C_SCALE = 1.0
T1N_THRESHOLD_PERCENTILE = 0.01
T1N_SCALE = 1.0
T2F_THRESHOLD_PERCENTILE = 0.01
T2F_SCALE = 1.0
T2W_THRESHOLD_PERCENTILE = 0.01
T2W_SCALE = 1.0

# Template Matching
T1C_TUMOUR_SIZE = 10
T1C_TEMPLATE_MATCH_THRESHOLD = 0.6
T1N_TUMOUR_SIZE = 10
T1N_TEMPLATE_MATCH_THRESHOLD = 0.6
T2F_TUMOUR_SIZE = 10
T2F_TEMPLATE_MATCH_THRESHOLD = 0.6
T2W_TUMOUR_SIZE = 10
T2W_TEMPLATE_MATCH_THRESHOLD = 0.6

def get_image_parent_path(path):
    return os.path.dirname(path)

def get_image_name_from_path(path):
    return os.path.basename(get_image_parent_path(path=path))

def normalize_image(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std

def load_image(filename):
    return nib.load(filename)

def save_image(img, filename, scan_type):
    output_image_name = '{}/{}-{}-{}.nii'.format(get_image_parent_path(path=filename),
                                                 get_image_name_from_path(path=filename),
                                                 scan_type,
                                                 PRE_PROCESSED_IMAGE_SUFFIX)
    nib.save(img, output_image_name)
    return output_image_name

def apply_threshold_contrast(img, scan_type, show_image=False):
    if scan_type == T1C_SCAN_TYPE:
        threshold_percentile = T1C_THRESHOLD_PERCENTILE
        scale = T1C_SCALE
    elif scan_type == T1N_SCAN_TYPE:
        threshold_percentile = T1N_THRESHOLD_PERCENTILE
        scale = T1N_SCALE
    elif scan_type == T2F_SCAN_TYPE:
        threshold_percentile = T2F_THRESHOLD_PERCENTILE
        scale = T2F_SCALE
    else:
        threshold_percentile = T2W_THRESHOLD_PERCENTILE
        scale = T2W_SCALE

    volume = img.get_fdata()
    normalized_img = normalize_image(volume)
    enhanced = np.copy(volume)
    threshold = np.percentile(normalized_img, (1 - threshold_percentile) * 100)
    enhanced[normalized_img <= threshold] = 0
    enhanced *= scale
    enhanced_data = np.clip(enhanced, 0, np.max(enhanced))
    return nib.Nifti1Image(enhanced_data, img.affine, img.header)

def create_spherical_template(radius):
    shape = (2 * radius + 1,) * 3
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) // 2
    distance = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    sphere = (distance <= radius).astype(np.float32)
    return gaussian_filter(sphere, sigma=1)

def apply_template_matching(img, scan_type, show_image=False):
    if scan_type == T1C_SCAN_TYPE:
        radius = T1C_TUMOUR_SIZE
        threshold = T1C_TEMPLATE_MATCH_THRESHOLD
    elif scan_type == T1N_SCAN_TYPE:
        radius = T1N_TUMOUR_SIZE
        threshold = T1N_TEMPLATE_MATCH_THRESHOLD
    elif scan_type == T2F_SCAN_TYPE:
        radius = T2F_TUMOUR_SIZE
        threshold = T2F_TEMPLATE_MATCH_THRESHOLD
    else:
        radius = T2W_TUMOUR_SIZE
        threshold = T2W_TEMPLATE_MATCH_THRESHOLD

    volume = img.get_fdata()
    radius_voxels = round(radius/(img.header.get_zooms()[0]))
    template = create_spherical_template(radius=radius_voxels)
    correlation = match_template(volume, template, pad_input=True)
    local_max = maximum_filter(correlation, size=template.shape) == correlation
    correlation_threshold = threshold * np.max(correlation)
    detected_peaks = (correlation > correlation_threshold) & local_max
    labeled, num_features = label(detected_peaks)
    match_coords_list = center_of_mass(detected_peaks, labeled, range(1, num_features + 1))
    match_coords_list = [tuple(map(int, coords)) for coords in match_coords_list]

    masked = np.zeros_like(volume)
    for x, y, z in match_coords_list:
        z_min, z_max = max(z - radius_voxels, 0), min(z + radius_voxels + 1, volume.shape[0])
        y_min, y_max = max(y - radius_voxels, 0), min(y + radius_voxels + 1, volume.shape[1])
        x_min, x_max = max(x - radius_voxels, 0), min(x + radius_voxels + 1, volume.shape[2])
        xx, yy, zz = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
        dist = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
        mask = dist <= radius_voxels
        masked[x_min:x_max, y_min:y_max, z_min:z_max][mask] = volume[x_min:x_max, y_min:y_max, z_min:z_max][mask]

    return nib.Nifti1Image(masked, img.affine, img.header), match_coords_list

def apply_img_processing(filename, scan_type, show_image=False):
    img = load_image(filename=filename)
    thresholded_img = apply_threshold_contrast(img=img, scan_type=scan_type)
    template_matched_image, _ = apply_template_matching(thresholded_img, scan_type)
    output_path = save_image(template_matched_image, filename, scan_type)
    print(output_path)
    return output_path

def extract_data():
    extracted_data_folder = os.path.join(DATASET_FOLDER)

    if not os.path.isdir(extracted_data_folder):
        raise FileNotFoundError(f"❌ Folder not found: {extracted_data_folder}")

    result = pd.DataFrame()
    t1c_scan_paths = []
    t1n_scan_paths = []
    t2f_scan_paths = []
    t2w_scan_paths = []
    label_paths = []
    preprocessed_paths = []
    scan_names = []

    for sample in os.listdir(extracted_data_folder):
        if PROJECT_NAME_PREFIX in sample:
            sample_folder_path = os.path.join(extracted_data_folder, sample)

            scan_names.append(sample)
            t1c_scan_paths.append(os.path.join(sample_folder_path, f"{sample}-{T1C_SCAN_TYPE}.nii"))
            t1n_scan_paths.append(os.path.join(sample_folder_path, f"{sample}-{T1N_SCAN_TYPE}.nii"))
            t2f_scan_paths.append(os.path.join(sample_folder_path, f"{sample}-{T2F_SCAN_TYPE}.nii"))
            t2w_scan_paths.append(os.path.join(sample_folder_path, f"{sample}-{T2W_SCAN_TYPE}.nii"))
            label_paths.append(os.path.join(sample_folder_path, f"{sample}-{LABEL_NAME}.nii"))
            preprocessed_paths.append(os.path.join(sample_folder_path, f"{sample}-{T1C_SCAN_TYPE}-{PRE_PROCESSED_IMAGE_SUFFIX}.nii"))

    result['scan_name'] = scan_names
    result['t1c_path'] = t1c_scan_paths
    result['t1n_path'] = t1n_scan_paths
    result['t2f_path'] = t2f_scan_paths
    result['t2w_path'] = t2w_scan_paths
    result['label_path'] = label_paths

    if APPLY_PRE_PROCESSING:
        print("🧪 Applying image processing for T1C scans...")
        result['t1c_processed_scan_path'] = result['t1c_path'].apply(
            lambda path: apply_img_processing(path, scan_type=T1C_SCAN_TYPE)
        )
    else:
        result['t1c_processed_scan_path'] = preprocessed_paths

    return result

class BRATSMetsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        t1c_img = nib.load(row['t1c_path']).get_fdata()
        t1c_img = np.expand_dims(t1c_img, axis=0)
        attn_mask = nib.load(row['t1c_processed_scan_path']).get_fdata()
        attn_mask = np.expand_dims(attn_mask, axis=0)
        seg = nib.load(row['label_path']).get_fdata()
        seg = np.expand_dims(seg, axis=0)
        t1c_img = (t1c_img - np.min(t1c_img)) / (np.max(t1c_img) - np.min(t1c_img) + 1e-5)
        attn_mask = (attn_mask - np.min(attn_mask)) / (np.max(attn_mask) - np.min(attn_mask) + 1e-5)
        sample = {
            'image': torch.tensor(t1c_img, dtype=torch.float32),
            'attention': torch.tensor(attn_mask, dtype=torch.float32),
            'label': torch.tensor(seg, dtype=torch.long)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

# Insert the fixed AttentionBlock3D and AttentionUNet3D classes here (use my corrected versions from above).

# Training function and main script remain unchanged.

# --- Training function ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch in dataloader:
        images = batch['image'].to(device)
        attn = batch['attention'].to(device)
        labels = batch['label'].to(device).squeeze(1)
        outputs = model(images, attn)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch complete. Loss: {:.4f}".format(loss.item()))

# --- Main script ---
if __name__ == "__main__":
    data = extract_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = AttentionUNet3D(in_channels=1, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = BRATSMetsDataset(dataframe=data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    train_one_epoch(model, dataloader, criterion, optimizer, device)
