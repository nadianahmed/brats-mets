
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
DATASET_FOLDER = '/project/def-sreeram/hsheikh1/brats-mets/Datasets/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'  
LABEL_NAME = 'seg'
T1C_SCAN_TYPE = 't1c'
T1N_SCAN_TYPE = 't1n'
T2F_SCAN_TYPE = 't2f'
T2W_SCAN_TYPE = 't2w'
PRE_PROCESSED_IMAGE_SUFFIX = 'processed'
PROJECT_NAME_PREFIX = 'BraTS'
APPLY_PRE_PROCESSING = True

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
    return save_image(template_matched_image, filename, scan_type)

def extract_data():
    extracted_data_folder = DATASET_FOLDER
    if not os.path.isdir(extracted_data_folder):
        raise FileNotFoundError(f"Expected folder not found: {extracted_data_folder}")

    result = pd.DataFrame()
    scan_names, t1c_scan_paths, t1n_scan_paths, t2f_scan_paths, t2w_scan_paths, label_paths, preprocessed_paths = [], [], [], [], [], [], []

    for sample in os.listdir(extracted_data_folder):
        path = os.path.join(extracted_data_folder, sample)
        if os.path.isdir(path) and sample.startswith("BraTS-MET-"):
            scan_base = os.path.basename(path)
            scan_names.append(scan_base)
            t1c_scan_paths.append(os.path.join(path, f"{scan_base}-{T1C_SCAN_TYPE}.nii"))
            t1n_scan_paths.append(os.path.join(path, f"{scan_base}-{T1N_SCAN_TYPE}.nii"))
            t2f_scan_paths.append(os.path.join(path, f"{scan_base}-{T2F_SCAN_TYPE}.nii"))
            t2w_scan_paths.append(os.path.join(path, f"{scan_base}-{T2W_SCAN_TYPE}.nii"))
            label_paths.append(os.path.join(path, f"{scan_base}-{LABEL_NAME}.nii"))
            preprocessed_paths.append(os.path.join(path, f"{scan_base}-{T1C_SCAN_TYPE}-{PRE_PROCESSED_IMAGE_SUFFIX}.nii"))

    result['scan_name'] = scan_names
    result['t1c_path'] = t1c_scan_paths
    result['t1n_path'] = t1n_scan_paths
    result['t2f_path'] = t2f_scan_paths
    result['t2w_path'] = t2w_scan_paths
    result['label_path'] = label_paths

    if APPLY_PRE_PROCESSING:
        result['t1c_processed_scan_path'] = result.apply(
            lambda row: apply_img_processing(row['t1c_path'], scan_type=T1C_SCAN_TYPE), axis=1)
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

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int + 1, 1, kernel_size=1), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, template_map):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        if template_map.shape[2:] != psi.shape[2:]:
            template_map = nn.functional.interpolate(template_map, size=psi.shape[2:], mode='trilinear', align_corners=False)
        psi = torch.cat([psi, template_map], dim=1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256]):
        super(AttentionUNet3D, self).__init__()
        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self._block(features[2], features[3])
        self.upconv3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(features[2], features[2], features[1])
        self.decoder3 = self._block(features[3], features[2])
        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(features[1], features[1], features[0])
        self.decoder2 = self._block(features[2], features[1])
        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(features[0], features[0], features[0]//2)
        self.decoder1 = self._block(features[1], features[0])
        self.conv_out = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, t1c_input, template_map):
        enc1 = self.encoder1(t1c_input)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        enc3 = self.att3(dec3, enc3, template_map)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2, template_map)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        enc1 = self.att1(dec1, enc1, template_map)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        return self.conv_out(dec1)

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
