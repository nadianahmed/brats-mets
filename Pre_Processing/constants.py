import os

# Paths
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_FOLDER_NAME = 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'
DATASET_FOLDER = 'Datasets/'
LABEL_NAME = 'seg'
T1C_SCAN_TYPE = 't1c'
T1N_SCAN_TYPE = 't1n'
T2F_SCAN_TYPE = 't2f'
T2W_SCAN_TYPE = 't2w'
PRE_PROCESSED_IMAGE_SUFFIX = 'processed'
PROJECT_NAME_PREFIX = 'BraTS'

# Image Thresholding
T1C_THRESHOLD_PERCENTILE=0.01 # The percentage of bright pixels to keep.
T1C_SCALE=1.0
T1N_THRESHOLD_PERCENTILE=0.01 # The percentage of bright pixels to keep.
T1N_SCALE=1.0
T2F_THRESHOLD_PERCENTILE=0.01 # The percentage of bright pixels to keep.
T2F_SCALE=1.0
T2W_THRESHOLD_PERCENTILE=0.01 # The percentage of bright pixels to keep.
T2W_SCALE=1.0

# Template Matching
T1C_TUMOUR_SIZE = 10 # Measured in milimeters
T1C_TEMPLATE_MATCH_THRESHOLD = 0.6
T1N_TUMOUR_SIZE = 10 # Measured in milimeters
T1N_TEMPLATE_MATCH_THRESHOLD = 0.6
T2F_TUMOUR_SIZE = 10 # Measured in milimeters
T2F_TEMPLATE_MATCH_THRESHOLD = 0.6
T2W_TUMOUR_SIZE = 10 # Measured in milimeters
T2W_TEMPLATE_MATCH_THRESHOLD = 0.6