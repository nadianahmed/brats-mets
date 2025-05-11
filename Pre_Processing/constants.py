import os

# Paths
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_FOLDER_NAME = 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'
DATASET_FOLDER = 'Datasets/'
LABEL_NAME = 'seg'
CHOSEN_SCAN_TYPE = 't1c'
PRE_PROCESSED_IMAGE_SUFFIX = 'processed'
PROJECT_NAME_PREFIX = 'BraTS'

# Image Thresholding
THRESHOLD=1 # The percentage of bright pixels to keep.
SCALE=1.0

# Template Match
TUMOUR_SIZE = 10 # Measured in milimeters
TEMPLATE_MATCH_THRESHOLD = 0.6