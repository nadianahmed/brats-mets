from Pre_Processing.data_preparation import extract_data
from Pre_Processing.image_analysis import apply_threshold_contrast

# Extracting the datasets
data = extract_data()

# Apply thresholding to all images.
apply_threshold_contrast(data.loc[data['scan_name'] == 'BraTS-MET-00002-000', 'scan_path'].values[0], save=True)

