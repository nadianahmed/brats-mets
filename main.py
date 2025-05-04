from Pre_Processing.data_preparation import extract_data
from Pre_Processing.image_analysis import apply_threshold_contrast

# Extracting the datasets
data = extract_data()

# Apply thresholding to all images.
data['thresholded_scan_path'] = data.apply(lambda row: apply_threshold_contrast(row['scan_path']), axis=1)

print(data['thresholded_scan_path'][0])
