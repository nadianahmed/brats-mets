from Pre_Processing.data_preparation import extract_data
from Pre_Processing.image_analysis import load_image, apply_threshold_contrast, save_image

def apply_img_processing(filename, display_img=False):
    '''
    Applies the correct pre-processing to the image.

    Parameters:
    - filename(String): the filename for the img.
    - display_img(Bool): whether to display the image or not.
    '''
    img, volume = load_image(filename=filename)
    thresholded_img = apply_threshold_contrast(img=img, volume=volume, display_image=display_img)
    processed_img_path = save_image(thresholded_img, filename=filename)
    return processed_img_path

# Extracting the datasets
data = extract_data()

# if you want a single sample
data[data['scan_name'] == 'BraTS-MET-00002-000'].apply(lambda row: apply_img_processing(row['scan_path'],
                                                                                        display_img=True), axis=1)

# Apply thresholding to all images.
# data['thresholded_scan_path'] = data.apply(lambda row: apply_img_processing(row['scan_path']), axis=1)