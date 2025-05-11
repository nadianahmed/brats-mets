from Pre_Processing.data_preparation import extract_data
from Pre_Processing.image_analysis import load_image, apply_threshold_contrast, apply_template_matching, save_image, display_image

from Helpers.file_helper import get_image_name_from_path

RUN_ALL = False
CHOSEN_TEST_SAMPLE = 'BraTS-MET-00002-000'

def apply_img_processing(filename, show_image=False):
    '''
    Applies the correct pre-processing to the image.

    Parameters:
    - filename(String): the filename for the img.
    - show_image(Bool): whether to display the image or not.

    Returns:
    - String: the path to the resulting image.
    '''
    img = load_image(filename=filename)
    thresholded_img = apply_threshold_contrast(img=img, show_image=show_image)
    template_matched_image, match_coords_list = apply_template_matching(img=thresholded_img, show_image=show_image)
    print("Suspected tumour locations for {}: {}".format(get_image_name_from_path(filename), match_coords_list))
    display_image(template_matched_image.get_fdata(), title="Final processed image")

    processed_img_path = save_image(template_matched_image, filename=filename)
    return processed_img_path

# Extracting the datasets
data = extract_data()

if RUN_ALL:
    # Apply thresholding to all images.
    data['thresholded_scan_path'] = data.apply(lambda row: apply_img_processing(row['scan_path']), axis=1)
else:
    # try a single sample
    data[data['scan_name'] == CHOSEN_TEST_SAMPLE].apply(lambda row: apply_img_processing(row['scan_path'], show_image=True), axis=1)


