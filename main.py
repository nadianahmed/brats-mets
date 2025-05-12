import matplotlib.pyplot as plt

from Pre_Processing.data_preparation import extract_data
from Pre_Processing.image_analysis import load_image, apply_threshold_contrast, apply_template_matching, save_image
import Pre_Processing.constants as constants

from Helpers.file_helper import get_image_name_from_path
from Helpers.scrollable_scan_viewer import ScrollableScanViewer

RUN_ALL = False
CHOSEN_TEST_SAMPLE = 'BraTS-MET-00002-000'
SCAN_TYPE = constants.T1C_SCAN_TYPE

def apply_img_processing(filename, scan_type, show_image=False):
    '''
    Applies the correct pre-processing to the image.

    Parameters:
    - filename(String): the filename for the img.
    - scan_type(String): the type of scan.
    - show_image(Bool): whether to display the image or not.

    Returns:
    - String: the path to the resulting image.
    '''
    img = load_image(filename=filename)
    if show_image:
        ScrollableScanViewer(img.get_fdata(), title="Original scan of type {}".format(scan_type))
    
    thresholded_img = apply_threshold_contrast(img=img, scan_type=scan_type, show_image=show_image)
    template_matched_image, match_coords_list = apply_template_matching(img=thresholded_img, scan_type=scan_type, show_image=show_image)
    
    print("Suspected tumour locations for {} of type {}: {}".format(get_image_name_from_path(filename), scan_type,
                                                                    match_coords_list))

    if show_image:
        ScrollableScanViewer(template_matched_image.get_fdata(), 
                             title="Final processed scan of type {}".format(scan_type))

    processed_img_path = save_image(template_matched_image, filename=filename, scan_type=scan_type)
    return processed_img_path

# Extracting the datasets
data = extract_data()

if RUN_ALL:
    # Apply thresholding to all images.
    print("Running the T1C scans:")
    data['t1c_processed_scan_path'] = data.apply(lambda row: apply_img_processing(row['t1c_path'], scan_type=constants.T1C_SCAN_TYPE), axis=1)

    print("Running the T2F scans:")
    data['t2f_processed_scan_path'] = data.apply(lambda row: apply_img_processing(row['t2f_path'], scan_type=constants.T2F_SCAN_TYPE), axis=1)

else:
    # try a single sample
    if SCAN_TYPE == constants.T1C_SCAN_TYPE:
        data[data['scan_name'] == CHOSEN_TEST_SAMPLE].apply(lambda row: apply_img_processing(row['t1c_path'], scan_type=SCAN_TYPE, show_image=True), axis=1)
    elif SCAN_TYPE == constants.T2F_SCAN_TYPE:
        data[data['scan_name'] == CHOSEN_TEST_SAMPLE].apply(lambda row: apply_img_processing(row['t2f_path'], scan_type=SCAN_TYPE, show_image=True), axis=1)
    else:
        print("DON'T USE THESE SCAN TYPES")

plt.show()