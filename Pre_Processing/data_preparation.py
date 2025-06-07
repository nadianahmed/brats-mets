import os
import zipfile
import pandas as pd

from Pre_Processing.image_analysis import load_image, apply_threshold_contrast, apply_template_matching, save_image
import Pre_Processing.constants as constants

from Helpers.progress_bar import print_progress_bar
from Helpers.file_helper import get_image_name_from_path
from Helpers.scrollable_scan_viewer import ScrollableScanViewer

def extract_data(apply_preprocessing, scan_type):
    '''
    Extracts the data from the zip file and returns a numpy of the relevant paths.

    Parameters:
    - apply_preprocessing(Bool): whether to apply preprocessing or not.
    - scan_type(String): the type of scan to analyze.

    Returns:
    - pd.Dataframe: a table of the relevant paths.
    '''
    extracted_data_folder = constants.ROOT_FOLDER + '/' + constants.DATASET_FOLDER + constants.EXTRACTED_FOLDER_NAME
    if not os.path.isdir(extracted_data_folder):
        print(f"📦 Extracting the dataset zipfile..")
        with zipfile.ZipFile(extracted_data_folder + '.zip', 'r') as zip_ref:
            zip_ref.extractall(constants.ROOT_FOLDER + '/' + constants.DATASET_FOLDER)

    result = pd.DataFrame()

    t1c_scan_paths = []
    t1n_scan_paths = []
    t2f_scan_paths = []
    t2w_scan_paths = []

    label_paths = []
    scan_names = []
    preprocessed_paths = []
    
    print(f"⏳ Loading the data..")
    for sample in os.listdir(extracted_data_folder):
        if constants.PROJECT_NAME_PREFIX in sample:
            sample_folder_path = os.path.join(extracted_data_folder, sample)

            scan_names.append(sample)
            t1c_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T1C_SCAN_TYPE + '.nii.gz')
            t1n_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T1N_SCAN_TYPE + '.nii.gz')
            t2f_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T2F_SCAN_TYPE + '.nii.gz')
            t2w_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T2W_SCAN_TYPE + '.nii.gz')
            label_paths.append(sample_folder_path + '/' + sample + '-' + constants.LABEL_NAME + '.nii.gz')
            preprocessed_paths.append(os.path.join(sample_folder_path, f"{sample}-{scan_type}-{constants.PRE_PROCESSED_IMAGE_SUFFIX}.nii.gz"))

    result['scan_name'] = scan_names
    result['t1c_path'] = t1c_scan_paths
    result['t1n_path'] = t1n_scan_paths
    result['t2f_path'] = t2f_scan_paths
    result['t2w_path'] = t2w_scan_paths
    result['label_path'] = label_paths
    if apply_preprocessing:
        print(f"🧪 Applying pre-processing for {scan_type} scans...")
        processed_paths = []
        print_progress_bar(0, len(scan_names))
        for i, path in enumerate(result[scan_type + '_path']):
            print_progress_bar(i+1, len(scan_names))
            processed = apply_pre_processing(path, scan_type=scan_type)
            processed_paths.append(processed)
        result[scan_type + '_processed_scan_path'] = processed_paths
    else:
        result[scan_type + '_processed_scan_path'] = preprocessed_paths

    return result

def apply_pre_processing(filename, scan_type, show_image=False):
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
    
    if show_image:
        print("Suspected tumour locations for {} of type {}: {}".format(get_image_name_from_path(filename), scan_type, match_coords_list))
        ScrollableScanViewer(template_matched_image.get_fdata(), 
                             title="Final processed scan of type {}".format(scan_type))

    processed_img_path = save_image(template_matched_image, filename=filename, scan_type=scan_type)
    return processed_img_path