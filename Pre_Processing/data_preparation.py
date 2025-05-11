import os
import zipfile
import pandas as pd

import Pre_Processing.constants as constants

def extract_data():
    '''
    Extracts the data from the zip file and returns a numpy of the relevant paths.

    Returns:
    - pd.Dataframe: a table of the relevant paths.
    '''
    extracted_data_folder = constants.ROOT_FOLDER + '/' + constants.DATASET_FOLDER + constants.EXTRACTED_FOLDER_NAME
    if not os.path.isdir(extracted_data_folder):
        with zipfile.ZipFile(extracted_data_folder + '.zip', 'r') as zip_ref:
            zip_ref.extractall(constants.ROOT_FOLDER + '/' + constants.DATASET_FOLDER)

    result = pd.DataFrame()

    t1c_scan_paths = []
    t1n_scan_paths = []
    t2f_scan_paths = []
    t2w_scan_paths = []

    label_paths = []
    scan_names = []
    for sample in os.listdir(extracted_data_folder):
        if constants.PROJECT_NAME_PREFIX in sample:
            sample_folder_path = os.path.join(extracted_data_folder, sample)

            scan_names.append(sample)
            t1c_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T1C_SCAN_TYPE + '.nii.gz')
            t1n_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T1N_SCAN_TYPE + '.nii.gz')
            t2f_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T2F_SCAN_TYPE + '.nii.gz')
            t2w_scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.T2W_SCAN_TYPE + '.nii.gz')
            label_paths.append(sample_folder_path + '/' + sample + '-' + constants.LABEL_NAME + '.nii.gz')

    result['scan_name'] = scan_names
    result['t1c_path'] = t1c_scan_paths
    result['t1n_path'] = t1n_scan_paths
    result['t2f_path'] = t2f_scan_paths
    result['t2w_path'] = t2w_scan_paths
    result['label_path'] = label_paths

    return result

