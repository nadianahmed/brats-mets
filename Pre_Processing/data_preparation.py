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

    scan_paths = []
    label_paths = []
    scan_names = []
    for sample in os.listdir(extracted_data_folder):
        if constants.PROJECT_NAME_PREFIX in sample:
            sample_folder_path = os.path.join(extracted_data_folder, sample)

            scan_names.append(sample)
            scan_paths.append(sample_folder_path + '/' + sample + '-' + constants.CHOSEN_SCAN_TYPE + '.nii.gz')
            label_paths.append(sample_folder_path + '/' + sample + '-' + constants.LABEL_NAME + '.nii.gz')

    result['scan_name'] = scan_names
    result['scan_path'] = scan_paths
    result['label_path'] = label_paths

    return result

