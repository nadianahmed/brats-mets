import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter

import Pre_Processing.constants as constants
from Helpers.scrollable_scan_viewer import ScrollableScanViewer

def load_image(filename):
    '''
    Loads the image from the given input file.

    Parameters:
    - filename(String): the path to the file.

    Returns:
    - img(Image): the loaded image.
    - volume(ndarray): the contents of the image.
    '''
    img = nib.load(filename)
    return img, img.get_fdata()
    
def save_image(img, filename):
    '''
    Saves the given image into the given filename.

    Parameters:
    - img(Image): the iamge to save.
    - filename(String): the name of the image file to save.

    Returns:
    - path(String): the path to the saved image.
    '''
    parent_dir = os.path.dirname(filename)
    output_image_name = '{}/{}-{}.nii.gz'.format(parent_dir, 
                                                 os.path.basename(parent_dir), 
                                                 constants.PRE_PROCESSED_IMAGE_SUFFIX)
    nib.save(img, output_image_name)

    return output_image_name

def normalize_image(img):
    '''
    Normalizes the input image using the z-score method.

    Parameters:
    - img(Image): the input image

    Returns:
    - img(Image): the output image after applying the zscore method.
    '''
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std
    
def apply_threshold_contrast(img, volume, threshold=constants.THRESHOLD, scale=constants.SCALE, display_image=False):
    '''
    Applies thresholding to the given image.

    Parameters:
    - img(Image): the input image.
    - threshold(Int): the threshold used on the brightness of the image.
    - scale(Int): the scale used to increase the brightness by.
    - display_image(Bool): whether the final image should be displayed using matplotlib or not.

    Returns:
    - result(Image): the output image.
    '''
    # Normalizing the image.
    normalized_img = normalize_image(volume)

    # Applying the threshold.
    enhanced = np.copy(volume)
    threshold = np.percentile(normalized_img, 100 - threshold)
    enhanced[normalized_img <= threshold] = 0
    enhanced *= scale

    # Clipping the output to the correct range.
    enhanced_data = np.clip(enhanced, 0, np.max(enhanced))

    # Generating the output image.
    enhanced_img = nib.Nifti1Image(enhanced_data, img.affine, img.header)

    # Displaying the output image.
    if display_image:
        ScrollableScanViewer(enhanced_data, title="Thresholded Image", axis=2)
        plt.show()

    return enhanced_img