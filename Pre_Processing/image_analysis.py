import os
import Pre_Processing.constants as constants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter

import Pre_Processing.constants as constants

def normalize_image(img):
    lower = np.percentile(img, 1)
    upper = np.percentile(img, 99)
    return np.clip((img - lower) / (upper - lower), 0, 1)
    
def apply_threshold_contrast(input_file, threshold=constants.THRESHOLD, scale=constants.SCALE, display_image=False):
    '''
    Applies thresholding to the given image.

    Parameters:
    - threshold(Int): the threshold used on the brightness of the image.
    - scale(Int): the scale used to increase the brightness by.
    - display_image(Bool): whether the final image should be displayed using matplotlib or not.

    Returns:
    - result(string): path to the saved image

    '''
    img = nib.load(input_file)
    volume = img.get_fdata()

    enhanced = normalize_image(volume)
    threshold = np.percentile(enhanced, 100 - 1)
    enhanced[enhanced < threshold] = 0
   # enhanced = (enhanced - threshold) * scale
    enhanced_data = np.clip(enhanced, 0, np.max(enhanced))

    enhanced_img = nib.Nifti1Image(enhanced_data, img.affine, img.header)

    parent_dir = os.path.dirname(input_file)
    nib.save(enhanced_img, parent_dir + '/' + os.path.basename(parent_dir) + '-thresholded.nii.gz')

    if display_image:
        plt.imshow(enhanced_data[:, :, enhanced_data.shape[2] // 2], cmap='gray')
        plt.title("Threshold-Enhanced MRI Slice")
        plt.axis('off')
        plt.show()

    return parent_dir + '/' + os.path.basename(parent_dir) + '-thresholded.nii.gz'