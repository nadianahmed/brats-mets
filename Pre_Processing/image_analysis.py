import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter

import Pre_Processing.constants as constants

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

    # Normalizing the image.
    normalized_img = normalize_image(volume)

    # Applying the threshold.
    enhanced = np.copy(volume)
    threshold = np.percentile(normalized_img, 100 - threshold)
    enhanced[normalized_img <= threshold] = 0

    # Clipping the output to the correct range.
    enhanced_data = np.clip(enhanced, 0, np.max(enhanced))

    # Generating the output image.
    enhanced_img = nib.Nifti1Image(enhanced_data, img.affine, img.header)

    # Saving the output image.
    parent_dir = os.path.dirname(input_file)
    output_image_name = parent_dir + '/' + os.path.basename(parent_dir) + '-thresholded.nii.gz'
    nib.save(enhanced_img, output_image_name)

    # Displaying the output image.
    if display_image:
        plt.imshow(enhanced_data[:, :, enhanced_data.shape[2] // 2], cmap='gray')
        plt.title('Threshold-Enhanced MRI Slice')
        plt.axis('off')
        plt.show()

    return output_image_name