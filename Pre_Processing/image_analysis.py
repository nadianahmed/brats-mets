import os
import Pre_Processing.constants as constants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter

def apply_threshold_contrast(input_file, threshold, scale=1.0, save=False, display_image=False):
    img = nib.load(input_file)
    volume = img.get_fdata()

    enhanced = np.copy(volume)
    enhanced[enhanced < threshold] = 0
    enhanced = (enhanced - threshold) * scale
    enhanced_data = np.clip(enhanced, 0, np.max(enhanced))

    enhanced_img = nib.Nifti1Image(enhanced_data, img.affine, img.header)

    if save:
        parent_dir = os.path.basename(os.path.dirname(input_file))
        print(parent_dir)
        nib.save(enhanced_img, constants.ROOT_FOLDER + '-thresholded.nii.gz')

    if display_image:
        plt.imshow(enhanced_data[:, :, enhanced_data.shape[2] // 2], cmap='gray')
        plt.title("Threshold-Enhanced MRI Slice")
        plt.axis('off')
        plt.show()