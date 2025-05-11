import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass

import Pre_Processing.constants as constants
from Helpers.scrollable_scan_viewer import ScrollableScanViewer
from Helpers.file_helper import get_image_name_from_path, get_image_parent_path

def load_image(filename):
    '''
    Loads the image from the given input file.

    Parameters:
    - filename(String): the path to the file.

    Returns:
    - nib.Nifti1Image: the loaded image.
    '''
    img = nib.load(filename)
    return img
    
def save_image(img, filename, scan_type):
    '''
    Saves the given image into the given filename.

    Parameters:
    - img(Image): the iamge to save.
    - filename(String): the name of the image file to save.
    - scan_type(String): the type of scan.

    Returns:
    - String: the path to the saved image.
    '''
    output_image_name = '{}/{}-{}-{}.nii.gz'.format(get_image_parent_path(path=filename), 
                                                    get_image_name_from_path(path=filename), 
                                                    scan_type,
                                                    constants.PRE_PROCESSED_IMAGE_SUFFIX)
    nib.save(img, output_image_name)

    return output_image_name

def display_image(img, title, match_coords=None, axis=2):
    '''
    Displays the current scan using the scrollable viewer.

    Parameters:
    - img(np.ndarray): the image data to display.
    - title(String): the title for the image.
    - match_coords_list([(Int, Int, Int)]): the coordinates for the points to display (default=None).
    - axis(Int): the orientation of the image.
    '''
    ScrollableScanViewer(img, title, match_coords=match_coords, axis=axis)
    plt.show()
    
def normalize_image(img):
    '''
    Normalizes the input image using the z-score method.

    Parameters:
    - img(np.ndarray): the input image

    Returns:
    - np.ndarray: the output image after applying the zscore method.
    '''
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std
    
def apply_threshold_contrast(img, scan_type, show_image=False):
    '''
    Applies thresholding to the given image.

    Parameters:
    - img(Image): the input image.
    - scan_type(String): the type of scan.
    - show_image(Bool): whether the final image should be displayed using matplotlib or not.

    Returns:
    - nib.Nifti1Image: the output image.
    '''
    if scan_type == constants.T1C_SCAN_TYPE:
        threshold_percentile = constants.T1C_THRESHOLD_PERCENTILE
        scale = constants.T1C_SCALE
    elif scan_type == constants.T1N_SCAN_TYPE:
        threshold_percentile = constants.T1N_THRESHOLD_PERCENTILE
        scale = constants.T1N_SCALE
    elif scan_type == constants.T2F_SCAN_TYPE:
        threshold_percentile = constants.T2F_THRESHOLD_PERCENTILE
        scale = constants.T2F_SCALE
    else:
        threshold_percentile = constants.T2W_THRESHOLD_PERCENTILE
        scale = constants.T2W_SCALE

    # Get the contents of the image.
    volume = img.get_fdata()

    # Normalize the image.
    normalized_img = normalize_image(volume)

    # Apply the threshold.
    enhanced = np.copy(volume)
    threshold = np.percentile(normalized_img, (1 - threshold_percentile) * 100)
    enhanced[normalized_img <= threshold] = 0
    enhanced *= scale

    # Clip the output to the correct range.
    enhanced_data = np.clip(enhanced, 0, np.max(enhanced))

    # Generate the output image.
    output_img = nib.Nifti1Image(enhanced_data, img.affine, img.header)

    # Display the output image.
    if show_image:
        display_image(enhanced_data, title="Thresholded scan of type {}".format(scan_type))

    return output_img

def create_spherical_template(radius):
    '''
    Create a 3D spherecial template with the given radius.

    Parameters:
    - radius(Int): Radius of the sphere in voxels.

    Returns:
    - np.ndarray: 3D binary spherical template.
    '''
    shape = (2 * radius + 1,) * 3
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) // 2
    distance = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    sphere = (distance <= radius).astype(np.float32)
    return gaussian_filter(sphere, sigma=1)

def apply_template_matching(img, scan_type, show_image=False):
    '''
    Applies thresholding to the given image.

    Parameters:
    - img(Image): the input image.
    - scan_type(String): the type of scan.
    - show_image(Bool): whether the final image should be displayed using matplotlib or not.

    Returns:
    - nib.Nifti1Image: the output image.
    - [(Int, Int, Int)]: a list of the possible template matches.
    '''
    if scan_type == constants.T1C_SCAN_TYPE:
        radius = constants.T1C_TUMOUR_SIZE
        threshold = constants.T1C_TEMPLATE_MATCH_THRESHOLD
    elif scan_type == constants.T1N_SCAN_TYPE:
        radius = constants.T1N_TUMOUR_SIZE
        threshold = constants.T1N_TEMPLATE_MATCH_THRESHOLD
    elif scan_type == constants.T2F_SCAN_TYPE:
        radius = constants.T2F_TUMOUR_SIZE
        threshold = constants.T2F_TEMPLATE_MATCH_THRESHOLD
    else:
        radius = constants.T2W_TUMOUR_SIZE
        threshold = constants.T2W_TEMPLATE_MATCH_THRESHOLD
    
    # Get the contents of the image.
    volume = img.get_fdata()

    # Converting milimeters to voxels.
    radius_voxels = round(radius/(img.header.get_zooms()[0]))

    # Generate a spherical template.
    template = create_spherical_template(radius=radius_voxels)
    if template.shape[0] > volume.shape[0] or \
       template.shape[1] > volume.shape[1] or \
       template.shape[2] > volume.shape[2]:
        raise ValueError("Template must be smaller than the volume in all dimensions.")
    
    # Find correlations to the template.
    correlation = match_template(volume, template, pad_input=True)

    # Find matches based on the correlations.
    local_max = maximum_filter(correlation, size=template.shape) == correlation
    correlation_threshold = threshold * np.max(correlation)
    detected_peaks = (correlation > correlation_threshold) & local_max
    labeled, num_features = label(detected_peaks)
    match_coords_list = center_of_mass(detected_peaks, labeled, range(1, num_features + 1))
    match_coords_list = [tuple(map(int, coords)) for coords in match_coords_list]

    # Display the matches.
    if show_image:
        display_image(volume, title="Template matched scan of type {}".format(scan_type), match_coords=match_coords_list)

    masked = np.zeros_like(volume)

    # Only keep the matches.
    for x, y, z in match_coords_list:
        # Define bounding box for sphere
        z_min = max(z - radius_voxels, 0)
        z_max = min(z + radius_voxels + 1, volume.shape[0])
        y_min = max(y - radius_voxels, 0)
        y_max = min(y + radius_voxels + 1, volume.shape[1])
        x_min = max(x - radius_voxels, 0)
        x_max = min(x + radius_voxels + 1, volume.shape[2])

        xx, yy, zz = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
        dist = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
        mask = dist <= radius_voxels

        # Apply spherical mask to copy values
        masked[x_min:x_max, y_min:y_max, z_min:z_max][mask] = \
            volume[x_min:x_max, y_min:y_max, z_min:z_max][mask]
        
    # Generate the output image.
    output_img = nib.Nifti1Image(masked, img.affine, img.header)

    return output_img, match_coords_list
