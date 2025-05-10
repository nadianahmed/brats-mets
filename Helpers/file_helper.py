import os


def get_image_parent_path(path):
    '''
    Returns the parent path for the image.

    Parameters:
    - path(String): the path to the image.

    Returns:
    - String: the path to the parent directory of the image.
    '''
    return os.path.dirname(path)

def get_image_name_from_path(path):
    '''
    Extracts the image name from its path.

    Parameters:
    - path(String): the path to the image.

    Returns:
    - String: the name of the image.
    '''
    return os.path.basename(get_image_parent_path(path=path))