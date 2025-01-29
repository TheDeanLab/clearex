# Standard Library Imports
import numpy as np

# Third Party Imports

# Local Imports


def add_median_border(image_data):
    """Add a border to the image data with the median intensity value.

    Parameters
    ----------
    image_data : np.ndarray
        The image data to add a border to.

    Returns
    -------
    np.ndarray
        The image data with a border added.
    """
    (z_len, y_len, x_len) = image_data.shape
    median_intensity = np.median(image_data)
    padded_image_data = np.full((z_len+2, y_len+2, x_len+2), median_intensity)
    padded_image_data[1:z_len+1, 1:y_len+1, 1:x_len+1]=image_data
    return padded_image_data