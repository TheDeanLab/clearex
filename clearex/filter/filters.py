# Standard Library Imports

# Third Party Imports
from scipy import ndimage
import cv2
import numpy as np


# Local Imports

def dog(sigma_high: float,
        sigma_low: float,
        vol: np.ndarray[...]) -> np.ndarray[...]:
    """ Difference of Gaussian (DoG) filter.

    Parameters
    ----------
    sigma_high : float
        The standard deviation of the high-pass filter.
    sigma_low : float
        The standard deviation of the low-pass filter.
    vol : np.ndarray
        The volume to filter.

    Returns
    -------
    np.ndarray
        The DoG filtered volume.
    """
    high_pass_filtered = ndimage.gaussian_filter(vol, sigma=sigma_high, mode="nearest")
    low_pass_filtered = ndimage.gaussian_filter(vol, sigma=sigma_low, mode="nearest")
    dog_filtered = high_pass_filtered - low_pass_filtered
    return dog_filtered


def dog_cv2(sigma_high: float, sigma_low: float, vol: np.ndarray) -> np.ndarray:
    """ Difference of Gaussian (DoG) filter using OpenCV.

    Parameters
    ----------
    sigma_high : float
        The standard deviation of the high-pass filter.
    sigma_low : float
        The standard deviation of the low-pass filter.
    vol : np.ndarray
        The volume to filter.

    Returns
    -------
    np.ndarray
        The DoG filtered volume.
    """

    # Determine kernel sizes. OpenCV requires kernel sizes to be odd and positive integers.
    # A common rule of thumb is kernel_size = 6 * sigma + 1 to capture most of the Gaussian.
    ksize_high = int(6 * sigma_high + 1) | 1  # ensure odd
    ksize_low  = int(6 * sigma_low  + 1) | 1  # ensure odd

    # Apply Gaussian blur with the high sigma value.
    high_blurred = cv2.GaussianBlur(vol, (ksize_high, ksize_high), sigmaX=sigma_high, borderType=cv2.BORDER_REFLECT)
    low_blurred = cv2.GaussianBlur(vol, (ksize_low, ksize_low), sigmaX=sigma_low, borderType=cv2.BORDER_REFLECT)
    dog_result = low_blurred - high_blurred
    print("min/max high:", np.min(high_blurred), np.max(high_blurred))
    print("min/max low:", np.min(low_blurred), np.max(low_blurred))
    print("min/max dog:", np.min(dog_result), np.max(dog_result))

    return dog_result