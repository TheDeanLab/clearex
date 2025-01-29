# Standard Library Imports

# Third Party Imports
from skimage.filters import threshold_otsu
import numpy as np
import dask.array as da
from scipy import ndimage
from skimage.morphology import dilation, erosion
import skimage.filters as filters
import skimage.measure as measure

from filter.kernels import gaussian_kernel, second_derivative_gaussian_kernel

# Local Imports
from filter.filters import dog

# Local Imports
from filter.kernels import make_3d_structured_element

# Local Imports

def difference_of_gaussian(vol, sigma_high=50, sigma_low=200, down_sampling=4,
                          n_classes=3):
    """
    vol: a 3D NumPy array of shape (Z, Y, X).
    Returns a 3D labeled volume of shape (Z, Y, X).
    """

    # Difference of Gaussian
    dog_filtered = dog(sigma_high, sigma_low, vol)
    sub = dog_filtered[::down_sampling, ::down_sampling, ::down_sampling]
    thresholds = filters.threshold_multiotsu(sub, classes=n_classes)
    threshold = thresholds[-1] if len(thresholds) > 1 else thresholds[0]
    binary = dog_filtered >= threshold
    return binary


def otsu(image_data, down_sampling=4):
    """Apply Otsu thresholding to a 3D image.

    Parameters
    ----------
    image_data : np.ndarray
        The image data to threshold.
    down_sampling : int
        The factor to downsample the image data by.

    Returns
    -------
    image_binary : np.ndarray
        The thresholded image data.
    """
    threshold = threshold_otsu(
        image_data[::down_sampling, ::down_sampling, ::down_sampling])
    image_binary = image_data > threshold
    return image_binary

def noise_normalized_otsu(image_data, down_sampling=4):
    """Normalized Otsu Thresholding for 3D images.

    Identifies the threshold for the image data using Otsu's method and normalizes
    the image data by subtracting the threshold and dividing by the standard deviation.

    Parameters
    ----------
    image_data : np.ndarray
        The image data to threshold and normalize.
    down_sampling : int
        The factor to downsample the image data by.

    Returns
    -------
    np.ndarray
        The normalized image data.
    """
    image_threshold = threshold_otsu(image_data[::down_sampling, ::down_sampling, ::down_sampling])
    normalized_cell = (image_data - image_threshold) / np.std(image_data - image_threshold)
    return normalized_cell


def gamma_otsu(image_data, gamma=0.5, gaussian_kernel=1, down_sampling=4):
    """Apply gamma correction and Otsu thresholding to a 3D image.

    Applies a gamma correction to the image data, followed by a Gaussian blur,
    Otsu thresholding, dilation, erosion, and a final Gaussian filter.

    Parameters
    ----------
    image_data : np.ndarray
        The image data to threshold and normalize.
    gamma : float
        The gamma value to apply to the image data.
    gaussian_kernel : float
        The standard deviation of the Gaussian kernel to apply to the image data.
    down_sampling : int
        The factor to downsample the image data by.

    Returns
    -------
    np.ndarray
        The thresholded image data.
    """

    if type(image_data) == np.ndarray:
        image_data = da.from_array(image_data)

    # Gamma Correction
    image_data = image_data ** gamma

    # Gaussian Blur
    blurred = image_data.map_overlap(
        lambda x: gaussian_filter(x, sigma=gaussian_kernel, order=0, mode="nearest"),
        depth=10
    )
    blur_image = blurred.compute()

    # Otsu Threshold
    image_threshold = threshold_otsu(
        blur_image[::down_sampling, ::down_sampling, ::down_sampling])
    image_binary = blurred.map_blocks(
        lambda block: block > image_threshold,
        dtype=bool)

    # Dilation Operation
    dilate_radius = 2
    structured_element = make_3d_structured_element(dilate_radius, shape='sphere')
    image_binary = image_binary.map_overlap(
        lambda block: dilation(block, structured_element),
        depth=10)

    # Fill holes
    image_binary = image_binary.map_overlap(
        lambda block: ndimage.binary_fill_holes(block),
        depth=10
    )
    # Erosion Operation
    erode_radius = 2
    structured_element = make_3d_structured_element(erode_radius, shape='sphere')
    image_binary = image_binary.map_overlap(
        lambda block: erosion(block, structured_element),
        depth=10).astype(np.double)

    # Final filtering operation
    image_binary = image_binary.map_overlap(
        lambda x: gaussian_filter(x, sigma=1, order=0, mode="nearest"),
        depth=10).compute()
    return image_binary