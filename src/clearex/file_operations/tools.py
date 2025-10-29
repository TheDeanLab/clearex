#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

# Standard imports
import sys
import pickle
import os
from pathlib import Path
from typing import Any, Iterable
from typing import Union, Tuple

# Third-party imports
import ants
import numpy as np
from numpy import ndarray

# Local imports
from clearex.segmentation.otsu import otsu


def get_moving_image_paths(directory: str | Path, idx: int) -> list[Path]:
    """
    Get a list of .tif and .tiff image paths for a given round index in a directory.

    Parameters
    ----------
    directory : str or Path
        The directory to search for image files.
    idx : int
        The round index to match in the filenames.

    Returns
    -------
    list of Path
        List of matching image file paths.
    """

    directory = Path(directory)

    # Match both .tif and .tiff
    tif_files = list(directory.glob(f"Round{idx}_*.tif"))

    tiff_files = list(directory.glob(f"Round{idx}_*.tiff"))
    return tif_files + tiff_files


def get_variable_size(variable: Any) -> float:
    """Get the size of a variable in MB.

    Parameters
    ----------
    variable : Any
        The variable to get the size of.

    Returns
    -------
    float
        The size of the variable in MB.
    """
    return sys.getsizeof(variable) / 1024**2


def save_variable_to_disk(variable: Any, path: str) -> None:
    """Save a variable to disk.

    Parameters
    ----------
    variable : Any
        The variable to save.
    path : str
        The path to save the variable to.

    Returns
    -------
    None
    """
    with open(path, "wb") as f:
        pickle.dump(variable, f)


def load_variable_from_disk(path: str) -> Any:
    """Load a variable from disk.

    Parameters
    ----------
    path : str
        The path to load the variable from.

    Returns
    -------
    Any
        The loaded variable

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def delete_filetype(data_path: str, filetype: str) -> None:
    """Delete any files with the designated filetype in the specified directory.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the files.
    filetype : str
        The filetype to delete. E.g. 'pdf'

    Returns
    -------
    None
    """
    if filetype[0] != ".":
        filetype = "." + filetype

    files = [f for f in os.listdir(data_path) if f.endswith(filetype)]
    for file in files:
        os.remove(os.path.join(data_path, file))


def get_roi_indices(
    image: np.ndarray, roi_size: int = 256
) -> Tuple[int, int, int, int, int, int]:
    """Get indices for a centered ROI of size roi_size x roi_size x roi_size

    Parameters
    ----------
    image : np.ndarray
        The input image from which to extract the ROI.
    roi_size : int, optional
        The size of the ROI to extract from the center of the image. Default is 256.

    Returns
    -------
    tuple
        A tuple containing the start and end indices for each dimension (z_start, z_end, y_start, y_end, x_start, x_end).
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3D array.")
    if roi_size <= 0:
        raise ValueError("ROI size must be a positive integer.")
    if roi_size > min(image.shape):
        raise ValueError(
            "ROI size must be less than or equal to the smallest dimension of the image."
        )

    # Calculate the start and end indices for the ROI
    distance = roi_size // 2
    b = image.shape
    z_start = b[0] // 2 - distance
    z_end = b[0] // 2 + distance
    y_start = b[1] // 2 - distance
    y_end = b[1] // 2 + distance
    x_start = b[2] // 2 - distance
    x_end = b[2] // 2 + distance

    # Ensure indices are within bounds
    z_start = max(0, z_start)
    z_end = min(b[0], z_end)
    y_start = max(0, y_start)
    y_end = min(b[1], y_end)
    x_start = max(0, x_start)
    x_end = min(b[2], x_end)
    return z_start, z_end, y_start, y_end, x_start, x_end


def identify_robust_bounding_box(
    binary: np.ndarray, lower_pct: float = 5, upper_pct: float = 95
) -> Tuple[int, int, int, int, int, int]:
    """Compute a robust bounding box from binary 3D mask by ignoring outliers.

    Parameters
    ----------
    binary : np.ndarray
        3D binary array (dtype=bool or 0/1).
    lower_pct : float
        Lower percentile cutoff (e.g., 5).
    upper_pct : float
        Upper percentile cutoff (e.g., 95).

    Returns
    -------
    (z0, z1, y0, y1, x0, x1) : Tuple of ints
        Bounding box indices in the form: [z_start:z_end, y_start:y_end, x_start:x_end]
    """

    assert binary.ndim == 3, "Input must be a 3D array."
    binary = binary.astype(np.uint8)

    # Sum across two axes to get marginal sums
    z_dist = binary.sum(axis=(1, 2)).astype(np.int32)
    y_dist = binary.sum(axis=(0, 2)).astype(np.int32)
    x_dist = binary.sum(axis=(0, 1)).astype(np.int32)

    def find_bounds(dist, low, high):
        cumsum = np.cumsum(dist)
        total = cumsum[-1]
        if total == 0:
            raise ValueError("No foreground found.")
        lower_idx = np.searchsorted(cumsum, total * (low / 100))
        upper_idx = np.searchsorted(cumsum, total * (high / 100))
        return lower_idx, upper_idx + 1  # upper bound is exclusive

    z0, z1 = find_bounds(z_dist, lower_pct, upper_pct)
    y0, y1 = find_bounds(y_dist, lower_pct, upper_pct)
    x0, x1 = find_bounds(x_dist, lower_pct, upper_pct)

    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def identify_minimal_bounding_box(
    image: Union[np.ndarray, ants.core.ants_image.ANTsImage],
    down_sampling: int = 8,
    robust=False,
    lower_pct: float = 5,
    upper_pct: float = 95,
) -> Tuple[int, int, int, int, int, int]:
    """Identify the minimal bounding box that encloses foreground signal in a 3D
    image using Otsu thresholding.

    This function performs downsampling-based Otsu thresholding to detect foreground
    voxels, computes the minimal bounding box in the downsampled space, and scales
    the result back to the original image resolution.

    Parameters
    ----------
    image : np.ndarray or ants.core.ants_image.ANTsImage
        A 3D image in either NumPy array or ANTsImage format. If an ANTsImage is
        provided, it will be converted to a NumPy array internally.

    down_sampling : int, optional
        Downsampling factor applied to the input image before Otsu thresholding.
        Used to speed up thresholding and bounding box computation. Default is 8.

    robust : bool
        Run the robust boundary detection method. Default is False.

    lower_pct : float
        Lower percentage for cut-off for robust boundary detection. Values between 0
        and 100 and valid.

    upper_pct : float
        Upper percentage for cut-off for robust boundary detection. Values between 0
        and 100 and valid.
    Returns
    -------
    z_start : int
        Starting index (inclusive) of the bounding box along the z-axis in full resolution.
    z_end : int
        Ending index (exclusive) of the bounding box along the z-axis in full resolution.
    y_start : int
        Starting index (inclusive) of the bounding box along the y-axis in full resolution.
    y_end : int
        Ending index (exclusive) of the bounding box along the y-axis in full resolution.
    x_start : int
        Starting index (inclusive) of the bounding box along the x-axis in full resolution.
    x_end : int
        Ending index (exclusive) of the bounding box along the x-axis in full resolution.

    Raises
    ------
    TypeError
        If the input image is not a NumPy array or ANTsImage.
    ValueError
        If no foreground signal is detected in the thresholded binary image.

    Notes
    -----
    The bounding box is returned in `(z_start, z_end, y_start, y_end, x_start,
    x_end)` order and is compatible with slicing: `image[z_start:z_end,
    y_start:y_end, x_start:x_end]`.

    Examples
    --------
    >>> bbox = identify_minimal_bounding_box(image, down_sampling=8)
    >>> z0, z1, y0, y1, x0, x1 = bbox
    >>> cropped = image[z0:z1, y0:y1, x0:x1]
    """

    # Make sure that the input image is a np.ndarray
    if isinstance(image, ants.core.ants_image.ANTsImage):
        image = image.numpy().astype(np.uint16)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(
            f"Unsupported file type: {type(image)}. Supported types are: "
            f"np.ndarray and ANTsImage"
        )

    # Otsu Thresholding
    binary_data = otsu(image_data=image, down_sampling=down_sampling)
    if robust:
        bounding_box = identify_robust_bounding_box(
            binary_data, lower_pct=lower_pct, upper_pct=upper_pct
        )
        z_start, z_end, y_start, y_end, x_start, x_end = bounding_box
        return z_start, z_end, y_start, y_end, x_start, x_end

    # Get coordinates of foreground voxels in binary mask
    coords = np.argwhere(binary_data)
    if coords.size == 0:
        raise ValueError("No foreground detected in binary image.")

    # Compute bounding box in down sampled space
    min_idx = coords.min(axis=0)
    max_idx = coords.max(axis=0) + 1

    # Ensure bounds do not exceed original image shape
    max_idx = np.minimum(max_idx, image.shape)

    z_start, y_start, x_start = min_idx
    z_end, y_end, x_end = max_idx

    return z_start, z_end, y_start, y_end, x_start, x_end


def merge_bounding_boxes(
    box1: Tuple[int, int, int, int, int, int],
    box2: Tuple[int, int, int, int, int, int],
) -> Tuple[slice, slice, slice]:
    """Compute a minimal bounding box that encompasses two input bounding boxes.

    This function takes two bounding boxes in the form of 6-element tuples
    (z_start, z_end, y_start, y_end, x_start, x_end), and returns a new bounding box
    that spans both inputs. It ensures start coordinates are minimized and end
    coordinates are maximized to fully include both regions. The output is returned
    as a tuple of `slice` objects for direct use in array indexing.

    Parameters
    ----------
    box1 : tuple of int or np.integer
        Bounding box coordinates in the format
        (z_start, z_end, y_start, y_end, x_start, x_end).
    box2 : tuple of int or np.integer
        Another bounding box to merge, in the same format as `box1`.

    Returns
    -------
    bounding_box_slices : tuple of slice
        A merged bounding box that fully encompasses both `box1` and `box2`,
        returned as a tuple of `slice` objects in the order (z, y, x), and
        ready for use in array slicing.

    Examples
    --------
    >>> bounding_box_1 = (10, 50, 20, 60, 30, 80)
    >>> bounding_box_2 = (15, 55, 10, 70, 25, 85)
    >>> merged_slices = merge_bounding_boxes(bounding_box_1, bounding_box_2)
    >>> image[merged_slices]  # Direct indexing into a 3D image
    """
    merged_coords = tuple(
        min(a, b) if i % 2 == 0 else max(a, b)
        for i, (a, b) in enumerate(zip(box1, box2))
    )
    merged_coords = tuple(int(x) for x in merged_coords)

    z_start, z_end, y_start, y_end, x_start, x_end = merged_coords
    return (
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end),
    )


def crop_overlapping_datasets(
    fixed_roi: Union[np.ndarray, ants.core.ants_image.ANTsImage],
    transformed_image: Union[np.ndarray, ants.core.ants_image.ANTsImage],
    robust: bool = False,
    lower_pct: float = 5,
    upper_pct: float = 95,
) -> tuple[
    ndarray[tuple[int, ...], Any],
    ndarray[tuple[int, ...], Any],
    Iterable | tuple[slice],
]:
    """Crop two 3D images to the maximal overlapping bounding box containing
    foreground signal.

    This function computes the minimal foreground bounding boxes for each input image
    and then determines a merged bounding box that encompasses both regions. Each image
    is then cropped to this merged region. Foreground detection is based on intensity
    thresholding (e.g., Otsu), with an optional robust mode to reduce sensitivity to
    outliers.

    Parameters
    ----------
    fixed_roi : np.ndarray or ants.core.ants_image.ANTsImage
        The fixed image or ROI volume. Must be a 3D image.

    transformed_image : np.ndarray or ants.core.ants_image.ANTsImage
        The transformed moving image aligned to the fixed ROI. Must also be 3D.

    robust : bool, optional
        If True, uses a robust method to detect foreground and exclude outliers when
        computing the bounding boxes. If False, uses strict minimum and maximum
        coordinates. Default is False.

    lower_pct : float, optional
        Lower percentage for cut-off for robust boundary detection. Values between 0
        and 100 and valid. Default is 5.

    upper_pct : float, optional
        Upper percentage for cut-off for robust boundary detection. Values between 0
        and 100 and valid. Default is 95.

    Returns
    -------
    fixed_cropped : np.ndarray
        Cropped version of the fixed image limited to the shared bounding box.

    transformed_cropped : np.ndarray
        Cropped version of the transformed image limited to the shared bounding box.

    bounding_box : np.ndarray
        The bounding box used to slice the data.

    Raises
    ------
    ValueError
        If no foreground is detected in either image.

    Notes
    -----
    If input images are ANTsImage objects, they will be converted to NumPy arrays (
    dtype uint16) before cropping. The bounding box is computed in `(z_start:z_end,
    y_start:y_end, x_start:x_end)` order.

    Examples
    --------
    >>> fixed_crop, moving_crop, bounding_box = crop_overlapping_datasets(fixed_img, moving_img, robust=True, lower_pct=2, upper_pct=98)
    """
    # Identify z_start, z_end, y_start, y_end, x_start, x_end, for each image.
    minimum_bounding_box_fixed = identify_minimal_bounding_box(
        fixed_roi, robust=robust, lower_pct=lower_pct, upper_pct=upper_pct
    )
    minimum_bounding_box_moving = identify_minimal_bounding_box(
        transformed_image, robust=robust, lower_pct=lower_pct, upper_pct=upper_pct
    )

    # Find the maximum overlapping region.
    bounding_box = merge_bounding_boxes(
        minimum_bounding_box_moving, minimum_bounding_box_fixed
    )

    # Convert to numpy to crop the data.
    if isinstance(fixed_roi, ants.core.ants_image.ANTsImage):
        fixed_roi = fixed_roi.numpy().astype(np.uint16)
    fixed_roi = fixed_roi[bounding_box]

    if isinstance(transformed_image, ants.core.ants_image.ANTsImage):
        transformed_image = transformed_image.numpy().astype(np.uint16)
    transformed_image = transformed_image[bounding_box]

    return fixed_roi, transformed_image, bounding_box
