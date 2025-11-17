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

# Standard Library Imports
import os
import logging

# Third Party Imports
import ants
import numpy as np
from tifffile import imwrite, imread

# Local Imports

# Start logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def export_affine_transform(
    affine_transform: ants.core.ants_transform.ANTsTransform, directory: str
) -> None:
    """Export an ants Affine Transform to disk.

    Parameters
    ----------
    affine_transform : ants.core.ants_transform.ANTsTransform
        The affine transform to export.
    directory : str
        The directory where the transform will be saved. If it does not exist, it will
        be created.

    Raises
    ------
    ValueError
        If the affine_transform is not an instance of ANTsTransform.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(affine_transform, ants.core.ants_transform.ANTsTransform):
        raise ValueError("The affine_transform must be an instance of ANTsTransform.")

    save_path = os.path.join(directory, str(affine_transform.type) + ".mat")
    ants.write_transform(affine_transform, save_path)


def export_tiff(
    image: ants.core.ants_image.ANTsImage,
    data_path: str,
    min_value: float = None,
    max_value: float = None,
    max_intensity_project: bool = False,
) -> None:
    """Export an ants.ANTsImage to a 16-bit tiff file.

    Export an ants.ANTsImage to a 16-bit tiff file. If the image min or max values
    are provided, the image will be scaled to that range. If not provided, the image
    will be scaled to its own min and max values.

    Parameters
    ----------
    image: ants.core.ants_image.ANTsImage
        Image to export
    data_path: str
        The location and name of the file to save the data to.
    min_value: float or None
        Minimum value of the reference range to map the image to. Default is None.
    max_value: float or None
        Maximum value of the reference range to map the image to. Default is None.
    max_intensity_project: bool
        If True, perform a max intensity projection along the z-axis before saving.
    """
    if isinstance(image, ants.core.ants_image.ANTsImage):
        image = image.numpy()
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported file type {type(image)}")

    # Get rid of NaNs and Infs
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image[image < 0] = 0.0

    # Determine min and max if not provided
    if min_value is None:
        min_value = image.min()

    if max_value is None:
        max_value = image.max()

    # Normalize to [0,1] with small offset.
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Map into reference range
    image = min_value + image * (max_value - min_value)

    # Convert to uint16
    image = image.astype(np.uint16)

    if max_intensity_project:
        # Max intensity project along z-axis
        image = np.max(image, axis=0)

    imwrite(data_path, image)


def import_tiff(data_path: str) -> ants.core.ants_image.ANTsImage:
    """Import a tiff file and convert to an ANTsImage.

    Parameters
    ----------
    data_path: str
        The path to the file to import.
    """
    return ants.from_numpy(imread(data_path))


def import_affine_transform(data_path: str) -> ants.core.ants_transform.ANTsTransform:
    """Import an ants Affine Transform.

    Parameters
    ----------
    data_path: str
        The path to the Affine Transform.

    Returns
    ------
    affine_transform: ants.core.ants_transform.ANTsTransform
        The affine transform.
    """
    affine_transform = ants.read_transform(data_path)
    return affine_transform


def calculate_metrics(fixed, moving, mask=None, sampling="regular", sampling_pct=1.0):
    """
    Compute normalized cross-correlation band Mattes Mutual Information etween two
    ANTsImage volumes.

    Parameters:
    -----------
    fixed : ants.ANTsImage
        The reference (fixed) image.
    moving : ants.ANTsImage
        The image to be compared or aligned.
    mask : ants.ANTsImage or None
        Optional mask applied to both fixed and moving.
    sampling : {'regular', 'random', None}
        Sampling strategy for computing metric.
    sampling_pct : float
        Fraction of voxels to sample (0–1).

    Returns:
    --------
    metric_results : dict
        Keys include 'Correlation' and 'MattesMutualInformation'. Values are metric
        results. For Correlation coefficient, range is from -1 to 1, where 1
        indicates perfect alignment.
    """

    if isinstance(fixed, np.ndarray):
        fixed = ants.from_numpy(fixed)
    if isinstance(moving, np.ndarray):
        moving = ants.from_numpy(moving)

    metric_results = {}
    for metric_type in ["Correlation", "MattesMutualInformation"]:
        value = ants.image_similarity(
            fixed,
            moving,
            metric_type=metric_type,
            fixed_mask=mask,
            moving_mask=mask,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
        )
        metric_results[metric_type] = -value
        logger.info(f"Image Metric: {metric_type}, value: {-value}")

    return metric_results
