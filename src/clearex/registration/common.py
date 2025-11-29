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
from logging import Logger
from pathlib import Path
from typing import Any, Sequence, Optional
import json

# Third Party Imports
import ants
import numpy as np
from numpy import ndarray
from tifffile import imwrite, imread

# Local Imports
from clearex.file_operations.tools import identify_minimal_bounding_box
from clearex.io.read import ImageOpener

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


def crop_data(
    logging_instance: Logger,
    crop: bool | None,
    imaging_round: int | None,
    image: ndarray[Any, Any],
    save_directory: str | os.PathLike[str],
    image_type: str = "",
) -> Any:
    """
    Crop an image to its minimal bounding box or load existing crop indices.

    Identifies and applies a minimal bounding box around non-background voxels
    in the image. Crop indices are saved to disk for reproducibility and reuse.

    Parameters
    ----------
    logging_instance : Logger
        Logger instance for recording crop operations.
    crop : bool or None
        Whether to perform cropping. If False or None, returns the image as an
        ANTsImage without cropping.
    imaging_round : int or None
        The round number used to identify the crop indices file. If None, the
        file is named without a round suffix.
    image : ndarray
        The image array to be cropped.
    save_directory : str or os.PathLike[str]
        Directory where crop indices JSON file will be saved or loaded from.
    image_type : str, optional
        Type of image being cropped, either "fixed" or "moving", used for naming
        the crop indices file. Default is "".

    Returns
    -------
    ants.core.ants_image.ANTsImage
        The cropped image as an ANTsImage (if crop=True), or the original image
        converted to ANTsImage (if crop=False).

    Notes
    -----
    Crop indices are saved as JSON files in the format:
    - `{image_type}_crop_indices_{imaging_round}.json` (if imaging_round is provided)
    - `{image_type}_crop_indices.json` (if imaging_round is None)
    """

    # Only crop the moving data since the fixed data defines the coordinate space.
    if crop:
        # # Convert to AntsImage for cropping
        # image = ants.from_numpy(image)

        if imaging_round is None:
            json_path = os.path.join(save_directory, f"{image_type}_crop_indices.json")
        else:
            json_path = os.path.join(
                save_directory, f"{image_type}_crop_indices_{imaging_round}.json"
            )

        if os.path.exists(json_path):
            logging_instance.info(f"Loading existing crop indices from: {json_path}")
            with open(json_path, "r") as f:
                z0, z1, y0, y1, x0, x1 = json.load(f)
        else:
            logging_instance.info("Identifying minimal bounding box for cropping...")
            z0, z1, y0, y1, x0, x1 = identify_minimal_bounding_box(
                image=image,
                down_sampling=4,
                robust=True,
                lower_pct=0.1,
                upper_pct=99.9,
            )

            # Save the bounding box coordinates for future reference
            with open(json_path, "w") as f:
                json.dump([z0, z1, y0, y1, x0, x1], f)
            logging_instance.info(f"Crop indices saved to: {json_path}")

        # image = ants.crop_indices(
        #     image, lowerind=(z0, y0, x0), upperind=(z1, y1, x1)
        # )

        image = image[z0:z1, y0:y1, x0:x1]
        logging_instance.info(f"Moving image cropped to: {image.shape}.")

    else:
        logging_instance.info("Cropping not enabled; proceeding without cropping.")

    return ants.from_numpy(image)


def bulk_transform_directory(
    fixed_image_path: str | Path,
    moving_image_paths: Sequence[str | Path],
    save_directory: str | Path,
    imaging_round: int,
) -> None:
    """
    Apply linear and nonlinear transformations to a list of moving images and save the results.

    Parameters
    ----------
    fixed_image_path : str or Path
        Path to the fixed reference image (TIFF file).
    moving_image_paths : Sequence[str or Path]
        List of paths to moving images to be transformed.
    save_directory : str or Path
        Directory where the transformed images will be saved.
    imaging_round : int
        The imaging_round number used to identify transformation files.

    Returns
    -------
    None
        The function saves the transformed images to disk and does not return anything.

    Raises
    ------
    FileNotFoundError
        If any of the required transformation files do not exist.
    """

    # Paths to transforms on disk
    linear_transformation_path = os.path.join(
        save_directory, f"linear_transform_{imaging_round}.mat"
    )
    nonlinear_transformation_path = os.path.join(
        save_directory, f"nonlinear_warp_{imaging_round}.nii.gz"
    )
    for p in (linear_transformation_path, nonlinear_transformation_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Convert to logging
    print("Linear Transformation Path:", linear_transformation_path)
    print("Nonlinear Transformation Path:", nonlinear_transformation_path)

    for moving_image_path in moving_image_paths:
        transform_image(
            fixed_image_path,
            linear_transformation_path,
            moving_image_path,
            nonlinear_transformation_path,
            save_directory,
        )


def transform_image(
    fixed_image_path: str | Path,
    linear_transformation_path: str | Path,
    moving_image_path: str | Path,
    nonlinear_transformation_path: str | Path,
    save_directory: str | Path,
) -> None:
    """
    Apply linear and nonlinear transformations to a moving image and save the result.

    Parameters
    ----------
    fixed_image_path : str or Path
        The path to the fixed reference image (TIFF file).
    linear_transformation_path : str or Path
        The path to the linear (affine) transformation file (.mat).
    moving_image_path : str or Path
        The path to the moving image to be transformed (TIFF file).
    nonlinear_transformation_path : str or Path
        The path to the nonlinear (warp) transformation file (.nii.gz).
    save_directory : str or Path
        Directory where the transformed image will be saved.

    Returns
    -------
    None
        The function saves the transformed image to disk and does not return anything.

    Raises
    ------
    FileNotFoundError
        If any of the provided file paths do not exist.
    """

    # Ensure all input parameters are Path objects
    fixed_image_path = (
        Path(fixed_image_path)
        if isinstance(fixed_image_path, str)
        else fixed_image_path
    )
    linear_transformation_path = (
        Path(linear_transformation_path)
        if isinstance(linear_transformation_path, str)
        else linear_transformation_path
    )
    moving_image_path = (
        Path(moving_image_path)
        if isinstance(moving_image_path, str)
        else moving_image_path
    )
    nonlinear_transformation_path = (
        Path(nonlinear_transformation_path)
        if isinstance(nonlinear_transformation_path, str)
        else nonlinear_transformation_path
    )
    save_directory = (
        Path(save_directory) if isinstance(save_directory, str) else save_directory
    )

    # Load the fixed image data and convert to AntsImage...
    image_opener = ImageOpener()
    fixed_image, _ = image_opener.open(fixed_image_path, prefer_dask=False)
    fixed_image = ants.from_numpy(fixed_image)

    moving_image, _ = image_opener.open(moving_image_path, prefer_dask=False)
    min_value = float(moving_image.min())
    max_value = float(moving_image.max())
    moving_image = ants.from_numpy(moving_image)

    # TODO: Use proper logging
    print(f"Loaded {moving_image_path}. Shape: {moving_image.shape}.")

    # Transform the data
    # ``ants.apply_transforms`` applies transforms in reverse order, i.e. the
    # last entry in the list is executed first.  For transforms generated by
    # ``ants.registration`` the forward transform order should therefore be
    # the nonlinear warp followed by the affine matrix to ensure that the
    # deformation field and affine offsets are composed correctly.

    nonlinear_transformed = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=[
            str(nonlinear_transformation_path),
            str(linear_transformation_path),
        ],
        whichtoinvert=[False, False],
        interpolator="linear",  # use "nearestNeighbor" for label images
    )
    print("Transformation Complete. Shape:", nonlinear_transformed.shape)

    transformed_image_path = os.path.join(save_directory, moving_image_path.name)
    export_tiff(
        nonlinear_transformed,
        transformed_image_path,
        min_value,
        max_value,
    )
    print("Image saved to:", transformed_image_path)
