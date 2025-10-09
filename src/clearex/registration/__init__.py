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
import logging
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

# Third Party Imports
import tifffile
import ants
import numpy as np

# Local Imports
import clearex
import clearex.registration.common
import clearex.registration.linear as linear
import clearex.registration.nonlinear as nonlinear
from clearex.io.log import (
    initialize_logging,
    log_and_echo as log,
    capture_c_level_output,
)
from clearex.file_operations.tools import (
    crop_overlapping_datasets,
    identify_minimal_bounding_box,
)
from clearex.io.read import ImageOpener
from clearex.registration.common import export_tiff


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def register_round(
    fixed_image_path: str | Path,
    moving_image_path: str | Path,
    save_directory: str | Path,
    imaging_round: int = 0,
    crop: bool = False,
    enable_logging: bool = True,
) -> None:
    """
    Register a moving image to a fixed image using linear and nonlinear transformations.

    Parameters
    ----------
    fixed_image_path : str or Path
        The path to the fixed reference image (TIFF file).
    moving_image_path : str or Path
        The path to the moving image to be registered (TIFF file).
    save_directory : str or Path
        Directory where the transformation results will be saved.
    imaging_round : int, optional
        The round number used to identify transformation files (default is 0).
    crop : bool, optional
        Whether to crop the moving image before registration (default is False).
    enable_logging : bool, optional
        Whether to enable logging (default is True).

    Returns
    -------
    None
        The function saves the registration results to disk and does not return anything.

    Raises
    ------
    FileNotFoundError
        If any of the provided file paths do not exist.
    """

    # Confirm that the save_directory exists.
    os.makedirs(save_directory, exist_ok=True)

    _log = initialize_logging(
        log_directory=save_directory, enable_logging=enable_logging
    )

    # Load the fixed image data and convert to AntsImage...
    image_opener = ImageOpener()
    fixed_image, _ = image_opener.open(fixed_image_path, prefer_dask=False)
    fixed_image = ants.from_numpy(fixed_image)
    _log.info(f"Loaded fixed image {fixed_image_path}. Shape: {fixed_image.shape}.")

    # Load the moving image data
    moving_image, _ = image_opener.open(moving_image_path, prefer_dask=False)
    _log.info(f"Loaded moving image {moving_image_path}. Shape: {moving_image.shape}.")

    # Only crop the moving data since the fixed data defines the coordinate space.
    if crop:
        z0, z1, y0, y1, x0, x1 = identify_minimal_bounding_box(
            image=moving_image,
            down_sampling=4,
            robust=True,
            lower_pct=0.1,
            upper_pct=99.9,
        )
        moving_image = ants.from_numpy(moving_image)
        moving_image = ants.crop_indices(
            moving_image, lowerind=(z0, y0, x0), upperind=(z1, y1, x1)
        )
        _log.info(f"Moving image cropped to: {moving_image.shape}.")
    else:
        moving_image = ants.from_numpy(moving_image)

    _log.info("Performing Linear Registration...")
    result, stdout, stderr = capture_c_level_output(
        clearex.registration.linear.register_image,
        moving_image=moving_image,
        fixed_image=fixed_image,
        registration_type="TRSAA",
        accuracy="low",
        verbose=True,
    )
    moving_image, transform = result
    if stdout:
        _log.info(stdout)
    if stderr:
        _log.error(stderr)

    linear_transformation_path = os.path.join(
        save_directory, f"linear_transform_{imaging_round}.mat"
    )
    ants.write_transform(transform, linear_transformation_path)
    _log.info(f"Linear Transform written to: {linear_transformation_path}")

    _log.info("Beginning Nonlinear Registration...")
    result, stdout, stderr = capture_c_level_output(
        clearex.registration.nonlinear.register_image,
        moving_image=moving_image,
        fixed_image=fixed_image,
        accuracy="high",
        verbose=True,
    )
    nonlinear_transformed, transform_path = result
    if stdout:
        _log.info(stdout)
    if stderr:
        _log.error(stderr)

    nonlinear_transformation_path = os.path.join(
        save_directory, f"nonlinear_warp_{imaging_round}.nii.gz"
    )
    shutil.copyfile(transform_path, nonlinear_transformation_path)
    _log.info(f"Linear Transform written to: {nonlinear_transformation_path}")


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

    print("Linear Transformation Path:", linear_transformation_path)
    print("Nonlinear Transformation Path;", nonlinear_transformation_path)

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
    print(f"Loaded {moving_image_path}. Shape: {moving_image.shape}.")
    nonlinear_transformed = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=[linear_transformation_path, nonlinear_transformation_path],
        whichtoinvert=[False, False],  # set True for any transform you want inverted
        interpolator="linear",  # use "nearestNeighbor" for label images
    )
    print("Transformation Complete. Shape:", nonlinear_transformed.shape)

    transformed_image_path = os.path.join(save_directory, moving_image_path.name)
    export_tiff(nonlinear_transformed, min_value, max_value, transformed_image_path)
    print("Image saved to:", transformed_image_path)
