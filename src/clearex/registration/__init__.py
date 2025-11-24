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

"""
Image registration module.

This module provides high-level registration functions that combine
linear and nonlinear transformations. Implementation details are in
submodules:
- linear: Linear/affine registration
- nonlinear: Deformable registration
- common: Shared utilities
"""

# Environment setup
import os

# Limit internal threading in BLAS/ITK/etc. to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

# Standard Library Imports
import shutil
from pathlib import Path
from logging import Logger
from typing import Any

# Third Party Imports
import ants

# Local Imports
from clearex.registration.linear import register_image as linear_registration
from clearex.registration.nonlinear import register_image as nonlinear_registration
from clearex.io.log import (
    initialize_logging,
    capture_c_level_output,
)
from clearex.io.read import ImageOpener
from clearex.registration.common import export_tiff, crop_data


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

    # Initialize logging
    _log = initialize_logging(
        log_directory=save_directory, enable_logging=enable_logging
    )
    _log.info(f"Image registration performed with antspyx: {ants.__version__}")

    # Load the fixed image data and convert to AntsImage...
    image_opener = ImageOpener()
    fixed_image, _ = image_opener.open(fixed_image_path, prefer_dask=False)
    fixed_image = ants.from_numpy(fixed_image)
    _log.info(f"Loaded fixed image {fixed_image_path}. Shape: {fixed_image.shape}.")

    # Load the moving image data
    moving_image, _ = image_opener.open(moving_image_path, prefer_dask=False)
    _log.info(f"Loaded moving image {moving_image_path}. Shape: {moving_image.shape}.")

    # Optionally crop the moving data to reduce computation time
    moving_image = crop_data(_log, crop, imaging_round, moving_image, save_directory)

    # Perform Linear Registration
    moving_linear_aligned = _linear_registration(_log, fixed_image, imaging_round,
                                                 moving_image, save_directory)

    # Perform Nonlinear Registration
    _nonlinear_registration(_log, fixed_image, imaging_round, moving_linear_aligned,
                            save_directory)


def _nonlinear_registration(_log: Logger, fixed_image, imaging_round: int | None,
                            moving_linear_aligned, save_directory: str | Path):
    # Perform Nonlinear Registration
    nonlinear_transformation_path = os.path.join(
        save_directory, f"nonlinear_warp_{imaging_round}.nii.gz"
    )
    if os.path.exists(nonlinear_transformation_path):
        _log.info(
            f"Nonlinear transformation already exists at "
            f"{nonlinear_transformation_path}. Skipping registration."
        )
        nonlinear_transform = ants.read_transform(nonlinear_transformation_path)

        # Resample the registered image to the target image.

    else:
        _log.info("Beginning Nonlinear Registration...")
        result, stdout, stderr = capture_c_level_output(
            nonlinear_registration,
            moving_image=moving_linear_aligned,
            fixed_image=fixed_image,
            accuracy="high",
            verbose=True,
        )
        nonlinear_transformed, transform_path = result
        if stdout:
            _log.info(stdout)
        if stderr:
            _log.error(stderr)

        shutil.copyfile(transform_path, nonlinear_transformation_path)
        _log.info(f"Nonlinear warp written to: {nonlinear_transformation_path}")


def _linear_registration(_log: Logger, fixed_image, imaging_round: int | None,
                         moving_image, save_directory: str | Path) -> Any:
    linear_transformation_path = os.path.join(
        save_directory, f"linear_transform_{imaging_round}.mat"
    )
    if os.path.exists(linear_transformation_path):
        _log.info(
            f"Linear transformation already exists at {linear_transformation_path}. "
            "Skipping registration."
        )
        # Read the transform from disk.
        transform = ants.read_transform(linear_transformation_path)

        # Resample the registered image to the target image.
        moving_linear_aligned = transform.apply_to_image(
            image=moving_image,
            reference=fixed_image,
            interpolation='linear'
        )
    else:
        _log.info("Performing Linear Registration...")
        result, stdout, stderr = capture_c_level_output(
            linear_registration,
            moving_image=moving_image,
            fixed_image=fixed_image,
            registration_type="TRSAA",
            accuracy="low",
            verbose=True,
        )
        moving_linear_aligned, transform = result
        if stdout:
            _log.info(stdout)
        if stderr:
            _log.error(stderr)

        ants.write_transform(transform, linear_transformation_path)
        _log.info(f"Linear Transform written to: {linear_transformation_path}")
    return moving_linear_aligned


