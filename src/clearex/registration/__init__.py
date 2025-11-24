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


class ImageRegistration:
    """
    A class for performing image registration using linear and nonlinear transformations.

    This class handles the complete registration workflow for whole images, including
    loading images, performing linear (affine) registration, followed by nonlinear
    (deformable) registration. It manages transform caching and logging.

    Attributes
    ----------
    fixed_image_path : str or Path
        Path to the fixed reference image.
    moving_image_path : str or Path
        Path to the moving image to be registered.
    save_directory : str or Path
        Directory where transformation results are saved.
    imaging_round : int
        The round number for identifying transformation files.
    crop : bool
        Whether to crop the moving image before registration.
    enable_logging : bool
        Whether to enable logging.
    _log : Logger
        Logger instance for this registration.
    _image_opener : ImageOpener
        Image opener for loading image data.
    """

    def __init__(
        self,
        fixed_image_path: str | Path = None,
        moving_image_path: str | Path = None,
        save_directory: str | Path = None,
        imaging_round: int = 0,
        crop: bool = False,
        enable_logging: bool = True,
    ):
        """
        Initialize the ImageRegistration instance.

        Parameters
        ----------
        fixed_image_path : str or Path, optional
            The path to the fixed reference image (TIFF file).
        moving_image_path : str or Path, optional
            The path to the moving image to be registered (TIFF file).
        save_directory : str or Path, optional
            Directory where the transformation results will be saved.
        imaging_round : int, optional
            The round number used to identify transformation files (default is 0).
        crop : bool, optional
            Whether to crop the moving image before registration (default is False).
        enable_logging : bool, optional
            Whether to enable logging (default is True).
        """
        self.fixed_image_path = fixed_image_path
        self.moving_image_path = moving_image_path
        self.save_directory = save_directory
        self.imaging_round = imaging_round
        self.crop = crop
        self.enable_logging = enable_logging
        self._log = None
        self._image_opener = ImageOpener()

    def register(
        self,
        fixed_image_path: str | Path = None,
        moving_image_path: str | Path = None,
        save_directory: str | Path = None,
        imaging_round: int = None,
    ) -> None:
        """
        Register a moving image to a fixed image using linear and nonlinear transformations.

        Parameters
        ----------
        fixed_image_path : str or Path, optional
            The path to the fixed reference image. If not provided, uses the instance attribute.
        moving_image_path : str or Path, optional
            The path to the moving image. If not provided, uses the instance attribute.
        save_directory : str or Path, optional
            Directory where results will be saved. If not provided, uses the instance attribute.
        imaging_round : int, optional
            The round number for transformation files. If not provided, uses the instance attribute.

        Returns
        -------
        None
            The function saves the registration results to disk.

        Raises
        ------
        FileNotFoundError
            If any of the provided file paths do not exist.
        ValueError
            If required paths are not provided either as arguments or instance attributes.
        """
        # Use provided values or fall back to instance attributes
        fixed_path = fixed_image_path or self.fixed_image_path
        moving_path = moving_image_path or self.moving_image_path
        save_dir = save_directory or self.save_directory
        round_num = imaging_round if imaging_round is not None else self.imaging_round

        # Validate required parameters
        if not all([fixed_path, moving_path, save_dir]):
            raise ValueError(
                "fixed_image_path, moving_image_path, and save_directory must be provided "
                "either as arguments or instance attributes."
            )

        # Confirm that the save_directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Initialize logging
        self._log = initialize_logging(
            log_directory=save_dir, enable_logging=self.enable_logging
        )
        self._log.info(f"Image registration performed with antspyx: {ants.__version__}")

        # Load the fixed image data and convert to AntsImage
        fixed_image, _ = self._image_opener.open(fixed_path, prefer_dask=False)
        fixed_image = ants.from_numpy(fixed_image)
        self._log.info(f"Loaded fixed image {fixed_path}. Shape: {fixed_image.shape}.")

        # Load the moving image data
        moving_image, _ = self._image_opener.open(moving_path, prefer_dask=False)
        self._log.info(f"Loaded moving image {moving_path}. Shape: {moving_image.shape}.")

        # Optionally crop the moving data to reduce computation time
        moving_image = crop_data(self._log, self.crop, round_num, moving_image, save_dir)

        # Perform Linear Registration
        moving_linear_aligned = self._perform_linear_registration(
            fixed_image, moving_image, round_num, save_dir
        )

        # Perform Nonlinear Registration
        self._perform_nonlinear_registration(
            fixed_image, moving_linear_aligned, round_num, save_dir
        )

    def _perform_linear_registration(
        self, fixed_image, moving_image, imaging_round: int, save_directory: str | Path
    ) -> Any:
        """
        Perform linear (affine) registration.

        Parameters
        ----------
        fixed_image : ants.ANTsImage
            The fixed reference image.
        moving_image : numpy.ndarray or ants.ANTsImage
            The moving image to be registered.
        imaging_round : int
            The round number for identifying the transformation file.
        save_directory : str or Path
            Directory where the transformation will be saved.

        Returns
        -------
        ants.ANTsImage
            The linearly registered moving image.
        """
        linear_transformation_path = os.path.join(
            save_directory, f"linear_transform_{imaging_round}.mat"
        )

        if os.path.exists(linear_transformation_path):
            self._log.info(
                f"Linear transformation already exists at {linear_transformation_path}. "
                "Skipping registration."
            )
            # Read the transform from disk
            transform = ants.read_transform(linear_transformation_path)

            # Resample the registered image to the target image
            moving_linear_aligned = transform.apply_to_image(
                image=moving_image,
                reference=fixed_image,
                interpolation='linear'
            )
        else:
            self._log.info("Performing Linear Registration...")
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
                self._log.info(stdout)
            if stderr:
                self._log.error(stderr)

            ants.write_transform(transform, linear_transformation_path)
            self._log.info(f"Linear Transform written to: {linear_transformation_path}")

        return moving_linear_aligned

    def _perform_nonlinear_registration(
        self, fixed_image, moving_linear_aligned, imaging_round: int, save_directory: str | Path
    ) -> None:
        """
        Perform nonlinear (deformable) registration.

        Parameters
        ----------
        fixed_image : ants.ANTsImage
            The fixed reference image.
        moving_linear_aligned : ants.ANTsImage
            The linearly registered moving image.
        imaging_round : int
            The round number for identifying the transformation file.
        save_directory : str or Path
            Directory where the transformation will be saved.
        """
        nonlinear_transformation_path = os.path.join(
            save_directory, f"nonlinear_warp_{imaging_round}.nii.gz"
        )

        if os.path.exists(nonlinear_transformation_path):
            self._log.info(
                f"Nonlinear transformation already exists at "
                f"{nonlinear_transformation_path}. Skipping registration."
            )
        else:
            self._log.info("Beginning Nonlinear Registration...")
            result, stdout, stderr = capture_c_level_output(
                nonlinear_registration,
                moving_image=moving_linear_aligned,
                fixed_image=fixed_image,
                accuracy="high",
                verbose=True,
            )
            nonlinear_transformed, transform_path = result
            if stdout:
                self._log.info(stdout)
            if stderr:
                self._log.error(stderr)

            shutil.copyfile(transform_path, nonlinear_transformation_path)
            self._log.info(f"Nonlinear warp written to: {nonlinear_transformation_path}")


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

    This is a convenience function that creates an ImageRegistration instance and
    calls its register method. For more control, consider using the ImageRegistration
    class directly.

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

    Examples
    --------
    >>> from clearex.registration import register_round
    >>> register_round(
    ...     fixed_image_path="reference.tif",
    ...     moving_image_path="round_1.tif",
    ...     save_directory="./results",
    ...     imaging_round=1
    ... )
    """
    registrar = ImageRegistration(
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        save_directory=save_directory,
        imaging_round=imaging_round,
        crop=crop,
        enable_logging=enable_logging,
    )
    registrar.register()


