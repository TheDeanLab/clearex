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
from typing import Any, Optional, Union, Tuple, List
import warnings
from dataclasses import dataclass

# Third Party Imports
import ants
import numpy as np

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
        self._image_opener = ImageOpener()

        # Initialize logging
        self._log = initialize_logging(
            log_directory=self.save_directory, enable_logging=enable_logging
        )
        self._log.info(f"Image registration performed with antspyx: {ants.__version__}")

        # Validate required parameters
        if not all([self.fixed_image_path, self.moving_image_path, self.save_directory]):
            raise ValueError(
                "fixed_image_path, moving_image_path, and save_directory must be provided "
                "either as arguments or instance attributes."
            )

        # Confirm that the save_directory exists
        os.makedirs(self.save_directory, exist_ok=True)

        # Load the image data as numpy array.
        self.fixed_image, self.fixed_image_info = self._image_opener.open(
            self.fixed_image_path,prefer_dask=False)
        self.moving_image, self.moving_image_info = self._image_opener.open(
            self.moving_image_path, prefer_dask=False)

        # Report the loaded image shapes.
        self._log.info(f"Loaded fixed image {self.fixed_image_path}. "
                       f"Shape: {self.fixed_image.shape}.")
        self._log.info(f"Loaded moving image {self.moving_image_path}. "
                       f"Shape: {self.moving_image.shape}.")


        # Optionally crop the data to reduce computation time.
        self.fixed_image = crop_data(
            self._log,
            self.crop,
            imaging_round,
            self.fixed_image,
            save_directory,
            image_type="fixed"
        )

        # Optionally crop the moving data to reduce computation time
        self.moving_image = crop_data(
            self._log,
            self.crop,
            imaging_round,
            self.moving_image,
            save_directory,
            image_type="moving"
        )



    def register(self) -> None:
        """
        Register a moving image to a fixed image using linear and nonlinear transformations.

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
        # Perform Linear Registration
        moving_linear_aligned = self._perform_linear_registration(
            fixed_image=self.fixed_image,
            moving_image=self.moving_image,
            imaging_round=self.imaging_round,
            save_directory=self.save_directory
        )

        # Perform Nonlinear Registration
        self._perform_nonlinear_registration(
            fixed_image=self.fixed_image,
            moving_linear_aligned=moving_linear_aligned,
            imaging_round=self.imaging_round,
            save_directory=self.save_directory
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
            self,
            fixed_image: ants.ANTsImage,
            moving_linear_aligned: ants.ANTsImage,
            imaging_round: int,
            save_directory: str | Path
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

@dataclass
class ChunkInfo:
    """Information about a chunk in the volume."""

    chunk_id: int

    # Actual data region (without overlap)
    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    # Extended region (with overlap for registration)
    z_start_ext: int
    z_end_ext: int
    y_start_ext: int
    y_end_ext: int
    x_start_ext: int
    x_end_ext: int

    # Transform paths
    linear_transform_path: Optional[Path] = None
    nonlinear_transform_path: Optional[Path] = None

class ChunkedImageRegistration(ImageRegistration):
    """
    A class for performing image registration using linear and nonlinear transformations.

    This class handles the complete registration workflow for chunked images, including
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
        Initialize the ChunkedImageRegistration instance.

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
        # Initialize the parent class
        super().__init__(
            fixed_image_path,
            moving_image_path,
            save_directory,
            imaging_round,
            crop,
            enable_logging
        )

        #  Chunking parameters    
        self.chunk_size: Tuple[int, int, int] = (256, 256, 256)  # (z, y, x)
        self.overlap_fraction: float = 0.15
        
        # Compute the chunk grid.
        self._compute_chunk_grid(self.fixed_image_info.shape)

        # Create directory for chunk transforms
        self.chunks_dir = self.save_directory / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)

    def _perform_nonlinear_registration(
            self,
            fixed_image: ants.ANTsImage,
            moving_linear_aligned: ants.ANTsImage,
            imaging_round: int,
            save_directory: str | Path
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
        self._log.info("Beginning Chunked Nonlinear Registration...")

        # Create an empty array to store the numpy transformed chunks to
        registered_image = np.zeros(self.fixed_image.shape)

        # Create an empty array to store the warp field
        nonlinear_transform = np.zeros(
            (*self.fixed_image.shape, 3),
            dtype=self.fixed_image.numpy().dtype)


        # Process each chunk
        for i, chunk in enumerate(self.chunk_info_list):
            self._log.info(f"Chunk {i+1}/{len(self.chunk_info_list)}")
            updated_chunk, local_warp = self.register_chunk(
                chunk=chunk,
                fixed_image=self.fixed_image,
                moving_image=moving_linear_aligned,
                nonlinear_type="SyNOnly",
                nonlinear_accuracy="low",
            )
            self.chunk_info_list[i] = updated_chunk

            # Place the registered chunk into the correct location in the full image
            registered_image[
                chunk.z_start:chunk.z_end,
                chunk.y_start:chunk.y_end,
                chunk.x_start:chunk.x_end
            ] = updated_chunk

            # Place the local warp into the correct location in the full warp field
            nonlinear_transform[
                chunk.z_start:chunk.z_end,
                chunk.y_start:chunk.y_end,
                chunk.x_start:chunk.x_end,
                :
            ] = local_warp[
                chunk.z_start - chunk.z_start_ext:
                chunk.z_end - chunk.z_start_ext,
                chunk.y_start - chunk.y_start_ext:
                chunk.y_end - chunk.y_start_ext,
                chunk.x_start - chunk.x_start_ext:
                chunk.x_end - chunk.x_start_ext,
                :
            ]

        self._log.info("All chunks registered!")

        nonlinear_transformation_path = os.path.join(
            save_directory, f"nonlinear_warp_{imaging_round}.nii.gz"
        )

        nonlinear_image_path = os.path.join(
            save_directory, f"nonlinear_registered_{imaging_round}.tif"
        )

        # Save the full registered image and warp field
        export_tiff(
            image=registered_image,
            data_path=nonlinear_image_path,
            min_value=self.moving_image.min(),
            max_value=self.moving_image.max(),
        )
        self._log.info(f"Nonlinear registered image written to: {nonlinear_image_path}")

        # Convert the warp field to an ANTs image and save
        nonlinear_transform = ants.from_numpy(nonlinear_transform)
        ants.write_transform(nonlinear_transform, nonlinear_transformation_path)
        self._log.info(f"Nonlinear warp written to: {nonlinear_transformation_path}")

    def _compute_chunk_grid(
            self,
            image_shape
    ) -> List[ChunkInfo]:
        """
        Compute the grid of overlapping chunks.

        Parameters
        ----------
        image_shape : tuple of int
            Shape of the full image (z, y, x).

        Returns
        -------
        list of ChunkInfo
            List of chunk information objects.
        """
        self._log.info("Computing Chunk Grid")

        # Unpack and convert shape dimensions to Python int
        shape_z, shape_y, shape_x = image_shape
        z_max: int = int(shape_z)
        y_max: int = int(shape_y)
        x_max: int = int(shape_x)

        # Unpack chunk size
        chunk_z, chunk_y, chunk_x = self.chunk_size

        # Calculate overlap in pixels
        overlap_z: int = int(chunk_z * self.overlap_fraction)
        overlap_y: int = int(chunk_y * self.overlap_fraction)
        overlap_x: int = int(chunk_x * self.overlap_fraction)

        self._log.info(f"Image shape: {image_shape}")
        self._log.info(f"Chunk size: {self.chunk_size}")
        self._log.info(f"Overlap: ({overlap_z}, {overlap_y}, {overlap_x})")

        chunks = []
        chunk_id = 0

        # Calculate step size (chunk size minus overlap)
        step_z: int = int(chunk_z - overlap_z)
        step_y: int = int(chunk_y - overlap_y)
        step_x: int = int(chunk_x - overlap_x)

        # Iterate through the volume
        z_positions = list(range(0, z_max, step_z))  # type: ignore[arg-type]
        y_positions = list(range(0, y_max, step_y))  # type: ignore[arg-type]
        x_positions = list(range(0, x_max, step_x))  # type: ignore[arg-type]

        for z_start in z_positions:
            for y_start in y_positions:
                for x_start in x_positions:
                    # Core chunk boundaries (without overlap)
                    z_end = min(z_start + chunk_z, z_max)
                    y_end = min(y_start + chunk_y, y_max)
                    x_end = min(x_start + chunk_x, x_max)

                    # Extended boundaries (with overlap)
                    z_start_ext = max(0, z_start - overlap_z // 2)
                    z_end_ext = min(z_max, z_end + overlap_z // 2)
                    y_start_ext = max(0, y_start - overlap_y // 2)
                    y_end_ext = min(y_max, y_end + overlap_y // 2)
                    x_start_ext = max(0, x_start - overlap_x // 2)
                    x_end_ext = min(x_max, x_end + overlap_x // 2)

                    chunk = ChunkInfo(
                        chunk_id=chunk_id,
                        z_start=z_start,
                        z_end=z_end,
                        y_start=y_start,
                        y_end=y_end,
                        x_start=x_start,
                        x_end=x_end,
                        z_start_ext=z_start_ext,
                        z_end_ext=z_end_ext,
                        y_start_ext=y_start_ext,
                        y_end_ext=y_end_ext,
                        x_start_ext=x_start_ext,
                        x_end_ext=x_end_ext,
                    )
                    chunks.append(chunk)
                    chunk_id += 1

        self._log.info(f"Created {len(chunks)} chunks")
        self._log.info(
            f"Grid dimensions: {len(z_positions)} x {len(y_positions)} x {len(x_positions)}"
        )

        self.chunk_info_list = chunks
        return chunks


    def register_chunk(
        self,
        chunk: ChunkInfo,
        fixed_image: ants.ANTsImage,
        moving_image: ants.ANTsImage,
        nonlinear_type: str = "SyNOnly",
        nonlinear_accuracy: str = "high",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register a single chunk with fine registration.

        Parameters
        ----------
        chunk : ChunkInfo
            Information about the chunk to register.
        fixed_image : ants.ANTsImage
            The full fixed image.
        moving_image : ants.ANTsImage
            The full moving image.
        nonlinear_type : str, optional
            Type of nonlinear registration (default: "SyNOnly").
        nonlinear_accuracy : str, optional
            Accuracy level for nonlinear registration (default: "high").

        Returns
        -------
        moving_chunk : np.ndarray
            Updated chunk with transform applied.
        warp : np.ndarray
            The warp field for the chunk.
        """
        self._log.info(f"Processing chunk {chunk.chunk_id}...")

        # Extract chunk from images (using extended boundaries)
        fixed_chunk = ants.crop_indices(
            fixed_image,
            lowerind=(chunk.z_start_ext, chunk.y_start_ext, chunk.x_start_ext),
            upperind=(chunk.z_end_ext, chunk.y_end_ext, chunk.x_end_ext)
        )
        moving_chunk = ants.crop_indices(
            moving_image,
            lowerind=(chunk.z_start_ext, chunk.y_start_ext, chunk.x_start_ext),
            upperind=(chunk.z_end_ext, chunk.y_end_ext, chunk.x_end_ext)
        )

        # Pre-registration checks.
        fixed_chunk_np = fixed_chunk.numpy()
        moving_chunk_np = moving_chunk.numpy()

        # 1. Check for sufficient variance
        moving_std = np.std(moving_chunk_np)
        fixed_std = np.std(fixed_chunk_np)
        if moving_std < 1e-6 or fixed_std < 1e-6:
            message = (
                f"Block has insufficient variance (moving_std={moving_std:.2e}, "
                f"fixed_std={fixed_std:.2e}). Skipping registration."
            )
            self._log.warning(message)
            warnings.warn(message)
            return moving_chunk_np

        # 2. Check for NaN or inf values
        if np.any(np.isnan(moving_chunk_np)) or np.any(np.isnan(fixed_chunk_np)):
            message = (
                "Block contains NaN values. Skipping registration."
            )
            self._log.warning(message)
            warnings.warn(message)
            return moving_chunk_np

        # 3. Check for inf values
        if np.any(np.isinf(moving_chunk_np)) or np.any(np.isinf(fixed_chunk_np)):
            message = (
                "Block contains inf values. Skipping registration."
            )
            self._log.warning(message)
            warnings.warn(message)
            return moving_chunk_np

        # Perform fine nonlinear registration on this chunk
        self._log.info("  Performing fine nonlinear registration...")
        result, stdout, stderr = capture_c_level_output(
            nonlinear_registration,
            moving_image=moving_chunk,
            fixed_image=fixed_chunk,
            registration_type=nonlinear_type,
            accuracy=nonlinear_accuracy,
            verbose=False,
        )
        nonlinear_transformed, local_transform_path = result
        if stdout:
            self._log.info(stdout)
        if stderr:
            self._log.error(stderr)

        # Save the local transform
        local_warp_path = self.chunks_dir / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz"
        shutil.copy2(local_transform_path, local_warp_path)

        # Update chunk info with transform path
        chunk.nonlinear_transform_path = local_warp_path
        self._log.info(f"  Local transform saved: {local_warp_path}")

        # Convert transformed chunk back to numpy array
        nonlinear_transformed = nonlinear_transformed.numpy()

        # Strip the overlap to return only the core chunk
        z_start_rel = chunk.z_start - chunk.z_start_ext
        z_end_rel = z_start_rel + (chunk.z_end - chunk.z_start)
        y_start_rel = chunk.y_start - chunk.y_start_ext
        y_end_rel = y_start_rel + (chunk.y_end - chunk.y_start)
        x_start_rel = chunk.x_start - chunk.x_start_ext
        x_end_rel = x_start_rel + (chunk.x_end - chunk.x_start)

        nonlinear_transformed = nonlinear_transformed[
            z_start_rel:z_end_rel,
            y_start_rel:y_end_rel,
            x_start_rel:x_end_rel
        ]

        # Read the nonlinear transform
        warp = ants.image_read(str(local_warp_path)).numpy()

        # Strip the overlap from the warp as well
        warp = warp[
            z_start_rel:z_end_rel,
            y_start_rel:y_end_rel,
            x_start_rel:x_end_rel,
            :
        ]

        return nonlinear_transformed, warp

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


