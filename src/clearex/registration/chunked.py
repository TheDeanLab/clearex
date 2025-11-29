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

"""Chunked registration for large-scale image volumes using multi-scale strategy."""

# Standard Library Imports
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third Party Imports
import ants
import dask.array as da
import numpy as np
import zarr

# Local Imports
from clearex.io.read import ImageOpener
from clearex.registration.linear import register_image as linear_registration
from clearex.registration.nonlinear import register_image as nonlinear_registration

# Start logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


class ChunkedRegistration:
    """
    Multi-scale chunked registration for large image volumes.

    This class implements a hierarchical registration strategy:
    1. Coarse global registration on downsampled data
    2. Fine registration on overlapping chunks
    3. Transform composition and boundary blending

    Parameters
    ----------
    fixed_image_path : str or Path
        Path to the fixed reference image.
    moving_image_path : str or Path
        Path to the moving image to be registered.
    output_dir : str or Path
        Directory to save all outputs (transforms, chunks, etc.).
    downsample_factor : int, optional
        Factor to downsample for coarse registration (default: 4).
    chunk_size : int or tuple of int, optional
        Size of chunks for fine registration. Can be a single int (cubic chunks)
        or a tuple (z, y, x). Default: 256.
    overlap_fraction : float, optional
        Fraction of chunk size to use for overlap (default: 0.15, i.e., 15%).
    prefer_dask : bool, optional
        Whether to load images as Dask arrays (default: True).
    """

    def __init__(
        self,
        fixed_image_path: Union[str, Path],
        moving_image_path: Union[str, Path],
        output_dir: Union[str, Path],
        downsample_factor: int = 4,
        chunk_size: Union[int, Tuple[int, int, int]] = 256,
        overlap_fraction: float = 0.15,
        prefer_dask: bool = True,
    ):
        self.fixed_image_path = Path(fixed_image_path)
        self.moving_image_path = Path(moving_image_path)
        self.output_dir = Path(output_dir)
        self.downsample_factor = downsample_factor
        self.overlap_fraction = overlap_fraction
        self.prefer_dask = prefer_dask

        # Ensure chunk_size is a tuple
        if isinstance(chunk_size, int):
            self.chunk_size = (chunk_size, chunk_size, chunk_size)
        else:
            self.chunk_size = chunk_size

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transforms_dir = self.output_dir / "transforms"
        self.transforms_dir.mkdir(exist_ok=True)
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)

        # Initialize image opener
        self.image_opener = ImageOpener()

        # Storage for transforms and chunks
        self.global_linear_transform_path: Optional[Path] = None
        self.global_nonlinear_transform_path: Optional[Path] = None
        self.chunk_info_list: List[ChunkInfo] = []

        logger.info("ChunkedRegistration initialized:")
        logger.info(f"  Fixed: {self.fixed_image_path}")
        logger.info(f"  Moving: {self.moving_image_path}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Downsample factor: {self.downsample_factor}")
        logger.info(f"  Chunk size: {self.chunk_size}")
        logger.info(f"  Overlap fraction: {self.overlap_fraction}")

    def _downsample_for_coarse_registration(
        self, image: np.ndarray, factor: int
    ) -> np.ndarray:
        """
        Downsample a 3D image by a given factor using local averaging.

        This avoids loading the full resolution into ANTs until after downsampling.

        Parameters
        ----------
        image : np.ndarray
            The input 3D image.
        factor : int
            The downsampling factor.

        Returns
        -------
        np.ndarray
            The downsampled image.
        """
        if factor == 1:
            return image

        # Use block averaging for downsampling
        # For Dask arrays, use coarsen
        if isinstance(image, da.Array):
            downsampled = da.coarsen(
                np.mean, image, {0: factor, 1: factor, 2: factor}, trim_excess=True
            ).compute()
        else:
            # For NumPy arrays, use simple slicing (faster than scipy.zoom)
            downsampled = image[::factor, ::factor, ::factor]

        logger.info(f"Downsampled from {image.shape} to {downsampled.shape}")
        return downsampled

    def perform_coarse_registration(
        self,
        linear_type: str = "Affine",
        nonlinear_type: str = "SyNOnly",
        linear_accuracy: str = "low",
        nonlinear_accuracy: str = "medium",
    ) -> Tuple[Path, Path]:
        """
        Perform coarse global registration on downsampled data.

        Parameters
        ----------
        linear_type : str, optional
            Type of linear registration (default: "Affine").
        nonlinear_type : str, optional
            Type of nonlinear registration (default: "SyNOnly").
        linear_accuracy : str, optional
            Accuracy level for linear registration (default: "low").
        nonlinear_accuracy : str, optional
            Accuracy level for nonlinear registration (default: "medium").

        Returns
        -------
        linear_path : Path
            Path to the saved linear transform.
        nonlinear_path : Path
            Path to the saved nonlinear transform.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Coarse Global Registration on Downsampled Data")
        logger.info("=" * 60)

        # Load images (potentially as Dask arrays)
        logger.info("Loading fixed image...")
        fixed_image, fixed_info = self.image_opener.open(
            self.fixed_image_path, prefer_dask=self.prefer_dask
        )

        logger.info("Loading moving image...")
        moving_image, moving_info = self.image_opener.open(
            self.moving_image_path, prefer_dask=self.prefer_dask
        )

        logger.info(f"Fixed image shape: {fixed_info.shape}")
        logger.info(f"Moving image shape: {moving_info.shape}")

        # Downsample before converting to ANTs
        logger.info(f"Downsampling by factor {self.downsample_factor}...")
        fixed_downsampled = self._downsample_for_coarse_registration(
            fixed_image, self.downsample_factor
        )
        moving_downsampled = self._downsample_for_coarse_registration(
            moving_image, self.downsample_factor
        )

        # Now convert to ANTs images (these are much smaller now)
        logger.info("Converting to ANTs images...")
        fixed_ants = ants.from_numpy(fixed_downsampled.astype(np.float32))
        moving_ants = ants.from_numpy(moving_downsampled.astype(np.float32))

        # Adjust spacing to account for downsampling
        original_spacing = [1.0, 1.0, 1.0]  # Default spacing
        downsampled_spacing = [s * self.downsample_factor for s in original_spacing]
        fixed_ants.set_spacing(downsampled_spacing)
        moving_ants.set_spacing(downsampled_spacing)

        # Perform linear registration
        logger.info(f"Performing linear registration ({linear_type})...")
        moving_linear, linear_transform = linear_registration(
            moving_image=moving_ants,
            fixed_image=fixed_ants,
            registration_type=linear_type,
            accuracy=linear_accuracy,
            verbose=True,
        )

        # Save linear transform
        linear_path = self.transforms_dir / "global_linear_transform.mat"
        ants.write_transform(linear_transform, str(linear_path))
        self.global_linear_transform_path = linear_path
        logger.info(f"Linear transform saved: {linear_path}")

        # Perform nonlinear registration
        logger.info(f"Performing nonlinear registration ({nonlinear_type})...")
        moving_nonlinear, nonlinear_transform_path = nonlinear_registration(
            moving_image=moving_linear,
            fixed_image=fixed_ants,
            registration_type=nonlinear_type,
            accuracy=nonlinear_accuracy,
            verbose=True,
        )

        # Copy nonlinear transform to our transforms directory
        nonlinear_path = self.transforms_dir / "global_nonlinear_warp.nii.gz"
        import shutil

        shutil.copy2(nonlinear_transform_path, nonlinear_path)
        self.global_nonlinear_transform_path = nonlinear_path
        logger.info(f"Nonlinear transform saved: {nonlinear_path}")

        logger.info("Coarse registration complete!")
        return linear_path, nonlinear_path

    def _compute_chunk_grid(self, image_shape: Tuple[int, int, int]) -> List[ChunkInfo]:
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
        logger.info("=" * 60)
        logger.info("STEP 2: Computing Chunk Grid")
        logger.info("=" * 60)

        z_max, y_max, x_max = image_shape
        chunk_z, chunk_y, chunk_x = self.chunk_size

        # Calculate overlap in pixels
        overlap_z = int(chunk_z * self.overlap_fraction)
        overlap_y = int(chunk_y * self.overlap_fraction)
        overlap_x = int(chunk_x * self.overlap_fraction)

        logger.info(f"Image shape: {image_shape}")
        logger.info(f"Chunk size: {self.chunk_size}")
        logger.info(f"Overlap: ({overlap_z}, {overlap_y}, {overlap_x})")

        chunks = []
        chunk_id = 0

        # Calculate step size (chunk size minus overlap)
        step_z = chunk_z - overlap_z
        step_y = chunk_y - overlap_y
        step_x = chunk_x - overlap_x

        # Iterate through the volume
        z_positions = list(range(0, z_max, step_z))
        y_positions = list(range(0, y_max, step_y))
        x_positions = list(range(0, x_max, step_x))

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

        logger.info(f"Created {len(chunks)} chunks")
        logger.info(
            f"Grid dimensions: {len(z_positions)} x {len(y_positions)} x {len(x_positions)}"
        )

        self.chunk_info_list = chunks
        return chunks

    def register_chunk(
        self,
        chunk: ChunkInfo,
        fixed_image: Union[np.ndarray, da.Array],
        moving_image: Union[np.ndarray, da.Array],
        linear_type: str = "TRSAA",
        linear_accuracy: str = "medium",
        nonlinear_type: str = "SyNOnly",
        nonlinear_accuracy: str = "high",
        perform_local_linear: bool = True,
    ) -> ChunkInfo:
        """
        Register a single chunk with fine registration.

        Parameters
        ----------
        chunk : ChunkInfo
            Information about the chunk to register.
        fixed_image : np.ndarray or da.Array
            The full fixed image.
        moving_image : np.ndarray or da.Array
            The full moving image.
        linear_type : str, optional
            Type of local linear registration (default: "TRSAA").
        linear_accuracy : str, optional
            Accuracy level for local linear registration (default: "medium").
        nonlinear_type : str, optional
            Type of nonlinear registration (default: "SyNOnly").
        nonlinear_accuracy : str, optional
            Accuracy level for nonlinear registration (default: "high").
        perform_local_linear : bool, optional
            Whether to perform local linear registration on the chunk before
            nonlinear registration (default: True).

        Returns
        -------
        ChunkInfo
            Updated chunk information with transform paths.
        """
        logger.info(f"Processing chunk {chunk.chunk_id}...")

        # Extract chunk from images (using extended boundaries)
        fixed_chunk = fixed_image[
            chunk.z_start_ext : chunk.z_end_ext,
            chunk.y_start_ext : chunk.y_end_ext,
            chunk.x_start_ext : chunk.x_end_ext,
        ]
        moving_chunk = moving_image[
            chunk.z_start_ext : chunk.z_end_ext,
            chunk.y_start_ext : chunk.y_end_ext,
            chunk.x_start_ext : chunk.x_end_ext,
        ]

        # Compute if Dask arrays
        if isinstance(fixed_chunk, da.Array):
            fixed_chunk = fixed_chunk.compute()
        if isinstance(moving_chunk, da.Array):
            moving_chunk = moving_chunk.compute()

        logger.info(f"  Chunk shape: {fixed_chunk.shape}")

        # Convert to ANTs images
        fixed_chunk_ants = ants.from_numpy(fixed_chunk.astype(np.float32))
        moving_chunk_ants = ants.from_numpy(moving_chunk.astype(np.float32))

        # Apply global transforms to moving chunk first
        if self.global_linear_transform_path and self.global_nonlinear_transform_path:
            logger.info("  Applying global transforms...")
            moving_chunk_ants = ants.apply_transforms(
                fixed=fixed_chunk_ants,
                moving=moving_chunk_ants,
                transformlist=[
                    str(self.global_linear_transform_path),
                    str(self.global_nonlinear_transform_path),
                ],
                whichtoinvert=[False, False],
                interpolator="linear",
            )

        # Perform local linear registration on chunk if requested
        if perform_local_linear:
            logger.info(f"  Performing local linear registration ({linear_type})...")
            moving_chunk_ants, local_linear_transform = linear_registration(
                moving_image=moving_chunk_ants,
                fixed_image=fixed_chunk_ants,
                registration_type=linear_type,
                accuracy=linear_accuracy,
                verbose=False,
            )

            # Save local linear transform
            local_linear_path = (
                self.chunks_dir / f"chunk_{chunk.chunk_id:04d}_linear.mat"
            )
            ants.write_transform(local_linear_transform, str(local_linear_path))
            chunk.linear_transform_path = local_linear_path
            logger.info(f"  Local linear transform saved: {local_linear_path}")

        # Perform fine nonlinear registration on this chunk
        logger.info("  Performing fine nonlinear registration...")
        _, local_transform_path = nonlinear_registration(
            moving_image=moving_chunk_ants,
            fixed_image=fixed_chunk_ants,
            registration_type=nonlinear_type,
            accuracy=nonlinear_accuracy,
            verbose=False,
        )

        # Save the local transform
        local_warp_path = self.chunks_dir / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz"
        import shutil

        shutil.copy2(local_transform_path, local_warp_path)

        chunk.nonlinear_transform_path = local_warp_path
        logger.info(f"  Local transform saved: {local_warp_path}")

        return chunk

    def register_all_chunks(
        self,
        linear_type: str = "TRSAA",
        linear_accuracy: str = "medium",
        nonlinear_type: str = "SyNOnly",
        nonlinear_accuracy: str = "high",
        perform_local_linear: bool = True,
        parallel: bool = False,
    ) -> None:
        """
        Register all chunks with fine registration.

        Parameters
        ----------
        linear_type : str, optional
            Type of local linear registration for each chunk (default: "TRSAA").
        linear_accuracy : str, optional
            Accuracy level for local linear registration (default: "medium").
        nonlinear_type : str, optional
            Type of nonlinear registration (default: "SyNOnly").
        nonlinear_accuracy : str, optional
            Accuracy level for nonlinear registration (default: "high").
        perform_local_linear : bool, optional
            Whether to perform local linear registration on each chunk before
            nonlinear registration (default: True).
        parallel : bool, optional
            Whether to process chunks in parallel using Dask (default: False).
            Note: ANTs may not be thread-safe, so use with caution.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Fine Registration on Chunks")
        logger.info("=" * 60)

        # Load full images
        logger.info("Loading full images...")
        fixed_image, fixed_info = self.image_opener.open(
            self.fixed_image_path, prefer_dask=self.prefer_dask
        )
        moving_image, moving_info = self.image_opener.open(
            self.moving_image_path, prefer_dask=self.prefer_dask
        )

        # Compute chunk grid if not already done
        if not self.chunk_info_list:
            self._compute_chunk_grid(fixed_info.shape)

        # Process each chunk
        for i, chunk in enumerate(self.chunk_info_list):
            logger.info(f"Chunk {i+1}/{len(self.chunk_info_list)}")
            updated_chunk = self.register_chunk(
                chunk=chunk,
                fixed_image=fixed_image,
                moving_image=moving_image,
                linear_type=linear_type,
                linear_accuracy=linear_accuracy,
                nonlinear_type=nonlinear_type,
                nonlinear_accuracy=nonlinear_accuracy,
                perform_local_linear=perform_local_linear,
            )
            self.chunk_info_list[i] = updated_chunk

        logger.info("All chunks registered!")

    def apply_transforms_to_chunk(
        self,
        chunk: ChunkInfo,
        fixed_image: Union[np.ndarray, da.Array],
        moving_image: Union[np.ndarray, da.Array],
        interpolator: str = "linear",
    ) -> np.ndarray:
        """
        Apply all transforms to a chunk and return the warped result.

        Parameters
        ----------
        chunk : ChunkInfo
            Information about the chunk.
        fixed_image : np.ndarray or da.Array
            The full fixed image.
        moving_image : np.ndarray or da.Array
            The full moving image.
        interpolator : str, optional
            Interpolation method (default: "linear").

        Returns
        -------
        np.ndarray
            The transformed chunk (core region only, without overlap).
        """
        # Extract extended chunk
        fixed_chunk = fixed_image[
            chunk.z_start_ext : chunk.z_end_ext,
            chunk.y_start_ext : chunk.y_end_ext,
            chunk.x_start_ext : chunk.x_end_ext,
        ]
        moving_chunk = moving_image[
            chunk.z_start_ext : chunk.z_end_ext,
            chunk.y_start_ext : chunk.y_end_ext,
            chunk.x_start_ext : chunk.x_end_ext,
        ]

        # Compute if Dask
        if isinstance(fixed_chunk, da.Array):
            fixed_chunk = fixed_chunk.compute()
        if isinstance(moving_chunk, da.Array):
            moving_chunk = moving_chunk.compute()

        # Convert to ANTs
        fixed_chunk_ants = ants.from_numpy(fixed_chunk.astype(np.float32))
        moving_chunk_ants = ants.from_numpy(moving_chunk.astype(np.float32))

        # Build transform list: global transforms + local transforms
        transform_list = []
        if self.global_linear_transform_path:
            transform_list.append(str(self.global_linear_transform_path))
        if self.global_nonlinear_transform_path:
            transform_list.append(str(self.global_nonlinear_transform_path))
        if chunk.linear_transform_path:
            transform_list.append(str(chunk.linear_transform_path))
        if chunk.nonlinear_transform_path:
            transform_list.append(str(chunk.nonlinear_transform_path))

        # Apply composed transforms
        transformed_chunk = ants.apply_transforms(
            fixed=fixed_chunk_ants,
            moving=moving_chunk_ants,
            transformlist=transform_list,
            whichtoinvert=[False] * len(transform_list),
            interpolator=interpolator,
        )

        # Extract core region (removing overlap)
        # Calculate offsets within extended chunk
        z_offset = chunk.z_start - chunk.z_start_ext
        y_offset = chunk.y_start - chunk.y_start_ext
        x_offset = chunk.x_start - chunk.x_start_ext

        core_size_z = chunk.z_end - chunk.z_start
        core_size_y = chunk.y_end - chunk.y_start
        core_size_x = chunk.x_end - chunk.x_start

        core_chunk = transformed_chunk.numpy()[
            z_offset : z_offset + core_size_z,
            y_offset : y_offset + core_size_y,
            x_offset : x_offset + core_size_x,
        ]

        return core_chunk

    def apply_transforms_and_blend(
        self,
        output_path: Union[str, Path],
        interpolator: str = "linear",
        blend_sigma: float = 10.0,
    ) -> None:
        """
        Apply transforms to all chunks and blend them into final output.

        Parameters
        ----------
        output_path : str or Path
            Path where the final transformed image will be saved.
        interpolator : str, optional
            Interpolation method (default: "linear").
        blend_sigma : float, optional
            Sigma for Gaussian blending in overlap regions (default: 10.0).
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Applying Transforms and Blending")
        logger.info("=" * 60)

        # Load images
        logger.info("Loading images...")
        fixed_image, fixed_info = self.image_opener.open(
            self.fixed_image_path, prefer_dask=self.prefer_dask
        )
        moving_image, moving_info = self.image_opener.open(
            self.moving_image_path, prefer_dask=self.prefer_dask
        )

        # Create output array
        output_shape = fixed_info.shape
        output_array = np.zeros(output_shape, dtype=np.float32)
        weight_array = np.zeros(output_shape, dtype=np.float32)

        logger.info(f"Output shape: {output_shape}")

        # Process each chunk
        for i, chunk in enumerate(self.chunk_info_list):
            logger.info(f"Transforming chunk {i+1}/{len(self.chunk_info_list)}")

            # Apply transforms to chunk
            transformed_core = self.apply_transforms_to_chunk(
                chunk=chunk,
                fixed_image=fixed_image,
                moving_image=moving_image,
                interpolator=interpolator,
            )

            # Create weight map for blending (higher weight in center, lower at edges)
            core_shape = transformed_core.shape
            weight_map = np.ones(core_shape, dtype=np.float32)

            # Apply Gaussian taper at boundaries if this is not an edge chunk
            if blend_sigma > 0:
                for axis in range(3):
                    size = core_shape[axis]
                    taper = np.ones(size)

                    # Taper at start if not at volume boundary
                    if (
                        (axis == 0 and chunk.z_start > 0)
                        or (axis == 1 and chunk.y_start > 0)
                        or (axis == 2 and chunk.x_start > 0)
                    ):
                        blend_size = int(size * self.overlap_fraction)
                        taper[:blend_size] = np.linspace(0, 1, blend_size)

                    # Taper at end if not at volume boundary
                    max_idx = output_shape[axis]
                    if (
                        (axis == 0 and chunk.z_end < max_idx)
                        or (axis == 1 and chunk.y_end < max_idx)
                        or (axis == 2 and chunk.x_end < max_idx)
                    ):
                        blend_size = int(size * self.overlap_fraction)
                        taper[-blend_size:] = np.linspace(1, 0, blend_size)

                    # Reshape taper for broadcasting
                    shape = [1, 1, 1]
                    shape[axis] = size
                    taper = taper.reshape(shape)
                    weight_map = weight_map * taper

            # Add to output with weighting
            output_array[
                chunk.z_start : chunk.z_end,
                chunk.y_start : chunk.y_end,
                chunk.x_start : chunk.x_end,
            ] += (
                transformed_core * weight_map
            )

            weight_array[
                chunk.z_start : chunk.z_end,
                chunk.y_start : chunk.y_end,
                chunk.x_start : chunk.x_end,
            ] += weight_map

        # Normalize by weights
        logger.info("Normalizing blended output...")
        output_array = np.divide(
            output_array,
            weight_array,
            out=np.zeros_like(output_array),
            where=weight_array > 0,
        )

        # Save output
        output_path = Path(output_path)
        logger.info(f"Saving output to {output_path}...")

        if output_path.suffix in [".zarr", ".n5"]:
            # Save as Zarr
            zarr.save(str(output_path), output_array)
        else:
            # Save as TIFF
            import tifffile

            tifffile.imwrite(str(output_path), output_array)

        logger.info("Transform application and blending complete!")

    def run_full_pipeline(
        self,
        output_path: Union[str, Path],
        linear_type: str = "Affine",
        nonlinear_type_coarse: str = "SyNOnly",
        nonlinear_type_fine: str = "SyNOnly",
        linear_accuracy: str = "low",
        coarse_accuracy: str = "medium",
        fine_accuracy: str = "high",
        interpolator: str = "linear",
        blend_sigma: float = 10.0,
    ) -> None:
        """
        Run the complete chunked registration pipeline.

        This executes all steps:
        1. Coarse global registration on downsampled data
        2. Break into overlapping chunks
        3. Fine registration on each chunk
        4. Apply transforms and blend

        Parameters
        ----------
        output_path : str or Path
            Path where the final transformed image will be saved.
        linear_type : str, optional
            Type of linear registration (default: "Affine").
        nonlinear_type_coarse : str, optional
            Type of nonlinear registration for coarse step (default: "SyNOnly").
        nonlinear_type_fine : str, optional
            Type of nonlinear registration for fine step (default: "SyNOnly").
        linear_accuracy : str, optional
            Accuracy level for linear registration (default: "low").
        coarse_accuracy : str, optional
            Accuracy level for coarse registration (default: "medium").
        fine_accuracy : str, optional
            Accuracy level for fine registration (default: "high").
        interpolator : str, optional
            Interpolation method for final transform application (default: "linear").
        blend_sigma : float, optional
            Sigma for Gaussian blending in overlap regions (default: 10.0).
        """
        logger.info("=" * 60)
        logger.info("CHUNKED REGISTRATION PIPELINE")
        logger.info("=" * 60)

        # Step 1: Coarse registration
        self.perform_coarse_registration(
            linear_type=linear_type,
            nonlinear_type=nonlinear_type_coarse,
            linear_accuracy=linear_accuracy,
            nonlinear_accuracy=coarse_accuracy,
        )

        # Step 2 & 3: Chunk and register
        self.register_all_chunks(
            nonlinear_type=nonlinear_type_fine,
            nonlinear_accuracy=fine_accuracy,
        )

        # Step 4: Apply and blend
        self.apply_transforms_and_blend(
            output_path=output_path,
            interpolator=interpolator,
            blend_sigma=blend_sigma,
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
