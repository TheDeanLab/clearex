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


# Third Party Imports
import numpy as np
from skimage.feature import blob_log
from scipy.ndimage import gaussian_filter

from clearex.detect.particles import remove_close_blobs, \
    eliminate_insignificant_point_sources
# Local Imports
from clearex.plot.images import mips
from clearex.preprocess.scale import resize_data


def detect_point_sources(
    input_chunk: np.ndarray,
    axial_pixel_size: float,
    lateral_pixel_size: float,
    distance: float = 10.0,
    plot_data: bool = False,
) -> np.ndarray:
    """Detect point sources in a 3D image.

    Parameters
    ----------
    input_chunk : np.ndarray
        The 3D image.
    axial_pixel_size : float
        The axial pixel size.
    lateral_pixel_size : float
        The lateral pixel size.
    distance : float
        The minimum distance between point sources.
    plot_data : bool
        Whether to plot the data.

    Returns
    -------
    masked_data : np.ndarray
        A boolean mask indicating the location of point sources.
    coordinates : np.ndarray
        Nx3 array of [z, y, x] coordinates for each point source.

    Notes
    -----
    For particles with FWHM ~6 pixels (XY) and ~10 pixels (Z):
    sigma ≈ FWHM / 2.355
    After isotropic resizing, all dimensions should have similar sigma
    XY: 6 / 2.355 ≈ 2.5, Z: 10 / 2.355 ≈ 4.2 (but scaled to ~2.5 after resize)
    """

    # Resize to isotropic
    print("Resizing data to isotropic voxel size...")
    chunk_iso: np.ndarray = resize_data(
        data=input_chunk,
        axial_pixel_size=axial_pixel_size,
        lateral_pixel_size=lateral_pixel_size,
    )

    # Detect blobs using Laplacian of Gaussian
    print("Detecting point sources using Laplacian of Gaussian...")
    particle_location = blob_log(
        image=chunk_iso,
        min_sigma=1.5,
        max_sigma=20.0,
        threshold=0.005,
        num_sigma=10,
        overlap=0.5,
    )
    print(f"Initial blob detection: {len(particle_location)} particles found")

    masked_data: np.ndarray = np.zeros_like(input_chunk, dtype=bool)
    if len(particle_location) == 0:
        print("No blobs detected after initial detection.")
        return masked_data, np.array([]).reshape(0, 3)

    # Locally evaluate whether the blob is statistically significant
    # relative to the local background.
    print("Filtering insignificant point sources...")
    significant_blobs = eliminate_insignificant_point_sources(
        chunk_iso, particle_location
    )
    print(f"After significance filtering: {len(significant_blobs)} particles remaining")
    if len(significant_blobs) == 0:
        print("No significant blobs remaining after significance filtering.")
        return masked_data, np.array([]).reshape(0, 3)

    # Iteratively remove blobs that are too close to each other
    print("Removing close point sources...")
    delta_blobs = True
    while delta_blobs:
        number_blobs: int = len(significant_blobs)
        significant_blobs = remove_close_blobs(
            blobs=significant_blobs, image=chunk_iso, min_dist=distance
        )
        delta_blobs = number_blobs > len(significant_blobs)
    print(f"After proximity filtering: {len(significant_blobs)} particles remaining")

    # Scale the z coordinate of the blobs
    particle_location = np.array(significant_blobs)
    particle_location[:, 0] = particle_location[:, 0] * (
        lateral_pixel_size / axial_pixel_size
    )

    # Convert blobs to type int
    particle_location = particle_location.astype(int)

    # Extract Z, Y, X coordinates
    coordinates = particle_location[:, :3]  # Shape: (N, 3) with columns [z, y, x]

    if plot_data:
        print("Plotting detected point sources...")
        mips(
            input_chunk,
            points=[coordinates],
            lut="nipy_spectral",
            scale_intensity=0.5,
        )

    # Create mask
    masked_data[
        particle_location[:, 0], particle_location[:, 1], particle_location[:, 2]
    ] = True
    return masked_data, coordinates


def background_correction(image: np.ndarray, sigma: float = 20) -> np.ndarray:
    """Compute background using Gaussian blur while excluding zero-valued voxels.

    Parameters
    ----------
    image : np.ndarray
        The 3D image.
    sigma : float
        The sigma for the Gaussian filter.

    Returns
    -------
    np.ndarray
        The background-corrected image.
    """
    # Create mask for non-zero voxels
    valid_mask = image > 0

    # Replace zeros with NaN
    image_masked = image.astype(float)
    image_masked[~valid_mask] = np.nan

    # Blur the valid data and the mask separately
    blurred_image = gaussian_filter(np.nan_to_num(image_masked), sigma=sigma)
    blurred_mask = gaussian_filter(valid_mask.astype(float), sigma=sigma)

    # Normalize by the blurred mask to get proper weighted average
    # This prevents zero regions from pulling down the background estimate
    background = np.divide(
        blurred_image,
        blurred_mask,
        out=np.zeros_like(blurred_image),
        where=blurred_mask > 0.01,  # Avoid division by very small numbers
    )

    background_corrected = image - background
    background_corrected[image == 0] = 0

    return background_corrected
