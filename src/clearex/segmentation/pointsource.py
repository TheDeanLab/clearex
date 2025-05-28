# Standard Library Imports
from typing import Optional

# Third Party Imports
import numpy as np
from skimage.feature import blob_log

# Local Imports
from clearex.plot.images import mips
from clearex.preprocess.scale import resize_data


def remove_close_blobs(
    blobs: np.ndarray, image: np.ndarray, min_dist: float
) -> np.ndarray:
    """Remove close particles.

    Remove blobs that are too close to each other using an ellipsoidal search volume
    defined by the blob's sigma values, discarding the blob with the lower absolute
    intensity at its centroid.

    Parameters
    ----------
    blobs : np.ndarray
        An Nx4 or Nx6 array of blobs. For isotropic blobs, each row is [z, y, x, sigma].
        For anisotropic blobs, each row is [z, y, x, z_sigma, y_sigma, x_sigma].
    image : np.ndarray
        The 3D image data from which intensities at blob centers will be extracted.
    min_dist : float
        A scaling factor for the sigma values to determine the ellipsoidal search
        volume.

    Returns
    -------
    np.ndarray
        The filtered array of blobs.
    """
    blobs = np.array(blobs)

    sorted_blobs = sort_by_point_source_intensity(blobs, image)

    final_blobs = []
    for blob in sorted_blobs:
        blob_center = blob[:3]

        too_close = False
        # Check against every already accepted blob
        for accepted_blob in final_blobs:
            accepted_center = accepted_blob[:3]

            if accepted_blob.size == 4:
                # Isotropic sigma
                accepted_radii = np.array([accepted_blob[3]] * 3) * min_dist
            elif accepted_blob.size == 6:
                # Anisotropic sigma
                accepted_radii = (
                    np.array([accepted_blob[3], accepted_blob[4], accepted_blob[5]])
                    * min_dist
                )
            else:
                raise ValueError(f"Unexpected blob format with length {len(blob)}")

            # Calculate the difference vector
            diff = blob_center - accepted_center

            # Normalized squared distance
            norm_sq = (
                (diff[0] / accepted_radii[0]) ** 2
                + (diff[1] / accepted_radii[1]) ** 2
                + (diff[2] / accepted_radii[2]) ** 2
            )

            if norm_sq < 1:
                too_close = True
                break

        if not too_close:
            final_blobs.append(blob)

    return np.array(final_blobs)


def sort_by_point_source_intensity(blobs: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Sort blobs by their intensity in the given image.

    Parameters
    ----------
    blobs : np.ndarray
        An Nx4 or Nx6 array of blobs. For isotropic blobs, each row is [z, y, x, sigma].
        For anisotropic blobs, each row is [z, y, x, z_sigma, y_sigma, x_sigma].
    image : np.ndarray
        The 3D image data from which intensities at blob centers will be extracted.

    Returns
    -------
    np.ndarray
        The sorted array of blobs.
    """
    # Compute intensity for each blob using the image data.
    intensities = []
    for blob in blobs:
        # Extract the center coordinates (assumes they lie within image bounds)
        z, y, x = blob[:3]
        intensity = image[int(round(z)), int(round(y)), int(round(x))]
        intensities.append(intensity)
    intensities = np.array(intensities)

    # Sort blobs in descending order of intensity (highest intensity first)
    order = np.argsort(intensities)[::-1]
    sorted_blobs = blobs[order]
    return sorted_blobs


def detect_point_sources(
    input_chunk: np.ndarray,
    axial_pixel_size: float,
    lateral_pixel_size: float,
    distance: Optional[float] = 10,
    plot_data: Optional[bool] = False,
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
    np.ndarray
        A boolean mask indicating the location of point sources.
    """

    # Resize to isotropic
    chunk_iso = resize_data(
        input_chunk,
        axial_pixel_size=axial_pixel_size,
        lateral_pixel_size=lateral_pixel_size,
    )

    particle_location = blob_log(
        chunk_iso,
        min_sigma=(10.0, 1.0, 1.0),
        max_sigma=(10.0, 1.0, 1.0),
        threshold=100,
        num_sigma=1,
        overlap=0.0,
    )

    masked_data = np.zeros_like(input_chunk, dtype=bool)
    if len(particle_location) == 0:
        return masked_data

    # Locally evaluate whether the blob is statistically significant
    # relative to the local background.
    significant_blobs = eliminate_insignificant_point_sources(
        chunk_iso, particle_location
    )
    if len(significant_blobs) == 0:
        return masked_data

    # Iteratively remove blobs that are too close to each other
    delta_blobs = True
    while delta_blobs:
        number_blobs = len(significant_blobs)
        significant_blobs = remove_close_blobs(
            blobs=significant_blobs, min_dist=distance, image=chunk_iso
        )
        delta_blobs = number_blobs > len(significant_blobs)

    # Scale the z coordinate of the blobs
    particle_location = np.array(significant_blobs)
    particle_location[:, 0] = particle_location[:, 0] * (
        lateral_pixel_size / axial_pixel_size
    )

    # Convert blobs to type int
    particle_location = particle_location.astype(int)

    if plot_data:
        mips(
            input_chunk,
            points=[particle_location[:, :3]],
            lut="nipy_spectral",
            scale_intensity=0.5,
        )

    # Create mask
    masked_data[
        particle_location[:, 0], particle_location[:, 1], particle_location[:, 2]
    ] = True
    return masked_data


def eliminate_insignificant_point_sources(
    chunk_iso: np.ndarray, particle_location: np.ndarray
) -> np.ndarray:
    """Eliminate insignificant point sources.

     Evaluate whether the point source is statistically significant relative to the
     local background. This is done by comparing the point sources's intensity to the
     mean and standard deviation of a local region. If the point sources's intensity is
     greater than the mean + 2 * std, it is considered significant.

    Parameters
    ----------
    chunk_iso : np.ndarray
        The 3D image.
    particle_location : np.ndarray
        The locations of the blobs.

    Returns
    -------
    significant_blobs : np.ndarray
        The significant blobs.
    """

    chunk_mean = np.mean(chunk_iso)
    chunk_std = np.std(chunk_iso)
    significant_blobs = []
    for i, blob in enumerate(particle_location):
        z, y, x, _, _, _ = blob
        z, y, x = int(z), int(y), int(x)
        if chunk_iso[z, y, x] < chunk_mean + 2 * chunk_std:
            continue
        significant_blobs.append(blob)
    return significant_blobs
