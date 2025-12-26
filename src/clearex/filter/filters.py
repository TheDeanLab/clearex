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

# Third Party Imports
from scipy import ndimage
import cv2
import numpy as np
from skimage import filters as skfilters


# Local Imports

def fwhm_to_sigma(fwhm_px: float) -> float:
    """ Convert from FWHM to sigma.

    FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.35482*sigma

    Parameters
    ----------
    fwhm_px : float
        The full width at half maximum in pixels.
    Returns
    -------
    float
        The standard deviation sigma.
    """
    return float(fwhm_px) / 2.354820045


def dog(sigma_high: float, sigma_low: float, vol: np.ndarray[...]) -> np.ndarray[...]:
    """Difference of Gaussian (DoG) filter.

    Parameters
    ----------
    sigma_high : float
        The standard deviation of the high-pass filter.
    sigma_low : float
        The standard deviation of the low-pass filter.
    vol : np.ndarray
        The volume to filter.

    Returns
    -------
    np.ndarray
        The DoG filtered volume.
    """
    high_pass_filtered = ndimage.gaussian_filter(vol, sigma=sigma_high, mode="nearest")
    low_pass_filtered = ndimage.gaussian_filter(vol, sigma=sigma_low, mode="nearest")
    dog_filtered = high_pass_filtered - low_pass_filtered
    return dog_filtered


def dog_cv2(sigma_high: float, sigma_low: float, vol: np.ndarray) -> np.ndarray:
    """Difference of Gaussian (DoG) filter using OpenCV.

    Parameters
    ----------
    sigma_high : float
        The standard deviation of the high-pass filter.
    sigma_low : float
        The standard deviation of the low-pass filter.
    vol : np.ndarray
        The volume to filter.

    Returns
    -------
    np.ndarray
        The DoG filtered volume.
    """

    # Determine kernel sizes.
    # OpenCV requires kernel sizes to be odd and positive integers.
    # A common rule of thumb is kernel_size = 6 * sigma + 1 to capture most of Gaussian.
    ksize_high = int(6 * sigma_high + 1) | 1  # ensure odd
    ksize_low = int(6 * sigma_low + 1) | 1  # ensure odd

    # Apply Gaussian blur with the high sigma value.
    high_blurred = cv2.GaussianBlur(
        vol, (ksize_high, ksize_high), sigmaX=sigma_high, borderType=cv2.BORDER_REFLECT
    )
    low_blurred = cv2.GaussianBlur(
        vol, (ksize_low, ksize_low), sigmaX=sigma_low, borderType=cv2.BORDER_REFLECT
    )
    dog_result = low_blurred - high_blurred
    print("min/max high:", np.min(high_blurred), np.max(high_blurred))
    print("min/max low:", np.min(low_blurred), np.max(low_blurred))
    print("min/max dog:", np.min(dog_result), np.max(dog_result))

    return dog_result


def meijering_filter(
    slice2d: np.ndarray, sigmas: list[float], black_ridges: bool
) -> np.ndarray:
    """Apply the Meijering filter to a 2D slice.

    Parameters
    ----------
    slice2d : np.ndarray
        A 2D slice of the data.
    sigmas : list of float
        Standard deviations for Gaussian smoothing.
    black_ridges : bool
        If True, return black ridges on a white background.

    Returns
    -------
    np.ndarray
        The filtered 2D slice.
    """
    return skfilters.meijering(image=slice2d, sigmas=sigmas, black_ridges=black_ridges)
