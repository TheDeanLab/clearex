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

# Standard imports
from typing import Any

# Third party imports
import numpy as np
from scipy.optimize import curve_fit

# Local imports


def gaussian_fit(
    x: np.ndarray, amplitude: float, x_offset: float, sigma: float, y_offset: float
) -> np.ndarray:
    """Fit a Gaussian to a line profile.

    Parameters
    ----------
    x : np.ndarray
        The x-axis values.
    amplitude : float
        The amplitude of the Gaussian.
    x_offset : float
        The offset of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.
    y_offset : float
        The y offset of the Gaussian.

    Returns
    -------
    np.ndarray
        The Gaussian fit to the line profile.
    """
    return amplitude * np.exp(-((x - x_offset) ** 2) / (2 * sigma**2)) + y_offset


def fit_line_profile(
    x_axis: np.ndarray, line_profile: np.ndarray, lateral_pixel_size: float
) -> (float | Any, Any, Any, Any, Any):
    """Fit a Gaussian to a line profile and calculate the FWHM

    Parameters
    ----------
    x_axis : np.ndarray
        The x-axis values.
    line_profile : np.ndarray
        The line profile to fit.
    lateral_pixel_size : float
        The lateral pixel size of the data.

    Returns
    -------
    fit_results : tuple(float | Any, Any, Any, Any)
        The full width half maximum of the Gaussian fit.
    """

    # Initial guesses
    amplitude = np.max(line_profile) - np.min(line_profile)
    x_offset = np.argmax(line_profile)
    sigma = 3
    y_offset = np.min(line_profile)

    # Fit the Gaussian
    params, cov = curve_fit(
        gaussian_fit, x_axis, line_profile, p0=[amplitude, x_offset, sigma, y_offset]
    )

    # Unpack the parameters
    amplitude, x_offset, sigma, y_offset = params
    full_width_half_maximum = 2 * np.sqrt(2 * np.log(2)) * sigma * lateral_pixel_size
    return full_width_half_maximum, amplitude, x_offset, sigma, y_offset
