# Standard imports
from typing import Any

# Third party imports
import numpy as np
from scipy.optimize import curve_fit

# Local imports

def gaussian_fit(x: np.ndarray,
                 amplitude: float,
                 x_offset: float,
                 sigma: float,
                 y_offset: float) -> np.ndarray:
    """ Fit a Gaussian to a line profile.

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


def fit_line_profile(x_axis: np.ndarray,
                     line_profile: np.ndarray,
                     lateral_pixel_size: float) -> (float | Any, Any, Any, Any, Any):
    """ Fit a Gaussian to a line profile and calculate the FWHM

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
    params, cov = curve_fit(gaussian_fit, x_axis, line_profile,
                            p0=[amplitude, x_offset, sigma, y_offset]
    )

    # Unpack the parameters
    amplitude, x_offset, sigma, y_offset = params
    full_width_half_maximum = 2 * np.sqrt(2 * np.log(2)) * sigma * lateral_pixel_size
    return full_width_half_maximum, amplitude, x_offset, sigma, y_offset