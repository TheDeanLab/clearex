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
import numpy as np

# Local Imports


def make_3d_structured_element(radius, shape="sphere"):
    """Create a 3D structuring element.

    Parameters
    ----------
    radius : int
        The radius of the structuring element.
    shape : str
        The shape of structuring element to create. Options are 'sphere'.

    Returns
    -------
    np.ndarray
        The 3D structuring element.
    """

    radius = int(radius)
    structured_element = np.zeros((radius, radius, radius))
    (z_len, y_len, x_len) = structured_element.shape

    if shape == "sphere":
        for i in range(int(z_len)):
            for j in range(int(y_len)):
                for k in range(int(x_len)):
                    if ((i**2 + j**2 + k**2) / radius**2) < 1:
                        structured_element[i, j, k] = 1
    else:
        raise ValueError("Invalid shape. Please choose 'sphere'.")
    return structured_element


def gaussian_kernel(sigma):
    """Create a 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        The 1D Gaussian kernel.
    """
    w = int(np.ceil(5 * sigma))
    x = np.arange(-w, w + 1, 1)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def second_derivative_gaussian_kernel(sigma):
    """Create a 1D second derivative Gaussian kernel.

    Parameters
    ----------
    sigma : float
        The standard deviation of the second derivative Gaussian kernel.

    Returns
    -------
    np.ndarray
        The 1D second derivative Gaussian kernel.
    """
    w = int(np.ceil(5 * sigma))
    x = np.arange(-w, w + 1, 1)
    d2g = -(x**2 / sigma**2 - 1) / sigma**2 * np.exp(-(x**2) / (2 * sigma**2))
    d2g /= d2g.sum()
    return d2g
