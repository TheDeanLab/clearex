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

# Third-Party Imports
from skimage.transform import resize
import numpy as np

# Local Imports


def resize_data(
    data: np.array, axial_pixel_size: float, lateral_pixel_size: float
) -> np.array:
    """Resize the data to isotropic resolution with linear interpolation.

    Parameters
    ----------
    data : np.array
        The 3D image data to resize.
    axial_pixel_size : float
        The axial pixel size of the data.
    lateral_pixel_size : float
        The lateral pixel size of the data.

    Returns
    -------
    np.array
        The resized 3D image data.
    """

    # Assert that the data is 3D
    assert data.ndim == 3, "Data must be 3D"

    original_dims = data.shape
    if axial_pixel_size == lateral_pixel_size:
        return data
    elif axial_pixel_size > lateral_pixel_size:
        # Make axial match lateral
        new_dims = (
            int(original_dims[0] * (axial_pixel_size / lateral_pixel_size)),
            original_dims[1],
            original_dims[2],
        )
    else:
        # Make lateral match axial
        new_dims = (
            original_dims[0],
            int(original_dims[1] * (lateral_pixel_size / axial_pixel_size)),
            original_dims[2] * (lateral_pixel_size / axial_pixel_size),
        )

    # linear interpolation specified by order. Should anti_aliasing=False?
    return resize(data, new_dims, preserve_range=True, order=1)
