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

import numpy as np
import pytest
from clearex.preprocess.scale import resize_data


def test_resize_data_isotropic():
    data = np.random.rand(10, 20, 30)
    axial_pixel_size = 1.0
    lateral_pixel_size = 1.0

    resized_data = resize_data(data, axial_pixel_size, lateral_pixel_size)
    assert resized_data.shape == data.shape
    assert np.allclose(resized_data, data)


def test_resize_data_axial_larger():
    data = np.random.rand(10, 20, 30)
    axial_pixel_size = 2.0
    lateral_pixel_size = 1.0

    resized_data = resize_data(data, axial_pixel_size, lateral_pixel_size)
    expected_shape = (20, 20, 30)
    assert resized_data.shape == expected_shape


def test_resize_data_lateral_larger():
    data = np.random.rand(10, 20, 30)
    axial_pixel_size = 1.0
    lateral_pixel_size = 2.0

    resized_data = resize_data(data, axial_pixel_size, lateral_pixel_size)
    expected_shape = (10, 40, 60)
    assert resized_data.shape == expected_shape


def test_resize_data_invalid_dimension():
    data = np.random.rand(10, 20)
    axial_pixel_size = 1.0
    lateral_pixel_size = 1.0

    with pytest.raises(AssertionError):
        resize_data(data, axial_pixel_size, lateral_pixel_size)
