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