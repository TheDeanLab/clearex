import numpy as np
import pytest
from clearex.segmentation.pointsource import (
    remove_close_blobs,
    sort_by_point_source_intensity,
    eliminate_insignificant_point_sources
)

def test_remove_close_blobs_isotropic():
    blobs = np.array([
        [10, 10, 10, 1],
        [11, 11, 11, 1],
        [20, 20, 20, 1]
    ])
    image = np.zeros((30, 30, 30))
    image[10, 10, 10] = 100
    image[11, 11, 11] = 90
    image[20, 20, 20] = 80

    min_dist = 2.0
    filtered_blobs = remove_close_blobs(blobs, image, min_dist)
    assert len(filtered_blobs) == 2
    assert np.array_equal(filtered_blobs, np.array([[10, 10, 10, 1], [20, 20, 20, 1]]))

def test_remove_close_blobs_anisotropic():
    blobs = np.array([
        [10, 10, 10, 1, 1, 1],
        [11, 11, 11, 1, 1, 1],
        [20, 20, 20, 1, 1, 1]
    ])
    image = np.zeros((30, 30, 30))
    image[10, 10, 10] = 100
    image[11, 11, 11] = 90
    image[20, 20, 20] = 80

    min_dist = 2.0
    filtered_blobs = remove_close_blobs(blobs, image, min_dist)
    assert len(filtered_blobs) == 2
    assert np.array_equal(filtered_blobs, np.array([[10, 10, 10, 1, 1, 1], [20, 20, 20, 1, 1, 1]]))

def test_remove_close_blobs_empty():
    blobs = np.array([])
    image = np.zeros((30, 30, 30))
    min_dist = 2.0
    filtered_blobs = remove_close_blobs(blobs, image, min_dist)
    assert len(filtered_blobs) == 0

def test_remove_close_blobs_invalid_format():
    blobs = np.array([
        [10, 10, 10],
        [11, 11, 11]
    ])
    image = np.zeros((30, 30, 30))
    min_dist = 2.0
    with pytest.raises(ValueError):
        remove_close_blobs(blobs, image, min_dist)

def test_sort_by_point_source_intensity_isotropic():
    blobs = np.array([
        [10, 10, 10, 1],
        [20, 20, 20, 1],
        [30, 30, 30, 1]
    ])
    image = np.zeros((40, 40, 40))
    image[10, 10, 10] = 80
    image[20, 20, 20] = 100
    image[30, 30, 30] = 90

    sorted_blobs = sort_by_point_source_intensity(blobs, image)
    expected_blobs = np.array([
        [20, 20, 20, 1],
        [30, 30, 30, 1],
        [10, 10, 10, 1]
    ])
    assert np.array_equal(sorted_blobs, expected_blobs)

def test_sort_by_point_source_intensity_anisotropic():
    blobs = np.array([
        [10, 10, 10, 1, 1, 1],
        [20, 20, 20, 1, 1, 1],
        [30, 30, 30, 1, 1, 1]
    ])
    image = np.zeros((40, 40, 40))
    image[10, 10, 10] = 80
    image[20, 20, 20] = 100
    image[30, 30, 30] = 90

    sorted_blobs = sort_by_point_source_intensity(blobs, image)
    expected_blobs = np.array([
        [20, 20, 20, 1, 1, 1],
        [30, 30, 30, 1, 1, 1],
        [10, 10, 10, 1, 1, 1]
    ])
    assert np.array_equal(sorted_blobs, expected_blobs)

def test_sort_by_point_source_intensity_empty():
    blobs = np.array([])
    image = np.zeros((40, 40, 40))

    sorted_blobs = sort_by_point_source_intensity(blobs, image)
    assert len(sorted_blobs) == 0

def test_eliminate_insignificant_point_sources_all_significant():
    chunk_iso = np.zeros((30, 30, 30))
    chunk_iso[10, 10, 10] = 150
    chunk_iso[20, 20, 20] = 160
    particle_location = np.array([
        [10, 10, 10, 1, 1, 1],
        [20, 20, 20, 1, 1, 1]
    ])

    significant_blobs = eliminate_insignificant_point_sources(chunk_iso, particle_location)
    assert len(significant_blobs) == 2
    assert np.array_equal(significant_blobs, particle_location)

def test_eliminate_insignificant_point_sources_some_insignificant():
    mean = 100
    sigma = 15
    offset = 50

    # Create a 3D volume of shape (30, 30, 30)
    chunk_iso = offset + np.random.normal(loc=mean, scale=sigma, size=(30, 30, 30))

    chunk_iso[10, 10, 10] = mean + offset
    chunk_iso[20, 20, 20] = mean + 5*sigma + offset
    particle_location = np.array([
        [10, 10, 10, 1, 1, 1],
        [20, 20, 20, 1, 1, 1]
    ])

    significant_blobs = eliminate_insignificant_point_sources(chunk_iso, particle_location)
    assert len(significant_blobs) == 1
    assert np.array_equal(significant_blobs, np.array([[20, 20, 20, 1, 1, 1]]))

def test_eliminate_insignificant_point_sources_empty():
    chunk_iso = np.zeros((30, 30, 30))
    particle_location = np.array([])

    significant_blobs = eliminate_insignificant_point_sources(chunk_iso, particle_location)
    assert len(significant_blobs) == 0