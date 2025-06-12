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
from clearex.segmentation.pointsource import (
    remove_close_blobs,
    sort_by_point_source_intensity,
    eliminate_insignificant_point_sources,
)


def test_remove_close_blobs_isotropic():
    blobs = np.array([[10, 10, 10, 1], [11, 11, 11, 1], [20, 20, 20, 1]])
    image = np.zeros((30, 30, 30))
    image[10, 10, 10] = 100
    image[11, 11, 11] = 90
    image[20, 20, 20] = 80

    min_dist = 2.0
    filtered_blobs = remove_close_blobs(blobs, image, min_dist)
    assert len(filtered_blobs) == 2
    assert np.array_equal(filtered_blobs, np.array([[10, 10, 10, 1], [20, 20, 20, 1]]))


def test_remove_close_blobs_anisotropic():
    blobs = np.array(
        [[10, 10, 10, 1, 1, 1], [11, 11, 11, 1, 1, 1], [20, 20, 20, 1, 1, 1]]
    )
    image = np.zeros((30, 30, 30))
    image[10, 10, 10] = 100
    image[11, 11, 11] = 90
    image[20, 20, 20] = 80

    min_dist = 2.0
    filtered_blobs = remove_close_blobs(blobs, image, min_dist)
    assert len(filtered_blobs) == 2
    assert np.array_equal(
        filtered_blobs, np.array([[10, 10, 10, 1, 1, 1], [20, 20, 20, 1, 1, 1]])
    )


def test_remove_close_blobs_empty():
    blobs = np.array([])
    image = np.zeros((30, 30, 30))
    min_dist = 2.0
    filtered_blobs = remove_close_blobs(blobs, image, min_dist)
    assert len(filtered_blobs) == 0


def test_remove_close_blobs_invalid_format():
    blobs = np.array([[10, 10, 10], [11, 11, 11]])
    image = np.zeros((30, 30, 30))
    min_dist = 2.0
    with pytest.raises(ValueError):
        remove_close_blobs(blobs, image, min_dist)


def test_sort_by_point_source_intensity_isotropic():
    blobs = np.array([[10, 10, 10, 1], [20, 20, 20, 1], [30, 30, 30, 1]])
    image = np.zeros((40, 40, 40))
    image[10, 10, 10] = 80
    image[20, 20, 20] = 100
    image[30, 30, 30] = 90

    sorted_blobs = sort_by_point_source_intensity(blobs, image)
    expected_blobs = np.array([[20, 20, 20, 1], [30, 30, 30, 1], [10, 10, 10, 1]])
    assert np.array_equal(sorted_blobs, expected_blobs)


def test_sort_by_point_source_intensity_anisotropic():
    blobs = np.array(
        [[10, 10, 10, 1, 1, 1], [20, 20, 20, 1, 1, 1], [30, 30, 30, 1, 1, 1]]
    )
    image = np.zeros((40, 40, 40))
    image[10, 10, 10] = 80
    image[20, 20, 20] = 100
    image[30, 30, 30] = 90

    sorted_blobs = sort_by_point_source_intensity(blobs, image)
    expected_blobs = np.array(
        [[20, 20, 20, 1, 1, 1], [30, 30, 30, 1, 1, 1], [10, 10, 10, 1, 1, 1]]
    )
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
    particle_location = np.array([[10, 10, 10, 1, 1, 1], [20, 20, 20, 1, 1, 1]])

    significant_blobs = eliminate_insignificant_point_sources(
        chunk_iso, particle_location
    )
    assert len(significant_blobs) == 2
    assert np.array_equal(significant_blobs, particle_location)


def test_eliminate_insignificant_point_sources_some_insignificant():
    mean = 100
    sigma = 15
    offset = 50

    # Create a 3D volume of shape (30, 30, 30)
    chunk_iso = offset + np.random.normal(loc=mean, scale=sigma, size=(30, 30, 30))

    chunk_iso[10, 10, 10] = mean + offset
    chunk_iso[20, 20, 20] = mean + 5 * sigma + offset
    particle_location = np.array([[10, 10, 10, 1, 1, 1], [20, 20, 20, 1, 1, 1]])

    significant_blobs = eliminate_insignificant_point_sources(
        chunk_iso, particle_location
    )
    assert len(significant_blobs) == 1
    assert np.array_equal(significant_blobs, np.array([[20, 20, 20, 1, 1, 1]]))


def test_eliminate_insignificant_point_sources_empty():
    chunk_iso = np.zeros((30, 30, 30))
    particle_location = np.array([])

    significant_blobs = eliminate_insignificant_point_sources(
        chunk_iso, particle_location
    )
    assert len(significant_blobs) == 0
