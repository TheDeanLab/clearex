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

import unittest
import os
import pickle
import numpy as np

from clearex.file_operations.tools import (
    save_variable_to_disk,
    get_variable_size,
    load_variable_from_disk,
    delete_filetype,
    get_roi_indices,
)


class TestGetVariableSize(unittest.TestCase):
    def test_get_variable_size_int(self):
        variable = 12345
        size = get_variable_size(variable)
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)

    def test_get_variable_size_list(self):
        variable = [1, 2, 3, 4, 5]
        size = get_variable_size(variable)
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)

    def test_get_variable_size_dict(self):
        variable = {"a": 1, "b": 2, "c": 3}
        size = get_variable_size(variable)
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)

    def test_get_variable_size_string(self):
        variable = "Hello, World!"
        size = get_variable_size(variable)
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)


class TestSaveVariableToDisk(unittest.TestCase):
    def setUp(self):
        self.test_file_path = "test_variable.pkl"

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_save_variable_to_disk_int(self):
        variable = 12345
        save_variable_to_disk(variable, self.test_file_path)
        self.assertTrue(os.path.exists(self.test_file_path))
        with open(self.test_file_path, "rb") as f:
            loaded_variable = pickle.load(f)
        self.assertEqual(variable, loaded_variable)

    def test_save_variable_to_disk_list(self):
        variable = [1, 2, 3, 4, 5]
        save_variable_to_disk(variable, self.test_file_path)
        self.assertTrue(os.path.exists(self.test_file_path))
        with open(self.test_file_path, "rb") as f:
            loaded_variable = pickle.load(f)
        self.assertEqual(variable, loaded_variable)

    def test_save_variable_to_disk_dict(self):
        variable = {"a": 1, "b": 2, "c": 3}
        save_variable_to_disk(variable, self.test_file_path)
        self.assertTrue(os.path.exists(self.test_file_path))
        with open(self.test_file_path, "rb") as f:
            loaded_variable = pickle.load(f)
        self.assertEqual(variable, loaded_variable)

    def test_save_variable_to_disk_string(self):
        variable = "Hello, World!"
        save_variable_to_disk(variable, self.test_file_path)
        self.assertTrue(os.path.exists(self.test_file_path))
        with open(self.test_file_path, "rb") as f:
            loaded_variable = pickle.load(f)
        self.assertEqual(variable, loaded_variable)


class TestLoadVariableFromDisk(unittest.TestCase):
    def setUp(self):
        self.test_file_path = "test_variable.pkl"

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_load_variable_from_disk_int(self):
        variable = 12345
        save_variable_to_disk(variable, self.test_file_path)
        loaded_variable = load_variable_from_disk(self.test_file_path)
        self.assertEqual(variable, loaded_variable)

    def test_load_variable_from_disk_list(self):
        variable = [1, 2, 3, 4, 5]
        save_variable_to_disk(variable, self.test_file_path)
        loaded_variable = load_variable_from_disk(self.test_file_path)
        self.assertEqual(variable, loaded_variable)

    def test_load_variable_from_disk_dict(self):
        variable = {"a": 1, "b": 2, "c": 3}
        save_variable_to_disk(variable, self.test_file_path)
        loaded_variable = load_variable_from_disk(self.test_file_path)
        self.assertEqual(variable, loaded_variable)

    def test_load_variable_from_disk_string(self):
        variable = "Hello, World!"
        save_variable_to_disk(variable, self.test_file_path)
        loaded_variable = load_variable_from_disk(self.test_file_path)
        self.assertEqual(variable, loaded_variable)


class TestDeleteFiletype(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_delete_filetype_dir"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_delete_filetype_with_dot(self):
        # Create test files
        test_files = ["file1.txt", "file2.txt", "file3.pdf"]
        for file in test_files:
            with open(os.path.join(self.test_dir, file), "w") as f:
                f.write("test content")

        delete_filetype(self.test_dir, ".txt")

        remaining_files = os.listdir(self.test_dir)
        self.assertEqual(len(remaining_files), 1)
        self.assertIn("file3.pdf", remaining_files)

    def test_delete_filetype_without_dot(self):
        # Create test files
        test_files = ["file1.pdf", "file2.pdf", "file3.txt"]
        for file in test_files:
            with open(os.path.join(self.test_dir, file), "w") as f:
                f.write("test content")

        delete_filetype(self.test_dir, "pdf")

        remaining_files = os.listdir(self.test_dir)
        self.assertEqual(len(remaining_files), 1)
        self.assertIn("file3.txt", remaining_files)

    def test_delete_filetype_no_matching_files(self):
        # Create test files
        test_files = ["file1.txt", "file2.txt"]
        for file in test_files:
            with open(os.path.join(self.test_dir, file), "w") as f:
                f.write("test content")

        delete_filetype(self.test_dir, ".pdf")

        remaining_files = os.listdir(self.test_dir)
        self.assertEqual(len(remaining_files), 2)

    def test_delete_filetype_empty_directory(self):
        delete_filetype(self.test_dir, ".txt")
        remaining_files = os.listdir(self.test_dir)
        self.assertEqual(len(remaining_files), 0)

    def test_delete_filetype_multiple_extensions(self):
        # Create test files with various extensions
        test_files = ["file1.txt", "file2.pdf", "file3.txt", "file4.jpg", "file5.txt"]
        for file in test_files:
            with open(os.path.join(self.test_dir, file), "w") as f:
                f.write("test content")

        delete_filetype(self.test_dir, "txt")

        remaining_files = os.listdir(self.test_dir)
        self.assertEqual(len(remaining_files), 2)
        self.assertIn("file2.pdf", remaining_files)
        self.assertIn("file4.jpg", remaining_files)


class TestGetROIIndices(unittest.TestCase):
    def test_get_roi_indices_default_size(self):
        # Create a test image
        image = np.zeros((512, 512, 512))
        z_start, z_end, y_start, y_end, x_start, x_end = get_roi_indices(image)

        # Check that ROI is centered
        self.assertEqual(z_start, 128)
        self.assertEqual(z_end, 384)
        self.assertEqual(y_start, 128)
        self.assertEqual(y_end, 384)
        self.assertEqual(x_start, 128)
        self.assertEqual(x_end, 384)

        # Check ROI size is correct
        self.assertEqual(z_end - z_start, 256)
        self.assertEqual(y_end - y_start, 256)
        self.assertEqual(x_end - x_start, 256)

    def test_get_roi_indices_custom_size(self):
        image = np.zeros((512, 512, 512))
        roi_size = 128
        z_start, z_end, y_start, y_end, x_start, x_end = get_roi_indices(
            image, roi_size
        )

        # Check ROI size is correct
        self.assertEqual(z_end - z_start, 128)
        self.assertEqual(y_end - y_start, 128)
        self.assertEqual(x_end - x_start, 128)

        # Check centering
        self.assertEqual(z_start, 192)
        self.assertEqual(z_end, 320)

    def test_get_roi_indices_non_cubic_image(self):
        # Test with non-cubic dimensions
        image = np.zeros((400, 600, 800))
        roi_size = 200
        z_start, z_end, y_start, y_end, x_start, x_end = get_roi_indices(
            image, roi_size
        )

        self.assertEqual(z_end - z_start, 200)
        self.assertEqual(y_end - y_start, 200)
        self.assertEqual(x_end - x_start, 200)

    def test_get_roi_indices_odd_dimensions(self):
        # Test with odd image dimensions
        image = np.zeros((511, 511, 511))
        roi_size = 256
        z_start, z_end, y_start, y_end, x_start, x_end = get_roi_indices(
            image, roi_size
        )

        # Verify indices are within bounds
        self.assertGreaterEqual(z_start, 0)
        self.assertLessEqual(z_end, 511)
        self.assertGreaterEqual(y_start, 0)
        self.assertLessEqual(y_end, 511)
        self.assertGreaterEqual(x_start, 0)
        self.assertLessEqual(x_end, 511)

    def test_get_roi_indices_raises_on_2d_image(self):
        # Should raise ValueError for 2D image
        image = np.zeros((512, 512))
        with self.assertRaises(ValueError) as context:
            get_roi_indices(image)
        self.assertIn("3D array", str(context.exception))

    def test_get_roi_indices_raises_on_negative_roi_size(self):
        image = np.zeros((512, 512, 512))
        with self.assertRaises(ValueError) as context:
            get_roi_indices(image, roi_size=-10)
        self.assertIn("positive integer", str(context.exception))

    def test_get_roi_indices_raises_on_zero_roi_size(self):
        image = np.zeros((512, 512, 512))
        with self.assertRaises(ValueError) as context:
            get_roi_indices(image, roi_size=0)
        self.assertIn("positive integer", str(context.exception))

    def test_get_roi_indices_raises_on_oversized_roi(self):
        # ROI size larger than image dimensions
        image = np.zeros((100, 100, 100))
        with self.assertRaises(ValueError) as context:
            get_roi_indices(image, roi_size=256)
        self.assertIn("smallest dimension", str(context.exception))

    def test_get_roi_indices_boundary_case_small_image(self):
        # Test with small image where ROI equals image size
        image = np.zeros((256, 256, 256))
        roi_size = 256
        z_start, z_end, y_start, y_end, x_start, x_end = get_roi_indices(
            image, roi_size
        )

        self.assertEqual(z_start, 0)
        self.assertEqual(z_end, 256)
        self.assertEqual(y_start, 0)
        self.assertEqual(y_end, 256)
        self.assertEqual(x_start, 0)
        self.assertEqual(x_end, 256)
