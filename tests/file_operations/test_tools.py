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

from clearex.file_operations.tools import (
    save_variable_to_disk,
    get_variable_size,
    load_variable_from_disk,
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
