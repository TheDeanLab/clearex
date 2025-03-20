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
