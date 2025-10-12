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
from unittest.mock import patch, MagicMock
import io
from clearex.context.environment import get_installed_packages


class TestGetInstalledPackages(unittest.TestCase):
    @patch("pkgutil.iter_modules")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_get_installed_packages_returns_dict(self, mock_stdout, mock_iter_modules):
        # Create mock module objects
        mock_module1 = MagicMock()
        mock_module1.name = "numpy"
        mock_module1.module_finder.path = "/usr/local/lib/python3.9/site-packages"

        mock_module2 = MagicMock()
        mock_module2.name = "pandas"
        mock_module2.module_finder.path = "/usr/local/lib/python3.9/site-packages"

        mock_iter_modules.return_value = [mock_module1, mock_module2]

        result = get_installed_packages()

        # Verify return type and content
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn("numpy", result)
        self.assertIn("pandas", result)
        self.assertEqual(result["numpy"], "/usr/local/lib/python3.9/site-packages")
        self.assertEqual(result["pandas"], "/usr/local/lib/python3.9/site-packages")

    @patch("pkgutil.iter_modules")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_get_installed_packages_prints_output(self, mock_stdout, mock_iter_modules):
        # Create mock module object
        mock_module = MagicMock()
        mock_module.name = "requests"
        mock_module.module_finder.path = "/usr/lib/python3.9/site-packages"

        mock_iter_modules.return_value = [mock_module]

        get_installed_packages()

        output = mock_stdout.getvalue()
        self.assertIn("requests: /usr/lib/python3.9/site-packages", output)

    @patch("pkgutil.iter_modules")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_get_installed_packages_empty_list(self, mock_stdout, mock_iter_modules):
        # Mock empty module list
        mock_iter_modules.return_value = []

        result = get_installed_packages()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)
        self.assertEqual(result, {})

    @patch("pkgutil.iter_modules")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_get_installed_packages_multiple_packages(
        self, mock_stdout, mock_iter_modules
    ):
        # Create multiple mock modules
        mock_modules = []
        expected_packages = {
            "scipy": "/path/to/scipy",
            "matplotlib": "/path/to/matplotlib",
            "pytest": "/path/to/pytest",
        }

        for name, path in expected_packages.items():
            mock_module = MagicMock()
            mock_module.name = name
            mock_module.module_finder.path = path
            mock_modules.append(mock_module)

        mock_iter_modules.return_value = mock_modules

        result = get_installed_packages()

        self.assertEqual(result, expected_packages)

        # Verify all packages were printed
        output = mock_stdout.getvalue()
        for name, path in expected_packages.items():
            self.assertIn(f"{name}: {path}", output)
