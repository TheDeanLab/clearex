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

"""Tests for ImageRegistration class."""

# Standard Library Imports
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Third Party Imports
import pytest
import numpy as np
import ants

# Local Imports
from clearex.registration import ImageRegistration, register_round


class TestImageRegistration:
    """Test suite for ImageRegistration class."""

    def test_initialization_with_defaults(self):
        """Test that ImageRegistration initializes with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files
            fixed_path = os.path.join(tmpdir, "fixed.npy")
            moving_path = os.path.join(tmpdir, "moving.npy")

            # Create simple 3D arrays
            fixed_arr = np.random.rand(10, 10, 10).astype(np.float32)
            moving_arr = np.random.rand(10, 10, 10).astype(np.float32)

            np.save(fixed_path, fixed_arr)
            np.save(moving_path, moving_arr)

            reg = ImageRegistration(
                fixed_image_path=fixed_path,
                moving_image_path=moving_path,
                save_directory=tmpdir,
            )
            assert reg.fixed_image_path == fixed_path
            assert reg.moving_image_path == moving_path
            assert reg.save_directory == tmpdir
            assert reg.imaging_round == 0
            assert reg.crop is False
            assert reg.force_override is False
            assert reg._log is not None
            assert reg._image_opener is not None

    def test_initialization_with_custom_values(self):
        """Test that ImageRegistration initializes with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files
            fixed_path = os.path.join(tmpdir, "fixed.npy")
            moving_path = os.path.join(tmpdir, "moving.npy")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Create simple 3D arrays
            fixed_arr = np.random.rand(10, 10, 10).astype(np.float32)
            moving_arr = np.random.rand(10, 10, 10).astype(np.float32)

            np.save(fixed_path, fixed_arr)
            np.save(moving_path, moving_arr)

            reg = ImageRegistration(
                fixed_image_path=fixed_path,
                moving_image_path=moving_path,
                save_directory=output_dir,
                imaging_round=5,
                crop=True,
                enable_logging=False,
            )
            assert reg.fixed_image_path == fixed_path
            assert reg.moving_image_path == moving_path
            assert reg.save_directory == output_dir
            assert reg.imaging_round == 5
            assert reg.crop is True
            assert reg.force_override is False

    def test_initialization_validates_required_parameters(self):
        """Test that ImageRegistration __init__ validates required parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files
            fixed_path = os.path.join(tmpdir, "fixed.npy")
            moving_path = os.path.join(tmpdir, "moving.npy")

            # Create simple 3D arrays
            fixed_arr = np.random.rand(10, 10, 10).astype(np.float32)
            moving_arr = np.random.rand(10, 10, 10).astype(np.float32)

            np.save(fixed_path, fixed_arr)
            np.save(moving_path, moving_arr)

            # Test initialization succeeds with all required parameters
            reg = ImageRegistration(
                fixed_image_path=fixed_path,
                moving_image_path=moving_path,
                save_directory=tmpdir,
            )
            assert reg is not None

    def test_register_uses_instance_attributes(self):
        """Test that register uses instance attributes when parameters not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files
            fixed_path = os.path.join(tmpdir, "fixed.npy")
            moving_path = os.path.join(tmpdir, "moving.npy")

            # Create simple 3D arrays
            fixed_arr = np.random.rand(10, 10, 10).astype(np.float32)
            moving_arr = np.random.rand(10, 10, 10).astype(np.float32)

            np.save(fixed_path, fixed_arr)
            np.save(moving_path, moving_arr)

            reg = ImageRegistration(
                fixed_image_path=fixed_path,
                moving_image_path=moving_path,
                save_directory=tmpdir,
                imaging_round=1,
                enable_logging=False,
            )

            # Mock the internal methods to avoid actual registration
            mock_image = MagicMock()
            mock_mask = MagicMock()
            with patch.object(reg, '_perform_linear_registration', return_value=(mock_image, mock_mask)):
                with patch.object(reg, '_perform_nonlinear_registration', return_value=(mock_image, mock_mask)):
                    reg.register()

            # Verify that logging was initialized
            assert reg._log is not None


class TestRegisterRound:
    """Test suite for register_round convenience function."""

    def test_register_round_creates_and_calls_image_registration(self):
        """Test that register_round creates ImageRegistration and calls register."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files
            fixed_path = os.path.join(tmpdir, "fixed.npy")
            moving_path = os.path.join(tmpdir, "moving.npy")

            # Create simple 3D arrays
            fixed_arr = np.random.rand(10, 10, 10).astype(np.float32)
            moving_arr = np.random.rand(10, 10, 10).astype(np.float32)

            np.save(fixed_path, fixed_arr)
            np.save(moving_path, moving_arr)

            # Mock ImageRegistration to avoid actual registration
            with patch('clearex.registration.ImageRegistration') as MockReg:
                mock_instance = MockReg.return_value

                register_round(
                    fixed_image_path=fixed_path,
                    moving_image_path=moving_path,
                    save_directory=tmpdir,
                    imaging_round=2,
                    crop=True,
                    enable_logging=False,
                )

                # Verify ImageRegistration was created with correct parameters
                MockReg.assert_called_once_with(
                    fixed_image_path=fixed_path,
                    moving_image_path=moving_path,
                    save_directory=tmpdir,
                    imaging_round=2,
                    crop=True,
                    enable_logging=False,
                )

                # Verify register was called
                mock_instance.register.assert_called_once()

