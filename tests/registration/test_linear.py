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

"""Tests for linear registration functions."""

# Standard Library Imports
from unittest.mock import Mock
import tempfile

# Third Party Imports
import pytest
import numpy as np
import ants
from scipy.spatial.transform import Rotation
from scipy.linalg import polar, rq

# Local Imports
from clearex.registration.linear import (
    inspect_affine_transform,
    _extract_translation,
    _extract_rotation,
    _extract_shear,
    _extract_scale,
)


class TestExtractTranslation:
    """Test suite for _extract_translation function."""

    def test_identity_transform_no_translation(self, capsys):
        """Test that identity transform has zero translation."""
        affine_matrix = np.eye(4)
        _extract_translation(affine_matrix)

        captured = capsys.readouterr()
        assert "Translation distance in Z: 0.00 voxels" in captured.out
        assert "Translation distance in Y: 0.00 voxels" in captured.out
        assert "Translation distance in X: 0.00 voxels" in captured.out

    def test_pure_translation_positive(self, capsys):
        """Test extraction of positive translation values."""
        affine_matrix = np.eye(4)
        affine_matrix[:3, 3] = [10.5, 20.3, 30.7]  # Z, Y, X translation

        _extract_translation(affine_matrix)

        captured = capsys.readouterr()
        assert "Translation distance in Z: 10.50 voxels" in captured.out
        assert "Translation distance in Y: 20.30 voxels" in captured.out
        assert "Translation distance in X: 30.70 voxels" in captured.out

    def test_pure_translation_negative(self, capsys):
        """Test extraction of negative translation values."""
        affine_matrix = np.eye(4)
        affine_matrix[:3, 3] = [-5.2, -15.8, -25.4]

        _extract_translation(affine_matrix)

        captured = capsys.readouterr()
        assert "Translation distance in Z: -5.20 voxels" in captured.out
        assert "Translation distance in Y: -15.80 voxels" in captured.out
        assert "Translation distance in X: -25.40 voxels" in captured.out

    def test_mixed_transform_with_translation(self, capsys):
        """Test translation extraction from a combined transform."""
        # Create a transform with rotation and translation
        affine_matrix = np.eye(4)
        # Add some rotation
        angle = np.radians(45)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        affine_matrix[:3, :3] = np.array(
            [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
        )
        # Add translation
        affine_matrix[:3, 3] = [12.0, 24.0, 36.0]

        _extract_translation(affine_matrix)

        captured = capsys.readouterr()
        assert "Translation distance in Z: 12.00 voxels" in captured.out
        assert "Translation distance in Y: 24.00 voxels" in captured.out
        assert "Translation distance in X: 36.00 voxels" in captured.out


class TestExtractRotation:
    """Test suite for _extract_rotation function."""

    def test_identity_no_rotation(self, capsys):
        """Test that identity transform has zero rotation."""
        affine_matrix = np.eye(4)
        _extract_rotation(affine_matrix)

        captured = capsys.readouterr()
        assert "Rotation around X (roll) axis: 0.00 degrees" in captured.out
        assert "Rotation around Y (pitch) axis: 0.00 degrees" in captured.out
        assert "Rotation around Z (yaw) axis: 0.00 degrees" in captured.out

    def test_rotation_around_x_axis(self, capsys):
        """Test extraction of rotation around X axis (roll)."""
        angle_deg = 30.0
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array(
            [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
        )

        _extract_rotation(affine_matrix)

        captured = capsys.readouterr()
        assert "Rotation around X (roll) axis: 30.00 degrees" in captured.out

    def test_rotation_around_y_axis(self, capsys):
        """Test extraction of rotation around Y axis (pitch)."""
        angle_deg = 45.0
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array(
            [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
        )

        _extract_rotation(affine_matrix)

        captured = capsys.readouterr()
        assert "Rotation around Y (pitch) axis: 45.00 degrees" in captured.out

    def test_rotation_around_z_axis(self, capsys):
        """Test extraction of rotation around Z axis (yaw)."""
        angle_deg = 60.0
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
        )

        _extract_rotation(affine_matrix)

        captured = capsys.readouterr()
        assert "Rotation around Z (yaw) axis: 60.00 degrees" in captured.out

    def test_combined_rotation_xyz(self, capsys):
        """Test extraction of combined rotation around all axes."""
        angles_deg = np.array([15.0, 25.0, 35.0])  # X, Y, Z rotations
        r = Rotation.from_euler("xyz", angles_deg, degrees=True)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = r.as_matrix()

        _extract_rotation(affine_matrix)

        captured = capsys.readouterr()
        # Check that all three rotation components are present
        assert "Rotation around X (roll) axis:" in captured.out
        assert "Rotation around Y (pitch) axis:" in captured.out
        assert "Rotation around Z (yaw) axis:" in captured.out

        # The extracted angles should be close to the original
        assert "15.00 degrees" in captured.out
        assert "25.00 degrees" in captured.out
        assert "35.00 degrees" in captured.out

    def test_negative_rotation(self, capsys):
        """Test extraction of negative rotation angles."""
        angle_deg = -45.0
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
        )

        _extract_rotation(affine_matrix)

        captured = capsys.readouterr()
        assert "Rotation around Z (yaw) axis: -45.00 degrees" in captured.out


class TestExtractScale:
    """Test suite for _extract_scale function."""

    def test_identity_uniform_scale(self, capsys):
        """Test that identity transform has unit scale."""
        affine_matrix = np.eye(4)
        _extract_scale(affine_matrix)

        captured = capsys.readouterr()
        assert "Z Scale: 1.00-fold" in captured.out
        assert "Y Scale: 1.00-fold" in captured.out
        assert "X Scale: 1.00-fold" in captured.out

    def test_uniform_scale_greater_than_one(self, capsys):
        """Test extraction of uniform scaling > 1."""
        scale_factor = 2.5
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.eye(3) * scale_factor

        _extract_scale(affine_matrix)

        captured = capsys.readouterr()
        assert "Z Scale: 2.50-fold" in captured.out
        assert "Y Scale: 2.50-fold" in captured.out
        assert "X Scale: 2.50-fold" in captured.out

    def test_uniform_scale_less_than_one(self, capsys):
        """Test extraction of uniform scaling < 1."""
        scale_factor = 0.5
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.eye(3) * scale_factor

        _extract_scale(affine_matrix)

        captured = capsys.readouterr()
        assert "Z Scale: 0.50-fold" in captured.out
        assert "Y Scale: 0.50-fold" in captured.out
        assert "X Scale: 0.50-fold" in captured.out

    def test_anisotropic_scale(self, capsys):
        """Test extraction of different scales for each axis."""
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.diag([0.5, 1.5, 2.0])  # Z, Y, X scales

        _extract_scale(affine_matrix)

        captured = capsys.readouterr()
        assert "Z Scale: 0.50-fold" in captured.out
        assert "Y Scale: 1.50-fold" in captured.out
        assert "X Scale: 2.00-fold" in captured.out

    def test_scale_with_rotation(self, capsys):
        """Test scale extraction from a transform with rotation."""
        # Create a scale matrix
        scales = np.array([0.8, 1.2, 1.5])

        # Create a rotation matrix (45 degrees around Z)
        angle = np.radians(45)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        # Combine: first scale, then rotate
        combined = rotation @ np.diag(scales)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = combined

        _extract_scale(affine_matrix)

        captured = capsys.readouterr()
        # Should extract the original scale values
        assert "Z Scale: 0.80-fold" in captured.out
        assert "Y Scale: 1.20-fold" in captured.out
        assert "X Scale: 1.50-fold" in captured.out

    def test_negative_determinant_scale(self, capsys):
        """Test scale extraction when determinant is negative (reflection)."""
        # Create a scale with one negative value (reflection)
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.diag([-1.0, 1.0, 1.0])  # Reflection in Z

        _extract_scale(affine_matrix)

        captured = capsys.readouterr()
        # Should report absolute values (as per the abs() in the function)
        assert "Z Scale: 1.00-fold" in captured.out
        assert "Y Scale: 1.00-fold" in captured.out
        assert "X Scale: 1.00-fold" in captured.out


class TestExtractShear:
    """Test suite for _extract_shear function."""

    def test_identity_no_shear(self, capsys):
        """Test that identity transform has zero shear."""
        affine_matrix = np.eye(4)
        _extract_shear(affine_matrix)

        captured = capsys.readouterr()
        assert "Shear angle XY: 0.00 degrees" in captured.out
        assert "Shear angle XZ: 0.00 degrees" in captured.out
        assert "Shear angle YZ: 0.00 degrees" in captured.out

    def test_shear_xy(self, capsys):
        """Test extraction of XY shear."""
        shear_factor = 0.5
        affine_matrix = np.eye(4)
        # Upper triangular shear matrix
        affine_matrix[:3, :3] = np.array([[1, shear_factor, 0], [0, 1, 0], [0, 0, 1]])

        _extract_shear(affine_matrix)

        captured = capsys.readouterr()
        expected_angle = np.degrees(np.arctan(shear_factor))
        assert f"Shear angle XY: {expected_angle:.2f} degrees" in captured.out

    def test_shear_xz(self, capsys):
        """Test extraction of XZ shear."""
        shear_factor = 0.3
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array([[1, 0, shear_factor], [0, 1, 0], [0, 0, 1]])

        _extract_shear(affine_matrix)

        captured = capsys.readouterr()
        expected_angle = np.degrees(np.arctan(shear_factor))
        assert f"Shear angle XZ: {expected_angle:.2f} degrees" in captured.out

    def test_shear_yz(self, capsys):
        """Test extraction of YZ shear."""
        shear_factor = 0.4
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array([[1, 0, 0], [0, 1, shear_factor], [0, 0, 1]])

        _extract_shear(affine_matrix)

        captured = capsys.readouterr()
        expected_angle = np.degrees(np.arctan(shear_factor))
        assert f"Shear angle YZ: {expected_angle:.2f} degrees" in captured.out

    def test_combined_shear(self, capsys):
        """Test extraction of multiple shear components."""
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array([[1, 0.2, 0.1], [0, 1, 0.3], [0, 0, 1]])

        _extract_shear(affine_matrix)

        captured = capsys.readouterr()
        assert "Shear angle XY:" in captured.out
        assert "Shear angle XZ:" in captured.out
        assert "Shear angle YZ:" in captured.out


class TestInspectAffineTransform:
    """Test suite for inspect_affine_transform function."""

    def create_mock_transform(self, affine_matrix, center=None):
        """Create a mock ANTs transform from a 4x4 affine matrix.

        Parameters
        ----------
        affine_matrix : np.ndarray
            4x4 affine transformation matrix
        center : np.ndarray, optional
            Center of rotation (3D point). Defaults to origin.

        Returns
        -------
        mock_transform : Mock
            Mock ANTsTransform object
        """
        if center is None:
            center = np.array([0.0, 0.0, 0.0])

        # Extract the rotation/scale/shear matrix and translation
        rotation_scale_shear = affine_matrix[:3, :3]
        translation = affine_matrix[:3, 3]

        # ANTs stores the transform differently:
        # The transformation is: y = A * (x - center) + translation + center
        # So we need to adjust the translation
        adjusted_translation = translation - center + rotation_scale_shear @ center

        # Flatten the 3x3 matrix and append translation to get 12 parameters
        parameters = np.concatenate(
            [rotation_scale_shear.flatten(), adjusted_translation]
        )

        mock_transform = Mock(spec=ants.core.ants_transform.ANTsTransform)
        mock_transform.parameters = parameters
        mock_transform.fixed_parameters = center

        return mock_transform

    def test_identity_transform(self, capsys):
        """Test inspection of identity transform."""
        affine_matrix = np.eye(4)
        mock_transform = self.create_mock_transform(affine_matrix)

        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        # Check for identity values
        assert "Translation distance in Z: 0.00 voxels" in captured.out
        assert "Translation distance in Y: 0.00 voxels" in captured.out
        assert "Translation distance in X: 0.00 voxels" in captured.out
        assert "Z Scale: 1.00-fold" in captured.out
        assert "Y Scale: 1.00-fold" in captured.out
        assert "X Scale: 1.00-fold" in captured.out
        assert "Complete Affine 4x4 matrix" in captured.out

    def test_pure_translation_transform(self, capsys):
        """Test inspection of pure translation."""
        affine_matrix = np.eye(4)
        affine_matrix[:3, 3] = [5.0, 10.0, 15.0]

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        assert "Translation distance in Z: 5.00 voxels" in captured.out
        assert "Translation distance in Y: 10.00 voxels" in captured.out
        assert "Translation distance in X: 15.00 voxels" in captured.out

    def test_pure_scale_transform(self, capsys):
        """Test inspection of pure scaling."""
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.diag([0.5, 1.5, 2.0])

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        assert "Z Scale: 0.50-fold" in captured.out
        assert "Y Scale: 1.50-fold" in captured.out
        assert "X Scale: 2.00-fold" in captured.out

    def test_pure_rotation_transform(self, capsys):
        """Test inspection of pure rotation."""
        angle_deg = 30.0
        r = Rotation.from_euler("z", angle_deg, degrees=True)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = r.as_matrix()

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        assert "Rotation around Z (yaw) axis: 30.00 degrees" in captured.out

    def test_combined_transform(self, capsys):
        """Test inspection of combined rotation, scale, and translation."""
        # Build a combined transform: Scale -> Rotate -> Translate
        scale = np.diag([0.8, 1.2, 1.5])
        rotation = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        translation = np.array([5.0, 10.0, 15.0])

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation @ scale
        affine_matrix[:3, 3] = translation

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        # Check that all components are extracted
        assert "Translation distance" in captured.out
        assert "Scale:" in captured.out
        assert "Rotation around" in captured.out
        assert "Shear angle" in captured.out

    def test_transform_with_non_zero_center(self, capsys):
        """Test transform with non-zero center of rotation."""
        center = np.array([50.0, 50.0, 50.0])

        # Simple rotation around non-zero center
        angle_deg = 45.0
        r = Rotation.from_euler("z", angle_deg, degrees=True)

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = r.as_matrix()
        affine_matrix[:3, 3] = [10.0, 20.0, 0.0]  # Some translation

        mock_transform = self.create_mock_transform(affine_matrix, center=center)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        assert "Complete Affine 4x4 matrix" in captured.out
        assert "Rotation around Z (yaw) axis: 45.00 degrees" in captured.out

    def test_transform_with_shear(self, capsys):
        """Test transform including shear components."""
        affine_matrix = np.eye(4)
        # Add some shear
        affine_matrix[:3, :3] = np.array(
            [[1.0, 0.3, 0.1], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]
        )
        affine_matrix[:3, 3] = [2.0, 4.0, 6.0]

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        assert "Shear angle XY:" in captured.out
        assert "Shear angle XZ:" in captured.out
        assert "Shear angle YZ:" in captured.out

    def test_realistic_registration_transform(self, capsys):
        """Test a realistic registration scenario with small rotation, scale, and translation."""
        # Typical registration might have:
        # - Small rotation (few degrees)
        # - Slight anisotropic scaling (e.g., different Z vs XY)
        # - Translation

        # Small rotation around each axis
        rotation = Rotation.from_euler("xyz", [2, -3, 1.5], degrees=True).as_matrix()

        # Anisotropic scale (common in microscopy - different Z and XY resolution)
        scale = np.diag([0.62, 1.02, 1.02])  # Z different from XY

        # Translation
        translation = np.array([1.5, -2.3, 3.7])

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation @ scale
        affine_matrix[:3, 3] = translation

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        # Verify all components are reported
        assert "Translation distance in Z:" in captured.out
        assert "Translation distance in Y:" in captured.out
        assert "Translation distance in X:" in captured.out
        assert "Z Scale: 0.62-fold" in captured.out
        assert "Y Scale: 1.02-fold" in captured.out
        assert "X Scale: 1.02-fold" in captured.out
        assert "Rotation around" in captured.out

    def test_reflection_transform(self, capsys):
        """Test transform with reflection (negative determinant)."""
        affine_matrix = np.eye(4)
        # Reflection in one axis
        affine_matrix[:3, :3] = np.diag([-1.0, 1.0, 1.0])

        mock_transform = self.create_mock_transform(affine_matrix)
        inspect_affine_transform(mock_transform)

        captured = capsys.readouterr()
        # Should handle gracefully and report absolute scale
        assert "Scale:" in captured.out


class TestAffineMatrixDecomposition:
    """Test the mathematical correctness of the decomposition algorithms."""

    def test_polar_decomposition_orthogonality(self):
        """Test that polar decomposition produces orthogonal rotation matrix."""
        # Create a random affine matrix
        np.random.seed(42)
        affine_matrix = np.random.randn(3, 3)

        rotation, scaling_shear = polar(affine_matrix)

        # Check that rotation is orthogonal (R @ R.T = I)
        identity = rotation @ rotation.T
        assert np.allclose(identity, np.eye(3), atol=1e-10)

        # Check determinant is +1 (proper rotation)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)

    def test_rq_decomposition_upper_triangular(self):
        """Test that RQ decomposition produces upper triangular matrix."""
        np.random.seed(42)
        matrix = np.random.randn(3, 3)

        scale_matrix, shear_matrix = rq(matrix)

        # R should be upper triangular
        # Check that lower triangle (excluding diagonal) is zero
        lower_triangle = np.tril(scale_matrix, k=-1)
        assert np.allclose(lower_triangle, 0, atol=1e-10)

    def test_decomposition_reconstruction(self):
        """Test that decomposed transform can be reconstructed."""
        # Create a known transform
        scale = np.diag([0.5, 1.5, 2.0])
        rotation = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
        combined = rotation @ scale

        # Decompose
        extracted_rotation, scaling_shear = polar(combined)
        extracted_scale, _ = rq(scaling_shear)

        # Reconstruct
        reconstructed = extracted_rotation @ scaling_shear

        # Should match original
        assert np.allclose(reconstructed, combined, atol=1e-10)

    def test_scale_extraction_accuracy(self):
        """Test that scale extraction is accurate for known scales."""
        original_scales = np.array([0.5, 1.5, 2.5])
        rotation = Rotation.from_euler("xyz", [15, 25, 35], degrees=True).as_matrix()

        combined = rotation @ np.diag(original_scales)

        # Extract scales using the same method as _extract_scale
        _, scaling_shear = polar(combined)
        scale_matrix, _ = rq(scaling_shear)
        extracted_scales = np.abs(np.diagonal(scale_matrix))

        # Should match original scales
        assert np.allclose(extracted_scales, original_scales, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
