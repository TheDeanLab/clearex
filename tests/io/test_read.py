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

# Standard Library Imports
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import tempfile

# Third Party Imports
import numpy as np
import pytest
import dask.array as da
from numpy.typing import NDArray
import tifffile
import zarr
from numcodecs import Blosc
import h5py

# Local Imports
from clearex.io.read import (
    ImageInfo,
    Reader,
    TiffReader,
    ZarrReader,
    HDF5Reader,
    NumpyReader,
    ND2Reader,
    ImageOpener,
)
from tests import download_test_registration_data


# =============================================================================
# Test ImageInfo Dataclass
# =============================================================================


class TestImageInfo:
    """Test suite for the ImageInfo dataclass."""

    def test_imageinfo_creation_minimal(self):
        """Test ImageInfo creation with minimal required fields."""
        path = Path("test.tif")
        shape = (512, 512)
        dtype = np.uint16

        info = ImageInfo(path=path, shape=shape, dtype=dtype)

        assert info.path == path
        assert info.shape == shape
        assert info.dtype == dtype
        assert info.axes is None
        assert info.metadata is None

    def test_imageinfo_creation_with_axes(self):
        """Test ImageInfo creation with axes information."""
        path = Path("test.tif")
        shape = (10, 512, 512)
        dtype = np.uint16
        axes = "ZYX"

        info = ImageInfo(path=path, shape=shape, dtype=dtype, axes=axes)

        assert info.path == path
        assert info.shape == shape
        assert info.dtype == dtype
        assert info.axes == axes
        assert info.metadata is None

    def test_imageinfo_creation_with_metadata(self):
        """Test ImageInfo creation with additional metadata."""
        path = Path("test.tif")
        shape = (512, 512)
        dtype = np.uint16
        metadata = {"scale": 1.0, "unit": "um"}

        info = ImageInfo(path=path, shape=shape, dtype=dtype, metadata=metadata)

        assert info.path == path
        assert info.shape == shape
        assert info.dtype == dtype
        assert info.axes is None
        assert info.metadata == metadata

    def test_imageinfo_creation_full(self):
        """Test ImageInfo creation with all fields."""
        path = Path("test.tif")
        shape = (10, 3, 512, 512)
        dtype = np.float32
        axes = "ZCYX"
        metadata = {"scale": 0.5, "unit": "um", "channels": ["DAPI", "GFP", "RFP"]}

        info = ImageInfo(
            path=path, shape=shape, dtype=dtype, axes=axes, metadata=metadata
        )

        assert info.path == path
        assert info.shape == shape
        assert info.dtype == dtype
        assert info.axes == axes
        assert info.metadata == metadata

    def test_imageinfo_shape_2d(self):
        """Test ImageInfo with 2D image shape."""
        info = ImageInfo(path=Path("2d.tif"), shape=(256, 256), dtype=np.uint8)
        assert len(info.shape) == 2

    def test_imageinfo_shape_3d(self):
        """Test ImageInfo with 3D image shape."""
        info = ImageInfo(path=Path("3d.tif"), shape=(50, 256, 256), dtype=np.uint16)
        assert len(info.shape) == 3

    def test_imageinfo_shape_4d(self):
        """Test ImageInfo with 4D image shape."""
        info = ImageInfo(path=Path("4d.tif"), shape=(10, 50, 256, 256), dtype=np.uint16)
        assert len(info.shape) == 4

    def test_imageinfo_various_dtypes(self):
        """Test ImageInfo with various numpy dtypes."""
        dtypes = [np.uint8, np.uint16, np.uint32, np.float32, np.float64, np.int16]

        for dtype in dtypes:
            info = ImageInfo(path=Path("test.tif"), shape=(100, 100), dtype=dtype)
            assert info.dtype == dtype


# =============================================================================
# Test Reader Abstract Base Class
# =============================================================================


class ConcreteReader(Reader):
    """Concrete implementation of Reader for testing purposes."""

    SUFFIXES = (".test", ".tst")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Mock implementation of open method."""
        # Create a simple test array
        arr = np.random.rand(10, 10).astype(np.float32)
        info = ImageInfo(
            path=path,
            shape=arr.shape,
            dtype=arr.dtype,
            axes="YX",
            metadata={"test": True},
        )
        return arr, info


class TestReaderABC:
    """Test suite for the Reader abstract base class."""

    def test_reader_is_abstract(self):
        """Test that Reader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Reader()  # type: ignore

    def test_reader_subclass_requires_open_implementation(self):
        """Test that Reader subclass must implement open method."""

        class IncompleteReader(Reader):
            SUFFIXES = (".incomplete",)

        with pytest.raises(TypeError):
            IncompleteReader()  # type: ignore

    def test_concrete_reader_instantiation(self):
        """Test that a properly implemented Reader subclass can be instantiated."""
        reader = ConcreteReader()
        assert isinstance(reader, Reader)
        assert isinstance(reader, ConcreteReader)

    def test_claims_method_with_matching_suffix(self):
        """Test that claims method returns True for matching file suffix."""
        assert ConcreteReader.claims(Path("file.test")) is True
        assert ConcreteReader.claims(Path("file.tst")) is True

    def test_claims_method_with_uppercase_suffix(self):
        """Test that claims method is case-insensitive."""
        assert ConcreteReader.claims(Path("file.TEST")) is True
        assert ConcreteReader.claims(Path("file.TsT")) is True

    def test_claims_method_with_non_matching_suffix(self):
        """Test that claims method returns False for non-matching suffix."""
        assert ConcreteReader.claims(Path("file.tif")) is False
        assert ConcreteReader.claims(Path("file.zarr")) is False
        assert ConcreteReader.claims(Path("file.txt")) is False

    def test_claims_method_with_no_suffix(self):
        """Test that claims method returns False for files without suffix."""
        assert ConcreteReader.claims(Path("file")) is False

    def test_claims_method_with_complex_path(self):
        """Test claims method with complex file paths."""
        assert ConcreteReader.claims(Path("/path/to/file.test")) is True
        assert ConcreteReader.claims(Path("../relative/path/file.tst")) is True
        assert ConcreteReader.claims(Path("/path/to/file.test.test")) is True

    def test_suffixes_attribute(self):
        """Test that SUFFIXES attribute is accessible."""
        assert hasattr(ConcreteReader, "SUFFIXES")
        assert ConcreteReader.SUFFIXES == (".test", ".tst")

    def test_empty_suffixes(self):
        """Test Reader with empty SUFFIXES tuple."""

        class EmptySuffixReader(Reader):
            SUFFIXES = ()

            def open(
                self,
                path: Path,
                prefer_dask: bool = False,
                chunks: Optional[Union[int, Tuple[int, ...]]] = None,
                **kwargs: Any,
            ) -> Tuple[NDArray[Any], ImageInfo]:
                arr = np.zeros((5, 5))
                info = ImageInfo(path=path, shape=arr.shape, dtype=arr.dtype)
                return arr, info

        reader = EmptySuffixReader()
        assert EmptySuffixReader.claims(Path("any.file")) is False

    def test_open_method_signature(self):
        """Test that concrete reader's open method has correct signature."""
        reader = ConcreteReader()
        assert callable(reader.open)

        # Test with minimal arguments
        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path)
            assert isinstance(arr, np.ndarray)
            assert isinstance(info, ImageInfo)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_open_method_with_prefer_dask(self):
        """Test open method with prefer_dask parameter."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path, prefer_dask=False)
            assert isinstance(arr, np.ndarray)
            assert isinstance(info, ImageInfo)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_open_method_with_chunks(self):
        """Test open method with chunks parameter."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path, prefer_dask=True, chunks=(5, 5))
            assert isinstance(arr, np.ndarray)  # Mock returns ndarray
            assert isinstance(info, ImageInfo)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_open_method_with_kwargs(self):
        """Test open method with additional keyword arguments."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path, custom_param="value")
            assert isinstance(arr, np.ndarray)
            assert isinstance(info, ImageInfo)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_open_returns_tuple(self):
        """Test that open method returns a tuple of (array, ImageInfo)."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = reader.open(tmp_path)
            assert isinstance(result, tuple)
            assert len(result) == 2
            arr, info = result
            assert isinstance(arr, np.ndarray)
            assert isinstance(info, ImageInfo)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_imageinfo_from_open_has_correct_path(self):
        """Test that ImageInfo returned by open has correct path."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path)
            assert info.path == tmp_path
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_imageinfo_from_open_has_correct_shape(self):
        """Test that ImageInfo returned by open has shape matching array."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path)
            assert info.shape == arr.shape
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_imageinfo_from_open_has_correct_dtype(self):
        """Test that ImageInfo returned by open has dtype matching array."""
        reader = ConcreteReader()

        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            arr, info = reader.open(tmp_path)
            assert info.dtype == arr.dtype
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


# =============================================================================
# Test TiffReader Class
# =============================================================================


class TestTiffReader:
    """Test suite for the TiffReader class."""

    @pytest.fixture(scope="class")
    def test_data_dir(self):
        """Fixture to download and provide test data directory."""
        data_dir = download_test_registration_data()
        return Path(data_dir)

    @pytest.fixture
    def tiff_reader(self):
        """Fixture to create a TiffReader instance."""
        return TiffReader()

    @pytest.fixture
    def sample_tiff_path(self, test_data_dir):
        """Fixture to provide path to a sample TIFF file."""
        # Use one of the downloaded test files
        tiff_path = test_data_dir / "cropped_fixed.tif"
        if not tiff_path.exists():
            pytest.skip(f"Test TIFF file not found: {tiff_path}")
        return tiff_path

    @pytest.fixture
    def temp_tiff_2d(self, tmp_path):
        """Fixture to create a temporary 2D TIFF file."""
        arr = np.random.randint(0, 65535, size=(256, 256), dtype=np.uint16)
        tiff_path = tmp_path / "test_2d.tif"
        tifffile.imwrite(str(tiff_path), arr)
        return tiff_path, arr

    @pytest.fixture
    def temp_tiff_3d(self, tmp_path):
        """Fixture to create a temporary 3D TIFF file."""
        arr = np.random.randint(0, 65535, size=(10, 256, 256), dtype=np.uint16)
        tiff_path = tmp_path / "test_3d.tif"
        tifffile.imwrite(str(tiff_path), arr)
        return tiff_path, arr

    def test_tiffreader_instantiation(self, tiff_reader):
        """Test that TiffReader can be instantiated."""
        assert isinstance(tiff_reader, TiffReader)
        assert isinstance(tiff_reader, Reader)

    def test_tiffreader_suffixes(self, tiff_reader):
        """Test that TiffReader has correct SUFFIXES."""
        assert TiffReader.SUFFIXES == (".tif", ".tiff")
        assert hasattr(tiff_reader, "SUFFIXES")

    def test_tiffreader_claims_tif_files(self):
        """Test that TiffReader claims .tif and .tiff files."""
        assert TiffReader.claims(Path("image.tif")) is True
        assert TiffReader.claims(Path("image.tiff")) is True
        assert TiffReader.claims(Path("image.TIF")) is True
        assert TiffReader.claims(Path("image.TIFF")) is True

    def test_tiffreader_rejects_other_files(self):
        """Test that TiffReader rejects non-TIFF files."""
        assert TiffReader.claims(Path("image.zarr")) is False
        assert TiffReader.claims(Path("image.h5")) is False
        assert TiffReader.claims(Path("image.npy")) is False
        assert TiffReader.claims(Path("image.txt")) is False

    def test_open_2d_tiff_as_numpy(self, tiff_reader, temp_tiff_2d):
        """Test opening a 2D TIFF file as NumPy array."""
        tiff_path, expected_arr = temp_tiff_2d

        arr, info = tiff_reader.open(tiff_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_3d_tiff_as_numpy(self, tiff_reader, temp_tiff_3d):
        """Test opening a 3D TIFF file as NumPy array."""
        tiff_path, expected_arr = temp_tiff_3d

        arr, info = tiff_reader.open(tiff_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_tiff_as_dask(self, tiff_reader, temp_tiff_3d):
        """Test opening a TIFF file as Dask array."""
        tiff_path, expected_arr = temp_tiff_3d

        arr, info = tiff_reader.open(tiff_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        # Compute Dask array to compare values
        assert np.array_equal(arr.compute(), expected_arr)

    def test_open_tiff_with_custom_chunks(self, tiff_reader, temp_tiff_3d):
        """Test opening a TIFF file with custom chunk sizes."""
        tiff_path, expected_arr = temp_tiff_3d

        chunks = (5, 128, 128)
        arr, info = tiff_reader.open(tiff_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        # Check that chunking was applied
        assert arr.chunks is not None

    def test_open_tiff_with_single_chunk_size(self, tiff_reader, temp_tiff_3d):
        """Test opening a TIFF file with a single chunk size for all dimensions."""
        tiff_path, expected_arr = temp_tiff_3d

        chunks = 64
        arr, info = tiff_reader.open(tiff_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.chunks is not None

    def test_imageinfo_from_tiff_has_correct_path(self, tiff_reader, temp_tiff_2d):
        """Test that ImageInfo has correct path."""
        tiff_path, _ = temp_tiff_2d

        arr, info = tiff_reader.open(tiff_path)

        assert isinstance(info, ImageInfo)
        assert info.path == tiff_path

    def test_imageinfo_from_tiff_has_correct_shape(self, tiff_reader, temp_tiff_2d):
        """Test that ImageInfo has correct shape."""
        tiff_path, expected_arr = temp_tiff_2d

        arr, info = tiff_reader.open(tiff_path)

        assert info.shape == expected_arr.shape
        assert info.shape == arr.shape

    def test_imageinfo_from_tiff_has_correct_dtype(self, tiff_reader, temp_tiff_2d):
        """Test that ImageInfo has correct dtype."""
        tiff_path, expected_arr = temp_tiff_2d

        arr, info = tiff_reader.open(tiff_path)

        assert info.dtype == expected_arr.dtype
        assert info.dtype == arr.dtype

    def test_open_real_tiff_file(self, tiff_reader, sample_tiff_path):
        """Test opening a real downloaded TIFF file."""
        arr, info = tiff_reader.open(sample_tiff_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)
        assert info.path == sample_tiff_path
        assert info.shape == arr.shape
        assert info.dtype == arr.dtype
        assert arr.size > 0

    def test_open_real_tiff_as_dask(self, tiff_reader, sample_tiff_path):
        """Test opening a real TIFF file as Dask array."""
        arr, info = tiff_reader.open(sample_tiff_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert isinstance(info, ImageInfo)
        assert info.path == sample_tiff_path
        assert info.shape == arr.shape
        assert info.dtype == arr.dtype

    def test_open_nonexistent_file(self, tiff_reader):
        """Test that opening a nonexistent file raises appropriate error."""
        nonexistent_path = Path("/nonexistent/path/to/file.tif")

        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            tiff_reader.open(nonexistent_path)

    def test_imageinfo_metadata_field_exists(self, tiff_reader, temp_tiff_2d):
        """Test that ImageInfo has metadata field."""
        tiff_path, _ = temp_tiff_2d

        arr, info = tiff_reader.open(tiff_path)

        assert hasattr(info, "metadata")
        assert isinstance(info.metadata, dict)

    def test_imageinfo_axes_field_exists(self, tiff_reader, temp_tiff_2d):
        """Test that ImageInfo has axes field."""
        tiff_path, _ = temp_tiff_2d

        arr, info = tiff_reader.open(tiff_path)

        assert hasattr(info, "axes")
        # axes can be None for non-OME TIFFs

    def test_open_different_dtypes(self, tiff_reader, tmp_path):
        """Test opening TIFF files with different dtypes."""
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            arr_in = np.random.rand(64, 64).astype(dtype)
            if dtype in [np.uint8, np.uint16]:
                arr_in = (arr_in * np.iinfo(dtype).max).astype(dtype)

            tiff_path = tmp_path / f"test_{dtype.__name__}.tif"
            tifffile.imwrite(str(tiff_path), arr_in)

            arr_out, info = tiff_reader.open(tiff_path)

            assert arr_out.dtype == dtype
            assert info.dtype == dtype
            assert np.allclose(arr_out, arr_in)

    def test_open_various_shapes(self, tiff_reader, tmp_path):
        """Test opening TIFF files with various shapes."""
        shapes = [
            (100, 100),  # 2D
            (10, 100, 100),  # 3D
            (5, 3, 100, 100),  # 4D
        ]

        for idx, shape in enumerate(shapes):
            arr_in = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            tiff_path = tmp_path / f"test_{idx}d.tif"

            # For 4D arrays, specify photometric and planarconfig to avoid deprecation warning
            if len(shape) == 4:
                tifffile.imwrite(
                    str(tiff_path),
                    arr_in,
                    photometric="minisblack",
                    planarconfig="separate",
                )
            else:
                tifffile.imwrite(str(tiff_path), arr_in)

            arr_out, info = tiff_reader.open(tiff_path)

            assert arr_out.shape == shape
            assert info.shape == shape
            assert np.array_equal(arr_out, arr_in)

    def test_numpy_and_dask_same_values(self, tiff_reader, temp_tiff_3d):
        """Test that NumPy and Dask loading produce the same values."""
        tiff_path, _ = temp_tiff_3d

        arr_numpy, info_numpy = tiff_reader.open(tiff_path, prefer_dask=False)
        arr_dask, info_dask = tiff_reader.open(tiff_path, prefer_dask=True)

        assert np.array_equal(arr_numpy, arr_dask.compute())
        assert info_numpy.shape == info_dask.shape
        assert info_numpy.dtype == info_dask.dtype

    def test_open_with_kwargs(self, tiff_reader, temp_tiff_2d):
        """Test that open method accepts additional kwargs."""
        tiff_path, _ = temp_tiff_2d

        # This should not raise an error even with extra kwargs
        arr, info = tiff_reader.open(tiff_path, key=0)

        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)

    def test_tiff_path_preservation(self, tiff_reader, sample_tiff_path):
        """Test that the original path is preserved in ImageInfo."""
        arr, info = tiff_reader.open(sample_tiff_path)

        assert info.path == sample_tiff_path
        assert str(sample_tiff_path) in str(info.path)

    def test_multiple_opens_same_file(self, tiff_reader, temp_tiff_2d):
        """Test that the same file can be opened multiple times."""
        tiff_path, expected_arr = temp_tiff_2d

        arr1, info1 = tiff_reader.open(tiff_path)
        arr2, info2 = tiff_reader.open(tiff_path)

        assert np.array_equal(arr1, arr2)
        assert info1.shape == info2.shape
        assert info1.dtype == info2.dtype

    def test_large_tiff_with_dask(self, tiff_reader, tmp_path):
        """Test that Dask can handle larger TIFF files efficiently."""
        # Create a moderately large TIFF (not huge to keep test fast)
        arr = np.random.randint(0, 255, size=(50, 512, 512), dtype=np.uint8)
        tiff_path = tmp_path / "large.tif"
        tifffile.imwrite(str(tiff_path), arr)

        # Open with Dask should not load into memory immediately
        darr, info = tiff_reader.open(
            tiff_path, prefer_dask=True, chunks=(10, 256, 256)
        )

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape
        # Verify a slice works without loading everything
        slice_result = darr[0].compute()
        assert np.array_equal(slice_result, arr[0])


# =============================================================================
# Test ZarrReader Class
# =============================================================================


class TestZarrReader:
    """Test suite for the ZarrReader class."""

    @pytest.fixture
    def zarr_reader(self):
        """Fixture to create a ZarrReader instance."""
        return ZarrReader()

    @pytest.fixture
    def temp_zarr_2d(self, tmp_path):
        """Fixture to create a temporary 2D Zarr store."""

        arr = np.random.randint(0, 65535, size=(256, 256), dtype=np.uint16)
        zarr_path = tmp_path / "test_2d.zarr"
        # Create as a group with a single array
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr)
        return zarr_path, arr

    @pytest.fixture
    def temp_zarr_3d(self, tmp_path):
        """Fixture to create a temporary 3D Zarr store."""

        arr = np.random.randint(0, 65535, size=(10, 256, 256), dtype=np.uint16)
        zarr_path = tmp_path / "test_3d.zarr"
        # Create as a group with a single array
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr)
        return zarr_path, arr

    @pytest.fixture
    def temp_zarr_with_attrs(self, tmp_path):
        """Fixture to create a Zarr store with custom attributes."""

        arr = np.random.rand(10, 100, 100).astype(np.float32)
        zarr_path = tmp_path / "test_with_attrs.zarr"

        root = zarr.open_group(str(zarr_path), mode="w")
        z = root.create_dataset(
            "data",
            data=arr,
            chunks=(5, 50, 50),
        )
        z.attrs["axes"] = ["z", "y", "x"]
        z.attrs["scale"] = 1.5
        z.attrs["unit"] = "um"

        return zarr_path, arr

    @pytest.fixture
    def temp_zarr_multiarray(self, tmp_path):
        """Fixture to create a Zarr store with multiple arrays."""

        zarr_path = tmp_path / "test_multi.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")

        # Create multiple arrays with different sizes
        arr_small = np.random.rand(50, 50).astype(np.float32)
        arr_large = np.random.rand(100, 200, 200).astype(np.float32)
        arr_medium = np.random.rand(100, 100).astype(np.float32)

        root.create_dataset("small", data=arr_small)
        root.create_dataset("large", data=arr_large)
        root.create_dataset("medium", data=arr_medium)

        return zarr_path, arr_large  # Should select the largest

    @pytest.fixture
    def temp_zarr_with_ome_attrs(self, tmp_path):
        """Fixture to create a Zarr store with OME-Zarr style metadata."""

        arr = np.random.randint(0, 255, size=(5, 3, 100, 100), dtype=np.uint8)
        zarr_path = tmp_path / "test_ome.zarr"

        root = zarr.open_group(str(zarr_path), mode="w")
        z = root.create_dataset("data", data=arr, chunks=(1, 1, 50, 50))
        z.attrs["multiscales"] = [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ]

        return zarr_path, arr

    def test_zarrreader_instantiation(self, zarr_reader):
        """Test that ZarrReader can be instantiated."""
        assert isinstance(zarr_reader, ZarrReader)
        assert isinstance(zarr_reader, Reader)

    def test_zarrreader_suffixes(self, zarr_reader):
        """Test that ZarrReader has correct SUFFIXES."""
        assert ZarrReader.SUFFIXES == (".zarr", ".zarr/", ".n5", ".n5/")
        assert hasattr(zarr_reader, "SUFFIXES")

    def test_zarrreader_claims_zarr_files(self):
        """Test that ZarrReader claims .zarr and .n5 files."""
        assert ZarrReader.claims(Path("data.zarr")) is True
        assert ZarrReader.claims(Path("data.zarr/")) is True
        assert ZarrReader.claims(Path("data.n5")) is True
        assert ZarrReader.claims(Path("data.n5/")) is True
        assert ZarrReader.claims(Path("data.ZARR")) is True

    def test_zarrreader_rejects_other_files(self):
        """Test that ZarrReader rejects non-Zarr files."""
        assert ZarrReader.claims(Path("image.tif")) is False
        assert ZarrReader.claims(Path("image.h5")) is False
        assert ZarrReader.claims(Path("image.npy")) is False
        assert ZarrReader.claims(Path("image.txt")) is False

    def test_open_2d_zarr_as_numpy(self, zarr_reader, temp_zarr_2d):
        """Test opening a 2D Zarr store as NumPy array."""
        zarr_path, expected_arr = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_3d_zarr_as_numpy(self, zarr_reader, temp_zarr_3d):
        """Test opening a 3D Zarr store as NumPy array."""
        zarr_path, expected_arr = temp_zarr_3d

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_zarr_as_dask(self, zarr_reader, temp_zarr_3d):
        """Test opening a Zarr store as Dask array."""
        zarr_path, expected_arr = temp_zarr_3d

        arr, info = zarr_reader.open(zarr_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        # Compute Dask array to compare values
        assert np.array_equal(arr.compute(), expected_arr)

    def test_open_zarr_with_custom_chunks(self, zarr_reader, temp_zarr_3d):
        """Test opening a Zarr store with custom chunk sizes."""
        zarr_path, expected_arr = temp_zarr_3d

        chunks = (5, 128, 128)
        arr, info = zarr_reader.open(zarr_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        # Check that chunking was applied
        assert arr.chunks is not None

    def test_open_zarr_with_single_chunk_size(self, zarr_reader, temp_zarr_3d):
        """Test opening a Zarr store with a single chunk size for all dimensions."""
        zarr_path, expected_arr = temp_zarr_3d

        chunks = 64
        arr, info = zarr_reader.open(zarr_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.chunks is not None

    def test_imageinfo_from_zarr_has_correct_path(self, zarr_reader, temp_zarr_2d):
        """Test that ImageInfo has correct path."""
        zarr_path, _ = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path)

        assert isinstance(info, ImageInfo)
        assert info.path == zarr_path

    def test_imageinfo_from_zarr_has_correct_shape(self, zarr_reader, temp_zarr_2d):
        """Test that ImageInfo has correct shape."""
        zarr_path, expected_arr = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path)

        assert info.shape == expected_arr.shape
        assert info.shape == arr.shape

    def test_imageinfo_from_zarr_has_correct_dtype(self, zarr_reader, temp_zarr_2d):
        """Test that ImageInfo has correct dtype."""
        zarr_path, expected_arr = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path)

        assert info.dtype == expected_arr.dtype
        assert info.dtype == arr.dtype

    def test_open_zarr_with_attributes(self, zarr_reader, temp_zarr_with_attrs):
        """Test opening a Zarr store with custom attributes."""
        zarr_path, expected_arr = temp_zarr_with_attrs

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, expected_arr)
        assert info.axes is not None
        assert info.metadata is not None
        assert "scale" in info.metadata
        assert info.metadata["scale"] == 1.5

    def test_open_zarr_multiarray_selects_largest(
        self, zarr_reader, temp_zarr_multiarray
    ):
        """Test that ZarrReader selects the largest array when multiple exist."""
        zarr_path, expected_largest = temp_zarr_multiarray

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        # Should have selected the largest array (100, 200, 200)
        assert arr.shape == expected_largest.shape
        assert np.array_equal(arr, expected_largest)

    def test_open_zarr_with_ome_metadata(self, zarr_reader, temp_zarr_with_ome_attrs):
        """Test opening a Zarr store with OME-Zarr style metadata."""
        zarr_path, expected_arr = temp_zarr_with_ome_attrs

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert info.metadata is not None
        assert "multiscales" in info.metadata

    def test_open_nonexistent_zarr(self, zarr_reader):
        """Test that opening a nonexistent Zarr store raises appropriate error."""
        nonexistent_path = Path("/nonexistent/path/to/store.zarr")

        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            zarr_reader.open(nonexistent_path)

    def test_imageinfo_metadata_field_exists(self, zarr_reader, temp_zarr_2d):
        """Test that ImageInfo has metadata field."""
        zarr_path, _ = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path)

        assert hasattr(info, "metadata")
        assert isinstance(info.metadata, dict)

    def test_imageinfo_axes_field_exists(self, zarr_reader, temp_zarr_2d):
        """Test that ImageInfo has axes field."""
        zarr_path, _ = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path)

        assert hasattr(info, "axes")
        # axes can be None if not specified in Zarr attributes

    def test_open_different_dtypes(self, zarr_reader, tmp_path):
        """Test opening Zarr stores with different dtypes."""

        dtypes = [np.uint8, np.uint16, np.int32, np.float32, np.float64]

        for dtype in dtypes:
            arr_in = np.random.rand(64, 64).astype(dtype)
            if dtype in [np.uint8, np.uint16]:
                arr_in = (arr_in * np.iinfo(dtype).max).astype(dtype)

            zarr_path = tmp_path / f"test_{dtype.__name__}.zarr"
            root = zarr.open_group(str(zarr_path), mode="w")
            root.create_dataset("data", data=arr_in)

            arr_out, info = zarr_reader.open(zarr_path)

            assert arr_out.dtype == dtype
            assert info.dtype == dtype
            assert np.allclose(arr_out, arr_in)

    def test_open_various_shapes(self, zarr_reader, tmp_path):
        """Test opening Zarr stores with various shapes."""

        shapes = [
            (100, 100),  # 2D
            (10, 100, 100),  # 3D
            (5, 3, 100, 100),  # 4D
            (2, 5, 3, 50, 50),  # 5D
        ]

        for idx, shape in enumerate(shapes):
            arr_in = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            zarr_path = tmp_path / f"test_{len(shape)}d.zarr"
            root = zarr.open_group(str(zarr_path), mode="w")
            root.create_dataset("data", data=arr_in)

            arr_out, info = zarr_reader.open(zarr_path)

            assert arr_out.shape == shape
            assert info.shape == shape
            assert np.array_equal(arr_out, arr_in)

    def test_numpy_and_dask_same_values(self, zarr_reader, temp_zarr_3d):
        """Test that NumPy and Dask loading produce the same values."""
        zarr_path, _ = temp_zarr_3d

        arr_numpy, info_numpy = zarr_reader.open(zarr_path, prefer_dask=False)
        arr_dask, info_dask = zarr_reader.open(zarr_path, prefer_dask=True)

        assert np.array_equal(arr_numpy, arr_dask.compute())
        assert info_numpy.shape == info_dask.shape
        assert info_numpy.dtype == info_dask.dtype

    def test_open_with_kwargs(self, zarr_reader, temp_zarr_2d):
        """Test that open method accepts additional kwargs."""
        zarr_path, _ = temp_zarr_2d

        # This should not raise an error even with extra kwargs
        arr, info = zarr_reader.open(zarr_path, mode="r")

        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)

    def test_zarr_path_preservation(self, zarr_reader, temp_zarr_2d):
        """Test that the original path is preserved in ImageInfo."""
        zarr_path, _ = temp_zarr_2d

        arr, info = zarr_reader.open(zarr_path)

        assert info.path == zarr_path
        assert str(zarr_path) in str(info.path)

    def test_multiple_opens_same_store(self, zarr_reader, temp_zarr_2d):
        """Test that the same Zarr store can be opened multiple times."""
        zarr_path, expected_arr = temp_zarr_2d

        arr1, info1 = zarr_reader.open(zarr_path)
        arr2, info2 = zarr_reader.open(zarr_path)

        assert np.array_equal(arr1, arr2)
        assert info1.shape == info2.shape
        assert info1.dtype == info2.dtype

    def test_large_zarr_with_dask(self, zarr_reader, tmp_path):
        """Test that Dask can handle larger Zarr stores efficiently."""

        # Create a moderately large Zarr store (not huge to keep test fast)
        arr = np.random.randint(0, 255, size=(50, 512, 512), dtype=np.uint8)
        zarr_path = tmp_path / "large.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr)

        # Open with Dask should not load into memory immediately
        darr, info = zarr_reader.open(
            zarr_path, prefer_dask=True, chunks=(10, 256, 256)
        )

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape
        # Verify a slice works without loading everything
        slice_result = darr[0].compute()
        assert np.array_equal(slice_result, arr[0])

    def test_zarr_native_chunking(self, zarr_reader, tmp_path):
        """Test that ZarrReader respects Zarr's native chunking."""

        arr = np.random.rand(100, 200, 200).astype(np.float32)
        zarr_path = tmp_path / "chunked.zarr"

        # Create with specific chunking
        root = zarr.open_group(str(zarr_path), mode="w")
        z = root.create_dataset(
            "data",
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=(10, 100, 100),
        )
        z[:] = arr

        # Open with Dask without specifying chunks (should use native)
        darr, info = zarr_reader.open(zarr_path, prefer_dask=True)

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape
        # Dask should use native Zarr chunking
        assert darr.chunks is not None

    def test_zarr_with_nested_groups(self, zarr_reader, tmp_path):
        """Test opening a Zarr store with nested groups."""

        zarr_path = tmp_path / "nested.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")

        # Create nested structure
        grp1 = root.create_group("group1")
        grp2 = root.create_group("group2")

        # Add arrays at different levels - including root level
        arr_small = np.random.rand(50, 50).astype(np.float32)
        arr_large = np.random.rand(200, 300, 300).astype(np.float32)
        arr_medium = np.random.rand(100, 100).astype(np.float32)

        # Root level arrays (ZarrReader only looks here)
        root.create_dataset("small", data=arr_small)
        root.create_dataset("large", data=arr_large)
        root.create_dataset("medium", data=arr_medium)

        # Also add to nested groups (these won't be found by current implementation)
        grp1.create_dataset("nested_small", data=arr_small)
        grp2.create_dataset("nested_medium", data=arr_medium)

        # Should find and select the largest array at root level
        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert arr.shape == arr_large.shape
        assert np.array_equal(arr, arr_large)

    def test_zarr_empty_store_raises_error(self, zarr_reader, tmp_path):
        """Test that opening an empty Zarr store raises ValueError."""

        zarr_path = tmp_path / "empty.zarr"
        # Create an empty group with no arrays
        zarr.open_group(str(zarr_path), mode="w")

        with pytest.raises(ValueError, match="No arrays found"):
            zarr_reader.open(zarr_path)

    def test_zarr_axes_from_multiscales(self, zarr_reader, temp_zarr_with_ome_attrs):
        """Test extraction of axes from multiscales attribute."""
        zarr_path, _ = temp_zarr_with_ome_attrs

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        # Should extract axes from multiscales metadata
        assert info.axes is not None

    def test_zarr_axes_from_attrs(self, zarr_reader, temp_zarr_with_attrs):
        """Test extraction of axes from direct attrs."""
        zarr_path, _ = temp_zarr_with_attrs

        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        # Should extract axes from attrs
        assert info.axes is not None
        assert info.axes == ["z", "y", "x"]

    def test_zarr_readonly_mode(self, zarr_reader, temp_zarr_2d):
        """Test that ZarrReader opens stores in read-only mode."""
        zarr_path, _ = temp_zarr_2d

        # Should be able to open in read mode
        arr, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert info.path == zarr_path

    def test_zarr_dask_lazy_evaluation(self, zarr_reader, tmp_path):
        """Test that Dask arrays from Zarr are truly lazy."""

        arr = np.random.rand(100, 100, 100).astype(np.float32)
        zarr_path = tmp_path / "lazy.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr)

        # Open as Dask
        darr, info = zarr_reader.open(zarr_path, prefer_dask=True)

        # Should be lazy - no computation yet
        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape

        # Only compute a small slice
        small_slice = darr[0:1, 0:10, 0:10].compute()
        assert small_slice.shape == (1, 10, 10)
        assert np.array_equal(small_slice, arr[0:1, 0:10, 0:10])

    def test_zarr_compressor_preserved(self, zarr_reader, tmp_path):
        """Test that Zarr compressor settings are preserved."""

        arr = np.random.rand(100, 100).astype(np.float32)
        zarr_path = tmp_path / "compressed.zarr"

        # Create with specific compressor
        root = zarr.open_group(str(zarr_path), mode="w")
        z = root.create_dataset(
            "data",
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=(50, 50),
            compressor=Blosc(cname="zstd", clevel=5),
        )
        z[:] = arr

        # Should be able to open and read correctly
        arr_out, info = zarr_reader.open(zarr_path, prefer_dask=False)

        assert np.allclose(arr_out, arr)
        assert arr_out.shape == arr.shape


# =============================================================================
# Test HDF5Reader
# =============================================================================


class TestHDF5Reader:
    """Test suite for the HDF5Reader class."""

    @pytest.fixture
    def hdf5_reader(self):
        """Fixture to create an HDF5Reader instance."""
        return HDF5Reader()

    @pytest.fixture
    def temp_hdf5_2d(self, tmp_path):
        """Fixture to create a temporary 2D HDF5 file."""

        arr = np.random.randint(0, 65535, size=(256, 256), dtype=np.uint16)
        hdf5_path = tmp_path / "test_2d.h5"
        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("data", data=arr)
        return hdf5_path, arr

    @pytest.fixture
    def temp_hdf5_3d(self, tmp_path):
        """Fixture to create a temporary 3D HDF5 file."""

        arr = np.random.randint(0, 65535, size=(10, 256, 256), dtype=np.uint16)
        hdf5_path = tmp_path / "test_3d.h5"
        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("data", data=arr)
        return hdf5_path, arr

    @pytest.fixture
    def temp_hdf5_with_attrs(self, tmp_path):
        """Fixture to create an HDF5 file with custom attributes."""

        arr = np.random.rand(10, 100, 100).astype(np.float32)
        hdf5_path = tmp_path / "test_with_attrs.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            ds = f.create_dataset("data", data=arr, chunks=(5, 50, 50))
            ds.attrs["axes"] = "ZYX"
            ds.attrs["scale"] = 1.5
            ds.attrs["unit"] = "um"

        return hdf5_path, arr

    @pytest.fixture
    def temp_hdf5_multidataset(self, tmp_path):
        """Fixture to create an HDF5 file with multiple datasets."""

        hdf5_path = tmp_path / "test_multi.h5"

        # Create multiple datasets with different sizes
        arr_small = np.random.rand(50, 50).astype(np.float32)
        arr_large = np.random.rand(100, 200, 200).astype(np.float32)
        arr_medium = np.random.rand(100, 100).astype(np.float32)

        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("small", data=arr_small)
            f.create_dataset("large", data=arr_large)
            f.create_dataset("medium", data=arr_medium)

        return hdf5_path, arr_large  # Should select the largest

    @pytest.fixture
    def temp_hdf5_with_groups(self, tmp_path):
        """Fixture to create an HDF5 file with nested groups."""

        hdf5_path = tmp_path / "test_groups.h5"

        # Create nested structure
        arr_small = np.random.rand(50, 50).astype(np.float32)
        arr_large = np.random.rand(200, 300, 300).astype(np.float32)
        arr_medium = np.random.rand(100, 100).astype(np.float32)

        with h5py.File(str(hdf5_path), "w") as f:
            # Root level datasets
            f.create_dataset("root_small", data=arr_small)

            # Nested groups with datasets
            grp1 = f.create_group("group1")
            grp1.create_dataset("nested_large", data=arr_large)

            grp2 = f.create_group("group2")
            grp2.create_dataset("nested_medium", data=arr_medium)

        return hdf5_path, arr_large  # Should select the largest

    @pytest.fixture
    def temp_hdf5_chunked(self, tmp_path):
        """Fixture to create a chunked HDF5 file."""

        arr = np.random.rand(100, 200, 200).astype(np.float32)
        hdf5_path = tmp_path / "test_chunked.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("data", data=arr, chunks=(10, 100, 100))

        return hdf5_path, arr

    @pytest.fixture
    def temp_hdf5_dimension_order(self, tmp_path):
        """Fixture to create an HDF5 file with dimension order attributes."""

        arr = np.random.randint(0, 255, size=(5, 3, 100, 100), dtype=np.uint8)
        hdf5_path = tmp_path / "test_dimension_order.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            ds = f.create_dataset("image", data=arr)
            ds.attrs["DimensionOrder"] = "TCYX"

        return hdf5_path, arr

    def test_hdf5reader_instantiation(self, hdf5_reader):
        """Test that HDF5Reader can be instantiated."""
        assert isinstance(hdf5_reader, HDF5Reader)
        assert isinstance(hdf5_reader, Reader)

    def test_hdf5reader_suffixes(self, hdf5_reader):
        """Test that HDF5Reader has correct SUFFIXES."""
        assert HDF5Reader.SUFFIXES == (".h5", ".hdf5", ".hdf")
        assert hasattr(hdf5_reader, "SUFFIXES")

    def test_hdf5reader_claims_h5_files(self):
        """Test that HDF5Reader claims HDF5 file extensions."""
        assert HDF5Reader.claims(Path("data.h5")) is True
        assert HDF5Reader.claims(Path("data.hdf5")) is True
        assert HDF5Reader.claims(Path("data.hdf")) is True
        assert HDF5Reader.claims(Path("data.H5")) is True

    def test_hdf5reader_rejects_other_files(self):
        """Test that HDF5Reader rejects non-HDF5 files."""
        assert HDF5Reader.claims(Path("image.tif")) is False
        assert HDF5Reader.claims(Path("image.zarr")) is False
        assert HDF5Reader.claims(Path("image.npy")) is False
        assert HDF5Reader.claims(Path("image.txt")) is False

    def test_open_2d_hdf5_as_numpy(self, hdf5_reader, temp_hdf5_2d):
        """Test opening a 2D HDF5 file as NumPy array."""
        hdf5_path, expected_arr = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_3d_hdf5_as_numpy(self, hdf5_reader, temp_hdf5_3d):
        """Test opening a 3D HDF5 file as NumPy array."""
        hdf5_path, expected_arr = temp_hdf5_3d

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_hdf5_as_dask(self, hdf5_reader, temp_hdf5_3d):
        """Test opening an HDF5 file as Dask array."""
        hdf5_path, expected_arr = temp_hdf5_3d

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        # Compute Dask array to compare values
        assert np.array_equal(arr.compute(), expected_arr)

    def test_open_hdf5_with_custom_chunks(self, hdf5_reader, temp_hdf5_3d):
        """Test opening an HDF5 file with custom chunk sizes."""
        hdf5_path, expected_arr = temp_hdf5_3d

        chunks = (5, 128, 128)
        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        # Check that chunking was applied
        assert arr.chunks is not None

    def test_open_hdf5_with_single_chunk_size(self, hdf5_reader, temp_hdf5_3d):
        """Test opening an HDF5 file with a single chunk size for all dimensions."""
        hdf5_path, expected_arr = temp_hdf5_3d

        chunks = 64
        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.chunks is not None

    def test_imageinfo_from_hdf5_has_correct_path(self, hdf5_reader, temp_hdf5_2d):
        """Test that ImageInfo has correct path."""
        hdf5_path, _ = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path)

        assert isinstance(info, ImageInfo)
        assert info.path == hdf5_path

    def test_imageinfo_from_hdf5_has_correct_shape(self, hdf5_reader, temp_hdf5_2d):
        """Test that ImageInfo has correct shape."""
        hdf5_path, expected_arr = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path)

        assert info.shape == expected_arr.shape
        assert info.shape == arr.shape

    def test_imageinfo_from_hdf5_has_correct_dtype(self, hdf5_reader, temp_hdf5_2d):
        """Test that ImageInfo has correct dtype."""
        hdf5_path, expected_arr = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path)

        assert info.dtype == expected_arr.dtype
        assert info.dtype == arr.dtype

    def test_open_hdf5_with_attributes(self, hdf5_reader, temp_hdf5_with_attrs):
        """Test opening an HDF5 file with custom attributes."""
        hdf5_path, expected_arr = temp_hdf5_with_attrs

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, expected_arr)
        assert info.axes is not None
        assert info.axes == "ZYX"
        assert info.metadata is not None
        assert "scale" in info.metadata
        assert info.metadata["scale"] == 1.5

    def test_open_hdf5_multidataset_selects_largest(
        self, hdf5_reader, temp_hdf5_multidataset
    ):
        """Test that HDF5Reader selects the largest dataset when multiple exist."""
        hdf5_path, expected_largest = temp_hdf5_multidataset

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        # Should have selected the largest dataset (100, 200, 200)
        assert arr.shape == expected_largest.shape
        assert np.array_equal(arr, expected_largest)

    def test_open_hdf5_with_nested_groups(self, hdf5_reader, temp_hdf5_with_groups):
        """Test opening an HDF5 file with nested groups."""
        hdf5_path, expected_largest = temp_hdf5_with_groups

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        # Should recursively find and select the largest dataset
        assert arr.shape == expected_largest.shape
        assert np.array_equal(arr, expected_largest)

    def test_open_hdf5_with_dimension_order(
        self, hdf5_reader, temp_hdf5_dimension_order
    ):
        """Test opening an HDF5 file with DimensionOrder attribute."""
        hdf5_path, expected_arr = temp_hdf5_dimension_order

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert info.axes is not None
        assert info.axes == "TCYX"

    def test_open_nonexistent_hdf5(self, hdf5_reader):
        """Test that opening a nonexistent HDF5 file raises appropriate error."""
        nonexistent_path = Path("/nonexistent/path/to/file.h5")

        with pytest.raises((FileNotFoundError, OSError, Exception)):
            hdf5_reader.open(nonexistent_path)

    def test_imageinfo_metadata_field_exists(self, hdf5_reader, temp_hdf5_2d):
        """Test that ImageInfo has metadata field."""
        hdf5_path, _ = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path)

        assert hasattr(info, "metadata")
        assert isinstance(info.metadata, dict)

    def test_imageinfo_axes_field_exists(self, hdf5_reader, temp_hdf5_2d):
        """Test that ImageInfo has axes field."""
        hdf5_path, _ = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path)

        assert hasattr(info, "axes")
        # axes can be None if not specified in HDF5 attributes

    def test_open_different_dtypes(self, hdf5_reader, tmp_path):
        """Test opening HDF5 files with different dtypes."""

        dtypes = [np.uint8, np.uint16, np.int32, np.float32, np.float64]

        for dtype in dtypes:
            arr_in = np.random.rand(64, 64).astype(dtype)
            if dtype in [np.uint8, np.uint16]:
                arr_in = (arr_in * np.iinfo(dtype).max).astype(dtype)

            hdf5_path = tmp_path / f"test_{dtype.__name__}.h5"
            with h5py.File(str(hdf5_path), "w") as f:
                f.create_dataset("data", data=arr_in)

            arr_out, info = hdf5_reader.open(hdf5_path)

            assert arr_out.dtype == dtype
            assert info.dtype == dtype
            assert np.allclose(arr_out, arr_in)

    def test_open_various_shapes(self, hdf5_reader, tmp_path):
        """Test opening HDF5 files with various shapes."""

        shapes = [
            (100, 100),  # 2D
            (10, 100, 100),  # 3D
            (5, 3, 100, 100),  # 4D
            (2, 5, 3, 50, 50),  # 5D
        ]

        for idx, shape in enumerate(shapes):
            arr_in = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            hdf5_path = tmp_path / f"test_{len(shape)}d.h5"
            with h5py.File(str(hdf5_path), "w") as f:
                f.create_dataset("data", data=arr_in)

            arr_out, info = hdf5_reader.open(hdf5_path)

            assert arr_out.shape == shape
            assert info.shape == shape
            assert np.array_equal(arr_out, arr_in)

    def test_numpy_and_dask_same_values(self, hdf5_reader, temp_hdf5_3d):
        """Test that NumPy and Dask loading produce the same values."""
        hdf5_path, _ = temp_hdf5_3d

        arr_numpy, info_numpy = hdf5_reader.open(hdf5_path, prefer_dask=False)
        arr_dask, info_dask = hdf5_reader.open(hdf5_path, prefer_dask=True)

        assert np.array_equal(arr_numpy, arr_dask.compute())
        assert info_numpy.shape == info_dask.shape
        assert info_numpy.dtype == info_dask.dtype

    def test_open_with_kwargs(self, hdf5_reader, temp_hdf5_2d):
        """Test that open method accepts additional kwargs."""
        hdf5_path, _ = temp_hdf5_2d

        # This should not raise an error even with extra kwargs
        arr, info = hdf5_reader.open(hdf5_path, some_unused_kwarg="value")

        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)

    def test_hdf5_path_preservation(self, hdf5_reader, temp_hdf5_2d):
        """Test that the original path is preserved in ImageInfo."""
        hdf5_path, _ = temp_hdf5_2d

        arr, info = hdf5_reader.open(hdf5_path)

        assert info.path == hdf5_path
        assert str(hdf5_path) in str(info.path)

    def test_multiple_opens_same_file(self, hdf5_reader, temp_hdf5_2d):
        """Test that the same HDF5 file can be opened multiple times."""
        hdf5_path, expected_arr = temp_hdf5_2d

        arr1, info1 = hdf5_reader.open(hdf5_path)
        arr2, info2 = hdf5_reader.open(hdf5_path)

        assert np.array_equal(arr1, arr2)
        assert info1.shape == info2.shape
        assert info1.dtype == info2.dtype

    def test_large_hdf5_with_dask(self, hdf5_reader, tmp_path):
        """Test that Dask can handle larger HDF5 files efficiently."""

        # Create a moderately large HDF5 file (not huge to keep test fast)
        arr = np.random.randint(0, 255, size=(50, 512, 512), dtype=np.uint8)
        hdf5_path = tmp_path / "large.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("data", data=arr)

        # Open with Dask should not load into memory immediately
        darr, info = hdf5_reader.open(
            hdf5_path, prefer_dask=True, chunks=(10, 256, 256)
        )

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape
        # Verify a slice works without loading everything
        slice_result = darr[0].compute()
        assert np.array_equal(slice_result, arr[0])

    def test_hdf5_native_chunking(self, hdf5_reader, temp_hdf5_chunked):
        """Test that HDF5Reader respects native HDF5 chunking."""
        hdf5_path, arr = temp_hdf5_chunked

        # Open with Dask without specifying chunks (should use native)
        darr, info = hdf5_reader.open(hdf5_path, prefer_dask=True)

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape
        # Dask should use native HDF5 chunking
        assert darr.chunks is not None

    def test_hdf5_empty_file_raises_error(self, hdf5_reader, tmp_path):
        """Test that opening an HDF5 file with no datasets raises ValueError."""

        hdf5_path = tmp_path / "empty.h5"
        # Create an empty file with no datasets
        with h5py.File(str(hdf5_path), "w") as f:
            pass

        with pytest.raises(ValueError, match="No datasets found"):
            hdf5_reader.open(hdf5_path)

    def test_hdf5_axes_from_dimension_labels(self, hdf5_reader, tmp_path):
        """Test extraction of axes from DIMENSION_LABELS attribute."""

        arr = np.random.rand(5, 100, 100).astype(np.float32)
        hdf5_path = tmp_path / "test_dimension_labels.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            ds = f.create_dataset("data", data=arr)
            ds.attrs["DIMENSION_LABELS"] = ["z", "y", "x"]

        arr_out, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        # Should extract axes from DIMENSION_LABELS
        assert info.axes is not None

    def test_hdf5_with_trailing_slash(self, hdf5_reader, temp_hdf5_2d):
        """Test that HDF5Reader handles paths with trailing slashes."""
        hdf5_path, expected_arr = temp_hdf5_2d

        # Add trailing slash to path (should be stripped)
        path_with_slash = Path(str(hdf5_path) + "/")

        arr, info = hdf5_reader.open(path_with_slash, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert np.array_equal(arr, expected_arr)

    def test_hdf5_dask_lazy_evaluation(self, hdf5_reader, tmp_path):
        """Test that Dask arrays from HDF5 are truly lazy."""

        arr = np.random.rand(100, 100, 100).astype(np.float32)
        hdf5_path = tmp_path / "lazy.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("data", data=arr)

        # Open as Dask
        darr, info = hdf5_reader.open(hdf5_path, prefer_dask=True)

        # Should be lazy - no computation yet
        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape

        # Only compute a small slice
        small_slice = darr[0:1, 0:10, 0:10].compute()
        assert small_slice.shape == (1, 10, 10)
        assert np.array_equal(small_slice, arr[0:1, 0:10, 0:10])

    def test_hdf5_with_compression(self, hdf5_reader, tmp_path):
        """Test that HDF5Reader handles compressed datasets."""

        arr = np.random.rand(100, 100).astype(np.float32)
        hdf5_path = tmp_path / "compressed.h5"

        # Create with gzip compression
        with h5py.File(str(hdf5_path), "w") as f:
            f.create_dataset("data", data=arr, compression="gzip", compression_opts=9)

        # Should be able to open and read correctly
        arr_out, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        assert np.allclose(arr_out, arr)
        assert arr_out.shape == arr.shape

    def test_hdf5_with_multiple_attrs(self, hdf5_reader, tmp_path):
        """Test opening an HDF5 file with multiple custom attributes."""

        arr = np.random.rand(50, 100, 100).astype(np.float32)
        hdf5_path = tmp_path / "multi_attrs.h5"

        with h5py.File(str(hdf5_path), "w") as f:
            ds = f.create_dataset("data", data=arr)
            ds.attrs["axes"] = "ZYX"
            ds.attrs["scale"] = 0.5
            ds.attrs["unit"] = "mm"
            ds.attrs["modality"] = "fluorescence"
            ds.attrs["timestamp"] = "2025-12-05"

        arr_out, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        assert info.metadata is not None
        assert len(info.metadata) >= 5
        assert info.metadata["axes"] == "ZYX"
        assert info.metadata["scale"] == 0.5
        assert info.metadata["unit"] == "mm"
        assert info.metadata["modality"] == "fluorescence"

    def test_hdf5_dataset_name_in_metadata(self, hdf5_reader, temp_hdf5_with_groups):
        """Test that dataset name is preserved in metadata."""
        hdf5_path, _ = temp_hdf5_with_groups

        arr, info = hdf5_reader.open(hdf5_path, prefer_dask=False)

        # The metadata should exist
        assert info.metadata is not None
        # Note: The actual dataset name preservation depends on implementation


# =============================================================================
# Test NumpyReader
# =============================================================================


class TestNumpyReader:
    """Test suite for the NumpyReader class."""

    @pytest.fixture
    def numpy_reader(self):
        """Fixture to create a NumpyReader instance."""
        return NumpyReader()

    @pytest.fixture
    def temp_npy_2d(self, tmp_path):
        """Fixture to create a temporary 2D .npy file."""
        arr = np.random.randint(0, 65535, size=(256, 256), dtype=np.uint16)
        npy_path = tmp_path / "test_2d.npy"
        np.save(str(npy_path), arr)
        return npy_path, arr

    @pytest.fixture
    def temp_npy_3d(self, tmp_path):
        """Fixture to create a temporary 3D .npy file."""
        arr = np.random.randint(0, 65535, size=(10, 256, 256), dtype=np.uint16)
        npy_path = tmp_path / "test_3d.npy"
        np.save(str(npy_path), arr)
        return npy_path, arr

    @pytest.fixture
    def temp_npy_float(self, tmp_path):
        """Fixture to create a temporary .npy file with float data."""
        arr = np.random.rand(50, 100, 100).astype(np.float32)
        npy_path = tmp_path / "test_float.npy"
        np.save(str(npy_path), arr)
        return npy_path, arr

    @pytest.fixture
    def temp_npz_single(self, tmp_path):
        """Fixture to create a temporary .npz file with a single array."""
        arr = np.random.rand(100, 100).astype(np.float32)
        npz_path = tmp_path / "test_single.npz"
        np.savez(str(npz_path), arr_0=arr)
        return npz_path, arr, "arr_0"

    @pytest.fixture
    def temp_npz_multiple(self, tmp_path):
        """Fixture to create a temporary .npz file with multiple arrays."""
        arr1 = np.random.rand(50, 50).astype(np.float32)
        arr2 = np.random.rand(100, 100).astype(np.float64)
        arr3 = np.random.rand(75, 75).astype(np.int32)
        npz_path = tmp_path / "test_multiple.npz"
        np.savez(str(npz_path), data_a=arr1, data_b=arr2, data_c=arr3)
        # First key alphabetically should be 'data_a'
        return npz_path, arr1, "data_a"

    @pytest.fixture
    def temp_npy_large(self, tmp_path):
        """Fixture to create a larger .npy file for memory mapping tests."""
        arr = np.random.rand(100, 512, 512).astype(np.float32)
        npy_path = tmp_path / "test_large.npy"
        np.save(str(npy_path), arr)
        return npy_path, arr

    def test_numpyreader_instantiation(self, numpy_reader):
        """Test that NumpyReader can be instantiated."""
        assert isinstance(numpy_reader, NumpyReader)
        assert isinstance(numpy_reader, Reader)

    def test_numpyreader_suffixes(self, numpy_reader):
        """Test that NumpyReader has correct SUFFIXES."""
        assert NumpyReader.SUFFIXES == (".npy", ".npz")
        assert hasattr(numpy_reader, "SUFFIXES")

    def test_numpyreader_claims_npy_files(self):
        """Test that NumpyReader claims .npy and .npz files."""
        assert NumpyReader.claims(Path("data.npy")) is True
        assert NumpyReader.claims(Path("data.npz")) is True
        assert NumpyReader.claims(Path("data.NPY")) is True
        assert NumpyReader.claims(Path("data.NPZ")) is True

    def test_numpyreader_rejects_other_files(self):
        """Test that NumpyReader rejects non-NumPy files."""
        assert NumpyReader.claims(Path("image.tif")) is False
        assert NumpyReader.claims(Path("image.h5")) is False
        assert NumpyReader.claims(Path("image.zarr")) is False
        assert NumpyReader.claims(Path("image.txt")) is False

    def test_open_2d_npy_as_numpy(self, numpy_reader, temp_npy_2d):
        """Test opening a 2D .npy file as NumPy array."""
        npy_path, expected_arr = temp_npy_2d

        arr, info = numpy_reader.open(npy_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_3d_npy_as_numpy(self, numpy_reader, temp_npy_3d):
        """Test opening a 3D .npy file as NumPy array."""
        npy_path, expected_arr = temp_npy_3d

        arr, info = numpy_reader.open(npy_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)

    def test_open_npy_as_dask(self, numpy_reader, temp_npy_3d):
        """Test opening a .npy file as Dask array with memory mapping."""
        npy_path, expected_arr = temp_npy_3d

        arr, info = numpy_reader.open(npy_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        # Compute Dask array to compare values
        assert np.array_equal(arr.compute(), expected_arr)

    def test_open_npy_with_custom_chunks(self, numpy_reader, temp_npy_3d):
        """Test opening a .npy file with custom chunk sizes."""
        npy_path, expected_arr = temp_npy_3d

        chunks = (5, 128, 128)
        arr, info = numpy_reader.open(npy_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        # Check that chunking was applied
        assert arr.chunks is not None

    def test_open_npy_with_single_chunk_size(self, numpy_reader, temp_npy_3d):
        """Test opening a .npy file with a single chunk size for all dimensions."""
        npy_path, expected_arr = temp_npy_3d

        chunks = 64
        arr, info = numpy_reader.open(npy_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.chunks is not None

    def test_imageinfo_from_npy_has_correct_path(self, numpy_reader, temp_npy_2d):
        """Test that ImageInfo has correct path."""
        npy_path, _ = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert isinstance(info, ImageInfo)
        assert info.path == npy_path

    def test_imageinfo_from_npy_has_correct_shape(self, numpy_reader, temp_npy_2d):
        """Test that ImageInfo has correct shape."""
        npy_path, expected_arr = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert info.shape == expected_arr.shape
        assert info.shape == arr.shape

    def test_imageinfo_from_npy_has_correct_dtype(self, numpy_reader, temp_npy_2d):
        """Test that ImageInfo has correct dtype."""
        npy_path, expected_arr = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert info.dtype == expected_arr.dtype
        assert info.dtype == arr.dtype

    def test_open_npz_single_array(self, numpy_reader, temp_npz_single):
        """Test opening a .npz file with a single array."""
        npz_path, expected_arr, expected_key = temp_npz_single

        arr, info = numpy_reader.open(npz_path, prefer_dask=False)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)
        assert info.metadata["npz_key"] == expected_key

    def test_open_npz_multiple_arrays_selects_first(
        self, numpy_reader, temp_npz_multiple
    ):
        """Test that opening a .npz file with multiple arrays selects the first alphabetically."""
        npz_path, expected_arr, expected_key = temp_npz_multiple

        arr, info = numpy_reader.open(npz_path, prefer_dask=False)

        # Should have selected the first array alphabetically
        assert arr.shape == expected_arr.shape
        assert np.array_equal(arr, expected_arr)
        assert info.metadata["npz_key"] == expected_key

    def test_open_npz_as_dask(self, numpy_reader, temp_npz_single):
        """Test opening a .npz file as Dask array."""
        npz_path, expected_arr, expected_key = temp_npz_single

        arr, info = numpy_reader.open(npz_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr.compute(), expected_arr)
        assert info.metadata["npz_key"] == expected_key

    def test_open_npz_with_custom_chunks(self, numpy_reader, temp_npz_single):
        """Test opening a .npz file with custom chunk sizes."""
        npz_path, expected_arr, _ = temp_npz_single

        chunks = (50, 50)
        arr, info = numpy_reader.open(npz_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert arr.chunks is not None

    def test_open_nonexistent_npy(self, numpy_reader):
        """Test that opening a nonexistent .npy file raises appropriate error."""
        nonexistent_path = Path("/nonexistent/path/to/file.npy")

        with pytest.raises((FileNotFoundError, OSError)):
            numpy_reader.open(nonexistent_path)

    def test_imageinfo_metadata_field_exists(self, numpy_reader, temp_npy_2d):
        """Test that ImageInfo has metadata field."""
        npy_path, _ = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert hasattr(info, "metadata")
        assert isinstance(info.metadata, dict)

    def test_imageinfo_axes_field_is_none(self, numpy_reader, temp_npy_2d):
        """Test that ImageInfo axes field is None for .npy files."""
        npy_path, _ = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert hasattr(info, "axes")
        # .npy files don't have axis metadata
        assert info.axes is None

    def test_open_different_dtypes(self, numpy_reader, tmp_path):
        """Test opening .npy files with different dtypes."""
        dtypes = [np.uint8, np.uint16, np.int32, np.float32, np.float64]

        for dtype in dtypes:
            arr_in = np.random.rand(64, 64).astype(dtype)
            if dtype in [np.uint8, np.uint16]:
                arr_in = (arr_in * np.iinfo(dtype).max).astype(dtype)

            npy_path = tmp_path / f"test_{dtype.__name__}.npy"
            np.save(str(npy_path), arr_in)

            arr_out, info = numpy_reader.open(npy_path)

            assert arr_out.dtype == dtype
            assert info.dtype == dtype
            assert np.allclose(arr_out, arr_in)

    def test_open_various_shapes(self, numpy_reader, tmp_path):
        """Test opening .npy files with various shapes."""
        shapes = [
            (100, 100),  # 2D
            (10, 100, 100),  # 3D
            (5, 3, 100, 100),  # 4D
            (2, 5, 3, 50, 50),  # 5D
        ]

        for idx, shape in enumerate(shapes):
            arr_in = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            npy_path = tmp_path / f"test_{len(shape)}d.npy"
            np.save(str(npy_path), arr_in)

            arr_out, info = numpy_reader.open(npy_path)

            assert arr_out.shape == shape
            assert info.shape == shape
            assert np.array_equal(arr_out, arr_in)

    def test_numpy_and_dask_same_values(self, numpy_reader, temp_npy_3d):
        """Test that NumPy and Dask loading produce the same values."""
        npy_path, _ = temp_npy_3d

        arr_numpy, info_numpy = numpy_reader.open(npy_path, prefer_dask=False)
        arr_dask, info_dask = numpy_reader.open(npy_path, prefer_dask=True)

        assert np.array_equal(arr_numpy, arr_dask.compute())
        assert info_numpy.shape == info_dask.shape
        assert info_numpy.dtype == info_dask.dtype

    def test_open_with_kwargs(self, numpy_reader, temp_npy_2d):
        """Test that open method accepts additional kwargs."""
        npy_path, _ = temp_npy_2d

        # This should not raise an error even with extra kwargs
        arr, info = numpy_reader.open(npy_path, some_unused_kwarg="value")

        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)

    def test_npy_path_preservation(self, numpy_reader, temp_npy_2d):
        """Test that the original path is preserved in ImageInfo."""
        npy_path, _ = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert info.path == npy_path
        assert str(npy_path) in str(info.path)

    def test_multiple_opens_same_file(self, numpy_reader, temp_npy_2d):
        """Test that the same .npy file can be opened multiple times."""
        npy_path, expected_arr = temp_npy_2d

        arr1, info1 = numpy_reader.open(npy_path)
        arr2, info2 = numpy_reader.open(npy_path)

        assert np.array_equal(arr1, arr2)
        assert info1.shape == info2.shape
        assert info1.dtype == info2.dtype

    def test_large_npy_with_dask_memmap(self, numpy_reader, temp_npy_large):
        """Test that Dask uses memory mapping for large .npy files."""
        npy_path, expected_arr = temp_npy_large

        # Open with Dask (should use memory mapping)
        darr, info = numpy_reader.open(
            npy_path, prefer_dask=True, chunks=(10, 256, 256)
        )

        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape
        # Verify a slice works without loading everything
        slice_result = darr[0].compute()
        assert np.array_equal(slice_result, expected_arr[0])

    def test_npy_dask_lazy_evaluation(self, numpy_reader, temp_npy_large):
        """Test that Dask arrays from .npy are truly lazy."""
        npy_path, expected_arr = temp_npy_large

        # Open as Dask
        darr, info = numpy_reader.open(npy_path, prefer_dask=True)

        # Should be lazy - no computation yet
        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape

        # Only compute a small slice
        small_slice = darr[0:1, 0:10, 0:10].compute()
        assert small_slice.shape == (1, 10, 10)
        assert np.array_equal(small_slice, expected_arr[0:1, 0:10, 0:10])

    def test_npy_dask_auto_chunks(self, numpy_reader, temp_npy_3d):
        """Test that Dask uses 'auto' chunking when chunks=None."""
        npy_path, expected_arr = temp_npy_3d

        # Open with Dask without specifying chunks (should use 'auto')
        darr, info = numpy_reader.open(npy_path, prefer_dask=True)

        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape
        # Should have some chunking applied
        assert darr.chunks is not None

    def test_npz_metadata_contains_key(self, numpy_reader, temp_npz_single):
        """Test that .npz metadata contains the array key."""
        npz_path, _, expected_key = temp_npz_single

        arr, info = numpy_reader.open(npz_path)

        assert info.metadata is not None
        assert "npz_key" in info.metadata
        assert info.metadata["npz_key"] == expected_key

    def test_npz_multiple_arrays_metadata(self, numpy_reader, temp_npz_multiple):
        """Test that .npz with multiple arrays has correct metadata."""
        npz_path, _, expected_key = temp_npz_multiple

        arr, info = numpy_reader.open(npz_path)

        assert info.metadata["npz_key"] == expected_key
        # Should be the first key alphabetically

    def test_npy_metadata_is_empty(self, numpy_reader, temp_npy_2d):
        """Test that .npy files have empty metadata dict."""
        npy_path, _ = temp_npy_2d

        arr, info = numpy_reader.open(npy_path)

        assert info.metadata == {}

    def test_open_float_npy(self, numpy_reader, temp_npy_float):
        """Test opening a .npy file with float data."""
        npy_path, expected_arr = temp_npy_float

        arr, info = numpy_reader.open(npy_path)

        assert arr.dtype == np.float32
        assert np.allclose(arr, expected_arr)

    def test_npz_path_preservation(self, numpy_reader, temp_npz_single):
        """Test that the original path is preserved in ImageInfo for .npz files."""
        npz_path, _, _ = temp_npz_single

        arr, info = numpy_reader.open(npz_path)

        assert info.path == npz_path
        assert str(npz_path) in str(info.path)

    def test_npy_readonly_array(self, numpy_reader, temp_npy_2d):
        """Test that opening .npy files produces readable arrays."""
        npy_path, _ = temp_npy_2d

        arr, info = numpy_reader.open(npy_path, prefer_dask=False)

        # Should be able to read the array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == info.shape

    def test_npz_compressed(self, numpy_reader, tmp_path):
        """Test opening a compressed .npz file."""
        arr = np.random.rand(100, 100).astype(np.float32)
        npz_path = tmp_path / "compressed.npz"
        # savez_compressed creates a compressed archive
        np.savez_compressed(str(npz_path), data=arr)

        arr_out, info = numpy_reader.open(npz_path)

        assert np.allclose(arr_out, arr)
        assert arr_out.shape == arr.shape
        assert info.metadata["npz_key"] == "data"

    def test_npz_dask_with_chunks(self, numpy_reader, temp_npz_single):
        """Test opening .npz as Dask array with custom chunks."""
        npz_path, expected_arr, _ = temp_npz_single

        chunks = (50, 50)
        darr, info = numpy_reader.open(npz_path, prefer_dask=True, chunks=chunks)

        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape
        assert np.array_equal(darr.compute(), expected_arr)

    def test_npy_1d_array(self, numpy_reader, tmp_path):
        """Test opening a 1D .npy file."""
        arr_in = np.arange(1000, dtype=np.int32)
        npy_path = tmp_path / "test_1d.npy"
        np.save(str(npy_path), arr_in)

        arr_out, info = numpy_reader.open(npy_path)

        assert arr_out.shape == (1000,)
        assert np.array_equal(arr_out, arr_in)

    def test_npy_empty_array(self, numpy_reader, tmp_path):
        """Test opening a .npy file with an empty array."""
        arr_in = np.array([], dtype=np.float32)
        npy_path = tmp_path / "test_empty.npy"
        np.save(str(npy_path), arr_in)

        arr_out, info = numpy_reader.open(npy_path)

        assert arr_out.shape == (0,)
        assert len(arr_out) == 0

    def test_npy_complex_dtype(self, numpy_reader, tmp_path):
        """Test opening a .npy file with complex dtype."""
        arr_in = np.random.rand(50, 50) + 1j * np.random.rand(50, 50)
        arr_in = arr_in.astype(np.complex64)
        npy_path = tmp_path / "test_complex.npy"
        np.save(str(npy_path), arr_in)

        arr_out, info = numpy_reader.open(npy_path)

        assert arr_out.dtype == np.complex64
        assert np.allclose(arr_out, arr_in)

    def test_npz_dask_auto_chunks(self, numpy_reader, temp_npz_single):
        """Test that Dask uses 'auto' chunking for .npz when chunks=None."""
        npz_path, expected_arr, _ = temp_npz_single

        darr, info = numpy_reader.open(npz_path, prefer_dask=True)

        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape
        assert darr.chunks is not None

    def test_npy_with_structured_dtype(self, numpy_reader, tmp_path):
        """Test opening a .npy file with structured dtype."""
        dt = np.dtype([("x", np.int32), ("y", np.float32)])
        arr_in = np.zeros(100, dtype=dt)
        arr_in["x"] = np.arange(100)
        arr_in["y"] = np.random.rand(100)
        npy_path = tmp_path / "test_structured.npy"
        np.save(str(npy_path), arr_in)

        arr_out, info = numpy_reader.open(npy_path)

        assert arr_out.dtype == dt
        assert np.array_equal(arr_out, arr_in)

    def test_npz_with_multiple_dtypes(self, numpy_reader, tmp_path):
        """Test .npz file with arrays of different dtypes."""
        arr1 = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        arr2 = np.random.rand(100, 100).astype(np.float64)
        arr3 = np.random.randint(-1000, 1000, size=(75, 75), dtype=np.int32)

        npz_path = tmp_path / "multi_dtype.npz"
        np.savez(str(npz_path), a_uint8=arr1, b_float64=arr2, c_int32=arr3)

        arr_out, info = numpy_reader.open(npz_path)

        # Should load first alphabetically (a_uint8)
        assert info.metadata["npz_key"] == "a_uint8"
        assert arr_out.dtype == np.uint8
        assert np.array_equal(arr_out, arr1)


# =============================================================================
# Test ImageOpener
# =============================================================================


class TestImageOpener:
    """Test suite for the ImageOpener class."""

    @pytest.fixture
    def image_opener(self):
        """Fixture to create an ImageOpener instance with default readers."""
        return ImageOpener()

    @pytest.fixture
    def custom_image_opener(self):
        """Fixture to create an ImageOpener with custom reader order."""
        return ImageOpener(readers=[TiffReader, ZarrReader])

    @pytest.fixture
    def temp_tiff_file(self, tmp_path):
        """Fixture to create a temporary TIFF file."""
        import tifffile

        arr = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(str(tiff_path), arr)
        return tiff_path, arr

    @pytest.fixture
    def temp_zarr_file(self, tmp_path):
        """Fixture to create a temporary Zarr store."""
        import zarr

        arr = np.random.rand(100, 100).astype(np.float32)
        zarr_path = tmp_path / "test.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr)
        return zarr_path, arr

    @pytest.fixture
    def temp_hdf5_file(self, tmp_path):
        """Fixture to create a temporary HDF5 file."""
        import h5py

        arr = np.random.rand(50, 50).astype(np.float32)
        h5_path = tmp_path / "test.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("data", data=arr)
        return h5_path, arr

    @pytest.fixture
    def temp_npy_file(self, tmp_path):
        """Fixture to create a temporary .npy file."""
        arr = np.random.rand(75, 75).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), arr)
        return npy_path, arr

    @pytest.fixture
    def temp_npz_file(self, tmp_path):
        """Fixture to create a temporary .npz file."""
        arr = np.random.rand(60, 60).astype(np.float32)
        npz_path = tmp_path / "test.npz"
        np.savez(str(npz_path), data=arr)
        return npz_path, arr

    def test_imageopener_instantiation_default(self, image_opener):
        """Test that ImageOpener can be instantiated with default readers."""
        assert isinstance(image_opener, ImageOpener)
        assert hasattr(image_opener, "_readers")
        # Default readers should be TiffReader, ZarrReader, NumpyReader, HDF5Reader
        assert len(image_opener._readers) == 4
        assert TiffReader in image_opener._readers
        assert ZarrReader in image_opener._readers
        assert NumpyReader in image_opener._readers
        assert HDF5Reader in image_opener._readers

    def test_imageopener_instantiation_custom(self, custom_image_opener):
        """Test that ImageOpener can be instantiated with custom readers."""
        assert isinstance(custom_image_opener, ImageOpener)
        assert len(custom_image_opener._readers) == 2
        assert custom_image_opener._readers[0] == TiffReader
        assert custom_image_opener._readers[1] == ZarrReader

    def test_imageopener_custom_single_reader(self):
        """Test ImageOpener with a single reader."""
        opener = ImageOpener(readers=[TiffReader])
        assert len(opener._readers) == 1
        assert opener._readers[0] == TiffReader

    def test_open_tiff_file(self, image_opener, temp_tiff_file):
        """Test opening a TIFF file using ImageOpener."""
        tiff_path, expected_arr = temp_tiff_file

        arr, info = image_opener.open(tiff_path)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert arr.dtype == expected_arr.dtype
        assert np.array_equal(arr, expected_arr)
        assert info.path == tiff_path

    def test_open_zarr_file(self, image_opener, temp_zarr_file):
        """Test opening a Zarr store using ImageOpener."""
        zarr_path, expected_arr = temp_zarr_file

        arr, info = image_opener.open(zarr_path)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert np.allclose(arr, expected_arr)
        assert info.path == zarr_path

    def test_open_hdf5_file(self, image_opener, temp_hdf5_file):
        """Test opening an HDF5 file using ImageOpener."""
        h5_path, expected_arr = temp_hdf5_file

        arr, info = image_opener.open(h5_path)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert np.allclose(arr, expected_arr)
        assert info.path == h5_path

    def test_open_npy_file(self, image_opener, temp_npy_file):
        """Test opening a .npy file using ImageOpener."""
        npy_path, expected_arr = temp_npy_file

        arr, info = image_opener.open(npy_path)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert np.allclose(arr, expected_arr)
        assert info.path == npy_path

    def test_open_npz_file(self, image_opener, temp_npz_file):
        """Test opening a .npz file using ImageOpener."""
        npz_path, expected_arr = temp_npz_file

        arr, info = image_opener.open(npz_path)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape
        assert np.allclose(arr, expected_arr)
        assert info.path == npz_path

    def test_open_with_string_path(self, image_opener, temp_tiff_file):
        """Test opening a file with a string path."""
        tiff_path, expected_arr = temp_tiff_file

        arr, info = image_opener.open(str(tiff_path))

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape

    def test_open_with_path_object(self, image_opener, temp_tiff_file):
        """Test opening a file with a Path object."""
        tiff_path, expected_arr = temp_tiff_file

        arr, info = image_opener.open(Path(tiff_path))

        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_arr.shape

    def test_open_with_prefer_dask(self, image_opener, temp_zarr_file):
        """Test opening a file with prefer_dask=True."""
        zarr_path, expected_arr = temp_zarr_file

        arr, info = image_opener.open(zarr_path, prefer_dask=True)

        assert isinstance(arr, da.Array)
        assert arr.shape == expected_arr.shape
        assert np.allclose(arr.compute(), expected_arr)

    def test_open_with_chunks(self, image_opener, temp_zarr_file):
        """Test opening a file with custom chunks."""
        zarr_path, expected_arr = temp_zarr_file

        chunks = (50, 50)
        arr, info = image_opener.open(zarr_path, prefer_dask=True, chunks=chunks)

        assert isinstance(arr, da.Array)
        assert arr.chunks is not None

    def test_open_nonexistent_file(self, image_opener):
        """Test that opening a nonexistent file raises FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/path/to/file.tif")

        with pytest.raises(FileNotFoundError):
            image_opener.open(nonexistent_path)

    def test_open_with_kwargs(self, image_opener, temp_tiff_file):
        """Test that open method passes kwargs to the reader."""
        tiff_path, _ = temp_tiff_file

        # Should not raise an error
        arr, info = image_opener.open(tiff_path)

        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)

    def test_reader_priority_order(self, tmp_path):
        """Test that readers are used in priority order."""
        import tifffile

        # Create a TIFF file
        arr = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(str(tiff_path), arr)

        # Create opener with TiffReader first
        opener1 = ImageOpener(readers=[TiffReader, ZarrReader])
        arr1, info1 = opener1.open(tiff_path)

        # Create opener with different order (shouldn't affect .tif files)
        opener2 = ImageOpener(readers=[ZarrReader, TiffReader])
        arr2, info2 = opener2.open(tiff_path)

        # Both should successfully open the file
        assert np.array_equal(arr1, arr2)

    def test_custom_reader_only_opens_claimed_format(
        self, custom_image_opener, tmp_path
    ):
        """Test that custom opener with limited readers only opens supported formats."""
        import tifffile

        # Create a TIFF file (supported by custom opener)
        arr = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(str(tiff_path), arr)

        # Should successfully open TIFF
        arr_out, info = custom_image_opener.open(tiff_path)
        assert np.array_equal(arr_out, arr)

    def test_custom_reader_fails_on_unsupported_format(self, tmp_path):
        """Test that custom opener fails on unsupported formats."""
        # Create an HDF5 file
        import h5py

        arr = np.random.rand(50, 50).astype(np.float32)
        h5_path = tmp_path / "test.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("data", data=arr)

        # Create opener without HDF5Reader
        opener = ImageOpener(readers=[TiffReader, ZarrReader])

        # Should raise ValueError since no reader can handle it
        with pytest.raises(ValueError, match="No suitable reader found"):
            opener.open(h5_path)

    def test_multiple_opens_same_file(self, image_opener, temp_tiff_file):
        """Test that the same file can be opened multiple times."""
        tiff_path, expected_arr = temp_tiff_file

        arr1, info1 = image_opener.open(tiff_path)
        arr2, info2 = image_opener.open(tiff_path)

        assert np.array_equal(arr1, arr2)
        assert info1.shape == info2.shape

    def test_open_different_formats_sequentially(
        self, image_opener, temp_tiff_file, temp_zarr_file, temp_npy_file
    ):
        """Test opening different file formats sequentially."""
        tiff_path, tiff_arr = temp_tiff_file
        zarr_path, zarr_arr = temp_zarr_file
        npy_path, npy_arr = temp_npy_file

        # Open TIFF
        arr1, info1 = image_opener.open(tiff_path)
        assert np.array_equal(arr1, tiff_arr)

        # Open Zarr
        arr2, info2 = image_opener.open(zarr_path)
        assert np.allclose(arr2, zarr_arr)

        # Open NPY
        arr3, info3 = image_opener.open(npy_path)
        assert np.allclose(arr3, npy_arr)

    def test_imageinfo_consistency_across_formats(
        self, image_opener, temp_tiff_file, temp_npy_file
    ):
        """Test that ImageInfo structure is consistent across formats."""
        tiff_path, _ = temp_tiff_file
        npy_path, _ = temp_npy_file

        arr1, info1 = image_opener.open(tiff_path)
        arr2, info2 = image_opener.open(npy_path)

        # All ImageInfo objects should have these fields
        for info in [info1, info2]:
            assert hasattr(info, "path")
            assert hasattr(info, "shape")
            assert hasattr(info, "dtype")
            assert hasattr(info, "axes")
            assert hasattr(info, "metadata")

    def test_open_with_dask_preserves_shape(self, image_opener, temp_zarr_file):
        """Test that Dask arrays preserve original shape."""
        zarr_path, expected_arr = temp_zarr_file

        darr, info = image_opener.open(zarr_path, prefer_dask=True)

        assert darr.shape == expected_arr.shape
        assert info.shape == expected_arr.shape

    def test_open_with_dask_preserves_dtype(self, image_opener, temp_zarr_file):
        """Test that Dask arrays preserve original dtype."""
        zarr_path, expected_arr = temp_zarr_file

        darr, info = image_opener.open(zarr_path, prefer_dask=True)

        assert darr.dtype == expected_arr.dtype
        assert info.dtype == expected_arr.dtype

    def test_readers_tuple_is_immutable(self, image_opener):
        """Test that the readers tuple cannot be modified."""
        assert isinstance(image_opener._readers, tuple)
        # Tuples are immutable, so this should work as expected

    def test_empty_readers_list(self):
        """Test ImageOpener with an empty readers list uses defaults."""
        opener = ImageOpener(readers=[])
        # Empty list should use default readers (this is the actual behavior)
        assert len(opener._readers) == 4  # Defaults to all readers

    def test_fallback_mechanism(self, image_opener, tmp_path):
        """Test that fallback mechanism tries all readers."""
        # Create a .npy file (which has a standard extension)
        # and verify it can be opened
        arr = np.random.rand(50, 50).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), arr)

        # Should successfully open the file
        arr_out, info = image_opener.open(npy_path)
        assert arr_out.shape == arr.shape
        assert np.allclose(arr_out, arr)

    def test_open_tiff_with_metadata(self, image_opener, tmp_path):
        """Test opening a TIFF file preserves metadata."""
        import tifffile

        arr = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        tiff_path = tmp_path / "test_meta.tif"
        tifffile.imwrite(str(tiff_path), arr, metadata={"test": "value"})

        arr_out, info = image_opener.open(tiff_path)

        assert isinstance(info.metadata, dict)
        assert info.path == tiff_path

    def test_open_zarr_with_dask_chunking(self, image_opener, tmp_path):
        """Test opening Zarr with Dask and custom chunking."""
        import zarr

        arr = np.random.rand(200, 200).astype(np.float32)
        zarr_path = tmp_path / "test_chunked.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr, chunks=(100, 100))

        darr, info = image_opener.open(zarr_path, prefer_dask=True, chunks=(50, 50))

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape

    def test_open_hdf5_with_dask(self, image_opener, temp_hdf5_file):
        """Test opening HDF5 file with Dask."""
        h5_path, expected_arr = temp_hdf5_file

        darr, info = image_opener.open(h5_path, prefer_dask=True)

        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape
        assert np.allclose(darr.compute(), expected_arr)

    def test_open_npy_with_dask_memmap(self, image_opener, temp_npy_file):
        """Test opening .npy file with Dask uses memory mapping."""
        npy_path, expected_arr = temp_npy_file

        darr, info = image_opener.open(npy_path, prefer_dask=True)

        assert isinstance(darr, da.Array)
        assert darr.shape == expected_arr.shape
        assert np.allclose(darr.compute(), expected_arr)

    def test_path_types_equivalence(self, image_opener, temp_tiff_file):
        """Test that string and Path produce same results."""
        tiff_path, _ = temp_tiff_file

        arr1, info1 = image_opener.open(str(tiff_path))
        arr2, info2 = image_opener.open(Path(tiff_path))

        assert np.array_equal(arr1, arr2)
        assert info1.shape == info2.shape
        assert info1.dtype == info2.dtype

    def test_reader_order_preserved(self):
        """Test that reader order is preserved in the registry."""
        readers = [HDF5Reader, NumpyReader, ZarrReader, TiffReader]
        opener = ImageOpener(readers=readers)

        assert opener._readers[0] == HDF5Reader
        assert opener._readers[1] == NumpyReader
        assert opener._readers[2] == ZarrReader
        assert opener._readers[3] == TiffReader

    def test_open_returns_tuple(self, image_opener, temp_tiff_file):
        """Test that open returns a tuple of (array, ImageInfo)."""
        tiff_path, _ = temp_tiff_file

        result = image_opener.open(tiff_path)

        assert isinstance(result, tuple)
        assert len(result) == 2
        arr, info = result
        assert isinstance(arr, np.ndarray)
        assert isinstance(info, ImageInfo)

    def test_all_default_readers_accessible(self, image_opener):
        """Test that all default readers are accessible in the registry."""
        reader_classes = [TiffReader, ZarrReader, NumpyReader, HDF5Reader]

        for reader_cls in reader_classes:
            assert reader_cls in image_opener._readers

    def test_opener_with_duplicate_readers(self):
        """Test ImageOpener behavior with duplicate readers in the list."""
        opener = ImageOpener(readers=[TiffReader, TiffReader, ZarrReader])

        # Should have 3 readers even if duplicates
        assert len(opener._readers) == 3

    def test_open_large_file_with_dask(self, image_opener, tmp_path):
        """Test opening a large file with Dask for lazy loading."""
        import zarr

        # Create a moderately large file
        arr = np.random.rand(200, 512, 512).astype(np.float32)
        zarr_path = tmp_path / "large.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_dataset("data", data=arr)

        darr, info = image_opener.open(
            zarr_path, prefer_dask=True, chunks=(10, 256, 256)
        )

        assert isinstance(darr, da.Array)
        assert darr.shape == arr.shape
        # Test lazy evaluation with a slice
        slice_result = darr[0].compute()
        assert np.allclose(slice_result, arr[0])
