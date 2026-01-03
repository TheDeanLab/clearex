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
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
import logging

# Third Party Imports
import numpy as np
import dask.array as da
import tifffile
import zarr
import h5py
from numpy.typing import NDArray

# Try to import ome-types for OME-TIFF metadata parsing
try:
    from ome_types import from_xml
    HAS_OME_TYPES = True
except ImportError:
    HAS_OME_TYPES = False

# Try to import nd2 for ND2 file support
try:
    import nd2
    HAS_ND2 = True
except ImportError:
    HAS_ND2 = False

# Local Imports

ArrayLike = Union[NDArray[Any], da.Array]

# Start logging
logger: logging.Logger = logging.getLogger(name=__name__)
logger.addHandler(hdlr=logging.NullHandler())


def _ensure_tuple(value: Any) -> Optional[Tuple[float, ...]]:
    """Convert various types to tuple of floats for pixel_size.
    
    Parameters
    ----------
    value : Any
        Value to convert (list, tuple, ndarray, or single value)
        
    Returns
    -------
    Optional[Tuple[float, ...]]
        Tuple of floats if conversion successful, None otherwise
    """
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            return tuple(float(x) for x in value)
        elif isinstance(value, np.ndarray):
            return tuple(float(x) for x in value.flat)
        else:
            # Single value
            return (float(value),)
    except (ValueError, TypeError):
        return None


def _normalize_axes(axes: Any) -> Optional[List[str]]:
    """Normalize axis information to OME-NGFF compatible list format.
    
    Converts various axis representations to a standardized list format
    compatible with OME-NGFF (up to 5 dimensions: t, c, z, y, x).
    
    Parameters
    ----------
    axes : Any
        Axis information in various formats:
        - String like "TCZYX" or "ZYX"
        - List of axis names like ["t", "c", "z", "y", "x"]
        - List of OME-NGFF axis dicts like [{"name": "t", "type": "time"}, ...]
        - Dict with keys like {'T': 10, 'C': 3, 'Z': 5, 'Y': 512, 'X': 512}
        
    Returns
    -------
    Optional[list]
        List of lowercase axis names (e.g., ["t", "c", "z", "y", "x"]) or None
        
    Notes
    -----
    Following OME-NGFF conventions:
    - Supports up to 5 dimensions
    - Common axis names: 't' (time), 'c' (channel), 'z', 'y', 'x' (spatial)
    - Preserves order from input
    - Normalizes to lowercase for consistency
    """
    if axes is None:
        return None
    
    try:
        # Handle string format (e.g., "TCZYX")
        if isinstance(axes, str):
            return [ax.lower() for ax in axes] if axes else None
        
        # Handle list of dicts (OME-NGFF format)
        if isinstance(axes, list):
            if not axes:  # Empty list
                return None
            if all(isinstance(item, dict) and "name" in item for item in axes):
                return [item["name"].lower() for item in axes]
            # Handle simple list of strings
            elif all(isinstance(item, str) for item in axes):
                return [ax.lower() for ax in axes]
        
        # Handle dict format (e.g., sizes dict from ND2)
        if isinstance(axes, dict):
            return [key.lower() for key in axes.keys()] if axes else None
            
    except Exception:
        pass
    
    return None


@dataclass
class ImageInfo:
    """Container for image metadata.

    This dataclass stores metadata about an opened image file, including
    its path, shape, data type, axis labels, pixel size, and any additional
    metadata extracted from the file format.

    Attributes
    ----------
    path : Path
        The file path to the image.
    shape : Tuple[int, ...]
        The shape of the image array (e.g., (height, width) for 2D,
        (depth, height, width) for 3D, or arbitrary N-dimensional shapes).
    dtype : Any
        The NumPy dtype of the image data (e.g., np.uint8, np.float32).
    axes : list of str, optional
        List of axis names describing dimension order, following OME-NGFF
        conventions. Supports up to 5 dimensions with common names:
        - 't' or 'time' for time dimension
        - 'c' or 'channel' for channel dimension
        - 'z' for Z spatial dimension
        - 'y' for Y spatial dimension
        - 'x' for X spatial dimension
        
        Examples: ["t", "c", "z", "y", "x"], ["z", "y", "x"], ["y", "x"]
        Order matches the actual array dimension order.
        Defaults to None if not available.
    pixel_size : Tuple[float, ...], optional
        Physical pixel/voxel sizes in micrometers, ordered to match the
        spatial axes in the axes list. For example, with axes=["z", "y", "x"]
        and pixel_size=(2.0, 0.65, 0.65), the Z spacing is 2.0 µm and
        XY resolution is 0.65 µm. Defaults to None if not available.
    metadata : Dict[str, Any], optional
        Additional metadata extracted from the file format, such as
        attributes from Zarr/HDF5 or custom tags. Defaults to None.

    Examples
    --------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> info = ImageInfo(
    ...     path=Path("image.tif"),
    ...     shape=(10, 512, 512),
    ...     dtype=np.uint16,
    ...     axes=["z", "y", "x"],
    ...     pixel_size=(2.0, 0.65, 0.65),
    ...     metadata={"scale": 1.0}
    ... )
    >>> print(info.shape)
    (10, 512, 512)
    >>> print(info.axes)
    ['z', 'y', 'x']
    >>> print(info.pixel_size)
    (2.0, 0.65, 0.65)
    
    >>> # 5D example
    >>> info_5d = ImageInfo(
    ...     path=Path("timeseries.nd2"),
    ...     shape=(20, 3, 10, 512, 512),
    ...     dtype=np.uint16,
    ...     axes=["t", "c", "z", "y", "x"],
    ...     pixel_size=(2.0, 0.65, 0.65),  # Z, Y, X only
    ... )
    >>> print(info_5d.axes)
    ['t', 'c', 'z', 'y', 'x']
    """

    path: Path
    shape: Tuple[int, ...]
    dtype: Any
    axes: Optional[List[str]] = None
    pixel_size: Optional[Tuple[float, ...]] = None
    metadata: Optional[Dict[str, Any]] = None


class Reader(ABC):
    """Abstract base class for image file readers.

    This class defines the interface for reading various image file formats.
    Subclasses implement the `open` method to handle specific file types
    (e.g., TIFF, Zarr, HDF5, NumPy). The reader pattern allows extensibility
    and consistent handling of different formats.

    Attributes
    ----------
    SUFFIXES : Tuple[str, ...]
        File extensions (e.g., ('.tif', '.tiff')) that this reader typically
        supports. Used by the `claims` method for format detection.

    See Also
    --------
    TiffReader : Reader for TIFF/OME-TIFF files.
    ZarrReader : Reader for Zarr stores.
    HDF5Reader : Reader for HDF5 files.
    NumpyReader : Reader for NumPy .npy/.npz files.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = TiffReader()
    >>> arr, info = reader.open(Path("image.tif"))
    >>> print(info.shape)
    (512, 512, 3)
    """

    # file suffixes this reader *typically* supports
    SUFFIXES: Tuple[str, ...] = ()

    @classmethod
    def claims(cls, path: Path) -> bool:
        """Check if this reader can handle the given file path.

        This is typically a lightweight test based on file extension,
        but can be overridden for more sophisticated detection.

        Parameters
        ----------
        path : Path
            The file path to check.

        Returns
        -------
        bool
            True if this reader claims to handle the file, False otherwise.

        Notes
        -----
        The default implementation checks if the file's suffix (lowercased)
        is in the `SUFFIXES` tuple. Subclasses may override this method
        for more complex detection logic.

        Examples
        --------
        >>> from pathlib import Path
        >>> TiffReader.claims(Path("image.tif"))
        True
        >>> TiffReader.claims(Path("image.zarr"))
        False
        """
        return path.suffix.lower() in cls.SUFFIXES

    @abstractmethod
    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open an image file and return the data and metadata.

        This is an abstract method that must be implemented by all subclasses.

        Parameters
        ----------
        path : Path
            The path to the image file.
        prefer_dask : bool, optional
            If True, return a Dask array when possible for lazy evaluation
            and out-of-core processing. If False, load the entire image
            into memory as a NumPy array. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses default or native chunking. Ignored if
            `prefer_dask` is False. Defaults to None.
        **kwargs : dict
            Additional keyword arguments specific to the reader implementation.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded image data. Type depends on `prefer_dask`:
            - If prefer_dask=False: NumPy ndarray
            - If prefer_dask=True: Dask array
        info : ImageInfo
            Metadata about the loaded image, including path, shape, dtype,
            axes information, and format-specific metadata.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file cannot be read or is not a valid format.

        Notes
        -----
        Implementations should handle both NumPy and Dask array returns
        based on the `prefer_dask` parameter. Dask arrays are recommended
        for large files that don't fit in memory.

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = TiffReader()
        >>> arr, info = reader.open(Path("image.tif"), prefer_dask=False)
        >>> print(type(arr).__name__)
        ndarray
        >>> darr, info = reader.open(Path("large.tif"), prefer_dask=True)
        >>> print(type(darr).__name__)
        Array
        """


class TiffReader(Reader):
    """Reader for TIFF and OME-TIFF files using the tifffile library.

    This reader handles standard TIFF files as well as OME-TIFF (Open Microscopy
    Environment) format files. It can extract axis order information from OME
    metadata when available.

    Attributes
    ----------
    SUFFIXES : tuple of str
        Supported file extensions: ('.tif', '.tiff').

    See Also
    --------
    Reader : Abstract base class for all readers.
    ZarrReader : Reader for Zarr stores.
    HDF5Reader : Reader for HDF5 files.

    Notes
    -----
    When `prefer_dask=True`, this reader uses tifffile's `aszarr=True` option
    to create a Zarr-backed Dask array, enabling lazy loading and efficient
    processing of large TIFF files without loading them entirely into memory.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = TiffReader()
    >>> arr, info = reader.open(Path("image.tif"))
    >>> print(arr.shape)
    (512, 512, 3)

    >>> # For large files, use Dask
    >>> darr, info = reader.open(Path("large.ome.tif"), prefer_dask=True)
    >>> print(info.axes)
    TCZYX
    """

    SUFFIXES = (".tif", ".tiff")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open a TIFF or OME-TIFF file and return the image data and metadata.

        This method reads TIFF files using the tifffile library. For OME-TIFF
        files, it attempts to extract axis order information (e.g., "TCZYX")
        from the OME-XML metadata.

        Parameters
        ----------
        path : Path
            The path to the TIFF or OME-TIFF file.
        prefer_dask : bool, optional
            If True, return a Dask array backed by a Zarr store for lazy
            evaluation. If False, load the entire image into memory as a
            NumPy array. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses tifffile's default chunking. Only relevant when
            `prefer_dask=True`. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to `tifffile.imread`.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded image data. Returns a NumPy ndarray if `prefer_dask=False`,
            or a Dask array if `prefer_dask=True`.
        info : ImageInfo
            Metadata about the loaded image, including path, shape, dtype,
            and axis order (extracted from OME-XML if available).

        Raises
        ------
        ValueError
            If the file cannot be read as a valid TIFF file.
        FileNotFoundError
            If the specified file does not exist.

        Notes
        -----
        When `prefer_dask=True`, the reader uses tifffile's `aszarr=True`
        option, which creates a Zarr store interface to the TIFF file. This
        allows Dask to read data lazily without loading the entire file into
        memory, which is beneficial for large microscopy images.

        For OME-TIFF files, the axis order is extracted from the OME-XML
        metadata using the DimensionOrder attribute. Common values include
        "TCZYX" (time, channel, Z, Y, X) or "ZCYX" (Z, channel, Y, X).

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = TiffReader()

        >>> # Load a standard TIFF into memory
        >>> arr, info = reader.open(Path("sample.tif"))
        >>> print(f"Shape: {info.shape}, dtype: {info.dtype}")
        Shape: (1024, 1024), dtype: uint16

        >>> # Load an OME-TIFF as a Dask array
        >>> darr, info = reader.open(Path("timelapse.ome.tif"), prefer_dask=True)
        >>> print(f"Axes: {info.axes}, Type: {type(darr).__name__}")
        Axes: ['t', 'c', 'z', 'y', 'x'], Type: Array

        >>> # Specify custom chunking for Dask
        >>> darr, info = reader.open(
        ...     Path("large.tif"),
        ...     prefer_dask=True,
        ...     chunks=(1, 512, 512)
        ... )
        >>> print(darr.chunksize)
        (1, 512, 512)
        """

        # Try OME-axes, pixel size, and metadata
        with tifffile.TiffFile(str(path)) as tf:
            ome_meta = getattr(tf, "omexml", None)
            axes = None
            pixel_size = None
            
            if ome_meta is not None:
                try:
                    axes = ome_meta.image().pixels().DimensionOrder  # e.g., "TCZYX"
                except Exception:
                    axes = None
            
            # Try to extract pixel size from OME metadata
            if HAS_OME_TYPES and hasattr(tf, 'ome_metadata') and tf.ome_metadata:
                try:
                    ome = from_xml(tf.ome_metadata)
                    if ome.images and ome.images[0].pixels:
                        pixels = ome.images[0].pixels
                        # Extract physical sizes (in micrometers by default in OME)
                        size_x = getattr(pixels, 'physical_size_x', None)
                        size_y = getattr(pixels, 'physical_size_y', None)
                        size_z = getattr(pixels, 'physical_size_z', None)
                        
                        # Build pixel_size tuple in ZYX order to match axes convention
                        if size_x is not None and size_y is not None:
                            if size_z is not None:
                                pixel_size = (size_z, size_y, size_x)
                            else:
                                pixel_size = (size_y, size_x)
                except Exception:
                    # If OME parsing fails, pixel_size remains None
                    pass
            
            # Fallback: try to extract from standard TIFF resolution tags
            if pixel_size is None:
                try:
                    page = tf.pages[0]
                    x_res = page.tags.get('XResolution')
                    y_res = page.tags.get('YResolution')
                    res_unit = page.tags.get('ResolutionUnit')
                    
                    if x_res and y_res and res_unit:
                        # Extract resolution values
                        x_val = x_res.value[0] / x_res.value[1] if isinstance(x_res.value, tuple) else x_res.value
                        y_val = y_res.value[0] / y_res.value[1] if isinstance(y_res.value, tuple) else y_res.value
                        
                        # Convert to micrometers per pixel based on unit
                        # Resolution unit: 1=none, 2=inch, 3=centimeter
                        if res_unit.value == 2:  # inch
                            # pixels per inch -> um per pixel
                            x_um = 25400.0 / x_val  # 1 inch = 25400 um
                            y_um = 25400.0 / y_val
                            pixel_size = (y_um, x_um)  # YX order
                        elif res_unit.value == 3:  # centimeter
                            # pixels per cm -> um per pixel
                            x_um = 10000.0 / x_val  # 1 cm = 10000 um
                            y_um = 10000.0 / y_val
                            pixel_size = (y_um, x_um)  # YX order
                except Exception:
                    # If standard TIFF tag parsing fails, pixel_size remains None
                    pass

        if prefer_dask:
            # Option A: use tifffile's OME-as-zarr path if possible
            # This keeps it lazy and chunked without loading into RAM
            store = tifffile.imread(str(path), aszarr=True)
            darr = da.from_zarr(store, chunks=chunks) if chunks else da.from_zarr(store)
            info = ImageInfo(
                path=path,
                shape=tuple(darr.shape),
                dtype=darr.dtype,
                axes=_normalize_axes(axes),
                pixel_size=pixel_size,
                metadata={},
            )
            logger.info(f"Loaded {path.name} as a Dask array.")
            return darr, info
        else:
            # Load to memory as NumPy
            arr = tifffile.imread(str(path))
            info = ImageInfo(
                path=path,
                shape=tuple(arr.shape),
                dtype=arr.dtype,
                axes=_normalize_axes(axes),
                pixel_size=pixel_size,
                metadata={},
            )
            logger.info(f"Loaded {path.name} as NumPy array.")
            return arr, info


class ZarrReader(Reader):
    """Reader for Zarr and N5 storage formats.

    This reader handles Zarr stores and N5 format
    directories. It automatically selects the largest array (by number of elements)
    from the store and can extract axis information and metadata from Zarr attributes.

    Attributes
    ----------
    SUFFIXES : tuple of str
        Supported file extensions: ('.zarr', '.zarr/', '.n5', '.n5/').

    See Also
    --------
    Reader : Abstract base class for all readers.
    TiffReader : Reader for TIFF/OME-TIFF files.
    HDF5Reader : Reader for HDF5 files.

    Notes
    -----
    Zarr is a chunked, compressed array storage format designed for efficient
    parallel and cloud-based computing. This reader is particularly well-suited
    for `prefer_dask=True` since Zarr's native chunking aligns perfectly with
    Dask's lazy evaluation model.

    When multiple arrays exist in a Zarr group, the reader selects the array
    with the largest number of elements (np.prod(shape)), which typically
    corresponds to the highest resolution image.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = ZarrReader()
    >>> arr, info = reader.open(Path("data.zarr"))
    >>> print(arr.shape)
    (100, 512, 512)

    >>> # For large datasets, use Dask for lazy loading
    >>> darr, info = reader.open(Path("large.zarr"), prefer_dask=True)
    >>> print(info.axes)
    ['z', 'y', 'x']
    """

    SUFFIXES = (".zarr", ".zarr/", ".n5", ".n5/")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open a Zarr or N5 store and return the image data and metadata.

        This method opens a Zarr store, identifies all arrays within the group,
        and selects the largest one by number of elements. Metadata including
        axis information and custom attributes are extracted when available.

        Parameters
        ----------
        path : Path
            The path to the Zarr store directory (e.g., 'data.zarr' or 'data.n5').
        prefer_dask : bool, optional
            If True, return a Dask array for lazy evaluation. If False, load
            the entire array into memory as a NumPy array. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses the Zarr store's native chunking. Only relevant when
            `prefer_dask=True`. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to `zarr.open_group`.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded image data. Returns a NumPy ndarray if `prefer_dask=False`,
            or a Dask array if `prefer_dask=True`.
        info : ImageInfo
            Metadata about the loaded image, including path, shape, dtype,
            axes information (if available from Zarr attributes), and any
            additional metadata stored in the array's attributes.

        Raises
        ------
        ValueError
            If the Zarr store contains no arrays.
        FileNotFoundError
            If the specified path does not exist.

        Notes
        -----
        When `prefer_dask=True`, this reader leverages Zarr's native chunking,
        which is optimal for Dask's lazy evaluation. The array remains on disk
        and is read only when computation is triggered.

        The reader attempts to extract axis information from multiple common
        attribute locations:
        - `multiscales[0]['axes']` (OME-Zarr format)
        - `axes` attribute (custom metadata)

        If multiple arrays exist in the Zarr group, the array with the largest
        number of elements (np.prod(shape)) is selected, typically corresponding
        to the full-resolution image in multi-resolution pyramids.

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = ZarrReader()

        >>> # Load a Zarr store into memory
        >>> arr, info = reader.open(Path("image.zarr"))
        >>> print(f"Shape: {info.shape}, dtype: {info.dtype}")
        Shape: (50, 1024, 1024), dtype: uint16

        >>> # Load as Dask array for large datasets
        >>> darr, info = reader.open(Path("large.zarr"), prefer_dask=True)
        >>> print(f"Type: {type(darr).__name__}, chunks: {darr.chunksize}")
        Type: Array, chunks: (1, 512, 512)

        >>> # Override native chunking
        >>> darr, info = reader.open(
        ...     Path("data.zarr"),
        ...     prefer_dask=True,
        ...     chunks=(10, 256, 256)
        ... )
        >>> print(darr.chunksize)
        (10, 256, 256)

        >>> # Access metadata
        >>> arr, info = reader.open(Path("ome.zarr"))
        >>> print(info.metadata.keys())
        dict_keys(['multiscales', 'omero', '_ARRAY_DIMENSIONS'])
        """
        grp = zarr.open_group(str(path), mode="r")

        # collect all arrays
        arrays = []
        if hasattr(grp, "array_keys") and callable(grp.array_keys):
            arrays = [grp[k] for k in grp.array_keys()]

        if not arrays:
            logger.error(f"No arrays found in Zarr group: {path}")
            raise ValueError(f"No arrays found in Zarr group: {path}")

        # Pick array with the largest number of elements
        array = max(arrays, key=lambda arr: np.prod(arr.shape))

        axes = None
        pixel_size = None
        meta = {}
        try:
            attrs = getattr(array, "attrs", {})
            axes = attrs.get("multiscales", [{}])[0].get("axes") or attrs.get("axes")
            meta = dict(attrs)
            
            # Try to extract pixel size from Zarr attributes
            # Check for OME-Zarr style multiscales metadata
            if "multiscales" in attrs and attrs["multiscales"]:
                multiscale = attrs["multiscales"][0]
                if "axes" in multiscale and isinstance(multiscale["axes"], list):
                    # Look for scale/transform information
                    if "datasets" in multiscale and multiscale["datasets"]:
                        dataset = multiscale["datasets"][0]
                        if "coordinateTransformations" in dataset:
                            for transform in dataset["coordinateTransformations"]:
                                if transform.get("type") == "scale" and "scale" in transform:
                                    # Scale values typically correspond to axis order
                                    pixel_size = tuple(transform["scale"])
                                    break
            
            # Fallback: check for direct pixel_size or scale attributes
            if pixel_size is None:
                if "pixel_size" in attrs:
                    pixel_size = _ensure_tuple(attrs["pixel_size"])
                elif "scale" in attrs:
                    pixel_size = _ensure_tuple(attrs["scale"])
                elif "resolution" in attrs:
                    pixel_size = _ensure_tuple(attrs["resolution"])
                    
        except Exception:
            pass

        if prefer_dask:
            darr = da.from_zarr(array, chunks=chunks) if chunks else da.from_zarr(array)
            logger.info(f"Loaded {path.name} as a Dask array.")
            info = ImageInfo(
                path=path,
                shape=tuple(darr.shape),
                dtype=darr.dtype,
                axes=_normalize_axes(axes),
                pixel_size=pixel_size,
                metadata=meta,
            )
            return darr, info
        else:
            # Load to memory as NumPy
            np_arr = np.array(array) if np is not None else array[:]
            logger.info(f"Loaded {path.name} as a NumPy array.")
            info = ImageInfo(
                path=path,
                shape=tuple(np_arr.shape),
                dtype=np_arr.dtype,
                axes=_normalize_axes(axes),
                pixel_size=pixel_size,
                metadata=meta,
            )
            return np_arr, info


class HDF5Reader(Reader):
    """Reader for HDF5 files using the h5py library.

    This reader handles HDF5 (Hierarchical Data Format version 5) files,
    a widely used format for storing large scientific datasets. When multiple
    datasets exist in the file, the reader automatically selects the largest
    one by number of elements (highest resolution).

    Attributes
    ----------
    SUFFIXES : tuple of str
        Supported file extensions: ('.h5', '.hdf5', '.hdf').

    See Also
    --------
    Reader : Abstract base class for all readers.
    TiffReader : Reader for TIFF/OME-TIFF files.
    ZarrReader : Reader for Zarr stores.

    Notes
    -----
    This reader searches the entire HDF5 file structure recursively and
    automatically selects the dataset with the largest number of elements.
    It does not accept subpath specifications (e.g., 'file.h5/group/dataset').

    When `prefer_dask=True`, the reader uses h5py's dataset interface with
    Dask's `from_array` method, enabling lazy loading with thread-safe access.
    This is particularly useful for large datasets that don't fit in memory.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = HDF5Reader()
    >>> arr, info = reader.open(Path("data.h5"))
    >>> print(arr.shape)
    (200, 512, 512)

    >>> # For large files, use Dask for lazy loading
    >>> darr, info = reader.open(Path("large.h5"), prefer_dask=True)
    >>> print(type(darr).__name__)
    Array
    """

    SUFFIXES = (".h5", ".hdf5", ".hdf")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open an HDF5 file and return the highest resolution dataset.

        This method opens an HDF5 file, recursively searches all groups for
        datasets, and selects the one with the largest number of elements.
        Metadata and axis information are extracted from the dataset's
        attributes when available.

        Parameters
        ----------
        path : Path
            Path to the HDF5 file (e.g., '/path/to/data.h5'). The reader
            does not accept subpath specifications; it automatically finds
            and selects the largest dataset in the file.
        prefer_dask : bool, optional
            If True, return a Dask array backed by the on-disk dataset for
            lazy evaluation with thread-safe access. If False, load the
            entire dataset into memory as a NumPy array. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses the dataset's native HDF5 chunking if available,
            otherwise defaults to 128 per dimension. Only relevant when
            `prefer_dask=True`. Defaults to None.
        **kwargs : dict
            Reserved for future compatibility with additional reader options.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded image data. Returns a NumPy ndarray if `prefer_dask=False`,
            or a Dask array if `prefer_dask=True`.
        info : ImageInfo
            Metadata about the loaded image, including path, shape, dtype,
            axes information (if available from HDF5 attributes), dataset
            name, and any additional attributes stored with the dataset.

        Raises
        ------
        ValueError
            If the file contains no datasets.
        FileNotFoundError
            If the specified file does not exist.

        Notes
        -----
        When `prefer_dask=True`, the reader creates a Dask array backed by
        the HDF5 dataset with thread-safe locking enabled. This allows
        multiple Dask workers to read from the same file safely. The dataset
        remains on disk and is read only when computation is triggered.

        The reader attempts to extract axis information from common HDF5
        attribute names:
        - `axes` : Custom axis labels
        - `dimension_order` or `DimensionOrder` : Axis ordering
        - `DIMENSION_LABELS` : Alternative axis label specification

        For datasets without native HDF5 chunking, the reader applies a
        conservative default chunk size of 128 elements per dimension when
        `prefer_dask=True` and no explicit chunking is specified.

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = HDF5Reader()

        >>> # Load an HDF5 file into memory
        >>> arr, info = reader.open(Path("image.h5"))
        >>> print(f"Shape: {info.shape}, dtype: {info.dtype}")
        Shape: (100, 1024, 1024), dtype: float32

        >>> # Load as Dask array for large files
        >>> darr, info = reader.open(Path("large.h5"), prefer_dask=True)
        >>> print(f"Type: {type(darr).__name__}, chunks: {darr.chunksize}")
        Type: Array, chunks: (10, 512, 512)

        >>> # Specify custom chunking
        >>> darr, info = reader.open(
        ...     Path("data.h5"),
        ...     prefer_dask=True,
        ...     chunks=(5, 256, 256)
        ... )
        >>> print(darr.chunksize)
        (5, 256, 256)

        >>> # Access metadata and dataset information
        >>> arr, info = reader.open(Path("annotated.h5"))
        >>> print(info.metadata.get('axes'))
        ZYX
        >>> print(f"Loaded dataset: {info.metadata.get('dataset_name')}")
        Loaded dataset: /images/channel_0
        """
        # Be tolerant of a trailing '/' on the filename.
        path_str = str(path).rstrip(os.sep)
        f = h5py.File(path_str, mode="r")

        # Recursively collect all datasets in the file
        def _collect_datasets(group: h5py.Group) -> list:
            out = []
            for key, obj in group.items():
                if isinstance(obj, h5py.Dataset):
                    out.append(obj)
                elif isinstance(obj, h5py.Group):
                    out.extend(_collect_datasets(obj))
            return out

        datasets = _collect_datasets(f)
        if not datasets:
            logger.error(f"No datasets found in HDF5 file: {path_str}")
            f.close()
            raise ValueError(f"No datasets found in HDF5 file: {path_str}")

        # Highest resolution ≈ largest number of elements
        ds = max(datasets, key=lambda d: int(np.prod(d.shape)))

        # Extract basic metadata / axes / pixel_size if present
        axes = None
        pixel_size = None
        meta: Dict[str, Any] = {}
        try:
            attrs = dict(ds.attrs) if hasattr(ds, "attrs") else {}
            meta = {str(k): v for k, v in attrs.items()}
            axes = (
                attrs.get("axes")
                or attrs.get("dimension_order")
                or attrs.get("DimensionOrder")
                or attrs.get("DIMENSION_LABELS")  # sometimes stored by other tools
            )
            
            # Try to extract pixel size from HDF5 attributes
            if "pixel_size" in attrs:
                pixel_size = _ensure_tuple(attrs["pixel_size"])
            elif "scale" in attrs:
                pixel_size = _ensure_tuple(attrs["scale"])
            elif "resolution" in attrs:
                pixel_size = _ensure_tuple(attrs["resolution"])
            # Check for individual axis scales
            elif "element_size_um" in attrs:
                pixel_size = _ensure_tuple(attrs["element_size_um"])
                
        except Exception:
            pass

        if prefer_dask:
            # Use native HDF5 chunking if available; otherwise a conservative fallback
            native_chunks = getattr(
                ds, "chunks", None
            )  # may be None if dataset isn't chunked
            eff_chunks = chunks or native_chunks
            if eff_chunks is None:
                eff_chunks = tuple(min(128, s) for s in ds.shape)

            # For h5py-backed arrays, keeping a reference to `ds` is sufficient for I/O
            darr = da.from_array(
                ds,
                chunks=eff_chunks,
                name=f"h5::{os.path.basename(path_str)}::{ds.name}",
                lock=True,  # ensure thread-safe access to HDF5
            )
            info = ImageInfo(
                path=Path(path_str),
                shape=tuple(darr.shape),
                dtype=darr.dtype,
                axes=_normalize_axes(axes),
                pixel_size=pixel_size,
                metadata=meta,
            )
            logger.info(
                f"Loaded {Path(path_str).name} (HDF5) as Dask array from dataset '{ds.name}'."
            )
            return darr, info

        np_arr = ds[...]
        f.close()

        info = ImageInfo(
            path=Path(path_str),
            shape=tuple(np_arr.shape),
            dtype=np_arr.dtype,
            axes=_normalize_axes(axes),
            pixel_size=pixel_size,
            metadata=meta,
        )
        logger.info(
            f"Loaded {Path(path_str).name} (HDF5) as NumPy array from dataset '{ds.name}'."
        )
        return np_arr, info


class NumpyReader(Reader):
    """Reader for NumPy binary files (.npy and .npz).

    This reader handles NumPy's native binary formats: .npy files (single arrays)
    and .npz files (archives containing multiple named arrays). For .npz files,
    the reader selects the first array in the archive.

    Attributes
    ----------
    SUFFIXES : tuple of str
        Supported file extensions: ('.npy', '.npz').

    See Also
    --------
    Reader : Abstract base class for all readers.
    TiffReader : Reader for TIFF/OME-TIFF files.
    ZarrReader : Reader for Zarr stores.
    HDF5Reader : Reader for HDF5 files.

    Notes
    -----
    For .npy files with `prefer_dask=True`, this reader uses NumPy's memory-mapped
    mode (`mmap_mode='r'`) which allows lazy loading without reading the entire
    file into memory. The memory-mapped array is then wrapped in a Dask array.

    For .npz archives, the first array (alphabetically by key name) is loaded.
    If you need to load a specific array from an .npz file, consider using
    `np.load()` directly or extending this reader.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = NumpyReader()
    >>> arr, info = reader.open(Path("data.npy"))
    >>> print(arr.shape)
    (512, 512)

    >>> # For large files, use Dask with memory mapping
    >>> darr, info = reader.open(Path("large.npy"), prefer_dask=True)
    >>> print(type(darr).__name__)
    Array
    """

    SUFFIXES = (".npy", ".npz")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open a NumPy .npy or .npz file and return the array data and metadata.

        This method handles both .npy (single array) and .npz (archive) formats.
        For .npy files with `prefer_dask=True`, memory-mapping is used for
        efficient lazy loading. For .npz archives, the first array is selected.

        Parameters
        ----------
        path : Path
            Path to the NumPy file (e.g., '/path/to/data.npy' or 'archive.npz').
        prefer_dask : bool, optional
            If True, return a Dask array. For .npy files, this uses memory-mapping
            for lazy loading. For .npz files, the array is wrapped in a Dask array
            after loading. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses 'auto' chunking. Only relevant when `prefer_dask=True`.
            Defaults to None.
        **kwargs : dict
            Reserved for future compatibility with additional reader options.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded array data. Returns a NumPy ndarray if `prefer_dask=False`,
            or a Dask array if `prefer_dask=True`.
        info : ImageInfo
            Metadata about the loaded array, including path, shape, and dtype.
            For .npz files, the metadata includes the 'npz_key' indicating which
            array was loaded from the archive.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the .npz archive is empty or cannot be read.

        Notes
        -----
        When `prefer_dask=True` for .npy files, the reader uses NumPy's
        memory-mapped mode, which maps the file directly into virtual memory
        without loading it into RAM. This is efficient for large arrays that
        don't fit in memory.

        For .npz archives, all arrays are technically loaded when the archive
        is opened, but only the first one is returned. The first array is
        determined by alphabetical ordering of the key names in the archive.

        NumPy binary files do not typically contain axis labels or other
        metadata, so the `axes` field in ImageInfo is always None, and
        the metadata dictionary is empty (except for 'npz_key' in .npz files).

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = NumpyReader()

        >>> # Load a .npy file into memory
        >>> arr, info = reader.open(Path("image.npy"))
        >>> print(f"Shape: {info.shape}, dtype: {info.dtype}")
        Shape: (1024, 1024), dtype: float64

        >>> # Load a .npy file as a Dask array with memory mapping
        >>> darr, info = reader.open(Path("large.npy"), prefer_dask=True)
        >>> print(f"Type: {type(darr).__name__}")
        Type: Array

        >>> # Specify custom chunking for Dask
        >>> darr, info = reader.open(
        ...     Path("data.npy"),
        ...     prefer_dask=True,
        ...     chunks=(512, 512)
        ... )
        >>> print(darr.chunksize)
        (512, 512)

        >>> # Load from a .npz archive
        >>> arr, info = reader.open(Path("archive.npz"))
        >>> print(f"Loaded array: {info.metadata['npz_key']}")
        Loaded array: arr_0

        >>> # Load .npz as Dask array
        >>> darr, info = reader.open(Path("archive.npz"), prefer_dask=True)
        >>> print(f"Key: {info.metadata['npz_key']}, Type: {type(darr).__name__}")
        Key: arr_0, Type: Array
        """

        if path.suffix.lower() == ".npy":
            if prefer_dask:
                # memmap keeps it lazy-ish; wrap with dask
                mem = np.load(str(path), mmap_mode="r")
                darr = da.from_array(mem, chunks=chunks or "auto")
                info = ImageInfo(
                    path=path,
                    shape=tuple(darr.shape),
                    dtype=darr.dtype,
                    axes=None,
                    pixel_size=None,
                    metadata={},
                )
                return darr, info
            else:
                arr = np.load(str(path))
                info = ImageInfo(
                    path=path,
                    shape=tuple(arr.shape),
                    dtype=arr.dtype,
                    axes=None,
                    pixel_size=None,
                    metadata={},
                )
                return arr, info
        else:
            # .npz: load first array by name (or all if you prefer)
            npz = np.load(str(path))
            first_key = list(npz.keys())[0]
            arr = npz[first_key]
            if prefer_dask:
                darr = da.from_array(arr, chunks=chunks or "auto")
                info = ImageInfo(
                    path=path,
                    shape=tuple(darr.shape),
                    dtype=darr.dtype,
                    axes=None,
                    pixel_size=None,
                    metadata={"npz_key": first_key},
                )
                return darr, info
            else:
                info = ImageInfo(
                    path=path,
                    shape=tuple(arr.shape),
                    dtype=arr.dtype,
                    axes=None,
                    pixel_size=None,
                    metadata={"npz_key": first_key},
                )
                return arr, info


class ND2Reader(Reader):
    """Reader for Nikon ND2 files using the nd2 library.

    This reader handles Nikon's proprietary ND2 format files commonly used
    in microscopy. It can extract comprehensive metadata including pixel/voxel
    sizes, axis information, and experimental parameters.

    Attributes
    ----------
    SUFFIXES : tuple of str
        Supported file extensions: ('.nd2',).

    See Also
    --------
    Reader : Abstract base class for all readers.
    TiffReader : Reader for TIFF/OME-TIFF files.
    ZarrReader : Reader for Zarr stores.

    Notes
    -----
    The nd2 library provides native support for both NumPy arrays and Dask
    arrays through its `to_dask()` method. When `prefer_dask=True`, this
    reader uses the native Dask support for efficient lazy loading of large
    ND2 files without loading them entirely into memory.

    Pixel size information is extracted from the Volume metadata and stored
    in the ImageInfo as (Z, Y, X) calibration values in micrometers, reordered
    from the ND2 native (X, Y, Z) format to match our axes convention.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = ND2Reader()
    >>> arr, info = reader.open(Path("image.nd2"))
    >>> print(arr.shape)
    (10, 512, 512)
    >>> print(info.pixel_size)
    (2.0, 0.65, 0.65)  # Z, Y, X in micrometers

    >>> # For large files, use Dask
    >>> darr, info = reader.open(Path("large.nd2"), prefer_dask=True)
    >>> print(type(darr).__name__)
    Array
    """

    SUFFIXES = (".nd2",)

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open an ND2 file and return the image data and metadata.

        This method reads Nikon ND2 files using the nd2 library. It extracts
        comprehensive metadata including pixel sizes, axis order, and other
        experimental parameters.

        Parameters
        ----------
        path : Path
            The path to the ND2 file.
        prefer_dask : bool, optional
            If True, return a Dask array for lazy evaluation. If False, load
            the entire image into memory as a NumPy array. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses nd2's default chunking. Only relevant when
            `prefer_dask=True`. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to `nd2.ND2File`.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded image data. Returns a NumPy ndarray if `prefer_dask=False`,
            or a Dask array if `prefer_dask=True`.
        info : ImageInfo
            Metadata about the loaded image, including path, shape, dtype,
            axes information, pixel size (in micrometers), and format-specific
            metadata.

        Raises
        ------
        ValueError
            If the file cannot be read as a valid ND2 file.
        FileNotFoundError
            If the specified file does not exist.

        Notes
        -----
        The reader extracts pixel/voxel sizes from the ND2 metadata's Volume
        section. The native ND2 format stores calibration as (X, Y, Z), but
        these are reordered to (Z, Y, X) to match our axes convention and
        stored in the `pixel_size` field of ImageInfo.

        Axes information is extracted from the ND2 file's dimension names and
        converted to a string format (e.g., "TCZYX").

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = ND2Reader()

        >>> # Load a standard ND2 file into memory
        >>> arr, info = reader.open(Path("sample.nd2"))
        >>> print(f"Shape: {info.shape}, dtype: {info.dtype}")
        Shape: (5, 512, 512), dtype: uint16
        >>> print(f"Pixel size: {info.pixel_size}")
        Pixel size: (1.0, 0.325, 0.325)  # Z, Y, X in micrometers

        >>> # Load an ND2 as a Dask array
        >>> darr, info = reader.open(Path("large.nd2"), prefer_dask=True)
        >>> print(f"Type: {type(darr).__name__}")
        Type: Array

        >>> # Specify custom chunking for Dask
        >>> darr, info = reader.open(
        ...     Path("timelapse.nd2"),
        ...     prefer_dask=True,
        ...     chunks=(1, 512, 512)
        ... )
        >>> print(darr.chunksize)
        (1, 512, 512)
        """
        
        if not HAS_ND2:
            raise ImportError(
                "The 'nd2' library is required to read ND2 files. "
                "Install it with: pip install nd2"
            )

        with nd2.ND2File(str(path), **kwargs) as nd2_file:
            # Extract metadata
            metadata_dict = {}
            axes = None
            pixel_size = None

            # Get axes information from sizes dict
            if hasattr(nd2_file, 'sizes') and nd2_file.sizes:
                # Pass the sizes dict to normalize_axes which will extract keys
                axes = nd2_file.sizes

            # Extract pixel size from metadata
            try:
                if nd2_file.metadata and nd2_file.metadata.channels:
                    # Get the first channel's volume information
                    channel = nd2_file.metadata.channels[0]
                    if channel.volume and channel.volume.axesCalibration:
                        # axesCalibration is (X, Y, Z) in micrometers
                        # Reorder to (Z, Y, X) to match our axes convention
                        x, y, z = channel.volume.axesCalibration
                        pixel_size = (z, y, x)
            except (AttributeError, IndexError, TypeError):
                # If metadata extraction fails, pixel_size remains None
                pass

            # Store additional metadata
            if nd2_file.metadata:
                metadata_dict['metadata'] = nd2_file.metadata
            if hasattr(nd2_file, 'attributes') and nd2_file.attributes:
                metadata_dict['attributes'] = nd2_file.attributes

            if prefer_dask:
                # Use nd2's native Dask support
                darr = nd2_file.to_dask()
                
                # Apply custom chunking if specified
                if chunks is not None:
                    darr = darr.rechunk(chunks)
                
                info = ImageInfo(
                    path=path,
                    shape=tuple(darr.shape),
                    dtype=darr.dtype,
                    axes=_normalize_axes(axes),
                    pixel_size=pixel_size,
                    metadata=metadata_dict,
                )
                logger.info(f"Loaded {path.name} as a Dask array.")
                return darr, info
            else:
                # Load to memory as NumPy
                arr = nd2_file.asarray()
                info = ImageInfo(
                    path=path,
                    shape=tuple(arr.shape),
                    dtype=arr.dtype,
                    axes=_normalize_axes(axes),
                    pixel_size=pixel_size,
                    metadata=metadata_dict,
                )
                logger.info(f"Loaded {path.name} as NumPy array.")
                return arr, info


class ImageOpener:
    """Automatic image file format detection and reading.

    ImageOpener provides a unified interface for opening various image file
    formats. It automatically selects the appropriate reader based on file
    extension and falls back to probing if the extension is ambiguous or unknown.

    The opener maintains a registry of reader classes and attempts to use them
    in priority order. This design allows easy extension with custom readers
    and provides a single entry point for loading images regardless of format.

    Attributes
    ----------
    _readers : Tuple[Type[Reader], ...]
        Ordered tuple of reader classes. The first reader to successfully
        claim and open a file is used. Priority is determined by order.

    See Also
    --------
    Reader : Abstract base class for all readers.
    TiffReader : Reader for TIFF/OME-TIFF files.
    ZarrReader : Reader for Zarr stores.
    HDF5Reader : Reader for HDF5 files.
    NumpyReader : Reader for NumPy .npy/.npz files.

    Notes
    -----
    The opener uses a two-phase strategy:
    1. Extension-based: Tries readers whose `claims()` method returns True
    2. Fallback: If no reader claims the file, probes all readers sequentially

    This approach handles both standard file extensions and edge cases where
    files may have unconventional naming or multiple valid interpretations.

    Examples
    --------
    >>> from pathlib import Path
    >>> opener = ImageOpener()
    >>> arr, info = opener.open("image.tif")
    >>> print(info.shape)
    (512, 512, 3)

    >>> # Works with any supported format
    >>> arr, info = opener.open("data.zarr", prefer_dask=True)
    >>> print(type(arr).__name__)
    Array

    >>> # Custom reader registry
    >>> opener = ImageOpener(readers=[TiffReader, ZarrReader])
    >>> arr, info = opener.open("image.tif")
    """

    def __init__(self, readers: Optional[Iterable[Type[Reader]]] = None) -> None:
        """Initialize the ImageOpener with a reader registry.

        Parameters
        ----------
        readers : Iterable of Reader subclasses, optional
            An ordered sequence of reader classes to use. If None, uses the
            default registry: (TiffReader, ZarrReader, NumpyReader, HDF5Reader,
            ND2Reader). The order determines priority when multiple readers
            could handle the same file.

        Notes
        -----
        Custom readers can be provided by passing a sequence of Reader subclasses.
        Readers earlier in the sequence have higher priority and will be tried
        first when opening files.

        Examples
        --------
        >>> # Use default readers
        >>> opener = ImageOpener()

        >>> # Custom reader priority
        >>> opener = ImageOpener(readers=[ZarrReader, TiffReader])

        >>> # Single reader only
        >>> opener = ImageOpener(readers=[TiffReader])
        """
        # Registry order is priority order
        self._readers: Tuple[Type[Reader], ...] = tuple(
            readers or (TiffReader, ZarrReader, NumpyReader, HDF5Reader, ND2Reader)
        )

    def open(
        self,
        path: Union[str, os.PathLike],
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[Any], ImageInfo]:
        """Open an image file using the appropriate reader.

        This method automatically detects the file format and selects the
        best reader from the registry. It first tries extension-based matching,
        then falls back to probing each reader if necessary.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the image file or directory (e.g., 'image.tif', 'data.zarr').
        prefer_dask : bool, optional
            If True, attempt to return a Dask array for lazy evaluation when
            the reader supports it. If False, load the entire image into memory
            as a NumPy array. Defaults to False.
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. Can be a single integer (applied to
            all dimensions) or a tuple specifying chunk size per dimension.
            If None, uses the reader's default chunking strategy. Only relevant
            when `prefer_dask=True`. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the selected reader's `open`
            method. These vary by reader implementation.

        Returns
        -------
        arr : NDArray[Any] or dask.array.Array
            The loaded image data. Returns a NumPy ndarray if `prefer_dask=False`,
            or a Dask array if `prefer_dask=True` and the reader supports it.
        info : ImageInfo
            Metadata about the loaded image, including path, shape, dtype,
            axes information (if available), and format-specific metadata.

        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.
        ValueError
            If no suitable reader in the registry can open the file.

        Notes
        -----
        The method uses a two-phase approach:

        Phase 1 - Extension-based selection:
            Iterates through registered readers in priority order and uses
            the first one whose `claims()` method returns True for the file.

        Phase 2 - Fallback probing:
            If no reader claims the file, attempts to open it with each
            reader sequentially until one succeeds.

        This strategy handles both common cases (correct file extension) and
        edge cases (missing extension, multiple interpretations, etc.).

        Examples
        --------
        >>> from pathlib import Path
        >>> opener = ImageOpener()

        >>> # Open a TIFF file into memory
        >>> arr, info = opener.open("sample.tif")
        >>> print(f"Shape: {info.shape}, dtype: {info.dtype}")
        Shape: (1024, 1024), dtype: uint16

        >>> # Open a Zarr store as Dask array
        >>> darr, info = opener.open("large.zarr", prefer_dask=True)
        >>> print(f"Type: {type(darr).__name__}")
        Type: Array

        >>> # Specify custom chunking
        >>> darr, info = opener.open(
        ...     "data.h5",
        ...     prefer_dask=True,
        ...     chunks=(10, 256, 256)
        ... )
        >>> print(darr.chunksize)
        (10, 256, 256)

        >>> # Open with Path object
        >>> arr, info = opener.open(Path("image.npy"))
        >>> print(info.path)
        image.npy

        >>> # Works with any format in the registry
        >>> for filepath in ["data.tif", "data.zarr", "data.h5", "data.npy"]:
        ...     arr, info = opener.open(filepath)
        ...     print(f"{filepath}: {info.shape}")
        """

        p = Path(path)
        if not p.exists():
            logger.error(msg=f"File {p} does not exist")
            raise FileNotFoundError(p)

        # 1) Extension-based selection
        logger.info(msg=f"Opening {p}")
        for reader_cls in self._readers:
            try:
                if reader_cls.claims(p):
                    reader = reader_cls()
                    logger.info(msg=f"Using reader: {reader_cls.__name__}.")
                    return reader.open(
                        path=p, prefer_dask=prefer_dask, chunks=chunks, **kwargs
                    )
            except Exception:
                pass

        # 2) Fallback: probe readers that didn't claim the file
        logger.info(
            msg=f"No suitable reader found for {p}. Attempting fallback readers."
        )
        for reader_cls in self._readers:
            try:
                reader: Reader = reader_cls()
                return reader.open(
                    path=p, prefer_dask=prefer_dask, chunks=chunks, **kwargs
                )
            except Exception:
                continue

        logger.error(msg=f"No suitable reader found for {p}")
        raise ValueError("No suitable reader found for:", p)


def rename_tiff_to_tif(base_path: str, recursive: bool = True) -> int:
    """Rename all .tiff files to .tif in the given directory.

    Args:
        base_path: Directory to search for .tiff files
        recursive: If True, search subdirectories as well

    Returns:
        Number of files renamed
    """
    base = Path(base_path)
    pattern = "**/*.tiff" if recursive else "*.tiff"
    count = 0

    for tiff_file in base.glob(pattern):
        new_name = tiff_file.with_suffix(".tif")
        tiff_file.rename(new_name)
        print(f"Renamed: {tiff_file} -> {new_name}")
        count += 1

    print(f"Renamed {count} files.")
    return count
