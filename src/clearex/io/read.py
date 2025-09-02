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
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union
import logging

# Optional deps (import guarded)
try:
    import numpy as np
except ImportError:  
    np = None  

try:
    import dask.array as da  
except ImportError:
    da = None  

try:
    import tifffile  
except ImportError:
    tifffile = None  

try:
    import zarr  
except ImportError:
    zarr = None  


ArrayLike = Union["np.ndarray", "da.Array"]  # noqa: F821

# Start logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass
class ImageInfo:
    path: Path
    shape: Tuple[int, ...]
    dtype: Any
    axes: Optional[str] = None
    metadata: Dict[str, Any] = None


class Reader(ABC):
    """Abstract strategy for opening image data."""

    # file suffixes this reader *typically* supports
    SUFFIXES: Tuple[str, ...] = ()

    @classmethod
    def claims(cls, path: Path) -> bool:
        """Lightweight test: usually just extension check; may be overridden."""
        return path.suffix.lower() in cls.SUFFIXES

    @abstractmethod
    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        """Return array (NumPy or Dask) and ImageInfo."""


# --------------------------
# Concrete Readers
# --------------------------

class TiffReader(Reader):
    """Reader for TIFF/OME-TIFF files using tifffile."""
    SUFFIXES = (".tif", ".tiff")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        """Open TIFF/OME-TIFF file.

        Parameters
        ----------
        path : Path
            The path to the TIFF file.
        prefer_dask : bool, optional
            If True, attempt to return a Dask array when possible.
            Defaults to False (return NumPy array).
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. If None, use default chunking.
            Ignored if `prefer_dask` is False. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to tifffile.imread.
        Returns
        -------
        arr : np.ndarray or dask.array.Array
            The loaded image data as a NumPy or Dask array.
        info : ImageInfo
            Metadata about the loaded image.
        Raises
        ------
        ImportError
            If tifffile is not installed.
        ValueError
            If the file cannot be read as a TIFF.
        """
        if tifffile is None:
            logger.error("The tifffile package is not installed.")
            raise ImportError("tifffile is required to read TIFF images")

        # Try OME-axes and metadata
        with tifffile.TiffFile(str(path)) as tf:
            ome_meta = getattr(tf, "omexml", None)
            axes = None
            if ome_meta is not None:
                try:
                    axes = ome_meta.image().pixels().DimensionOrder  # e.g., "TCZYX"
                except Exception:
                    axes = None

        if prefer_dask:
            # Option A: use tifffile's OME-as-zarr path if possible
            # This keeps it lazy and chunked without loading into RAM

            if da is None:
                logger.error("The dask package is not installed.")
                raise ImportError("dask[array] not available (install dask)")

            store = tifffile.imread(str(path), aszarr=True)
            darr = da.from_zarr(store, chunks=chunks) if chunks else da.from_zarr(store)
            info = ImageInfo(path=path, shape=tuple(darr.shape), dtype=darr.dtype, axes=axes, metadata={})
            logger.info(f"Loaded {path.name} as a Dask array.")
            return darr, info
        else:
            # Load to memory as NumPy
            arr = tifffile.imread(str(path))
            info = ImageInfo(path=path, shape=tuple(arr.shape), dtype=arr.dtype, axes=axes, metadata={})
            logger.info(f"Loaded {path.name} as NumPy array.")
            return arr, info


class ZarrReader(Reader):
    SUFFIXES = (".zarr",)

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        if zarr is None:
            raise ImportError("zarr is required to read .zarr stores")

        grp = zarr.open_group(str(path), mode="r")
        # Heuristic: if group, pick the first array key; if array, use directly
        # arr_candidate = grp
        # if hasattr(grp, "array_keys") and callable(grp.array_keys) and len(grp.array_keys()) > 0:
        #     first_key = sorted(grp.array_keys())[0]
        #     arr_candidate = grp[first_key]
        print(grp.info)
        print(grp.attrs)
        print(grp.shape)

        # axes = None
        # meta = {}
        # try:
        #     attrs = getattr(arr_candidate, "attrs", {})
        #     axes = attrs.get("multiscales", [{}])[0].get("axes") or attrs.get("axes")
        #     meta = dict(attrs)
        # except Exception:
        #     pass
        #
        # if prefer_dask:
        #     if da is None:
        #         raise ImportError("dask[array] not available (install dask)")
        #     darr = da.from_zarr(arr_candidate, chunks=chunks) if chunks else da.from_zarr(arr_candidate)
        #     info = ImageInfo(path=path, shape=tuple(darr.shape), dtype=darr.dtype, axes=axes, metadata=meta)
        #     return darr, info
        # else:
        #     # Load to memory as NumPy
        #     np_arr = np.array(arr_candidate) if np is not None else arr_candidate[:]
        #     info = ImageInfo(path=path, shape=tuple(np_arr.shape), dtype=np_arr.dtype, axes=axes, metadata=meta)
        #     return np_arr, info


class NpyReader(Reader):
    SUFFIXES = (".npy", ".npz")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        if np is None:
            raise ImportError("numpy is required for .npy/.npz")

        if path.suffix.lower() == ".npy":
            if prefer_dask:
                if da is None:
                    raise ImportError("dask[array] not available (install dask)")
                # memmap keeps it lazy-ish; wrap with dask
                mem = np.load(str(path), mmap_mode="r")
                darr = da.from_array(mem, chunks=chunks or "auto")
                info = ImageInfo(path=path, shape=tuple(darr.shape), dtype=darr.dtype, axes=None, metadata={})
                return darr, info
            else:
                arr = np.load(str(path))
                info = ImageInfo(path=path, shape=tuple(arr.shape), dtype=arr.dtype, axes=None, metadata={})
                return arr, info
        else:
            # .npz: load first array by name (or all if you prefer)
            npz = np.load(str(path))
            first_key = list(npz.keys())[0]
            arr = npz[first_key]
            if prefer_dask:
                if da is None:
                    raise ImportError("dask[array] not available (install dask)")
                darr = da.from_array(arr, chunks=chunks or "auto")
                info = ImageInfo(path=path, shape=tuple(darr.shape), dtype=darr.dtype, axes=None, metadata={"npz_key": first_key})
                return darr, info
            else:
                info = ImageInfo(path=path, shape=tuple(arr.shape), dtype=arr.dtype, axes=None, metadata={"npz_key": first_key})
                return arr, info



class ImageOpener:
    """Generic image opener that selects an appropriate reader."""

    def __init__(
            self,
            readers: Optional[Iterable[Type[Reader]]] = None) -> None:

        # Registry order is priority order
        self._readers: Tuple[Type[Reader], ...] = tuple(
            readers or (TiffReader, ZarrReader, NpyReader)
        )

    def open(
        self,
        path: Union[str, os.PathLike],
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        """Open image file with appropriate reader.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the image file or directory.
        prefer_dask : bool, optional
            If True, attempt to return a Dask array when possible.
            Defaults to False (return NumPy array).
        chunks : int or tuple of int, optional
            Chunk size for Dask arrays. If None, use default chunking.
            Ignored if `prefer_dask` is False. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the reader's `open` method.
        Returns
        -------
        arr : np.ndarray or dask.array.Array
            The loaded image data as a NumPy or Dask array.
        info : ImageInfo
            Metadata about the loaded image.
        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.
        ValueError
            If no suitable reader is found for the file.
        """

        p = Path(path)
        if not p.exists():
            logger.error(f"File {p} does not exist")
            raise FileNotFoundError(p)

        # 1) Extension-based selection
        logger.info(f"Opening {p}")
        for reader_cls in self._readers:
            try:
                if reader_cls.claims(p):
                    reader = reader_cls()
                    logger.info(f"Using reader: {reader_cls.__name__}.")
                    return reader.open(p, prefer_dask=prefer_dask, chunks=chunks, **kwargs)
            except Exception:
                # If a reader "claims" but fails to open, fall through to next
                pass

        # 2) Fallback: probe readers that didn't claim the file (e.g., exotic cases)
        logger.info(f"No suitable reader found for {p}. Attempting fallback readers.")
        for reader_cls in self._readers:
            try:
                reader = reader_cls()
                return reader.open(p, prefer_dask=prefer_dask, chunks=chunks, **kwargs)
            except Exception:
                continue

        logger.error(f"No suitable reader found for {p}")
        raise ValueError(f"No suitable reader found for: {p}")


