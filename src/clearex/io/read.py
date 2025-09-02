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

# Third Party Imports
import numpy as np
import dask.array as da
import tifffile
import zarr

# Local Imports

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

class N5Reader(Reader):
    """Reader for N5 containers."""
    SUFFIXES = (".n5", ".n5/")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        """
        Open an N5 container and return the highest-resolution dataset.

        Parameters
        ----------
        path : Path
            The path to the N5 container (e.g., /path/to/data.n5)
        prefer_dask : bool, optional
            If True, return a Dask array backed by the on-disk dataset.
        chunks : int or tuple of int, optional
            Desired Dask chunking. If None, tries to use the dataset's native chunk/block size.
        **kwargs : dict
            Unused; reserved for future compatibility.

        Returns
        -------
        arr : np.ndarray or dask.array.Array
            The loaded image data as a NumPy or Dask array.
        info : ImageInfo
            Metadata about the loaded image.

        Raises
        ------
        ImportError
            If zarr (or dask when prefer_dask=True) is not installed.
        ValueError
            If the container has no datasets.
        """

        # Normalize path and allow optional dataset subpath after ".n5"
        path_str = str(path)
        if path_str.endswith(os.sep):
            path_str = path_str[:-1]

        # Split potential "container.n5/some/sub/dataset" into (store_path, ds_sub_path)
        store_path, ds_sub_path = path_str, None
        split_idx = path_str.lower().find(".n5")
        if split_idx != -1:
            end = split_idx + len(".n5")
            store_path = path_str[:end]

            # If there's a subpath after ".n5", strip leading slash
            tail = path_str[end:]
            if tail.startswith(os.sep):
                ds_sub_path = tail[len(os.sep):] if len(tail) > 1 else None

        # Open the container
        f = zarr.N5Store(store_path, mode="r")

        # Recursively collect datasets (leaves) under a group-like node
        def _collect_datasets(node):
            items = []
            try:
                for k in node.keys():
                    child = node[k]
                    if hasattr(child, "shape") and hasattr(child, "__getitem__"):
                        items.append(child)
                    else:
                        try:
                            items.extend(_collect_datasets(child))
                        except Exception:
                            # Not a group; ignore
                            pass
            except Exception:
                pass
            return items

        # Choose the dataset: explicit subpath if given, else the largest by voxel count
        if ds_sub_path:
            try:
                ds = f[ds_sub_path]
                if not (hasattr(ds, "shape") and hasattr(ds, "__getitem__")):
                    raise ValueError(f"Path points to a group, not a dataset: {ds_sub_path}")
                datasets = [ds]
            except Exception as e:
                logger.error(f"Dataset subpath not found in N5: {ds_sub_path}")
                raise
        else:
            datasets = _collect_datasets(f)
            if not datasets:
                logger.error(f"No datasets found in N5 container: {store_path}")
                raise ValueError(f"No datasets found in N5 container: {store_path}")

        # Highest resolution ≈ largest number of elements
        dataset = max(datasets, key=lambda d: int(np.prod(d.shape)))

        # Extract basic metadata / axes if present
        axes = None
        meta: Dict[str, Any] = {}
        try:
            attrs = getattr(dataset, "attrs", {})
            meta = dict(attrs) if isinstance(attrs, dict) else {}
            axes = (
                meta.get("axes")
                or meta.get("dimensionNames")            # some tools store this
                or (meta.get("multiscales", [{}])[0].get("axes") if "multiscales" in meta else None)
            )
        except Exception:
            pass

        # Dask or NumPy return
        if prefer_dask:

            # Prefer native block/chunk shape if available
            native_chunks = getattr(dataset, "chunks", None)
            eff_chunks = chunks or native_chunks
            if eff_chunks is None:
                # conservative fallback if chunk info is unavailable
                eff_chunks = tuple(min(128, s) for s in dataset.shape)

            darr = da.from_array(dataset, chunks=eff_chunks, name=f"n5::{getattr(dataset, 'path', 'dataset')}")
            info = ImageInfo(path=Path(store_path), shape=tuple(darr.shape), dtype=darr.dtype, axes=axes, metadata=meta)
            logger.info(f"Loaded {Path(store_path).name} (N5) as Dask array from '{getattr(dataset, 'path', '/')}'.")
            return darr, info

        # Eager NumPy read
        np_arr = dataset[:]  # read all data into memory
        info = ImageInfo(path=Path(store_path), shape=tuple(np_arr.shape), dtype=np_arr.dtype, axes=axes, metadata=meta)
        logger.info(f"Loaded {Path(store_path).name} (N5) as NumPy array from '{getattr(dataset, 'path', '/')}'.")
        return np_arr, info

class ZarrReader(Reader):
    SUFFIXES = (".zarr", ".zarr/")

    def open(
        self,
        path: Path,
        prefer_dask: bool = False,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tuple[ArrayLike, ImageInfo]:
        """Open Zarr file.

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
            Additional keyword arguments passed to zarr.open.
        Returns
        -------
        arr : np.ndarray or dask.array.Array
            The loaded image data as a NumPy or Dask array.
        info : ImageInfo
            Metadata about the loaded image.
        Raises
        ------
        ImportError
            If zarr is not installed.
        ValueError
            If the file cannot be read as a zarr store.
        """

        if zarr is None:
            logger.error("The zarr package is not installed.")
            raise ImportError("zarr is required to read .zarr stores")

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
        meta = {}
        try:
            attrs = getattr(array, "attrs", {})
            axes = attrs.get("multiscales", [{}])[0].get("axes") or attrs.get("axes")
            meta = dict(attrs)
        except Exception:
            pass

        if prefer_dask:
            if da is None:
                logger.error("The dask package is not installed.")
                raise ImportError("The dask package is not installed.")
            darr = da.from_zarr(array, chunks=chunks) if chunks else da.from_zarr(array)
            logger.info(f"Loaded {path.name} as a Dask array.")
            info = ImageInfo(path=path, shape=tuple(darr.shape), dtype=darr.dtype, axes=axes, metadata=meta)
            return darr, info
        else:
            # Load to memory as NumPy
            np_arr = np.array(array) if np is not None else array[:]
            logger.info(f"Loaded {path.name} as a NumPy array.")
            info = ImageInfo(path=path, shape=tuple(np_arr.shape), dtype=np_arr.dtype, axes=axes, metadata=meta)
            return np_arr, info


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
            readers or (TiffReader, ZarrReader, N5Reader, NpyReader)
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
                pass

        # 2) Fallback: probe readers that didn't claim the file
        logger.info(f"No suitable reader found for {p}. Attempting fallback readers.")
        for reader_cls in self._readers:
            try:
                reader = reader_cls()
                return reader.open(p, prefer_dask=prefer_dask, chunks=chunks, **kwargs)
            except Exception:
                continue

        logger.error(f"No suitable reader found for {p}")
        raise ValueError("No suitable reader found for:", p)


