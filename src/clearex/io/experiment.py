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

"""Navigate experiment ingestion and 6D Zarr analysis-store utilities."""

from __future__ import annotations

# Standard Library Imports
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
import json
import re
import warnings
import xml.etree.ElementTree as ET

# Third Party Imports
import dask
import dask.array as da
import h5py
import numpy as np
import tifffile
import zarr

# Local Imports
from clearex.io.read import ImageInfo

if TYPE_CHECKING:
    from dask.delayed import Delayed
    from dask.distributed import Client

# YAML parsing is optional because many Navigate ``experiment.yml`` files are JSON.
try:
    import yaml

    HAS_PYYAML = True
except Exception:
    HAS_PYYAML = False


ArrayLike = Union[np.ndarray, da.Array]
AxesSpec = Optional[tuple[str, ...]]
CanonicalShapeTpczyx = tuple[int, int, int, int, int, int]
ProgressCallback = Callable[[int, str], None]


def _is_zarr_like_path(path: Path) -> bool:
    """Return whether a path is a Zarr or N5 directory store.

    Parameters
    ----------
    path : pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` when ``path`` is a directory with ``.zarr`` or ``.n5`` suffix.
    """
    return path.is_dir() and path.suffix.lower() in {".zarr", ".n5"}


def _normalize_axis_token(token: Any) -> Optional[str]:
    """Normalize a single axis descriptor into canonical one-letter form.

    Parameters
    ----------
    token : Any
        Axis token candidate.

    Returns
    -------
    str, optional
        Canonical axis token in ``{"t", "p", "c", "z", "y", "x"}``, or
        ``None`` when the token cannot be mapped.
    """
    if token is None:
        return None
    if isinstance(token, bytes):
        text = token.decode("utf-8", errors="ignore")
    else:
        text = str(token)
    normalized = text.strip().lower()
    if not normalized:
        return None

    aliases = {
        "t": "t",
        "time": "t",
        "c": "c",
        "ch": "c",
        "channel": "c",
        "z": "z",
        "y": "y",
        "x": "x",
        "p": "p",
        "s": "p",
        "series": "p",
        "position": "p",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized[0] in aliases:
        return aliases[normalized[0]]
    return None


def _normalize_axes_descriptor(axes: Any, *, ndim: int) -> AxesSpec:
    """Normalize axis metadata into ordered canonical axis tokens.

    Parameters
    ----------
    axes : Any
        Axis descriptor from source metadata. Supported forms include strings,
        lists of strings, and OME-Zarr axis dictionaries with ``name`` keys.
    ndim : int
        Expected number of dimensions in the source array.

    Returns
    -------
    tuple of str, optional
        Normalized axis tokens matching ``ndim``, or ``None`` when metadata is
        missing or incompatible.
    """
    tokens: list[Any]
    if axes is None:
        return None
    if isinstance(axes, str):
        tokens = list(axes)
    elif isinstance(axes, (tuple, list)):
        tokens = []
        for axis in axes:
            if isinstance(axis, dict):
                tokens.append(axis.get("name"))
            else:
                tokens.append(axis)
    else:
        return None

    if len(tokens) != ndim:
        return None

    normalized = tuple(_normalize_axis_token(token) for token in tokens)
    if any(token is None for token in normalized):
        return None
    return tuple(str(token) for token in normalized)


def _extract_zarr_axes(array: Any, group_attrs: dict[str, Any]) -> AxesSpec:
    """Extract and normalize axis metadata from a Zarr array/group.

    Parameters
    ----------
    array : Any
        Zarr array-like object.
    group_attrs : dict[str, Any]
        Parent group attributes.

    Returns
    -------
    tuple of str, optional
        Normalized source axes, when present.
    """
    attrs = dict(getattr(array, "attrs", {}))
    raw_axes = (
        attrs.get("multiscales", [{}])[0].get("axes")
        or group_attrs.get("multiscales", [{}])[0].get("axes")
        or attrs.get("_ARRAY_DIMENSIONS")
        or group_attrs.get("_ARRAY_DIMENSIONS")
        or attrs.get("axes")
        or group_attrs.get("axes")
    )
    return _normalize_axes_descriptor(raw_axes, ndim=len(tuple(array.shape)))


def _infer_source_axes(
    shape: tuple[int, ...], experiment: NavigateExperiment
) -> tuple[str, ...]:
    """Infer source axis ordering when metadata is unavailable.

    Parameters
    ----------
    shape : tuple[int, ...]
        Source array shape.
    experiment : NavigateExperiment
        Parsed experiment metadata used for weak disambiguation.

    Returns
    -------
    tuple[str, ...]
        Inferred axis tokens corresponding to ``shape``.

    Raises
    ------
    ValueError
        If the source dimensionality is unsupported.
    """
    ndim = len(shape)
    if ndim == 2:
        return ("y", "x")
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 4:
        if shape[0] == experiment.timepoints and shape[1] == experiment.number_z_steps:
            return ("t", "z", "y", "x")
        if (
            shape[0] == experiment.number_z_steps
            and shape[1] == experiment.channel_count
        ):
            return ("z", "c", "y", "x")
        return ("c", "z", "y", "x")
    if ndim == 5:
        if (
            shape[0] == experiment.multiposition_count
            and shape[1] == experiment.channel_count
        ):
            return ("p", "c", "z", "y", "x")
        return ("t", "c", "z", "y", "x")
    if ndim == 6:
        return ("t", "p", "c", "z", "y", "x")
    raise ValueError(
        "Unsupported source dimensionality "
        f"{ndim}; expected 2D-6D arrays for ingestion."
    )


def _coerce_to_tpczyx(
    source: da.Array,
    *,
    experiment: NavigateExperiment,
    source_axes: AxesSpec = None,
) -> da.Array:
    """Convert source data into canonical ``(t, p, c, z, y, x)`` ordering.

    Parameters
    ----------
    source : dask.array.Array
        Source image data.
    experiment : NavigateExperiment
        Parsed experiment metadata.
    source_axes : tuple[str, ...], optional
        Source axis tokens. If omitted, axis order is inferred from shape.

    Returns
    -------
    dask.array.Array
        View of ``source`` reordered/expanded to ``(t, p, c, z, y, x)``.

    Raises
    ------
    ValueError
        If source axis metadata is inconsistent or missing required spatial
        dimensions.
    """
    canonical_axes = ["t", "p", "c", "z", "y", "x"]
    axes = tuple(source_axes or _infer_source_axes(tuple(source.shape), experiment))
    if len(axes) != source.ndim:
        raise ValueError(
            f"Source axes length ({len(axes)}) does not match ndim ({source.ndim})."
        )
    if not {"y", "x"}.issubset(set(axes)):
        raise ValueError(f"Source axes must include y/x dimensions, got {axes}.")

    if len(set(axes)) != len(axes):
        raise ValueError(f"Source axes contain duplicates: {axes}.")

    present_axes = [axis for axis in canonical_axes if axis in axes]
    perm = [axes.index(axis) for axis in present_axes]
    reordered = source.transpose(tuple(perm))

    current_axes = list(present_axes)
    for idx, axis in enumerate(canonical_axes):
        if axis in current_axes:
            continue
        reordered = da.expand_dims(reordered, axis=idx)
        current_axes.insert(idx, axis)

    return reordered


def _collect_largest_zarr_array(store_path: Path) -> tuple[Any, str, AxesSpec]:
    """Open a Zarr/N5 store and select the largest contained array.

    Parameters
    ----------
    store_path : pathlib.Path
        Source Zarr or N5 store path.

    Returns
    -------
    tuple
        ``(array, component, axes)`` where ``component`` is the selected array
        path relative to the store root.

    Raises
    ------
    ValueError
        If no arrays are found in the store.
    """
    root_object = zarr.open(str(store_path), mode="r")
    if hasattr(root_object, "shape") and not hasattr(root_object, "array_keys"):
        axes = _extract_zarr_axes(root_object, {})
        return root_object, "", axes

    group = zarr.open_group(str(store_path), mode="r")
    group_attrs = dict(getattr(group, "attrs", {}))
    arrays: list[tuple[str, Any]] = []

    def _walk(group_node: Any, prefix: str = "") -> None:
        if hasattr(group_node, "array_keys") and callable(group_node.array_keys):
            for key in sorted(group_node.array_keys()):
                arrays.append((f"{prefix}{key}", group_node[key]))
        if hasattr(group_node, "group_keys") and callable(group_node.group_keys):
            for key in sorted(group_node.group_keys()):
                _walk(group_node[key], f"{prefix}{key}/")

    _walk(group)
    if not arrays:
        raise ValueError(f"No arrays found in Zarr/N5 store: {store_path}")

    arrays.sort(key=lambda item: (-int(np.prod(item[1].shape)), item[0]))
    component, array = arrays[0]
    axes = _extract_zarr_axes(array, group_attrs)
    return array, component, axes


def _open_source_as_dask(
    source_path: Path, *, exit_stack: ExitStack
) -> tuple[da.Array, AxesSpec, dict[str, Any]]:
    """Open source image data as a Dask array using format-specific fast paths.

    Parameters
    ----------
    source_path : pathlib.Path
        Source image path.
    exit_stack : contextlib.ExitStack
        Exit stack used to manage open file handles for formats that require
        persistent references (for example HDF5).

    Returns
    -------
    tuple
        ``(source_array, source_axes, metadata)``.

    Raises
    ------
    ValueError
        If the source format is unsupported.
    """
    suffix = source_path.suffix.lower()
    meta: dict[str, Any] = {"source_path": str(source_path)}

    if _is_zarr_like_path(source_path):
        array, component, axes = _collect_largest_zarr_array(source_path)
        source_array = (
            da.from_zarr(str(source_path))
            if not component
            else da.from_zarr(str(source_path), component=component)
        )
        meta["source_component"] = component or "."
        return source_array, axes, meta

    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(str(source_path)) as tf:
            series = tf.series[0]
            axes = _normalize_axes_descriptor(
                getattr(series, "axes", None), ndim=len(tuple(series.shape))
            )
        tiff_store = tifffile.imread(str(source_path), aszarr=True)
        source_array = da.from_zarr(tiff_store)
        return source_array, axes, meta

    if suffix in {".h5", ".hdf5", ".hdf"}:
        h5_file = exit_stack.enter_context(h5py.File(str(source_path), mode="r"))
        datasets: list[h5py.Dataset] = []

        def _collect_datasets(group: h5py.Group) -> None:
            for _, obj in group.items():
                if isinstance(obj, h5py.Dataset):
                    datasets.append(obj)
                elif isinstance(obj, h5py.Group):
                    _collect_datasets(obj)

        _collect_datasets(h5_file)
        if not datasets:
            raise ValueError(f"No datasets found in HDF5 file: {source_path}")

        dataset = max(datasets, key=lambda item: int(np.prod(item.shape)))
        raw_axes = (
            dataset.attrs.get("axes")
            or dataset.attrs.get("dimension_order")
            or dataset.attrs.get("DimensionOrder")
            or dataset.attrs.get("DIMENSION_LABELS")
        )
        axes = _normalize_axes_descriptor(raw_axes, ndim=len(tuple(dataset.shape)))
        chunks = dataset.chunks or tuple(min(128, int(size)) for size in dataset.shape)
        source_array = da.from_array(dataset, chunks=chunks, lock=True)
        meta["source_component"] = dataset.name
        return source_array, axes, meta

    if suffix == ".npy":
        mmap = np.load(str(source_path), mmap_mode="r")
        source_array = da.from_array(mmap, chunks="auto")
        return source_array, None, meta

    if suffix == ".npz":
        npz_file = exit_stack.enter_context(np.load(str(source_path)))
        keys = sorted(npz_file.keys())
        if not keys:
            raise ValueError(f"No arrays found in NPZ file: {source_path}")
        key = keys[0]
        source_array = da.from_array(npz_file[key], chunks="auto")
        meta["source_component"] = key
        return source_array, None, meta

    raise ValueError(f"Unsupported source format for ingestion: {source_path}")


def _parse_navigate_bdv_setup_index_map(
    source_path: Path,
) -> Optional[dict[int, tuple[int, int]]]:
    """Parse Navigate BDV XML setup metadata into position/channel indices.

    Parameters
    ----------
    source_path : pathlib.Path
        Source BDV data path (``.h5``/``.hdf5``/``.hdf`` or ``.n5``).

    Returns
    -------
    dict[int, tuple[int, int]], optional
        Mapping from setup index to ``(position_index, channel_index)``.
        Returns ``None`` when companion XML metadata is unavailable or
        incompatible.

    Raises
    ------
    None
        Parsing is best-effort and falls back to ``None``.
    """
    suffix = source_path.suffix.lower()
    if suffix not in {".h5", ".hdf5", ".hdf", ".n5"}:
        return None

    xml_path = source_path.with_suffix(".xml")
    if not xml_path.exists():
        return None

    try:
        root = ET.fromstring(xml_path.read_text())
    except Exception:
        return None

    image_loader = root.find("SequenceDescription/ImageLoader")
    if image_loader is None:
        return None
    image_format = str(image_loader.attrib.get("format", "")).strip().lower()

    if suffix in {".h5", ".hdf5", ".hdf"} and image_format != "bdv.hdf5":
        return None
    if suffix == ".n5" and image_format != "bdv.n5":
        return None

    raw_entries: list[tuple[int, int, int]] = []
    for view_setup in root.findall("SequenceDescription/ViewSetups/ViewSetup"):
        setup_text = view_setup.findtext("id")
        channel_text = view_setup.findtext("attributes/channel")
        tile_text = view_setup.findtext("attributes/tile")
        if setup_text is None or channel_text is None or tile_text is None:
            continue
        try:
            setup_index = int(setup_text)
            channel_index = int(channel_text)
            tile_index = int(tile_text)
        except ValueError:
            continue
        raw_entries.append((setup_index, channel_index, tile_index))

    if not raw_entries:
        return None

    channel_values = sorted({entry[1] for entry in raw_entries})
    tile_values = sorted({entry[2] for entry in raw_entries})
    channel_lookup = {value: idx for idx, value in enumerate(channel_values)}
    tile_lookup = {value: idx for idx, value in enumerate(tile_values)}

    return {
        int(setup_index): (
            int(tile_lookup[tile_index]),
            int(channel_lookup[channel_index]),
        )
        for setup_index, channel_index, tile_index in raw_entries
    }


def _open_navigate_bdv_collection_as_dask(
    *,
    experiment: "NavigateExperiment",
    source_path: Path,
    exit_stack: ExitStack,
) -> Optional[tuple[da.Array, AxesSpec, dict[str, Any]]]:
    """Open Navigate BDV H5/N5 acquisitions as stacked ``(t, p, c, ...)``.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    source_path : pathlib.Path
        Source acquisition path.
    exit_stack : contextlib.ExitStack
        Exit stack for file-handle lifecycle management.

    Returns
    -------
    tuple, optional
        ``(source_array, source_axes, metadata)`` for BDV collection mode,
        or ``None`` when the source does not match BDV collection patterns.

    Raises
    ------
    ValueError
        If indexed source arrays have inconsistent shape/axes or contain
        missing position/channel combinations.
    """
    suffix = source_path.suffix.lower()
    is_h5 = suffix in {".h5", ".hdf5", ".hdf"}
    is_n5 = suffix == ".n5" and _is_zarr_like_path(source_path)
    if not is_h5 and not is_n5:
        return None

    setup_map = _parse_navigate_bdv_setup_index_map(source_path)
    inferred_positions = max(1, int(experiment.multiposition_count))

    arrays_by_index: dict[tuple[int, int, int], da.Array] = {}
    base_axes: AxesSpec = None
    base_shape: Optional[tuple[int, ...]] = None

    if is_h5:
        h5_file = exit_stack.enter_context(h5py.File(str(source_path), mode="r"))
        entries: list[tuple[int, int, h5py.Dataset]] = []

        def _collect_h5_entries(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            match = re.match(r"^t(\d+)/s(\d+)/(\d+)/cells$", name)
            if match is None:
                return
            if int(match.group(3)) != 0:
                return
            entries.append((int(match.group(1)), int(match.group(2)), obj))

        h5_file.visititems(_collect_h5_entries)
        if len(entries) <= 1:
            return None

        for time_index, setup_index, dataset in sorted(
            entries, key=lambda item: (item[0], item[1])
        ):
            if setup_map is not None and setup_index not in setup_map:
                continue
            if setup_map is not None:
                position_index, channel_index = setup_map[setup_index]
            else:
                channel_index = int(setup_index // inferred_positions)
                position_index = int(setup_index % inferred_positions)

            key = (int(time_index), int(position_index), int(channel_index))
            if key in arrays_by_index:
                continue

            raw_axes = (
                dataset.attrs.get("axes")
                or dataset.attrs.get("dimension_order")
                or dataset.attrs.get("DimensionOrder")
                or dataset.attrs.get("DIMENSION_LABELS")
            )
            source_axes = _normalize_axes_descriptor(
                raw_axes, ndim=len(tuple(dataset.shape))
            )
            source_array = da.from_array(
                dataset,
                chunks=dataset.chunks
                or tuple(min(128, int(size)) for size in dataset.shape),
                lock=True,
            )
            normalized_axes = tuple(
                source_axes or _infer_source_axes(tuple(source_array.shape), experiment)
            )
            if any(axis in {"t", "p", "c"} for axis in normalized_axes):
                return None
            source_shape = tuple(int(size) for size in source_array.shape)
            if base_axes is None:
                base_axes = normalized_axes
                base_shape = source_shape
            else:
                if normalized_axes != base_axes:
                    raise ValueError(
                        "Navigate BDV H5 collection has inconsistent source axes: "
                        f"expected {base_axes}, got {normalized_axes} at "
                        f"t={time_index}, setup={setup_index}."
                    )
                if source_shape != base_shape:
                    raise ValueError(
                        "Navigate BDV H5 collection has inconsistent source shapes: "
                        f"expected {base_shape}, got {source_shape} at "
                        f"t={time_index}, setup={setup_index}."
                    )
            arrays_by_index[key] = source_array
    else:
        root = zarr.open_group(str(source_path), mode="r")
        group_attrs = dict(getattr(root, "attrs", {}))
        entries: list[tuple[int, int, str, AxesSpec]] = []

        def _walk(group_node: Any, prefix: str = "") -> None:
            for key in sorted(group_node.array_keys()):
                component = f"{prefix}{key}"
                match = re.match(r"^setup(\d+)/timepoint(\d+)/s(\d+)$", component)
                if match is None or int(match.group(3)) != 0:
                    continue
                array = group_node[key]
                entries.append(
                    (
                        int(match.group(2)),
                        int(match.group(1)),
                        component,
                        _extract_zarr_axes(array, group_attrs),
                    )
                )
            for key in sorted(group_node.group_keys()):
                _walk(group_node[key], f"{prefix}{key}/")

        _walk(root)
        if len(entries) <= 1:
            return None

        for time_index, setup_index, component, source_axes in sorted(
            entries, key=lambda item: (item[0], item[1], item[2])
        ):
            if setup_map is not None and setup_index not in setup_map:
                continue
            if setup_map is not None:
                position_index, channel_index = setup_map[setup_index]
            else:
                channel_index = int(setup_index // inferred_positions)
                position_index = int(setup_index % inferred_positions)

            key = (int(time_index), int(position_index), int(channel_index))
            if key in arrays_by_index:
                continue

            source_array = da.from_zarr(str(source_path), component=component)
            normalized_axes = tuple(
                source_axes or _infer_source_axes(tuple(source_array.shape), experiment)
            )
            if any(axis in {"t", "p", "c"} for axis in normalized_axes):
                return None
            source_shape = tuple(int(size) for size in source_array.shape)
            if base_axes is None:
                base_axes = normalized_axes
                base_shape = source_shape
            else:
                if normalized_axes != base_axes:
                    raise ValueError(
                        "Navigate BDV N5 collection has inconsistent source axes: "
                        f"expected {base_axes}, got {normalized_axes} at "
                        f"t={time_index}, setup={setup_index}."
                    )
                if source_shape != base_shape:
                    raise ValueError(
                        "Navigate BDV N5 collection has inconsistent source shapes: "
                        f"expected {base_shape}, got {source_shape} at "
                        f"t={time_index}, setup={setup_index}."
                    )
            arrays_by_index[key] = source_array

    if len(arrays_by_index) <= 1:
        return None

    time_indices = sorted({int(key[0]) for key in arrays_by_index})
    position_indices = sorted({int(key[1]) for key in arrays_by_index})
    channel_indices = sorted({int(key[2]) for key in arrays_by_index})

    missing: list[tuple[int, int, int]] = []
    stacked_by_time: list[da.Array] = []
    for time_index in time_indices:
        stacked_by_position: list[da.Array] = []
        for position_index in position_indices:
            stacked_by_channel: list[da.Array] = []
            for channel_index in channel_indices:
                key = (time_index, position_index, channel_index)
                array = arrays_by_index.get(key)
                if array is None:
                    missing.append(key)
                    continue
                stacked_by_channel.append(array)
            if missing:
                continue
            stacked_by_position.append(da.stack(stacked_by_channel, axis=0))
        if missing:
            continue
        stacked_by_time.append(da.stack(stacked_by_position, axis=0))

    if missing:
        preview = ", ".join(
            f"(t={time_idx}, p={position_idx}, c={channel_idx})"
            for time_idx, position_idx, channel_idx in missing[:8]
        )
        raise ValueError(
            "Navigate BDV collection is missing expected position/channel "
            f"combinations: {preview}."
        )

    source_array = da.stack(stacked_by_time, axis=0)
    source_axes = ("t", "p", "c", *(base_axes or ()))
    metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "source_file_count": int(len(arrays_by_index)),
        "source_collection_time_indices": [int(value) for value in time_indices],
        "source_collection_position_indices": [
            int(value) for value in position_indices
        ],
        "source_collection_channel_indices": [int(value) for value in channel_indices],
        "source_collection_type": "bdv_h5" if is_h5 else "bdv_n5",
    }
    return source_array, source_axes, metadata


def _parse_navigate_tiff_indices(path: Path) -> tuple[int, int, int]:
    """Parse ``(time, position, channel)`` indices from a Navigate TIFF path.

    Parameters
    ----------
    path : pathlib.Path
        TIFF file path from a Navigate acquisition directory.

    Returns
    -------
    tuple[int, int, int]
        Parsed index tuple in ``(t, p, c)`` order. Missing values default to
        ``0``.

    Raises
    ------
    None
        Parsing is best-effort and falls back to ``0`` for missing indices.
    """
    stem = path.stem

    position_index = 0
    for part in reversed(path.parts):
        match = re.fullmatch(r"Position(\d+)", part, flags=re.IGNORECASE)
        if match is not None:
            position_index = int(match.group(1))
            break
    else:
        match = re.search(r"(?:^|[_-])P(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
        if match is not None:
            position_index = int(match.group(1))

    channel_index = 0
    match = re.search(r"CH(\d+)", stem, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(?:^|[_-])C(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
    if match is not None:
        channel_index = int(match.group(1))

    time_index = 0
    match = re.search(r"CH\d+_(\d+)", stem, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(?:^|[_-])T(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
    if match is not None:
        time_index = int(match.group(1))

    return time_index, position_index, channel_index


def _open_navigate_tiff_collection_as_dask(
    *,
    experiment: "NavigateExperiment",
    exit_stack: ExitStack,
) -> Optional[tuple[da.Array, AxesSpec, dict[str, Any]]]:
    """Open Navigate TIFF acquisitions as one Dask array over ``(t, p, c)``.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    exit_stack : contextlib.ExitStack
        Exit stack passed through to source-open helpers.

    Returns
    -------
    tuple, optional
        ``(source_array, source_axes, metadata)`` for collection-based TIFF
        ingestion, or ``None`` when the layout is not a multi-file
        position/channel collection.

    Raises
    ------
    ValueError
        If the TIFF collection has incompatible array shapes/axes or missing
        position-channel combinations.
    """
    if _normalize_file_type(experiment.file_type) != "TIFF":
        return None

    candidates = [
        path.resolve()
        for path in find_experiment_data_candidates(experiment)
        if path.suffix.lower() in {".tif", ".tiff"}
    ]
    if len(candidates) <= 1:
        return None

    indexed_paths: dict[tuple[int, int, int], Path] = {}
    for path in candidates:
        key = _parse_navigate_tiff_indices(path)
        indexed_paths.setdefault(key, path)

    if len(indexed_paths) <= 1:
        return None

    time_indices = sorted({int(key[0]) for key in indexed_paths})
    position_indices = sorted({int(key[1]) for key in indexed_paths})
    channel_indices = sorted({int(key[2]) for key in indexed_paths})

    arrays_by_index: dict[tuple[int, int, int], da.Array] = {}
    base_axes: AxesSpec = None
    base_shape: Optional[tuple[int, ...]] = None
    for key in sorted(indexed_paths):
        source_array, source_axes, _ = _open_source_as_dask(
            indexed_paths[key],
            exit_stack=exit_stack,
        )
        normalized_axes = tuple(
            source_axes or _infer_source_axes(tuple(source_array.shape), experiment)
        )
        if any(axis in {"t", "p", "c"} for axis in normalized_axes):
            return None

        source_shape = tuple(int(size) for size in source_array.shape)
        if base_axes is None:
            base_axes = normalized_axes
            base_shape = source_shape
        else:
            if normalized_axes != base_axes:
                raise ValueError(
                    "Navigate TIFF collection has inconsistent source axes: "
                    f"expected {base_axes}, got {normalized_axes} at "
                    f"{indexed_paths[key]}."
                )
            if source_shape != base_shape:
                raise ValueError(
                    "Navigate TIFF collection has inconsistent source shapes: "
                    f"expected {base_shape}, got {source_shape} at "
                    f"{indexed_paths[key]}."
                )
        arrays_by_index[key] = source_array

    missing: list[tuple[int, int, int]] = []
    stacked_by_time: list[da.Array] = []
    for time_index in time_indices:
        stacked_by_position: list[da.Array] = []
        for position_index in position_indices:
            stacked_by_channel: list[da.Array] = []
            for channel_index in channel_indices:
                key = (time_index, position_index, channel_index)
                array = arrays_by_index.get(key)
                if array is None:
                    missing.append(key)
                    continue
                stacked_by_channel.append(array)
            if missing:
                continue
            stacked_by_position.append(da.stack(stacked_by_channel, axis=0))
        if missing:
            continue
        stacked_by_time.append(da.stack(stacked_by_position, axis=0))

    if missing:
        preview = ", ".join(
            f"(t={time_idx}, p={position_idx}, c={channel_idx})"
            for time_idx, position_idx, channel_idx in missing[:8]
        )
        raise ValueError(
            "Navigate TIFF collection is missing expected position/channel "
            f"combinations: {preview}."
        )

    source_array = da.stack(stacked_by_time, axis=0)
    source_axes = ("t", "p", "c", *(base_axes or ()))
    metadata: dict[str, Any] = {
        "source_path": str(experiment.save_directory),
        "source_file_count": int(len(indexed_paths)),
        "source_collection_time_indices": [int(value) for value in time_indices],
        "source_collection_position_indices": [
            int(value) for value in position_indices
        ],
        "source_collection_channel_indices": [int(value) for value in channel_indices],
    }
    return source_array, source_axes, metadata


def _format_axes_for_image_info(axes: AxesSpec) -> Optional[str]:
    """Format normalized axis tokens as an uppercase axis string.

    Parameters
    ----------
    axes : tuple[str, ...], optional
        Normalized lowercase axis tokens.

    Returns
    -------
    str, optional
        Uppercase axis string suitable for :class:`clearex.io.read.ImageInfo`,
        or ``None`` when ``axes`` is not available.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    if axes is None:
        return None
    return "".join(token.upper() for token in axes)


def _normalize_tpczyx_shape(shape: tuple[int, ...]) -> CanonicalShapeTpczyx:
    """Validate and normalize canonical ``(t, p, c, z, y, x)`` shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        Candidate shape tuple.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Normalized canonical shape.

    Raises
    ------
    ValueError
        If the shape does not define exactly six positive dimensions.
    """
    if len(shape) != 6:
        raise ValueError(
            "Canonical data shape must define exactly six dimensions "
            "(t, p, c, z, y, x)."
        )
    normalized = tuple(int(size) for size in shape)
    if any(size <= 0 for size in normalized):
        raise ValueError(
            f"Canonical data shape values must be greater than zero; got {normalized}."
        )
    return (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
        normalized[4],
        normalized[5],
    )


def _normalize_write_chunks(
    shape_tpczyx: CanonicalShapeTpczyx,
    chunks: tuple[int, int, int, int, int, int],
) -> CanonicalShapeTpczyx:
    """Normalize requested chunk sizes against target canonical shape.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Target canonical shape.
    chunks : tuple[int, int, int, int, int, int]
        Requested chunk sizes.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Effective chunk sizes with per-axis values clipped to array bounds.

    Raises
    ------
    ValueError
        If chunk specification is not six positive integers.
    """
    if len(chunks) != 6:
        raise ValueError("chunks must define six values in (t, p, c, z, y, x) order.")
    requested = tuple(int(chunk) for chunk in chunks)
    if any(chunk <= 0 for chunk in requested):
        raise ValueError("chunks values must be greater than zero.")
    normalized = tuple(
        min(int(chunk), int(dim))
        for chunk, dim in zip(requested, shape_tpczyx, strict=False)
    )
    return (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
        normalized[4],
        normalized[5],
    )


def _normalize_pyramid_level_factors(
    pyramid_factors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
) -> tuple[CanonicalShapeTpczyx, ...]:
    """Normalize per-axis pyramid factors into concrete per-level factors.

    Parameters
    ----------
    pyramid_factors : tuple[tuple[int, ...], ...]
        Per-axis factors in ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...]
        Per-level downsampling factors. Level ``0`` is always
        ``(1, 1, 1, 1, 1, 1)``.

    Raises
    ------
    ValueError
        If factor definitions are missing, malformed, or invalid.

    Notes
    -----
    If axes define different numbers of levels, shorter axes repeat their last
    factor for deeper levels so all levels resolve to complete 6D factor tuples.
    """
    axis_names = ("t", "p", "c", "z", "y", "x")
    if len(pyramid_factors) != len(axis_names):
        raise ValueError(
            "pyramid_factors must define six axis entries in (t, p, c, z, y, x) order."
        )

    normalized_axes: list[tuple[int, ...]] = []
    for axis_name, axis_levels in zip(axis_names, pyramid_factors, strict=False):
        parsed_levels = tuple(int(level) for level in axis_levels)
        if not parsed_levels:
            raise ValueError(f"pyramid_factors for axis '{axis_name}' cannot be empty.")
        if any(level <= 0 for level in parsed_levels):
            raise ValueError(
                f"pyramid_factors for axis '{axis_name}' must be greater than zero."
            )
        if parsed_levels[0] != 1:
            raise ValueError(
                f"pyramid_factors for axis '{axis_name}' must start with 1."
            )
        normalized_axes.append(parsed_levels)

    max_levels = max(len(levels) for levels in normalized_axes)
    levels: list[CanonicalShapeTpczyx] = []
    for level_index in range(max_levels):
        factors: CanonicalShapeTpczyx = (
            int(normalized_axes[0][min(level_index, len(normalized_axes[0]) - 1)]),
            int(normalized_axes[1][min(level_index, len(normalized_axes[1]) - 1)]),
            int(normalized_axes[2][min(level_index, len(normalized_axes[2]) - 1)]),
            int(normalized_axes[3][min(level_index, len(normalized_axes[3]) - 1)]),
            int(normalized_axes[4][min(level_index, len(normalized_axes[4]) - 1)]),
            int(normalized_axes[5][min(level_index, len(normalized_axes[5]) - 1)]),
        )
        if levels and factors == levels[-1]:
            continue
        levels.append(factors)

    return tuple(levels)


def _downsample_tpczyx_by_stride(
    array: da.Array,
    factors_tpczyx: CanonicalShapeTpczyx,
) -> da.Array:
    """Create a strided downsampled pyramid level from a canonical array.

    Parameters
    ----------
    array : dask.array.Array
        Source canonical array in ``(t, p, c, z, y, x)`` order.
    factors_tpczyx : tuple[int, int, int, int, int, int]
        Integer stride factors for each canonical axis.

    Returns
    -------
    dask.array.Array
        Downsampled view with nearest-neighbor decimation.

    Raises
    ------
    ValueError
        If ``factors_tpczyx`` does not define six positive integers.
    """
    if len(factors_tpczyx) != 6:
        raise ValueError(
            "factors_tpczyx must define six values in (t, p, c, z, y, x) order."
        )
    if any(int(factor) <= 0 for factor in factors_tpczyx):
        raise ValueError("factors_tpczyx values must be greater than zero.")

    slices = tuple(slice(None, None, int(factor)) for factor in factors_tpczyx)
    return array[slices]


def _derive_pyramid_level_chunks(
    *,
    base_chunks_tpczyx: CanonicalShapeTpczyx,
    level_shape_tpczyx: CanonicalShapeTpczyx,
    level_factors_tpczyx: CanonicalShapeTpczyx,
) -> CanonicalShapeTpczyx:
    """Derive chunk sizes for one pyramid level from base chunks and factors.

    Parameters
    ----------
    base_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Base-level chunk shape in canonical order.
    level_shape_tpczyx : tuple[int, int, int, int, int, int]
        Target level shape.
    level_factors_tpczyx : tuple[int, int, int, int, int, int]
        Absolute level downsampling factors in canonical order.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Normalized chunk sizes for the level.

    Raises
    ------
    ValueError
        If any provided tuple is malformed or contains non-positive values.
    """
    requested_chunks: CanonicalShapeTpczyx = (
        max(1, int(base_chunks_tpczyx[0]) // int(level_factors_tpczyx[0])),
        max(1, int(base_chunks_tpczyx[1]) // int(level_factors_tpczyx[1])),
        max(1, int(base_chunks_tpczyx[2]) // int(level_factors_tpczyx[2])),
        max(1, int(base_chunks_tpczyx[3]) // int(level_factors_tpczyx[3])),
        max(1, int(base_chunks_tpczyx[4]) // int(level_factors_tpczyx[4])),
        max(1, int(base_chunks_tpczyx[5]) // int(level_factors_tpczyx[5])),
    )
    return _normalize_write_chunks(level_shape_tpczyx, requested_chunks)


def _materialize_data_pyramid(
    *,
    store_path: Path,
    base_chunks_tpczyx: CanonicalShapeTpczyx,
    pyramid_factors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 60,
    progress_end: int = 96,
) -> list[str]:
    """Build and persist downsampled Zarr pyramid levels in canonical store.

    Parameters
    ----------
    store_path : pathlib.Path
        Target Zarr store containing canonical ``data`` array.
    base_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective base chunking in canonical order.
    pyramid_factors : tuple[tuple[int, ...], ...]
        Per-axis pyramid factors in ``(t, p, c, z, y, x)`` order.
    client : dask.distributed.Client, optional
        Active Dask client used for parallel writes.
    progress_callback : callable, optional
        Optional callback invoked as ``callback(percent, message)``.
    progress_start : int, default=60
        Start percentage for pyramid generation progress.
    progress_end : int, default=96
        End percentage for pyramid generation progress.

    Returns
    -------
    list[str]
        Ordered component paths including base level at index ``0``.

    Raises
    ------
    ValueError
        If canonical base data or pyramid configuration is invalid.

    Notes
    -----
    Levels are stored under ``data_pyramid/level_<n>`` where ``n`` starts at 1.
    Downsampling uses stride-based nearest-neighbor decimation for speed and
    deterministic dtype preservation.
    """
    level_factors = _normalize_pyramid_level_factors(pyramid_factors)
    root = zarr.open_group(str(store_path), mode="a")
    if "data" not in root:
        raise ValueError(f"Expected canonical data array at {store_path}/data.")

    if "data_pyramid" in root:
        del root["data_pyramid"]
    root.require_group("data_pyramid")

    base_shape = _normalize_tpczyx_shape(
        tuple(int(size) for size in root["data"].shape)
    )
    level_paths = ["data"]
    level_factor_payload = [[int(value) for value in level_factors[0]]]
    level_shapes_payload = [[int(value) for value in base_shape]]
    total_downsample_levels = max(0, len(level_factors) - 1)

    if total_downsample_levels == 0:
        root["data"].attrs.update(
            {
                "pyramid_levels": level_paths,
                "pyramid_factors_tpczyx": level_factor_payload,
            }
        )
        root.attrs.update(
            {
                "data_pyramid_levels": level_paths,
                "data_pyramid_factors_tpczyx": level_factor_payload,
                "data_pyramid_shapes_tpczyx": level_shapes_payload,
            }
        )
        if progress_callback is not None:
            progress_callback(
                int(progress_end), "Pyramid configuration has only base level"
            )
        return level_paths

    prior_component = "data"
    prior_factors = level_factors[0]
    for level_index, absolute_factors in enumerate(level_factors[1:], start=1):
        all_relative = all(
            int(current) % int(previous) == 0
            for current, previous in zip(absolute_factors, prior_factors, strict=False)
        )
        if all_relative:
            relative_factors: CanonicalShapeTpczyx = (
                int(absolute_factors[0] // prior_factors[0]),
                int(absolute_factors[1] // prior_factors[1]),
                int(absolute_factors[2] // prior_factors[2]),
                int(absolute_factors[3] // prior_factors[3]),
                int(absolute_factors[4] // prior_factors[4]),
                int(absolute_factors[5] // prior_factors[5]),
            )
            source_component = prior_component
            downsample_factors = relative_factors
        else:
            source_component = "data"
            downsample_factors = absolute_factors

        message = (
            f"Writing pyramid level {level_index}/{total_downsample_levels} "
            f"(factors={absolute_factors})"
        )
        if progress_callback is not None:
            progress = progress_start + int(
                ((level_index - 1) / total_downsample_levels)
                * max(0, progress_end - progress_start)
            )
            progress_callback(progress, message)

        source_array = da.from_zarr(str(store_path), component=source_component)
        downsampled = _downsample_tpczyx_by_stride(source_array, downsample_factors)
        level_shape = _normalize_tpczyx_shape(
            tuple(int(size) for size in downsampled.shape)
        )
        level_chunks = _derive_pyramid_level_chunks(
            base_chunks_tpczyx=base_chunks_tpczyx,
            level_shape_tpczyx=level_shape,
            level_factors_tpczyx=absolute_factors,
        )
        with dask.config.set({"array.rechunk.method": "tasks"}):
            downsampled = downsampled.rechunk(level_chunks)

        component = f"data_pyramid/level_{level_index}"
        write_graph = da.to_zarr(
            downsampled,
            url=str(store_path),
            component=component,
            overwrite=True,
            compute=False,
        )
        _compute_dask_graph(write_graph, client=client)

        root = zarr.open_group(str(store_path), mode="a")
        root[component].attrs.update(
            {
                "axes": ["t", "p", "c", "z", "y", "x"],
                "pyramid_level": int(level_index),
                "downsample_factors_tpczyx": [int(value) for value in absolute_factors],
                "chunk_shape_tpczyx": [int(value) for value in level_chunks],
                "source_component": source_component,
            }
        )
        level_paths.append(component)
        level_factor_payload.append([int(value) for value in absolute_factors])
        level_shapes_payload.append([int(value) for value in level_shape])
        prior_component = component
        prior_factors = absolute_factors

    root["data"].attrs.update(
        {
            "pyramid_levels": level_paths,
            "pyramid_factors_tpczyx": level_factor_payload,
        }
    )
    root.attrs.update(
        {
            "data_pyramid_levels": level_paths,
            "data_pyramid_factors_tpczyx": level_factor_payload,
            "data_pyramid_shapes_tpczyx": level_shapes_payload,
        }
    )
    if progress_callback is not None:
        progress_callback(int(progress_end), "Pyramid generation complete")
    return level_paths


def _compute_dask_graph(graph: Any, *, client: Optional["Client"] = None) -> None:
    """Execute a Dask graph via a configured client or local scheduler.

    Parameters
    ----------
    graph : Any
        Dask delayed graph, future-like object, or nested collection of graphs.
    client : dask.distributed.Client, optional
        Active distributed client. When omitted, local Dask scheduler is used.

    Returns
    -------
    None
        Execution side effects only.

    Raises
    ------
    Exception
        Propagates scheduler or task execution errors from Dask when
        local fallback is not applicable.

    Notes
    -----
    Some file-backed graphs (for example TIFF/HDF stores with thread locks)
    cannot be serialized for distributed workers. In those cases, execution
    automatically falls back to local Dask compute.
    """

    def _compute_with_threads() -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Running on a single-machine scheduler when a distributed client "
                    "is active might lead to unexpected results."
                ),
            )
            dask.compute(graph, scheduler="threads")

    if client is None:
        _compute_with_threads()
        return
    try:
        futures = client.compute(graph)
        client.gather(futures)
    except Exception as exc:
        message = str(exc)
        if (
            "Could not serialize object" not in message
            and "cannot pickle" not in message
        ):
            raise
        _compute_with_threads()


@dataclass(frozen=True)
class MaterializedDataStore:
    """Metadata for a source-to-Zarr materialization run.

    Attributes
    ----------
    source_path : pathlib.Path
        Resolved acquisition source path used for reading.
    store_path : pathlib.Path
        Resolved target Zarr store path containing canonical ``data`` array.
    source_component : str, optional
        Component path selected within the source store, when applicable.
    source_image_info : ImageInfo
        Source image metadata used for logging/provenance.
    data_image_info : ImageInfo
        Canonical materialized ``data`` array metadata.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective write chunks in canonical axis order.
    """

    source_path: Path
    store_path: Path
    source_component: Optional[str]
    source_image_info: ImageInfo
    data_image_info: ImageInfo
    chunks_tpczyx: CanonicalShapeTpczyx


def resolve_data_store_path(
    experiment: "NavigateExperiment",
    source_path: Union[str, Path],
) -> Path:
    """Resolve destination Zarr store path for materialized source data.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    source_path : str or pathlib.Path
        Resolved or candidate source path.

    Returns
    -------
    pathlib.Path
        Destination Zarr store path. Existing Zarr/N5 sources are reused
        in-place; non-Zarr sources are materialized as ``data_store.zarr``
        next to ``experiment.yml``.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    source = Path(source_path).expanduser().resolve()
    if _is_zarr_like_path(source):
        return source
    return (experiment.path.parent / "data_store.zarr").resolve()


def materialize_experiment_data_store(
    *,
    experiment: "NavigateExperiment",
    source_path: Union[str, Path],
    chunks: tuple[int, int, int, int, int, int] = (1, 1, 1, 256, 256, 256),
    pyramid_factors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ] = ((1,), (1,), (1,), (1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)),
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> MaterializedDataStore:
    """Materialize experiment source data into canonical Zarr ``data`` pyramid.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    source_path : str or pathlib.Path
        Acquisition source path resolved from experiment metadata.
    chunks : tuple[int, int, int, int, int, int], default=(1,1,1,256,256,256)
        Target chunking in ``(t, p, c, z, y, x)`` order.
    pyramid_factors : tuple[tuple[int, ...], ...], optional
        Resolution pyramid factors in ``(t, p, c, z, y, x)`` order.
    client : dask.distributed.Client, optional
        Active Dask distributed client used to execute graph writes.
    progress_callback : callable, optional
        Callback invoked as ``progress_callback(percent, message)`` with
        stage-level progress updates from ``0`` to ``100``.

    Returns
    -------
    MaterializedDataStore
        Materialization summary including source/store metadata for base level.

    Raises
    ------
    FileNotFoundError
        If ``source_path`` does not exist.
    ValueError
        If source format or dimensionality cannot be normalized.

    Notes
    -----
    The workflow writes canonical base data to ``data`` and then writes
    downsampled levels under ``data_pyramid/level_<n>`` according to
    ``pyramid_factors``. When ``source_path`` is already a Zarr/N5 store,
    conversion is performed in the same store path rather than creating
    a duplicate store.
    """

    def _emit_progress(percent: int, message: str) -> None:
        """Emit optional stage progress updates.

        Parameters
        ----------
        percent : int
            Progress value in ``[0, 100]``.
        message : str
            Human-readable stage description.

        Returns
        -------
        None
            Callback side effects only.
        """
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    source_resolved = Path(source_path).expanduser().resolve()
    if not source_resolved.exists():
        raise FileNotFoundError(source_resolved)

    store_path = resolve_data_store_path(
        experiment=experiment, source_path=source_resolved
    )
    write_client = client if _is_zarr_like_path(source_resolved) else None
    _emit_progress(5, "Opening source data")

    with ExitStack() as exit_stack:
        source_payload = _open_navigate_tiff_collection_as_dask(
            experiment=experiment,
            exit_stack=exit_stack,
        )
        if source_payload is None:
            source_payload = _open_navigate_bdv_collection_as_dask(
                experiment=experiment,
                source_path=source_resolved,
                exit_stack=exit_stack,
            )
        if source_payload is not None:
            source_array, source_axes, source_meta = source_payload
        else:
            source_array, source_axes, source_meta = _open_source_as_dask(
                source_resolved,
                exit_stack=exit_stack,
            )
        _emit_progress(20, "Loaded source metadata")
        source_shape = tuple(int(size) for size in source_array.shape)
        source_dtype = np.dtype(source_array.dtype)
        source_component_value = str(source_meta.get("source_component", "")).strip()
        source_component = source_component_value or None

        canonical = _coerce_to_tpczyx(
            source_array,
            experiment=experiment,
            source_axes=source_axes,
        )
        _emit_progress(35, "Normalizing canonical axes")
        canonical_shape = _normalize_tpczyx_shape(
            tuple(int(size) for size in canonical.shape)
        )
        normalized_chunks = _normalize_write_chunks(
            shape_tpczyx=canonical_shape,
            chunks=chunks,
        )
        with dask.config.set({"array.rechunk.method": "tasks"}):
            canonical = canonical.rechunk(normalized_chunks)
        _emit_progress(45, "Preparing chunked write graph")

        should_stage_same_component = (
            store_path == source_resolved and source_component == "data"
        )
        if should_stage_same_component:
            temp_component = "__clearex_tmp_data"
            root = zarr.open_group(str(store_path), mode="a")
            if temp_component in root:
                del root[temp_component]
            write_graph = da.to_zarr(
                canonical,
                url=str(store_path),
                component=temp_component,
                overwrite=True,
                compute=False,
            )
            _emit_progress(55, "Writing staged data to existing store")
            _compute_dask_graph(write_graph, client=write_client)
            _emit_progress(82, "Swapping staged data into canonical component")
            if "data" in root:
                del root["data"]
            root.move(temp_component, "data")
            initialize_analysis_store(
                experiment=experiment,
                zarr_path=store_path,
                overwrite=False,
                chunks=chunks,
                pyramid_factors=pyramid_factors,
                dtype=source_dtype.name,
                shape_tpczyx=canonical_shape,
            )
            _materialize_data_pyramid(
                store_path=store_path,
                base_chunks_tpczyx=normalized_chunks,
                pyramid_factors=pyramid_factors,
                client=client,
                progress_callback=progress_callback,
                progress_start=86,
                progress_end=96,
            )
            _emit_progress(97, "Finalizing store metadata")
        else:
            initialize_analysis_store(
                experiment=experiment,
                zarr_path=store_path,
                overwrite=True,
                chunks=chunks,
                pyramid_factors=pyramid_factors,
                dtype=source_dtype.name,
                shape_tpczyx=canonical_shape,
            )
            root = zarr.open_group(str(store_path), mode="a")
            write_graph = da.store(
                canonical,
                root["data"],
                lock=False,
                compute=False,
            )
            _emit_progress(55, "Writing canonical data")
            _compute_dask_graph(write_graph, client=write_client)
            _materialize_data_pyramid(
                store_path=store_path,
                base_chunks_tpczyx=normalized_chunks,
                pyramid_factors=pyramid_factors,
                client=client,
                progress_callback=progress_callback,
                progress_start=60,
                progress_end=96,
            )
            _emit_progress(97, "Finalizing store metadata")

    root = zarr.open_group(str(store_path), mode="a")
    source_axes_attr = list(source_axes) if source_axes is not None else None
    source_metadata_path = str(source_meta.get("source_path", source_resolved))
    voxel_size_um_zyx = None
    if (
        experiment.z_step_um is not None
        and experiment.xy_pixel_size_um is not None
        and experiment.z_step_um > 0
        and experiment.xy_pixel_size_um > 0
    ):
        voxel_size_um_zyx = [
            float(experiment.z_step_um),
            float(experiment.xy_pixel_size_um),
            float(experiment.xy_pixel_size_um),
        ]
    root["data"].attrs.update(
        {
            "source_path": source_metadata_path,
            "source_axes": source_axes_attr,
            "voxel_size_um_zyx": voxel_size_um_zyx,
        }
    )
    if source_component is not None:
        root["data"].attrs["source_component"] = source_component
    root.attrs.update(
        {
            "source_data_path": source_metadata_path,
            "source_data_axes": source_axes_attr,
            "source_data_component": source_component,
            "voxel_size_um_zyx": voxel_size_um_zyx,
        }
    )

    source_image_path = Path(source_metadata_path).expanduser()
    if not source_image_path.is_absolute():
        source_image_path = source_resolved

    source_image_info = ImageInfo(
        path=source_image_path,
        shape=source_shape,
        dtype=source_dtype,
        axes=_format_axes_for_image_info(source_axes),
        metadata=dict(source_meta),
    )
    data_image_info = ImageInfo(
        path=store_path,
        shape=canonical_shape,
        dtype=source_dtype,
        axes="TPCZYX",
        metadata={"component": "data"},
    )
    _emit_progress(100, "Materialization complete")
    return MaterializedDataStore(
        source_path=source_resolved,
        store_path=store_path,
        source_component=source_component,
        source_image_info=source_image_info,
        data_image_info=data_image_info,
        chunks_tpczyx=normalized_chunks,
    )


@dataclass
class NavigateChannel:
    """Selected channel metadata from a Navigate experiment.

    Attributes
    ----------
    name : str
        Channel key in the experiment state (e.g., ``"channel_1"``).
    laser : str, optional
        Laser label assigned for the channel.
    laser_index : int, optional
        Laser index used by acquisition.
    exposure_ms : float, optional
        Camera exposure time in milliseconds.
    is_selected : bool
        Whether the channel was selected for acquisition.
    """

    name: str
    laser: Optional[str]
    laser_index: Optional[int]
    exposure_ms: Optional[float]
    is_selected: bool


@dataclass
class NavigateExperiment:
    """Parsed Navigate experiment metadata required by ClearEx.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to the source ``experiment.yml``.
    raw : dict[str, Any]
        Full parsed experiment mapping.
    save_directory : pathlib.Path
        Acquisition output directory.
    file_type : str
        Declared output file type from acquisition settings.
    microscope_name : str, optional
        Active microscope profile name used during acquisition.
    image_mode : str, optional
        Acquisition mode (e.g., ``"z-stack"``).
    timepoints : int
        Number of timepoints captured.
    number_z_steps : int
        Number of z slices per stack.
    y_pixels : int
        Image height in pixels.
    x_pixels : int
        Image width in pixels.
    multiposition_count : int
        Number of multiposition entries recorded.
    selected_channels : list[NavigateChannel]
        Channels marked as selected in acquisition state.
    xy_pixel_size_um : float, optional
        Estimated sample-space XY pixel size in microns.
    z_step_um : float, optional
        Z-step size in microns.
    """

    path: Path
    raw: Dict[str, Any]
    save_directory: Path
    file_type: str
    microscope_name: Optional[str]
    image_mode: Optional[str]
    timepoints: int
    number_z_steps: int
    y_pixels: int
    x_pixels: int
    multiposition_count: int
    selected_channels: list[NavigateChannel]
    xy_pixel_size_um: Optional[float] = None
    z_step_um: Optional[float] = None

    @property
    def channel_count(self) -> int:
        """Return number of selected channels.

        Returns
        -------
        int
            Count of selected channels, defaulting to ``1``.
        """
        return max(1, len(self.selected_channels))

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert parsed experiment into JSON-friendly metadata.

        Returns
        -------
        dict[str, Any]
            Compact metadata mapping suitable for Zarr attrs/provenance.
        """
        return {
            "experiment_path": str(self.path),
            "save_directory": str(self.save_directory),
            "file_type": self.file_type,
            "microscope_name": self.microscope_name,
            "image_mode": self.image_mode,
            "timepoints": self.timepoints,
            "number_z_steps": self.number_z_steps,
            "multiposition_count": self.multiposition_count,
            "channel_count": self.channel_count,
            "xy_pixel_size_um": self.xy_pixel_size_um,
            "z_step_um": self.z_step_um,
            "selected_channels": [
                {
                    "name": channel.name,
                    "laser": channel.laser,
                    "laser_index": channel.laser_index,
                    "exposure_ms": channel.exposure_ms,
                    "is_selected": channel.is_selected,
                }
                for channel in self.selected_channels
            ],
        }


def is_navigate_experiment_file(path: Union[str, Path]) -> bool:
    """Return whether a path is a Navigate experiment descriptor file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` when basename is ``experiment.yml`` or ``experiment.yaml``.
    """
    name = Path(path).name.lower()
    return name in {"experiment.yml", "experiment.yaml"}


def _parse_serialized_text(text: str, *, context: str) -> Any:
    """Parse JSON text with optional YAML fallback.

    Parameters
    ----------
    text : str
        Input file content.
    context : str
        Human-readable context used in parse error messages.

    Returns
    -------
    Any
        Parsed Python object.

    Raises
    ------
    ValueError
        If text cannot be parsed.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not HAS_PYYAML:
            raise ValueError(f"{context} is not valid JSON and PyYAML is unavailable.")
        return yaml.safe_load(text)


def _parse_experiment_text(text: str) -> Dict[str, Any]:
    """Parse Navigate experiment text as JSON with YAML fallback.

    Parameters
    ----------
    text : str
        File content.

    Returns
    -------
    dict[str, Any]
        Parsed mapping.

    Raises
    ------
    ValueError
        If text cannot be parsed into a mapping.
    """
    parsed = _parse_serialized_text(text=text, context="experiment.yml")

    if not isinstance(parsed, dict):
        raise ValueError("Parsed experiment must be a mapping.")
    return parsed


def _looks_like_position_header(row: Any) -> bool:
    """Return whether a multiposition row is a header row.

    Parameters
    ----------
    row : Any
        Candidate row value.

    Returns
    -------
    bool
        ``True`` when the row appears to be a list/tuple of header strings.
    """
    if not isinstance(row, (list, tuple)):
        return False
    if not row:
        return False
    return all(isinstance(value, str) for value in row)


def _extract_position_rows_from_payload(payload: Any) -> Optional[list[Any]]:
    """Extract position rows from serialized multiposition payload.

    Parameters
    ----------
    payload : Any
        Parsed payload from ``multi_positions.yml``.

    Returns
    -------
    list[Any], optional
        Position rows with any header row removed, or ``None`` when payload
        does not contain an expected structure.
    """
    rows: Optional[list[Any]]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("positions", "MultiPositions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            rows = None
    else:
        rows = None

    if rows is None:
        return None
    if rows and _looks_like_position_header(rows[0]):
        return rows[1:]
    return rows


def _load_multiposition_rows(save_directory: Path) -> Optional[list[Any]]:
    """Load multiposition rows from ``multi_positions.yml`` when available.

    Parameters
    ----------
    save_directory : pathlib.Path
        Acquisition save directory containing experiment sidecar files.

    Returns
    -------
    list[Any], optional
        Parsed position rows without header row when available; otherwise
        ``None`` if file is missing or not parseable.
    """
    path = save_directory / "multi_positions.yml"
    if not path.exists():
        return None
    try:
        payload = _parse_serialized_text(
            text=path.read_text(), context="multi_positions.yml"
        )
        return _extract_position_rows_from_payload(payload)
    except Exception:
        return None


def _infer_multiposition_count(
    raw: Dict[str, Any],
    state: Dict[str, Any],
    save_directory: Path,
) -> int:
    """Infer position count from sidecar metadata and experiment payload.

    Parameters
    ----------
    raw : dict[str, Any]
        Parsed experiment payload.
    state : dict[str, Any]
        ``MicroscopeState`` mapping.
    save_directory : pathlib.Path
        Acquisition save directory.

    Returns
    -------
    int
        Position count with a minimum of ``1``.
    """
    is_multiposition = bool(state.get("is_multiposition", False))

    # Navigate records detailed position lists in the sidecar file.
    if is_multiposition:
        rows = _load_multiposition_rows(save_directory=save_directory)
        if rows is not None:
            return max(1, len(rows))

    fallback = raw.get("MultiPositions", [])
    if isinstance(fallback, list) and fallback:
        if _looks_like_position_header(fallback[0]):
            fallback = fallback[1:]
        return max(1, len(fallback))
    return 1


def _safe_optional_float(value: Any) -> Optional[float]:
    """Parse an optional float value.

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    float, optional
        Parsed float value, or ``None`` when parsing fails.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _camera_profile_mapping(
    *,
    camera: Dict[str, Any],
    microscope_name: Optional[str],
) -> Dict[str, Any]:
    """Resolve microscope-specific camera parameter mapping when available.

    Parameters
    ----------
    camera : dict[str, Any]
        Top-level ``CameraParameters`` mapping.
    microscope_name : str, optional
        Active microscope profile name.

    Returns
    -------
    dict[str, Any]
        Profile-specific mapping when available, otherwise ``camera``.
    """
    if microscope_name and isinstance(camera.get(microscope_name), dict):
        return camera[microscope_name]
    return camera


def _infer_xy_pixel_size_um(
    *,
    camera: Dict[str, Any],
    microscope_name: Optional[str],
) -> Optional[float]:
    """Infer sample-space XY pixel size in microns.

    Parameters
    ----------
    camera : dict[str, Any]
        ``CameraParameters`` mapping from experiment metadata.
    microscope_name : str, optional
        Active microscope profile name.

    Returns
    -------
    float, optional
        Estimated XY pixel size in microns, or ``None`` when unavailable.

    Notes
    -----
    This uses profile-specific values when available. Preferred estimate is
    ``fov_x / img_x_pixels`` (or ``fov_y / img_y_pixels``). If unavailable,
    ``pixel_size`` is used as a fallback.
    """
    profile = _camera_profile_mapping(
        camera=camera,
        microscope_name=microscope_name,
    )
    fov_x = _safe_optional_float(profile.get("fov_x"))
    img_x = _safe_optional_float(profile.get("img_x_pixels", profile.get("x_pixels")))
    if fov_x is not None and img_x is not None and img_x > 0:
        return float(fov_x / img_x)

    fov_y = _safe_optional_float(profile.get("fov_y"))
    img_y = _safe_optional_float(profile.get("img_y_pixels", profile.get("y_pixels")))
    if fov_y is not None and img_y is not None and img_y > 0:
        return float(fov_y / img_y)

    pixel_size = _safe_optional_float(profile.get("pixel_size"))
    if pixel_size is not None and pixel_size > 0:
        return float(pixel_size)
    return None


def load_navigate_experiment(path: Union[str, Path]) -> NavigateExperiment:
    """Load and normalize Navigate ``experiment.yml`` metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``experiment.yml``.

    Returns
    -------
    NavigateExperiment
        Parsed experiment metadata.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If file content is invalid or missing required structure.
    """
    experiment_path = Path(path).expanduser().resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(experiment_path)

    raw = _parse_experiment_text(experiment_path.read_text())

    saving = raw.get("Saving", {}) if isinstance(raw.get("Saving", {}), dict) else {}
    state = (
        raw.get("MicroscopeState", {})
        if isinstance(raw.get("MicroscopeState", {}), dict)
        else {}
    )
    camera = (
        raw.get("CameraParameters", {})
        if isinstance(raw.get("CameraParameters", {}), dict)
        else {}
    )

    save_directory = Path(
        saving.get("save_directory", str(experiment_path.parent))
    ).expanduser()
    if not save_directory.is_absolute():
        save_directory = (experiment_path.parent / save_directory).resolve()

    selected_channels: list[NavigateChannel] = []
    channels_obj = state.get("channels", {})
    if isinstance(channels_obj, dict):
        for channel_name in sorted(channels_obj.keys()):
            channel_value = channels_obj.get(channel_name, {})
            if not isinstance(channel_value, dict):
                continue
            if not bool(channel_value.get("is_selected", False)):
                continue
            selected_channels.append(
                NavigateChannel(
                    name=channel_name,
                    laser=(
                        str(channel_value.get("laser"))
                        if channel_value.get("laser") is not None
                        else None
                    ),
                    laser_index=(
                        int(channel_value["laser_index"])
                        if channel_value.get("laser_index") is not None
                        else None
                    ),
                    exposure_ms=(
                        float(channel_value["camera_exposure_time"])
                        if channel_value.get("camera_exposure_time") is not None
                        else None
                    ),
                    is_selected=True,
                )
            )

    multiposition_count = _infer_multiposition_count(
        raw=raw,
        state=state,
        save_directory=save_directory,
    )
    microscope_name = (
        str(state.get("microscope_name"))
        if state.get("microscope_name") is not None
        else None
    )
    xy_pixel_size_um = _infer_xy_pixel_size_um(
        camera=camera,
        microscope_name=microscope_name,
    )
    z_step_um = _safe_optional_float(state.get("step_size"))

    return NavigateExperiment(
        path=experiment_path,
        raw=raw,
        save_directory=save_directory,
        file_type=str(saving.get("file_type", "UNKNOWN")).upper(),
        microscope_name=microscope_name,
        image_mode=str(state.get("image_mode")) if state.get("image_mode") else None,
        timepoints=max(1, int(state.get("timepoints", 1))),
        number_z_steps=max(1, int(state.get("number_z_steps", 1))),
        y_pixels=max(1, int(camera.get("img_y_pixels", camera.get("y_pixels", 1)))),
        x_pixels=max(1, int(camera.get("img_x_pixels", camera.get("x_pixels", 1)))),
        multiposition_count=max(1, multiposition_count),
        selected_channels=selected_channels,
        xy_pixel_size_um=xy_pixel_size_um,
        z_step_um=z_step_um,
    )


def _h5_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort H5 acquisition files by channel and time when available.

    Parameters
    ----------
    path : pathlib.Path
        H5 file path.

    Returns
    -------
    tuple[int, int, str]
        Sort key tuple ``(channel_index, time_index, name)``.
    """
    match = re.match(r"CH(\d+)_(\d+)\.(?:h5|hdf5|hdf)$", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    return 9999, 9999, path.name


def _normalize_file_type(file_type: str) -> str:
    """Normalize acquisition file-type labels to canonical format tokens.

    Parameters
    ----------
    file_type : str
        Raw file-type value from acquisition metadata.

    Returns
    -------
    str
        Canonical token in ``{"TIFF", "H5", "N5", "ZARR", "UNKNOWN"}``.
    """
    token = str(file_type or "").strip().upper().replace("_", "-")
    token = token.replace(" ", "")

    if token in {
        "TIFF",
        "TIF",
        ".TIFF",
        ".TIF",
        "OME-TIFF",
        "OME-TIF",
        "OMETIFF",
        "OMETIF",
    }:
        return "TIFF"
    if token in {"H5", "HDF5", "HDF", ".H5", ".HDF5", ".HDF"}:
        return "H5"
    if token in {"N5", ".N5", "OME-N5", "OMEN5"}:
        return "N5"
    if token in {
        "ZARR",
        ".ZARR",
        "OME-ZARR",
        "OMEZARR",
        "OME-NGFF",
        "OMENGFF",
        "NGFF",
    }:
        return "ZARR"
    return "UNKNOWN"


def _is_mip_path(path: Path) -> bool:
    """Return whether a path appears to be a MIP/preview image.

    Parameters
    ----------
    path : pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` when path is located in a ``MIP`` directory or filename
        indicates a MIP artifact.
    """
    if "MIP" in path.name.upper():
        return True
    return any(part.upper() == "MIP" for part in path.parts)


def find_experiment_data_candidates(experiment: NavigateExperiment) -> list[Path]:
    """Find candidate acquisition data files/stores for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.

    Returns
    -------
    list[pathlib.Path]
        Candidate paths sorted by deterministic preference.
    """
    base = experiment.save_directory
    file_type = _normalize_file_type(experiment.file_type)

    if file_type == "H5":
        candidates = sorted(
            [p for p in base.glob("*") if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}],
            key=_h5_sort_key,
        )
        return candidates

    if file_type == "ZARR":
        return sorted([p for p in base.glob("*.zarr") if p.is_dir()])

    if file_type == "N5":
        return sorted([p for p in base.glob("*.n5") if p.is_dir()])

    if file_type == "TIFF":
        all_tiffs = sorted(
            {*base.rglob("*.tif"), *base.rglob("*.tiff")},
            key=lambda p: str(p),
        )
        primary = [p for p in all_tiffs if not _is_mip_path(p)]
        return primary or all_tiffs

    # Fallback: search known formats in priority order.
    fallback_tiffs = sorted(
        [
            p
            for p in {*base.rglob("*.tif"), *base.rglob("*.tiff")}
            if not _is_mip_path(p)
        ],
        key=lambda p: str(p),
    )
    if not fallback_tiffs:
        fallback_tiffs = sorted(
            {*base.rglob("*.tif"), *base.rglob("*.tiff")},
            key=lambda p: str(p),
        )

    fallback = [
        *sorted([p for p in base.glob("*.zarr") if p.is_dir()]),
        *sorted([p for p in base.glob("*.n5") if p.is_dir()]),
        *sorted(
            [p for p in base.glob("*") if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}],
            key=_h5_sort_key,
        ),
        *fallback_tiffs,
    ]
    return fallback


def resolve_experiment_data_path(experiment: NavigateExperiment) -> Path:
    """Resolve primary acquisition data path from experiment metadata.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.

    Returns
    -------
    pathlib.Path
        Selected source data path.

    Raises
    ------
    FileNotFoundError
        If no compatible source data can be found.
    """
    candidates = find_experiment_data_candidates(experiment)
    if not candidates:
        raise FileNotFoundError(
            f"No source data candidates found in {experiment.save_directory} "
            f"for file_type={experiment.file_type}."
        )
    return candidates[0]


def default_analysis_store_path(experiment: NavigateExperiment) -> Path:
    """Return canonical 6D analysis Zarr store path for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.

    Returns
    -------
    pathlib.Path
        Path to canonical analysis store (``analysis_6d.zarr``).
    """
    return experiment.save_directory / "analysis_6d.zarr"


def infer_zyx_shape(
    experiment: NavigateExperiment, image_info: Optional[ImageInfo]
) -> tuple[int, int, int]:
    """Infer ``(z, y, x)`` shape for 6D analysis store allocation.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    image_info : ImageInfo, optional
        Metadata from source image reader.

    Returns
    -------
    tuple[int, int, int]
        Inferred spatial dimensions ``(z, y, x)``.
    """
    if image_info is not None:
        shape = tuple(int(v) for v in image_info.shape)
        if len(shape) >= 3:
            return shape[-3], shape[-2], shape[-1]
    return experiment.number_z_steps, experiment.y_pixels, experiment.x_pixels


def initialize_analysis_store(
    experiment: NavigateExperiment,
    zarr_path: Union[str, Path],
    *,
    image_info: Optional[ImageInfo] = None,
    shape_tpczyx: Optional[CanonicalShapeTpczyx] = None,
    overwrite: bool = False,
    chunks: tuple[int, int, int, int, int, int] = (1, 1, 1, 256, 256, 256),
    pyramid_factors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ] = ((1,), (1,), (1,), (1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)),
    dtype: Optional[str] = None,
) -> Path:
    """Initialize canonical 6D analysis Zarr store for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    zarr_path : str or pathlib.Path
        Output Zarr store path.
    image_info : ImageInfo, optional
        Source image metadata used for dtype/shape inference.
    shape_tpczyx : tuple[int, int, int, int, int, int], optional
        Explicit canonical shape override in ``(t, p, c, z, y, x)`` order.
    overwrite : bool, default=False
        Whether to overwrite existing ``data`` array when present.
    chunks : tuple[int, int, int, int, int, int], default=(1,1,1,256,256,256)
        Target 6D chunking in ``(t, p, c, z, y, x)`` order.
    pyramid_factors : tuple[tuple[int, ...], ...], optional
        Downsampling factors in ``(t, p, c, z, y, x)`` order. Each axis must
        provide at least one positive factor and start with ``1``.
    dtype : str, optional
        Explicit output dtype. Defaults to source dtype or ``uint16``.

    Returns
    -------
    pathlib.Path
        Resolved Zarr store path.

    Raises
    ------
    ValueError
        If ``chunks`` or ``pyramid_factors`` are not valid for six axes.
    """
    axis_names = ("t", "p", "c", "z", "y", "x")
    if len(chunks) != len(axis_names):
        raise ValueError("chunks must define six values in (t, p, c, z, y, x) order.")
    requested_chunks = tuple(int(chunk) for chunk in chunks)
    if any(chunk <= 0 for chunk in requested_chunks):
        raise ValueError("chunks values must be greater than zero.")

    if len(pyramid_factors) != len(axis_names):
        raise ValueError(
            "pyramid_factors must define six axis entries in (t, p, c, z, y, x) order."
        )
    normalized_pyramid: list[tuple[int, ...]] = []
    for axis_name, levels in zip(axis_names, pyramid_factors, strict=False):
        parsed_levels = tuple(int(level) for level in levels)
        if not parsed_levels:
            raise ValueError(f"pyramid_factors for axis '{axis_name}' cannot be empty.")
        if any(level <= 0 for level in parsed_levels):
            raise ValueError(
                f"pyramid_factors for axis '{axis_name}' must be greater than zero."
            )
        if parsed_levels[0] != 1:
            raise ValueError(
                f"pyramid_factors for axis '{axis_name}' must start with 1."
            )
        normalized_pyramid.append(parsed_levels)

    output_path = Path(zarr_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shape_tpczyx is None:
        z_size, y_size, x_size = infer_zyx_shape(
            experiment=experiment, image_info=image_info
        )
        shape = (
            int(experiment.timepoints),
            int(experiment.multiposition_count),
            int(experiment.channel_count),
            int(z_size),
            int(y_size),
            int(x_size),
        )
    else:
        if len(shape_tpczyx) != len(axis_names):
            raise ValueError(
                "shape_tpczyx must define six values in (t, p, c, z, y, x) order."
            )
        shape = tuple(int(size) for size in shape_tpczyx)
        if any(size <= 0 for size in shape):
            raise ValueError("shape_tpczyx values must be greater than zero.")

    if dtype is None:
        if image_info is not None:
            dtype = np.dtype(image_info.dtype).name
        else:
            dtype = "uint16"
    else:
        dtype = np.dtype(dtype).name

    normalized_chunks = tuple(
        min(int(chunk), int(dim))
        for chunk, dim in zip(requested_chunks, shape, strict=False)
    )
    pyramid_payload = [list(levels) for levels in normalized_pyramid]
    voxel_size_um_zyx = None
    if (
        experiment.z_step_um is not None
        and experiment.xy_pixel_size_um is not None
        and experiment.z_step_um > 0
        and experiment.xy_pixel_size_um > 0
    ):
        voxel_size_um_zyx = [
            float(experiment.z_step_um),
            float(experiment.xy_pixel_size_um),
            float(experiment.xy_pixel_size_um),
        ]

    root = zarr.open_group(str(output_path), mode="a")
    root.require_group("results")
    root.require_group("provenance")
    if "data" in root:
        if overwrite:
            del root["data"]
        else:
            existing = root["data"]
            existing_chunks = (
                [int(chunk) for chunk in existing.chunks]
                if existing.chunks is not None
                else None
            )
            existing.attrs.update(
                {
                    "axes": ["t", "p", "c", "z", "y", "x"],
                    "storage_policy": "latest_only",
                    "chunk_shape_tpczyx": existing_chunks,
                    "configured_chunks_tpczyx": [
                        int(chunk) for chunk in requested_chunks
                    ],
                    "resolution_pyramid_factors_tpczyx": pyramid_payload,
                    "voxel_size_um_zyx": voxel_size_um_zyx,
                }
            )
            root.attrs.update(
                {
                    "schema": "clearex.analysis_store.v1",
                    "axes": ["t", "p", "c", "z", "y", "x"],
                    "source_experiment": str(experiment.path),
                    "navigate_experiment": experiment.to_metadata_dict(),
                    "storage_policy_analysis_outputs": "latest_only",
                    "storage_policy_provenance": "append_only",
                    "chunk_shape_tpczyx": existing_chunks,
                    "configured_chunks_tpczyx": [
                        int(chunk) for chunk in requested_chunks
                    ],
                    "resolution_pyramid_factors_tpczyx": pyramid_payload,
                    "voxel_size_um_zyx": voxel_size_um_zyx,
                }
            )
            return output_path

    root.create_dataset(
        name="data",
        shape=shape,
        chunks=normalized_chunks,
        dtype=dtype,
        overwrite=True,
    )
    root["data"].attrs.update(
        {
            "axes": ["t", "p", "c", "z", "y", "x"],
            "storage_policy": "latest_only",
            "chunk_shape_tpczyx": [int(chunk) for chunk in normalized_chunks],
            "configured_chunks_tpczyx": [int(chunk) for chunk in requested_chunks],
            "resolution_pyramid_factors_tpczyx": pyramid_payload,
            "voxel_size_um_zyx": voxel_size_um_zyx,
        }
    )
    root.attrs.update(
        {
            "schema": "clearex.analysis_store.v1",
            "axes": ["t", "p", "c", "z", "y", "x"],
            "source_experiment": str(experiment.path),
            "navigate_experiment": experiment.to_metadata_dict(),
            "storage_policy_analysis_outputs": "latest_only",
            "storage_policy_provenance": "append_only",
            "chunk_shape_tpczyx": [int(chunk) for chunk in normalized_chunks],
            "configured_chunks_tpczyx": [int(chunk) for chunk in requested_chunks],
            "resolution_pyramid_factors_tpczyx": pyramid_payload,
            "voxel_size_um_zyx": voxel_size_um_zyx,
        }
    )
    return output_path


def create_dask_client(
    *,
    scheduler_address: Optional[str] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    processes: bool = False,
    memory_limit: Union[str, float] = "auto",
    local_directory: Optional[Union[str, Path]] = None,
    dashboard_address: Optional[str] = ":0",
) -> "Client":
    """Create Dask distributed client (local default, cluster optional).

    Parameters
    ----------
    scheduler_address : str, optional
        Existing Dask scheduler address for multi-node operation.
    n_workers : int, optional
        Number of local workers when using local mode.
    threads_per_worker : int, default=1
        Threads per worker for local mode.
    processes : bool, default=False
        Whether local workers should be process-based. I/O-dominant workloads
        typically perform better with thread-based workers.
    memory_limit : str or float, default="auto"
        Memory limit per worker for local mode.
    local_directory : str or pathlib.Path, optional
        Worker local directory for spills/temp files.
    dashboard_address : str, optional, default=":0"
        Dashboard bind address for local mode. ``":0"`` requests an available
        ephemeral port to avoid collisions when multiple local clusters run.

    Returns
    -------
    dask.distributed.Client
        Connected Dask client.

    Raises
    ------
    ImportError
        If ``dask.distributed`` is not available.
    """
    from dask.distributed import Client, LocalCluster

    if scheduler_address:
        return Client(scheduler_address)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        local_directory=str(local_directory) if local_directory is not None else None,
        dashboard_address=dashboard_address,
    )
    return Client(cluster)


def write_zyx_block(
    zarr_path: Union[str, Path],
    block: ArrayLike,
    *,
    t_index: int,
    p_index: int,
    c_index: int,
    compute: bool = True,
) -> "Delayed | None":
    """Write one non-overlapping ``(z, y, x)`` block into 6D analysis store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Analysis Zarr store path.
    block : numpy.ndarray or dask.array.Array
        ``(z, y, x)`` volume to write.
    t_index : int
        Time index.
    p_index : int
        Position index.
    c_index : int
        Channel index.
    compute : bool, default=True
        If ``block`` is Dask, whether to execute the write immediately.

    Returns
    -------
    Any
        ``None`` for NumPy writes. For Dask writes, returns a delayed object
        when ``compute=False``.

    Raises
    ------
    ValueError
        If ``block`` is not 3D.
    TypeError
        If ``block`` type is unsupported.
    """
    if block.ndim != 3:
        raise ValueError(f"Expected 3D block (z, y, x), got shape={block.shape}.")

    z_size, y_size, x_size = (int(v) for v in block.shape)
    region = (
        slice(int(t_index), int(t_index) + 1),
        slice(int(p_index), int(p_index) + 1),
        slice(int(c_index), int(c_index) + 1),
        slice(0, z_size),
        slice(0, y_size),
        slice(0, x_size),
    )

    if isinstance(block, da.Array):
        block_6d = block[None, None, None, :, :, :]
        return da.to_zarr(
            block_6d,
            url=str(zarr_path),
            component="data",
            region=region,
            overwrite=False,
            compute=compute,
        )

    if isinstance(block, np.ndarray):
        root = zarr.open_group(str(zarr_path), mode="a")
        root["data"][region] = block[None, None, None, :, :, :]
        return None

    raise TypeError(
        f"Unsupported block type {type(block).__name__}; expected ndarray or Dask Array."
    )
