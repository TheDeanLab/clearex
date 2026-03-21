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
from datetime import datetime, timezone
from itertools import islice, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Sequence, Union
import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET

# Third Party Imports
import dask
import dask.array as da
import h5py
import numpy as np
import tifffile
import zarr
from dask.delayed import delayed

# Local Imports
from clearex.io.read import ImageInfo
from clearex.io.zarr_storage import (
    clear_component,
    create_or_overwrite_array,
    detect_store_format,
    extract_raw_axes_metadata,
    is_clearex_analysis_store,
    open_group as open_zarr_group,
    replace_store_path,
    resolve_external_analysis_store_path,
    resolve_legacy_v2_store_path,
    resolve_staging_store_path,
    to_jsonable,
)
from clearex.workflow import (
    SpatialCalibrationConfig,
    spatial_calibration_from_dict,
    spatial_calibration_to_dict,
)

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
IngestionProgressRecord = dict[str, Any]

_INGESTION_PROGRESS_SCHEMA = "clearex.ingestion_progress.v1"
_INGESTION_PROGRESS_ATTR = "ingestion_progress"
_SPATIAL_CALIBRATION_ATTR = "spatial_calibration"


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


def has_canonical_data_component(zarr_path: Union[str, Path]) -> bool:
    """Return whether a store contains canonical 6D analysis data.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Candidate Zarr/N5 store path.

    Returns
    -------
    bool
        ``True`` when the store exposes a ``data`` array in canonical
        ``(t, p, c, z, y, x)`` form. If axis metadata is present, it must
        also normalize to that same canonical order.

    Notes
    -----
    This helper is intentionally conservative and returns ``False`` for any
    unreadable, missing, or malformed ``data`` component.
    """
    try:
        root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="r")
    except Exception:
        return False

    if "data" not in root:
        return False

    data = root["data"]
    if not hasattr(data, "shape"):
        return False

    shape = tuple(int(size) for size in tuple(data.shape))
    if len(shape) != 6 or any(size <= 0 for size in shape):
        return False

    axes = _extract_zarr_axes(data, dict(getattr(root, "attrs", {})))
    if axes is not None and axes != ("t", "p", "c", "z", "y", "x"):
        return False
    return True


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Current UTC timestamp with timezone offset.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def _read_ingestion_progress_record(root: Any) -> Optional[IngestionProgressRecord]:
    """Read ingestion progress metadata from a Zarr root group.

    Parameters
    ----------
    root : Any
        Opened root Zarr group.

    Returns
    -------
    dict, optional
        Parsed ingestion progress record when present and dictionary-shaped.
        Returns ``None`` when progress metadata is missing or malformed.

    Raises
    ------
    None
        Invalid payloads are treated as absent metadata.
    """
    try:
        payload = root.attrs.get(_INGESTION_PROGRESS_ATTR)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def _write_ingestion_progress_record(
    *,
    store_path: Path,
    record: IngestionProgressRecord,
) -> None:
    """Persist ingestion progress metadata to a Zarr root group.

    Parameters
    ----------
    store_path : pathlib.Path
        Target Zarr/N5 store path.
    record : dict
        Ingestion progress record payload.

    Returns
    -------
    None
        Side-effect writes only.

    Raises
    ------
    TypeError
        If ``record`` cannot be serialized to JSON-compatible metadata.

    Notes
    -----
    Payload is round-tripped through JSON to guarantee metadata compatibility
    across Zarr backends.
    """
    serialized = json.loads(json.dumps(record))
    root = zarr.open_group(str(store_path), mode="a")
    root.attrs[_INGESTION_PROGRESS_ATTR] = serialized


def load_store_spatial_calibration(
    zarr_path: Union[str, Path],
) -> SpatialCalibrationConfig:
    """Load store-level spatial calibration from root Zarr attrs.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Analysis-store path.

    Returns
    -------
    SpatialCalibrationConfig
        Parsed store calibration. Missing attrs resolve to identity.

    Raises
    ------
    ValueError
        If stored spatial calibration metadata is malformed.
    """
    root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="r")
    return spatial_calibration_from_dict(root.attrs.get(_SPATIAL_CALIBRATION_ATTR))


def save_store_spatial_calibration(
    zarr_path: Union[str, Path],
    calibration: SpatialCalibrationConfig,
) -> SpatialCalibrationConfig:
    """Persist store-level spatial calibration into root Zarr attrs.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Analysis-store path.
    calibration : SpatialCalibrationConfig
        Calibration payload to persist.

    Returns
    -------
    SpatialCalibrationConfig
        Normalized calibration written to the store.
    """
    normalized = spatial_calibration_from_dict(calibration)
    serialized = json.loads(json.dumps(spatial_calibration_to_dict(normalized)))
    root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="a")
    root.attrs[_SPATIAL_CALIBRATION_ATTR] = serialized
    return normalized


def _resolve_expected_pyramid_level_factors(
    *,
    root: Any,
    expected_pyramid_factors: Optional[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = None,
) -> Optional[tuple[CanonicalShapeTpczyx, ...]]:
    """Resolve expected absolute pyramid factors for completeness checks.

    Parameters
    ----------
    root : Any
        Opened root Zarr group.
    expected_pyramid_factors : tuple[tuple[int, ...], ...], optional
        Explicit per-axis factors in ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...], optional
        Normalized per-level absolute factors. Returns ``None`` when factors
        are unavailable or malformed.

    Raises
    ------
    None
        Invalid factor metadata returns ``None``.
    """
    if expected_pyramid_factors is not None:
        try:
            return _normalize_pyramid_level_factors(expected_pyramid_factors)
        except Exception:
            return None

    raw = root.attrs.get("resolution_pyramid_factors_tpczyx")
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        return None
    try:
        parsed = tuple(
            tuple(int(value) for value in axis_levels)  # type: ignore[arg-type]
            for axis_levels in raw
        )
    except Exception:
        return None
    try:
        return _normalize_pyramid_level_factors(parsed)  # type: ignore[arg-type]
    except Exception:
        return None


def _expected_pyramid_components(
    level_factors: tuple[CanonicalShapeTpczyx, ...],
) -> list[str]:
    """Return canonical component paths implied by pyramid level factors.

    Parameters
    ----------
    level_factors : tuple[tuple[int, int, int, int, int, int], ...]
        Absolute factors per pyramid level, including base level.

    Returns
    -------
    list[str]
        Ordered component paths beginning with ``"data"``.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    return ["data", *[f"data_pyramid/level_{idx}" for idx in range(1, len(level_factors))]]


def _has_expected_pyramid_structure(
    *,
    root: Any,
    level_factors: Optional[tuple[CanonicalShapeTpczyx, ...]],
) -> bool:
    """Validate that required pyramid arrays exist with compatible structure.

    Parameters
    ----------
    root : Any
        Opened root Zarr group.
    level_factors : tuple[tuple[int, int, int, int, int, int], ...], optional
        Expected absolute factors per level. When ``None``, only base ``data``
        validation is enforced.

    Returns
    -------
    bool
        ``True`` when required levels exist and expose expected shape/chunks.

    Raises
    ------
    None
        Structural mismatches return ``False``.
    """
    if "data" not in root:
        return False
    data = root["data"]
    if not hasattr(data, "shape") or not hasattr(data, "chunks"):
        return False
    try:
        base_shape = _normalize_tpczyx_shape(tuple(int(v) for v in data.shape))
    except Exception:
        return False
    if data.chunks is None:
        return False
    try:
        base_chunks = _normalize_write_chunks(
            shape_tpczyx=base_shape,
            chunks=tuple(int(v) for v in data.chunks),
        )
    except Exception:
        return False

    if level_factors is None:
        return True

    expected_components = _expected_pyramid_components(level_factors)
    configured_levels = root.attrs.get("data_pyramid_levels")
    if isinstance(configured_levels, (list, tuple)):
        configured_paths = [str(value) for value in configured_levels]
        if configured_paths != expected_components:
            return False

    for level_index, factors in enumerate(level_factors[1:], start=1):
        component = f"data_pyramid/level_{level_index}"
        if component not in root:
            return False
        level_array = root[component]
        if not hasattr(level_array, "shape") or not hasattr(level_array, "chunks"):
            return False
        if level_array.chunks is None:
            return False

        expected_shape: CanonicalShapeTpczyx = (
            max(1, int(math.ceil(int(base_shape[0]) / int(factors[0])))),
            max(1, int(math.ceil(int(base_shape[1]) / int(factors[1])))),
            max(1, int(math.ceil(int(base_shape[2]) / int(factors[2])))),
            max(1, int(math.ceil(int(base_shape[3]) / int(factors[3])))),
            max(1, int(math.ceil(int(base_shape[4]) / int(factors[4])))),
            max(1, int(math.ceil(int(base_shape[5]) / int(factors[5])))),
        )
        actual_shape = tuple(int(size) for size in level_array.shape)
        if actual_shape != expected_shape:
            return False
        expected_chunks = _derive_pyramid_level_chunks(
            base_chunks_tpczyx=base_chunks,
            level_shape_tpczyx=expected_shape,
            level_factors_tpczyx=factors,
        )
        actual_chunks = tuple(int(size) for size in level_array.chunks)
        if actual_chunks != expected_chunks:
            return False
        raw_factors = level_array.attrs.get("downsample_factors_tpczyx")
        if isinstance(raw_factors, (list, tuple)):
            try:
                attr_factors = tuple(int(value) for value in raw_factors)
            except Exception:
                return False
            if attr_factors != tuple(int(value) for value in factors):
                return False

    return True


def _ingestion_record_is_complete(
    *,
    record: IngestionProgressRecord,
    required_components: list[str],
) -> bool:
    """Return whether progress metadata marks ingestion as complete.

    Parameters
    ----------
    record : dict
        Ingestion progress record payload.
    required_components : list[str]
        Required canonical components implied by expected configuration.

    Returns
    -------
    bool
        ``True`` when status is completed and tracked region counters indicate
        full completion for base data and required pyramid levels.

    Raises
    ------
    None
        Malformed payloads return ``False``.
    """
    schema = str(record.get("schema", "")).strip()
    if schema and schema != _INGESTION_PROGRESS_SCHEMA:
        return False
    status = str(record.get("status", "")).strip().lower()
    if status != "completed":
        return False

    base_progress = record.get("base_progress", {})
    if not isinstance(base_progress, dict):
        return False
    try:
        base_total = int(base_progress.get("total_regions", 0))
        base_completed = int(base_progress.get("completed_regions", -1))
    except Exception:
        return False
    if base_total <= 0 or base_completed < base_total:
        return False

    pyramid_progress = record.get("pyramid_progress", {})
    if not isinstance(pyramid_progress, dict):
        pyramid_progress = {}
    for component in required_components:
        if component == "data":
            continue
        level_progress = pyramid_progress.get(component, {})
        if not isinstance(level_progress, dict):
            return False
        try:
            level_total = int(level_progress.get("total_regions", 0))
            level_completed = int(level_progress.get("completed_regions", -1))
        except Exception:
            return False
        if level_total <= 0 or level_completed < level_total:
            return False

    return True


def has_complete_canonical_data_store(
    zarr_path: Union[str, Path],
    *,
    expected_chunks_tpczyx: Optional[CanonicalShapeTpczyx] = None,
    expected_pyramid_factors: Optional[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = None,
) -> bool:
    """Return whether canonical ingestion is complete for a Zarr/N5 store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Candidate Zarr/N5 store path.
    expected_chunks_tpczyx : tuple[int, int, int, int, int, int], optional
        Expected base-level chunking in canonical order.
    expected_pyramid_factors : tuple[tuple[int, ...], ...], optional
        Expected per-axis pyramid factors in canonical order.

    Returns
    -------
    bool
        ``True`` when base data is canonical, expected pyramid structure exists,
        and ingestion progress metadata indicates a completed run.

    Raises
    ------
    None
        Unreadable or malformed stores return ``False``.

    Notes
    -----
    Stores created before ingestion progress tracking existed are accepted when
    structural validation succeeds and no progress record is present.
    """
    if not has_canonical_data_component(zarr_path):
        return False
    try:
        root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="r")
    except Exception:
        return False

    data = root.get("data")
    if data is None or not hasattr(data, "shape") or not hasattr(data, "chunks"):
        return False
    if data.chunks is None:
        return False
    try:
        base_shape = _normalize_tpczyx_shape(tuple(int(size) for size in data.shape))
    except Exception:
        return False

    if expected_chunks_tpczyx is not None:
        try:
            expected_chunks = _normalize_write_chunks(
                shape_tpczyx=base_shape,
                chunks=expected_chunks_tpczyx,
            )
        except Exception:
            return False
        actual_chunks = tuple(int(size) for size in data.chunks)
        if actual_chunks != expected_chunks:
            return False

    level_factors = _resolve_expected_pyramid_level_factors(
        root=root,
        expected_pyramid_factors=expected_pyramid_factors,
    )
    if not _has_expected_pyramid_structure(root=root, level_factors=level_factors):
        return False
    required_components = (
        _expected_pyramid_components(level_factors) if level_factors is not None else ["data"]
    )

    record = _read_ingestion_progress_record(root)
    if record is None:
        return True
    return _ingestion_record_is_complete(
        record=record,
        required_components=required_components,
    )


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
    raw_axes = extract_raw_axes_metadata(array, group_attrs)
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
        Source BDV data path (``.h5``/``.hdf5``/``.hdf``/``.n5``/``.zarr``).

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
    def _candidate_bdv_xml_paths(path: Path) -> list[Path]:
        """Return candidate BDV XML sidecar paths for one source path.

        Parameters
        ----------
        path : pathlib.Path
            Source BDV container path.

        Returns
        -------
        list[pathlib.Path]
            Ordered XML candidate paths, deduplicated while preserving order.
        """
        candidates: list[Path] = []
        candidates.append(path.with_suffix(".xml"))
        candidates.append(path.parent / f"{path.name}.xml")

        lower_name = path.name.lower()
        for token in (".ome.zarr", ".zarr", ".n5", ".hdf5", ".hdf", ".h5"):
            if not lower_name.endswith(token):
                continue
            stem = path.name[: -len(token)]
            if stem:
                candidates.append(path.parent / f"{stem}.xml")
            break

        ordered: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        return ordered

    def _loader_format_matches_source_suffix(*, suffix: str, image_format: str) -> bool:
        """Return whether a BDV XML loader format matches the source suffix.

        Parameters
        ----------
        suffix : str
            Source path suffix.
        image_format : str
            XML ``ImageLoader`` format value.

        Returns
        -------
        bool
            ``True`` when ``image_format`` is compatible with ``suffix``.
        """
        normalized = str(image_format).strip().lower()
        if suffix in {".h5", ".hdf5", ".hdf"}:
            return normalized in {"bdv.hdf5", "bdv.h5", "bdv.hdf"}
        if suffix == ".n5":
            return normalized == "bdv.n5" or normalized.startswith("bdv.n5.")
        if suffix == ".zarr":
            if normalized in {
                "bdv.zarr",
                "bdv.ome.zarr",
                "bdv.ngff",
                "bdv.omezarr",
                "bdv.omengff",
                "ome.zarr",
                "ome-zarr",
                "ome.ngff",
                "ome-ngff",
                "ngff",
                "zarr",
            }:
                return True
            return (
                normalized.startswith("bdv.zarr.")
                or normalized.startswith("bdv.ome.zarr.")
                or normalized.startswith("bdv.ngff.")
                or normalized.startswith("ome.zarr.")
                or normalized.startswith("ome.ngff.")
            )
        return False

    suffix = source_path.suffix.lower()
    if suffix not in {".h5", ".hdf5", ".hdf", ".n5", ".zarr"}:
        return None

    xml_candidates = _candidate_bdv_xml_paths(source_path)
    for xml_path in xml_candidates:
        if not xml_path.exists():
            continue
        try:
            root = ET.fromstring(xml_path.read_text())
        except Exception:
            continue

        image_loader = root.find("SequenceDescription/ImageLoader")
        if image_loader is None:
            continue
        image_format = str(image_loader.attrib.get("format", ""))
        if not _loader_format_matches_source_suffix(
            suffix=suffix,
            image_format=image_format,
        ):
            continue

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
            continue

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
    return None


def _open_navigate_bdv_collection_as_dask(
    *,
    experiment: "NavigateExperiment",
    source_path: Path,
    exit_stack: ExitStack,
) -> Optional[tuple[da.Array, AxesSpec, dict[str, Any]]]:
    """Open Navigate BDV H5/N5/Zarr acquisitions as stacked ``(t, p, c, ...)``.

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
    is_zarr = suffix == ".zarr" and _is_zarr_like_path(source_path)
    if not is_h5 and not is_n5 and not is_zarr:
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
        entries: list[tuple[int, int, str, AxesSpec, Any]] = []

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
                        array,
                    )
                )
            for key in sorted(group_node.group_keys()):
                _walk(group_node[key], f"{prefix}{key}/")

        _walk(root)
        if len(entries) <= 1:
            return None

        for time_index, setup_index, component, source_axes, source_zarr_array in sorted(
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

            source_array = da.from_zarr(source_zarr_array)
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
                        "Navigate BDV collection has inconsistent source axes: "
                        f"expected {base_axes}, got {normalized_axes} at "
                        f"t={time_index}, setup={setup_index}."
                    )
                if source_shape != base_shape:
                    raise ValueError(
                        "Navigate BDV collection has inconsistent source shapes: "
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
        "source_collection_type": (
            "bdv_h5" if is_h5 else "bdv_n5" if is_n5 else "bdv_zarr"
        ),
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


def _axis_chunk_bounds(size: int, chunk_size: int) -> list[tuple[int, int]]:
    """Build contiguous chunk bounds for one axis.

    Parameters
    ----------
    size : int
        Axis length.
    chunk_size : int
        Chunk size for the axis.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(start, stop)`` bounds covering the full axis.
    """
    return [
        (start, min(start + chunk_size, size))
        for start in range(0, size, chunk_size)
    ]


def _iter_tpczyx_chunk_regions(
    *,
    shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
) -> Iterator[tuple[slice, slice, slice, slice, slice, slice]]:
    """Build canonical chunk-aligned regions for one array.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full canonical array shape.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective chunk shape.

    Returns
    -------
    iterator of tuple[slice, ...]
        Ordered canonical chunk-aligned region slices in ``(t, p, c, z, y, x)``
        order.
    """
    axis_bounds = [
        _axis_chunk_bounds(int(size), int(chunk))
        for size, chunk in zip(shape_tpczyx, chunks_tpczyx, strict=False)
    ]
    for bounds in product(*axis_bounds):
        yield (
            slice(int(bounds[0][0]), int(bounds[0][1])),
            slice(int(bounds[1][0]), int(bounds[1][1])),
            slice(int(bounds[2][0]), int(bounds[2][1])),
            slice(int(bounds[3][0]), int(bounds[3][1])),
            slice(int(bounds[4][0]), int(bounds[4][1])),
            slice(int(bounds[5][0]), int(bounds[5][1])),
        )


def _count_tpczyx_chunk_regions(
    *,
    shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
) -> int:
    """Return the number of canonical chunk-aligned write regions.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full canonical array shape.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective chunk shape.

    Returns
    -------
    int
        Number of chunk-aligned regions needed to cover the full array.
    """
    return int(
        math.prod(
            max(1, math.ceil(int(size) / max(1, int(chunk))))
            for size, chunk in zip(shape_tpczyx, chunks_tpczyx, strict=False)
        )
    )


def _estimate_write_batch_region_count(
    *,
    chunks_tpczyx: CanonicalShapeTpczyx,
    dtype_itemsize: int,
) -> int:
    """Estimate how many chunk regions to submit per write batch.

    Parameters
    ----------
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective chunk shape in canonical order.
    dtype_itemsize : int
        Bytes per array element.

    Returns
    -------
    int
        Recommended number of chunk regions per submission batch.
    """
    chunk_bytes = max(1, math.prod(int(value) for value in chunks_tpczyx) * int(dtype_itemsize))
    target_batch_bytes = 512 << 20
    return max(1, min(64, target_batch_bytes // chunk_bytes))


def _chunk_axis_min_max(axis_chunks: tuple[int, ...]) -> tuple[int, int]:
    """Return minimum and maximum chunk sizes for one Dask axis.

    Parameters
    ----------
    axis_chunks : tuple[int, ...]
        Chunk-size sequence for a single axis from ``dask.array.Array.chunks``.

    Returns
    -------
    tuple[int, int]
        ``(min_chunk, max_chunk)`` for the axis.

    Raises
    ------
    ValueError
        If ``axis_chunks`` is empty.
    """
    if not axis_chunks:
        raise ValueError("axis_chunks cannot be empty.")
    return (
        min(int(value) for value in axis_chunks),
        max(int(value) for value in axis_chunks),
    )


def _dask_chunk_min_max_tpczyx(
    array: da.Array,
) -> tuple[CanonicalShapeTpczyx, CanonicalShapeTpczyx]:
    """Return per-axis min/max chunk sizes for canonical 6D arrays.

    Parameters
    ----------
    array : dask.array.Array
        Canonical ``(t, p, c, z, y, x)`` array.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...]]
        Pair of ``(min_chunks_tpczyx, max_chunks_tpczyx)``.

    Raises
    ------
    ValueError
        If ``array`` is not six-dimensional.
    """
    if array.ndim != 6:
        raise ValueError(
            "Expected canonical 6D array (t, p, c, z, y, x) for chunk inspection."
        )
    axis_min_max = [_chunk_axis_min_max(axis_chunks) for axis_chunks in array.chunks]
    min_chunks: CanonicalShapeTpczyx = (
        int(axis_min_max[0][0]),
        int(axis_min_max[1][0]),
        int(axis_min_max[2][0]),
        int(axis_min_max[3][0]),
        int(axis_min_max[4][0]),
        int(axis_min_max[5][0]),
    )
    max_chunks: CanonicalShapeTpczyx = (
        int(axis_min_max[0][1]),
        int(axis_min_max[1][1]),
        int(axis_min_max[2][1]),
        int(axis_min_max[3][1]),
        int(axis_min_max[4][1]),
        int(axis_min_max[5][1]),
    )
    return min_chunks, max_chunks


def _should_use_source_aligned_plane_writes(
    *,
    array: da.Array,
    shape_tpczyx: CanonicalShapeTpczyx,
    target_chunks_tpczyx: CanonicalShapeTpczyx,
) -> bool:
    """Return whether source-aligned z-batch writes should be preferred.

    Parameters
    ----------
    array : dask.array.Array
        Canonical source array in ``(t, p, c, z, y, x)`` order.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full canonical output shape.
    target_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Configured output chunk shape.

    Returns
    -------
    bool
        ``True`` when source chunks are single-plane along ``z`` and full-frame
        along ``y/x`` while the target chunking splits the lateral dimensions.

    Raises
    ------
    ValueError
        Propagates chunk-inspection errors for non-6D arrays.

    Notes
    -----
    In this pattern, chunk-batched writes over small ``(y, x)`` tiles cause
    repeated decompression of the same source plane chunks across many write
    batches. Source-aligned z-batch writes avoid this read amplification.
    """
    min_chunks, max_chunks = _dask_chunk_min_max_tpczyx(array)
    source_is_single_plane_z = min_chunks[3] == 1 and max_chunks[3] == 1
    source_is_full_lateral = (
        min_chunks[4] >= int(shape_tpczyx[4]) and min_chunks[5] >= int(shape_tpczyx[5])
    )
    target_splits_lateral = (
        int(target_chunks_tpczyx[4]) < int(shape_tpczyx[4])
        or int(target_chunks_tpczyx[5]) < int(shape_tpczyx[5])
    )
    return bool(source_is_single_plane_z and source_is_full_lateral and target_splits_lateral)


def _detect_runtime_memory_bytes() -> int:
    """Return an approximate runtime memory budget in bytes.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Detected physical memory in bytes, with a conservative fallback.

    Notes
    -----
    This helper is intentionally lightweight and best-effort. It is used only
    to scale source-aligned write aggressiveness and should never raise.
    """
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and phys_pages > 0:
                return page_size * phys_pages
        except Exception:
            pass

    return 8 << 30


def _detect_client_worker_resources(
    client: Optional["Client"],
) -> tuple[Optional[int], Optional[int]]:
    """Return active worker count and smallest worker memory limit.

    Parameters
    ----------
    client : dask.distributed.Client, optional
        Active distributed client.

    Returns
    -------
    tuple[int | None, int | None]
        ``(worker_count, min_worker_memory_limit_bytes)``. Values are ``None``
        when the scheduler metadata is unavailable.

    Notes
    -----
    This helper is best-effort and never raises. It uses scheduler metadata to
    constrain ingestion batching to real worker memory limits.
    """
    if client is None:
        return None, None
    try:
        scheduler = client.scheduler_info()
    except Exception:
        return None, None

    workers = scheduler.get("workers")
    if not isinstance(workers, dict) or not workers:
        return None, None

    worker_count = len(workers)
    memory_limits: list[int] = []
    for worker_payload in workers.values():
        if not isinstance(worker_payload, dict):
            continue
        raw_limit = worker_payload.get("memory_limit")
        try:
            parsed_limit = int(raw_limit)
        except Exception:
            continue
        if parsed_limit > 0:
            memory_limits.append(parsed_limit)

    min_worker_memory_limit = min(memory_limits) if memory_limits else None
    return int(worker_count), (
        None if min_worker_memory_limit is None else int(min_worker_memory_limit)
    )


def _estimate_source_plane_batch_depth(
    *,
    shape_tpczyx: CanonicalShapeTpczyx,
    target_chunks_tpczyx: CanonicalShapeTpczyx,
    dtype_itemsize: int,
    max_planes_per_batch: int = 128,
    target_batch_bytes: Optional[int] = None,
    worker_memory_limit_bytes: Optional[int] = None,
) -> int:
    """Estimate z-depth per source-aligned write region.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Canonical array shape.
    target_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Canonical target chunk shape.
    dtype_itemsize : int
        Bytes per element.
    max_planes_per_batch : int, default=128
        Hard cap on z-planes read per source-aligned write region.
    target_batch_bytes : int, optional
        Approximate in-memory byte budget per source-aligned write region.
        When omitted, runtime memory heuristics choose an aggressive value.
    worker_memory_limit_bytes : int, optional
        Best-effort per-worker memory limit in bytes. When provided, this
        constrains the batch depth against real Dask worker limits.

    Returns
    -------
    int
        Estimated z-plane count per source-aligned region.

    Raises
    ------
    ValueError
        If provided limits are not positive.

    Notes
    -----
    The estimate is conservative and memory-first. It prefers source-aligned
    batches that fit comfortably in worker memory and only snaps to target
    ``z`` chunk multiples when they fit inside that budget.
    """
    if max_planes_per_batch <= 0:
        raise ValueError("max_planes_per_batch must be greater than zero.")
    z_size = max(1, int(shape_tpczyx[3]))
    y_size = max(1, int(shape_tpczyx[4]))
    x_size = max(1, int(shape_tpczyx[5]))
    plane_bytes = max(1, y_size * x_size * max(1, int(dtype_itemsize)))

    target_z = max(1, int(target_chunks_tpczyx[3]))
    if worker_memory_limit_bytes is not None and int(worker_memory_limit_bytes) > 0:
        max_safe_batch_bytes = max(
            512 << 20,
            min(
                12 << 30,
                int(int(worker_memory_limit_bytes) * 0.30),
            ),
        )
    else:
        runtime_memory = _detect_runtime_memory_bytes()
        max_safe_batch_bytes = max(
            512 << 20,
            min(8 << 30, int(runtime_memory // 16)),
        )
    if target_batch_bytes is not None and target_batch_bytes <= 0:
        raise ValueError("target_batch_bytes must be greater than zero.")
    desired_chunk_aligned_bytes = max(
        plane_bytes,
        int(target_z) * int(plane_bytes),
    )
    candidate_budgets = [int(max_safe_batch_bytes), int(desired_chunk_aligned_bytes)]
    if target_batch_bytes is not None:
        candidate_budgets.append(int(target_batch_bytes))
    effective_target_batch_bytes = max(plane_bytes, min(candidate_budgets))

    by_bytes = max(1, int(effective_target_batch_bytes) // plane_bytes)
    depth = min(z_size, int(max_planes_per_batch), by_bytes)

    if depth >= target_z:
        depth = max(target_z, (depth // target_z) * target_z)

    return max(1, int(depth))


def _iter_source_aligned_plane_regions(
    *,
    shape_tpczyx: CanonicalShapeTpczyx,
    z_batch_depth: int,
) -> Iterator[tuple[slice, slice, slice, slice, slice, slice]]:
    """Yield canonical write regions that preserve full ``(y, x)`` source planes.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full canonical output shape.
    z_batch_depth : int
        Number of z-planes per region.

    Returns
    -------
    iterator of tuple[slice, ...]
        Source-aligned write regions in ``(t, p, c, z, y, x)`` order.

    Raises
    ------
    ValueError
        If ``z_batch_depth`` is not positive.
    """
    if z_batch_depth <= 0:
        raise ValueError("z_batch_depth must be greater than zero.")

    t_size, p_size, c_size, z_size, y_size, x_size = (
        int(shape_tpczyx[0]),
        int(shape_tpczyx[1]),
        int(shape_tpczyx[2]),
        int(shape_tpczyx[3]),
        int(shape_tpczyx[4]),
        int(shape_tpczyx[5]),
    )
    for t_index in range(t_size):
        for p_index in range(p_size):
            for c_index in range(c_size):
                for z_start in range(0, z_size, int(z_batch_depth)):
                    z_stop = min(z_size, z_start + int(z_batch_depth))
                    yield (
                        slice(t_index, t_index + 1),
                        slice(p_index, p_index + 1),
                        slice(c_index, c_index + 1),
                        slice(z_start, z_stop),
                        slice(0, y_size),
                        slice(0, x_size),
                    )


def _count_source_aligned_plane_regions(
    *,
    shape_tpczyx: CanonicalShapeTpczyx,
    z_batch_depth: int,
) -> int:
    """Return number of source-aligned write regions.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full canonical output shape.
    z_batch_depth : int
        Number of z-planes per source-aligned region.

    Returns
    -------
    int
        Number of write regions required to cover the full array.
    """
    t_size, p_size, c_size, z_size = (
        int(shape_tpczyx[0]),
        int(shape_tpczyx[1]),
        int(shape_tpczyx[2]),
        int(shape_tpczyx[3]),
    )
    z_regions = max(1, math.ceil(z_size / max(1, int(z_batch_depth))))
    return int(max(1, t_size * p_size * c_size * z_regions))


def _estimate_source_aligned_submission_batch_count(
    *,
    shape_tpczyx: CanonicalShapeTpczyx,
    z_batch_depth: int,
    dtype_itemsize: int,
    worker_count: Optional[int] = None,
    worker_memory_limit_bytes: Optional[int] = None,
) -> int:
    """Estimate regions per compute submission for source-aligned writes.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full canonical output shape.
    z_batch_depth : int
        Source-aligned z-planes per region.
    dtype_itemsize : int
        Bytes per element.
    worker_count : int, optional
        Active Dask worker count when distributed execution is enabled.
    worker_memory_limit_bytes : int, optional
        Per-worker memory limit in bytes when available.

    Returns
    -------
    int
        Recommended source-aligned regions per graph submission.
    """
    region_bytes = max(
        1,
        int(z_batch_depth)
        * int(shape_tpczyx[4])
        * int(shape_tpczyx[5])
        * max(1, int(dtype_itemsize)),
    )
    runtime_memory = _detect_runtime_memory_bytes()
    cpu_count = max(1, int(os.cpu_count() or 1))
    effective_worker_count = max(
        1,
        int(worker_count) if worker_count is not None and int(worker_count) > 0 else cpu_count,
    )

    per_worker_memory_limit = (
        int(worker_memory_limit_bytes)
        if worker_memory_limit_bytes is not None and int(worker_memory_limit_bytes) > 0
        else None
    )
    if per_worker_memory_limit is not None:
        target_submission_bytes = max(
            2 << 30,
            min(
                128 << 30,
                int(per_worker_memory_limit * effective_worker_count * 0.50),
            ),
        )
        max_regions_per_worker = max(
            1,
            int((int(per_worker_memory_limit) * 0.60) // region_bytes),
        )
        max_regions_by_worker_memory = max(
            1,
            int(max_regions_per_worker) * int(effective_worker_count),
        )
        max_regions_by_concurrency = max(1, int(effective_worker_count) * 2)
        hard_cap = 128
    else:
        target_submission_bytes = max(
            2 << 30,
            min(64 << 30, int(runtime_memory // 3)),
        )
        max_regions_by_worker_memory = max(1, int(effective_worker_count))
        max_regions_by_concurrency = max(1, int(cpu_count))
        hard_cap = 64

    return max(
        1,
        min(
            int(hard_cap),
            int(max_regions_by_concurrency),
            int(max_regions_by_worker_memory),
            int(target_submission_bytes // region_bytes),
        ),
    )


def _write_numpy_region(
    block: np.ndarray,
    *,
    zarr_path: str,
    component: str,
    region: tuple[slice, slice, slice, slice, slice, slice],
) -> int:
    """Write one computed canonical block into a Zarr/N5 component.

    Parameters
    ----------
    block : numpy.ndarray
        Computed data block aligned with ``region``.
    zarr_path : str
        Target Zarr/N5 store path.
    component : str
        Target component path relative to the store root.
    region : tuple[slice, slice, slice, slice, slice, slice]
        Canonical write region in ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    int
        Constant ``1`` for completion accounting.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    root[component][region] = np.asarray(block)
    return 1


def _write_dask_array_in_batches(
    *,
    array: da.Array,
    store_path: Path,
    component: str,
    shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
    dtype_itemsize: int,
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 100,
    progress_label: str = "Writing data",
    start_region_index: int = 0,
    batch_completed_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Write a canonical Dask array to Zarr/N5 using bounded region batches.

    Parameters
    ----------
    array : dask.array.Array
        Canonical source array in ``(t, p, c, z, y, x)`` order.
    store_path : pathlib.Path
        Target Zarr/N5 store path.
    component : str
        Target component path.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full output shape.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective output chunk shape.
    dtype_itemsize : int
        Bytes per array element.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Optional callback invoked as ``callback(percent, message)``.
    progress_start : int, default=0
        Start percentage for write progress.
    progress_end : int, default=100
        End percentage for write progress.
    progress_label : str, default="Writing data"
        Human-readable stage label.
    start_region_index : int, default=0
        Number of initial chunk regions to skip before writing. This enables
        resumable writes from a prior interrupted run.
    batch_completed_callback : callable, optional
        Optional callback invoked as
        ``batch_completed_callback(completed_regions, total_regions)`` after
        each successful batch.

    Returns
    -------
    None
        Side-effect writes only.
    """
    total_regions = _count_tpczyx_chunk_regions(
        shape_tpczyx=shape_tpczyx,
        chunks_tpczyx=chunks_tpczyx,
    )
    if total_regions == 0:
        return
    start_region = max(0, min(int(start_region_index), int(total_regions)))
    if start_region >= total_regions:
        if progress_callback is not None:
            progress_callback(
                int(progress_end),
                f"{progress_label}: resumed with all regions already complete "
                f"({start_region}/{total_regions})",
            )
        if batch_completed_callback is not None:
            batch_completed_callback(int(total_regions), int(total_regions))
        return

    batch_region_count = _estimate_write_batch_region_count(
        chunks_tpczyx=chunks_tpczyx,
        dtype_itemsize=dtype_itemsize,
    )
    remaining_regions = int(total_regions - start_region)
    total_batches = max(1, math.ceil(remaining_regions / batch_region_count))
    completed_regions = int(start_region)
    region_iter = islice(
        _iter_tpczyx_chunk_regions(
        shape_tpczyx=shape_tpczyx,
        chunks_tpczyx=chunks_tpczyx,
        ),
        int(start_region),
        None,
    )

    for batch_index in range(1, total_batches + 1):
        batch_regions = list(islice(region_iter, batch_region_count))
        if not batch_regions:
            break
        batch_tasks = [
            delayed(_write_numpy_region)(
                array[region],
                zarr_path=str(store_path),
                component=component,
                region=region,
            )
            for region in batch_regions
        ]
        _compute_dask_graph(batch_tasks, client=client)
        completed_regions += len(batch_regions)
        if progress_callback is not None:
            progress = int(
                progress_start
                + (completed_regions / total_regions)
                * max(0, progress_end - progress_start)
            )
            progress_callback(
                progress,
                f"{progress_label}: batch {batch_index}/{total_batches} "
                f"({completed_regions}/{total_regions} regions)",
            )
        if batch_completed_callback is not None:
            batch_completed_callback(int(completed_regions), int(total_regions))


def _write_dask_array_source_aligned_plane_batches(
    *,
    array: da.Array,
    store_path: Path,
    component: str,
    shape_tpczyx: CanonicalShapeTpczyx,
    z_batch_depth: int,
    dtype_itemsize: int,
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 100,
    progress_label: str = "Writing data",
    worker_count: Optional[int] = None,
    worker_memory_limit_bytes: Optional[int] = None,
    start_region_index: int = 0,
    batch_completed_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Write canonical data using source-aligned full-frame z-batches.

    Parameters
    ----------
    array : dask.array.Array
        Canonical source array in ``(t, p, c, z, y, x)`` order.
    store_path : pathlib.Path
        Target Zarr/N5 store path.
    component : str
        Target component path.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full output shape.
    z_batch_depth : int
        Number of z-planes per source-aligned write region.
    dtype_itemsize : int
        Bytes per array element.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Optional callback invoked as ``callback(percent, message)``.
    progress_start : int, default=0
        Start percentage for write progress.
    progress_end : int, default=100
        End percentage for write progress.
    progress_label : str, default="Writing data"
        Human-readable stage label.
    worker_count : int, optional
        Active Dask worker count. When omitted, this is auto-detected from the
        provided client when possible.
    worker_memory_limit_bytes : int, optional
        Per-worker memory limit in bytes. When omitted, this is auto-detected
        from the provided client when possible.
    start_region_index : int, default=0
        Number of initial source-aligned regions to skip before writing. This
        enables resumable writes from a prior interrupted run.
    batch_completed_callback : callable, optional
        Optional callback invoked as
        ``batch_completed_callback(completed_regions, total_regions)`` after
        each successful batch.

    Returns
    -------
    None
        Side-effect writes only.
    """
    total_regions = _count_source_aligned_plane_regions(
        shape_tpczyx=shape_tpczyx,
        z_batch_depth=z_batch_depth,
    )
    if total_regions == 0:
        return
    start_region = max(0, min(int(start_region_index), int(total_regions)))
    if start_region >= total_regions:
        if progress_callback is not None:
            progress_callback(
                int(progress_end),
                f"{progress_label}: resumed with all regions already complete "
                f"({start_region}/{total_regions} regions; z_batch={z_batch_depth})",
            )
        if batch_completed_callback is not None:
            batch_completed_callback(int(total_regions), int(total_regions))
        return

    detected_worker_count = worker_count
    detected_worker_memory_limit_bytes = worker_memory_limit_bytes
    if detected_worker_count is None or detected_worker_memory_limit_bytes is None:
        auto_worker_count, auto_worker_memory_limit = _detect_client_worker_resources(client)
        if detected_worker_count is None:
            detected_worker_count = auto_worker_count
        if detected_worker_memory_limit_bytes is None:
            detected_worker_memory_limit_bytes = auto_worker_memory_limit

    regions_per_submission = _estimate_source_aligned_submission_batch_count(
        shape_tpczyx=shape_tpczyx,
        z_batch_depth=z_batch_depth,
        dtype_itemsize=dtype_itemsize,
        worker_count=detected_worker_count,
        worker_memory_limit_bytes=detected_worker_memory_limit_bytes,
    )
    remaining_regions = int(total_regions - start_region)
    total_batches = max(1, math.ceil(remaining_regions / regions_per_submission))
    completed_regions = int(start_region)
    region_iter = islice(
        _iter_source_aligned_plane_regions(
            shape_tpczyx=shape_tpczyx,
            z_batch_depth=z_batch_depth,
        ),
        int(start_region),
        None,
    )

    for batch_index in range(1, total_batches + 1):
        batch_regions = list(islice(region_iter, regions_per_submission))
        if not batch_regions:
            break
        batch_tasks = [
            delayed(_write_numpy_region)(
                array[region],
                zarr_path=str(store_path),
                component=component,
                region=region,
            )
            for region in batch_regions
        ]
        _compute_dask_graph(batch_tasks, client=client)
        completed_regions += len(batch_regions)
        if progress_callback is not None:
            progress = int(
                progress_start
                + (completed_regions / total_regions)
                * max(0, progress_end - progress_start)
            )
            progress_callback(
                progress,
                f"{progress_label}: batch {batch_index}/{total_batches} "
                f"({completed_regions}/{total_regions} regions; z_batch={z_batch_depth})",
            )
        if batch_completed_callback is not None:
            batch_completed_callback(int(completed_regions), int(total_regions))


def _component_matches_shape_and_chunks(
    *,
    root: Any,
    component: str,
    shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
) -> bool:
    """Return whether a component exists with expected shape and chunks.

    Parameters
    ----------
    root : Any
        Opened root Zarr group.
    component : str
        Component path to validate.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Expected canonical shape.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Expected canonical chunk shape.

    Returns
    -------
    bool
        ``True`` when component exists and matches expected structure.

    Raises
    ------
    None
        Invalid or missing components return ``False``.
    """
    if component not in root:
        return False
    target = root[component]
    if not hasattr(target, "shape") or not hasattr(target, "chunks"):
        return False
    if target.chunks is None:
        return False
    try:
        actual_shape = tuple(int(size) for size in target.shape)
        actual_chunks = tuple(int(size) for size in target.chunks)
    except Exception:
        return False
    return actual_shape == tuple(int(v) for v in shape_tpczyx) and actual_chunks == tuple(
        int(v) for v in chunks_tpczyx
    )


def _create_ingestion_progress_record(
    *,
    source_path: Path,
    source_component: Optional[str],
    target_component: str,
    canonical_shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
    level_factors_tpczyx: tuple[CanonicalShapeTpczyx, ...],
    write_mode: str,
    z_batch_depth: Optional[int],
    base_total_regions: int,
) -> IngestionProgressRecord:
    """Create an initial ingestion progress record.

    Parameters
    ----------
    source_path : pathlib.Path
        Resolved source acquisition path.
    source_component : str, optional
        Source component path when reading from a Zarr/N5 store.
    target_component : str
        Destination component path for base canonical data writes.
    canonical_shape_tpczyx : tuple[int, int, int, int, int, int]
        Canonical output shape.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective canonical output chunks.
    level_factors_tpczyx : tuple[tuple[int, int, int, int, int, int], ...]
        Normalized absolute pyramid factors including base level.
    write_mode : str
        Base write mode identifier.
    z_batch_depth : int, optional
        Source-aligned z-batch depth when applicable.
    base_total_regions : int
        Total write regions for the base component.

    Returns
    -------
    dict
        Initialized in-progress record.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    now = _utc_now_iso()
    return {
        "schema": _INGESTION_PROGRESS_SCHEMA,
        "status": "in_progress",
        "source_path": str(source_path),
        "source_component": source_component,
        "target_component": str(target_component),
        "write_mode": str(write_mode),
        "z_batch_depth": None if z_batch_depth is None else int(z_batch_depth),
        "canonical_shape_tpczyx": [int(value) for value in canonical_shape_tpczyx],
        "chunks_tpczyx": [int(value) for value in chunks_tpczyx],
        "pyramid_factors_tpczyx": [
            [int(value) for value in level] for level in level_factors_tpczyx
        ],
        "base_progress": {
            "total_regions": int(base_total_regions),
            "completed_regions": 0,
        },
        "pyramid_progress": {},
        "swap_completed": False,
        "started_utc": now,
        "updated_utc": now,
        "completed_utc": None,
    }


def _ingestion_progress_record_matches(
    *,
    record: IngestionProgressRecord,
    source_path: Path,
    source_component: Optional[str],
    target_component: str,
    canonical_shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
    level_factors_tpczyx: tuple[CanonicalShapeTpczyx, ...],
    write_mode: str,
    z_batch_depth: Optional[int],
) -> bool:
    """Return whether a progress record matches current ingestion configuration.

    Parameters
    ----------
    record : dict
        Existing progress record.
    source_path : pathlib.Path
        Resolved source acquisition path.
    source_component : str, optional
        Source component path when reading from a store.
    target_component : str
        Destination component path for base writes.
    canonical_shape_tpczyx : tuple[int, int, int, int, int, int]
        Canonical output shape.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Effective canonical output chunks.
    level_factors_tpczyx : tuple[tuple[int, int, int, int, int, int], ...]
        Normalized absolute pyramid factors.
    write_mode : str
        Base write mode identifier.
    z_batch_depth : int, optional
        Source-aligned z-batch depth when applicable.

    Returns
    -------
    bool
        ``True`` when record configuration matches the active run.

    Raises
    ------
    None
        Malformed records return ``False``.
    """
    if str(record.get("schema", "")).strip() != _INGESTION_PROGRESS_SCHEMA:
        return False
    if str(record.get("status", "")).strip().lower() != "in_progress":
        return False
    if str(record.get("source_path", "")).strip() != str(source_path):
        return False
    record_source_component = record.get("source_component")
    normalized_record_source_component = (
        ""
        if record_source_component is None
        else str(record_source_component).strip()
    )
    normalized_source_component = (
        "" if source_component is None else str(source_component).strip()
    )
    if normalized_record_source_component != normalized_source_component:
        return False
    if str(record.get("target_component", "")).strip() != str(target_component):
        return False
    if str(record.get("write_mode", "")).strip() != str(write_mode):
        return False
    try:
        record_shape = tuple(int(value) for value in record.get("canonical_shape_tpczyx", []))
        record_chunks = tuple(int(value) for value in record.get("chunks_tpczyx", []))
        record_factors = tuple(
            tuple(int(value) for value in level)
            for level in record.get("pyramid_factors_tpczyx", [])
        )
    except Exception:
        return False
    if record_shape != tuple(int(value) for value in canonical_shape_tpczyx):
        return False
    if record_chunks != tuple(int(value) for value in chunks_tpczyx):
        return False
    if record_factors != tuple(
        tuple(int(value) for value in level) for level in level_factors_tpczyx
    ):
        return False
    record_z_batch = record.get("z_batch_depth")
    if z_batch_depth is None:
        return record_z_batch is None
    try:
        return int(record_z_batch) == int(z_batch_depth)
    except Exception:
        return False


def _set_ingestion_base_progress(
    *,
    record: IngestionProgressRecord,
    completed_regions: int,
    total_regions: int,
) -> None:
    """Update base-component progress counters in an ingestion record.

    Parameters
    ----------
    record : dict
        Mutable ingestion record payload.
    completed_regions : int
        Number of base regions completed.
    total_regions : int
        Total number of base regions.

    Returns
    -------
    None
        Record is modified in place.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    record["base_progress"] = {
        "completed_regions": int(max(0, completed_regions)),
        "total_regions": int(max(0, total_regions)),
    }
    record["updated_utc"] = _utc_now_iso()


def _set_ingestion_level_progress(
    *,
    record: IngestionProgressRecord,
    component: str,
    completed_regions: int,
    total_regions: int,
    shape_tpczyx: CanonicalShapeTpczyx,
    chunks_tpczyx: CanonicalShapeTpczyx,
    downsample_factors_tpczyx: CanonicalShapeTpczyx,
    source_component: str,
) -> None:
    """Update one pyramid-level progress entry in an ingestion record.

    Parameters
    ----------
    record : dict
        Mutable ingestion record payload.
    component : str
        Pyramid component path (for example ``"data_pyramid/level_1"``).
    completed_regions : int
        Number of written regions for this level.
    total_regions : int
        Total number of regions for this level.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Level shape in canonical axis order.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Level chunk shape in canonical axis order.
    downsample_factors_tpczyx : tuple[int, int, int, int, int, int]
        Absolute level downsample factors.
    source_component : str
        Source component used to generate this level.

    Returns
    -------
    None
        Record is modified in place.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    pyramid_progress = record.get("pyramid_progress")
    if not isinstance(pyramid_progress, dict):
        pyramid_progress = {}
        record["pyramid_progress"] = pyramid_progress
    pyramid_progress[str(component)] = {
        "completed_regions": int(max(0, completed_regions)),
        "total_regions": int(max(0, total_regions)),
        "shape_tpczyx": [int(value) for value in shape_tpczyx],
        "chunks_tpczyx": [int(value) for value in chunks_tpczyx],
        "downsample_factors_tpczyx": [
            int(value) for value in downsample_factors_tpczyx
        ],
        "source_component": str(source_component),
    }
    record["updated_utc"] = _utc_now_iso()


def _mark_ingestion_completed(*, record: IngestionProgressRecord) -> None:
    """Mark ingestion record status as completed.

    Parameters
    ----------
    record : dict
        Mutable ingestion record payload.

    Returns
    -------
    None
        Record is modified in place.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    now = _utc_now_iso()
    record["status"] = "completed"
    record["swap_completed"] = True
    record["updated_utc"] = now
    record["completed_utc"] = now


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
    start_regions_by_component: Optional[Dict[str, int]] = None,
    preserve_existing: bool = False,
    level_progress_callback: Optional[
        Callable[
            [
                str,
                int,
                int,
                CanonicalShapeTpczyx,
                CanonicalShapeTpczyx,
                CanonicalShapeTpczyx,
                str,
            ],
            None,
        ]
    ] = None,
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
    start_regions_by_component : dict[str, int], optional
        Optional mapping of component path to completed-region count for
        resumable level writes.
    preserve_existing : bool, default=False
        Whether to keep existing ``data_pyramid`` components and resume in
        place when their structure matches expected shape/chunks.
    level_progress_callback : callable, optional
        Optional callback invoked as
        ``callback(component, completed, total, shape, chunks, factors, source)``
        after successful write batches.

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
    base_dtype = np.dtype(root["data"].dtype)

    if not preserve_existing and "data_pyramid" in root:
        del root["data_pyramid"]
    root.require_group("data_pyramid")
    resume_offsets = dict(start_regions_by_component or {})

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
        level_progress_start = progress_start + int(
            ((level_index - 1) / total_downsample_levels)
            * max(0, progress_end - progress_start)
        )
        level_progress_end = progress_start + int(
            (level_index / total_downsample_levels)
            * max(0, progress_end - progress_start)
        )
        if progress_callback is not None:
            progress_callback(level_progress_start, message)

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
        root = zarr.open_group(str(store_path), mode="a")
        level_total_regions = _count_tpczyx_chunk_regions(
            shape_tpczyx=level_shape,
            chunks_tpczyx=level_chunks,
        )
        start_region_index = max(
            0,
            min(
                int(resume_offsets.get(component, 0)),
                int(level_total_regions),
            ),
        )
        should_overwrite_level = True
        if preserve_existing and _component_matches_shape_and_chunks(
            root=root,
            component=component,
            shape_tpczyx=level_shape,
            chunks_tpczyx=level_chunks,
        ):
            should_overwrite_level = False
        if should_overwrite_level:
            create_or_overwrite_array(
                root=root,
                name=component,
                shape=level_shape,
                chunks=level_chunks,
                dtype=base_dtype.name,
                overwrite=True,
            )
            start_region_index = 0

        def _emit_level_progress(completed: int, total: int) -> None:
            if level_progress_callback is None:
                return
            level_progress_callback(
                component,
                int(completed),
                int(total),
                level_shape,
                level_chunks,
                absolute_factors,
                source_component,
            )

        _write_dask_array_in_batches(
            array=downsampled,
            store_path=store_path,
            component=component,
            shape_tpczyx=level_shape,
            chunks_tpczyx=level_chunks,
            dtype_itemsize=int(base_dtype.itemsize),
            client=client,
            progress_callback=progress_callback,
            progress_start=level_progress_start,
            progress_end=level_progress_end,
            progress_label=f"Writing pyramid level {level_index}/{total_downsample_levels}",
            start_region_index=start_region_index,
            batch_completed_callback=_emit_level_progress,
        )

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
        Destination Zarr store path. ClearEx-managed stores are reused
        in-place; external Zarr/N5 sources are materialized into a sibling
        ClearEx-managed store; non-Zarr sources are materialized as
        ``data_store.zarr`` next to ``experiment.yml``.

    Raises
    ------
    None
        This helper does not raise custom exceptions.
    """
    override_path = str(os.environ.get("CLEAREX_OVERRIDE_ANALYSIS_STORE_PATH", "")).strip()
    if override_path:
        return Path(override_path).expanduser().resolve()

    source = Path(source_path).expanduser().resolve()
    if _is_zarr_like_path(source):
        if is_clearex_analysis_store(source):
            return source
        return resolve_external_analysis_store_path(source)
    return (experiment.path.parent / "data_store.zarr").resolve()


def _create_synthetic_experiment(
    *,
    source_path: Path,
    source_shape: tuple[int, ...],
    source_axes: AxesSpec,
) -> "NavigateExperiment":
    """Create a minimal synthetic experiment for direct-source materialization."""
    axes = tuple(source_axes or ())
    axis_sizes = {axis: int(source_shape[idx]) for idx, axis in enumerate(axes)}
    channel_count = max(1, int(axis_sizes.get("c", 1)))
    return NavigateExperiment(
        path=source_path,
        raw={"source_path": str(source_path), "synthetic": True},
        save_directory=source_path.parent,
        file_type=str(source_path.suffix).upper().lstrip(".") or "ZARR",
        microscope_name=None,
        image_mode=None,
        timepoints=max(1, int(axis_sizes.get("t", 1))),
        number_z_steps=max(1, int(axis_sizes.get("z", source_shape[-3] if len(source_shape) >= 3 else 1))),
        y_pixels=max(1, int(axis_sizes.get("y", source_shape[-2] if len(source_shape) >= 2 else 1))),
        x_pixels=max(1, int(axis_sizes.get("x", source_shape[-1] if len(source_shape) >= 1 else 1))),
        multiposition_count=max(1, int(axis_sizes.get("p", 1))),
        selected_channels=[
            NavigateChannel(name=f"channel_{idx}", laser=None, laser_index=None, exposure_ms=None, is_selected=True)
            for idx in range(channel_count)
        ],
        xy_pixel_size_um=None,
        z_step_um=None,
    )


def _legacy_n5_helper_python() -> Optional[str]:
    """Return a Python executable that still exposes ``zarr.N5Store``."""
    candidates: list[str] = []
    env_candidate = str(os.environ.get("CLEAREX_LEGACY_N5_PYTHON", "")).strip()
    if env_candidate:
        candidates.append(env_candidate)
    default_candidates = [
        "/opt/anaconda3/bin/python",
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for candidate in default_candidates:
        if candidate:
            candidates.append(str(candidate))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            probe = subprocess.run(
                [
                    candidate,
                    "-c",
                    "import zarr,sys; sys.exit(0 if hasattr(zarr, 'N5Store') else 1)",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        if probe.returncode == 0:
            return candidate
    return None


def _legacy_n5_helper_command_prefix() -> Optional[tuple[str, ...]]:
    """Return a command prefix that can run Python with ``zarr.N5Store``.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[str, ...], optional
        Command prefix ending in ``python``. Returns a direct Python executable
        when available, otherwise falls back to ``uv run --with zarr<3 python``
        when that probe succeeds.

    Notes
    -----
    The ``uv`` fallback avoids requiring users to pre-create a separate
    legacy environment in common setups where ClearEx already runs under uv.
    """
    legacy_python = _legacy_n5_helper_python()
    if legacy_python is not None:
        return (legacy_python,)

    uv_candidates: list[str] = []
    uv_from_path = shutil.which("uv")
    if uv_from_path:
        uv_candidates.append(str(uv_from_path))
    uv_tool_bin_dir = str(os.environ.get("UV_TOOL_BIN_DIR", "")).strip()
    if uv_tool_bin_dir:
        uv_candidates.append(str((Path(uv_tool_bin_dir).expanduser() / "uv")))
    uv_install_dir = str(os.environ.get("UV_INSTALL_DIR", "")).strip()
    if uv_install_dir:
        uv_candidates.append(str((Path(uv_install_dir).expanduser() / "uv")))

    seen: set[str] = set()
    for uv_executable in uv_candidates:
        if not uv_executable or uv_executable in seen:
            continue
        seen.add(uv_executable)
        probe_command = [
            uv_executable,
            "run",
            "--with",
            "zarr<3",
            "python",
            "-c",
            "import zarr,sys; sys.exit(0 if hasattr(zarr, 'N5Store') else 1)",
        ]
        try:
            probe = subprocess.run(
                probe_command,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        if probe.returncode == 0:
            return (
                uv_executable,
                "run",
                "--with",
                "zarr<3",
                "python",
            )
    return None


def _extract_client_scheduler_address(client: Optional["Client"]) -> Optional[str]:
    """Return scheduler address for a connected Dask client.

    Parameters
    ----------
    client : dask.distributed.Client, optional
        Connected client instance.

    Returns
    -------
    str, optional
        Scheduler address when it can be resolved from the client.

    Notes
    -----
    This helper is best-effort and never raises. It checks both direct client
    attributes and scheduler metadata for compatibility across distributed
    versions.
    """
    if client is None:
        return None

    try:
        scheduler = getattr(client, "scheduler", None)
        address = getattr(scheduler, "address", None)
        if isinstance(address, str) and address.strip():
            return address.strip()
    except Exception:
        pass

    try:
        scheduler_info = client.scheduler_info()
    except Exception:
        return None
    if not isinstance(scheduler_info, dict):
        return None
    address_value = scheduler_info.get("address")
    if isinstance(address_value, str) and address_value.strip():
        return address_value.strip()
    return None


def _materialize_n5_via_legacy_helper(
    *,
    experiment: "NavigateExperiment",
    source_path: Path,
    output_store_path: Path,
    chunks: CanonicalShapeTpczyx,
    pyramid_factors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    client: Optional["Client"] = None,
) -> Path:
    """Materialize an N5 source into an intermediate v2 ClearEx store.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    source_path : pathlib.Path
        Source N5 path.
    output_store_path : pathlib.Path
        Canonical analysis-store destination path.
    chunks : tuple[int, int, int, int, int, int]
        Canonical write chunks.
    pyramid_factors : tuple[tuple[int, ...], ...]
        Canonical pyramid factors.
    client : dask.distributed.Client, optional
        Active Dask client. When provided, the helper reconnects to this
        scheduler so legacy N5 writes honor the selected backend.

    Returns
    -------
    pathlib.Path
        Path to the produced legacy-v2 handoff store.
    """
    helper_command_prefix = _legacy_n5_helper_command_prefix()
    if helper_command_prefix is None:
        raise RuntimeError(
            "N5 ingestion requires a legacy Python environment with zarr.N5Store. "
            "Set CLEAREX_LEGACY_N5_PYTHON to a compatible interpreter or "
            "ensure `uv` is available so ClearEx can run the helper with "
            "`zarr<3` automatically."
        )

    legacy_output = resolve_legacy_v2_store_path(output_store_path)
    repo_root = Path(__file__).resolve().parents[3]
    command = [
        *helper_command_prefix,
        "-m",
        "clearex.io.n5_legacy_helper",
        "--experiment-path",
        str(experiment.path),
        "--source-path",
        str(source_path),
        "--output-store",
        str(legacy_output),
        "--chunks",
        ",".join(str(int(value)) for value in chunks),
        "--pyramid-factors",
        json.dumps([[int(value) for value in axis_levels] for axis_levels in pyramid_factors]),
    ]
    scheduler_address = _extract_client_scheduler_address(client)
    if scheduler_address:
        command.extend(
            [
                "--scheduler-address",
                str(scheduler_address),
            ]
        )
    subprocess.run(
        command,
        check=True,
        cwd=str(repo_root),
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        },
    )
    return legacy_output


def migrate_analysis_store(
    zarr_path: Union[str, Path],
    *,
    keep_backup: bool = True,
) -> Path:
    """Convert an existing ClearEx-managed store to Zarr v3 in place."""
    store_path = Path(zarr_path).expanduser().resolve()
    if not _is_zarr_like_path(store_path):
        raise ValueError(f"Path is not a Zarr store: {store_path}")
    if not is_clearex_analysis_store(store_path):
        raise ValueError(f"Path is not a ClearEx-managed analysis store: {store_path}")
    if detect_store_format(store_path) == 3:
        return store_path

    staging_path = resolve_staging_store_path(store_path)
    if staging_path.exists():
        shutil.rmtree(staging_path)

    source_root = zarr.open_group(str(store_path), mode="r")
    target_root = open_zarr_group(staging_path, mode="a", zarr_format=3)

    def _copy_group(source_group: Any, target_group: Any) -> None:
        target_group.attrs.update(to_jsonable(dict(getattr(source_group, "attrs", {}))))
        for array_key in sorted(source_group.array_keys()):
            source_array = source_group[array_key]
            chunks = getattr(source_array, "chunks", None)
            target_array = create_or_overwrite_array(
                root=target_group,
                name=str(array_key),
                shape=tuple(int(v) for v in source_array.shape),
                chunks=tuple(int(v) for v in chunks) if chunks is not None else None,
                dtype=source_array.dtype,
                overwrite=True,
            )
            da.to_zarr(da.from_zarr(source_array), target_array, compute=True)
            target_array.attrs.update(
                to_jsonable(dict(getattr(source_array, "attrs", {})))
            )
        for group_key in sorted(source_group.group_keys()):
            child_target = target_group.require_group(str(group_key))
            _copy_group(source_group[group_key], child_target)

    _copy_group(source_root, target_root)
    _ = replace_store_path(
        staging_path=staging_path,
        target_path=store_path,
        keep_backup=keep_backup,
    )
    return store_path


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
    force_rebuild: bool = False,
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
    force_rebuild : bool, default=False
        Whether to rebuild the canonical store even when a complete canonical
        store already exists.
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

    final_store_path = resolve_data_store_path(
        experiment=experiment, source_path=source_resolved
    )
    if (
        source_resolved.suffix.lower() == ".n5"
        and not is_clearex_analysis_store(source_resolved)
        and str(os.environ.get("CLEAREX_LEGACY_N5_ACTIVE", "")).strip() != "1"
    ):
        _emit_progress(10, "Materializing N5 source via legacy helper")
        legacy_store_path = _materialize_n5_via_legacy_helper(
            experiment=experiment,
            source_path=source_resolved,
            output_store_path=final_store_path,
            chunks=chunks,
            pyramid_factors=pyramid_factors,
            client=client,
        )
        migrated_legacy_store = migrate_analysis_store(
            legacy_store_path,
            keep_backup=False,
        )
        _ = replace_store_path(
            staging_path=migrated_legacy_store,
            target_path=final_store_path,
            keep_backup=False,
        )
        root = zarr.open_group(str(final_store_path), mode="r")
        data = root["data"]
        canonical_shape = _normalize_tpczyx_shape(tuple(int(size) for size in data.shape))
        canonical_chunks = _normalize_write_chunks(
            shape_tpczyx=canonical_shape,
            chunks=tuple(int(value) for value in (data.chunks or chunks)),
        )
        return MaterializedDataStore(
            source_path=source_resolved,
            store_path=final_store_path,
            source_component=None,
            source_image_info=ImageInfo(
                path=source_resolved,
                shape=canonical_shape,
                dtype=np.dtype(data.dtype),
                axes="TPCZYX",
                metadata={"legacy_n5_helper": True},
            ),
            data_image_info=ImageInfo(
                path=final_store_path,
                shape=canonical_shape,
                dtype=np.dtype(data.dtype),
                axes="TPCZYX",
                metadata={"component": "data", "legacy_n5_helper": True},
            ),
            chunks_tpczyx=canonical_chunks,
        )

    working_store_path = resolve_staging_store_path(final_store_path)
    write_client = client if _is_zarr_like_path(source_resolved) else None
    source_aligned_worker_count: Optional[int] = None
    source_aligned_worker_memory_limit_bytes: Optional[int] = None
    if write_client is not None:
        (
            source_aligned_worker_count,
            source_aligned_worker_memory_limit_bytes,
        ) = _detect_client_worker_resources(write_client)
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
        level_factors_tpczyx = _normalize_pyramid_level_factors(pyramid_factors)
        use_source_aligned_plane_writes = _should_use_source_aligned_plane_writes(
            array=canonical,
            shape_tpczyx=canonical_shape,
            target_chunks_tpczyx=normalized_chunks,
        )
        source_aligned_z_batch_depth: Optional[int] = None
        if use_source_aligned_plane_writes:
            source_aligned_z_batch_depth = _estimate_source_plane_batch_depth(
                shape_tpczyx=canonical_shape,
                target_chunks_tpczyx=normalized_chunks,
                dtype_itemsize=int(source_dtype.itemsize),
                worker_memory_limit_bytes=source_aligned_worker_memory_limit_bytes,
            )
            _emit_progress(
                45,
                "Preparing source-aligned canonical writes "
                f"(z_batch={source_aligned_z_batch_depth})",
            )
        else:
            with dask.config.set({"array.rechunk.method": "tasks"}):
                canonical = canonical.rechunk(normalized_chunks)
            _emit_progress(45, "Preparing chunk-batched canonical writes")

        if (not force_rebuild) and has_complete_canonical_data_store(final_store_path):
            _emit_progress(100, "Canonical data store is already complete")
            data_root = zarr.open_group(str(final_store_path), mode="r")
            data_chunks = tuple(
                int(value) for value in (data_root["data"].chunks or normalized_chunks)
            )
            return MaterializedDataStore(
                source_path=source_resolved,
                store_path=final_store_path,
                source_component=source_component,
                source_image_info=ImageInfo(
                    path=source_resolved,
                    shape=source_shape,
                    dtype=source_dtype,
                    axes=_format_axes_for_image_info(source_axes),
                    metadata=dict(source_meta),
                ),
                data_image_info=ImageInfo(
                    path=final_store_path,
                    shape=canonical_shape,
                    dtype=source_dtype,
                    axes="TPCZYX",
                    metadata={"component": "data"},
                ),
                chunks_tpczyx=_normalize_write_chunks(
                    shape_tpczyx=canonical_shape,
                    chunks=data_chunks,
                ),
            )

        store_path = working_store_path

        def _write_canonical_component(
            *,
            component: str,
            progress_start: int,
            progress_end: int,
            progress_label: str,
            start_region_index: int = 0,
            batch_completed_callback: Optional[Callable[[int, int], None]] = None,
        ) -> None:
            """Write canonical base data to one target component.

            Parameters
            ----------
            component : str
                Target Zarr component path.
            progress_start : int
                Stage progress lower bound.
            progress_end : int
                Stage progress upper bound.
            progress_label : str
                Human-readable progress prefix.
            start_region_index : int, default=0
                Number of initial base write regions to skip for resuming.
            batch_completed_callback : callable, optional
                Callback invoked after each successful batch.

            Returns
            -------
            None
                Side-effect writes only.
            """
            if (
                use_source_aligned_plane_writes
                and source_aligned_z_batch_depth is not None
            ):
                _write_dask_array_source_aligned_plane_batches(
                    array=canonical,
                    store_path=store_path,
                    component=component,
                    shape_tpczyx=canonical_shape,
                    z_batch_depth=source_aligned_z_batch_depth,
                    dtype_itemsize=int(source_dtype.itemsize),
                    client=write_client,
                    progress_callback=progress_callback,
                    progress_start=progress_start,
                    progress_end=progress_end,
                    progress_label=progress_label,
                    worker_count=source_aligned_worker_count,
                    worker_memory_limit_bytes=source_aligned_worker_memory_limit_bytes,
                    start_region_index=start_region_index,
                    batch_completed_callback=batch_completed_callback,
                )
                return
            _write_dask_array_in_batches(
                array=canonical,
                store_path=store_path,
                component=component,
                shape_tpczyx=canonical_shape,
                chunks_tpczyx=normalized_chunks,
                dtype_itemsize=int(source_dtype.itemsize),
                client=write_client,
                progress_callback=progress_callback,
                progress_start=progress_start,
                progress_end=progress_end,
                progress_label=progress_label,
                start_region_index=start_region_index,
                batch_completed_callback=batch_completed_callback,
            )

        write_mode = (
            "source_aligned_plane_batches"
            if use_source_aligned_plane_writes and source_aligned_z_batch_depth is not None
            else "chunk_region_batches"
        )
        if write_mode == "source_aligned_plane_batches":
            base_total_regions = _count_source_aligned_plane_regions(
                shape_tpczyx=canonical_shape,
                z_batch_depth=int(source_aligned_z_batch_depth or 1),
            )
        else:
            base_total_regions = _count_tpczyx_chunk_regions(
                shape_tpczyx=canonical_shape,
                chunks_tpczyx=normalized_chunks,
            )

        should_stage_same_component = False
        checkpoint_resume_supported = not should_stage_same_component
        root = zarr.open_group(str(store_path), mode="a")
        existing_progress_record = _read_ingestion_progress_record(root)
        resume_from_checkpoint = False
        base_start_region = 0
        if (
            not force_rebuild
            and
            checkpoint_resume_supported
            and existing_progress_record is not None
            and _ingestion_progress_record_matches(
                record=existing_progress_record,
                source_path=source_resolved,
                source_component=source_component,
                target_component="data",
                canonical_shape_tpczyx=canonical_shape,
                chunks_tpczyx=normalized_chunks,
                level_factors_tpczyx=level_factors_tpczyx,
                write_mode=write_mode,
                z_batch_depth=source_aligned_z_batch_depth,
            )
            and _component_matches_shape_and_chunks(
                root=root,
                component="data",
                shape_tpczyx=canonical_shape,
                chunks_tpczyx=normalized_chunks,
            )
        ):
            ingestion_record = dict(existing_progress_record)
            base_progress = ingestion_record.get("base_progress", {})
            if isinstance(base_progress, dict):
                try:
                    base_start_region = int(base_progress.get("completed_regions", 0))
                except Exception:
                    base_start_region = 0
            base_start_region = max(
                0,
                min(int(base_start_region), int(base_total_regions)),
            )
            resume_from_checkpoint = True
            _emit_progress(
                50,
                "Resuming canonical data writes "
                f"({base_start_region}/{base_total_regions} regions complete)",
            )
        else:
            ingestion_record = _create_ingestion_progress_record(
                source_path=source_resolved,
                source_component=source_component,
                target_component="data",
                canonical_shape_tpczyx=canonical_shape,
                chunks_tpczyx=normalized_chunks,
                level_factors_tpczyx=level_factors_tpczyx,
                write_mode=write_mode,
                z_batch_depth=source_aligned_z_batch_depth,
                base_total_regions=int(base_total_regions),
            )
            _write_ingestion_progress_record(
                store_path=store_path,
                record=ingestion_record,
            )

        def _persist_base_progress(completed: int, total: int) -> None:
            _set_ingestion_base_progress(
                record=ingestion_record,
                completed_regions=int(completed),
                total_regions=int(total),
            )
            _write_ingestion_progress_record(
                store_path=store_path,
                record=ingestion_record,
            )

        def _persist_level_progress(
            component: str,
            completed: int,
            total: int,
            level_shape_tpczyx: CanonicalShapeTpczyx,
            level_chunks_tpczyx: CanonicalShapeTpczyx,
            downsample_factors_tpczyx: CanonicalShapeTpczyx,
            level_source_component: str,
        ) -> None:
            _set_ingestion_level_progress(
                record=ingestion_record,
                component=component,
                completed_regions=int(completed),
                total_regions=int(total),
                shape_tpczyx=level_shape_tpczyx,
                chunks_tpczyx=level_chunks_tpczyx,
                downsample_factors_tpczyx=downsample_factors_tpczyx,
                source_component=level_source_component,
            )
            _write_ingestion_progress_record(
                store_path=store_path,
                record=ingestion_record,
            )

        initialize_analysis_store(
            experiment=experiment,
            zarr_path=store_path,
            overwrite=not resume_from_checkpoint,
            chunks=chunks,
            pyramid_factors=pyramid_factors,
            dtype=source_dtype.name,
            shape_tpczyx=canonical_shape,
        )
        _write_canonical_component(
            component="data",
            progress_start=55,
            progress_end=70,
            progress_label="Writing canonical data",
            start_region_index=base_start_region,
            batch_completed_callback=_persist_base_progress,
        )
        ingestion_record["swap_completed"] = True
        ingestion_record["updated_utc"] = _utc_now_iso()
        _write_ingestion_progress_record(
            store_path=store_path,
            record=ingestion_record,
        )

        start_regions_by_component: Dict[str, int] = {}
        if resume_from_checkpoint:
            pyramid_progress = ingestion_record.get("pyramid_progress", {})
            if isinstance(pyramid_progress, dict):
                for component, payload in pyramid_progress.items():
                    if not isinstance(payload, dict):
                        continue
                    try:
                        completed = int(payload.get("completed_regions", 0))
                    except Exception:
                        continue
                    start_regions_by_component[str(component)] = max(0, int(completed))

        _materialize_data_pyramid(
            store_path=store_path,
            base_chunks_tpczyx=normalized_chunks,
            pyramid_factors=pyramid_factors,
            client=client,
            progress_callback=progress_callback,
            progress_start=72,
            progress_end=96,
            start_regions_by_component=start_regions_by_component,
            preserve_existing=resume_from_checkpoint,
            level_progress_callback=_persist_level_progress,
        )
        _emit_progress(97, "Finalizing store metadata")

        _mark_ingestion_completed(record=ingestion_record)
        _write_ingestion_progress_record(
            store_path=store_path,
            record=ingestion_record,
        )

    _ = replace_store_path(
        staging_path=store_path,
        target_path=final_store_path,
        keep_backup=False,
    )

    root = zarr.open_group(str(final_store_path), mode="a")
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
    write_strategy = (
        "source_aligned_plane_batches"
        if use_source_aligned_plane_writes
        else "chunk_region_batches"
    )
    root["data"].attrs.update(
        {
            "source_path": source_metadata_path,
            "source_axes": source_axes_attr,
            "voxel_size_um_zyx": voxel_size_um_zyx,
            "materialization_write_strategy": write_strategy,
            "source_aligned_z_batch_depth": (
                int(source_aligned_z_batch_depth)
                if source_aligned_z_batch_depth is not None
                else None
            ),
            "source_aligned_worker_count": (
                int(source_aligned_worker_count)
                if source_aligned_worker_count is not None
                else None
            ),
            "source_aligned_worker_memory_limit_bytes": (
                int(source_aligned_worker_memory_limit_bytes)
                if source_aligned_worker_memory_limit_bytes is not None
                else None
            ),
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
            "materialization_write_strategy": write_strategy,
            "source_aligned_z_batch_depth": (
                int(source_aligned_z_batch_depth)
                if source_aligned_z_batch_depth is not None
                else None
            ),
            "source_aligned_worker_count": (
                int(source_aligned_worker_count)
                if source_aligned_worker_count is not None
                else None
            ),
            "source_aligned_worker_memory_limit_bytes": (
                int(source_aligned_worker_memory_limit_bytes)
                if source_aligned_worker_memory_limit_bytes is not None
                else None
            ),
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
        path=final_store_path,
        shape=canonical_shape,
        dtype=source_dtype,
        axes="TPCZYX",
        metadata={"component": "data"},
    )
    _emit_progress(100, "Materialization complete")
    return MaterializedDataStore(
        source_path=source_resolved,
        store_path=final_store_path,
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


class ExperimentDataResolutionError(FileNotFoundError):
    """Raised when acquisition data cannot be located for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata for the failed lookup.
    searched_directories : tuple[pathlib.Path, ...]
        Directories inspected for candidate acquisition data.
    """

    def __init__(
        self,
        *,
        experiment: NavigateExperiment,
        searched_directories: tuple[Path, ...],
    ) -> None:
        self.experiment = experiment
        self.searched_directories = searched_directories
        searched_lines = "\n".join(
            f"- {directory}" for directory in searched_directories
        )
        super().__init__(
            "No source data candidates found for "
            f"file_type={experiment.file_type}.\n"
            "Searched directories:\n"
            f"{searched_lines}"
        )


def _looks_like_windows_absolute_path(path_text: str) -> bool:
    """Return whether text appears to encode a Windows absolute path.

    Parameters
    ----------
    path_text : str
        Path text to inspect.

    Returns
    -------
    bool
        ``True`` when ``path_text`` starts with a drive letter or UNC prefix.
    """
    normalized = str(path_text).strip()
    return bool(re.match(r"^[A-Za-z]:[\\/]", normalized)) or normalized.startswith(
        "\\\\"
    )


def _resolve_experiment_save_directory(path_text: Any, *, experiment_dir: Path) -> Path:
    """Resolve the acquisition save directory recorded in experiment metadata.

    Parameters
    ----------
    path_text : Any
        Raw ``Saving.save_directory`` payload.
    experiment_dir : pathlib.Path
        Directory containing ``experiment.yml``.

    Returns
    -------
    pathlib.Path
        Resolved save-directory path. Windows absolute paths are preserved as
        written when running on non-Windows platforms.
    """
    normalized = str(path_text or str(experiment_dir)).strip()
    if not normalized:
        return experiment_dir.resolve()

    directory = Path(normalized).expanduser()
    if directory.is_absolute():
        return directory.resolve()
    if _looks_like_windows_absolute_path(normalized):
        return directory
    return (experiment_dir / directory).resolve()


def _search_directory_identity(path: Path) -> str:
    """Return a deterministic identity string for a search directory.

    Parameters
    ----------
    path : pathlib.Path
        Search-directory candidate.

    Returns
    -------
    str
        Normalized identity used for de-duplication.
    """
    text = str(path)
    if _looks_like_windows_absolute_path(text):
        return text.lower()
    try:
        return str(path.resolve())
    except OSError:
        return text


def experiment_data_search_directories(
    experiment: NavigateExperiment,
    *,
    search_directory: Optional[Union[str, Path]] = None,
) -> list[Path]:
    """Return ordered directories to inspect for acquisition data.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    search_directory : str or pathlib.Path, optional
        Explicit user-selected override directory. This is searched first.

    Returns
    -------
    list[pathlib.Path]
        Ordered search roots with duplicates removed.
    """
    directories: list[Path] = []
    if search_directory is not None:
        override_path = Path(search_directory).expanduser().resolve()
        directories.append(override_path)

    directories.append(experiment.save_directory)
    directories.append(experiment.path.parent.resolve())

    ordered: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        identity = _search_directory_identity(directory)
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(directory)
    return ordered


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


def _parse_zoom_factor(value: Any) -> Optional[float]:
    """Parse microscope zoom factor from Navigate metadata values.

    Parameters
    ----------
    value : Any
        Raw zoom value (for example ``"0.63x"`` or ``0.63``).

    Returns
    -------
    float, optional
        Parsed positive zoom factor, or ``None`` when unavailable.

    Raises
    ------
    None
        Invalid values are handled internally and return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if np.isfinite(parsed) and parsed > 0:
            return float(parsed)
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if match is None:
        return None
    try:
        parsed = float(match.group(1))
    except ValueError:
        return None
    if not np.isfinite(parsed) or parsed <= 0:
        return None
    return float(parsed)


def _parse_binning_xy(value: Any) -> tuple[float, float]:
    """Parse camera binning value into positive ``(x, y)`` factors.

    Parameters
    ----------
    value : Any
        Raw binning descriptor (for example ``"2x2"`` or ``[2, 2]``).

    Returns
    -------
    tuple[float, float]
        Parsed binning factors. Defaults to ``(1.0, 1.0)`` when unavailable.

    Raises
    ------
    None
        Invalid values are handled internally and return defaults.
    """
    if value is None:
        return 1.0, 1.0
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            bx = float(value[0])
            by = float(value[1])
        except (TypeError, ValueError):
            return 1.0, 1.0
        if bx > 0 and by > 0 and np.isfinite(bx) and np.isfinite(by):
            return float(bx), float(by)
        return 1.0, 1.0

    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return 1.0, 1.0
    match = re.match(r"([0-9]*\.?[0-9]+)[x,]([0-9]*\.?[0-9]+)$", text)
    if match is None:
        parsed = _safe_optional_float(text)
        if parsed is None or parsed <= 0:
            return 1.0, 1.0
        return float(parsed), float(parsed)
    try:
        bx = float(match.group(1))
        by = float(match.group(2))
    except ValueError:
        return 1.0, 1.0
    if bx <= 0 or by <= 0 or not np.isfinite(bx) or not np.isfinite(by):
        return 1.0, 1.0
    return float(bx), float(by)


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
    microscope_state: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Infer sample-space XY pixel size in microns.

    Parameters
    ----------
    camera : dict[str, Any]
        ``CameraParameters`` mapping from experiment metadata.
    microscope_name : str, optional
        Active microscope profile name.
    microscope_state : dict[str, Any], optional
        ``MicroscopeState`` mapping used for zoom inference.

    Returns
    -------
    float, optional
        Estimated XY pixel size in microns, or ``None`` when unavailable.

    Notes
    -----
    This uses profile-specific values when available. Preferred estimate is
    ``fov_x / img_x_pixels`` (or ``fov_y / img_y_pixels``). If unavailable,
    fallback uses ``pixel_size / zoom`` adjusted by camera binning. Finally,
    ``pixel_size`` is used when zoom is missing.
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
    zoom = _parse_zoom_factor(
        (microscope_state or {}).get("zoom") if microscope_state else None
    )
    binning_x, binning_y = _parse_binning_xy(
        profile.get("binning", camera.get("binning"))
    )
    if pixel_size is not None and pixel_size > 0 and zoom is not None:
        return float(pixel_size / zoom * ((binning_x + binning_y) / 2.0))

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

    save_directory = _resolve_experiment_save_directory(
        saving.get("save_directory", str(experiment_path.parent)),
        experiment_dir=experiment_path.parent,
    )

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
        microscope_state=state,
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


def _find_experiment_data_candidates_in_directory(
    *,
    base: Path,
    file_type: str,
) -> list[Path]:
    """Find acquisition candidates within one directory root.

    Parameters
    ----------
    base : pathlib.Path
        Directory root to inspect.
    file_type : str
        Normalized acquisition file type.

    Returns
    -------
    list[pathlib.Path]
        Candidate paths found under ``base``.
    """
    if not base.exists() or not base.is_dir():
        return []

    if file_type == "H5":
        return sorted(
            [p for p in base.glob("*") if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}],
            key=_h5_sort_key,
        )

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

    return [
        *sorted([p for p in base.glob("*.zarr") if p.is_dir()]),
        *sorted([p for p in base.glob("*.n5") if p.is_dir()]),
        *sorted(
            [p for p in base.glob("*") if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}],
            key=_h5_sort_key,
        ),
        *fallback_tiffs,
    ]


def find_experiment_data_candidates(
    experiment: NavigateExperiment,
    *,
    search_directory: Optional[Union[str, Path]] = None,
) -> list[Path]:
    """Find candidate acquisition data files/stores for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    search_directory : str or pathlib.Path, optional
        Explicit user-selected override directory searched ahead of metadata
        defaults.

    Returns
    -------
    list[pathlib.Path]
        Candidate paths sorted by deterministic preference.
    """
    file_type = _normalize_file_type(experiment.file_type)
    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in experiment_data_search_directories(
        experiment,
        search_directory=search_directory,
    ):
        for candidate in _find_experiment_data_candidates_in_directory(
            base=directory,
            file_type=file_type,
        ):
            identity = str(candidate.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate.resolve())
    return candidates


def resolve_experiment_data_path(
    experiment: NavigateExperiment,
    *,
    search_directory: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve primary acquisition data path from experiment metadata.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    search_directory : str or pathlib.Path, optional
        Explicit user-selected override directory searched ahead of metadata
        defaults.

    Returns
    -------
    pathlib.Path
        Selected source data path.

    Raises
    ------
    FileNotFoundError
        If no compatible source data can be found.
    """
    candidates = find_experiment_data_candidates(
        experiment,
        search_directory=search_directory,
    )
    if not candidates:
        raise ExperimentDataResolutionError(
            experiment=experiment,
            searched_directories=tuple(
                experiment_data_search_directories(
                    experiment,
                    search_directory=search_directory,
                )
            ),
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
    target_zarr_format = int(str(os.environ.get("CLEAREX_TARGET_ZARR_FORMAT", "3")))

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

    root = open_zarr_group(
        output_path,
        mode="a",
        zarr_format=target_zarr_format if detect_store_format(output_path) is None else None,
    )
    root.require_group("results")
    root.require_group("provenance")
    spatial_calibration_payload = spatial_calibration_to_dict(
        spatial_calibration_from_dict(root.attrs.get(_SPATIAL_CALIBRATION_ATTR))
    )
    if "data" in root:
        if overwrite:
            clear_component(root, "data")
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
                    _SPATIAL_CALIBRATION_ATTR: spatial_calibration_payload,
                    "chunk_shape_tpczyx": existing_chunks,
                    "configured_chunks_tpczyx": [
                        int(chunk) for chunk in requested_chunks
                    ],
                    "resolution_pyramid_factors_tpczyx": pyramid_payload,
                    "voxel_size_um_zyx": voxel_size_um_zyx,
                }
            )
            return output_path

    create_or_overwrite_array(
        root=root,
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
            _SPATIAL_CALIBRATION_ATTR: spatial_calibration_payload,
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
    gpu_enabled: bool = False,
    gpu_device_ids: Optional[Sequence[Union[int, str]]] = None,
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
    gpu_enabled : bool, default=False
        Whether to launch a GPU-pinned local cluster with one worker process
        per assigned CUDA device.
    gpu_device_ids : sequence of int or str, optional
        Explicit CUDA device identifiers (for example ``[0, 1, 2, 3]``).
        When omitted and ``gpu_enabled`` is true, devices are auto-detected
        from ``nvidia-smi``.

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

    if gpu_enabled:
        from dask.distributed import Nanny, Scheduler
        from distributed.deploy.spec import SpecCluster

        assigned_gpu_ids = _normalize_gpu_device_ids(gpu_device_ids)
        if not assigned_gpu_ids:
            assigned_gpu_ids = _detect_visible_gpu_device_ids()
        if not assigned_gpu_ids:
            raise RuntimeError(
                "GPU-aware LocalCluster startup requested, but no CUDA devices "
                "were detected via nvidia-smi."
            )

        worker_count = (
            max(1, int(n_workers))
            if n_workers is not None
            else len(assigned_gpu_ids)
        )
        worker_gpu_ids = [
            str(assigned_gpu_ids[idx % len(assigned_gpu_ids)])
            for idx in range(worker_count)
        ]
        local_directory_text = (
            str(local_directory) if local_directory is not None else None
        )
        cuda_library_paths = _detect_cuda_library_paths()
        worker_library_env_updates = _build_library_path_environment_updates(
            cuda_library_paths
        )
        for env_key, env_value in worker_library_env_updates.items():
            os.environ[env_key] = env_value

        workers: Dict[str, Dict[str, Any]] = {}
        for worker_index, gpu_id in enumerate(worker_gpu_ids):
            worker_env: Dict[str, str] = {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "CLEAREX_GPU_DEVICE_ID": str(gpu_id),
                "CLEAREX_GPU_WORKER_INDEX": str(worker_index),
            }
            worker_env.update(worker_library_env_updates)
            options: Dict[str, Any] = {
                "nthreads": max(1, int(threads_per_worker)),
                "memory_limit": memory_limit,
                "env": worker_env,
                "resources": {"GPU": 1.0},
            }
            if local_directory_text:
                options["local_directory"] = local_directory_text
            workers[f"gpu-worker-{worker_index:02d}"] = {
                "cls": Nanny,
                "options": options,
            }

        scheduler_options: Dict[str, Any] = {}
        if dashboard_address is not None:
            scheduler_options["dashboard_address"] = dashboard_address
        cluster = SpecCluster(
            scheduler={"cls": Scheduler, "options": scheduler_options},
            workers=workers,
            asynchronous=False,
        )
        return Client(cluster)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        local_directory=str(local_directory) if local_directory is not None else None,
        dashboard_address=dashboard_address,
    )
    return Client(cluster)


def _normalize_gpu_device_ids(
    gpu_device_ids: Optional[Sequence[Union[int, str]]],
) -> list[str]:
    """Normalize explicit CUDA device identifiers.

    Parameters
    ----------
    gpu_device_ids : sequence[int | str], optional
        Raw GPU identifiers from runtime config.

    Returns
    -------
    list[str]
        Distinct normalized device identifiers in original order.
    """
    if gpu_device_ids is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in gpu_device_ids:
        text = str(value).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _library_path_env_vars_for_platform(
    *,
    os_name: Optional[str] = None,
    platform: Optional[str] = None,
) -> tuple[str, ...]:
    """Return dynamic-library environment variable names for current platform.

    Parameters
    ----------
    os_name : str, optional
        Override for ``os.name`` used by tests.
    platform : str, optional
        Override for ``sys.platform`` used by tests.

    Returns
    -------
    tuple[str, ...]
        Ordered environment variable names used for runtime library lookup.
    """
    effective_os_name = str(os.name if os_name is None else os_name).strip().lower()
    effective_platform = str(
        sys.platform if platform is None else platform
    ).strip().lower()
    if effective_os_name == "nt":
        return ("PATH",)
    if effective_platform == "darwin":
        return ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")
    return ("LD_LIBRARY_PATH",)


def _split_library_path_entries(value: str) -> list[str]:
    """Split a platform library-path string into normalized entries.

    Parameters
    ----------
    value : str
        Raw library-path text.

    Returns
    -------
    list[str]
        Non-empty path entries in input order.
    """
    text = str(value).strip()
    if not text:
        return []
    return [entry.strip() for entry in text.split(os.pathsep) if entry.strip()]


def _build_library_path_environment_updates(
    discovered_paths: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    env_var_names: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Build environment updates for runtime library search paths.

    Parameters
    ----------
    discovered_paths : sequence of str
        Candidate runtime library directories to prioritize.
    env : dict[str, str], optional
        Environment mapping source used for inherited path discovery.
        Defaults to ``os.environ``.
    env_var_names : sequence of str, optional
        Explicit path variable names. Defaults to platform-appropriate
        variables from :func:`_library_path_env_vars_for_platform`.

    Returns
    -------
    dict[str, str]
        Environment updates keyed by selected path variable names.
    """
    effective_env = os.environ if env is None else env
    selected_env_vars = tuple(
        str(value).strip()
        for value in (
            _library_path_env_vars_for_platform()
            if env_var_names is None
            else tuple(env_var_names)
        )
        if str(value).strip()
    )
    inherited_entries: list[str] = []
    for env_key in selected_env_vars:
        inherited_entries.extend(
            _split_library_path_entries(str(effective_env.get(env_key, "")))
        )
    merged_entries = _deduplicate_library_path_entries(
        [*list(discovered_paths), *inherited_entries]
    )
    if not merged_entries:
        return {}
    joined = os.pathsep.join(merged_entries)
    return {env_key: joined for env_key in selected_env_vars}


def _detect_cuda_library_paths() -> list[str]:
    """Return CUDA/cuDNN library directories useful for worker environments.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Existing library directories that contain CUDA runtime, NVRTC, or
        cuDNN shared objects. Returns an empty list when no candidate path is
        found.
    """
    candidate_dirs: list[Path] = []
    nvidia_runtime_modules: tuple[str, ...] = (
        "nvidia.cuda_nvrtc",
        "nvidia.cuda_runtime",
        "nvidia.cudnn",
    )
    for module_name in nvidia_runtime_modules:
        try:
            module = __import__(module_name, fromlist=["__file__"])
            module_paths: list[Path] = []
            module_file = getattr(module, "__file__", None)
            if module_file:
                module_paths.append(Path(str(module_file)).resolve().parent)
            module_search_paths = getattr(module, "__path__", None)
            if module_search_paths is not None:
                module_paths.extend(
                    Path(str(path_entry)).resolve()
                    for path_entry in list(module_search_paths)
                )
        except Exception:
            continue
        for module_path in module_paths:
            candidate_dirs.append(module_path / "lib")
            candidate_dirs.append(module_path / "bin")
    for env_key in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        root_text = str(os.environ.get(env_key, "")).strip()
        if not root_text:
            continue
        root = Path(root_text)
        candidate_dirs.append(root / "lib64")
        candidate_dirs.append(root / "lib")
        candidate_dirs.append(root / "bin")

    if os.name == "nt":
        marker_patterns = ("cudart64*.dll", "nvrtc64*.dll", "cudnn*.dll")
    elif sys.platform == "darwin":
        marker_patterns = ("libcudart*.dylib", "libnvrtc*.dylib", "libcudnn*.dylib")
    else:
        marker_patterns = ("libcudart.so*", "libnvrtc.so*", "libcudnn*.so*")

    discovered: list[str] = []
    seen: set[str] = set()
    for directory in candidate_dirs:
        try:
            resolved = directory.resolve()
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        has_runtime_lib = any(
            any(resolved.glob(pattern)) for pattern in marker_patterns
        )
        if not has_runtime_lib:
            continue
        text = str(resolved)
        if text in seen:
            continue
        seen.add(text)
        discovered.append(text)
    return discovered


def _deduplicate_library_path_entries(entries: Sequence[str]) -> list[str]:
    """Deduplicate library path entries while preserving order.

    Parameters
    ----------
    entries : sequence of str
        Candidate path entries.

    Returns
    -------
    list[str]
        Distinct non-empty normalized entries in first-seen order.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for value in entries:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _detect_visible_gpu_device_ids() -> list[str]:
    """Detect locally visible CUDA device indices via ``nvidia-smi``.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        CUDA device indices as strings (for example ``["0", "1"]``).

    Notes
    -----
    Detection failures are non-fatal and return an empty list.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return _normalize_gpu_device_ids(result.stdout.splitlines())


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
