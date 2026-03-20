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

"""BaSiCPy-backed flatfield correction on canonical 6D Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from itertools import product
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union, cast

import dask
import dask.array as da
from dask import delayed
import numpy as np
from numpy.typing import NDArray
import zarr

from clearex.io.provenance import register_latest_output_reference
from clearex.io.zarr_storage import write_dask_array

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]
Float32Array = NDArray[np.float32]
RegionBounds = tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]
SpatialBounds = tuple[tuple[int, int], tuple[int, int]]
CanonicalShapeTpczyx = tuple[int, int, int, int, int, int]
RESUME_SCHEMA_VERSION = "clearex.flatfield.resume.v1"
CHECKPOINT_GROUP_NAME = "checkpoint"


@dataclass(frozen=True)
class FlatfieldProfileResult:
    """Fitted BaSiC profile for one ``(position, channel)`` pair.

    Attributes
    ----------
    position_index : int
        Zero-based position index.
    channel_index : int
        Zero-based channel index.
    flatfield_yx : numpy.ndarray
        Estimated flatfield image in ``(y, x)`` order.
    darkfield_yx : numpy.ndarray
        Estimated darkfield image in ``(y, x)`` order.
    baseline_tz : numpy.ndarray
        Estimated baseline values reshaped into ``(t, z)`` order.
    """

    position_index: int
    channel_index: int
    flatfield_yx: Float32Array
    darkfield_yx: Float32Array
    baseline_tz: Float32Array


@dataclass(frozen=True)
class FlatfieldTileSpec:
    """Tile specification for one flatfield fit task.

    Attributes
    ----------
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    tile_y_index : int
        Tile-grid row index in ``y``.
    tile_x_index : int
        Tile-grid column index in ``x``.
    y_core_bounds : tuple[int, int]
        Non-overlapping output bounds in ``y``.
    x_core_bounds : tuple[int, int]
        Non-overlapping output bounds in ``x``.
    y_read_bounds : tuple[int, int]
        Read bounds in ``y`` including overlap halo.
    x_read_bounds : tuple[int, int]
        Read bounds in ``x`` including overlap halo.
    """

    position_index: int
    channel_index: int
    tile_y_index: int
    tile_x_index: int
    y_core_bounds: tuple[int, int]
    x_core_bounds: tuple[int, int]
    y_read_bounds: tuple[int, int]
    x_read_bounds: tuple[int, int]


@dataclass(frozen=True)
class FlatfieldTileFitResult:
    """One tile-level flatfield fit payload.

    Attributes
    ----------
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    tile_y_index : int
        Tile-grid row index in ``y``.
    tile_x_index : int
        Tile-grid column index in ``x``.
    y_bounds : tuple[int, int]
        Destination bounds in ``y`` for payload arrays.
    x_bounds : tuple[int, int]
        Destination bounds in ``x`` for payload arrays.
    flatfield_payload_yx : numpy.ndarray
        Tile payload in ``(y, x)`` order. Contains either direct estimates
        (no blending) or weighted contributions (blending mode).
    darkfield_payload_yx : numpy.ndarray
        Tile payload in ``(y, x)`` order. Contains either direct estimates
        (no blending) or weighted contributions (blending mode).
    baseline_tz : numpy.ndarray
        Baseline estimate for the tile in ``(t, z)`` order.
    weight_payload_yx : numpy.ndarray, optional
        Tile blending weights in ``(y, x)`` order. ``None`` when blending is
        disabled.
    """

    position_index: int
    channel_index: int
    tile_y_index: int
    tile_x_index: int
    y_bounds: tuple[int, int]
    x_bounds: tuple[int, int]
    flatfield_payload_yx: Float32Array
    darkfield_payload_yx: Float32Array
    baseline_tz: Float32Array
    weight_payload_yx: Optional[Float32Array]


@dataclass(frozen=True)
class FlatfieldSummary:
    """Summary metadata for one flatfield-correction run.

    Attributes
    ----------
    component : str
        Output latest group component path.
    data_component : str
        Corrected data array component path.
    flatfield_component : str
        Flatfield artifact component path.
    darkfield_component : str
        Darkfield artifact component path.
    baseline_component : str
        Baseline artifact component path.
    source_component : str
        Source data component used as input.
    profile_count : int
        Number of fitted ``(position, channel)`` profiles.
    transformed_volumes : int
        Number of transformed ``(time, position, channel)`` volumes.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Chunk shape used for corrected output.
    output_dtype : str
        Saved corrected output dtype.
    basicpy_version : str, optional
        Installed BaSiCPy package version when available.
    """

    component: str
    data_component: str
    flatfield_component: str
    darkfield_component: str
    baseline_component: str
    source_component: str
    profile_count: int
    transformed_volumes: int
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]
    output_dtype: str
    basicpy_version: Optional[str]


@dataclass(frozen=True)
class FlatfieldTransformSpec:
    """Transform-task specification for one corrected output chunk.

    Attributes
    ----------
    t_index : int
        Time index in ``t``.
    p_index : int
        Position index in ``p``.
    c_index : int
        Channel index in ``c``.
    y_chunk_index : int
        Y-axis chunk-grid index.
    x_chunk_index : int
        X-axis chunk-grid index.
    read_region : RegionBounds
        Expanded read region including overlap halo.
    core_region : RegionBounds
        Non-overlapping write region.
    """

    t_index: int
    p_index: int
    c_index: int
    y_chunk_index: int
    x_chunk_index: int
    read_region: RegionBounds
    core_region: RegionBounds


@dataclass(frozen=True)
class FlatfieldOutputLayout:
    """Prepared output layout and checkpoint metadata for one run.

    Attributes
    ----------
    component : str
        Latest-group component path.
    data_component : str
        Corrected data component path.
    flatfield_component : str
        Flatfield artifact component path.
    darkfield_component : str
        Darkfield artifact component path.
    baseline_component : str
        Baseline artifact component path.
    checkpoint_component : str
        Resume-checkpoint group component path.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Output chunk shape in canonical axis order.
    fit_grid_yx : tuple[int, int]
        Fit tile grid shape as ``(n_y_tiles, n_x_tiles)``.
    transform_grid_yx : tuple[int, int]
        Transform chunk grid shape as ``(n_y_chunks, n_x_chunks)``.
    fit_mode : str
        Effective fit mode.
    blend_tiles_effective : bool
        Whether blended tiled reduction is active.
    parameter_fingerprint : str
        Stable hash of effective flatfield parameters.
    resumed : bool
        Whether this run reuses an existing checkpoint.
    """

    component: str
    data_component: str
    flatfield_component: str
    darkfield_component: str
    baseline_component: str
    checkpoint_component: str
    shape_tpczyx: tuple[int, int, int, int, int, int]
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]
    fit_grid_yx: tuple[int, int]
    transform_grid_yx: tuple[int, int]
    fit_mode: str
    blend_tiles_effective: bool
    parameter_fingerprint: str
    resumed: bool


def _basicpy_version() -> Optional[str]:
    """Return installed BaSiCPy package version when available.

    Parameters
    ----------
    None

    Returns
    -------
    str, optional
        Installed version string, otherwise ``None``.
    """
    for package_name in ("BaSiCPy", "basicpy"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue
    return None


def _load_basic_class() -> Any:
    """Import and return the BaSiCPy ``BaSiC`` class.

    Parameters
    ----------
    None

    Returns
    -------
    Any
        Imported ``BaSiC`` class.

    Raises
    ------
    ImportError
        If BaSiCPy is unavailable in the current environment.
    """
    try:
        module = import_module("basicpy")
    except ImportError as exc:
        raise ImportError(
            "Flatfield correction requires the 'basicpy' package, but it is not "
            "available in the current environment."
        ) from exc
    return getattr(module, "BaSiC")


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flatfield runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate flatfield parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized parameter mapping with concrete types.

    Raises
    ------
    ValueError
        If overlap shape or scalar values are invalid.
    """
    normalized = dict(parameters)
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["force_rerun"] = bool(normalized.get("force_rerun", False))
    normalized["chunk_basis"] = str(normalized.get("chunk_basis", "3d")).strip() or "3d"
    normalized["detect_2d_per_slice"] = bool(
        normalized.get("detect_2d_per_slice", False)
    )
    normalized["use_map_overlap"] = bool(normalized.get("use_map_overlap", True))
    overlap_zyx = normalized.get("overlap_zyx", [0, 32, 32])
    if not isinstance(overlap_zyx, (tuple, list)) or len(overlap_zyx) != 3:
        raise ValueError("flatfield overlap_zyx must define (z, y, x) overlap values.")
    normalized["overlap_zyx"] = tuple(max(0, int(v)) for v in overlap_zyx)
    normalized["memory_overhead_factor"] = float(
        normalized.get("memory_overhead_factor", 2.0)
    )
    if normalized["memory_overhead_factor"] <= 0:
        raise ValueError("flatfield memory_overhead_factor must be greater than zero.")
    normalized["get_darkfield"] = bool(normalized.get("get_darkfield", False))
    normalized["smoothness_flatfield"] = float(
        normalized.get("smoothness_flatfield", 1.0)
    )
    if normalized["smoothness_flatfield"] <= 0:
        raise ValueError("flatfield smoothness_flatfield must be greater than zero.")
    normalized["working_size"] = max(1, int(normalized.get("working_size", 128)))
    normalized["is_timelapse"] = bool(normalized.get("is_timelapse", False))
    fit_mode = str(normalized.get("fit_mode", "tiled")).strip().lower().replace("-", "_")
    if fit_mode not in {"tiled", "full_volume"}:
        raise ValueError("flatfield fit_mode must be 'tiled' or 'full_volume'.")
    normalized["fit_mode"] = fit_mode
    fit_tile_shape = normalized.get("fit_tile_shape_yx", [256, 256])
    if (
        not isinstance(fit_tile_shape, (tuple, list))
        or len(fit_tile_shape) != 2
    ):
        raise ValueError(
            "flatfield fit_tile_shape_yx must define tile sizes in (y, x) order."
        )
    fit_tile_shape_yx = (max(1, int(fit_tile_shape[0])), max(1, int(fit_tile_shape[1])))
    normalized["fit_tile_shape_yx"] = fit_tile_shape_yx
    normalized["blend_tiles"] = bool(normalized.get("blend_tiles", False))
    return normalized


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Current UTC timestamp.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    """Recursively convert values into JSON-serializable primitives.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    Any
        JSON-serializable representation.
    """
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resume_parameter_payload(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic parameter payload used for resume matching.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Normalized flatfield parameters.

    Returns
    -------
    dict[str, Any]
        JSON-serializable payload excluding run-control flags.
    """
    payload = {
        str(key): value
        for key, value in dict(parameters).items()
        if str(key) != "force_rerun"
    }
    return cast(dict[str, Any], _to_jsonable(payload))


def _resume_parameter_fingerprint(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Return the serialized payload and SHA256 digest for resume checks.

    Parameters
    ----------
    payload : mapping[str, Any]
        JSON-serializable parameter payload.

    Returns
    -------
    tuple[str, str]
        Canonical JSON string and hexadecimal digest.
    """
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, digest


def _fit_grid_shape(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    fit_tile_shape_yx: tuple[int, int],
) -> tuple[int, int]:
    """Return the tile-grid shape for tiled flatfield fitting.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    fit_tile_shape_yx : tuple[int, int]
        Requested tile shape in ``(y, x)`` order.

    Returns
    -------
    tuple[int, int]
        Tile-grid shape as ``(n_y_tiles, n_x_tiles)``.
    """
    y_count = int(shape_tpczyx[4])
    x_count = int(shape_tpczyx[5])
    tile_y = max(1, min(int(fit_tile_shape_yx[0]), y_count))
    tile_x = max(1, min(int(fit_tile_shape_yx[1]), x_count))
    return (
        len(_axis_chunk_bounds(y_count, tile_y)),
        len(_axis_chunk_bounds(x_count, tile_x)),
    )


def _transform_grid_shape(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    chunks_tpczyx: tuple[int, int, int, int, int, int],
) -> tuple[int, int]:
    """Return the transform chunk-grid shape in spatial axes.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Chunk shape in canonical axis order.

    Returns
    -------
    tuple[int, int]
        Transform grid shape as ``(n_y_chunks, n_x_chunks)``.
    """
    return (
        len(_axis_chunk_bounds(int(shape_tpczyx[4]), int(chunks_tpczyx[4]))),
        len(_axis_chunk_bounds(int(shape_tpczyx[5]), int(chunks_tpczyx[5]))),
    )


def _has_dataset(
    group: zarr.Group,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> bool:
    """Return whether a dataset exists with expected shape and dtype.

    Parameters
    ----------
    group : zarr.Group
        Parent group.
    name : str
        Dataset name.
    shape : tuple[int, ...]
        Expected dataset shape.
    dtype : Any
        Expected dtype.

    Returns
    -------
    bool
        ``True`` when the dataset exists and matches expectations.
    """
    candidate = group.get(name)
    if candidate is None:
        return False
    if not isinstance(candidate, zarr.Array):
        return False
    candidate_shape = tuple(int(v) for v in candidate.shape)
    if candidate_shape != tuple(int(v) for v in shape):
        return False
    return np.dtype(candidate.dtype) == np.dtype(dtype)


def _full_chunk_probe_selection(
    array: zarr.Array,
    *,
    leading_indices: Sequence[int] = (),
) -> tuple[slice, ...]:
    """Build a selection that reads one full chunk from an array.

    Parameters
    ----------
    array : zarr.Array
        Target array.
    leading_indices : sequence[int], optional
        Fixed index values for leading axes. Fixed axes are emitted as
        unit-width slices so the dimensionality is preserved.

    Returns
    -------
    tuple[slice, ...]
        Selection that covers one full chunk from each unfixed axis.

    Raises
    ------
    ValueError
        If indices exceed rank or any fixed index is out of bounds.
    """
    shape = tuple(int(v) for v in array.shape)
    chunks = tuple(int(v) for v in array.chunks)
    fixed = tuple(int(v) for v in leading_indices)
    if len(fixed) > len(shape):
        raise ValueError("Leading indices exceed array rank.")

    selection: list[slice] = []
    for axis, axis_size in enumerate(shape):
        if axis < len(fixed):
            index = int(fixed[axis])
            if index < 0 or index >= axis_size:
                raise ValueError("Leading index is out of bounds for chunk probe.")
            selection.append(slice(index, index + 1))
            continue
        chunk_size = max(1, int(chunks[axis]))
        selection.append(slice(0, min(axis_size, chunk_size)))
    return tuple(selection)


def _dataset_chunk_probe_is_readable(
    array: zarr.Array,
    *,
    scan_profile_axes: bool = False,
) -> bool:
    """Return whether representative full-chunk reads decode successfully.

    Parameters
    ----------
    array : zarr.Array
        Dataset to probe.
    scan_profile_axes : bool, optional
        Whether to probe one chunk for each ``(axis0, axis1)`` pair. This is
        used for ``(p, c, ...)`` profile/channel datasets where malformed chunk
        headers may only affect specific profile/channel combinations.

    Returns
    -------
    bool
        ``True`` when all selected probes decode cleanly.

    Notes
    -----
    Probes intentionally request full chunk extents to trigger strict N5 chunk
    header validation and reject rank-mismatched chunks before resume.
    """
    shape = tuple(int(v) for v in array.shape)
    if any(int(v) == 0 for v in shape):
        return True

    try:
        if bool(scan_profile_axes) and array.ndim >= 2:
            for axis0, axis1 in product(range(shape[0]), range(shape[1])):
                selection = _full_chunk_probe_selection(
                    array,
                    leading_indices=(int(axis0), int(axis1)),
                )
                np.asarray(array[selection], dtype=array.dtype)
            return True

        selection = _full_chunk_probe_selection(array)
        np.asarray(array[selection], dtype=array.dtype)
        return True
    except Exception:
        return False


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


def _expand_bounds_with_overlap(
    bounds: tuple[int, int],
    *,
    overlap: int,
    axis_size: int,
) -> tuple[int, int]:
    """Expand one-dimensional bounds by overlap and clip to axis limits.

    Parameters
    ----------
    bounds : tuple[int, int]
        Input ``(start, stop)`` bounds.
    overlap : int
        Overlap width to apply on both sides.
    axis_size : int
        Full axis size.

    Returns
    -------
    tuple[int, int]
        Expanded and clipped bounds.
    """
    return (
        max(0, int(bounds[0]) - int(overlap)),
        min(int(axis_size), int(bounds[1]) + int(overlap)),
    )


def _axis_blend_weights(length: int, left_core: int, right_core: int) -> Float32Array:
    """Return 1D feather weights for one tile axis.

    Parameters
    ----------
    length : int
        Tile axis length.
    left_core : int
        Inclusive core start offset inside the tile.
    right_core : int
        Exclusive core stop offset inside the tile.

    Returns
    -------
    numpy.ndarray
        Feather weights in ``(length,)`` shape.
    """
    weights = np.ones(max(1, int(length)), dtype=np.float32)
    left = max(0, int(left_core))
    right = min(int(length), max(left, int(right_core)))
    if left > 0:
        ramp = np.linspace(0.0, 1.0, left + 2, dtype=np.float32)[1:-1]
        weights[:left] = ramp
    if right < int(length):
        tail = int(length) - right
        ramp = np.linspace(1.0, 0.0, tail + 2, dtype=np.float32)[1:-1]
        weights[right:] = ramp
    return np.maximum(weights, np.float32(1e-3))


def _tile_blend_weights(
    *,
    shape_yx: tuple[int, int],
    core_offsets_yx: SpatialBounds,
) -> Float32Array:
    """Build 2D blending weights for one read tile.

    Parameters
    ----------
    shape_yx : tuple[int, int]
        Read tile shape in ``(y, x)`` order.
    core_offsets_yx : tuple[tuple[int, int], tuple[int, int]]
        Core offsets inside the read tile as ``((y0, y1), (x0, x1))``.

    Returns
    -------
    numpy.ndarray
        2D blending weights in ``(y, x)`` order.
    """
    y_weights = _axis_blend_weights(
        int(shape_yx[0]),
        int(core_offsets_yx[0][0]),
        int(core_offsets_yx[0][1]),
    )
    x_weights = _axis_blend_weights(
        int(shape_yx[1]),
        int(core_offsets_yx[1][0]),
        int(core_offsets_yx[1][1]),
    )
    return np.outer(y_weights, x_weights).astype(np.float32, copy=False)


def _fit_basic_profile(
    *,
    fit_images: Float32Array,
    basic_class: Any,
    parameters: Mapping[str, Any],
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Fit BaSiCPy artifacts for a stack of 2D images.

    Parameters
    ----------
    fit_images : numpy.ndarray
        Input images in ``(n_images, y, x)`` order.
    basic_class : Any
        Imported BaSiCPy ``BaSiC`` class.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Flatfield, darkfield, and baseline arrays.
    """
    basic = basic_class(
        fitting_mode="approximate",
        get_darkfield=bool(parameters.get("get_darkfield", False)),
        smoothness_flatfield=float(parameters.get("smoothness_flatfield", 1.0)),
        working_size=int(parameters.get("working_size", 128)),
        device="cpu",
    )
    basic.fit(fit_images, skip_shape_warning=True)
    flatfield_yx = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield_yx = np.asarray(basic.darkfield, dtype=np.float32)
    baseline = np.asarray(basic.baseline, dtype=np.float32)
    return (
        flatfield_yx,
        darkfield_yx,
        baseline,
    )


def _region_to_slices(region: RegionBounds) -> tuple[slice, slice, slice, slice, slice, slice]:
    """Convert integer region bounds into six-axis Python slices.

    Parameters
    ----------
    region : RegionBounds
        Bounds in canonical ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    tuple[slice, slice, slice, slice, slice, slice]
        Equivalent slicing tuple.
    """
    return (
        slice(int(region[0][0]), int(region[0][1])),
        slice(int(region[1][0]), int(region[1][1])),
        slice(int(region[2][0]), int(region[2][1])),
        slice(int(region[3][0]), int(region[3][1])),
        slice(int(region[4][0]), int(region[4][1])),
        slice(int(region[5][0]), int(region[5][1])),
    )


def _expand_region_with_overlap(
    core_region: RegionBounds,
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    overlap_zyx: tuple[int, int, int],
    use_overlap: bool,
) -> RegionBounds:
    """Expand a core region by overlap margins in spatial axes.

    Parameters
    ----------
    core_region : RegionBounds
        Core bounds in canonical axis order.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full array shape.
    overlap_zyx : tuple[int, int, int]
        Overlap depth in ``(z, y, x)`` order.
    use_overlap : bool
        Whether overlap expansion should be applied.

    Returns
    -------
    RegionBounds
        Expanded bounds clipped to array limits.
    """
    if not use_overlap:
        return core_region

    z_overlap, y_overlap, x_overlap = overlap_zyx
    t_bounds, p_bounds, c_bounds, z_bounds, y_bounds, x_bounds = core_region
    return (
        t_bounds,
        p_bounds,
        c_bounds,
        (
            max(0, int(z_bounds[0]) - z_overlap),
            min(int(shape_tpczyx[3]), int(z_bounds[1]) + z_overlap),
        ),
        (
            max(0, int(y_bounds[0]) - y_overlap),
            min(int(shape_tpczyx[4]), int(y_bounds[1]) + y_overlap),
        ),
        (
            max(0, int(x_bounds[0]) - x_overlap),
            min(int(shape_tpczyx[5]), int(x_bounds[1]) + x_overlap),
        ),
    )


def _crop_slices_for_core(
    *,
    read_region: RegionBounds,
    core_region: RegionBounds,
) -> tuple[slice, slice, slice, slice, slice, slice]:
    """Build crop slices that trim a read region back to its core bounds.

    Parameters
    ----------
    read_region : RegionBounds
        Expanded region used for reading.
    core_region : RegionBounds
        Core output region.

    Returns
    -------
    tuple[slice, slice, slice, slice, slice, slice]
        Crop slices in canonical ``(t, p, c, z, y, x)`` order.
    """
    return (
        slice(
            int(core_region[0][0]) - int(read_region[0][0]),
            int(core_region[0][1]) - int(read_region[0][0]),
        ),
        slice(
            int(core_region[1][0]) - int(read_region[1][0]),
            int(core_region[1][1]) - int(read_region[1][0]),
        ),
        slice(
            int(core_region[2][0]) - int(read_region[2][0]),
            int(core_region[2][1]) - int(read_region[2][0]),
        ),
        slice(
            int(core_region[3][0]) - int(read_region[3][0]),
            int(core_region[3][1]) - int(read_region[3][0]),
        ),
        slice(
            int(core_region[4][0]) - int(read_region[4][0]),
            int(core_region[4][1]) - int(read_region[4][0]),
        ),
        slice(
            int(core_region[5][0]) - int(read_region[5][0]),
            int(core_region[5][1]) - int(read_region[5][0]),
        ),
    )


def _copy_source_array_attrs(
    *,
    root: zarr.Group,
    source_component: str,
    target_array: zarr.Array,
    output_chunks: tuple[int, int, int, int, int, int],
) -> None:
    """Copy key visualization and layout metadata from the source array.

    Parameters
    ----------
    root : zarr.Group
        Open Zarr root group.
    source_component : str
        Source array component path.
    target_array : zarr.Array
        Newly created corrected output array.
    output_chunks : tuple[int, int, int, int, int, int]
        Chunk shape for the corrected array.

    Returns
    -------
    None
        Target attrs are updated in-place.
    """
    source_attrs = dict(root[source_component].attrs)
    root_attrs = dict(root.attrs)
    copied: dict[str, Any] = {
        "source_component": str(source_component),
        "chunk_shape_tpczyx": [int(v) for v in output_chunks],
        "pyramid_levels": ["results/flatfield/latest/data"],
    }
    for key in (
        "scale_tpczyx",
        "physical_scale_tpczyx",
        "resolution_pyramid_factors_tpczyx",
    ):
        if key in source_attrs:
            copied[key] = source_attrs.get(key)
        elif key in root_attrs:
            copied[key] = root_attrs.get(key)
    target_array.attrs.update(copied)


def _normalize_level_factor_payload(
    payload: Any,
) -> Optional[tuple[CanonicalShapeTpczyx, ...]]:
    """Normalize per-level absolute pyramid factors.

    Parameters
    ----------
    payload : Any
        Candidate payload where each entry defines one
        ``(t, p, c, z, y, x)`` level factor tuple.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...], optional
        Normalized level factors, or ``None`` when payload is missing or invalid.

    Raises
    ------
    None
        Invalid payloads return ``None``.
    """
    if not isinstance(payload, (list, tuple)) or len(payload) <= 0:
        return None
    parsed: list[CanonicalShapeTpczyx] = []
    for level in payload:
        if not isinstance(level, (list, tuple)) or len(level) != 6:
            return None
        try:
            factors: CanonicalShapeTpczyx = (
                int(level[0]),
                int(level[1]),
                int(level[2]),
                int(level[3]),
                int(level[4]),
                int(level[5]),
            )
        except Exception:
            return None
        if any(value <= 0 for value in factors):
            return None
        if parsed and factors == parsed[-1]:
            continue
        parsed.append(factors)
    if not parsed:
        return None
    if parsed[0] != (1, 1, 1, 1, 1, 1):
        return None
    return tuple(parsed)


def _normalize_axis_factor_payload(
    payload: Any,
) -> Optional[tuple[CanonicalShapeTpczyx, ...]]:
    """Normalize per-axis pyramid factors into absolute level factors.

    Parameters
    ----------
    payload : Any
        Candidate payload where each entry defines one axis factor sequence in
        canonical ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...], optional
        Normalized level factors, or ``None`` when payload is missing or invalid.

    Raises
    ------
    None
        Invalid payloads return ``None``.
    """
    if not isinstance(payload, (list, tuple)) or len(payload) != 6:
        return None

    axis_levels: list[tuple[int, ...]] = []
    for axis_payload in payload:
        if not isinstance(axis_payload, (list, tuple)) or len(axis_payload) <= 0:
            return None
        try:
            parsed_levels = tuple(int(value) for value in axis_payload)
        except Exception:
            return None
        if any(value <= 0 for value in parsed_levels):
            return None
        if parsed_levels[0] != 1:
            return None
        axis_levels.append(parsed_levels)

    max_levels = max(len(levels) for levels in axis_levels)
    resolved: list[CanonicalShapeTpczyx] = []
    for level_index in range(max_levels):
        factors: CanonicalShapeTpczyx = (
            int(axis_levels[0][min(level_index, len(axis_levels[0]) - 1)]),
            int(axis_levels[1][min(level_index, len(axis_levels[1]) - 1)]),
            int(axis_levels[2][min(level_index, len(axis_levels[2]) - 1)]),
            int(axis_levels[3][min(level_index, len(axis_levels[3]) - 1)]),
            int(axis_levels[4][min(level_index, len(axis_levels[4]) - 1)]),
            int(axis_levels[5][min(level_index, len(axis_levels[5]) - 1)]),
        )
        if resolved and factors == resolved[-1]:
            continue
        resolved.append(factors)

    if not resolved:
        return None
    if resolved[0] != (1, 1, 1, 1, 1, 1):
        return None
    return tuple(resolved)


def _resolve_output_level_factors(
    *,
    root: zarr.Group,
    source_component: str,
) -> tuple[CanonicalShapeTpczyx, ...]:
    """Resolve pyramid level factors to apply to flatfield corrected output.

    Parameters
    ----------
    root : zarr.Group
        Open Zarr root group.
    source_component : str
        Source component used for flatfield correction.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...]
        Absolute downsample factors in canonical ``(t, p, c, z, y, x)`` order.
        Always includes a base level ``(1, 1, 1, 1, 1, 1)``.

    Raises
    ------
    None
        Unavailable metadata falls back to a base-only level definition.
    """
    source_attrs = dict(root[source_component].attrs)
    root_attrs = dict(root.attrs)

    for candidate in (
        source_attrs.get("pyramid_factors_tpczyx"),
        root_attrs.get("data_pyramid_factors_tpczyx"),
    ):
        normalized = _normalize_level_factor_payload(candidate)
        if normalized is not None:
            return normalized

    for candidate in (
        source_attrs.get("resolution_pyramid_factors_tpczyx"),
        root_attrs.get("resolution_pyramid_factors_tpczyx"),
    ):
        normalized = _normalize_axis_factor_payload(candidate)
        if normalized is not None:
            return normalized

    return ((1, 1, 1, 1, 1, 1),)


def _normalize_level_chunks(
    *,
    level_shape_tpczyx: CanonicalShapeTpczyx,
    requested_chunks_tpczyx: CanonicalShapeTpczyx,
) -> CanonicalShapeTpczyx:
    """Normalize requested chunk sizes for one canonical level.

    Parameters
    ----------
    level_shape_tpczyx : tuple[int, int, int, int, int, int]
        Level shape in canonical axis order.
    requested_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Requested chunks in canonical axis order.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Chunks clipped to valid positive values per axis.

    Raises
    ------
    ValueError
        If any tuple is malformed.
    """
    if len(level_shape_tpczyx) != 6 or len(requested_chunks_tpczyx) != 6:
        raise ValueError("Expected six-axis canonical shape/chunk tuples.")

    normalized: list[int] = []
    for axis_size, chunk_size in zip(
        level_shape_tpczyx, requested_chunks_tpczyx, strict=False
    ):
        if int(axis_size) <= 0:
            raise ValueError("Level shape values must be greater than zero.")
        normalized.append(max(1, min(int(axis_size), int(chunk_size))))
    return (
        int(normalized[0]),
        int(normalized[1]),
        int(normalized[2]),
        int(normalized[3]),
        int(normalized[4]),
        int(normalized[5]),
    )


def _derive_level_chunks(
    *,
    base_chunks_tpczyx: CanonicalShapeTpczyx,
    level_shape_tpczyx: CanonicalShapeTpczyx,
    level_factors_tpczyx: CanonicalShapeTpczyx,
) -> CanonicalShapeTpczyx:
    """Derive downsampled-level chunks from base chunks and absolute factors.

    Parameters
    ----------
    base_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Base chunk shape in canonical order.
    level_shape_tpczyx : tuple[int, int, int, int, int, int]
        Target level shape in canonical order.
    level_factors_tpczyx : tuple[int, int, int, int, int, int]
        Absolute level factors in canonical order.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Normalized chunks for the level.

    Raises
    ------
    ValueError
        If any factor is invalid.
    """
    if any(int(factor) <= 0 for factor in level_factors_tpczyx):
        raise ValueError("Level factors must be greater than zero.")
    requested_chunks: CanonicalShapeTpczyx = (
        max(1, int(base_chunks_tpczyx[0]) // int(level_factors_tpczyx[0])),
        max(1, int(base_chunks_tpczyx[1]) // int(level_factors_tpczyx[1])),
        max(1, int(base_chunks_tpczyx[2]) // int(level_factors_tpczyx[2])),
        max(1, int(base_chunks_tpczyx[3]) // int(level_factors_tpczyx[3])),
        max(1, int(base_chunks_tpczyx[4]) // int(level_factors_tpczyx[4])),
        max(1, int(base_chunks_tpczyx[5]) // int(level_factors_tpczyx[5])),
    )
    return _normalize_level_chunks(
        level_shape_tpczyx=level_shape_tpczyx,
        requested_chunks_tpczyx=requested_chunks,
    )


def _materialize_output_pyramid(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    base_component: str,
    base_chunks_tpczyx: CanonicalShapeTpczyx,
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 95,
    progress_end: int = 99,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Materialize a multiscale pyramid for corrected flatfield output data.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Target analysis-store path.
    source_component : str
        Source component used during flatfield correction.
    base_component : str
        Base corrected output component (``results/flatfield/latest/data``).
    base_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Base corrected-output chunk shape in canonical axis order.
    client : dask.distributed.Client, optional
        Active Dask client for distributed writes.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.
    progress_start : int, default=95
        Progress range start.
    progress_end : int, default=99
        Progress range end.

    Returns
    -------
    tuple[list[str], list[list[int]], list[list[int]]]
        ``(level_paths, level_factor_payload, level_shape_payload)`` including
        base level at index ``0``.

    Raises
    ------
    ValueError
        If base output component shape/chunks are not canonical.
    """
    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    root = zarr.open_group(str(zarr_path), mode="a")
    base_array = root[base_component]
    base_shape = tuple(int(value) for value in base_array.shape)
    if len(base_shape) != 6 or len(tuple(base_chunks_tpczyx)) != 6:
        raise ValueError(
            "Flatfield output pyramid requires canonical 6D data "
            f"for component '{base_component}'."
        )

    level_factors = _resolve_output_level_factors(
        root=root,
        source_component=source_component,
    )

    base_parent = str(base_component).rsplit("/", 1)[0]
    latest_group = root[base_parent]
    if "data_pyramid" in latest_group:
        del latest_group["data_pyramid"]
    latest_group.require_group("data_pyramid")

    level_paths = [str(base_component)]
    level_factor_payload = [[1, 1, 1, 1, 1, 1]]
    level_shape_payload = [[int(value) for value in base_shape]]

    if len(level_factors) <= 1:
        base_array.attrs.update(
            {
                "pyramid_levels": list(level_paths),
                "pyramid_factors_tpczyx": list(level_factor_payload),
                "pyramid_shapes_tpczyx": list(level_shape_payload),
            }
        )
        latest_group.attrs.update(
            {
                "data_pyramid_levels": list(level_paths),
                "data_pyramid_factors_tpczyx": list(level_factor_payload),
                "data_pyramid_shapes_tpczyx": list(level_shape_payload),
            }
        )
        _emit(progress_end, "Flatfield pyramid generation complete")
        return level_paths, level_factor_payload, level_shape_payload

    prior_component = str(base_component)
    prior_factors = level_factors[0]
    total_levels = max(1, int(len(level_factors) - 1))
    for level_index, absolute_factors in enumerate(level_factors[1:], start=1):
        if all(
            int(current) % int(previous) == 0
            for current, previous in zip(absolute_factors, prior_factors, strict=False)
        ):
            relative_factors: CanonicalShapeTpczyx = (
                int(absolute_factors[0] // prior_factors[0]),
                int(absolute_factors[1] // prior_factors[1]),
                int(absolute_factors[2] // prior_factors[2]),
                int(absolute_factors[3] // prior_factors[3]),
                int(absolute_factors[4] // prior_factors[4]),
                int(absolute_factors[5] // prior_factors[5]),
            )
            level_source_component = str(prior_component)
            downsample_factors = relative_factors
        else:
            level_source_component = str(base_component)
            downsample_factors = absolute_factors

        level_progress_start = progress_start + int(
            ((level_index - 1) / total_levels) * max(0, progress_end - progress_start)
        )
        level_progress_end = progress_start + int(
            (level_index / total_levels) * max(0, progress_end - progress_start)
        )
        _emit(
            level_progress_start,
            "Writing flatfield pyramid "
            f"level {level_index}/{total_levels} (factors={absolute_factors})",
        )

        source_array = da.from_zarr(str(zarr_path), component=level_source_component)
        downsampled = source_array[
            tuple(slice(None, None, int(factor)) for factor in downsample_factors)
        ]
        level_shape: CanonicalShapeTpczyx = (
            int(downsampled.shape[0]),
            int(downsampled.shape[1]),
            int(downsampled.shape[2]),
            int(downsampled.shape[3]),
            int(downsampled.shape[4]),
            int(downsampled.shape[5]),
        )
        level_chunks = _derive_level_chunks(
            base_chunks_tpczyx=base_chunks_tpczyx,
            level_shape_tpczyx=level_shape,
            level_factors_tpczyx=absolute_factors,
        )
        with dask.config.set({"array.rechunk.method": "tasks"}):
            downsampled = downsampled.rechunk(level_chunks)

        level_component = f"{base_parent}/data_pyramid/level_{level_index}"
        write_task = write_dask_array(
            zarr_path=zarr_path,
            component=level_component,
            array=downsampled,
            chunks=level_chunks,
            overwrite=True,
            compute=False,
        )
        if client is None:
            dask.compute(write_task, scheduler="processes")
        else:
            client.compute(write_task).result()

        root = zarr.open_group(str(zarr_path), mode="a")
        root[level_component].attrs.update(
            {
                "axes": ["t", "p", "c", "z", "y", "x"],
                "pyramid_level": int(level_index),
                "downsample_factors_tpczyx": [
                    int(value) for value in absolute_factors
                ],
                "chunk_shape_tpczyx": [int(value) for value in level_chunks],
                "source_component": str(level_source_component),
            }
        )
        level_paths.append(level_component)
        level_factor_payload.append([int(value) for value in absolute_factors])
        level_shape_payload.append([int(value) for value in level_shape])
        prior_component = level_component
        prior_factors = absolute_factors
        _emit(
            level_progress_end,
            f"Wrote flatfield pyramid level {level_index}/{total_levels}",
        )

    root = zarr.open_group(str(zarr_path), mode="a")
    root[base_component].attrs.update(
        {
            "pyramid_levels": list(level_paths),
            "pyramid_factors_tpczyx": list(level_factor_payload),
            "pyramid_shapes_tpczyx": list(level_shape_payload),
        }
    )
    root[base_parent].attrs.update(
        {
            "data_pyramid_levels": list(level_paths),
            "data_pyramid_factors_tpczyx": list(level_factor_payload),
            "data_pyramid_shapes_tpczyx": list(level_shape_payload),
        }
    )
    _emit(progress_end, "Flatfield pyramid generation complete")
    return level_paths, level_factor_payload, level_shape_payload


def _checkpoint_dataset_specs(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    fit_grid_yx: tuple[int, int],
    transform_grid_yx: tuple[int, int],
    fit_mode: str,
    blend_tiles_effective: bool,
) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Return expected checkpoint dataset specs for the current run setup.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    fit_grid_yx : tuple[int, int]
        Fit tile-grid shape in ``(n_y_tiles, n_x_tiles)`` order.
    transform_grid_yx : tuple[int, int]
        Transform chunk-grid shape in ``(n_y_chunks, n_x_chunks)`` order.
    fit_mode : str
        Effective fit mode.
    blend_tiles_effective : bool
        Whether blended tile reduction is enabled.

    Returns
    -------
    dict[str, tuple[tuple[int, ...], Any]]
        Mapping from checkpoint dataset name to ``(shape, dtype)``.
    """
    t_count, p_count, c_count, z_count, y_count, x_count = (
        int(shape_tpczyx[0]),
        int(shape_tpczyx[1]),
        int(shape_tpczyx[2]),
        int(shape_tpczyx[3]),
        int(shape_tpczyx[4]),
        int(shape_tpczyx[5]),
    )
    specs: dict[str, tuple[tuple[int, ...], Any]] = {
        "fit_profile_done_pc": ((p_count, c_count), np.bool_),
        "transform_done_tpcyx": (
            (t_count, p_count, c_count, int(transform_grid_yx[0]), int(transform_grid_yx[1])),
            np.bool_,
        ),
    }
    if str(fit_mode) == "tiled":
        specs.update(
            {
                "fit_tile_done_pcyx": (
                    (p_count, c_count, int(fit_grid_yx[0]), int(fit_grid_yx[1])),
                    np.bool_,
                ),
                "fit_baseline_sum_pctz": ((p_count, c_count, t_count, z_count), np.float32),
                "fit_baseline_count_pc": ((p_count, c_count), np.uint32),
            }
        )
        if bool(blend_tiles_effective):
            specs.update(
                {
                    "fit_flatfield_sum_pcyx": ((p_count, c_count, y_count, x_count), np.float32),
                    "fit_darkfield_sum_pcyx": ((p_count, c_count, y_count, x_count), np.float32),
                    "fit_weight_sum_pcyx": ((p_count, c_count, y_count, x_count), np.float32),
                }
            )
    return specs


def _checkpoint_is_compatible(
    *,
    latest_group: zarr.Group,
    source_component: str,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    chunks_tpczyx: tuple[int, int, int, int, int, int],
    fit_grid_yx: tuple[int, int],
    transform_grid_yx: tuple[int, int],
    fit_mode: str,
    blend_tiles_effective: bool,
    parameter_fingerprint: str,
) -> bool:
    """Return whether an existing latest group can be resumed safely.

    Parameters
    ----------
    latest_group : zarr.Group
        Existing latest-output group.
    source_component : str
        Requested source component path.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Output chunk shape in canonical axis order.
    fit_grid_yx : tuple[int, int]
        Expected fit tile-grid shape.
    transform_grid_yx : tuple[int, int]
        Expected transform chunk-grid shape.
    fit_mode : str
        Effective fit mode.
    blend_tiles_effective : bool
        Whether blended tile reduction is enabled.
    parameter_fingerprint : str
        Expected parameter fingerprint.

    Returns
    -------
    bool
        ``True`` when checkpoint metadata and datasets are compatible.
    """
    p_count, c_count, t_count, z_count, y_count, x_count = (
        int(shape_tpczyx[1]),
        int(shape_tpczyx[2]),
        int(shape_tpczyx[0]),
        int(shape_tpczyx[3]),
        int(shape_tpczyx[4]),
        int(shape_tpczyx[5]),
    )
    if not _has_dataset(latest_group, name="data", shape=shape_tpczyx, dtype=np.float32):
        return False
    if tuple(int(v) for v in latest_group["data"].chunks) != tuple(int(v) for v in chunks_tpczyx):
        return False
    if not _has_dataset(
        latest_group,
        name="flatfield_pcyx",
        shape=(p_count, c_count, y_count, x_count),
        dtype=np.float32,
    ):
        return False
    if not _has_dataset(
        latest_group,
        name="darkfield_pcyx",
        shape=(p_count, c_count, y_count, x_count),
        dtype=np.float32,
    ):
        return False
    if not _has_dataset(
        latest_group,
        name="baseline_pctz",
        shape=(p_count, c_count, t_count, z_count),
        dtype=np.float32,
    ):
        return False

    checkpoint_group = latest_group.get(CHECKPOINT_GROUP_NAME)
    if checkpoint_group is None or not isinstance(checkpoint_group, zarr.Group):
        return False

    checkpoint_attrs = dict(checkpoint_group.attrs)
    if str(checkpoint_attrs.get("schema_version", "")) != RESUME_SCHEMA_VERSION:
        return False
    if str(checkpoint_attrs.get("parameter_fingerprint", "")) != str(parameter_fingerprint):
        return False
    if str(checkpoint_attrs.get("source_component", "")) != str(source_component):
        return False
    if tuple(checkpoint_attrs.get("source_shape_tpczyx", [])) != tuple(shape_tpczyx):
        return False
    if tuple(checkpoint_attrs.get("source_chunks_tpczyx", [])) != tuple(chunks_tpczyx):
        return False
    if str(checkpoint_attrs.get("fit_mode", "")) != str(fit_mode):
        return False
    if tuple(checkpoint_attrs.get("fit_grid_yx", [])) != tuple(fit_grid_yx):
        return False
    if tuple(checkpoint_attrs.get("transform_grid_yx", [])) != tuple(transform_grid_yx):
        return False
    if bool(checkpoint_attrs.get("blend_tiles_effective", False)) != bool(
        blend_tiles_effective
    ):
        return False

    expected_specs = _checkpoint_dataset_specs(
        shape_tpczyx=shape_tpczyx,
        fit_grid_yx=fit_grid_yx,
        transform_grid_yx=transform_grid_yx,
        fit_mode=fit_mode,
        blend_tiles_effective=blend_tiles_effective,
    )
    for name, (shape, dtype) in expected_specs.items():
        if not _has_dataset(checkpoint_group, name=name, shape=shape, dtype=dtype):
            return False

    # Ensure pre-existing chunks decode cleanly before attempting resume.
    for name in ("flatfield_pcyx", "darkfield_pcyx", "baseline_pctz"):
        candidate = latest_group.get(name)
        if candidate is None or not isinstance(candidate, zarr.Array):
            return False
        if not _dataset_chunk_probe_is_readable(candidate, scan_profile_axes=True):
            return False

    for name in expected_specs:
        candidate = checkpoint_group.get(name)
        if candidate is None or not isinstance(candidate, zarr.Array):
            return False
        if not _dataset_chunk_probe_is_readable(
            candidate,
            scan_profile_axes=name
            in {
                "fit_tile_done_pcyx",
                "fit_baseline_sum_pctz",
                "fit_flatfield_sum_pcyx",
                "fit_darkfield_sum_pcyx",
                "fit_weight_sum_pcyx",
            },
        ):
            return False
    return True


def _create_checkpoint_datasets(
    *,
    checkpoint_group: zarr.Group,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    chunks_tpczyx: tuple[int, int, int, int, int, int],
    fit_grid_yx: tuple[int, int],
    transform_grid_yx: tuple[int, int],
    fit_mode: str,
    blend_tiles_effective: bool,
) -> None:
    """Create fresh checkpoint datasets for resumable flatfield execution.

    Parameters
    ----------
    checkpoint_group : zarr.Group
        Checkpoint group under ``results/flatfield/latest``.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Output chunk shape in canonical axis order.
    fit_grid_yx : tuple[int, int]
        Fit tile-grid shape.
    transform_grid_yx : tuple[int, int]
        Transform chunk-grid shape.
    fit_mode : str
        Effective fit mode.
    blend_tiles_effective : bool
        Whether blended tile reduction is enabled.

    Returns
    -------
    None
        Datasets are created in-place.
    """
    p_count = int(shape_tpczyx[1])
    c_count = int(shape_tpczyx[2])
    t_count = int(shape_tpczyx[0])
    z_count = int(shape_tpczyx[3])
    y_count = int(shape_tpczyx[4])
    x_count = int(shape_tpczyx[5])
    y_chunk_count = int(transform_grid_yx[0])
    x_chunk_count = int(transform_grid_yx[1])

    checkpoint_group.create_dataset(
        name="fit_profile_done_pc",
        shape=(p_count, c_count),
        chunks=(1, 1),
        dtype=np.bool_,
        overwrite=True,
        fill_value=False,
    )
    checkpoint_group.create_dataset(
        name="transform_done_tpcyx",
        shape=(t_count, p_count, c_count, y_chunk_count, x_chunk_count),
        chunks=(1, 1, 1, 1, 1),
        dtype=np.bool_,
        overwrite=True,
        fill_value=False,
    )

    if str(fit_mode) != "tiled":
        return

    artifact_chunks_pcyx = (
        1,
        1,
        max(1, min(int(chunks_tpczyx[4]), y_count)),
        max(1, min(int(chunks_tpczyx[5]), x_count)),
    )
    checkpoint_group.create_dataset(
        name="fit_tile_done_pcyx",
        shape=(p_count, c_count, int(fit_grid_yx[0]), int(fit_grid_yx[1])),
        chunks=(1, 1, 1, 1),
        dtype=np.bool_,
        overwrite=True,
        fill_value=False,
    )
    checkpoint_group.create_dataset(
        name="fit_baseline_sum_pctz",
        shape=(p_count, c_count, t_count, z_count),
        chunks=(1, 1, 1, max(1, min(z_count, int(chunks_tpczyx[3])))),
        dtype=np.float32,
        overwrite=True,
        fill_value=np.float32(0.0),
    )
    checkpoint_group.create_dataset(
        name="fit_baseline_count_pc",
        shape=(p_count, c_count),
        chunks=(1, 1),
        dtype=np.uint32,
        overwrite=True,
        fill_value=np.uint32(0),
    )

    if not bool(blend_tiles_effective):
        return

    checkpoint_group.create_dataset(
        name="fit_flatfield_sum_pcyx",
        shape=(p_count, c_count, y_count, x_count),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
        fill_value=np.float32(0.0),
    )
    checkpoint_group.create_dataset(
        name="fit_darkfield_sum_pcyx",
        shape=(p_count, c_count, y_count, x_count),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
        fill_value=np.float32(0.0),
    )
    checkpoint_group.create_dataset(
        name="fit_weight_sum_pcyx",
        shape=(p_count, c_count, y_count, x_count),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
        fill_value=np.float32(0.0),
    )


def _initialize_latest_flatfield_group(
    *,
    root: zarr.Group,
    source_component: str,
    parameters: Mapping[str, Any],
    parameter_payload: Mapping[str, Any],
    parameter_json: str,
    parameter_fingerprint: str,
    basicpy_version: Optional[str],
    shape_tpczyx: tuple[int, int, int, int, int, int],
    chunks_tpczyx: tuple[int, int, int, int, int, int],
    fit_mode: str,
    fit_grid_yx: tuple[int, int],
    transform_grid_yx: tuple[int, int],
    blend_tiles_effective: bool,
) -> zarr.Group:
    """Initialize a fresh ``results/flatfield/latest`` output and checkpoint tree.

    Parameters
    ----------
    root : zarr.Group
        Open Zarr root in append mode.
    source_component : str
        Source component path.
    parameters : mapping[str, Any]
        Normalized run parameters.
    parameter_payload : mapping[str, Any]
        JSON-serializable parameter payload used for resume matching.
    parameter_json : str
        Canonical serialized payload.
    parameter_fingerprint : str
        SHA256 digest of ``parameter_json``.
    basicpy_version : str, optional
        Installed BaSiCPy version when available.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    chunks_tpczyx : tuple[int, int, int, int, int, int]
        Output chunk shape in canonical axis order.
    fit_mode : str
        Effective fit mode.
    fit_grid_yx : tuple[int, int]
        Fit tile-grid shape.
    transform_grid_yx : tuple[int, int]
        Transform chunk-grid shape.
    blend_tiles_effective : bool
        Whether blended tiled reduction is enabled.

    Returns
    -------
    zarr.Group
        Newly created latest group.
    """
    results_group = root.require_group("results")
    flatfield_group = results_group.require_group("flatfield")
    if "latest" in flatfield_group:
        del flatfield_group["latest"]
    latest_group = flatfield_group.create_group("latest")

    data_array = latest_group.create_dataset(
        name="data",
        shape=shape_tpczyx,
        chunks=chunks_tpczyx,
        dtype=np.float32,
        overwrite=True,
    )
    _copy_source_array_attrs(
        root=root,
        source_component=source_component,
        target_array=data_array,
        output_chunks=chunks_tpczyx,
    )

    artifact_chunks_pcyx = (
        1,
        1,
        max(1, min(int(chunks_tpczyx[4]), int(shape_tpczyx[4]))),
        max(1, min(int(chunks_tpczyx[5]), int(shape_tpczyx[5]))),
    )
    latest_group.create_dataset(
        name="flatfield_pcyx",
        shape=(
            int(shape_tpczyx[1]),
            int(shape_tpczyx[2]),
            int(shape_tpczyx[4]),
            int(shape_tpczyx[5]),
        ),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="darkfield_pcyx",
        shape=(
            int(shape_tpczyx[1]),
            int(shape_tpczyx[2]),
            int(shape_tpczyx[4]),
            int(shape_tpczyx[5]),
        ),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="baseline_pctz",
        shape=(
            int(shape_tpczyx[1]),
            int(shape_tpczyx[2]),
            int(shape_tpczyx[0]),
            int(shape_tpczyx[3]),
        ),
        chunks=(
            1,
            1,
            1,
            max(1, min(int(shape_tpczyx[3]), int(chunks_tpczyx[3]))),
        ),
        dtype=np.float32,
        overwrite=True,
    )

    latest_group.attrs.update(
        {
            "storage_policy": "latest_only",
            "run_id": None,
            "source_component": str(source_component),
            "data_component": "results/flatfield/latest/data",
            "flatfield_component": "results/flatfield/latest/flatfield_pcyx",
            "darkfield_component": "results/flatfield/latest/darkfield_pcyx",
            "baseline_component": "results/flatfield/latest/baseline_pctz",
            "parameters": _to_jsonable(dict(parameters)),
            "resume_parameters": dict(parameter_payload),
            "resume_parameters_json": str(parameter_json),
            "resume_parameter_fingerprint": str(parameter_fingerprint),
            "output_dtype": "float32",
            "output_chunks_tpczyx": [int(v) for v in chunks_tpczyx],
            "basicpy_version": basicpy_version,
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "updated_utc": _utc_now_iso(),
        }
    )

    checkpoint_group = latest_group.require_group(CHECKPOINT_GROUP_NAME)
    _create_checkpoint_datasets(
        checkpoint_group=checkpoint_group,
        shape_tpczyx=shape_tpczyx,
        chunks_tpczyx=chunks_tpczyx,
        fit_grid_yx=fit_grid_yx,
        transform_grid_yx=transform_grid_yx,
        fit_mode=fit_mode,
        blend_tiles_effective=blend_tiles_effective,
    )
    checkpoint_group.attrs.update(
        {
            "schema_version": RESUME_SCHEMA_VERSION,
            "source_component": str(source_component),
            "source_shape_tpczyx": [int(v) for v in shape_tpczyx],
            "source_chunks_tpczyx": [int(v) for v in chunks_tpczyx],
            "fit_mode": str(fit_mode),
            "fit_grid_yx": [int(v) for v in fit_grid_yx],
            "transform_grid_yx": [int(v) for v in transform_grid_yx],
            "blend_tiles_effective": bool(blend_tiles_effective),
            "parameter_payload": dict(parameter_payload),
            "parameter_json": str(parameter_json),
            "parameter_fingerprint": str(parameter_fingerprint),
            "fit_stage_status": "pending",
            "fit_outputs_materialized": False,
            "transform_stage_status": "pending",
            "run_status": "running",
            "created_utc": _utc_now_iso(),
            "updated_utc": _utc_now_iso(),
            "completed_utc": None,
            "last_error": None,
            "fit_warning_count": 0,
            "fit_fallback_records": [],
        }
    )
    return latest_group


def _prepare_output_arrays(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    parameters: Mapping[str, Any],
    basicpy_version: Optional[str],
) -> FlatfieldOutputLayout:
    """Prepare or resume latest flatfield output datasets in the Zarr store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis store path.
    source_component : str
        Source component path.
    parameters : mapping[str, Any]
        Normalized flatfield parameters.
    basicpy_version : str, optional
        Installed BaSiCPy version string.

    Returns
    -------
    FlatfieldOutputLayout
        Output component metadata and resume context.

    Raises
    ------
    ValueError
        If the source component is incompatible with canonical 6D data.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    source = root[source_component]
    shape_tpczyx = tuple(int(v) for v in source.shape)
    chunks_tpczyx = tuple(int(v) for v in source.chunks)
    if len(shape_tpczyx) != 6 or len(chunks_tpczyx) != 6:
        raise ValueError(
            "Flatfield correction requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    fit_mode = str(parameters.get("fit_mode", "tiled")).strip().lower().replace("-", "_")
    fit_tile_shape_yx = cast(
        tuple[int, int],
        parameters.get("fit_tile_shape_yx", (256, 256)),
    )
    fit_grid_yx = _fit_grid_shape(
        shape_tpczyx=shape_tpczyx,
        fit_tile_shape_yx=(int(fit_tile_shape_yx[0]), int(fit_tile_shape_yx[1])),
    )
    transform_grid_yx = _transform_grid_shape(
        shape_tpczyx=shape_tpczyx,
        chunks_tpczyx=chunks_tpczyx,
    )
    blend_tiles_effective = bool(
        parameters.get("blend_tiles", False) and parameters.get("use_map_overlap", True)
    )
    parameter_payload = _resume_parameter_payload(parameters)
    parameter_json, parameter_fingerprint = _resume_parameter_fingerprint(parameter_payload)

    component = "results/flatfield/latest"
    data_component = "results/flatfield/latest/data"
    flatfield_component = "results/flatfield/latest/flatfield_pcyx"
    darkfield_component = "results/flatfield/latest/darkfield_pcyx"
    baseline_component = "results/flatfield/latest/baseline_pctz"
    checkpoint_component = "results/flatfield/latest/checkpoint"

    results_group = root.require_group("results")
    flatfield_group = results_group.require_group("flatfield")
    latest_group = flatfield_group.get("latest")
    should_resume = False
    if (
        latest_group is not None
        and isinstance(latest_group, zarr.Group)
        and not bool(parameters.get("force_rerun", False))
    ):
        should_resume = _checkpoint_is_compatible(
            latest_group=latest_group,
            source_component=source_component,
            shape_tpczyx=shape_tpczyx,
            chunks_tpczyx=chunks_tpczyx,
            fit_grid_yx=fit_grid_yx,
            transform_grid_yx=transform_grid_yx,
            fit_mode=fit_mode,
            blend_tiles_effective=blend_tiles_effective,
            parameter_fingerprint=parameter_fingerprint,
        )

    if not should_resume:
        latest_group = _initialize_latest_flatfield_group(
            root=root,
            source_component=source_component,
            parameters=parameters,
            parameter_payload=parameter_payload,
            parameter_json=parameter_json,
            parameter_fingerprint=parameter_fingerprint,
            basicpy_version=basicpy_version,
            shape_tpczyx=shape_tpczyx,
            chunks_tpczyx=chunks_tpczyx,
            fit_mode=fit_mode,
            fit_grid_yx=fit_grid_yx,
            transform_grid_yx=transform_grid_yx,
            blend_tiles_effective=blend_tiles_effective,
        )
    else:
        latest_group.attrs.update(
            {
                "parameters": _to_jsonable(dict(parameters)),
                "resume_parameters": dict(parameter_payload),
                "resume_parameters_json": str(parameter_json),
                "resume_parameter_fingerprint": str(parameter_fingerprint),
                "basicpy_version": basicpy_version,
                "updated_utc": _utc_now_iso(),
            }
        )
        checkpoint_group = latest_group[CHECKPOINT_GROUP_NAME]
        checkpoint_group.attrs.update(
            {
                "fit_stage_status": str(
                    checkpoint_group.attrs.get("fit_stage_status", "pending")
                ),
                "transform_stage_status": str(
                    checkpoint_group.attrs.get("transform_stage_status", "pending")
                ),
                "run_status": "running",
                "updated_utc": _utc_now_iso(),
                "last_error": None,
                "fit_warning_count": int(
                    checkpoint_group.attrs.get(
                        "fit_warning_count",
                        len(checkpoint_group.attrs.get("fit_fallback_records", [])),
                    )
                ),
                "fit_fallback_records": _to_jsonable(
                    checkpoint_group.attrs.get("fit_fallback_records", [])
                ),
            }
        )

    return FlatfieldOutputLayout(
        component=component,
        data_component=data_component,
        flatfield_component=flatfield_component,
        darkfield_component=darkfield_component,
        baseline_component=baseline_component,
        checkpoint_component=checkpoint_component,
        shape_tpczyx=shape_tpczyx,
        output_chunks_tpczyx=chunks_tpczyx,
        fit_grid_yx=fit_grid_yx,
        transform_grid_yx=transform_grid_yx,
        fit_mode=fit_mode,
        blend_tiles_effective=blend_tiles_effective,
        parameter_fingerprint=parameter_fingerprint,
        resumed=bool(should_resume),
    )


def _fit_profile(
    *,
    zarr_path: str,
    source_component: str,
    position_index: int,
    channel_index: int,
    parameters: Mapping[str, Any],
) -> FlatfieldProfileResult:
    """Fit a BaSiC profile for one ``(position, channel)`` input subset.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component.
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    FlatfieldProfileResult
        Fitted profile artifacts for the requested ``(position, channel)`` pair.

    Raises
    ------
    ValueError
        If the fitted baseline length cannot be reshaped back to ``(t, z)``.
    """
    fit_mode = str(parameters.get("fit_mode", "tiled")).strip().lower().replace("-", "_")
    if fit_mode == "full_volume":
        return _fit_profile_full_volume(
            zarr_path=zarr_path,
            source_component=source_component,
            position_index=position_index,
            channel_index=channel_index,
            parameters=parameters,
        )
    return _fit_profile_tiled(
        zarr_path=zarr_path,
        source_component=source_component,
        position_index=position_index,
        channel_index=channel_index,
        parameters=parameters,
    )


def _fit_profile_full_volume(
    *,
    zarr_path: str,
    source_component: str,
    position_index: int,
    channel_index: int,
    parameters: Mapping[str, Any],
) -> FlatfieldProfileResult:
    """Fit one flatfield profile by materializing the full ``(t, z, y, x)`` volume.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component.
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    FlatfieldProfileResult
        Fitted profile artifacts for the requested ``(position, channel)`` pair.

    Raises
    ------
    ValueError
        If the fitted baseline length cannot be reshaped back to ``(t, z)``.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    volume_tzyx = np.asarray(
        source[:, position_index, channel_index, :, :, :],
        dtype=np.float32,
    )
    t_count, z_count, y_count, x_count = volume_tzyx.shape
    fit_images = volume_tzyx.reshape(t_count * z_count, y_count, x_count)
    basic_class = _load_basic_class()
    flatfield_yx, darkfield_yx, baseline = _fit_basic_profile(
        fit_images=fit_images,
        basic_class=basic_class,
        parameters=parameters,
    )
    if int(baseline.size) != int(t_count * z_count):
        raise ValueError(
            "Flatfield baseline shape mismatch: "
            f"expected {t_count * z_count} values, got {baseline.size}."
        )

    return FlatfieldProfileResult(
        position_index=int(position_index),
        channel_index=int(channel_index),
        flatfield_yx=flatfield_yx,
        darkfield_yx=darkfield_yx,
        baseline_tz=baseline.reshape(t_count, z_count),
    )


def _fit_profile_tiled(
    *,
    zarr_path: str,
    source_component: str,
    position_index: int,
    channel_index: int,
    parameters: Mapping[str, Any],
) -> FlatfieldProfileResult:
    """Fit one flatfield profile by iterating over spatial tiles.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component.
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    FlatfieldProfileResult
        Fitted profile artifacts for the requested ``(position, channel)`` pair.

    Raises
    ------
    ValueError
        If a fitted tile baseline cannot be reshaped back to ``(t, z)``.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    shape_tpczyx = tuple(int(v) for v in source.shape)
    if len(shape_tpczyx) != 6:
        raise ValueError(
            "Flatfield tiled fitting requires canonical 6D data (t,p,c,z,y,x)."
        )
    t_count = int(shape_tpczyx[0])
    z_count = int(shape_tpczyx[3])
    y_count = int(shape_tpczyx[4])
    x_count = int(shape_tpczyx[5])

    fit_tile_shape = cast(
        tuple[int, int],
        parameters.get("fit_tile_shape_yx", (256, 256)),
    )
    tile_y = max(1, min(int(fit_tile_shape[0]), y_count))
    tile_x = max(1, min(int(fit_tile_shape[1]), x_count))
    blend_tiles = bool(parameters.get("blend_tiles", False))
    use_overlap = bool(parameters.get("use_map_overlap", True))
    overlap_zyx = cast(tuple[int, int, int], parameters.get("overlap_zyx", (0, 32, 32)))
    overlap_y = int(overlap_zyx[1]) if use_overlap else 0
    overlap_x = int(overlap_zyx[2]) if use_overlap else 0

    y_core_regions = _axis_chunk_bounds(y_count, tile_y)
    x_core_regions = _axis_chunk_bounds(x_count, tile_x)
    tile_count = max(1, len(y_core_regions) * len(x_core_regions))
    basic_class = _load_basic_class()

    baseline_sum = np.zeros((t_count, z_count), dtype=np.float64)
    if blend_tiles:
        flatfield_sum = np.zeros((y_count, x_count), dtype=np.float64)
        darkfield_sum = np.zeros((y_count, x_count), dtype=np.float64)
        weight_sum = np.zeros((y_count, x_count), dtype=np.float64)
    else:
        flatfield_out = np.zeros((y_count, x_count), dtype=np.float32)
        darkfield_out = np.zeros((y_count, x_count), dtype=np.float32)

    for y_core_region in y_core_regions:
        for x_core_region in x_core_regions:
            y_core_start, y_core_stop = int(y_core_region[0]), int(y_core_region[1])
            x_core_start, x_core_stop = int(x_core_region[0]), int(x_core_region[1])
            y_read_start, y_read_stop = _expand_bounds_with_overlap(
                (y_core_start, y_core_stop),
                overlap=overlap_y,
                axis_size=y_count,
            )
            x_read_start, x_read_stop = _expand_bounds_with_overlap(
                (x_core_start, x_core_stop),
                overlap=overlap_x,
                axis_size=x_count,
            )
            y_core_offset_start = y_core_start - y_read_start
            y_core_offset_stop = y_core_stop - y_read_start
            x_core_offset_start = x_core_start - x_read_start
            x_core_offset_stop = x_core_stop - x_read_start

            volume_tzyx = np.asarray(
                source[
                    :,
                    position_index,
                    channel_index,
                    :,
                    y_read_start:y_read_stop,
                    x_read_start:x_read_stop,
                ],
                dtype=np.float32,
            )
            _, _, y_read_count, x_read_count = volume_tzyx.shape
            fit_images = volume_tzyx.reshape(t_count * z_count, y_read_count, x_read_count)
            flatfield_tile, darkfield_tile, baseline = _fit_basic_profile(
                fit_images=fit_images,
                basic_class=basic_class,
                parameters=parameters,
            )
            if int(baseline.size) != int(t_count * z_count):
                raise ValueError(
                    "Flatfield baseline shape mismatch: "
                    f"expected {t_count * z_count} values, got {baseline.size}."
                )
            baseline_sum += baseline.reshape(t_count, z_count).astype(np.float64)

            if blend_tiles:
                blend_weights = _tile_blend_weights(
                    shape_yx=(y_read_count, x_read_count),
                    core_offsets_yx=(
                        (y_core_offset_start, y_core_offset_stop),
                        (x_core_offset_start, x_core_offset_stop),
                    ),
                )
                blend_weights_64 = blend_weights.astype(np.float64)
                flatfield_sum[
                    y_read_start:y_read_stop,
                    x_read_start:x_read_stop,
                ] += flatfield_tile.astype(np.float64) * blend_weights_64
                darkfield_sum[
                    y_read_start:y_read_stop,
                    x_read_start:x_read_stop,
                ] += darkfield_tile.astype(np.float64) * blend_weights_64
                weight_sum[
                    y_read_start:y_read_stop,
                    x_read_start:x_read_stop,
                ] += blend_weights_64
            else:
                flatfield_out[y_core_start:y_core_stop, x_core_start:x_core_stop] = (
                    flatfield_tile[
                        y_core_offset_start:y_core_offset_stop,
                        x_core_offset_start:x_core_offset_stop,
                    ]
                )
                darkfield_out[y_core_start:y_core_stop, x_core_start:x_core_stop] = (
                    darkfield_tile[
                        y_core_offset_start:y_core_offset_stop,
                        x_core_offset_start:x_core_offset_stop,
                    ]
                )

    baseline_tz = (baseline_sum / float(tile_count)).astype(np.float32)
    if blend_tiles:
        denom = np.maximum(weight_sum, np.float64(1e-6))
        flatfield_yx = (flatfield_sum / denom).astype(np.float32)
        darkfield_yx = (darkfield_sum / denom).astype(np.float32)
    else:
        flatfield_yx = flatfield_out
        darkfield_yx = darkfield_out

    return FlatfieldProfileResult(
        position_index=int(position_index),
        channel_index=int(channel_index),
        flatfield_yx=flatfield_yx,
        darkfield_yx=darkfield_yx,
        baseline_tz=baseline_tz,
    )


def _build_profile_tile_specs(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    parameters: Mapping[str, Any],
) -> list[FlatfieldTileSpec]:
    """Build tiled-fit task specs across all ``(position, channel)`` pairs.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source data shape in canonical axis order.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    list[FlatfieldTileSpec]
        Ordered tile specs for distributed fitting.
    """
    p_count = int(shape_tpczyx[1])
    c_count = int(shape_tpczyx[2])
    y_count = int(shape_tpczyx[4])
    x_count = int(shape_tpczyx[5])
    fit_tile_shape = cast(
        tuple[int, int],
        parameters.get("fit_tile_shape_yx", (256, 256)),
    )
    tile_y = max(1, min(int(fit_tile_shape[0]), y_count))
    tile_x = max(1, min(int(fit_tile_shape[1]), x_count))
    use_overlap = bool(parameters.get("use_map_overlap", True))
    overlap_zyx = cast(tuple[int, int, int], parameters.get("overlap_zyx", (0, 32, 32)))
    overlap_y = int(overlap_zyx[1]) if use_overlap else 0
    overlap_x = int(overlap_zyx[2]) if use_overlap else 0
    y_core_regions = _axis_chunk_bounds(y_count, tile_y)
    x_core_regions = _axis_chunk_bounds(x_count, tile_x)
    specs: list[FlatfieldTileSpec] = []
    for position_index, channel_index in product(range(p_count), range(c_count)):
        for tile_y_index, y_core_region in enumerate(y_core_regions):
            for tile_x_index, x_core_region in enumerate(x_core_regions):
                y_core_bounds = (int(y_core_region[0]), int(y_core_region[1]))
                x_core_bounds = (int(x_core_region[0]), int(x_core_region[1]))
                y_read_bounds = _expand_bounds_with_overlap(
                    y_core_bounds,
                    overlap=overlap_y,
                    axis_size=y_count,
                )
                x_read_bounds = _expand_bounds_with_overlap(
                    x_core_bounds,
                    overlap=overlap_x,
                    axis_size=x_count,
                )
                specs.append(
                    FlatfieldTileSpec(
                        position_index=int(position_index),
                        channel_index=int(channel_index),
                        tile_y_index=int(tile_y_index),
                        tile_x_index=int(tile_x_index),
                        y_core_bounds=y_core_bounds,
                        x_core_bounds=x_core_bounds,
                        y_read_bounds=(int(y_read_bounds[0]), int(y_read_bounds[1])),
                        x_read_bounds=(int(x_read_bounds[0]), int(x_read_bounds[1])),
                    )
                )
    return specs


def _fit_profile_tile(
    *,
    zarr_path: str,
    source_component: str,
    tile: FlatfieldTileSpec,
    parameters: Mapping[str, Any],
) -> FlatfieldTileFitResult:
    """Fit one flatfield tile for a specific ``(position, channel)`` profile.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component.
    tile : FlatfieldTileSpec
        Tile specification including core/read bounds.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    FlatfieldTileFitResult
        Tile payload used by the reduction/stitch stage.

    Raises
    ------
    ValueError
        If baseline output cannot be reshaped to ``(t, z)``.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    volume_tzyx = np.asarray(
        source[
            :,
            tile.position_index,
            tile.channel_index,
            :,
            tile.y_read_bounds[0] : tile.y_read_bounds[1],
            tile.x_read_bounds[0] : tile.x_read_bounds[1],
        ],
        dtype=np.float32,
    )
    t_count, z_count, y_read_count, x_read_count = volume_tzyx.shape
    fit_images = volume_tzyx.reshape(t_count * z_count, y_read_count, x_read_count)
    flatfield_tile, darkfield_tile, baseline = _fit_basic_profile(
        fit_images=fit_images,
        basic_class=_load_basic_class(),
        parameters=parameters,
    )
    if int(baseline.size) != int(t_count * z_count):
        raise ValueError(
            "Flatfield baseline shape mismatch: "
            f"expected {t_count * z_count} values, got {baseline.size}."
        )
    baseline_tz = baseline.reshape(t_count, z_count).astype(np.float32, copy=False)

    blend_tiles = bool(
        parameters.get("blend_tiles", False) and parameters.get("use_map_overlap", True)
    )
    y_core_offset_start = int(tile.y_core_bounds[0]) - int(tile.y_read_bounds[0])
    y_core_offset_stop = int(tile.y_core_bounds[1]) - int(tile.y_read_bounds[0])
    x_core_offset_start = int(tile.x_core_bounds[0]) - int(tile.x_read_bounds[0])
    x_core_offset_stop = int(tile.x_core_bounds[1]) - int(tile.x_read_bounds[0])
    if blend_tiles:
        blend_weights = _tile_blend_weights(
            shape_yx=(int(y_read_count), int(x_read_count)),
            core_offsets_yx=(
                (int(y_core_offset_start), int(y_core_offset_stop)),
                (int(x_core_offset_start), int(x_core_offset_stop)),
            ),
        )
        return FlatfieldTileFitResult(
            position_index=int(tile.position_index),
            channel_index=int(tile.channel_index),
            tile_y_index=int(tile.tile_y_index),
            tile_x_index=int(tile.tile_x_index),
            y_bounds=(int(tile.y_read_bounds[0]), int(tile.y_read_bounds[1])),
            x_bounds=(int(tile.x_read_bounds[0]), int(tile.x_read_bounds[1])),
            flatfield_payload_yx=(
                flatfield_tile.astype(np.float32, copy=False) * blend_weights
            ),
            darkfield_payload_yx=(
                darkfield_tile.astype(np.float32, copy=False) * blend_weights
            ),
            baseline_tz=baseline_tz,
            weight_payload_yx=blend_weights.astype(np.float32, copy=False),
        )

    return FlatfieldTileFitResult(
        position_index=int(tile.position_index),
        channel_index=int(tile.channel_index),
        tile_y_index=int(tile.tile_y_index),
        tile_x_index=int(tile.tile_x_index),
        y_bounds=(int(tile.y_core_bounds[0]), int(tile.y_core_bounds[1])),
        x_bounds=(int(tile.x_core_bounds[0]), int(tile.x_core_bounds[1])),
        flatfield_payload_yx=flatfield_tile[
            y_core_offset_start:y_core_offset_stop,
            x_core_offset_start:x_core_offset_stop,
        ].astype(np.float32, copy=False),
        darkfield_payload_yx=darkfield_tile[
            y_core_offset_start:y_core_offset_stop,
            x_core_offset_start:x_core_offset_stop,
        ].astype(np.float32, copy=False),
        baseline_tz=baseline_tz,
        weight_payload_yx=None,
    )


def _is_recoverable_svd_failure(error: BaseException) -> bool:
    """Return whether an exception represents a recoverable SVD fit failure.

    Parameters
    ----------
    error : BaseException
        Exception raised while fitting a flatfield tile/profile.

    Returns
    -------
    bool
        ``True`` when the error message indicates an SVD convergence failure.
    """
    message = str(error).strip().lower()
    if "linalg.svd" not in message:
        return False
    if "failed to converge" in message:
        return True
    return "ill-conditioned" in message


def _identity_profile_result(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    position_index: int,
    channel_index: int,
) -> FlatfieldProfileResult:
    """Build an identity flatfield profile that leaves source data unchanged.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source dataset shape in canonical ``(t, p, c, z, y, x)`` order.
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.

    Returns
    -------
    FlatfieldProfileResult
        Identity profile with ``flatfield=1``, ``darkfield=0``, and ``baseline=0``.
    """
    t_count = int(shape_tpczyx[0])
    z_count = int(shape_tpczyx[3])
    y_count = int(shape_tpczyx[4])
    x_count = int(shape_tpczyx[5])
    return FlatfieldProfileResult(
        position_index=int(position_index),
        channel_index=int(channel_index),
        flatfield_yx=np.ones((y_count, x_count), dtype=np.float32),
        darkfield_yx=np.zeros((y_count, x_count), dtype=np.float32),
        baseline_tz=np.zeros((t_count, z_count), dtype=np.float32),
    )


def _profile_blend_weight_sum(
    *,
    profile_tiles: list[FlatfieldTileSpec],
    shape_yx: tuple[int, int],
) -> Float32Array:
    """Return aggregate blending weights for one ``(position, channel)`` profile.

    Parameters
    ----------
    profile_tiles : list[FlatfieldTileSpec]
        Tile specifications for a single profile.
    shape_yx : tuple[int, int]
        Full spatial shape as ``(y, x)``.

    Returns
    -------
    numpy.ndarray
        Aggregate blending-weight image in ``(y, x)`` order.
    """
    weight_sum = np.zeros((int(shape_yx[0]), int(shape_yx[1])), dtype=np.float32)
    for tile in profile_tiles:
        y_start, y_stop = int(tile.y_read_bounds[0]), int(tile.y_read_bounds[1])
        x_start, x_stop = int(tile.x_read_bounds[0]), int(tile.x_read_bounds[1])
        blend_weights = _tile_blend_weights(
            shape_yx=(int(y_stop - y_start), int(x_stop - x_start)),
            core_offsets_yx=(
                (
                    int(tile.y_core_bounds[0]) - int(y_start),
                    int(tile.y_core_bounds[1]) - int(y_start),
                ),
                (
                    int(tile.x_core_bounds[0]) - int(x_start),
                    int(tile.x_core_bounds[1]) - int(x_start),
                ),
            ),
        ).astype(np.float32, copy=False)
        weight_sum[y_start:y_stop, x_start:x_stop] += blend_weights
    return weight_sum


def _apply_tiled_profile_fallback(
    *,
    profile: FlatfieldProfileResult,
    profile_tiles: list[FlatfieldTileSpec],
    blend_tiles_effective: bool,
    fit_profile_done_array: Any,
    fit_tile_done_array: Any,
    baseline_sum_array: Any,
    baseline_count_array: Any,
    flatfield_array: Any,
    darkfield_array: Any,
    flatfield_sum_array: Optional[Any],
    darkfield_sum_array: Optional[Any],
    weight_sum_array: Optional[Any],
    shape_tpczyx: tuple[int, int, int, int, int, int],
) -> None:
    """Apply one profile-level fallback and mark all profile tiles complete.

    Parameters
    ----------
    profile : FlatfieldProfileResult
        Profile artifacts produced by full-volume retry or identity fallback.
    profile_tiles : list[FlatfieldTileSpec]
        Tile specifications that belong to ``profile``.
    blend_tiles_effective : bool
        Whether tiled blending accumulators are active.
    fit_profile_done_array : Any
        Checkpoint profile completion dataset.
    fit_tile_done_array : Any
        Checkpoint tile completion dataset.
    baseline_sum_array : Any
        Checkpoint baseline accumulation dataset.
    baseline_count_array : Any
        Checkpoint baseline sample-count dataset.
    flatfield_array : Any
        Final flatfield artifact array.
    darkfield_array : Any
        Final darkfield artifact array.
    flatfield_sum_array : Any, optional
        Blend accumulation array for flatfield payloads.
    darkfield_sum_array : Any, optional
        Blend accumulation array for darkfield payloads.
    weight_sum_array : Any, optional
        Blend accumulation array for weights.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.

    Returns
    -------
    None
        Fallback artifacts are written in-place to checkpoint/output datasets.

    Raises
    ------
    ValueError
        If blend accumulators are required but missing.
    """
    position_index = int(profile.position_index)
    channel_index = int(profile.channel_index)
    tile_count = max(1, int(len(profile_tiles)))
    profile_selection = (
        slice(position_index, position_index + 1),
        slice(channel_index, channel_index + 1),
        slice(None),
        slice(None),
    )

    fit_profile_done_array[position_index, channel_index] = np.bool_(True)
    fit_tile_done_array[
        slice(position_index, position_index + 1),
        slice(channel_index, channel_index + 1),
        slice(None),
        slice(None),
    ] = np.bool_(True)

    baseline_sum_array[profile_selection] = (
        np.asarray(profile.baseline_tz, dtype=np.float32)[None, None, :, :]
        * np.float32(float(tile_count))
    ).astype(np.float32, copy=False)
    baseline_count_array[position_index, channel_index] = np.uint32(tile_count)

    if bool(blend_tiles_effective):
        if (
            flatfield_sum_array is None
            or darkfield_sum_array is None
            or weight_sum_array is None
        ):
            raise ValueError("Flatfield fallback requires blend-accumulator datasets.")
        weight_sum = _profile_blend_weight_sum(
            profile_tiles=profile_tiles,
            shape_yx=(int(shape_tpczyx[4]), int(shape_tpczyx[5])),
        )
        flatfield_sum_array[profile_selection] = (
            np.asarray(profile.flatfield_yx, dtype=np.float32)[None, None, :, :]
            * weight_sum[None, None, :, :]
        ).astype(np.float32, copy=False)
        darkfield_sum_array[profile_selection] = (
            np.asarray(profile.darkfield_yx, dtype=np.float32)[None, None, :, :]
            * weight_sum[None, None, :, :]
        ).astype(np.float32, copy=False)
        weight_sum_array[profile_selection] = weight_sum[None, None, :, :]
        return

    flatfield_array[profile_selection] = np.asarray(
        profile.flatfield_yx,
        dtype=np.float32,
    )[None, None, :, :]
    darkfield_array[profile_selection] = np.asarray(
        profile.darkfield_yx,
        dtype=np.float32,
    )[None, None, :, :]


def _persist_profile_artifacts(
    *,
    zarr_path: str,
    flatfield_component: str,
    darkfield_component: str,
    baseline_component: str,
    profile: FlatfieldProfileResult,
) -> None:
    """Persist one fitted profile artifact bundle to the Zarr store.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    flatfield_component : str
        Flatfield artifact component path.
    darkfield_component : str
        Darkfield artifact component path.
    baseline_component : str
        Baseline artifact component path.
    profile : FlatfieldProfileResult
        Fitted profile payload.

    Returns
    -------
    None
        Artifacts are written in-place to the store.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    profile_selection = (
        slice(int(profile.position_index), int(profile.position_index) + 1),
        slice(int(profile.channel_index), int(profile.channel_index) + 1),
        slice(None),
        slice(None),
    )
    root[flatfield_component][profile_selection] = np.asarray(
        profile.flatfield_yx,
        dtype=np.float32,
    )[None, None, :, :]
    root[darkfield_component][profile_selection] = np.asarray(
        profile.darkfield_yx,
        dtype=np.float32,
    )[None, None, :, :]
    root[baseline_component][profile_selection] = np.asarray(
        profile.baseline_tz,
        dtype=np.float32,
    )[None, None, :, :]


def _transform_region(
    *,
    zarr_path: str,
    source_component: str,
    output_data_component: str,
    flatfield_component: str,
    darkfield_component: str,
    baseline_component: str,
    position_index: int,
    channel_index: int,
    read_region: RegionBounds,
    core_region: RegionBounds,
    is_timelapse: bool,
) -> int:
    """Apply flatfield correction for one chunk region and write the core.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component path.
    output_data_component : str
        Corrected output data component path.
    flatfield_component : str
        Flatfield artifact component path.
    darkfield_component : str
        Darkfield artifact component path.
    baseline_component : str
        Baseline artifact component path.
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    read_region : RegionBounds
        Expanded read bounds including overlap halo.
    core_region : RegionBounds
        Non-overlapping write bounds.
    is_timelapse : bool
        Whether baseline subtraction should be applied.

    Returns
    -------
    int
        Constant ``1`` used for task completion accounting.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    chunk = np.asarray(
        root[source_component][_region_to_slices(read_region)],
        dtype=np.float32,
    )

    y_start, y_stop = int(read_region[4][0]), int(read_region[4][1])
    x_start, x_stop = int(read_region[5][0]), int(read_region[5][1])
    denominator = np.maximum(
        np.asarray(
            root[flatfield_component][
                position_index,
                channel_index,
                y_start:y_stop,
                x_start:x_stop,
            ],
            dtype=np.float32,
        ),
        np.float32(1e-6),
    )
    darkfield = np.asarray(
        root[darkfield_component][
            position_index,
            channel_index,
            y_start:y_stop,
            x_start:x_stop,
        ],
        dtype=np.float32,
    )
    corrected = (chunk - darkfield[None, None, None, None, :, :]) / denominator[
        None,
        None,
        None,
        None,
        :,
        :,
    ]

    if is_timelapse:
        baseline = np.asarray(
            root[baseline_component][
                position_index,
                channel_index,
                read_region[0][0] : read_region[0][1],
                read_region[3][0] : read_region[3][1],
            ],
            dtype=np.float32,
        )
        corrected = corrected - baseline[:, None, None, :, None, None]

    corrected_core = corrected[_crop_slices_for_core(
        read_region=read_region,
        core_region=core_region,
    )].astype(np.float32, copy=False)

    write_root = zarr.open_group(str(zarr_path), mode="a")
    write_root[output_data_component][_region_to_slices(core_region)] = corrected_core
    return 1


def _build_transform_specs(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    output_chunks_tpczyx: tuple[int, int, int, int, int, int],
    overlap_zyx: tuple[int, int, int],
    use_overlap: bool,
) -> list[FlatfieldTransformSpec]:
    """Build transform-task specs over all corrected output chunks.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Output chunk shape in canonical axis order.
    overlap_zyx : tuple[int, int, int]
        Overlap halo in ``(z, y, x)`` order.
    use_overlap : bool
        Whether overlap halo should be applied.

    Returns
    -------
    list[FlatfieldTransformSpec]
        Ordered transform specs with chunk-grid indices.
    """
    y_bounds = _axis_chunk_bounds(int(shape_tpczyx[4]), int(output_chunks_tpczyx[4]))
    x_bounds = _axis_chunk_bounds(int(shape_tpczyx[5]), int(output_chunks_tpczyx[5]))
    specs: list[FlatfieldTransformSpec] = []
    for t_index, p_index, c_index in product(
        range(int(shape_tpczyx[0])),
        range(int(shape_tpczyx[1])),
        range(int(shape_tpczyx[2])),
    ):
        for y_chunk_index, y_bounds_chunk in enumerate(y_bounds):
            for x_chunk_index, x_bounds_chunk in enumerate(x_bounds):
                core_region: RegionBounds = (
                    (int(t_index), int(t_index) + 1),
                    (int(p_index), int(p_index) + 1),
                    (int(c_index), int(c_index) + 1),
                    (0, int(shape_tpczyx[3])),
                    (int(y_bounds_chunk[0]), int(y_bounds_chunk[1])),
                    (int(x_bounds_chunk[0]), int(x_bounds_chunk[1])),
                )
                specs.append(
                    FlatfieldTransformSpec(
                        t_index=int(t_index),
                        p_index=int(p_index),
                        c_index=int(c_index),
                        y_chunk_index=int(y_chunk_index),
                        x_chunk_index=int(x_chunk_index),
                        read_region=_expand_region_with_overlap(
                            core_region,
                            shape_tpczyx=shape_tpczyx,
                            overlap_zyx=overlap_zyx,
                            use_overlap=bool(use_overlap),
                        ),
                        core_region=core_region,
                    )
                )
    return specs


def run_flatfield_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> FlatfieldSummary:
    """Run BaSiCPy flatfield correction and persist latest outputs.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Flatfield-correction parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    FlatfieldSummary
        Summary metadata for the completed flatfield run.

    Raises
    ------
    ValueError
        If the source component is missing or incompatible.
    ImportError
        If BaSiCPy is unavailable in the current environment.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    def _execute_tasks(
        *,
        task_inputs: list[Any],
        build_task: Callable[[Any], Any],
        consume_result: Callable[[Any, Any], None],
        handle_task_error: Optional[Callable[[Any, BaseException], bool]] = None,
        progress_start: int,
        progress_span: int,
        progress_message: str,
    ) -> None:
        """Execute delayed tasks and consume results incrementally.

        Parameters
        ----------
        task_inputs : list[Any]
            Task descriptors used for building delayed computations.
        build_task : callable
            Builder returning a delayed task for one descriptor.
        consume_result : callable
            Callback invoked as ``consume_result(task_input, result)``.
        handle_task_error : callable, optional
            Error handler invoked as ``handle_task_error(task_input, error)``.
            Return ``True`` to treat the failure as handled and continue.
        progress_start : int
            Inclusive progress lower bound.
        progress_span : int
            Progress points consumed by this stage.
        progress_message : str
            Message template with ``{completed}`` and ``{total}`` fields.

        Returns
        -------
        None
            Tasks are executed and consumed in-place.
        """
        if not task_inputs:
            return

        total = max(1, int(len(task_inputs)))
        if client is None:
            for completed, task_input in enumerate(task_inputs, start=1):
                try:
                    result = dask.compute(build_task(task_input), scheduler="processes")[0]
                except Exception as exc:  # pragma: no cover - mirrored distributed path
                    if handle_task_error is None or not bool(
                        handle_task_error(task_input, exc)
                    ):
                        raise
                else:
                    consume_result(task_input, result)
                progress = progress_start + int((completed / total) * progress_span)
                _emit(
                    progress,
                    progress_message.format(completed=completed, total=total),
                )
            return

        from dask.distributed import as_completed

        delayed_tasks = [build_task(task_input) for task_input in task_inputs]
        futures = cast(list[Any], client.compute(delayed_tasks))
        future_to_input = {future: task_inputs[index] for index, future in enumerate(futures)}
        completed = 0
        for future in as_completed(futures):
            task_input = future_to_input[future]
            try:
                result = future.result()
            except Exception as exc:
                if handle_task_error is None or not bool(handle_task_error(task_input, exc)):
                    raise
            else:
                consume_result(task_input, result)
            completed += 1
            progress = progress_start + int((completed / total) * progress_span)
            _emit(
                progress,
                progress_message.format(completed=completed, total=total),
            )

    normalized = _normalize_parameters(parameters)
    basicpy_version = _basicpy_version()
    root = zarr.open_group(str(zarr_path), mode="r")
    source_component = str(normalized.get("input_source", "data")).strip() or "data"
    try:
        source = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"Flatfield input component '{source_component}' was not found in {zarr_path}."
        ) from exc

    shape = tuple(int(v) for v in source.shape)
    chunks = tuple(int(v) for v in source.chunks)
    if len(shape) != 6 or len(chunks) != 6:
        raise ValueError(
            "Flatfield correction requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    _emit(5, "Preparing flatfield output datasets")
    layout = _prepare_output_arrays(
        zarr_path=zarr_path,
        source_component=source_component,
        parameters=normalized,
        basicpy_version=basicpy_version,
    )
    shape_tpczyx = layout.shape_tpczyx
    output_chunks = layout.output_chunks_tpczyx
    profile_pairs = [
        (int(position_index), int(channel_index))
        for position_index, channel_index in product(
            range(int(shape_tpczyx[1])),
            range(int(shape_tpczyx[2])),
        )
    ]
    profile_count = int(len(profile_pairs))
    _emit(
        6,
        "Resuming existing flatfield checkpoint"
        if layout.resumed
        else "Initialized fresh flatfield checkpoint",
    )

    checkpoint_group = zarr.open_group(str(zarr_path), mode="a")[layout.checkpoint_component]
    raw_fallback_records = checkpoint_group.attrs.get("fit_fallback_records", [])
    fit_fallback_records: list[dict[str, Any]] = []
    if isinstance(raw_fallback_records, list):
        for record in raw_fallback_records:
            if isinstance(record, Mapping):
                fit_fallback_records.append(cast(dict[str, Any], _to_jsonable(dict(record))))
    checkpoint_group.attrs.update(
        {
            "run_status": "running",
            "updated_utc": _utc_now_iso(),
            "last_error": None,
            "fit_warning_count": int(len(fit_fallback_records)),
            "fit_fallback_records": _to_jsonable(fit_fallback_records),
        }
    )

    try:
        fit_profile_done_array = checkpoint_group["fit_profile_done_pc"]

        if layout.fit_mode == "full_volume":
            fit_done_mask = np.asarray(fit_profile_done_array, dtype=bool)
            pending_profile_pairs = [
                (int(position_index), int(channel_index))
                for position_index, channel_index in profile_pairs
                if not bool(fit_done_mask[position_index, channel_index])
            ]
            _emit(
                10,
                "Prepared "
                f"{len(pending_profile_pairs)} pending flatfield profile fit tasks "
                f"(total_profiles={profile_count})",
            )
            if pending_profile_pairs:
                checkpoint_group.attrs.update(
                    {
                        "fit_stage_status": "running",
                        "updated_utc": _utc_now_iso(),
                    }
                )

                def _build_profile_task(profile_key: tuple[int, int]) -> Any:
                    position_index, channel_index = profile_key
                    return delayed(_fit_profile_full_volume)(
                        zarr_path=str(zarr_path),
                        source_component=source_component,
                        position_index=int(position_index),
                        channel_index=int(channel_index),
                        parameters=normalized,
                    )

                def _consume_profile_result(
                    profile_key: tuple[int, int],
                    profile_result: FlatfieldProfileResult,
                ) -> None:
                    _persist_profile_artifacts(
                        zarr_path=str(zarr_path),
                        flatfield_component=layout.flatfield_component,
                        darkfield_component=layout.darkfield_component,
                        baseline_component=layout.baseline_component,
                        profile=profile_result,
                    )
                    position_index, channel_index = profile_key
                    fit_profile_done_array[position_index, channel_index] = np.bool_(True)

                _execute_tasks(
                    task_inputs=pending_profile_pairs,
                    build_task=_build_profile_task,
                    consume_result=_consume_profile_result,
                    progress_start=10,
                    progress_span=25,
                    progress_message="Fitted flatfield profile {completed}/{total}",
                )

            fit_done_mask = np.asarray(fit_profile_done_array, dtype=bool)
            completed_profiles = int(np.count_nonzero(fit_done_mask))
            if fit_done_mask.size == 0 or bool(np.all(fit_done_mask)):
                checkpoint_group.attrs.update(
                    {
                        "fit_stage_status": "complete",
                        "fit_outputs_materialized": True,
                        "updated_utc": _utc_now_iso(),
                    }
                )
            _emit(
                40,
                f"Persisted {completed_profiles} fitted flatfield profiles",
            )
        else:
            tile_specs = _build_profile_tile_specs(
                shape_tpczyx=shape_tpczyx,
                parameters=normalized,
            )
            tiles_by_profile: dict[tuple[int, int], list[FlatfieldTileSpec]] = {}
            for tile in tile_specs:
                key = (int(tile.position_index), int(tile.channel_index))
                tiles_by_profile.setdefault(key, []).append(tile)
            fallback_profile_keys: set[tuple[int, int]] = {
                (
                    int(record["position_index"]),
                    int(record["channel_index"]),
                )
                for record in fit_fallback_records
                if "position_index" in record and "channel_index" in record
            }
            tiles_per_profile = (
                int(len(tile_specs) // max(1, profile_count)) if profile_count > 0 else 0
            )
            tile_done_array = checkpoint_group["fit_tile_done_pcyx"]
            tile_done_mask = np.asarray(tile_done_array, dtype=bool)
            pending_tile_specs = [
                tile
                for tile in tile_specs
                if not bool(
                    tile_done_mask[
                        tile.position_index,
                        tile.channel_index,
                        tile.tile_y_index,
                        tile.tile_x_index,
                    ]
                )
                and (
                    int(tile.position_index),
                    int(tile.channel_index),
                )
                not in fallback_profile_keys
            ]
            _emit(
                10,
                "Prepared "
                f"{len(pending_tile_specs)} pending flatfield tile fit tasks "
                f"(total_tiles={len(tile_specs)}, profiles={profile_count}, "
                f"tiles_per_profile~{tiles_per_profile})",
            )
            write_root = zarr.open_group(str(zarr_path), mode="a")
            flatfield_array = write_root[layout.flatfield_component]
            darkfield_array = write_root[layout.darkfield_component]
            baseline_array = write_root[layout.baseline_component]
            baseline_sum_array = checkpoint_group["fit_baseline_sum_pctz"]
            baseline_count_array = checkpoint_group["fit_baseline_count_pc"]
            flatfield_sum_array = (
                checkpoint_group["fit_flatfield_sum_pcyx"]
                if layout.blend_tiles_effective
                else None
            )
            darkfield_sum_array = (
                checkpoint_group["fit_darkfield_sum_pcyx"]
                if layout.blend_tiles_effective
                else None
            )
            weight_sum_array = (
                checkpoint_group["fit_weight_sum_pcyx"]
                if layout.blend_tiles_effective
                else None
            )

            if pending_tile_specs:
                checkpoint_group.attrs.update(
                    {
                        "fit_stage_status": "running",
                        "updated_utc": _utc_now_iso(),
                    }
                )

                def _build_tile_task(tile: FlatfieldTileSpec) -> Any:
                    return delayed(_fit_profile_tile)(
                        zarr_path=str(zarr_path),
                        source_component=source_component,
                        tile=tile,
                        parameters=normalized,
                    )

                def _record_profile_fallback(
                    *,
                    profile_key: tuple[int, int],
                    fallback_mode: str,
                    trigger_error: BaseException,
                    retry_error: Optional[BaseException],
                ) -> None:
                    """Persist one profile-level fallback record to checkpoint attrs.

                    Parameters
                    ----------
                    profile_key : tuple[int, int]
                        Profile key in ``(position, channel)`` order.
                    fallback_mode : str
                        Applied fallback mode (for example ``full_volume`` or ``identity``).
                    trigger_error : BaseException
                        Original exception raised by tile fitting.
                    retry_error : BaseException, optional
                        Exception from full-volume retry, when fallback used identity.

                    Returns
                    -------
                    None
                        Checkpoint attributes are updated in-place.
                    """
                    record = {
                        "position_index": int(profile_key[0]),
                        "channel_index": int(profile_key[1]),
                        "fallback_mode": str(fallback_mode),
                        "trigger_error": str(trigger_error),
                        "retry_error": None if retry_error is None else str(retry_error),
                        "recorded_utc": _utc_now_iso(),
                    }
                    fit_fallback_records.append(record)
                    checkpoint_group.attrs.update(
                        {
                            "fit_warning_count": int(len(fit_fallback_records)),
                            "fit_fallback_records": _to_jsonable(fit_fallback_records),
                            "updated_utc": _utc_now_iso(),
                        }
                    )

                def _handle_tile_fit_error(
                    tile: FlatfieldTileSpec,
                    error: BaseException,
                ) -> bool:
                    """Handle recoverable tiled-fit failures with profile-level fallback.

                    Parameters
                    ----------
                    tile : FlatfieldTileSpec
                        Tile descriptor associated with the failed task.
                    error : BaseException
                        Exception raised by the task.

                    Returns
                    -------
                    bool
                        ``True`` when a fallback was applied and execution can continue.
                    """
                    profile_key = (int(tile.position_index), int(tile.channel_index))
                    if profile_key in fallback_profile_keys:
                        return True
                    if not _is_recoverable_svd_failure(error):
                        return False

                    profile_tiles = tiles_by_profile.get(profile_key, [])
                    if not profile_tiles:
                        return False

                    retry_error: Optional[BaseException] = None
                    try:
                        fallback_profile = _fit_profile_full_volume(
                            zarr_path=str(zarr_path),
                            source_component=source_component,
                            position_index=int(profile_key[0]),
                            channel_index=int(profile_key[1]),
                            parameters=normalized,
                        )
                        fallback_mode = "full_volume"
                    except Exception as exc:
                        retry_error = exc
                        fallback_profile = _identity_profile_result(
                            shape_tpczyx=shape_tpczyx,
                            position_index=int(profile_key[0]),
                            channel_index=int(profile_key[1]),
                        )
                        fallback_mode = "identity"

                    _apply_tiled_profile_fallback(
                        profile=fallback_profile,
                        profile_tiles=profile_tiles,
                        blend_tiles_effective=layout.blend_tiles_effective,
                        fit_profile_done_array=fit_profile_done_array,
                        fit_tile_done_array=tile_done_array,
                        baseline_sum_array=baseline_sum_array,
                        baseline_count_array=baseline_count_array,
                        flatfield_array=flatfield_array,
                        darkfield_array=darkfield_array,
                        flatfield_sum_array=flatfield_sum_array,
                        darkfield_sum_array=darkfield_sum_array,
                        weight_sum_array=weight_sum_array,
                        shape_tpczyx=shape_tpczyx,
                    )
                    fallback_profile_keys.add(profile_key)
                    _record_profile_fallback(
                        profile_key=profile_key,
                        fallback_mode=fallback_mode,
                        trigger_error=error,
                        retry_error=retry_error,
                    )
                    _emit(
                        10,
                        "Recovered flatfield profile "
                        f"p{profile_key[0]} c{profile_key[1]} via {fallback_mode} fallback",
                    )
                    return True

                def _consume_tile_result(
                    _tile: FlatfieldTileSpec,
                    tile_result: FlatfieldTileFitResult,
                ) -> None:
                    position_index = int(tile_result.position_index)
                    channel_index = int(tile_result.channel_index)
                    profile_key = (position_index, channel_index)
                    if profile_key in fallback_profile_keys:
                        return
                    y_start, y_stop = int(tile_result.y_bounds[0]), int(tile_result.y_bounds[1])
                    x_start, x_stop = int(tile_result.x_bounds[0]), int(tile_result.x_bounds[1])
                    profile_selection = (
                        slice(position_index, position_index + 1),
                        slice(channel_index, channel_index + 1),
                        slice(None),
                        slice(None),
                    )
                    tile_selection = (
                        slice(position_index, position_index + 1),
                        slice(channel_index, channel_index + 1),
                        slice(y_start, y_stop),
                        slice(x_start, x_stop),
                    )

                    baseline_sum = np.asarray(
                        baseline_sum_array[profile_selection],
                        dtype=np.float32,
                    )
                    baseline_sum += np.asarray(tile_result.baseline_tz, dtype=np.float32)[
                        None,
                        None,
                        :,
                        :,
                    ]
                    baseline_sum_array[profile_selection] = baseline_sum
                    baseline_count = int(baseline_count_array[position_index, channel_index])
                    baseline_count_array[position_index, channel_index] = np.uint32(
                        baseline_count + 1
                    )

                    if layout.blend_tiles_effective:
                        if tile_result.weight_payload_yx is None:
                            raise ValueError(
                                "Flatfield tiled reduction expected blend weights for a tile result."
                            )
                        if (
                            flatfield_sum_array is None
                            or darkfield_sum_array is None
                            or weight_sum_array is None
                        ):
                            raise ValueError(
                                "Flatfield checkpoint is missing blend-accumulator datasets."
                            )
                        flatfield_sum = np.asarray(
                            flatfield_sum_array[tile_selection],
                            dtype=np.float32,
                        )
                        flatfield_sum += np.asarray(
                            tile_result.flatfield_payload_yx,
                            dtype=np.float32,
                        )[None, None, :, :]
                        flatfield_sum_array[tile_selection] = flatfield_sum

                        darkfield_sum = np.asarray(
                            darkfield_sum_array[tile_selection],
                            dtype=np.float32,
                        )
                        darkfield_sum += np.asarray(
                            tile_result.darkfield_payload_yx,
                            dtype=np.float32,
                        )[None, None, :, :]
                        darkfield_sum_array[tile_selection] = darkfield_sum

                        weight_sum = np.asarray(
                            weight_sum_array[tile_selection],
                            dtype=np.float32,
                        )
                        weight_sum += np.asarray(
                            tile_result.weight_payload_yx,
                            dtype=np.float32,
                        )[None, None, :, :]
                        weight_sum_array[tile_selection] = weight_sum
                    else:
                        flatfield_array[tile_selection] = np.asarray(
                            tile_result.flatfield_payload_yx,
                            dtype=np.float32,
                        )[None, None, :, :]
                        darkfield_array[tile_selection] = np.asarray(
                            tile_result.darkfield_payload_yx,
                            dtype=np.float32,
                        )[None, None, :, :]

                    tile_done_array[
                        position_index,
                        channel_index,
                        int(tile_result.tile_y_index),
                        int(tile_result.tile_x_index),
                    ] = np.bool_(True)

                _execute_tasks(
                    task_inputs=pending_tile_specs,
                    build_task=_build_tile_task,
                    consume_result=_consume_tile_result,
                    handle_task_error=_handle_tile_fit_error,
                    progress_start=10,
                    progress_span=25,
                    progress_message="Fitted flatfield tile {completed}/{total}",
                )

            tile_done_mask = np.asarray(tile_done_array, dtype=bool)
            fit_outputs_materialized = bool(
                checkpoint_group.attrs.get("fit_outputs_materialized", False)
            )
            if tile_done_mask.size == 0 or bool(np.all(tile_done_mask)):
                if (not fit_outputs_materialized) or bool(pending_tile_specs):
                    for position_index, channel_index in profile_pairs:
                        profile_selection = (
                            slice(position_index, position_index + 1),
                            slice(channel_index, channel_index + 1),
                            slice(None),
                            slice(None),
                        )
                        baseline_sum = np.asarray(
                            baseline_sum_array[profile_selection],
                            dtype=np.float32,
                        )
                        baseline_count = max(
                            1,
                            int(baseline_count_array[position_index, channel_index]),
                        )
                        baseline_array[profile_selection] = (
                            baseline_sum / np.float32(float(baseline_count))
                        ).astype(np.float32, copy=False)
                        if layout.blend_tiles_effective:
                            if (
                                flatfield_sum_array is None
                                or darkfield_sum_array is None
                                or weight_sum_array is None
                            ):
                                raise ValueError(
                                    "Flatfield checkpoint is missing blend-accumulator datasets."
                                )
                            weight_sum = np.asarray(
                                weight_sum_array[profile_selection],
                                dtype=np.float32,
                            )
                            denominator = np.maximum(weight_sum, np.float32(1e-6))
                            flatfield_sum = np.asarray(
                                flatfield_sum_array[profile_selection],
                                dtype=np.float32,
                            )
                            darkfield_sum = np.asarray(
                                darkfield_sum_array[profile_selection],
                                dtype=np.float32,
                            )
                            flatfield_array[profile_selection] = (
                                flatfield_sum / denominator
                            ).astype(np.float32, copy=False)
                            darkfield_array[profile_selection] = (
                                darkfield_sum / denominator
                            ).astype(np.float32, copy=False)

                checkpoint_group.attrs.update(
                    {
                        "fit_stage_status": "complete",
                        "fit_outputs_materialized": True,
                        "updated_utc": _utc_now_iso(),
                    }
                )
            completed_tiles = int(np.count_nonzero(tile_done_mask))
            _emit(
                40,
                "Persisted "
                f"{profile_count} fitted flatfield profiles from "
                f"{completed_tiles}/{len(tile_specs)} completed tile tasks",
            )

        transform_done_array = checkpoint_group["transform_done_tpcyx"]
        transform_specs = _build_transform_specs(
            shape_tpczyx=shape_tpczyx,
            output_chunks_tpczyx=output_chunks,
            overlap_zyx=cast(tuple[int, int, int], normalized["overlap_zyx"]),
            use_overlap=bool(normalized["use_map_overlap"]),
        )
        transform_done_mask = np.asarray(transform_done_array, dtype=bool)
        pending_transform_specs = [
            spec
            for spec in transform_specs
            if not bool(
                transform_done_mask[
                    spec.t_index,
                    spec.p_index,
                    spec.c_index,
                    spec.y_chunk_index,
                    spec.x_chunk_index,
                ]
            )
        ]
        _emit(
            45,
            "Prepared "
            f"{len(pending_transform_specs)} pending flatfield transform tasks "
            f"(total_chunks={len(transform_specs)})",
        )
        if pending_transform_specs:
            checkpoint_group.attrs.update(
                {
                    "transform_stage_status": "running",
                    "updated_utc": _utc_now_iso(),
                }
            )

            def _build_transform_task(spec: FlatfieldTransformSpec) -> Any:
                return delayed(_transform_region)(
                    zarr_path=str(zarr_path),
                    source_component=source_component,
                    output_data_component=layout.data_component,
                    flatfield_component=layout.flatfield_component,
                    darkfield_component=layout.darkfield_component,
                    baseline_component=layout.baseline_component,
                    position_index=int(spec.p_index),
                    channel_index=int(spec.c_index),
                    read_region=spec.read_region,
                    core_region=spec.core_region,
                    is_timelapse=bool(normalized["is_timelapse"]),
                )

            def _consume_transform_result(
                spec: FlatfieldTransformSpec,
                _result: int,
            ) -> None:
                transform_done_array[
                    int(spec.t_index),
                    int(spec.p_index),
                    int(spec.c_index),
                    int(spec.y_chunk_index),
                    int(spec.x_chunk_index),
                ] = np.bool_(True)

            _execute_tasks(
                task_inputs=pending_transform_specs,
                build_task=_build_transform_task,
                consume_result=_consume_transform_result,
                progress_start=45,
                progress_span=50,
                progress_message="Corrected chunk {completed}/{total}",
            )

        transform_done_mask = np.asarray(transform_done_array, dtype=bool)
        warning_count = int(len(fit_fallback_records))
        completed_run_status = (
            "complete_with_warnings" if warning_count > 0 else "complete"
        )
        if transform_done_mask.size == 0 or bool(np.all(transform_done_mask)):
            checkpoint_group.attrs.update(
                {
                    "transform_stage_status": "complete",
                    "run_status": completed_run_status,
                    "fit_warning_count": int(warning_count),
                    "fit_fallback_records": _to_jsonable(fit_fallback_records),
                    "completed_utc": _utc_now_iso(),
                    "updated_utc": _utc_now_iso(),
                }
            )

        transformed_volumes = int(shape_tpczyx[0] * shape_tpczyx[1] * shape_tpczyx[2])
        pyramid_levels, pyramid_factors_tpczyx, pyramid_shapes_tpczyx = (
            _materialize_output_pyramid(
                zarr_path=str(zarr_path),
                source_component=source_component,
                base_component=layout.data_component,
                base_chunks_tpczyx=output_chunks,
                client=client,
                progress_callback=progress_callback,
                progress_start=95,
                progress_end=99,
            )
        )
        latest_group = zarr.open_group(str(zarr_path), mode="a")[layout.component]
        latest_group.attrs.update(
            {
                "profile_count": int(profile_count),
                "transformed_volumes": int(transformed_volumes),
                "source_component": str(source_component),
                "data_component": layout.data_component,
                "flatfield_component": layout.flatfield_component,
                "darkfield_component": layout.darkfield_component,
                "baseline_component": layout.baseline_component,
                "parameters": _to_jsonable(dict(normalized)),
                "resume_parameter_fingerprint": str(layout.parameter_fingerprint),
                "resumed_from_checkpoint": bool(layout.resumed),
                "output_dtype": "float32",
                "output_chunks_tpczyx": [int(v) for v in output_chunks],
                "pyramid_levels": list(pyramid_levels),
                "pyramid_factors_tpczyx": list(pyramid_factors_tpczyx),
                "pyramid_shapes_tpczyx": list(pyramid_shapes_tpczyx),
                "basicpy_version": basicpy_version,
                "run_status": completed_run_status,
                "fit_warning_count": int(warning_count),
                "fit_fallback_records": _to_jsonable(fit_fallback_records),
                "updated_utc": _utc_now_iso(),
            }
        )
        register_latest_output_reference(
            zarr_path=zarr_path,
            analysis_name="flatfield",
            component=layout.component,
            metadata={
                "data_component": layout.data_component,
                "flatfield_component": layout.flatfield_component,
                "darkfield_component": layout.darkfield_component,
                "baseline_component": layout.baseline_component,
                "source_component": source_component,
                "profile_count": int(profile_count),
                "transformed_volumes": int(transformed_volumes),
                "output_dtype": "float32",
                "output_chunks_tpczyx": [int(v) for v in output_chunks],
                "pyramid_levels": list(pyramid_levels),
                "pyramid_factors_tpczyx": list(pyramid_factors_tpczyx),
                "pyramid_shapes_tpczyx": list(pyramid_shapes_tpczyx),
                "basicpy_version": basicpy_version,
                "resume_parameter_fingerprint": str(layout.parameter_fingerprint),
                "resumed_from_checkpoint": bool(layout.resumed),
                "run_status": completed_run_status,
                "fit_warning_count": int(warning_count),
            },
        )
    except Exception as exc:
        failed_checkpoint_group = zarr.open_group(str(zarr_path), mode="a")[
            layout.checkpoint_component
        ]
        failed_checkpoint_group.attrs.update(
            {
                "run_status": "failed",
                "updated_utc": _utc_now_iso(),
                "last_error": str(exc),
            }
        )
        raise

    _emit(100, "Flatfield correction complete")
    return FlatfieldSummary(
        component=layout.component,
        data_component=layout.data_component,
        flatfield_component=layout.flatfield_component,
        darkfield_component=layout.darkfield_component,
        baseline_component=layout.baseline_component,
        source_component=source_component,
        profile_count=int(profile_count),
        transformed_volumes=int(shape_tpczyx[0] * shape_tpczyx[1] * shape_tpczyx[2]),
        output_chunks_tpczyx=output_chunks,
        output_dtype="float32",
        basicpy_version=basicpy_version,
    )
