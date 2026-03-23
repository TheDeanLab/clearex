#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
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

"""Chunk-parallel shear and rotation workflow for canonical 6D analysis stores."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

# Third Party Imports
import ants
import dask
from dask import delayed
import numpy as np
import zarr

# Local Imports
from clearex.io.ome_store import (
    analysis_auxiliary_root,
    analysis_cache_data_component,
    analysis_cache_root,
    public_analysis_root,
)
from clearex.io.provenance import register_latest_output_reference

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]
Region3DBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
_XYZ_FROM_ZYX = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_AUTO_ESTIMATE_EXTREME_X_FRACTION_DEFAULT = 0.03
_AUTO_ESTIMATE_ZY_STRIDE_DEFAULT = 4
_AUTO_ESTIMATE_SIGNAL_FRACTION_DEFAULT = 0.10
_AUTO_ESTIMATE_MIN_VALID_COLUMNS = 16
_AUTO_ESTIMATE_MIN_FOREGROUND_PIXELS = 64
_RESAMPLE_SUPPORT_EPS = np.float32(1e-3)


def _coerce_bool(value: Any) -> bool:
    """Coerce boolean-like values into ``bool``.

    Parameters
    ----------
    value : Any
        Candidate boolean-like value.

    Returns
    -------
    bool
        Parsed boolean value.

    Raises
    ------
    None
        Unrecognized values are coerced with Python truthiness semantics.
    """
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class ShearTransformSummary:
    """Summary metadata for one shear-transform run.

    Attributes
    ----------
    component : str
        Output latest group component path.
    data_component : str
        Output latest data-array component path.
    volumes_processed : int
        Number of transformed ``(t, p, c)`` volumes.
    output_shape_tpczyx : tuple[int, int, int, int, int, int]
        Shape of transformed output array.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Chunk shape used for transformed output.
    voxel_size_um_zyx : tuple[float, float, float]
        Effective voxel size in ``(z, y, x)`` order used for physical transforms.
    applied_shear_xy : float
        Applied XY shear coefficient in physical-space affine matrix.
    applied_shear_xz : float
        Applied XZ shear coefficient in physical-space affine matrix.
    applied_shear_yz : float
        Applied YZ shear coefficient in physical-space affine matrix.
    applied_rotation_deg_xyz : tuple[float, float, float]
        Applied rotation angles in degrees around ``(x, y, z)`` axes.
    interpolation : str
        Interpolation mode used by ANTsPy resampling.
    output_dtype : str
        Saved transformed output dtype name.
    """

    component: str
    data_component: str
    volumes_processed: int
    output_shape_tpczyx: tuple[int, int, int, int, int, int]
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]
    voxel_size_um_zyx: tuple[float, float, float]
    applied_shear_xy: float
    applied_shear_xz: float
    applied_shear_yz: float
    applied_rotation_deg_xyz: tuple[float, float, float]
    interpolation: str
    output_dtype: str


@dataclass(frozen=True)
class _AffineGeometry:
    """Resolved physical-space affine geometry for one transform run.

    Attributes
    ----------
    matrix_xyz : numpy.ndarray
        Forward linear transform matrix in physical XYZ coordinates.
    offset_xyz : numpy.ndarray
        Forward affine offset in physical XYZ coordinates.
    inverse_matrix_xyz : numpy.ndarray
        Inverse linear transform matrix in physical XYZ coordinates.
    inverse_offset_xyz : numpy.ndarray
        Inverse affine offset in physical XYZ coordinates.
    output_origin_xyz : tuple[float, float, float]
        Output image origin in physical XYZ coordinates.
    output_shape_zyx : tuple[int, int, int]
        Output shape in ``(z, y, x)`` order.
    applied_rotation_deg_xyz : tuple[float, float, float]
        Effective rotation applied in degrees around XYZ axes.
    """

    matrix_xyz: np.ndarray
    offset_xyz: np.ndarray
    inverse_matrix_xyz: np.ndarray
    inverse_offset_xyz: np.ndarray
    output_origin_xyz: tuple[float, float, float]
    output_shape_zyx: tuple[int, int, int]
    applied_rotation_deg_xyz: tuple[float, float, float]


def _axis_chunk_bounds(size: int, chunk_size: int) -> list[tuple[int, int]]:
    """Build contiguous chunk bounds for one axis.

    Parameters
    ----------
    size : int
        Axis length.
    chunk_size : int
        Chunk size.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(start, stop)`` bounds covering the full axis.
    """
    return [
        (start, min(start + chunk_size, size))
        for start in range(0, int(size), int(chunk_size))
    ]


def _iter_output_tile_bounds(
    *,
    out_z_bounds: Sequence[tuple[int, int]],
    out_y_bounds: Sequence[tuple[int, int]],
    out_x_bounds: Sequence[tuple[int, int]],
) -> Iterator[Region3DBounds]:
    """Yield output tile bounds in canonical ``(z, y, x)`` order.

    Parameters
    ----------
    out_z_bounds : sequence[tuple[int, int]]
        Output Z-axis bounds.
    out_y_bounds : sequence[tuple[int, int]]
        Output Y-axis bounds.
    out_x_bounds : sequence[tuple[int, int]]
        Output X-axis bounds.

    Yields
    ------
    Region3DBounds
        One output tile bound tuple.
    """
    for out_z, out_y, out_x in product(out_z_bounds, out_y_bounds, out_x_bounds):
        yield (out_z, out_y, out_x)


def _iter_output_tile_specs(
    *,
    t_count: int,
    p_count: int,
    c_count: int,
    out_z_bounds: Sequence[tuple[int, int]],
    out_y_bounds: Sequence[tuple[int, int]],
    out_x_bounds: Sequence[tuple[int, int]],
) -> Iterator[tuple[int, int, int, Region3DBounds]]:
    """Yield tile specifications for distributed chunk execution.

    Parameters
    ----------
    t_count : int
        Number of time points.
    p_count : int
        Number of positions.
    c_count : int
        Number of channels.
    out_z_bounds : sequence[tuple[int, int]]
        Output Z-axis tile bounds.
    out_y_bounds : sequence[tuple[int, int]]
        Output Y-axis tile bounds.
    out_x_bounds : sequence[tuple[int, int]]
        Output X-axis tile bounds.

    Yields
    ------
    tuple[int, int, int, Region3DBounds]
        ``(t_index, p_index, c_index, output_bounds_zyx)`` tile spec.
    """
    for t_index, p_index, c_index in product(
        range(int(t_count)),
        range(int(p_count)),
        range(int(c_count)),
    ):
        for output_bounds_zyx in _iter_output_tile_bounds(
            out_z_bounds=out_z_bounds,
            out_y_bounds=out_y_bounds,
            out_x_bounds=out_x_bounds,
        ):
            yield (
                int(t_index),
                int(p_index),
                int(c_index),
                output_bounds_zyx,
            )


def _estimate_worker_thread_capacity(client: "Client") -> int:
    """Estimate total worker-thread capacity from Dask scheduler metadata.

    Parameters
    ----------
    client : dask.distributed.Client
        Active Dask client.

    Returns
    -------
    int
        Sum of worker thread counts, clamped to at least ``1``.

    Notes
    -----
    If scheduler metadata cannot be read, this falls back to ``1``.
    """
    try:
        scheduler_info = client.scheduler_info()
    except Exception:
        return 1

    workers = scheduler_info.get("workers", {})
    if not isinstance(workers, dict):
        return 1

    capacity = 0
    for metadata in workers.values():
        if not isinstance(metadata, dict):
            continue
        nthreads = metadata.get("nthreads", 1)
        try:
            capacity += max(1, int(nthreads))
        except (TypeError, ValueError):
            capacity += 1
    return max(1, int(capacity))


def _normalize_roi_padding(value: Any) -> tuple[int, int, int]:
    """Normalize ROI-padding values in ``(z, y, x)`` order.

    Parameters
    ----------
    value : Any
        Candidate padding value.

    Returns
    -------
    tuple[int, int, int]
        Non-negative integer padding values.

    Raises
    ------
    ValueError
        If input does not define exactly three values.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("shear_transform roi_padding_zyx must define three values.")
    return (
        max(0, int(value[0])),
        max(0, int(value[1])),
        max(0, int(value[2])),
    )


def _fit_envelope_tilt_angle_deg(
    *,
    projection_zy: np.ndarray,
    z_spacing_um: float,
    y_spacing_um: float,
    signal_fraction: float,
) -> Optional[float]:
    """Estimate dominant envelope tilt from one ``(z, y)`` projection.

    Parameters
    ----------
    projection_zy : numpy.ndarray
        Input projection in ``(z, y)`` order.
    z_spacing_um : float
        Physical spacing for the projection row axis.
    y_spacing_um : float
        Physical spacing for the projection column axis.
    signal_fraction : float
        Fraction of dynamic range used to derive the foreground threshold.

    Returns
    -------
    float, optional
        Estimated physical-space tilt angle in degrees. Returns ``None`` when
        an envelope cannot be fit robustly.

    Raises
    ------
    None
        Invalid or degenerate inputs return ``None``.
    """
    image = np.asarray(projection_zy, dtype=np.float32)
    if image.ndim != 2 or image.size == 0:
        return None
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return None

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return None

    frac = float(signal_fraction)
    threshold = minimum + (max(0.0, min(1.0, frac)) * (maximum - minimum))
    mask = image > threshold
    if int(np.count_nonzero(mask)) < int(_AUTO_ESTIMATE_MIN_FOREGROUND_PIXELS):
        quantile_threshold = float(np.quantile(finite, 0.90))
        mask = image >= quantile_threshold
    if int(np.count_nonzero(mask)) < int(_AUTO_ESTIMATE_MIN_FOREGROUND_PIXELS):
        return None

    y_indices = np.where(np.any(mask, axis=0))[0]
    if int(y_indices.size) < int(_AUTO_ESTIMATE_MIN_VALID_COLUMNS):
        return None

    top_z = np.full(y_indices.shape, np.nan, dtype=np.float64)
    bottom_z = np.full(y_indices.shape, np.nan, dtype=np.float64)
    for idx, y_index in enumerate(y_indices):
        z_hits = np.where(mask[:, int(y_index)])[0]
        if z_hits.size == 0:
            continue
        top_z[int(idx)] = float(z_hits[0])
        bottom_z[int(idx)] = float(z_hits[-1])

    envelope_angles: list[float] = []
    for envelope in (top_z, bottom_z):
        valid = np.isfinite(envelope)
        if int(np.count_nonzero(valid)) < int(_AUTO_ESTIMATE_MIN_VALID_COLUMNS):
            continue
        y_fit = y_indices[valid].astype(np.float64)
        z_fit = envelope[valid].astype(np.float64)
        z_low, z_high = np.quantile(z_fit, [0.05, 0.95])
        keep = (z_fit >= float(z_low)) & (z_fit <= float(z_high))
        if int(np.count_nonzero(keep)) < int(_AUTO_ESTIMATE_MIN_VALID_COLUMNS):
            continue
        y_keep = y_fit[keep]
        z_keep = z_fit[keep]
        design = np.column_stack((y_keep, np.ones_like(y_keep)))
        slope_px, _ = np.linalg.lstsq(design, z_keep, rcond=None)[0]
        slope_um = float(slope_px) * float(z_spacing_um) / float(y_spacing_um)
        envelope_angles.append(float(np.rad2deg(np.arctan(slope_um))))

    if not envelope_angles:
        return None
    return float(np.median(np.asarray(envelope_angles, dtype=np.float64)))


def _estimate_shear_yz_deg_from_source_extremes(
    *,
    source_array: Any,
    voxel_size_um_zyx: tuple[float, float, float],
    parameters: Mapping[str, Any],
) -> Optional[float]:
    """Estimate ``shear_yz_deg`` from YZ projections of X-extreme source slabs.

    Parameters
    ----------
    source_array : Any
        Source canonical array exposing ``shape`` and NumPy-style indexing.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel spacing in microns for ``(z, y, x)``.
    parameters : mapping[str, Any]
        Normalized runtime parameter mapping.

    Returns
    -------
    float, optional
        Estimated ``shear_yz_deg`` value in physical-space degrees, or ``None``
        when estimation fails.

    Raises
    ------
    None
        Invalid sampling settings or sparse signal return ``None``.
    """
    shape_tpczyx = tuple(int(v) for v in getattr(source_array, "shape", ()))
    if len(shape_tpczyx) != 6:
        return None
    _, p_count, c_count, _, _, x_count = shape_tpczyx
    if x_count <= 0:
        return None

    t_index = max(0, int(parameters.get("auto_estimate_t_index", 0)))
    p_index = max(0, int(parameters.get("auto_estimate_p_index", 0)))
    c_index = max(0, int(parameters.get("auto_estimate_c_index", 0)))
    t_index = min(t_index, int(shape_tpczyx[0]) - 1)
    p_index = min(p_index, int(p_count) - 1)
    c_index = min(c_index, int(c_count) - 1)

    x_fraction = float(
        parameters.get(
            "auto_estimate_extreme_fraction_x",
            _AUTO_ESTIMATE_EXTREME_X_FRACTION_DEFAULT,
        )
    )
    x_fraction = max(1e-6, min(0.5, x_fraction))
    x_span = max(1, int(np.round(float(x_count) * float(x_fraction))))
    x_span = min(int(x_count), int(x_span))

    zy_stride = max(
        1,
        int(
            parameters.get("auto_estimate_zy_stride", _AUTO_ESTIMATE_ZY_STRIDE_DEFAULT)
        ),
    )
    signal_fraction = float(
        parameters.get(
            "auto_estimate_signal_fraction",
            _AUTO_ESTIMATE_SIGNAL_FRACTION_DEFAULT,
        )
    )
    signal_fraction = max(0.0, min(1.0, signal_fraction))

    z_um, y_um, _ = (float(v) for v in voxel_size_um_zyx)
    z_spacing_um = float(z_um) * float(zy_stride)
    y_spacing_um = float(y_um) * float(zy_stride)

    left_block = np.asarray(
        source_array[
            int(t_index),
            int(p_index),
            int(c_index),
            :: int(zy_stride),
            :: int(zy_stride),
            0 : int(x_span),
        ]
    )
    right_block = np.asarray(
        source_array[
            int(t_index),
            int(p_index),
            int(c_index),
            :: int(zy_stride),
            :: int(zy_stride),
            int(x_count - x_span) : int(x_count),
        ]
    )

    estimated_angles: list[float] = []
    for slab in (left_block, right_block):
        slab_zy = np.max(np.asarray(slab), axis=2)
        angle = _fit_envelope_tilt_angle_deg(
            projection_zy=np.asarray(slab_zy),
            z_spacing_um=float(z_spacing_um),
            y_spacing_um=float(y_spacing_um),
            signal_fraction=float(signal_fraction),
        )
        if angle is None:
            continue
        estimated_angles.append(float(angle))
    if not estimated_angles:
        return None
    return float(np.median(np.asarray(estimated_angles, dtype=np.float64)))


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize shear-transform runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized parameter mapping.

    Raises
    ------
    ValueError
        If interpolation mode, dtype, or auto-estimation settings are invalid.
    """
    normalized = dict(parameters)
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    normalized["force_rerun"] = bool(normalized.get("force_rerun", False))
    normalized["chunk_basis"] = "3d"
    normalized["detect_2d_per_slice"] = False
    normalized["use_map_overlap"] = False
    normalized["overlap_zyx"] = [0, 0, 0]
    normalized["memory_overhead_factor"] = float(
        normalized.get("memory_overhead_factor", 2.0)
    )
    for axis_suffix in ("xy", "xz", "yz"):
        shear_key = f"shear_{axis_suffix}"
        shear_deg_key = f"shear_{axis_suffix}_deg"
        if shear_deg_key in normalized:
            angle_deg = float(normalized.get(shear_deg_key, 0.0))
            normalized[shear_deg_key] = angle_deg
            normalized[shear_key] = float(np.tan(np.deg2rad(angle_deg)))
            continue
        shear_value = float(normalized.get(shear_key, 0.0))
        normalized[shear_key] = shear_value
        normalized[shear_deg_key] = float(np.rad2deg(np.arctan(shear_value)))
    normalized["rotation_deg_x"] = float(normalized.get("rotation_deg_x", 0.0))
    normalized["rotation_deg_y"] = float(normalized.get("rotation_deg_y", 0.0))
    normalized["rotation_deg_z"] = float(normalized.get("rotation_deg_z", 0.0))
    normalized["auto_rotate_from_shear"] = _coerce_bool(
        normalized.get("auto_rotate_from_shear", False)
    )
    normalized["auto_estimate_shear_yz"] = _coerce_bool(
        normalized.get("auto_estimate_shear_yz", False)
    )
    normalized["auto_estimate_extreme_fraction_x"] = float(
        normalized.get(
            "auto_estimate_extreme_fraction_x",
            _AUTO_ESTIMATE_EXTREME_X_FRACTION_DEFAULT,
        )
    )
    if (
        normalized["auto_estimate_extreme_fraction_x"] <= 0.0
        or normalized["auto_estimate_extreme_fraction_x"] > 0.5
    ):
        raise ValueError(
            "shear_transform auto_estimate_extreme_fraction_x must be within (0, 0.5]."
        )
    normalized["auto_estimate_zy_stride"] = max(
        1,
        int(
            normalized.get("auto_estimate_zy_stride", _AUTO_ESTIMATE_ZY_STRIDE_DEFAULT)
        ),
    )
    normalized["auto_estimate_signal_fraction"] = float(
        normalized.get(
            "auto_estimate_signal_fraction", _AUTO_ESTIMATE_SIGNAL_FRACTION_DEFAULT
        )
    )
    if (
        normalized["auto_estimate_signal_fraction"] < 0.0
        or normalized["auto_estimate_signal_fraction"] > 1.0
    ):
        raise ValueError(
            "shear_transform auto_estimate_signal_fraction must be within [0, 1]."
        )
    normalized["auto_estimate_t_index"] = max(
        0,
        int(normalized.get("auto_estimate_t_index", 0)),
    )
    normalized["auto_estimate_p_index"] = max(
        0,
        int(normalized.get("auto_estimate_p_index", 0)),
    )
    normalized["auto_estimate_c_index"] = max(
        0,
        int(normalized.get("auto_estimate_c_index", 0)),
    )
    interpolation = (
        str(normalized.get("interpolation", "linear")).strip().lower() or "linear"
    )
    if interpolation not in {"linear", "nearestneighbor", "bspline"}:
        raise ValueError(
            "shear_transform interpolation must be one of "
            "linear, nearestneighbor, or bspline."
        )
    normalized["interpolation"] = interpolation
    normalized["fill_value"] = float(normalized.get("fill_value", 0.0))
    dtype_name = str(normalized.get("output_dtype", "float32")).strip() or "float32"
    normalized["output_dtype"] = np.dtype(dtype_name).name
    normalized["roi_padding_zyx"] = _normalize_roi_padding(
        normalized.get("roi_padding_zyx", [2, 2, 2])
    )
    return normalized


def _extract_voxel_size_um_zyx(
    *,
    root: zarr.Group,
    source_component: str,
) -> tuple[float, float, float]:
    """Extract voxel size metadata in ``(z, y, x)`` order.

    Parameters
    ----------
    root : zarr.Group
        Open analysis-store root group.
    source_component : str
        Source component path.

    Returns
    -------
    tuple[float, float, float]
        Voxel sizes in microns for ``(z, y, x)``.

    Notes
    -----
    Missing metadata falls back to isotropic ``(1.0, 1.0, 1.0)`` microns.
    """
    root_attrs = dict(root.attrs)
    source_attrs: dict[str, Any] = {}
    try:
        source_attrs = dict(root[source_component].attrs)
    except Exception:
        source_attrs = {}

    for attrs in (source_attrs, root_attrs):
        voxel = attrs.get("voxel_size_um_zyx")
        if not isinstance(voxel, (tuple, list)) or len(voxel) < 3:
            continue
        z_um = float(voxel[0])
        y_um = float(voxel[1])
        x_um = float(voxel[2])
        if z_um > 0 and y_um > 0 and x_um > 0:
            return z_um, y_um, x_um

    for attrs in (source_attrs, root_attrs):
        navigate = attrs.get("navigate_experiment")
        if not isinstance(navigate, dict):
            continue
        xy_value = navigate.get("xy_pixel_size_um")
        z_value = navigate.get("z_step_um")
        if xy_value is None or z_value is None:
            continue
        xy_um = float(xy_value)
        z_um = float(z_value)
        if xy_um > 0 and z_um > 0:
            return z_um, xy_um, xy_um

    return 1.0, 1.0, 1.0


def _rotation_matrix_xyz(*, deg_x: float, deg_y: float, deg_z: float) -> np.ndarray:
    """Build rotation matrix from XYZ Euler angles in degrees.

    Parameters
    ----------
    deg_x : float
        Rotation around X axis in degrees.
    deg_y : float
        Rotation around Y axis in degrees.
    deg_z : float
        Rotation around Z axis in degrees.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` rotation matrix using ``Rz @ Ry @ Rx`` composition.
    """
    theta_x = np.deg2rad(float(deg_x))
    theta_y = np.deg2rad(float(deg_y))
    theta_z = np.deg2rad(float(deg_z))
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_x), -np.sin(theta_x)],
            [0.0, np.sin(theta_x), np.cos(theta_x)],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [np.cos(theta_y), 0.0, np.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta_y), 0.0, np.cos(theta_y)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0.0],
            [np.sin(theta_z), np.cos(theta_z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _resolve_affine_geometry(
    *,
    source_shape_zyx: tuple[int, int, int],
    voxel_size_um_zyx: tuple[float, float, float],
    parameters: Mapping[str, Any],
) -> _AffineGeometry:
    """Resolve global physical-space affine geometry.

    Parameters
    ----------
    source_shape_zyx : tuple[int, int, int]
        Source array shape in ``(z, y, x)`` order.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel size in microns for ``(z, y, x)``.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    _AffineGeometry
        Resolved forward/inverse affine geometry and output bounds.
    """
    z_size, y_size, x_size = source_shape_zyx
    z_um, y_um, x_um = voxel_size_um_zyx
    center_xyz = np.array(
        [
            ((x_size - 1) * x_um) / 2.0,
            ((y_size - 1) * y_um) / 2.0,
            ((z_size - 1) * z_um) / 2.0,
        ],
        dtype=np.float64,
    )

    shear_xy = float(parameters.get("shear_xy", 0.0))
    shear_xz = float(parameters.get("shear_xz", 0.0))
    shear_yz = float(parameters.get("shear_yz", 0.0))
    rotation_deg_x = float(parameters.get("rotation_deg_x", 0.0))
    rotation_deg_y = float(parameters.get("rotation_deg_y", 0.0))
    rotation_deg_z = float(parameters.get("rotation_deg_z", 0.0))
    if bool(parameters.get("auto_rotate_from_shear", False)):
        rotation_deg_x += -float(np.rad2deg(np.arctan(shear_yz)))
        rotation_deg_y += float(np.rad2deg(np.arctan(shear_xz)))
        rotation_deg_z += -float(np.rad2deg(np.arctan(shear_xy)))

    shear_matrix = np.array(
        [
            [1.0, shear_xy, shear_xz],
            [0.0, 1.0, shear_yz],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation_matrix = _rotation_matrix_xyz(
        deg_x=rotation_deg_x,
        deg_y=rotation_deg_y,
        deg_z=rotation_deg_z,
    )
    matrix_xyz = rotation_matrix @ shear_matrix
    offset_xyz = center_xyz - (matrix_xyz @ center_xyz)

    x_max = (x_size - 1) * x_um
    y_max = (y_size - 1) * y_um
    z_max = (z_size - 1) * z_um
    corners_xyz = np.asarray(
        [
            [x, y, z]
            for x, y, z in product(
                (0.0, float(x_max)),
                (0.0, float(y_max)),
                (0.0, float(z_max)),
            )
        ],
        dtype=np.float64,
    )
    transformed_xyz = (corners_xyz @ matrix_xyz.T) + offset_xyz
    min_xyz = np.min(transformed_xyz, axis=0)
    max_xyz = np.max(transformed_xyz, axis=0)

    output_x = max(1, int(np.floor((max_xyz[0] - min_xyz[0]) / x_um)) + 1)
    output_y = max(1, int(np.floor((max_xyz[1] - min_xyz[1]) / y_um)) + 1)
    output_z = max(1, int(np.floor((max_xyz[2] - min_xyz[2]) / z_um)) + 1)
    output_shape_zyx = (int(output_z), int(output_y), int(output_x))
    output_origin_xyz = (float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]))

    inverse_matrix_xyz = np.linalg.inv(matrix_xyz)
    inverse_offset_xyz = -inverse_matrix_xyz @ offset_xyz
    return _AffineGeometry(
        matrix_xyz=matrix_xyz,
        offset_xyz=offset_xyz,
        inverse_matrix_xyz=inverse_matrix_xyz,
        inverse_offset_xyz=inverse_offset_xyz,
        output_origin_xyz=output_origin_xyz,
        output_shape_zyx=output_shape_zyx,
        applied_rotation_deg_xyz=(
            float(rotation_deg_x),
            float(rotation_deg_y),
            float(rotation_deg_z),
        ),
    )


def _convert_affine_xyz_to_zyx(
    *,
    matrix_xyz: np.ndarray,
    offset_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert affine terms from XYZ coordinate basis to ZYX basis.

    Parameters
    ----------
    matrix_xyz : numpy.ndarray
        Affine linear matrix in physical XYZ basis.
    offset_xyz : numpy.ndarray
        Affine translation in physical XYZ basis.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Converted ``(matrix_zyx, offset_zyx)`` in physical ZYX basis.
    """
    permutation = _XYZ_FROM_ZYX
    matrix_zyx = permutation.T @ np.asarray(matrix_xyz, dtype=np.float64) @ permutation
    offset_zyx = permutation.T @ np.asarray(offset_xyz, dtype=np.float64)
    return matrix_zyx, offset_zyx


def _source_bounds_for_output_region(
    *,
    output_bounds_zyx: Region3DBounds,
    output_origin_xyz: tuple[float, float, float],
    voxel_size_um_zyx: tuple[float, float, float],
    inverse_matrix_xyz: np.ndarray,
    inverse_offset_xyz: np.ndarray,
    source_shape_zyx: tuple[int, int, int],
    roi_padding_zyx: tuple[int, int, int],
) -> Optional[Region3DBounds]:
    """Estimate source-region bounds required for one output tile.

    Parameters
    ----------
    output_bounds_zyx : Region3DBounds
        Output tile bounds in ``(z, y, x)`` index order.
    output_origin_xyz : tuple[float, float, float]
        Output physical origin in ``(x, y, z)``.
    voxel_size_um_zyx : tuple[float, float, float]
        Voxel size in microns for ``(z, y, x)``.
    inverse_matrix_xyz : numpy.ndarray
        Inverse affine linear matrix in physical XYZ coordinates.
    inverse_offset_xyz : numpy.ndarray
        Inverse affine offset in physical XYZ coordinates.
    source_shape_zyx : tuple[int, int, int]
        Source shape in ``(z, y, x)`` order.
    roi_padding_zyx : tuple[int, int, int]
        Additional read padding in source-voxel units.

    Returns
    -------
    Region3DBounds, optional
        Source bounds in ``(z, y, x)`` or ``None`` when tile does not overlap source.
    """
    (z_start, z_stop), (y_start, y_stop), (x_start, x_stop) = output_bounds_zyx
    z_candidates = (int(z_start), max(int(z_start), int(z_stop) - 1))
    y_candidates = (int(y_start), max(int(y_start), int(y_stop) - 1))
    x_candidates = (int(x_start), max(int(x_start), int(x_stop) - 1))
    z_um, y_um, x_um = voxel_size_um_zyx
    out_origin_x, out_origin_y, out_origin_z = output_origin_xyz

    output_points_xyz = np.asarray(
        [
            [
                out_origin_x + (x_index * x_um),
                out_origin_y + (y_index * y_um),
                out_origin_z + (z_index * z_um),
            ]
            for z_index, y_index, x_index in product(
                z_candidates,
                y_candidates,
                x_candidates,
            )
        ],
        dtype=np.float64,
    )
    source_points_xyz = (output_points_xyz @ inverse_matrix_xyz.T) + inverse_offset_xyz
    min_xyz = np.min(source_points_xyz, axis=0)
    max_xyz = np.max(source_points_xyz, axis=0)

    pad_z, pad_y, pad_x = roi_padding_zyx
    src_z0 = max(0, int(np.floor(min_xyz[2] / z_um)) - int(pad_z))
    src_z1 = min(
        int(source_shape_zyx[0]), int(np.ceil(max_xyz[2] / z_um)) + int(pad_z) + 1
    )
    src_y0 = max(0, int(np.floor(min_xyz[1] / y_um)) - int(pad_y))
    src_y1 = min(
        int(source_shape_zyx[1]), int(np.ceil(max_xyz[1] / y_um)) + int(pad_y) + 1
    )
    src_x0 = max(0, int(np.floor(min_xyz[0] / x_um)) - int(pad_x))
    src_x1 = min(
        int(source_shape_zyx[2]), int(np.ceil(max_xyz[0] / x_um)) + int(pad_x) + 1
    )

    if src_z0 >= src_z1 or src_y0 >= src_y1 or src_x0 >= src_x1:
        return None
    return ((src_z0, src_z1), (src_y0, src_y1), (src_x0, src_x1))


def _cast_output_dtype(data_zyx: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """Cast transformed tile data to configured output dtype.

    Parameters
    ----------
    data_zyx : numpy.ndarray
        Tile data in ``(z, y, x)`` order.
    dtype : numpy.dtype
        Target output dtype.

    Returns
    -------
    numpy.ndarray
        Cast tile with clipping/rounding for integer targets.
    """
    array = np.asarray(data_zyx)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        array = np.clip(np.rint(array), info.min, info.max)
    return np.asarray(array, dtype=dtype)


def _process_and_write_tile(
    *,
    zarr_path: str,
    source_component: str,
    output_component: str,
    t_index: int,
    p_index: int,
    c_index: int,
    output_bounds_zyx: Region3DBounds,
    output_origin_xyz: tuple[float, float, float],
    source_shape_zyx: tuple[int, int, int],
    roi_padding_zyx: tuple[int, int, int],
    voxel_size_um_zyx: tuple[float, float, float],
    inverse_matrix_xyz: np.ndarray,
    inverse_offset_xyz: np.ndarray,
    inverse_matrix_zyx: np.ndarray,
    inverse_offset_zyx: np.ndarray,
    interpolation: str,
    output_dtype: str,
    fill_value: float,
) -> int:
    """Transform one output tile and write it to the result store.

    Parameters
    ----------
    zarr_path : str
        Analysis-store path.
    source_component : str
        Source data component path.
    output_component : str
        Output data component path.
    t_index, p_index, c_index : int
        Selected time, position, and channel index.
    output_bounds_zyx : Region3DBounds
        Output tile bounds in ``(z, y, x)`` order.
    output_origin_xyz : tuple[float, float, float]
        Output physical origin in ``(x, y, z)``.
    source_shape_zyx : tuple[int, int, int]
        Source shape in ``(z, y, x)`` order.
    roi_padding_zyx : tuple[int, int, int]
        Additional read padding in source-voxel units.
    voxel_size_um_zyx : tuple[float, float, float]
        Voxel size in microns for ``(z, y, x)``.
    inverse_matrix_xyz : numpy.ndarray
        Inverse affine linear matrix in physical XYZ coordinates.
    inverse_offset_xyz : numpy.ndarray
        Inverse affine offset in physical XYZ coordinates.
    inverse_matrix_zyx : numpy.ndarray
        Inverse affine linear matrix in physical ZYX coordinates.
    inverse_offset_zyx : numpy.ndarray
        Inverse affine offset in physical ZYX coordinates.
    interpolation : str
        ANTs interpolation mode.
    output_dtype : str
        Output dtype name.
    fill_value : float
        Fill value used for tiles that do not intersect source bounds.

    Returns
    -------
    int
        Always returns ``1`` after writing one tile.
    """
    target_dtype = np.dtype(output_dtype)
    root = zarr.open_group(str(zarr_path), mode="a")
    output_array = root[output_component]
    (out_z0, out_z1), (out_y0, out_y1), (out_x0, out_x1) = output_bounds_zyx
    out_shape_zyx = (
        int(out_z1 - out_z0),
        int(out_y1 - out_y0),
        int(out_x1 - out_x0),
    )
    source_bounds_zyx = _source_bounds_for_output_region(
        output_bounds_zyx=output_bounds_zyx,
        output_origin_xyz=output_origin_xyz,
        voxel_size_um_zyx=voxel_size_um_zyx,
        inverse_matrix_xyz=inverse_matrix_xyz,
        inverse_offset_xyz=inverse_offset_xyz,
        source_shape_zyx=source_shape_zyx,
        roi_padding_zyx=roi_padding_zyx,
    )
    out_slices = (
        slice(int(t_index), int(t_index) + 1),
        slice(int(p_index), int(p_index) + 1),
        slice(int(c_index), int(c_index) + 1),
        slice(int(out_z0), int(out_z1)),
        slice(int(out_y0), int(out_y1)),
        slice(int(out_x0), int(out_x1)),
    )

    if source_bounds_zyx is None:
        tile_data = np.full(out_shape_zyx, fill_value, dtype=target_dtype)
        output_array[out_slices] = tile_data[np.newaxis, np.newaxis, np.newaxis, ...]
        return 1

    source_array = root[source_component]
    (src_z0, src_z1), (src_y0, src_y1), (src_x0, src_x1) = source_bounds_zyx
    src_slices = (
        slice(int(t_index), int(t_index) + 1),
        slice(int(p_index), int(p_index) + 1),
        slice(int(c_index), int(c_index) + 1),
        slice(int(src_z0), int(src_z1)),
        slice(int(src_y0), int(src_y1)),
        slice(int(src_x0), int(src_x1)),
    )
    source_chunk = np.asarray(source_array[src_slices])
    if source_chunk.size == 0:
        tile_data = np.full(out_shape_zyx, fill_value, dtype=target_dtype)
        output_array[out_slices] = tile_data[np.newaxis, np.newaxis, np.newaxis, ...]
        return 1

    source_zyx = np.asarray(source_chunk[0, 0, 0, :, :, :], dtype=np.float32)
    z_um, y_um, x_um = voxel_size_um_zyx
    src_origin_zyx = (float(src_z0 * z_um), float(src_y0 * y_um), float(src_x0 * x_um))
    out_origin_zyx = (
        float(output_origin_xyz[2] + (out_z0 * z_um)),
        float(output_origin_xyz[1] + (out_y0 * y_um)),
        float(output_origin_xyz[0] + (out_x0 * x_um)),
    )

    source_image = ants.from_numpy(source_zyx)
    source_image.set_spacing((z_um, y_um, x_um))
    source_image.set_origin(src_origin_zyx)

    reference_image = ants.from_numpy(np.zeros(out_shape_zyx, dtype=np.float32))
    reference_image.set_spacing((z_um, y_um, x_um))
    reference_image.set_origin(out_origin_zyx)

    transform = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        matrix=np.asarray(inverse_matrix_zyx, dtype=np.float64),
        translation=tuple(float(v) for v in np.asarray(inverse_offset_zyx)),
        center=(0.0, 0.0, 0.0),
    )
    transformed = transform.apply_to_image(
        source_image,
        reference=reference_image,
        interpolation=str(interpolation),
    )
    support_source_image = ants.from_numpy(np.ones_like(source_zyx, dtype=np.float32))
    support_source_image.set_spacing((z_um, y_um, x_um))
    support_source_image.set_origin(src_origin_zyx)
    transformed_support = transform.apply_to_image(
        support_source_image,
        reference=reference_image,
        interpolation="linear",
    )
    transformed_data = np.asarray(transformed.numpy(), dtype=np.float32)
    support_data = np.asarray(transformed_support.numpy(), dtype=np.float32)
    normalized_data = np.full_like(
        transformed_data,
        np.float32(fill_value),
        dtype=np.float32,
    )
    valid_support = support_data > _RESAMPLE_SUPPORT_EPS
    np.divide(
        transformed_data,
        support_data,
        out=normalized_data,
        where=valid_support,
    )
    tile_data = _cast_output_dtype(normalized_data, dtype=target_dtype)
    output_array[out_slices] = tile_data[np.newaxis, np.newaxis, np.newaxis, ...]
    return 1


def run_shear_transform_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> ShearTransformSummary:
    """Run chunk-parallel shear/rotation transform and persist latest output.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Shear-transform parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    ShearTransformSummary
        Summary including output component, geometry, and transform parameters.

    Raises
    ------
    ValueError
        If source component is missing or shape is incompatible.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_parameters(parameters)
    root = zarr.open_group(str(zarr_path), mode="r")
    source_component = str(normalized.get("input_source", "data")).strip() or "data"
    try:
        source_array = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"shear_transform input component '{source_component}' was not found in {zarr_path}."
        ) from exc

    source_shape_tpczyx = tuple(int(v) for v in source_array.shape)
    source_chunks_tpczyx = tuple(int(v) for v in source_array.chunks)
    if len(source_shape_tpczyx) != 6 or len(source_chunks_tpczyx) != 6:
        raise ValueError(
            "shear_transform requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    voxel_size_um_zyx = _extract_voxel_size_um_zyx(
        root=root,
        source_component=source_component,
    )
    if bool(normalized.get("auto_estimate_shear_yz", False)):
        _emit(3, "Estimating shear_yz_deg from x-extreme source slabs")
        estimated_shear_yz_deg = _estimate_shear_yz_deg_from_source_extremes(
            source_array=source_array,
            voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
            parameters=normalized,
        )
        if estimated_shear_yz_deg is not None and np.isfinite(estimated_shear_yz_deg):
            normalized["shear_yz_deg"] = float(estimated_shear_yz_deg)
            normalized["shear_yz"] = float(np.tan(np.deg2rad(estimated_shear_yz_deg)))
            _emit(
                4,
                "Auto-estimated shear_yz_deg=" f"{float(estimated_shear_yz_deg):.3f}",
            )
        else:
            _emit(4, "Auto-estimation failed; using configured shear parameters")
    geometry = _resolve_affine_geometry(
        source_shape_zyx=(
            source_shape_tpczyx[3],
            source_shape_tpczyx[4],
            source_shape_tpczyx[5],
        ),
        voxel_size_um_zyx=voxel_size_um_zyx,
        parameters=normalized,
    )
    inverse_matrix_zyx, inverse_offset_zyx = _convert_affine_xyz_to_zyx(
        matrix_xyz=geometry.inverse_matrix_xyz,
        offset_xyz=geometry.inverse_offset_xyz,
    )

    output_shape_tpczyx = (
        source_shape_tpczyx[0],
        source_shape_tpczyx[1],
        source_shape_tpczyx[2],
        geometry.output_shape_zyx[0],
        geometry.output_shape_zyx[1],
        geometry.output_shape_zyx[2],
    )
    output_chunks_tpczyx = (
        min(source_chunks_tpczyx[0], output_shape_tpczyx[0]),
        min(source_chunks_tpczyx[1], output_shape_tpczyx[1]),
        min(source_chunks_tpczyx[2], output_shape_tpczyx[2]),
        min(source_chunks_tpczyx[3], output_shape_tpczyx[3]),
        min(source_chunks_tpczyx[4], output_shape_tpczyx[4]),
        min(source_chunks_tpczyx[5], output_shape_tpczyx[5]),
    )
    component = public_analysis_root("shear_transform")
    data_component = analysis_cache_data_component("shear_transform")
    cache_root = analysis_cache_root("shear_transform")
    auxiliary_root = analysis_auxiliary_root("shear_transform")
    output_dtype = str(normalized["output_dtype"])

    _emit(5, "Preparing shear-transform output layout")
    root_w = zarr.open_group(str(zarr_path), mode="a")
    if cache_root in root_w:
        del root_w[cache_root]
    if auxiliary_root in root_w:
        del root_w[auxiliary_root]
    latest_group = root_w.require_group(cache_root)
    latest_group.create_dataset(
        name="data",
        shape=output_shape_tpczyx,
        chunks=output_chunks_tpczyx,
        dtype=output_dtype,
        fill_value=float(normalized["fill_value"]),
        overwrite=True,
    )
    latest_group["data"].attrs.update(
        {
            "axes": ["t", "p", "c", "z", "y", "x"],
            "voxel_size_um_zyx": [float(v) for v in voxel_size_um_zyx],
            "source_component": source_component,
            "output_origin_xyz_um": [float(v) for v in geometry.output_origin_xyz],
            "affine_matrix_xyz": geometry.matrix_xyz.tolist(),
            "affine_offset_xyz_um": geometry.offset_xyz.tolist(),
            "inverse_affine_matrix_xyz": geometry.inverse_matrix_xyz.tolist(),
            "inverse_affine_offset_xyz_um": geometry.inverse_offset_xyz.tolist(),
            "applied_rotation_deg_xyz": [
                float(v) for v in geometry.applied_rotation_deg_xyz
            ],
            "storage_policy": "latest_only",
        }
    )
    latest_group.attrs.update(
        {
            "storage_policy": "latest_only",
            "source_component": source_component,
            "parameters": {str(k): v for k, v in dict(normalized).items()},
            "output_shape_tpczyx": [int(v) for v in output_shape_tpczyx],
            "output_chunks_tpczyx": [int(v) for v in output_chunks_tpczyx],
            "output_origin_xyz_um": [float(v) for v in geometry.output_origin_xyz],
            "voxel_size_um_zyx": [float(v) for v in voxel_size_um_zyx],
        }
    )
    root_w.require_group(auxiliary_root).attrs.update(dict(latest_group.attrs))

    out_z_bounds = _axis_chunk_bounds(output_shape_tpczyx[3], output_chunks_tpczyx[3])
    out_y_bounds = _axis_chunk_bounds(output_shape_tpczyx[4], output_chunks_tpczyx[4])
    out_x_bounds = _axis_chunk_bounds(output_shape_tpczyx[5], output_chunks_tpczyx[5])
    roi_padding_zyx = tuple(int(v) for v in normalized["roi_padding_zyx"])

    tile_count_per_volume = (
        int(len(out_z_bounds)) * int(len(out_y_bounds)) * int(len(out_x_bounds))
    )
    total_tiles = (
        int(output_shape_tpczyx[0])
        * int(output_shape_tpczyx[1])
        * int(output_shape_tpczyx[2])
        * int(tile_count_per_volume)
    )
    source_shape_zyx = (
        int(source_shape_tpczyx[3]),
        int(source_shape_tpczyx[4]),
        int(source_shape_tpczyx[5]),
    )
    _emit(10, f"Prepared {total_tiles} shear-transform chunk tasks")
    if client is None:
        delayed_tasks = [
            delayed(_process_and_write_tile)(
                zarr_path=str(zarr_path),
                source_component=source_component,
                output_component=data_component,
                t_index=t_index,
                p_index=p_index,
                c_index=c_index,
                output_bounds_zyx=output_bounds_zyx,
                output_origin_xyz=geometry.output_origin_xyz,
                source_shape_zyx=source_shape_zyx,
                roi_padding_zyx=roi_padding_zyx,
                voxel_size_um_zyx=voxel_size_um_zyx,
                inverse_matrix_xyz=geometry.inverse_matrix_xyz,
                inverse_offset_xyz=geometry.inverse_offset_xyz,
                inverse_matrix_zyx=inverse_matrix_zyx,
                inverse_offset_zyx=inverse_offset_zyx,
                interpolation=str(normalized["interpolation"]),
                output_dtype=output_dtype,
                fill_value=float(normalized["fill_value"]),
            )
            for t_index, p_index, c_index, output_bounds_zyx in _iter_output_tile_specs(
                t_count=output_shape_tpczyx[0],
                p_count=output_shape_tpczyx[1],
                c_count=output_shape_tpczyx[2],
                out_z_bounds=out_z_bounds,
                out_y_bounds=out_y_bounds,
                out_x_bounds=out_x_bounds,
            )
        ]
        _emit(15, "Running shear-transform tasks with local process scheduler")
        dask.compute(*delayed_tasks, scheduler="processes")
        _emit(95, "Completed shear-transform chunk computation")
    else:
        from dask.distributed import as_completed

        thread_capacity = _estimate_worker_thread_capacity(client)
        max_in_flight = max(32, int(thread_capacity) * 4)
        _emit(
            15,
            "Submitting shear-transform chunk tasks to Dask client "
            f"(worker_threads={thread_capacity}, max_in_flight={max_in_flight})",
        )
        completed = 0
        pending_count = 0
        completion_queue = as_completed()

        for t_index, p_index, c_index, output_bounds_zyx in _iter_output_tile_specs(
            t_count=output_shape_tpczyx[0],
            p_count=output_shape_tpczyx[1],
            c_count=output_shape_tpczyx[2],
            out_z_bounds=out_z_bounds,
            out_y_bounds=out_y_bounds,
            out_x_bounds=out_x_bounds,
        ):
            completion_queue.add(
                client.submit(
                    _process_and_write_tile,
                    zarr_path=str(zarr_path),
                    source_component=source_component,
                    output_component=data_component,
                    t_index=t_index,
                    p_index=p_index,
                    c_index=c_index,
                    output_bounds_zyx=output_bounds_zyx,
                    output_origin_xyz=geometry.output_origin_xyz,
                    source_shape_zyx=source_shape_zyx,
                    roi_padding_zyx=roi_padding_zyx,
                    voxel_size_um_zyx=voxel_size_um_zyx,
                    inverse_matrix_xyz=geometry.inverse_matrix_xyz,
                    inverse_offset_xyz=geometry.inverse_offset_xyz,
                    inverse_matrix_zyx=inverse_matrix_zyx,
                    inverse_offset_zyx=inverse_offset_zyx,
                    interpolation=str(normalized["interpolation"]),
                    output_dtype=output_dtype,
                    fill_value=float(normalized["fill_value"]),
                    pure=False,
                )
            )
            pending_count += 1
            if pending_count < max_in_flight:
                continue
            completed_future = next(completion_queue)
            _ = completed_future.result()
            pending_count -= 1
            completed += 1
            progress = 15 + int((completed / max(1, total_tiles)) * 80)
            _emit(progress, f"Processed chunk {completed}/{total_tiles}")

        for completed_future in completion_queue:
            _ = completed_future.result()
            completed += 1
            progress = 15 + int((completed / max(1, total_tiles)) * 80)
            _emit(progress, f"Processed chunk {completed}/{total_tiles}")

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="shear_transform",
        component=component,
        metadata={
            "data_component": data_component,
            "source_component": source_component,
            "output_shape_tpczyx": [int(v) for v in output_shape_tpczyx],
            "output_chunks_tpczyx": [int(v) for v in output_chunks_tpczyx],
            "voxel_size_um_zyx": [float(v) for v in voxel_size_um_zyx],
            "output_origin_xyz_um": [float(v) for v in geometry.output_origin_xyz],
            "applied_shear": {
                "xy": float(normalized["shear_xy"]),
                "xz": float(normalized["shear_xz"]),
                "yz": float(normalized["shear_yz"]),
            },
            "applied_rotation_deg_xyz": [
                float(v) for v in geometry.applied_rotation_deg_xyz
            ],
            "interpolation": str(normalized["interpolation"]),
            "output_dtype": output_dtype,
        },
    )
    _emit(100, "Shear transform complete")
    return ShearTransformSummary(
        component=component,
        data_component=data_component,
        volumes_processed=int(
            output_shape_tpczyx[0] * output_shape_tpczyx[1] * output_shape_tpczyx[2]
        ),
        output_shape_tpczyx=output_shape_tpczyx,
        output_chunks_tpczyx=output_chunks_tpczyx,
        voxel_size_um_zyx=voxel_size_um_zyx,
        applied_shear_xy=float(normalized["shear_xy"]),
        applied_shear_xz=float(normalized["shear_xz"]),
        applied_shear_yz=float(normalized["shear_yz"]),
        applied_rotation_deg_xyz=geometry.applied_rotation_deg_xyz,
        interpolation=str(normalized["interpolation"]),
        output_dtype=output_dtype,
    )
