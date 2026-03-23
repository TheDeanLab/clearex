#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

"""Chunked tile-registration workflow for canonical 6D analysis stores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union

import dask
from dask import delayed
import numpy as np
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation
import zarr

try:
    import ants
except ImportError:  # pragma: no cover - exercised via runtime fallback tests
    ants = None

try:
    from skimage.registration import phase_cross_correlation as _phase_cross_correlation
except ImportError:  # pragma: no cover - optional fast-path dependency
    _phase_cross_correlation = None  # type: ignore[assignment]

from clearex.io.experiment import load_navigate_experiment
from clearex.io.ome_store import (
    analysis_auxiliary_root,
    analysis_cache_data_component,
    analysis_cache_root,
    load_store_metadata,
    public_analysis_root,
)
from clearex.io.provenance import register_latest_output_reference
from clearex.workflow import SpatialCalibrationConfig, spatial_calibration_from_dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]
_ANTS_AFF_ITERATIONS_LEGACY = (2000, 1000, 500, 250)
_ANTS_AFF_ITERATIONS = (200, 100, 50, 25)
_ANTS_AFF_SHRINK_FACTORS = (8, 4, 2, 1)
_ANTS_AFF_SMOOTHING_SIGMAS = (3, 2, 1, 0)
_ANTS_RANDOM_SAMPLING_RATE = 0.20
_ABSOLUTE_RESIDUAL_THRESHOLD_PX = 3.5
_RELATIVE_RESIDUAL_THRESHOLD = 2.5
_DEFAULT_MAX_PAIRWISE_VOXELS = 500_000
_PHASE_CORRELATION_UPSAMPLE_FACTOR = 10
_WEIGHT_EPS = np.float32(1e-6)
_PERMUTE_ZYX_TO_XYZ = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class RegistrationSummary:
    """Summary metadata for one registration run.

    Attributes
    ----------
    component : str
        Output latest-group component.
    data_component : str
        Fused output data component.
    affines_component : str
        Optimized affine dataset component.
    source_component : str
        Full-resolution source component used for fusion.
    requested_source_component : str
        Requested input source before resolution-level expansion.
    pairwise_source_component : str
        Effective selected-level source used for pairwise registration.
    input_resolution_level : int
        Effective registration level used for pairwise estimation.
    requested_input_resolution_level : int
        Requested registration level.
    registration_channel : int
        Channel used to estimate transforms.
    registration_type : str
        Registration transform family.
    anchor_positions : tuple[int, ...]
        Anchor tile index used for each timepoint.
    positions : int
        Number of source positions.
    timepoints : int
        Number of processed timepoints.
    edge_count : int
        Total candidate graph edges.
    active_edge_count : int
        Total active edges after pruning across timepoints.
    dropped_edge_count : int
        Total dropped edges across timepoints.
    output_shape_tpczyx : tuple[int, int, int, int, int, int]
        Output fused shape.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Output chunk layout.
    blend_mode : str
        Blend mode applied during fusion.
    """

    component: str
    data_component: str
    affines_component: str
    source_component: str
    requested_source_component: str
    pairwise_source_component: str
    input_resolution_level: int
    requested_input_resolution_level: int
    registration_channel: int
    registration_type: str
    anchor_positions: tuple[int, ...]
    positions: int
    timepoints: int
    edge_count: int
    active_edge_count: int
    dropped_edge_count: int
    output_shape_tpczyx: tuple[int, int, int, int, int, int]
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]
    blend_mode: str


@dataclass(frozen=True)
class _EdgeSpec:
    """Nominal overlap edge between two positions."""

    fixed_position: int
    moving_position: int
    overlap_bbox_xyz: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    overlap_voxels: int


def _emit(
    progress_callback: Optional[ProgressCallback], percent: int, message: str
) -> None:
    """Emit progress when a callback is available."""
    if progress_callback is None:
        return
    progress_callback(int(percent), str(message))


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    """Parse a float with fallback."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _looks_like_multiposition_header(row: Any) -> bool:
    """Return whether a row resembles a Navigate multiposition header."""
    if not isinstance(row, (list, tuple)) or not row:
        return False
    labels = {str(value).strip().upper() for value in row}
    return {"X", "Y", "Z"}.issubset(labels)


def _parse_multiposition_stage_rows(payload: Any) -> list[dict[str, float]]:
    """Parse stage-coordinate rows from multiposition payloads."""
    if not isinstance(payload, list):
        return []

    rows = list(payload)
    header_index: dict[str, int] = {}
    if rows and _looks_like_multiposition_header(rows[0]):
        header = rows.pop(0)
        if isinstance(header, (list, tuple)):
            for idx, value in enumerate(header):
                header_index[str(value).strip().upper()] = int(idx)

    parsed: list[dict[str, float]] = []
    for row in rows:
        if isinstance(row, Mapping):
            parsed.append(
                {
                    "x": _safe_float(row.get("x", row.get("X")), default=0.0),
                    "y": _safe_float(row.get("y", row.get("Y")), default=0.0),
                    "z": _safe_float(row.get("z", row.get("Z")), default=0.0),
                    "theta": _safe_float(
                        row.get("theta", row.get("THETA")), default=0.0
                    ),
                    "f": _safe_float(row.get("f", row.get("F")), default=0.0),
                }
            )
            continue

        if not isinstance(row, (list, tuple)):
            continue

        def _value(field: str, fallback_index: int) -> float:
            index = header_index.get(field, fallback_index)
            if index < 0 or index >= len(row):
                return 0.0
            return _safe_float(row[index], default=0.0)

        parsed.append(
            {
                "x": _value("X", 0),
                "y": _value("Y", 1),
                "z": _value("Z", 2),
                "theta": _value("THETA", 3),
                "f": _value("F", 4),
            }
        )
    return parsed


def _load_stage_rows(root_attrs: Mapping[str, Any]) -> list[dict[str, float]]:
    """Load multiposition stage rows from experiment metadata."""
    parsed = _parse_multiposition_stage_rows(root_attrs.get("stage_rows"))
    if parsed:
        return parsed

    navigate_payload = root_attrs.get("navigate_experiment")
    if isinstance(navigate_payload, Mapping):
        parsed = _parse_multiposition_stage_rows(navigate_payload.get("MultiPositions"))
        if parsed:
            return parsed

    source_experiment = root_attrs.get("source_experiment")
    if not isinstance(source_experiment, str):
        return []

    experiment_path = Path(source_experiment).expanduser()
    if not experiment_path.exists():
        return []

    sidecar_path = experiment_path.parent / "multi_positions.yml"
    if sidecar_path.exists():
        try:
            text = sidecar_path.read_text()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                try:
                    import yaml  # type: ignore[import-not-found]

                    payload = yaml.safe_load(text)
                except Exception:
                    payload = None
            parsed = _parse_multiposition_stage_rows(payload)
            if parsed:
                return parsed
        except Exception:
            pass

    try:
        experiment = load_navigate_experiment(experiment_path)
    except Exception:
        return []
    return _parse_multiposition_stage_rows(experiment.raw.get("MultiPositions"))


def _resolve_world_axis_delta(
    *,
    row: Mapping[str, float],
    reference: Mapping[str, float],
    binding: str,
) -> float:
    """Resolve one world-axis delta from stage rows."""
    if binding == "none":
        return 0.0
    sign = -1.0 if binding.startswith("-") else 1.0
    source_axis = binding[1:]
    return sign * float(row[source_axis] - reference[source_axis])


def _rotation_matrix_x(theta_deg: float) -> np.ndarray:
    """Build a right-handed rotation matrix around X."""
    theta_rad = math.radians(float(theta_deg))
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cos_theta, -sin_theta], [0.0, sin_theta, cos_theta]],
        dtype=np.float64,
    )


def _position_centroid_anchor(
    stage_rows: Sequence[Mapping[str, float]],
    spatial_calibration: SpatialCalibrationConfig,
    positions: Sequence[int],
) -> int:
    """Return the position closest to the centroid in world space."""
    if not positions:
        return 0
    stage_axis_map = spatial_calibration.stage_axis_map_by_world_axis()
    reference = stage_rows[positions[0]]
    coords: dict[int, np.ndarray] = {}
    for position in positions:
        row = stage_rows[position]
        coords[int(position)] = np.asarray(
            [
                _resolve_world_axis_delta(
                    row=row, reference=reference, binding=stage_axis_map["x"]
                ),
                _resolve_world_axis_delta(
                    row=row, reference=reference, binding=stage_axis_map["y"]
                ),
                _resolve_world_axis_delta(
                    row=row, reference=reference, binding=stage_axis_map["z"]
                ),
            ],
            dtype=np.float64,
        )
    centroid = np.mean(list(coords.values()), axis=0)
    return min(
        positions,
        key=lambda position: float(
            np.linalg.norm(coords[int(position)] - centroid, ord=2)
        ),
    )


def _build_nominal_transforms_xyz(
    stage_rows: Sequence[Mapping[str, float]],
    spatial_calibration: SpatialCalibrationConfig,
    *,
    anchor_position: int,
    positions: Sequence[int],
) -> dict[int, np.ndarray]:
    """Build nominal stage-derived transforms in physical XYZ coordinates."""
    stage_axis_map = spatial_calibration.stage_axis_map_by_world_axis()
    reference = stage_rows[anchor_position]
    transforms: dict[int, np.ndarray] = {}
    for position in positions:
        row = stage_rows[position]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _rotation_matrix_x(float(row["theta"] - reference["theta"]))
        transform[:3, 3] = np.asarray(
            [
                _resolve_world_axis_delta(
                    row=row, reference=reference, binding=stage_axis_map["x"]
                ),
                _resolve_world_axis_delta(
                    row=row, reference=reference, binding=stage_axis_map["y"]
                ),
                _resolve_world_axis_delta(
                    row=row, reference=reference, binding=stage_axis_map["z"]
                ),
            ],
            dtype=np.float64,
        )
        transforms[int(position)] = transform
    return transforms


def _extract_voxel_size_um_zyx(
    root: zarr.hierarchy.Group, source_component: str
) -> tuple[float, float, float]:
    """Extract voxel size in ``(z, y, x)`` order."""
    try:
        source = root[source_component]
    except Exception:
        source = None

    if source is not None:
        try:
            payload = source.attrs.get("voxel_size_um_zyx")
            if isinstance(payload, (list, tuple)) and len(payload) >= 3:
                parsed = tuple(float(value) for value in payload[:3])
                if all(value > 0 for value in parsed):
                    return parsed  # type: ignore[return-value]
        except Exception:
            pass

    try:
        payload = root.attrs.get("voxel_size_um_zyx")
        if isinstance(payload, (list, tuple)) and len(payload) >= 3:
            parsed = tuple(float(value) for value in payload[:3])
            if all(value > 0 for value in parsed):
                return parsed  # type: ignore[return-value]
    except Exception:
        pass

    navigate = root.attrs.get("navigate_experiment")
    if isinstance(navigate, Mapping):
        xy_um = _safe_float(navigate.get("xy_pixel_size_um"), default=1.0)
        z_um = _safe_float(navigate.get("z_step_um"), default=1.0)
        if xy_um > 0 and z_um > 0:
            return (float(z_um), float(xy_um), float(xy_um))
    return (1.0, 1.0, 1.0)


def _component_level_suffix(component: str) -> Optional[int]:
    """Parse a trailing ``level_<n>`` suffix."""
    token = str(component).strip().split("/")[-1]
    if not token.startswith("level_"):
        return None
    try:
        parsed = int(token.split("_", maxsplit=1)[1])
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _resolve_source_components_for_level(
    *,
    root: zarr.hierarchy.Group,
    requested_source_component: str,
    input_resolution_level: int,
) -> tuple[str, str, int]:
    """Resolve full-resolution and effective selected-level source components."""
    requested = str(requested_source_component).strip() or "data"
    full_resolution_source = requested
    if requested.startswith("data_pyramid/level_"):
        full_resolution_source = "data"
    elif "_pyramid/level_" in requested:
        full_resolution_source = requested.split("_pyramid/level_", maxsplit=1)[0]

    if full_resolution_source not in root:
        raise ValueError(
            f"registration input component '{full_resolution_source}' was not found."
        )

    effective_level = max(0, int(input_resolution_level))
    if effective_level <= 0:
        return full_resolution_source, full_resolution_source, 0

    direct_level = _component_level_suffix(requested)
    if (
        direct_level is not None
        and requested in root
        and direct_level == effective_level
    ):
        return full_resolution_source, requested, effective_level

    candidate_components = [
        f"{full_resolution_source}_pyramid/level_{effective_level}",
    ]
    if full_resolution_source == "data":
        candidate_components.insert(0, f"data_pyramid/level_{effective_level}")
    for candidate in candidate_components:
        if candidate in root:
            return full_resolution_source, candidate, effective_level

    raise ValueError(
        "registration input_resolution_level="
        f"{effective_level} was requested for '{full_resolution_source}', "
        "but no matching pyramid component exists."
    )


def _pyramid_factor_zyx_for_level(
    root: zarr.hierarchy.Group,
    *,
    level: int,
    source_component: Optional[str] = None,
) -> tuple[float, float, float]:
    """Return per-axis pyramid factors in ``(z, y, x)`` order."""
    if level <= 0:
        return (1.0, 1.0, 1.0)

    candidate_factors: list[Any] = []
    if source_component:
        try:
            source_attrs = dict(root[str(source_component)].attrs)
        except Exception:
            source_attrs = {}
        candidate_factors.extend(
            [
                source_attrs.get("pyramid_factors_tpczyx"),
                source_attrs.get("resolution_pyramid_factors_tpczyx"),
            ]
        )
    candidate_factors.append(root.attrs.get("data_pyramid_factors_tpczyx"))
    for factors in candidate_factors:
        if isinstance(factors, (tuple, list)) and len(factors) > level:
            entry = factors[level]
            if isinstance(entry, (tuple, list)) and len(entry) >= 6:
                try:
                    return (
                        max(1.0, float(entry[3])),
                        max(1.0, float(entry[4])),
                        max(1.0, float(entry[5])),
                    )
                except Exception:
                    pass
    uniform = float(2 ** int(level))
    return (uniform, uniform, uniform)


def _tile_extent_xyz(
    shape_zyx: Sequence[int], voxel_size_um_zyx: Sequence[float]
) -> np.ndarray:
    """Return physical tile extent in XYZ order."""
    return np.asarray(
        [
            float(shape_zyx[2]) * float(voxel_size_um_zyx[2]),
            float(shape_zyx[1]) * float(voxel_size_um_zyx[1]),
            float(shape_zyx[0]) * float(voxel_size_um_zyx[0]),
        ],
        dtype=np.float64,
    )


def _transform_points_xyz(
    transform_xyz: np.ndarray, points_xyz: np.ndarray
) -> np.ndarray:
    """Apply a homogeneous affine to point rows."""
    homogeneous = np.concatenate(
        [
            points_xyz.astype(np.float64),
            np.ones((points_xyz.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    return (transform_xyz @ homogeneous.T).T[:, :3]


def _bbox_from_points_xyz(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return axis-aligned bounding box from point rows."""
    return (
        np.min(points_xyz, axis=0, initial=np.inf).astype(np.float64),
        np.max(points_xyz, axis=0, initial=-np.inf).astype(np.float64),
    )


def _tile_bbox_xyz(
    transform_xyz: np.ndarray, tile_extent_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed axis-aligned bounding box for one tile."""
    ex, ey, ez = (float(value) for value in tile_extent_xyz)
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [ex, 0.0, 0.0],
            [0.0, ey, 0.0],
            [0.0, 0.0, ez],
            [ex, ey, 0.0],
            [ex, 0.0, ez],
            [0.0, ey, ez],
            [ex, ey, ez],
        ],
        dtype=np.float64,
    )
    return _bbox_from_points_xyz(_transform_points_xyz(transform_xyz, corners))


def _bbox_intersection_xyz(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return XYZ bbox intersection when overlap volume is positive."""
    minimum = np.maximum(left[0], right[0])
    maximum = np.minimum(left[1], right[1])
    if np.any(maximum <= minimum):
        return None
    return minimum, maximum


def _bbox_volume_voxels(
    minimum_xyz: np.ndarray,
    maximum_xyz: np.ndarray,
    voxel_size_um_zyx: Sequence[float],
) -> int:
    """Estimate overlap volume in voxels for one bbox."""
    shape_xyz = (maximum_xyz - minimum_xyz).astype(np.float64)
    voxel_xyz = np.asarray(
        [
            float(voxel_size_um_zyx[2]),
            float(voxel_size_um_zyx[1]),
            float(voxel_size_um_zyx[0]),
        ],
        dtype=np.float64,
    )
    counts = np.maximum(1, np.floor(shape_xyz / np.maximum(voxel_xyz, 1e-6))).astype(
        int
    )
    return int(np.prod(counts, dtype=np.int64))


def _build_edge_specs(
    nominal_transforms_xyz: Mapping[int, np.ndarray],
    *,
    positions: Sequence[int],
    tile_extent_xyz: np.ndarray,
    voxel_size_um_zyx: Sequence[float],
) -> list[_EdgeSpec]:
    """Build overlap graph edges from nominal transformed tile boxes."""
    bboxes = {
        int(position): _tile_bbox_xyz(
            nominal_transforms_xyz[int(position)], tile_extent_xyz
        )
        for position in positions
    }
    edges: list[_EdgeSpec] = []
    ordered = [int(position) for position in positions]
    for idx, fixed_position in enumerate(ordered):
        for moving_position in ordered[idx + 1 :]:
            overlap = _bbox_intersection_xyz(
                bboxes[int(fixed_position)], bboxes[int(moving_position)]
            )
            if overlap is None:
                continue
            overlap_voxels = _bbox_volume_voxels(
                overlap[0], overlap[1], voxel_size_um_zyx
            )
            edges.append(
                _EdgeSpec(
                    fixed_position=int(fixed_position),
                    moving_position=int(moving_position),
                    overlap_bbox_xyz=(
                        (float(overlap[0][0]), float(overlap[1][0])),
                        (float(overlap[0][1]), float(overlap[1][1])),
                        (float(overlap[0][2]), float(overlap[1][2])),
                    ),
                    overlap_voxels=int(overlap_voxels),
                )
            )
    return edges


def _xyz_scale_diagonal(voxel_size_xyz: Sequence[float]) -> np.ndarray:
    """Return diagonal scaling matrix for XYZ voxel sizes."""
    return np.diag(
        [
            float(voxel_size_xyz[0]),
            float(voxel_size_xyz[1]),
            float(voxel_size_xyz[2]),
        ]
    ).astype(np.float64)


def _world_to_input_affine_zyx(
    local_to_world_xyz: np.ndarray,
    *,
    reference_origin_xyz: np.ndarray,
    voxel_size_um_zyx: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``ndimage.affine_transform`` mapping into ZYX input indices."""
    voxel_size_xyz = (
        float(voxel_size_um_zyx[2]),
        float(voxel_size_um_zyx[1]),
        float(voxel_size_um_zyx[0]),
    )
    scale_xyz = _xyz_scale_diagonal(voxel_size_xyz)
    scale_inv_xyz = np.diag([1.0 / value for value in voxel_size_xyz]).astype(
        np.float64
    )
    world_to_local_xyz = np.linalg.inv(local_to_world_xyz)
    matrix = (
        _PERMUTE_ZYX_TO_XYZ
        @ scale_inv_xyz
        @ world_to_local_xyz[:3, :3]
        @ scale_xyz
        @ _PERMUTE_ZYX_TO_XYZ
    )
    offset = (
        _PERMUTE_ZYX_TO_XYZ
        @ scale_inv_xyz
        @ (
            (world_to_local_xyz[:3, :3] @ reference_origin_xyz)
            + world_to_local_xyz[:3, 3]
        )
    )
    return matrix.astype(np.float64), offset.astype(np.float64)


def _resample_source_to_world_grid(
    source_zyx: np.ndarray,
    local_to_world_xyz: np.ndarray,
    *,
    reference_origin_xyz: np.ndarray,
    reference_shape_zyx: tuple[int, int, int],
    voxel_size_um_zyx: Sequence[float],
    order: int,
    cval: float,
) -> np.ndarray:
    """Resample one source volume onto a world-aligned crop/output grid."""
    matrix, offset = _world_to_input_affine_zyx(
        local_to_world_xyz,
        reference_origin_xyz=reference_origin_xyz,
        voxel_size_um_zyx=voxel_size_um_zyx,
    )
    # TODO: GPU acceleration — when a CuPy-backed Dask array backend is active,
    # replace with ``cupyx.scipy.ndimage.affine_transform`` for 10-50× speedup
    # on large volumes.  Requires the same ``matrix``, ``offset``, and
    # ``output_shape`` arguments.  See also the deconvolution subsystem which
    # already supports GPU-pinned LocalCluster workers via
    # ``create_dask_client(..., gpu_enabled=True)``.
    return ndimage.affine_transform(
        np.asarray(source_zyx),
        matrix=matrix,
        offset=offset,
        output_shape=reference_shape_zyx,
        order=int(order),
        mode="constant",
        cval=float(cval),
        prefilter=bool(order > 1),
    ).astype(np.float32, copy=False)


def _source_subvolume_for_overlap(
    local_to_world_xyz: np.ndarray,
    *,
    reference_origin_xyz: np.ndarray,
    reference_shape_zyx: tuple[int, int, int],
    source_shape_zyx: tuple[int, int, int],
    voxel_size_um_zyx: Sequence[float],
    padding: int = 2,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    """Compute the minimal source sub-volume that covers a world-grid crop.

    Returns the ZYX slices into the source array and an adjusted
    ``local_to_world_xyz`` that accounts for the sub-volume origin shift.

    Parameters
    ----------
    local_to_world_xyz : numpy.ndarray
        4×4 homogeneous transform from source local space to world space.
    reference_origin_xyz : numpy.ndarray
        World-space origin of the output crop.
    reference_shape_zyx : tuple[int, int, int]
        Output crop shape in ``(z, y, x)`` order.
    source_shape_zyx : tuple[int, int, int]
        Full source tile shape in ``(z, y, x)`` order.
    voxel_size_um_zyx : sequence of float
        Voxel size in ``(z, y, x)`` order.
    padding : int, optional
        Extra voxel padding per side for interpolation safety (default 2).

    Returns
    -------
    tuple[tuple[slice, slice, slice], numpy.ndarray]
        ``(slices_zyx, adjusted_local_to_world_xyz)`` — Zarr-compatible index
        slices and a translation-adjusted copy of the input transform.
    """
    matrix, offset = _world_to_input_affine_zyx(
        local_to_world_xyz,
        reference_origin_xyz=reference_origin_xyz,
        voxel_size_um_zyx=voxel_size_um_zyx,
    )
    # Build the eight corners of the output crop in output-index space.
    rz, ry, rx = (int(reference_shape_zyx[0]), int(reference_shape_zyx[1]),
                  int(reference_shape_zyx[2]))
    output_corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [rz, 0.0, 0.0],
            [0.0, ry, 0.0],
            [0.0, 0.0, rx],
            [rz, ry, 0.0],
            [rz, 0.0, rx],
            [0.0, ry, rx],
            [rz, ry, rx],
        ],
        dtype=np.float64,
    )
    # Map each corner into source-index space: src_idx = matrix @ out_idx + offset
    source_indices = (matrix @ output_corners.T).T + offset[np.newaxis, :]
    src_min = np.floor(np.min(source_indices, axis=0)).astype(int) - int(padding)
    src_max = np.ceil(np.max(source_indices, axis=0)).astype(int) + int(padding)

    sz, sy, sx = (int(source_shape_zyx[0]), int(source_shape_zyx[1]),
                  int(source_shape_zyx[2]))
    z0, y0, x0 = (max(0, int(src_min[0])), max(0, int(src_min[1])),
                  max(0, int(src_min[2])))
    z1, y1, x1 = (min(sz, int(src_max[0])), min(sy, int(src_max[1])),
                  min(sx, int(src_max[2])))

    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return (
            (slice(0, sz), slice(0, sy), slice(0, sx)),
            local_to_world_xyz.copy(),
        )

    slices_zyx = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    # Shift the transform origin to account for the sub-volume offset.
    voxel_xyz = np.asarray(
        [float(voxel_size_um_zyx[2]), float(voxel_size_um_zyx[1]),
         float(voxel_size_um_zyx[0])],
        dtype=np.float64,
    )
    offset_xyz = np.asarray([float(x0), float(y0), float(z0)],
                            dtype=np.float64) * voxel_xyz
    adjusted = local_to_world_xyz.copy()
    adjusted[:3, 3] = adjusted[:3, 3] + (adjusted[:3, :3] @ offset_xyz)
    return slices_zyx, adjusted


def _downsample_crop_for_registration(
    volume_zyx: np.ndarray,
    *,
    max_voxels: int,
    voxel_size_um_zyx: Sequence[float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Optionally downsample a crop volume to keep it within a voxel budget.

    Parameters
    ----------
    volume_zyx : numpy.ndarray
        3D volume in ``(z, y, x)`` order.
    max_voxels : int
        Maximum total voxels allowed.  If the volume is smaller, it is
        returned unchanged.
    voxel_size_um_zyx : sequence of float
        Original voxel size in ``(z, y, x)`` order.

    Returns
    -------
    tuple[numpy.ndarray, tuple[float, float, float]]
        ``(downsampled_volume, effective_voxel_size_um_zyx)``
    """
    total = int(np.prod(volume_zyx.shape))
    if total <= max(1, int(max_voxels)):
        return volume_zyx, tuple(float(v) for v in voxel_size_um_zyx)  # type: ignore[return-value]
    factor = max(2, int(math.ceil((total / max(1, int(max_voxels))) ** (1.0 / 3.0))))
    zoom_factors = tuple(1.0 / float(factor) for _ in range(3))
    downsampled = ndimage.zoom(
        volume_zyx.astype(np.float32, copy=False),
        zoom_factors,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32, copy=False)
    effective_voxel = (
        float(voxel_size_um_zyx[0]) * float(factor),
        float(voxel_size_um_zyx[1]) * float(factor),
        float(voxel_size_um_zyx[2]) * float(factor),
    )
    return downsampled, effective_voxel


def _crop_from_overlap_bbox(
    overlap_bbox_xyz: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
    *,
    voxel_size_um_zyx: Sequence[float],
    overlap_zyx: Sequence[int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return crop origin and shape for one overlap bbox."""
    pad_xyz = np.asarray(
        [
            float(overlap_zyx[2]) * float(voxel_size_um_zyx[2]),
            float(overlap_zyx[1]) * float(voxel_size_um_zyx[1]),
            float(overlap_zyx[0]) * float(voxel_size_um_zyx[0]),
        ],
        dtype=np.float64,
    )
    minimum_xyz = (
        np.asarray(
            [overlap_bbox_xyz[0][0], overlap_bbox_xyz[1][0], overlap_bbox_xyz[2][0]],
            dtype=np.float64,
        )
        - pad_xyz
    )
    maximum_xyz = (
        np.asarray(
            [overlap_bbox_xyz[0][1], overlap_bbox_xyz[1][1], overlap_bbox_xyz[2][1]],
            dtype=np.float64,
        )
        + pad_xyz
    )
    voxel_xyz = np.asarray(
        [
            float(voxel_size_um_zyx[2]),
            float(voxel_size_um_zyx[1]),
            float(voxel_size_um_zyx[0]),
        ],
        dtype=np.float64,
    )
    size_xyz = np.maximum(voxel_xyz, maximum_xyz - minimum_xyz)
    shape_xyz = np.maximum(1, np.ceil(size_xyz / np.maximum(voxel_xyz, 1e-6))).astype(
        int
    )
    shape_zyx = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
    return minimum_xyz, shape_zyx


def _ants_image_from_zyx(
    volume_zyx: np.ndarray, *, voxel_size_um_zyx: Sequence[float]
) -> Any:
    """Create an ANTs image from a ZYX NumPy array with XYZ spacing."""
    if ants is None:
        raise RuntimeError(
            "registration requires antspyx/ants to estimate tile transforms."
        )
    image = ants.from_numpy(np.asarray(volume_zyx, dtype=np.float32))
    image.set_spacing(
        (
            float(voxel_size_um_zyx[2]),
            float(voxel_size_um_zyx[1]),
            float(voxel_size_um_zyx[0]),
        )
    )
    image.set_origin((0.0, 0.0, 0.0))
    return image


def _ants_transform_to_matrix_xyz(transform: Any) -> np.ndarray:
    """Convert an ANTs affine transform into a homogeneous XYZ matrix."""
    parameters = np.asarray(transform.parameters, dtype=np.float64)
    if parameters.size < 12:
        raise ValueError("Expected a 3D affine transform with 12 parameters.")
    matrix = parameters[:9].reshape(3, 3)
    translation = parameters[9:12]
    center = np.asarray(transform.fixed_parameters, dtype=np.float64)
    offset = translation + center - (matrix @ center)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = matrix
    affine[:3, 3] = offset
    return affine


def _registration_type_to_ants(value: str) -> str:
    """Map normalized registration type to ANTsPy value."""
    mapping = {
        "translation": "Translation",
        "rigid": "Rigid",
        "similarity": "Similarity",
    }
    return mapping[str(value).strip().lower()]


def _phase_correlation_translation(
    fixed_crop_zyx: np.ndarray,
    moving_crop_zyx: np.ndarray,
    *,
    voxel_size_um_zyx: Sequence[float],
) -> np.ndarray:
    """Estimate a translation-only correction via FFT phase correlation.

    Parameters
    ----------
    fixed_crop_zyx : numpy.ndarray
        Fixed reference crop in ``(z, y, x)`` order.
    moving_crop_zyx : numpy.ndarray
        Moving crop in ``(z, y, x)`` order (same shape as *fixed_crop_zyx*).
    voxel_size_um_zyx : sequence of float
        Voxel size in ``(z, y, x)`` order.

    Returns
    -------
    numpy.ndarray
        4×4 homogeneous translation correction matrix in XYZ µm.

    Raises
    ------
    RuntimeError
        If ``scikit-image`` is not installed.
    """
    if _phase_cross_correlation is None:
        raise RuntimeError(
            "phase correlation requires scikit-image "
            "(skimage.registration.phase_cross_correlation)."
        )
    shift_zyx, _error, _phasediff = _phase_cross_correlation(
        fixed_crop_zyx.astype(np.float32, copy=False),
        moving_crop_zyx.astype(np.float32, copy=False),
        upsample_factor=int(_PHASE_CORRELATION_UPSAMPLE_FACTOR),
    )
    shift_zyx = np.asarray(shift_zyx, dtype=np.float64)
    # Convert voxel shift to physical XYZ translation.
    shift_xyz_um = np.asarray(
        [
            float(shift_zyx[2]) * float(voxel_size_um_zyx[2]),
            float(shift_zyx[1]) * float(voxel_size_um_zyx[1]),
            float(shift_zyx[0]) * float(voxel_size_um_zyx[0]),
        ],
        dtype=np.float64,
    )
    correction = np.eye(4, dtype=np.float64)
    correction[:3, 3] = shift_xyz_um
    return correction


def _register_pairwise_overlap(
    *,
    zarr_path: str,
    source_component: str,
    t_index: int,
    registration_channel: int,
    edge: _EdgeSpec,
    nominal_fixed_transform_xyz: np.ndarray,
    nominal_moving_transform_xyz: np.ndarray,
    voxel_size_um_zyx: Sequence[float],
    overlap_zyx: Sequence[int],
    registration_type: str,
    max_pairwise_voxels: int = _DEFAULT_MAX_PAIRWISE_VOXELS,
    ants_iterations: Sequence[int] = _ANTS_AFF_ITERATIONS,
    ants_sampling_rate: float = _ANTS_RANDOM_SAMPLING_RATE,
    use_phase_correlation: bool = False,
    use_fft_initial_alignment: bool = True,
) -> dict[str, Any]:
    """Register one nominal overlap crop and return a correction transform.

    Parameters
    ----------
    zarr_path : str
        Path to canonical analysis-store.
    source_component : str
        Source data component path within the store.
    t_index : int
        Time index.
    registration_channel : int
        Channel used for pairwise estimation.
    edge : _EdgeSpec
        Overlap edge specification.
    nominal_fixed_transform_xyz : numpy.ndarray
        4×4 nominal world transform for the fixed tile.
    nominal_moving_transform_xyz : numpy.ndarray
        4×4 nominal world transform for the moving tile.
    voxel_size_um_zyx : sequence of float
        Voxel size in ``(z, y, x)`` order for the registration level.
    overlap_zyx : sequence of int
        Overlap padding in ``(z, y, x)`` voxels.
    registration_type : str
        Registration family (``translation``, ``rigid``, or ``similarity``).
    max_pairwise_voxels : int, optional
        Maximum voxel budget for overlap crops before sub-sampling.
    ants_iterations : sequence of int, optional
        ANTsPy affine iteration schedule per multi-resolution level.
    ants_sampling_rate : float, optional
        ANTsPy random sampling rate.
    use_phase_correlation : bool, optional
        When ``True`` and ``registration_type`` is ``translation``, use
        FFT-based phase correlation instead of ANTsPy for faster estimation.
    use_fft_initial_alignment : bool, optional
        When ``True``, run FFT phase correlation first to pre-align the
        moving crop before ANTs, reducing iteration requirements.  The
        ANTs result is then composed with the FFT correction.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    source_shape_zyx = tuple(int(v) for v in source.shape[3:])

    crop_origin_xyz, crop_shape_zyx = _crop_from_overlap_bbox(
        edge.overlap_bbox_xyz,
        voxel_size_um_zyx=voxel_size_um_zyx,
        overlap_zyx=overlap_zyx,
    )

    # Compute minimal source sub-volumes covering the overlap crop.
    fixed_slices, fixed_adj_transform = _source_subvolume_for_overlap(
        nominal_fixed_transform_xyz,
        reference_origin_xyz=crop_origin_xyz,
        reference_shape_zyx=crop_shape_zyx,
        source_shape_zyx=source_shape_zyx,
        voxel_size_um_zyx=voxel_size_um_zyx,
    )
    moving_slices, moving_adj_transform = _source_subvolume_for_overlap(
        nominal_moving_transform_xyz,
        reference_origin_xyz=crop_origin_xyz,
        reference_shape_zyx=crop_shape_zyx,
        source_shape_zyx=source_shape_zyx,
        voxel_size_um_zyx=voxel_size_um_zyx,
    )

    fixed_source = np.asarray(
        source[
            int(t_index), int(edge.fixed_position), int(registration_channel),
            fixed_slices[0], fixed_slices[1], fixed_slices[2],
        ],
        dtype=np.float32,
    )
    moving_source = np.asarray(
        source[
            int(t_index), int(edge.moving_position), int(registration_channel),
            moving_slices[0], moving_slices[1], moving_slices[2],
        ],
        dtype=np.float32,
    )

    fixed_crop = _resample_source_to_world_grid(
        fixed_source,
        fixed_adj_transform,
        reference_origin_xyz=crop_origin_xyz,
        reference_shape_zyx=crop_shape_zyx,
        voxel_size_um_zyx=voxel_size_um_zyx,
        order=1,
        cval=0.0,
    )
    moving_crop = _resample_source_to_world_grid(
        moving_source,
        moving_adj_transform,
        reference_origin_xyz=crop_origin_xyz,
        reference_shape_zyx=crop_shape_zyx,
        voxel_size_um_zyx=voxel_size_um_zyx,
        order=1,
        cval=0.0,
    )

    fixed_mask = np.asarray(fixed_crop > 0, dtype=np.float32)
    moving_mask = np.asarray(moving_crop > 0, dtype=np.float32)
    overlap_pixels = int(np.count_nonzero((fixed_mask > 0) & (moving_mask > 0)))
    if (
        overlap_pixels <= 0
        or float(np.std(fixed_crop)) <= 1e-6
        or float(np.std(moving_crop)) <= 1e-6
    ):
        return {
            "fixed_position": int(edge.fixed_position),
            "moving_position": int(edge.moving_position),
            "success": False,
            "reason": "insufficient_overlap_signal",
            "correction_matrix_xyz": np.eye(4, dtype=np.float64).tolist(),
            "overlap_voxels": int(edge.overlap_voxels),
            "nominal_overlap_pixels": int(overlap_pixels),
        }

    # Phase-correlation fast path — skip ANTs entirely for translation-only.
    if (
        bool(use_phase_correlation)
        and registration_type == "translation"
        and _phase_cross_correlation is not None
    ):
        try:
            correction_matrix_xyz = _phase_correlation_translation(
                fixed_crop,
                moving_crop,
                voxel_size_um_zyx=voxel_size_um_zyx,
            )
            return {
                "fixed_position": int(edge.fixed_position),
                "moving_position": int(edge.moving_position),
                "success": True,
                "reason": "phase_correlation",
                "correction_matrix_xyz": correction_matrix_xyz.tolist(),
                "overlap_voxels": int(edge.overlap_voxels),
                "nominal_overlap_pixels": int(overlap_pixels),
            }
        except Exception:
            pass  # Fall through to ANTs path.

    # Sub-sample large crops to stay within the voxel budget.
    reg_voxel_size: Sequence[float] = voxel_size_um_zyx
    reg_fixed = fixed_crop
    reg_moving = moving_crop
    reg_fixed_mask = fixed_mask
    reg_moving_mask = moving_mask
    if int(max_pairwise_voxels) > 0:
        reg_fixed, reg_voxel_size = _downsample_crop_for_registration(
            fixed_crop,
            max_voxels=int(max_pairwise_voxels),
            voxel_size_um_zyx=voxel_size_um_zyx,
        )
        reg_moving, reg_voxel_size = _downsample_crop_for_registration(
            moving_crop,
            max_voxels=int(max_pairwise_voxels),
            voxel_size_um_zyx=voxel_size_um_zyx,
        )
        reg_fixed_mask = np.asarray(reg_fixed > 0, dtype=np.float32)
        reg_moving_mask = np.asarray(reg_moving > 0, dtype=np.float32)

    # FFT initial alignment — pre-align moving crop before ANTs so ANTs
    # starts near the solution and converges with fewer iterations.
    fft_correction_xyz = np.eye(4, dtype=np.float64)
    if bool(use_fft_initial_alignment) and _phase_cross_correlation is not None:
        try:
            shift_zyx, _err, _pd = _phase_cross_correlation(
                reg_fixed.astype(np.float32, copy=False),
                reg_moving.astype(np.float32, copy=False),
                upsample_factor=int(_PHASE_CORRELATION_UPSAMPLE_FACTOR),
            )
            shift_zyx = np.asarray(shift_zyx, dtype=np.float64)
            # Pre-align the moving crop using the detected shift.
            reg_moving = ndimage.shift(
                reg_moving, shift_zyx, order=1, mode="constant", cval=0.0,
            ).astype(np.float32, copy=False)
            reg_moving_mask = np.asarray(reg_moving > 0, dtype=np.float32)
            # Build the FFT correction in physical XYZ coordinates.
            fft_correction_xyz[:3, 3] = np.asarray(
                [
                    float(shift_zyx[2]) * float(reg_voxel_size[2]),
                    float(shift_zyx[1]) * float(reg_voxel_size[1]),
                    float(shift_zyx[0]) * float(reg_voxel_size[0]),
                ],
                dtype=np.float64,
            )
        except Exception:
            fft_correction_xyz = np.eye(4, dtype=np.float64)

    try:
        fixed_image = _ants_image_from_zyx(
            reg_fixed, voxel_size_um_zyx=reg_voxel_size
        )
        moving_image = _ants_image_from_zyx(
            reg_moving, voxel_size_um_zyx=reg_voxel_size
        )
        fixed_mask_image = _ants_image_from_zyx(
            reg_fixed_mask, voxel_size_um_zyx=reg_voxel_size
        )
        moving_mask_image = _ants_image_from_zyx(
            reg_moving_mask, voxel_size_um_zyx=reg_voxel_size
        )
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            fixed_mask=fixed_mask_image,
            moving_mask=moving_mask_image,
            type_of_transform=_registration_type_to_ants(registration_type),
            initial_transform="Identity",
            aff_metric="mattes",
            aff_sampling=32,
            aff_random_sampling_rate=float(ants_sampling_rate),
            aff_iterations=tuple(int(v) for v in ants_iterations),
            aff_shrink_factors=_ANTS_AFF_SHRINK_FACTORS,
            aff_smoothing_sigmas=_ANTS_AFF_SMOOTHING_SIGMAS,
            mask_all_stages=True,
            verbose=False,
        )
        transform = ants.read_transform(registration["fwdtransforms"][0])
        ants_correction_xyz = _ants_transform_to_matrix_xyz(transform)
        # Compose: total = ANTs residual on top of FFT pre-alignment.
        correction_matrix_xyz = ants_correction_xyz @ fft_correction_xyz
        success = True
        reason = ""
    except Exception as exc:
        correction_matrix_xyz = fft_correction_xyz.copy()
        success = bool(not np.allclose(fft_correction_xyz, np.eye(4)))
        reason = str(exc) if not success else "fft_only_fallback"

    return {
        "fixed_position": int(edge.fixed_position),
        "moving_position": int(edge.moving_position),
        "success": bool(success),
        "reason": reason,
        "correction_matrix_xyz": correction_matrix_xyz.tolist(),
        "overlap_voxels": int(edge.overlap_voxels),
        "nominal_overlap_pixels": int(overlap_pixels),
    }


def _matrix_from_pose(params: np.ndarray, registration_type: str) -> np.ndarray:
    """Build a homogeneous correction matrix from pose parameters."""
    matrix = np.eye(4, dtype=np.float64)
    if registration_type == "translation":
        matrix[:3, 3] = np.asarray(params[:3], dtype=np.float64)
        return matrix

    rotation = Rotation.from_rotvec(
        np.asarray(params[:3], dtype=np.float64)
    ).as_matrix()
    translation = np.asarray(params[3:6], dtype=np.float64)
    if registration_type == "rigid":
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix

    scale = float(np.exp(float(params[6])))
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = translation
    return matrix


def _pose_from_matrix(matrix_xyz: np.ndarray, registration_type: str) -> np.ndarray:
    """Convert a homogeneous correction matrix into optimizer parameters."""
    translation = np.asarray(matrix_xyz[:3, 3], dtype=np.float64)
    if registration_type == "translation":
        return translation

    linear = np.asarray(matrix_xyz[:3, :3], dtype=np.float64)
    if registration_type == "similarity":
        scale = float(np.cbrt(max(np.linalg.det(linear), 1e-12)))
        rotation_matrix = linear / max(scale, 1e-12)
        rotation = Rotation.from_matrix(rotation_matrix)
        return np.concatenate(
            [rotation.as_rotvec(), translation, np.asarray([math.log(scale)])]
        )

    rotation = Rotation.from_matrix(linear)
    return np.concatenate([rotation.as_rotvec(), translation])


def _translation_residual_voxels(
    measured_xyz: np.ndarray,
    predicted_xyz: np.ndarray,
    *,
    voxel_size_um_zyx: Sequence[float],
) -> float:
    """Return translation residual magnitude in selected-level voxels."""
    delta_xyz = np.asarray(predicted_xyz[:3, 3] - measured_xyz[:3, 3], dtype=np.float64)
    mean_voxel = float(np.mean(np.asarray(voxel_size_um_zyx, dtype=np.float64)))
    return float(np.linalg.norm(delta_xyz, ord=2) / max(mean_voxel, 1e-6))


def _component_positions(
    positions: Sequence[int],
    active_edge_indices: Sequence[int],
    edges: Sequence[_EdgeSpec],
) -> list[list[int]]:
    """Return connected position components for the active edge graph."""
    adjacency: dict[int, set[int]] = {int(position): set() for position in positions}
    for edge_index in active_edge_indices:
        edge = edges[int(edge_index)]
        adjacency[int(edge.fixed_position)].add(int(edge.moving_position))
        adjacency[int(edge.moving_position)].add(int(edge.fixed_position))

    components: list[list[int]] = []
    visited: set[int] = set()
    for position in positions:
        node = int(position)
        if node in visited:
            continue
        queue: deque[int] = deque([node])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(int(current))
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in visited:
                    queue.append(int(neighbor))
        components.append(component)
    return components


def _solve_translation_component(
    *,
    component_positions: Sequence[int],
    active_edge_indices: Sequence[int],
    edge_results: Sequence[Mapping[str, Any]],
    edges: Sequence[_EdgeSpec],
    anchor_position: int,
) -> dict[int, np.ndarray]:
    """Solve translation-only correction poses for one connected component."""
    solved = {
        int(position): np.eye(4, dtype=np.float64) for position in component_positions
    }
    variable_positions = [
        int(position)
        for position in component_positions
        if int(position) != int(anchor_position)
    ]
    if not variable_positions:
        return solved

    column_index = {position: idx for idx, position in enumerate(variable_positions)}
    equations: list[list[float]] = []
    targets: list[float] = []
    for edge_index in active_edge_indices:
        edge = edges[int(edge_index)]
        fixed_position = int(edge.fixed_position)
        moving_position = int(edge.moving_position)
        if (
            fixed_position not in column_index
            and fixed_position != int(anchor_position)
            and fixed_position not in component_positions
        ):
            continue
        if (
            moving_position not in column_index
            and moving_position != int(anchor_position)
            and moving_position not in component_positions
        ):
            continue
        measurement = np.asarray(
            edge_results[int(edge_index)]["correction_matrix_xyz"], dtype=np.float64
        )
        weight = math.sqrt(
            max(1.0, float(edge_results[int(edge_index)].get("overlap_voxels", 1)))
        )
        delta = np.asarray(measurement[:3, 3], dtype=np.float64)
        for axis in range(3):
            row = [0.0] * (len(variable_positions) * 3)
            if fixed_position in column_index:
                row[(column_index[fixed_position] * 3) + axis] -= float(weight)
            if moving_position in column_index:
                row[(column_index[moving_position] * 3) + axis] += float(weight)
            equations.append(row)
            targets.append(float(weight) * float(delta[axis]))

    if not equations:
        return solved

    design = np.asarray(equations, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    for position in variable_positions:
        base = column_index[position] * 3
        solved[int(position)][:3, 3] = np.asarray(
            coefficients[base : base + 3], dtype=np.float64
        )
    return solved


def _solve_nonlinear_component(
    *,
    component_positions: Sequence[int],
    active_edge_indices: Sequence[int],
    edge_results: Sequence[Mapping[str, Any]],
    edges: Sequence[_EdgeSpec],
    registration_type: str,
    anchor_position: int,
    voxel_size_um_zyx: Sequence[float],
) -> dict[int, np.ndarray]:
    """Solve rigid/similarity correction poses for one connected component."""
    solved = {
        int(position): np.eye(4, dtype=np.float64) for position in component_positions
    }
    variable_positions = [
        int(position)
        for position in component_positions
        if int(position) != int(anchor_position)
    ]
    if not variable_positions:
        return solved

    pose_size = 6 if registration_type == "rigid" else 7
    variable_index = {position: idx for idx, position in enumerate(variable_positions)}
    initial = np.zeros(len(variable_positions) * pose_size, dtype=np.float64)
    for edge_index in active_edge_indices:
        edge = edges[int(edge_index)]
        if int(edge.moving_position) not in variable_index:
            continue
        if int(edge.fixed_position) != int(anchor_position):
            continue
        measurement = np.asarray(
            edge_results[int(edge_index)]["correction_matrix_xyz"], dtype=np.float64
        )
        base = variable_index[int(edge.moving_position)] * pose_size
        initial[base : base + pose_size] = _pose_from_matrix(
            measurement, registration_type
        )

    def _matrix_for(position: int, params: np.ndarray) -> np.ndarray:
        if int(position) == int(anchor_position):
            return np.eye(4, dtype=np.float64)
        base = variable_index[int(position)] * pose_size
        return _matrix_from_pose(params[base : base + pose_size], registration_type)

    def _residuals(params: np.ndarray) -> np.ndarray:
        residual_values: list[float] = []
        mean_voxel = float(np.mean(np.asarray(voxel_size_um_zyx, dtype=np.float64)))
        for edge_index in active_edge_indices:
            edge = edges[int(edge_index)]
            measured = np.asarray(
                edge_results[int(edge_index)]["correction_matrix_xyz"], dtype=np.float64
            )
            weight = math.sqrt(
                max(1.0, float(edge_results[int(edge_index)].get("overlap_voxels", 1)))
            )
            fixed_matrix = _matrix_for(int(edge.fixed_position), params)
            moving_matrix = _matrix_for(int(edge.moving_position), params)
            predicted = np.linalg.inv(fixed_matrix) @ moving_matrix

            measured_linear = np.asarray(measured[:3, :3], dtype=np.float64)
            predicted_linear = np.asarray(predicted[:3, :3], dtype=np.float64)
            if registration_type == "similarity":
                measured_scale = float(
                    np.cbrt(max(np.linalg.det(measured_linear), 1e-12))
                )
                predicted_scale = float(
                    np.cbrt(max(np.linalg.det(predicted_linear), 1e-12))
                )
                measured_rotation = measured_linear / max(measured_scale, 1e-12)
                predicted_rotation = predicted_linear / max(predicted_scale, 1e-12)
                scale_residual = math.log(
                    max(predicted_scale, 1e-12) / max(measured_scale, 1e-12)
                )
            else:
                measured_rotation = measured_linear
                predicted_rotation = predicted_linear
                scale_residual = 0.0

            delta_rotation = Rotation.from_matrix(
                measured_rotation.T @ predicted_rotation
            ).as_rotvec()
            delta_translation = np.asarray(
                predicted[:3, 3] - measured[:3, 3], dtype=np.float64
            ) / max(mean_voxel, 1e-6)
            residual_values.extend((weight * delta_rotation).tolist())
            residual_values.extend((weight * delta_translation).tolist())
            if registration_type == "similarity":
                residual_values.append(float(weight) * float(scale_residual))
        return np.asarray(residual_values, dtype=np.float64)

    result = optimize.least_squares(
        _residuals,
        initial,
        method="trf",
        x_scale="jac",
        max_nfev=200,
    )
    solved = {
        int(position): np.eye(4, dtype=np.float64) for position in component_positions
    }
    for position in variable_positions:
        base = variable_index[int(position)] * pose_size
        solved[int(position)] = _matrix_from_pose(
            result.x[base : base + pose_size], registration_type
        )
    return solved


def _solve_with_pruning(
    *,
    positions: Sequence[int],
    edges: Sequence[_EdgeSpec],
    edge_results: Sequence[Mapping[str, Any]],
    anchor_position: int,
    registration_type: str,
    voxel_size_um_zyx: Sequence[float],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Solve correction transforms with simple bad-link pruning."""
    success_mask = np.asarray(
        [bool(result.get("success", False)) for result in edge_results], dtype=bool
    )
    active_mask = success_mask.copy()
    residuals = np.full(len(edges), np.nan, dtype=np.float64)
    solved: dict[int, np.ndarray] = {
        int(position): np.eye(4, dtype=np.float64) for position in positions
    }
    if not np.any(active_mask):
        return solved, active_mask, residuals

    while True:
        solved = {int(position): np.eye(4, dtype=np.float64) for position in positions}
        active_edge_indices = np.flatnonzero(active_mask)
        components = _component_positions(positions, active_edge_indices, edges)
        for component_positions in components:
            component_edges = [
                int(edge_index)
                for edge_index in active_edge_indices
                if int(edges[int(edge_index)].fixed_position) in component_positions
                and int(edges[int(edge_index)].moving_position) in component_positions
            ]
            component_anchor = (
                int(anchor_position)
                if int(anchor_position) in component_positions
                else int(component_positions[0])
            )
            if registration_type == "translation":
                component_solution = _solve_translation_component(
                    component_positions=component_positions,
                    active_edge_indices=component_edges,
                    edge_results=edge_results,
                    edges=edges,
                    anchor_position=component_anchor,
                )
            else:
                component_solution = _solve_nonlinear_component(
                    component_positions=component_positions,
                    active_edge_indices=component_edges,
                    edge_results=edge_results,
                    edges=edges,
                    registration_type=registration_type,
                    anchor_position=component_anchor,
                    voxel_size_um_zyx=voxel_size_um_zyx,
                )
            solved.update(component_solution)

        residuals[:] = np.nan
        for edge_index in active_edge_indices:
            edge = edges[int(edge_index)]
            measured = np.asarray(
                edge_results[int(edge_index)]["correction_matrix_xyz"], dtype=np.float64
            )
            predicted = (
                np.linalg.inv(solved[int(edge.fixed_position)])
                @ solved[int(edge.moving_position)]
            )
            residuals[int(edge_index)] = _translation_residual_voxels(
                measured, predicted, voxel_size_um_zyx=voxel_size_um_zyx
            )

        active_residuals = residuals[np.isfinite(residuals)]
        if active_residuals.size == 0:
            break
        worst_edge_index = int(np.nanargmax(residuals))
        worst_residual = float(residuals[worst_edge_index])
        mean_residual = float(np.mean(active_residuals))
        if worst_residual <= float(
            _ABSOLUTE_RESIDUAL_THRESHOLD_PX
        ) or worst_residual <= max(
            float(_ABSOLUTE_RESIDUAL_THRESHOLD_PX),
            mean_residual * float(_RELATIVE_RESIDUAL_THRESHOLD),
        ):
            break

        edge = edges[worst_edge_index]
        component_positions = next(
            component
            for component in components
            if int(edge.fixed_position) in component
            and int(edge.moving_position) in component
        )
        component_active = [
            int(edge_index)
            for edge_index in active_edge_indices
            if int(edges[int(edge_index)].fixed_position) in component_positions
            and int(edges[int(edge_index)].moving_position) in component_positions
        ]
        if len(component_active) <= max(0, len(component_positions) - 1):
            break
        active_mask[worst_edge_index] = False

    return solved, active_mask, residuals


def _blend_weight_volume(
    shape_zyx: Sequence[int],
    *,
    blend_mode: str,
    overlap_zyx: Sequence[int],
) -> np.ndarray:
    """Build a separable edge-feathered weight volume.

    .. warning::

       This materializes the full 3D volume.  For large tiles use
       :func:`_blend_weight_profiles` and reconstruct sub-volumes
       on-the-fly from the 1D profiles.
    """
    shape = tuple(int(value) for value in shape_zyx)
    if str(blend_mode).strip().lower() == "average":
        return np.ones(shape, dtype=np.float32)

    profiles = _blend_weight_profiles(
        shape_zyx, blend_mode=blend_mode, overlap_zyx=overlap_zyx
    )
    return (
        profiles[0][:, np.newaxis, np.newaxis]
        * profiles[1][np.newaxis, :, np.newaxis]
        * profiles[2][np.newaxis, np.newaxis, :]
    ).astype(np.float32)


def _blend_weight_profiles(
    shape_zyx: Sequence[int],
    *,
    blend_mode: str,
    overlap_zyx: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three separable 1D blend-weight profiles for ``(z, y, x)``.

    The full 3D weight volume is the outer product of the three profiles.
    This function avoids materializing it, making it safe for arbitrarily
    large tiles.

    Parameters
    ----------
    shape_zyx : sequence of int
        Source tile shape in ``(z, y, x)`` order.
    blend_mode : str
        ``"feather"`` or ``"average"``.
    overlap_zyx : sequence of int
        Ramp width per axis in ``(z, y, x)`` order.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(profile_z, profile_y, profile_x)`` — each a 1D float32 array.
    """
    shape = tuple(int(value) for value in shape_zyx)
    profiles: list[np.ndarray] = []
    for axis_size, ramp_width in zip(shape, overlap_zyx, strict=False):
        if str(blend_mode).strip().lower() == "average":
            profiles.append(np.ones(int(axis_size), dtype=np.float32))
            continue
        profile = np.ones(int(axis_size), dtype=np.float32)
        width = max(0, min(int(ramp_width), max(0, int(axis_size // 2))))
        if width > 0:
            ramp = 0.5 - 0.5 * np.cos(
                np.linspace(0.0, np.pi, width, dtype=np.float32)
            )
            profile[:width] = ramp
            profile[-width:] = np.minimum(profile[-width:], ramp[::-1])
        profiles.append(profile)
    return (profiles[0], profiles[1], profiles[2])


def _blend_weight_subvolume_from_profiles(
    profile_z: np.ndarray,
    profile_y: np.ndarray,
    profile_x: np.ndarray,
    slices_zyx: tuple[slice, slice, slice],
) -> np.ndarray:
    """Reconstruct a sub-volume of the blend weight from 1D profiles.

    Only the sub-volume defined by *slices_zyx* is materialized, keeping
    memory proportional to the sub-volume rather than the full tile.

    Parameters
    ----------
    profile_z, profile_y, profile_x : numpy.ndarray
        1D blend-weight profiles for each axis.
    slices_zyx : tuple[slice, slice, slice]
        Sub-volume slices into the full tile.

    Returns
    -------
    numpy.ndarray
        3D float32 weight sub-volume.
    """
    pz = profile_z[slices_zyx[0]]
    py = profile_y[slices_zyx[1]]
    px = profile_x[slices_zyx[2]]
    return (
        pz[:, np.newaxis, np.newaxis]
        * py[np.newaxis, :, np.newaxis]
        * px[np.newaxis, np.newaxis, :]
    ).astype(np.float32, copy=False)


def _cast_to_dtype(data: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """Cast float output to the requested dtype."""
    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        return np.clip(np.rint(data), info.min, info.max).astype(out_dtype, copy=False)
    return np.asarray(data, dtype=out_dtype)


def _axis_chunk_bounds(size: int, chunk_size: int) -> list[tuple[int, int]]:
    """Build contiguous chunk bounds for one axis."""
    return [
        (start, min(start + chunk_size, int(size)))
        for start in range(0, int(size), int(chunk_size))
    ]


def _estimate_fusion_chunk_bytes(
    chunk_shape_zyx: Sequence[int],
    source_shape_zyx: Sequence[int],
    n_positions: int,
) -> int:
    """Estimate peak memory for a single fusion-worker invocation.

    The worker simultaneously holds:

    * Two float32 accumulation buffers (``chunk_sum``, ``chunk_weight``)
      sized to the output chunk.
    * Per overlapping position (worst-case *n_positions*): a source sub-volume
      read, a resampled volume, a weight sub-volume, and a resampled weight,
      each up to source-tile-sized but more typically chunk-sized.  Only one
      position is live at a time, so we estimate for one.

    Returns
    -------
    int
        Estimated bytes.
    """
    chunk_voxels = int(np.prod(list(chunk_shape_zyx)))
    # Worst-case sub-volume is full tile (clipped by _source_subvolume_for_overlap
    # but for estimation purposes we use the full tile).
    source_voxels = int(np.prod(list(source_shape_zyx)))
    # Two accumulators + 4 transient arrays per position (only 1 position live).
    bytes_per_float32 = 4
    return (2 * chunk_voxels + 4 * source_voxels) * bytes_per_float32


def _process_and_write_registration_chunk(
    *,
    zarr_path: str,
    source_component: str,
    output_component: str,
    affines_component: str,
    transformed_bboxes_component: str,
    blend_weights_component: str,
    t_index: int,
    c_index: int,
    z_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    x_bounds: tuple[int, int],
    output_origin_xyz: Sequence[float],
    voxel_size_um_zyx: Sequence[float],
    output_dtype: str,
) -> int:
    """Render one fused output chunk into the registration result store.

    Parameters
    ----------
    zarr_path : str
        Canonical analysis-store path.
    source_component : str
        Full-resolution source data component.
    output_component : str
        Output fused data component.
    affines_component : str
        Per-tile affine dataset component.
    transformed_bboxes_component : str
        Per-tile transformed bounding box dataset component.
    blend_weights_component : str
        Pre-computed blend weight volume component (lazily loaded per worker).
    t_index : int
        Time index.
    c_index : int
        Channel index.
    z_bounds, y_bounds, x_bounds : tuple[int, int]
        Inclusive start and exclusive end for each spatial axis.
    output_origin_xyz : sequence of float
        Output mosaic origin in XYZ µm.
    voxel_size_um_zyx : sequence of float
        Voxel size in ``(z, y, x)`` order.
    output_dtype : str
        Desired output dtype string.

    Returns
    -------
    int
        Always ``1`` on success.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    source_shape_zyx = tuple(int(v) for v in source.shape[3:])
    z0, z1 = (int(z_bounds[0]), int(z_bounds[1]))
    y0, y1 = (int(y_bounds[0]), int(y_bounds[1]))
    x0, x1 = (int(x_bounds[0]), int(x_bounds[1]))
    chunk_shape_zyx = (z1 - z0, y1 - y0, x1 - x0)
    chunk_origin_xyz = np.asarray(
        [
            float(output_origin_xyz[0]) + (float(x0) * float(voxel_size_um_zyx[2])),
            float(output_origin_xyz[1]) + (float(y0) * float(voxel_size_um_zyx[1])),
            float(output_origin_xyz[2]) + (float(z0) * float(voxel_size_um_zyx[0])),
        ],
        dtype=np.float64,
    )
    chunk_bbox_xyz = np.asarray(
        [
            [
                float(chunk_origin_xyz[0]),
                float(chunk_origin_xyz[0])
                + (float(chunk_shape_zyx[2]) * float(voxel_size_um_zyx[2])),
            ],
            [
                float(chunk_origin_xyz[1]),
                float(chunk_origin_xyz[1])
                + (float(chunk_shape_zyx[1]) * float(voxel_size_um_zyx[1])),
            ],
            [
                float(chunk_origin_xyz[2]),
                float(chunk_origin_xyz[2])
                + (float(chunk_shape_zyx[0]) * float(voxel_size_um_zyx[0])),
            ],
        ],
        dtype=np.float64,
    )

    # Load the pre-computed 1D blend-weight profiles (tiny — sum of axis
    # lengths × 4 bytes) and reconstruct sub-volumes on-the-fly per tile.
    blend_group = root[blend_weights_component]
    profile_z = np.asarray(blend_group["profile_z"][:], dtype=np.float32)
    profile_y = np.asarray(blend_group["profile_y"][:], dtype=np.float32)
    profile_x = np.asarray(blend_group["profile_x"][:], dtype=np.float32)

    chunk_sum = np.zeros(chunk_shape_zyx, dtype=np.float32)
    chunk_weight = np.zeros(chunk_shape_zyx, dtype=np.float32)
    position_count = int(source.shape[1])
    affines = root[affines_component]
    transformed_bboxes = root[transformed_bboxes_component]
    for position_index in range(position_count):
        bbox_payload = np.asarray(
            transformed_bboxes[int(t_index), int(position_index)], dtype=np.float64
        )
        bbox_min = bbox_payload[:3]
        bbox_max = bbox_payload[3:]
        if np.any(bbox_max <= chunk_bbox_xyz[:, 0]) or np.any(
            chunk_bbox_xyz[:, 1] <= bbox_min
        ):
            continue
        transform_xyz = np.asarray(
            affines[int(t_index), int(position_index)], dtype=np.float64
        )

        # Compute the minimal source sub-volume needed for this output chunk.
        src_slices, adj_transform = _source_subvolume_for_overlap(
            transform_xyz,
            reference_origin_xyz=chunk_origin_xyz,
            reference_shape_zyx=chunk_shape_zyx,
            source_shape_zyx=source_shape_zyx,
            voxel_size_um_zyx=voxel_size_um_zyx,
        )
        source_volume = np.asarray(
            source[
                int(t_index), int(position_index), int(c_index),
                src_slices[0], src_slices[1], src_slices[2],
            ],
            dtype=np.float32,
        )
        warped_volume = _resample_source_to_world_grid(
            source_volume,
            adj_transform,
            reference_origin_xyz=chunk_origin_xyz,
            reference_shape_zyx=chunk_shape_zyx,
            voxel_size_um_zyx=voxel_size_um_zyx,
            order=1,
            cval=0.0,
        )

        # Reconstruct blend weights for only the needed source sub-volume
        # from the 1D profiles — avoids materializing the full 3D weight volume.
        weight_sub = _blend_weight_subvolume_from_profiles(
            profile_z, profile_y, profile_x, src_slices,
        )
        warped_weight = _resample_source_to_world_grid(
            weight_sub,
            adj_transform,
            reference_origin_xyz=chunk_origin_xyz,
            reference_shape_zyx=chunk_shape_zyx,
            voxel_size_um_zyx=voxel_size_um_zyx,
            order=1,
            cval=0.0,
        )
        chunk_sum += warped_volume * warped_weight
        chunk_weight += warped_weight

    normalized = np.zeros(chunk_shape_zyx, dtype=np.float32)
    np.divide(
        chunk_sum,
        np.maximum(chunk_weight, _WEIGHT_EPS),
        out=normalized,
        where=chunk_weight > 0,
    )
    write_root = zarr.open_group(str(zarr_path), mode="a")
    write_root[output_component][int(t_index), 0, int(c_index), z0:z1, y0:y1, x0:x1] = (
        _cast_to_dtype(normalized, np.dtype(output_dtype))
    )
    return 1


def _prepare_output_group(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    parameters: Mapping[str, Any],
    output_shape_tpczyx: tuple[int, int, int, int, int, int],
    output_chunks_tpczyx: tuple[int, int, int, int, int, int],
    voxel_size_um_zyx: Sequence[float],
    output_origin_xyz: Sequence[float],
    source_tile_shape_zyx: tuple[int, int, int],
    blend_mode: str,
    overlap_zyx: Sequence[int],
) -> tuple[str, str, str, str]:
    """Create latest registration output datasets.

    Returns
    -------
    tuple[str, str, str, str]
        ``(component, data_component, affines_component,
        blend_weights_component)``
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    cache_root = analysis_cache_root("registration")
    auxiliary_root = analysis_auxiliary_root("registration")
    if cache_root in root:
        del root[cache_root]
    if auxiliary_root in root:
        del root[auxiliary_root]
    latest = root.require_group(cache_root)
    latest.create_dataset(
        name="data",
        shape=output_shape_tpczyx,
        chunks=output_chunks_tpczyx,
        dtype=root[source_component].dtype,
        overwrite=True,
    )
    latest["data"].attrs.update(
        {
            "axes": ["t", "p", "c", "z", "y", "x"],
            "source_component": str(source_component),
            "voxel_size_um_zyx": [float(value) for value in voxel_size_um_zyx],
            "output_origin_xyz_um": [float(value) for value in output_origin_xyz],
            "storage_policy": "latest_only",
        }
    )
    auxiliary_group = root.require_group(auxiliary_root)
    auxiliary_group.attrs.update(
        {
            "storage_policy": "latest_only",
            "source_component": str(source_component),
            "parameters": {str(key): value for key, value in dict(parameters).items()},
            "output_shape_tpczyx": [int(value) for value in output_shape_tpczyx],
            "output_chunks_tpczyx": [int(value) for value in output_chunks_tpczyx],
            "voxel_size_um_zyx": [float(value) for value in voxel_size_um_zyx],
            "output_origin_xyz_um": [float(value) for value in output_origin_xyz],
            "data_component": analysis_cache_data_component("registration"),
        }
    )
    auxiliary_group.create_dataset(
        name="affines_tpx44",
        shape=(output_shape_tpczyx[0], int(root[source_component].shape[1]), 4, 4),
        dtype=np.float64,
        overwrite=True,
    )

    # Store the separable 1D blend-weight profiles so fusion workers can
    # reconstruct only the sub-volume they need on-the-fly.  The three
    # profiles are tiny (sum of axis lengths × 4 bytes) regardless of tile
    # size, avoiding the catastrophic allocation that a full 3D outer-product
    # volume would require for large tiles.
    prof_z, prof_y, prof_x = _blend_weight_profiles(
        source_tile_shape_zyx,
        blend_mode=str(blend_mode),
        overlap_zyx=overlap_zyx,
    )
    blend_group = auxiliary_group.create_group("blend_weights", overwrite=True)
    blend_group.create_dataset(name="profile_z", data=prof_z, dtype=np.float32,
                               overwrite=True)
    blend_group.create_dataset(name="profile_y", data=prof_y, dtype=np.float32,
                               overwrite=True)
    blend_group.create_dataset(name="profile_x", data=prof_x, dtype=np.float32,
                               overwrite=True)

    return (
        public_analysis_root("registration"),
        analysis_cache_data_component("registration"),
        f"{auxiliary_root}/affines_tpx44",
        f"{auxiliary_root}/blend_weights",
    )


def _estimate_worker_thread_capacity(client: "Client") -> int:
    """Estimate available worker thread capacity for throttled submission."""
    try:
        info = client.scheduler_info()
        workers = info.get("workers", {})
        thread_total = sum(
            max(1, int(details.get("nthreads", 1))) for details in workers.values()
        )
        return max(1, int(thread_total))
    except Exception:
        return 1


def run_registration_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> RegistrationSummary:
    """Run 3D tile registration for a canonical multiposition analysis store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr/N5 object.
    parameters : mapping[str, Any]
        Normalized registration parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    RegistrationSummary
        Summary metadata for the completed registration run.

    Raises
    ------
    ValueError
        If source components or required stage metadata are missing.
    """
    _emit(progress_callback, 1, "Loading store metadata and stage coordinates")
    root = zarr.open_group(str(zarr_path), mode="r")
    requested_source_component = (
        str(parameters.get("input_source", "data")).strip() or "data"
    )
    requested_resolution_level = max(
        0, int(parameters.get("input_resolution_level", 0))
    )
    source_component, pairwise_source_component, effective_level = (
        _resolve_source_components_for_level(
            root=root,
            requested_source_component=requested_source_component,
            input_resolution_level=requested_resolution_level,
        )
    )
    full_source = root[source_component]
    pairwise_source = root[pairwise_source_component]
    source_shape_tpczyx = tuple(int(value) for value in full_source.shape)
    pairwise_shape_tpczyx = tuple(int(value) for value in pairwise_source.shape)
    if len(source_shape_tpczyx) != 6 or len(pairwise_shape_tpczyx) != 6:
        raise ValueError(
            "registration requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )
    if source_shape_tpczyx[1] != pairwise_shape_tpczyx[1]:
        raise ValueError("registration source position count mismatch between levels.")

    positions = [int(index) for index in range(int(source_shape_tpczyx[1]))]
    channel_count = int(source_shape_tpczyx[2])
    registration_channel = max(0, int(parameters.get("registration_channel", 0)))
    if registration_channel >= channel_count:
        raise ValueError(
            f"registration_channel={registration_channel} is out of bounds for {channel_count} channels."
        )

    root_attrs = dict(root.attrs)
    try:
        store_metadata = load_store_metadata(root)
    except Exception:
        store_metadata = {}
    merged_metadata = dict(root_attrs)
    if isinstance(store_metadata, Mapping):
        merged_metadata.update(dict(store_metadata))

    spatial_calibration = spatial_calibration_from_dict(
        merged_metadata.get("spatial_calibration")
    )
    stage_rows = _load_stage_rows(merged_metadata)
    if len(positions) > 1 and len(stage_rows) < len(positions):
        raise ValueError(
            "registration requires multiposition stage metadata when more than one position is present."
        )
    if not stage_rows:
        stage_rows = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0, "f": 0.0} for _ in positions
        ]

    configured_anchor = parameters.get("anchor_position")
    if (
        str(parameters.get("anchor_mode", "central")).strip().lower() == "manual"
        and configured_anchor is not None
    ):
        anchor_position = int(configured_anchor)
    else:
        anchor_position = _position_centroid_anchor(
            stage_rows, spatial_calibration, positions
        )
    if anchor_position < 0 or anchor_position >= len(positions):
        raise ValueError("registration anchor_position is out of bounds.")

    full_voxel_size_um_zyx = _extract_voxel_size_um_zyx(root, source_component)
    level_factor_zyx = _pyramid_factor_zyx_for_level(
        root,
        level=effective_level,
        source_component=pairwise_source_component,
    )
    pairwise_voxel_size_um_zyx = (
        float(full_voxel_size_um_zyx[0]) * float(level_factor_zyx[0]),
        float(full_voxel_size_um_zyx[1]) * float(level_factor_zyx[1]),
        float(full_voxel_size_um_zyx[2]) * float(level_factor_zyx[2]),
    )

    _emit(progress_callback, 3, "Building nominal stage transforms")
    nominal_transforms_xyz = _build_nominal_transforms_xyz(
        stage_rows,
        spatial_calibration,
        anchor_position=anchor_position,
        positions=positions,
    )
    pairwise_tile_extent_xyz = _tile_extent_xyz(
        pairwise_shape_tpczyx[3:], pairwise_voxel_size_um_zyx
    )
    edge_specs = _build_edge_specs(
        nominal_transforms_xyz,
        positions=positions,
        tile_extent_xyz=pairwise_tile_extent_xyz,
        voxel_size_um_zyx=pairwise_voxel_size_um_zyx,
    )
    edge_count = len(edge_specs)
    _emit(progress_callback, 5, f"Prepared {edge_count} registration graph edges")

    correction_affines_tex44 = np.zeros(
        (int(source_shape_tpczyx[0]), int(edge_count), 4, 4), dtype=np.float64
    )
    edge_status_te = np.zeros(
        (int(source_shape_tpczyx[0]), int(edge_count)), dtype=np.uint8
    )
    edge_residual_te = np.full(
        (int(source_shape_tpczyx[0]), int(edge_count)), np.nan, dtype=np.float32
    )
    anchor_positions: list[int] = []
    effective_corrections_tpx44 = np.repeat(
        np.eye(4, dtype=np.float64)[np.newaxis, np.newaxis, :, :],
        int(source_shape_tpczyx[0]),
        axis=0,
    )
    effective_corrections_tpx44 = np.repeat(
        effective_corrections_tpx44,
        int(source_shape_tpczyx[1]),
        axis=1,
    )
    successful_edge_count = 0

    # Resolve configurable pairwise estimation parameters.
    effective_ants_iterations = tuple(
        int(v)
        for v in parameters.get("ants_iterations", _ANTS_AFF_ITERATIONS)
    )
    effective_ants_sampling_rate = float(
        parameters.get("ants_sampling_rate", _ANTS_RANDOM_SAMPLING_RATE)
    )
    effective_max_pairwise_voxels = int(
        parameters.get("max_pairwise_voxels", _DEFAULT_MAX_PAIRWISE_VOXELS)
    )
    effective_use_phase_correlation = bool(
        parameters.get("use_phase_correlation", False)
    )
    effective_use_fft_initial_alignment = bool(
        parameters.get("use_fft_initial_alignment", True)
    )

    for t_index in range(int(source_shape_tpczyx[0])):
        anchor_positions.append(int(anchor_position))
        if edge_count == 0:
            continue
        delayed_edges = [
            delayed(_register_pairwise_overlap)(
                zarr_path=str(zarr_path),
                source_component=pairwise_source_component,
                t_index=int(t_index),
                registration_channel=int(registration_channel),
                edge=edge,
                nominal_fixed_transform_xyz=nominal_transforms_xyz[
                    int(edge.fixed_position)
                ],
                nominal_moving_transform_xyz=nominal_transforms_xyz[
                    int(edge.moving_position)
                ],
                voxel_size_um_zyx=pairwise_voxel_size_um_zyx,
                overlap_zyx=[
                    int(value) for value in parameters.get("overlap_zyx", [8, 32, 32])
                ],
                registration_type=str(parameters.get("registration_type", "translation"))
                .strip()
                .lower(),
                max_pairwise_voxels=effective_max_pairwise_voxels,
                ants_iterations=effective_ants_iterations,
                ants_sampling_rate=effective_ants_sampling_rate,
                use_phase_correlation=effective_use_phase_correlation,
                use_fft_initial_alignment=effective_use_fft_initial_alignment,
            )
            for edge in edge_specs
        ]
        if client is None:
            pairwise_results = []
            total = max(1, len(delayed_edges))
            batch_size = max(1, min(total, 4))
            for batch_start in range(0, total, batch_size):
                batch = delayed_edges[batch_start : batch_start + batch_size]
                batch_results = list(
                    dask.compute(*batch, scheduler="processes")
                )
                pairwise_results.extend(batch_results)
                completed = len(pairwise_results)
                _emit(
                    progress_callback,
                    5 + int((completed / total) * 35),
                    f"Pairwise registration {completed}/{total} for t={t_index}",
                )
        else:
            from dask.distributed import as_completed

            pairwise_results = []
            futures = client.compute(delayed_edges)
            completed = 0
            total = max(1, len(futures))
            for future in as_completed(futures):
                pairwise_results.append(future.result())
                completed += 1
                _emit(
                    progress_callback,
                    5 + int((completed / total) * 35),
                    f"Pairwise registration {completed}/{total} for t={t_index}",
                )
        successful_edge_count += sum(
            1 for result in pairwise_results if bool(result.get("success", False))
        )

        _emit(
            progress_callback,
            41,
            f"Solving global optimization for t={t_index}",
        )
        solved, active_mask, residuals = _solve_with_pruning(
            positions=positions,
            edges=edge_specs,
            edge_results=pairwise_results,
            anchor_position=anchor_position,
            registration_type=str(parameters.get("registration_type", "translation"))
            .strip()
            .lower(),
            voxel_size_um_zyx=pairwise_voxel_size_um_zyx,
        )
        for position_index in positions:
            effective_corrections_tpx44[int(t_index), int(position_index)] = solved[
                int(position_index)
            ]
        for edge_index, result in enumerate(pairwise_results):
            correction_affines_tex44[int(t_index), int(edge_index)] = np.asarray(
                result["correction_matrix_xyz"], dtype=np.float64
            )
            edge_status_te[int(t_index), int(edge_index)] = (
                1 if bool(active_mask[int(edge_index)]) else 0
            )
            edge_residual_te[int(t_index), int(edge_index)] = float(
                residuals[int(edge_index)]
            )

    _emit(progress_callback, 43, "Computing output layout and bounding boxes")
    effective_transforms_tpx44 = np.zeros(
        (int(source_shape_tpczyx[0]), int(source_shape_tpczyx[1]), 4, 4),
        dtype=np.float64,
    )
    transformed_bboxes_tpx6 = np.zeros(
        (int(source_shape_tpczyx[0]), int(source_shape_tpczyx[1]), 6),
        dtype=np.float64,
    )
    full_tile_extent_xyz = _tile_extent_xyz(
        source_shape_tpczyx[3:], full_voxel_size_um_zyx
    )
    all_bbox_mins: list[np.ndarray] = []
    all_bbox_maxs: list[np.ndarray] = []
    for t_index in range(int(source_shape_tpczyx[0])):
        for position_index in positions:
            effective_transform = (
                effective_corrections_tpx44[int(t_index), int(position_index)]
                @ nominal_transforms_xyz[int(position_index)]
            )
            effective_transforms_tpx44[int(t_index), int(position_index)] = (
                effective_transform
            )
            bbox_min, bbox_max = _tile_bbox_xyz(
                effective_transform, full_tile_extent_xyz
            )
            transformed_bboxes_tpx6[int(t_index), int(position_index), :3] = bbox_min
            transformed_bboxes_tpx6[int(t_index), int(position_index), 3:] = bbox_max
            all_bbox_mins.append(bbox_min)
            all_bbox_maxs.append(bbox_max)

    if all_bbox_mins:
        output_min_xyz = np.min(np.vstack(all_bbox_mins), axis=0)
        output_max_xyz = np.max(np.vstack(all_bbox_maxs), axis=0)
    else:
        output_min_xyz = np.zeros(3, dtype=np.float64)
        output_max_xyz = np.asarray(full_tile_extent_xyz, dtype=np.float64)
    output_shape_xyz = np.maximum(
        1,
        np.ceil(
            (output_max_xyz - output_min_xyz)
            / np.asarray(
                [
                    float(full_voxel_size_um_zyx[2]),
                    float(full_voxel_size_um_zyx[1]),
                    float(full_voxel_size_um_zyx[0]),
                ],
                dtype=np.float64,
            )
        ).astype(int),
    )
    output_shape_tpczyx = (
        int(source_shape_tpczyx[0]),
        1,
        int(source_shape_tpczyx[2]),
        int(output_shape_xyz[2]),
        int(output_shape_xyz[1]),
        int(output_shape_xyz[0]),
    )
    source_chunks = tuple(int(value) for value in full_source.chunks)
    output_chunks_tpczyx = (
        1,
        1,
        1,
        max(1, min(int(source_chunks[3]), int(output_shape_tpczyx[3]))),
        max(1, min(int(source_chunks[4]), int(output_shape_tpczyx[4]))),
        max(1, min(int(source_chunks[5]), int(output_shape_tpczyx[5]))),
    )
    blend_mode = str(parameters.get("blend_mode", "feather")).strip().lower()
    overlap_zyx = [
        int(value) for value in parameters.get("overlap_zyx", [8, 32, 32])
    ]

    # --- Memory guard ---------------------------------------------------------
    source_tile_shape_zyx = tuple(int(v) for v in source_shape_tpczyx[3:])
    chunk_shape_zyx = output_chunks_tpczyx[3:]
    est_bytes = _estimate_fusion_chunk_bytes(
        chunk_shape_zyx,
        source_tile_shape_zyx,
        n_positions=int(source_shape_tpczyx[1]),
    )
    est_gib = est_bytes / (1024 ** 3)
    logger.info(
        "Fusion memory estimate: %.1f GiB per worker (chunk %s, tile %s)",
        est_gib,
        chunk_shape_zyx,
        source_tile_shape_zyx,
    )
    if est_gib > 64:
        logger.warning(
            "Estimated per-worker memory (%.1f GiB) exceeds 64 GiB. "
            "Consider reducing output chunk sizes or downsampling tiles "
            "before registration to avoid out-of-memory failures.",
            est_gib,
        )
    # --------------------------------------------------------------------------

    component, data_component, affines_component, blend_weights_component = (
        _prepare_output_group(
            zarr_path=zarr_path,
            source_component=source_component,
            parameters=parameters,
            output_shape_tpczyx=output_shape_tpczyx,
            output_chunks_tpczyx=output_chunks_tpczyx,
            voxel_size_um_zyx=full_voxel_size_um_zyx,
            output_origin_xyz=output_min_xyz,
            source_tile_shape_zyx=source_tile_shape_zyx,  # type: ignore[arg-type]
            blend_mode=blend_mode,
            overlap_zyx=overlap_zyx,
        )
    )
    write_root = zarr.open_group(str(zarr_path), mode="a")
    latest_group = write_root[analysis_auxiliary_root("registration")]
    latest_group.create_dataset(
        name="edges_pe2",
        data=(
            np.asarray(
                [
                    [int(edge.fixed_position), int(edge.moving_position)]
                    for edge in edge_specs
                ],
                dtype=np.int32,
            ).reshape(-1, 2)
            if edge_specs
            else np.zeros((0, 2), dtype=np.int32)
        ),
        overwrite=True,
    )
    latest_group.create_dataset(
        name="pairwise_affines_tex44",
        data=correction_affines_tex44,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="edge_status_te",
        data=edge_status_te,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="edge_residual_te",
        data=edge_residual_te,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="transformed_bboxes_tpx6",
        data=transformed_bboxes_tpx6,
        overwrite=True,
    )
    latest_group["affines_tpx44"][:] = effective_transforms_tpx44

    z_bounds = _axis_chunk_bounds(output_shape_tpczyx[3], output_chunks_tpczyx[3])
    y_bounds = _axis_chunk_bounds(output_shape_tpczyx[4], output_chunks_tpczyx[4])
    x_bounds = _axis_chunk_bounds(output_shape_tpczyx[5], output_chunks_tpczyx[5])
    fusion_tasks = [
        (
            int(t_index),
            int(c_index),
            z_chunk,
            y_chunk,
            x_chunk,
        )
        for t_index in range(int(output_shape_tpczyx[0]))
        for c_index in range(int(output_shape_tpczyx[2]))
        for z_chunk in z_bounds
        for y_chunk in y_bounds
        for x_chunk in x_bounds
    ]
    _emit(progress_callback, 45, f"Prepared {len(fusion_tasks)} fusion chunk tasks")

    task_kwargs = [
        dict(
            zarr_path=str(zarr_path),
            source_component=source_component,
            output_component=data_component,
            affines_component=affines_component,
            transformed_bboxes_component=(
                f"{analysis_auxiliary_root('registration')}/transformed_bboxes_tpx6"
            ),
            blend_weights_component=blend_weights_component,
            t_index=t_index,
            c_index=c_index,
            z_bounds=z_chunk,
            y_bounds=y_chunk,
            x_bounds=x_chunk,
            output_origin_xyz=output_min_xyz,
            voxel_size_um_zyx=full_voxel_size_um_zyx,
            output_dtype=str(full_source.dtype),
        )
        for t_index, c_index, z_chunk, y_chunk, x_chunk in fusion_tasks
    ]
    if client is None:
        total = max(1, len(task_kwargs))
        batch_size = max(1, min(total, 4))
        completed = 0
        for batch_start in range(0, total, batch_size):
            batch = [
                delayed(_process_and_write_registration_chunk)(**kwargs)
                for kwargs in task_kwargs[batch_start : batch_start + batch_size]
            ]
            dask.compute(*batch, scheduler="processes")
            completed += len(batch)
            _emit(
                progress_callback,
                45 + int((completed / total) * 50),
                f"Fusion chunk {completed}/{total}",
            )
    else:
        from dask.distributed import as_completed

        max_in_flight = max(16, int(_estimate_worker_thread_capacity(client)) * 2)
        completion_queue = as_completed()
        pending_count = 0
        completed = 0
        total = max(1, len(task_kwargs))
        for kwargs in task_kwargs:
            completion_queue.add(
                client.submit(
                    _process_and_write_registration_chunk,
                    **kwargs,
                    pure=False,
                )
            )
            pending_count += 1
            if pending_count < max_in_flight:
                continue
            finished = next(completion_queue)
            _ = finished.result()
            pending_count -= 1
            completed += 1
            _emit(
                progress_callback,
                45 + int((completed / total) * 50),
                f"Fusion chunk {completed}/{total}",
            )
        for finished in completion_queue:
            _ = finished.result()
            completed += 1
            _emit(
                progress_callback,
                45 + int((completed / total) * 50),
                f"Fusion chunk {completed}/{total}",
            )

    _emit(progress_callback, 96, "Writing registration metadata and provenance")
    active_edge_count = int(np.count_nonzero(edge_status_te))
    dropped_edge_count = max(0, int(successful_edge_count) - int(active_edge_count))
    latest_group.attrs.update(
        {
            "requested_source_component": str(requested_source_component),
            "source_component": str(source_component),
            "pairwise_source_component": str(pairwise_source_component),
            "requested_input_resolution_level": int(requested_resolution_level),
            "input_resolution_level": int(effective_level),
            "registration_channel": int(registration_channel),
            "registration_type": str(parameters.get("registration_type", "translation")),
            "anchor_positions": [int(value) for value in anchor_positions],
            "edge_count": int(edge_count),
            "active_edge_count": int(active_edge_count),
            "dropped_edge_count": int(dropped_edge_count),
            "blend_mode": blend_mode,
            "max_pairwise_voxels": int(effective_max_pairwise_voxels),
            "ants_iterations": [int(v) for v in effective_ants_iterations],
            "ants_sampling_rate": float(effective_ants_sampling_rate),
            "use_phase_correlation": bool(effective_use_phase_correlation),
            "use_fft_initial_alignment": bool(effective_use_fft_initial_alignment),
        }
    )

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="registration",
        component=component,
        metadata={
            "data_component": data_component,
            "affines_component": affines_component,
            "requested_source_component": requested_source_component,
            "source_component": source_component,
            "pairwise_source_component": pairwise_source_component,
            "requested_input_resolution_level": int(requested_resolution_level),
            "input_resolution_level": int(effective_level),
            "registration_channel": int(registration_channel),
            "registration_type": str(parameters.get("registration_type", "translation")),
            "anchor_positions": [int(value) for value in anchor_positions],
            "edge_count": int(edge_count),
            "active_edge_count": int(active_edge_count),
            "dropped_edge_count": int(dropped_edge_count),
            "blend_mode": blend_mode,
            "max_pairwise_voxels": int(effective_max_pairwise_voxels),
            "ants_iterations": [int(v) for v in effective_ants_iterations],
            "ants_sampling_rate": float(effective_ants_sampling_rate),
            "use_phase_correlation": bool(effective_use_phase_correlation),
            "use_fft_initial_alignment": bool(effective_use_fft_initial_alignment),
            "output_shape_tpczyx": [int(value) for value in output_shape_tpczyx],
            "output_chunks_tpczyx": [int(value) for value in output_chunks_tpczyx],
            "voxel_size_um_zyx": [float(value) for value in full_voxel_size_um_zyx],
            "output_origin_xyz_um": [float(value) for value in output_min_xyz],
        },
    )
    _emit(progress_callback, 100, "Registration complete")
    return RegistrationSummary(
        component=component,
        data_component=data_component,
        affines_component=affines_component,
        source_component=source_component,
        requested_source_component=requested_source_component,
        pairwise_source_component=pairwise_source_component,
        input_resolution_level=int(effective_level),
        requested_input_resolution_level=int(requested_resolution_level),
        registration_channel=int(registration_channel),
        registration_type=str(parameters.get("registration_type", "translation"))
        .strip()
        .lower(),
        anchor_positions=tuple(int(value) for value in anchor_positions),
        positions=int(source_shape_tpczyx[1]),
        timepoints=int(source_shape_tpczyx[0]),
        edge_count=int(edge_count),
        active_edge_count=int(active_edge_count),
        dropped_edge_count=int(dropped_edge_count),
        output_shape_tpczyx=tuple(int(value) for value in output_shape_tpczyx),
        output_chunks_tpczyx=tuple(int(value) for value in output_chunks_tpczyx),
        blend_mode=str(parameters.get("blend_mode", "feather")).strip().lower(),
    )
