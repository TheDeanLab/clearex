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

"""Napari visualization workflow for canonical ClearEx analysis stores."""

from __future__ import annotations

# Standard Library Imports
import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

# Third Party Imports
import dask.array as da
import numpy as np
import zarr

# Local Imports
from clearex.io.experiment import load_navigate_experiment
from clearex.io.provenance import register_latest_output_reference


ProgressCallback = Callable[[int, str], None]
_AXIS_LABELS_TCZYX = ("t", "c", "z", "y", "x")
_TPCZYX_TO_TCZYX = (0, 2, 3, 4, 5)
_MAX_SAFE_SINGLE_LEVEL_VOLUME_TEXTURE_BYTES = 2 * 1024 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizationSummary:
    """Summary metadata for one visualization run.

    Attributes
    ----------
    component : str
        Output component path inside the Zarr store.
    source_component : str
        Source image component rendered in napari.
    source_components : tuple[str, ...]
        Ordered source levels used for multiscale rendering.
    position_index : int
        Reference position index from the ``p`` axis. When
        ``show_all_positions`` is ``False`` this is the single selected
        position. When ``True`` this is the first rendered position.
    selected_positions : tuple[int, ...]
        Rendered position indices from the ``p`` axis.
    show_all_positions : bool
        Whether all positions were rendered in one napari scene.
    overlay_points_count : int
        Number of particle points overlaid in the viewer.
    launch_mode : str
        Effective launch mode: ``in_process`` or ``subprocess``.
    viewer_pid : int, optional
        Spawned viewer process ID when subprocess launch mode is used.
    """

    component: str
    source_component: str
    source_components: tuple[str, ...]
    position_index: int
    selected_positions: tuple[int, ...]
    show_all_positions: bool
    overlay_points_count: int
    launch_mode: str
    viewer_pid: Optional[int]


@dataclass(frozen=True)
class NapariLayerPayload:
    """Payload describing napari layer metadata and calibration.

    Attributes
    ----------
    axis_labels_tczyx : tuple[str, str, str, str, str]
        Display axis labels for rendered arrays in ``(t, c, z, y, x)`` order.
    scale_tczyx : tuple[float, float, float, float, float]
        Physical/index scaling factors for rendered arrays in
        ``(t, c, z, y, x)`` order.
    image_metadata : dict[str, Any]
        Metadata payload attached to the napari image layer.
    points_metadata : dict[str, Any]
        Metadata payload attached to the napari points layer.
    """

    axis_labels_tczyx: tuple[str, str, str, str, str]
    scale_tczyx: tuple[float, float, float, float, float]
    image_metadata: Dict[str, Any]
    points_metadata: Dict[str, Any]


def _sanitize_metadata_value(value: Any) -> Any:
    """Convert metadata values into napari-safe Python primitives.

    Parameters
    ----------
    value : Any
        Metadata value candidate.

    Returns
    -------
    Any
        Metadata value converted to primitive Python containers/scalars.

    Raises
    ------
    None
        Conversion is best-effort and never raises custom exceptions.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _sanitize_metadata_value(value.item())
    if isinstance(value, np.ndarray):
        return _sanitize_metadata_value(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_metadata_value(item) for item in value]
    return str(value)


def _coerce_numeric_sequence(value: Any) -> Optional[tuple[float, ...]]:
    """Coerce a metadata value into a finite numeric sequence.

    Parameters
    ----------
    value : Any
        Candidate sequence value.

    Returns
    -------
    tuple[float, ...], optional
        Parsed sequence when conversion succeeds; otherwise ``None``.

    Raises
    ------
    None
        Invalid values are handled internally and return ``None``.
    """
    normalized = _sanitize_metadata_value(value)
    if not isinstance(normalized, list):
        return None

    parsed: list[float] = []
    for item in normalized:
        if isinstance(item, list):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        parsed.append(float(number))
    return tuple(parsed)


def _normalize_scale_tczyx(
    values: Sequence[float],
) -> tuple[float, float, float, float, float]:
    """Normalize scale values into positive finite ``(t, c, z, y, x)`` factors.

    Parameters
    ----------
    values : sequence of float
        Candidate scale values in ``(t, c, z, y, x)`` order.

    Returns
    -------
    tuple[float, float, float, float, float]
        Scale tuple where invalid values are replaced with ``1.0``.

    Raises
    ------
    ValueError
        If ``values`` does not define exactly five entries.
    """
    if len(values) != 5:
        raise ValueError("Scale values must define exactly five entries.")
    normalized: list[float] = []
    for item in values:
        value = float(item)
        if not np.isfinite(value) or value <= 0:
            normalized.append(1.0)
        else:
            normalized.append(value)
    return (
        float(normalized[0]),
        float(normalized[1]),
        float(normalized[2]),
        float(normalized[3]),
        float(normalized[4]),
    )


def _first_positive_float(candidates: Sequence[Any]) -> Optional[float]:
    """Return first positive finite float from candidate values.

    Parameters
    ----------
    candidates : sequence of Any
        Candidate values to parse.

    Returns
    -------
    float, optional
        First parsed positive finite value, or ``None`` if unavailable.

    Raises
    ------
    None
        Invalid candidates are skipped and do not raise custom exceptions.
    """
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return float(value)
    return None


def _extract_scale_tczyx_from_attrs(
    *,
    root_attrs: Mapping[str, Any],
    source_attrs: Mapping[str, Any],
) -> Optional[tuple[float, float, float, float, float]]:
    """Extract image scale from known store/array attribute keys.

    Parameters
    ----------
    root_attrs : mapping[str, Any]
        Root Zarr attributes.
    source_attrs : mapping[str, Any]
        Source-array attributes.

    Returns
    -------
    tuple[float, float, float, float, float], optional
        Scale in ``(t, c, z, y, x)`` order when available.

    Raises
    ------
    None
        Missing/invalid keys are ignored.
    """
    candidate_keys = (
        "scale_tczyx",
        "scale_tpczyx",
        "physical_scale_tpczyx",
        "voxel_size_tpczyx",
        "pixel_size_tpczyx",
        "physical_pixel_size_tpczyx",
        "voxel_size_um_zyx",
        "voxel_size_zyx",
        "pixel_size_zyx",
        "physical_pixel_size_zyx",
        "scale",
    )

    for attrs in (source_attrs, root_attrs):
        for key in candidate_keys:
            if key not in attrs:
                continue
            parsed = _coerce_numeric_sequence(attrs.get(key))
            if parsed is None:
                continue
            if len(parsed) == 6:
                mapped = tuple(float(parsed[idx]) for idx in _TPCZYX_TO_TCZYX)
                return _normalize_scale_tczyx(mapped)
            if len(parsed) == 5:
                return _normalize_scale_tczyx(parsed)
            if len(parsed) == 3:
                return _normalize_scale_tczyx(
                    (
                        1.0,
                        1.0,
                        float(parsed[0]),
                        float(parsed[1]),
                        float(parsed[2]),
                    )
                )
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
        return parsed if np.isfinite(parsed) and parsed > 0 else None

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
    """Parse camera binning into ``(x, y)`` factors.

    Parameters
    ----------
    value : Any
        Raw binning descriptor (for example ``"2x2"``, ``[2, 2]``).

    Returns
    -------
    tuple[float, float]
        Positive ``(x, y)`` binning factors. Defaults to ``(1.0, 1.0)``.

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
        parsed = _first_positive_float((text,))
        if parsed is None:
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


def _load_source_experiment_raw(
    root_attrs: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Load source Navigate experiment raw metadata when available.

    Parameters
    ----------
    root_attrs : mapping[str, Any]
        Root Zarr attributes.

    Returns
    -------
    dict[str, Any], optional
        Parsed ``experiment.yml`` payload from ``source_experiment``.

    Raises
    ------
    None
        Failures are handled internally and return ``None``.
    """
    source_experiment = root_attrs.get("source_experiment")
    if not isinstance(source_experiment, str):
        return None
    text = source_experiment.strip()
    if not text:
        return None

    experiment_path = Path(text).expanduser()
    if not experiment_path.exists():
        return None

    try:
        experiment = load_navigate_experiment(experiment_path)
    except Exception:
        return None

    if not isinstance(experiment.raw, dict):
        return None
    return dict(experiment.raw)


def _extract_scale_tczyx_from_navigate_raw(
    navigate_raw: Optional[Mapping[str, Any]],
) -> Optional[tuple[float, float, float, float, float]]:
    """Extract ``(t, c, z, y, x)`` scale from Navigate experiment metadata.

    Parameters
    ----------
    navigate_raw : mapping[str, Any], optional
        Raw parsed experiment metadata.

    Returns
    -------
    tuple[float, float, float, float, float], optional
        Scale tuple when at least one physical spacing field is available.

    Raises
    ------
    None
        Invalid structures are handled internally and return ``None``.
    """
    if not isinstance(navigate_raw, Mapping):
        return None

    state = navigate_raw.get("MicroscopeState")
    state_mapping = state if isinstance(state, Mapping) else {}
    camera = navigate_raw.get("CameraParameters")
    camera_mapping = camera if isinstance(camera, Mapping) else {}

    microscope_name_raw = state_mapping.get("microscope_name")
    microscope_name = str(microscope_name_raw).strip() if microscope_name_raw else ""
    microscope_camera = camera_mapping.get(microscope_name)
    microscope_camera_mapping = (
        microscope_camera if isinstance(microscope_camera, Mapping) else {}
    )

    profile_mapping: Mapping[str, Any]
    if microscope_camera_mapping:
        profile_mapping = microscope_camera_mapping
    else:
        profile_mapping = camera_mapping

    fov_x = _first_positive_float((profile_mapping.get("fov_x"),))
    fov_y = _first_positive_float((profile_mapping.get("fov_y"),))
    img_x = _first_positive_float(
        (profile_mapping.get("img_x_pixels"), profile_mapping.get("x_pixels"))
    )
    img_y = _first_positive_float(
        (profile_mapping.get("img_y_pixels"), profile_mapping.get("y_pixels"))
    )
    lateral_from_fov = _first_positive_float(
        (
            (fov_x / img_x) if fov_x is not None and img_x is not None and img_x > 0 else None,
            (fov_y / img_y) if fov_y is not None and img_y is not None and img_y > 0 else None,
        )
    )

    pixel_size = _first_positive_float(
        (
            profile_mapping.get("pixel_size"),
            camera_mapping.get("pixel_size"),
        )
    )
    zoom = _parse_zoom_factor(state_mapping.get("zoom"))
    binning_x, binning_y = _parse_binning_xy(
        profile_mapping.get("binning") or camera_mapping.get("binning")
    )
    lateral_from_pixel_zoom = None
    if pixel_size is not None and zoom is not None:
        lateral_from_pixel_zoom = float(pixel_size / zoom * ((binning_x + binning_y) / 2.0))

    lateral_size = _first_positive_float(
        (
            lateral_from_fov,
            lateral_from_pixel_zoom,
            pixel_size,
        )
    )
    axial_size = _first_positive_float((state_mapping.get("step_size"),))
    temporal_size = _first_positive_float(
        (
            state_mapping.get("timepoint_interval"),
            state_mapping.get("stack_pause"),
        )
    )

    if lateral_size is None and axial_size is None and temporal_size is None:
        return None

    return _normalize_scale_tczyx(
        (
            float(temporal_size) if temporal_size is not None else 1.0,
            1.0,
            float(axial_size) if axial_size is not None else 1.0,
            float(lateral_size) if lateral_size is not None else 1.0,
            float(lateral_size) if lateral_size is not None else 1.0,
        )
    )


def _collect_multiscale_level_metadata(
    *,
    root: zarr.hierarchy.Group,
    source_components: Sequence[str],
) -> list[Dict[str, Any]]:
    """Collect shape/chunk metadata for each rendered multiscale level.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened Zarr root group.
    source_components : sequence of str
        Ordered source component paths.

    Returns
    -------
    list[dict[str, Any]]
        Per-level metadata dictionaries.

    Raises
    ------
    KeyError
        If a referenced component is missing.
    """
    levels: list[Dict[str, Any]] = []
    for component in source_components:
        array = root[str(component)]
        chunks = (
            [int(chunk) for chunk in array.chunks]
            if getattr(array, "chunks", None) is not None
            else None
        )
        levels.append(
            {
                "component": str(component),
                "shape_tpczyx": [int(size) for size in tuple(array.shape)],
                "chunks_tpczyx": chunks,
                "dtype": str(array.dtype),
            }
        )
    return levels


def _build_napari_layer_payload(
    *,
    zarr_path: Union[str, Path],
    root: zarr.hierarchy.Group,
    source_component: str,
    source_components: Sequence[str],
    position_index: int,
    parameters: Mapping[str, Any],
    overlay_points_count: int,
    point_property_names: Sequence[str],
) -> NapariLayerPayload:
    """Build image/points metadata payloads for napari layers.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    root : zarr.hierarchy.Group
        Opened Zarr root group.
    source_component : str
        Base source component path.
    source_components : sequence of str
        Ordered multiscale source components.
    position_index : int
        Selected position index.
    parameters : mapping[str, Any]
        Effective visualization parameters.
    overlay_points_count : int
        Number of overlay points.
    point_property_names : sequence of str
        Overlay point property names.

    Returns
    -------
    NapariLayerPayload
        Prepared payload for napari rendering.

    Raises
    ------
    KeyError
        If referenced source components are missing.
    """
    root_attrs = dict(root.attrs)
    source_array = root[str(source_component)]
    source_attrs = dict(source_array.attrs)
    source_experiment_raw = _load_source_experiment_raw(root_attrs)
    scale_tczyx = (
        _extract_scale_tczyx_from_attrs(
            root_attrs=root_attrs,
            source_attrs=source_attrs,
        )
        or _extract_scale_tczyx_from_navigate_raw(source_experiment_raw)
        or (1.0, 1.0, 1.0, 1.0, 1.0)
    )
    multiscale_levels = _collect_multiscale_level_metadata(
        root=root,
        source_components=source_components,
    )

    image_metadata_raw: Dict[str, Any] = {
        "schema": root_attrs.get("schema"),
        "store_path": str(Path(zarr_path).expanduser().resolve()),
        "axis_labels_tczyx": list(_AXIS_LABELS_TCZYX),
        "scale_tczyx": [float(value) for value in scale_tczyx],
        "source_component": str(source_component),
        "source_components": [str(item) for item in source_components],
        "position_index": int(position_index),
        "source_data_path": root_attrs.get("source_data_path"),
        "source_data_component": root_attrs.get("source_data_component"),
        "source_data_axes": root_attrs.get("source_data_axes"),
        "source_experiment": root_attrs.get("source_experiment"),
        "navigate_experiment": root_attrs.get("navigate_experiment"),
        "source_experiment_metadata": source_experiment_raw,
        "configured_chunks_tpczyx": root_attrs.get("configured_chunks_tpczyx"),
        "chunk_shape_tpczyx": (
            source_attrs.get("chunk_shape_tpczyx")
            or root_attrs.get("chunk_shape_tpczyx")
        ),
        "resolution_pyramid_factors_tpczyx": (
            source_attrs.get("resolution_pyramid_factors_tpczyx")
            or root_attrs.get("resolution_pyramid_factors_tpczyx")
        ),
        "data_pyramid_levels": root_attrs.get("data_pyramid_levels"),
        "data_pyramid_factors_tpczyx": root_attrs.get("data_pyramid_factors_tpczyx"),
        "multiscale_levels": multiscale_levels,
        "source_array_attrs": source_attrs,
        "root_attrs": root_attrs,
        "visualization_parameters": dict(parameters),
    }
    points_metadata_raw: Dict[str, Any] = {
        "coordinate_axes_tczyx": list(_AXIS_LABELS_TCZYX),
        "scale_tczyx": [float(value) for value in scale_tczyx],
        "source_component": str(source_component),
        "position_index": int(position_index),
        "overlay_points_count": int(overlay_points_count),
        "point_properties": [str(name) for name in point_property_names],
        "detection_component": str(
            parameters.get(
                "particle_detection_component",
                "results/particle_detection/latest/detections",
            )
        ),
        "source_data_path": root_attrs.get("source_data_path"),
    }

    image_metadata = _sanitize_metadata_value(image_metadata_raw)
    points_metadata = _sanitize_metadata_value(points_metadata_raw)
    if not isinstance(image_metadata, dict):
        image_metadata = {"metadata": image_metadata}
    if not isinstance(points_metadata, dict):
        points_metadata = {"metadata": points_metadata}

    return NapariLayerPayload(
        axis_labels_tczyx=_AXIS_LABELS_TCZYX,
        scale_tczyx=scale_tczyx,
        image_metadata={str(key): value for key, value in image_metadata.items()},
        points_metadata={str(key): value for key, value in points_metadata.items()},
    )


def _normalize_visualization_parameters(
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize visualization runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized visualization parameters.

    Raises
    ------
    ValueError
        If ``launch_mode`` is unsupported.
    """
    normalized = dict(parameters)
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["show_all_positions"] = bool(
        normalized.get("show_all_positions", False)
    )
    normalized["position_index"] = max(0, int(normalized.get("position_index", 0)))
    normalized["use_multiscale"] = bool(normalized.get("use_multiscale", True))
    normalized["overlay_particle_detections"] = bool(
        normalized.get("overlay_particle_detections", True)
    )
    normalized["particle_detection_component"] = (
        str(
            normalized.get(
                "particle_detection_component",
                "results/particle_detection/latest/detections",
            )
        ).strip()
        or "results/particle_detection/latest/detections"
    )

    launch_mode = str(normalized.get("launch_mode", "auto")).strip().lower() or "auto"
    if launch_mode not in {"auto", "in_process", "subprocess"}:
        raise ValueError(
            "visualization launch_mode must be one of auto, in_process, subprocess."
        )
    normalized["launch_mode"] = launch_mode
    return normalized


def _estimate_single_channel_texture_bytes_tczyx(array: Any) -> int:
    """Estimate per-channel texture allocation size for a ``(t, c, z, y, x)`` array.

    Parameters
    ----------
    array : Any
        Array-like value exposing ``shape`` and ``dtype``.

    Returns
    -------
    int
        Estimated bytes required for one channel texture. Returns ``0`` when
        estimation is not possible.

    Raises
    ------
    None
        Invalid shapes/dtypes are handled internally and return ``0``.
    """
    shape = tuple(int(value) for value in getattr(array, "shape", tuple()))
    if len(shape) != 5:
        return 0
    t_size, _channel_size, z_size, y_size, x_size = shape
    if min(t_size, z_size, y_size, x_size) <= 0:
        return 0
    try:
        dtype = np.dtype(getattr(array, "dtype", np.float32))
        itemsize = int(dtype.itemsize)
    except Exception:
        itemsize = 4
    voxel_count = int(t_size) * int(z_size) * int(y_size) * int(x_size)
    if voxel_count <= 0:
        return 0
    return int(voxel_count) * int(max(1, itemsize))


def _resolve_initial_viewer_ndisplay(
    *,
    first_level_data_tczyx: Any,
    source_components: Sequence[str],
) -> tuple[int, Optional[str]]:
    """Resolve initial napari ``ndisplay`` mode for image rendering.

    Parameters
    ----------
    first_level_data_tczyx : Any
        First source level data for one position in ``(t, c, z, y, x)`` order.
    source_components : sequence of str
        Source component paths used for rendering.

    Returns
    -------
    tuple[int, str or None]
        ``(ndisplay, reason)``. ``reason`` is populated when 2D fallback is
        selected to avoid GPU texture allocation failures.

    Raises
    ------
    None
        Unsupported input values are handled internally via fallback logic.

    Notes
    -----
    Single-level (non-pyramid) sources can trigger OpenGL out-of-memory errors
    when napari attempts 3D texture allocation for very large volumes.
    """
    if len(tuple(source_components)) > 1:
        return 3, None

    estimated_bytes = _estimate_single_channel_texture_bytes_tczyx(
        first_level_data_tczyx
    )
    if estimated_bytes <= _MAX_SAFE_SINGLE_LEVEL_VOLUME_TEXTURE_BYTES:
        return 3, None

    gib = float(estimated_bytes) / float(1024**3)
    threshold_gib = float(_MAX_SAFE_SINGLE_LEVEL_VOLUME_TEXTURE_BYTES) / float(1024**3)
    reason = (
        "Large single-level volume detected "
        f"(estimated {gib:.2f} GiB per channel exceeds {threshold_gib:.2f} GiB "
        "safety threshold). Using 2D slicing view."
    )
    return 2, reason


def _resolve_effective_launch_mode(requested_mode: str) -> str:
    """Resolve effective viewer launch mode for current runtime context.

    Parameters
    ----------
    requested_mode : str
        Requested launch mode (``auto``, ``in_process``, or ``subprocess``).

    Returns
    -------
    str
        Effective launch mode (``in_process`` or ``subprocess``).
    """
    mode = str(requested_mode).strip().lower() or "auto"
    if mode == "auto":
        if threading.current_thread() is threading.main_thread():
            return "in_process"
        return "subprocess"
    return mode


def _resolve_multiscale_components(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
    use_multiscale: bool,
) -> tuple[str, ...]:
    """Resolve ordered multiscale component paths for a source image.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened Zarr root group.
    source_component : str
        Source component path.
    use_multiscale : bool
        Whether multiscale loading is enabled.

    Returns
    -------
    tuple[str, ...]
        Ordered component paths, always including ``source_component``.
    """
    try:
        source_array = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"Visualization input component '{source_component}' was not found."
        ) from exc

    if len(tuple(source_array.shape)) != 6:
        raise ValueError(
            "Visualization requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    if not use_multiscale:
        return (source_component,)

    candidates: list[str] = []
    source_levels = source_array.attrs.get("pyramid_levels")
    if isinstance(source_levels, (tuple, list)):
        candidates.extend(str(item) for item in source_levels)

    if source_component == "data":
        root_levels = root.attrs.get("data_pyramid_levels")
        if isinstance(root_levels, (tuple, list)):
            candidates.extend(str(item) for item in root_levels)

    ordered: list[str] = [source_component]
    ordered.extend(candidates)
    unique: list[str] = []
    for component in ordered:
        text = str(component).strip()
        if not text or text in unique:
            continue
        try:
            array = root[text]
        except Exception:
            continue
        if len(tuple(array.shape)) != 6:
            continue
        unique.append(text)
    return tuple(unique) if unique else (source_component,)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    """Parse a finite float value with fallback.

    Parameters
    ----------
    value : Any
        Candidate value to parse.
    default : float, default=0.0
        Fallback value used when parsing fails.

    Returns
    -------
    float
        Parsed finite float value, or ``default`` when unavailable.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _looks_like_multiposition_header(row: Any) -> bool:
    """Return whether a row resembles a Navigate multiposition header.

    Parameters
    ----------
    row : Any
        Candidate row value.

    Returns
    -------
    bool
        ``True`` when row contains at least ``X``, ``Y``, and ``Z`` labels.
    """
    if not isinstance(row, (list, tuple)) or not row:
        return False
    labels = {str(value).strip().upper() for value in row}
    return {"X", "Y", "Z"}.issubset(labels)


def _parse_multiposition_stage_rows(payload: Any) -> list[dict[str, float]]:
    """Parse stage coordinates from a ``multi_positions.yml`` payload.

    Parameters
    ----------
    payload : Any
        Parsed sidecar payload. Expected shape is a list of rows.

    Returns
    -------
    list[dict[str, float]]
        Parsed rows with ``x``, ``y``, ``z``, and ``theta`` values.
    """
    if not isinstance(payload, list):
        return []

    rows = list(payload)
    header_index: dict[str, int] = {}
    if rows and _looks_like_multiposition_header(rows[0]):
        header_row = rows.pop(0)
        if isinstance(header_row, (list, tuple)):
            for idx, value in enumerate(header_row):
                header_index[str(value).strip().upper()] = int(idx)

    parsed_rows: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue

        def _value(field: str, fallback_index: int) -> float:
            index = header_index.get(field, fallback_index)
            if index < 0 or index >= len(row):
                return 0.0
            return _safe_float(row[index], default=0.0)

        parsed_rows.append(
            {
                "x": _value("X", 0),
                "y": _value("Y", 1),
                "z": _value("Z", 2),
                "theta": _value("THETA", 3),
            }
        )
    return parsed_rows


def _load_multiposition_stage_rows(
    root_attrs: Mapping[str, Any],
) -> list[dict[str, float]]:
    """Load multiposition stage rows from sidecar metadata when available.

    Parameters
    ----------
    root_attrs : mapping[str, Any]
        Root Zarr attributes.

    Returns
    -------
    list[dict[str, float]]
        Parsed stage rows from ``multi_positions.yml`` or fallback metadata.
        Returns an empty list when stage metadata cannot be resolved.
    """
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


def _build_position_affine_tczyx(
    *,
    delta_x: float,
    delta_y: float,
    delta_z: float,
    delta_theta_deg: float,
    scale_tczyx: Sequence[float],
) -> np.ndarray:
    """Build a 5D napari affine matrix for one position offset.

    Parameters
    ----------
    delta_x : float
        Stage X translation delta relative to reference position.
    delta_y : float
        Stage Y translation delta relative to reference position.
    delta_z : float
        Stage Z translation delta relative to reference position.
    delta_theta_deg : float
        Stage rotation delta (degrees) around sample X axis.
    scale_tczyx : sequence of float
        Effective napari layer scale in ``(t, c, z, y, x)`` order.

    Returns
    -------
    numpy.ndarray
        ``(6, 6)`` homogeneous affine matrix for ``(t, c, z, y, x)`` data.
    """
    affine = np.eye(6, dtype=np.float64)
    theta_rad = math.radians(float(delta_theta_deg))
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # Rotate sample around X by rotating the Z/Y plane in TCZYX coordinates.
    affine[2, 2] = cos_theta
    affine[2, 3] = sin_theta
    affine[3, 2] = -sin_theta
    affine[3, 3] = cos_theta

    del scale_tczyx

    # Stage coordinates are reported in microns. Napari affine translation is
    # interpreted in world units, so pass micron offsets directly.
    affine[2, 5] = float(delta_z)
    affine[3, 5] = float(delta_y)
    affine[4, 5] = float(delta_x)
    return affine


def _resolve_position_affines_tczyx(
    *,
    root_attrs: Mapping[str, Any],
    selected_positions: Sequence[int],
    scale_tczyx: Sequence[float],
) -> tuple[dict[int, np.ndarray], list[dict[str, float]]]:
    """Resolve per-position affines for napari rendering.

    Parameters
    ----------
    root_attrs : mapping[str, Any]
        Root Zarr attributes.
    selected_positions : sequence of int
        Position indices selected for visualization.
    scale_tczyx : sequence of float
        Effective napari layer scale in ``(t, c, z, y, x)`` order.

    Returns
    -------
    tuple[dict[int, numpy.ndarray], list[dict[str, float]]]
        Mapping of position index to affine matrix and parsed stage rows.
    """
    affines: dict[int, np.ndarray] = {
        int(index): np.eye(6, dtype=np.float64) for index in selected_positions
    }
    stage_rows = _load_multiposition_stage_rows(root_attrs)
    if not stage_rows:
        return affines, []

    reference = stage_rows[0]
    for position_index in selected_positions:
        idx = int(position_index)
        if idx < 0 or idx >= len(stage_rows):
            continue
        row = stage_rows[idx]
        affines[idx] = _build_position_affine_tczyx(
            delta_x=float(row["x"] - reference["x"]),
            delta_y=float(row["y"] - reference["y"]),
            delta_z=float(row["z"] - reference["z"]),
            delta_theta_deg=float(row["theta"] - reference["theta"]),
            scale_tczyx=scale_tczyx,
        )
    return affines, stage_rows


def _load_particle_overlay_points(
    *,
    root: zarr.hierarchy.Group,
    detection_component: str,
    position_index: int,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load particle detections as napari points for one position index.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened Zarr root group.
    detection_component : str
        Detection-table component path.
    position_index : int
        Requested position index.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, numpy.ndarray]]
        Napari points in ``(t, c, z, y, x)`` order and associated properties.

    Notes
    -----
    Supported table layouts:

    - ``(N, >=6)`` interpreted as
      ``(t, p, c, z, y, x, sigma?, intensity?, ...)``.
    - ``(N, 5)`` interpreted as ``(t, c, z, y, x)``.
    - ``(N, 4)`` interpreted as ``(t, z, y, x)`` with channel fixed at 0.
    """
    try:
        detection_array = root[detection_component]
    except Exception:
        return np.empty((0, 5), dtype=np.float32), {}

    table = np.asarray(detection_array, dtype=np.float32)
    if table.ndim != 2 or table.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32), {}

    columns = int(table.shape[1])
    if columns >= 6:
        mask = np.isclose(table[:, 1], float(position_index))
        filtered = table[mask]
        if filtered.shape[0] == 0:
            return np.empty((0, 5), dtype=np.float32), {}
        points = filtered[:, [0, 2, 3, 4, 5]].astype(np.float32, copy=False)
        properties: Dict[str, np.ndarray] = {}
        if columns > 6:
            properties["sigma"] = filtered[:, 6].astype(np.float32, copy=False)
        if columns > 7:
            properties["intensity"] = filtered[:, 7].astype(np.float32, copy=False)
        return points, properties

    if columns == 5:
        return table.astype(np.float32, copy=False), {}

    if columns == 4:
        channel = np.zeros((table.shape[0], 1), dtype=np.float32)
        points = np.concatenate((table[:, [0]], channel, table[:, 1:4]), axis=1)
        return points.astype(np.float32, copy=False), {}

    return np.empty((0, 5), dtype=np.float32), {}


def _launch_napari_subprocess(
    *,
    zarr_path: Union[str, Path],
    normalized_parameters: Mapping[str, Any],
) -> int:
    """Launch napari visualization in a subprocess.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    normalized_parameters : mapping[str, Any]
        Normalized visualization parameters.

    Returns
    -------
    int
        Spawned process ID.
    """
    payload = json.dumps(dict(normalized_parameters), separators=(",", ":"))
    command = [
        sys.executable,
        "-m",
        "clearex.visualization.pipeline",
        "--zarr-path",
        str(zarr_path),
        "--parameters-json",
        payload,
    ]
    process = subprocess.Popen(command)
    return int(process.pid)


def _launch_napari_viewer(
    *,
    zarr_path: Union[str, Path],
    source_components: Sequence[str],
    source_component: str,
    selected_positions: Sequence[int],
    points_by_position: Mapping[int, np.ndarray],
    point_properties_by_position: Mapping[int, Mapping[str, np.ndarray]],
    position_affines_tczyx: Mapping[int, np.ndarray],
    axis_labels: Sequence[str],
    scale_tczyx: Sequence[float],
    image_metadata: Mapping[str, Any],
    points_metadata: Mapping[str, Any],
) -> None:
    """Open a napari viewer and render image + optional particle overlays.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    source_components : sequence of str
        Ordered source component paths for multiscale rendering.
    source_component : str
        Base source component path used for naming.
    selected_positions : sequence of int
        Position indices selected from the ``p`` axis.
    points_by_position : mapping[int, numpy.ndarray]
        Overlay points keyed by position index in ``(t, c, z, y, x)`` order.
    point_properties_by_position : mapping[int, mapping[str, numpy.ndarray]]
        Optional point-property mappings keyed by position index.
    position_affines_tczyx : mapping[int, numpy.ndarray]
        Per-position affine matrices in homogeneous ``(6, 6)`` form.
    axis_labels : sequence of str
        Axis labels in ``(t, c, z, y, x)`` order.
    scale_tczyx : sequence of float
        Layer scale values in ``(t, c, z, y, x)`` order.
    image_metadata : mapping[str, Any]
        Metadata payload attached to the image layer.
    points_metadata : mapping[str, Any]
        Metadata payload attached to the points layer.

    Returns
    -------
    None
        Viewer/event-loop side effects only.

    Raises
    ------
    ImportError
        If napari is unavailable.
    """
    import napari

    def _channel_colormap_cycle(channel_count: int) -> tuple[str, ...]:
        """Return deterministic channel colormaps for visualization layers.

        Parameters
        ----------
        channel_count : int
            Number of channels to color.

        Returns
        -------
        tuple[str, ...]
            Colormap names, one per channel.
        """
        if channel_count <= 0:
            return tuple()
        base = ("green", "magenta", "bop orange")
        extras = (
            "cyan",
            "yellow",
            "blue",
            "red",
            "gray",
            "turquoise",
            "hotpink",
        )
        palette: list[str] = []
        for index in range(channel_count):
            if index < len(base):
                palette.append(base[index])
            else:
                extra_index = (index - len(base)) % len(extras)
                palette.append(extras[extra_index])
        return tuple(palette)

    def _default_channel_opacity(channel_count: int) -> float:
        """Return default per-layer opacity based on channel count.

        Parameters
        ----------
        channel_count : int
            Number of channels being rendered.

        Returns
        -------
        float
            Layer opacity in ``[0.55, 1.0]``.
        """
        if channel_count <= 1:
            return 1.0
        return float(max(0.55, 1.0 - (0.1 * (channel_count - 1))))

    def _percentile_contrast_limits(
        data: Any,
        *,
        low_percentile: float = 1.0,
        high_percentile: float = 95.0,
        target_sample_size: int = 500_000,
    ) -> tuple[float, float]:
        """Estimate robust contrast limits from data percentiles.

        Parameters
        ----------
        data : Any
            Image-like array.
        low_percentile : float, default=1.0
            Lower percentile.
        high_percentile : float, default=95.0
            Upper percentile.
        target_sample_size : int, default=500_000
            Approximate number of sampled voxels used for percentile estimation.

        Returns
        -------
        tuple[float, float]
            ``(min, max)`` contrast limits.
        """
        if isinstance(data, da.Array):
            array = data
        else:
            array = np.asarray(data)

        shape = tuple(int(value) for value in getattr(array, "shape", tuple()))
        if not shape or int(np.prod(shape, dtype=np.int64)) <= 0:
            return 0.0, 1.0

        sampled = array
        total_size = int(np.prod(shape, dtype=np.int64))
        if total_size > int(target_sample_size):
            stride = int(
                max(
                    1,
                    math.ceil((total_size / float(target_sample_size)) ** (1.0 / len(shape))),
                )
            )
            sampled = sampled[tuple(slice(None, None, stride) for _ in shape)]

        low_value: float
        high_value: float
        try:
            if isinstance(sampled, da.Array):
                percentiles = da.percentile(
                    sampled.reshape(-1),
                    [float(low_percentile), float(high_percentile)],
                ).compute()
            else:
                percentiles = np.percentile(
                    np.asarray(sampled).reshape(-1),
                    [float(low_percentile), float(high_percentile)],
                )
            low_value = float(percentiles[0])
            high_value = float(percentiles[1])
        except Exception:
            try:
                if isinstance(sampled, da.Array):
                    minimum, maximum = da.compute(sampled.min(), sampled.max())
                else:
                    minimum = np.min(np.asarray(sampled))
                    maximum = np.max(np.asarray(sampled))
                low_value = float(minimum)
                high_value = float(maximum)
            except Exception:
                return 0.0, 1.0

        if not np.isfinite(low_value):
            low_value = 0.0
        if not np.isfinite(high_value):
            high_value = low_value + 1.0
        if high_value <= low_value:
            high_value = low_value + 1.0
        return float(low_value), float(high_value)

    scale = tuple(float(value) for value in scale_tczyx)
    selected = tuple(int(index) for index in selected_positions)
    first_position_index = int(selected[0]) if selected else 0
    first_level_data_tczyx = da.from_zarr(
        str(zarr_path), component=str(source_components[0])
    )[:, first_position_index, :, :, :, :]
    ndisplay, ndisplay_reason = _resolve_initial_viewer_ndisplay(
        first_level_data_tczyx=first_level_data_tczyx,
        source_components=source_components,
    )
    if ndisplay_reason:
        LOGGER.warning(ndisplay_reason)

    viewer = napari.Viewer(ndisplay=int(ndisplay), show=True)
    axis_labels_tuple = tuple(str(label) for label in axis_labels)
    axis_labels_applied = False
    volume_rendering_enabled = int(ndisplay) == 3

    multi_position = len(selected) > 1
    for position_index in selected:
        level_arrays = [
            da.from_zarr(str(zarr_path), component=component)[
                :, position_index, :, :, :, :
            ]
            for component in source_components
        ]
        image_data: Any
        is_multiscale = len(level_arrays) > 1
        channel_count = int(level_arrays[0].shape[1]) if level_arrays else 1
        channel_count = max(1, channel_count)
        colormaps = _channel_colormap_cycle(channel_count)
        channel_opacity = _default_channel_opacity(channel_count)

        affine_base = np.asarray(
            position_affines_tczyx.get(position_index, np.eye(6, dtype=np.float64)),
            dtype=np.float64,
        )
        for channel_index in range(channel_count):
            if is_multiscale:
                image_data = [
                    level[:, channel_index : channel_index + 1, :, :, :]
                    for level in level_arrays
                ]
            else:
                image_data = level_arrays[0][
                    :, channel_index : channel_index + 1, :, :, :
                ]

            channel_affine = np.asarray(affine_base, dtype=np.float64).copy()

            contrast_limits = _percentile_contrast_limits(
                image_data[-1] if is_multiscale else image_data
            )
            image_layer_metadata = dict(image_metadata)
            image_layer_metadata["position_index"] = int(position_index)
            image_layer_metadata["channel_index"] = int(channel_index)
            image_layer_metadata["channel_colormap"] = str(colormaps[channel_index])
            image_layer_metadata["contrast_limits"] = [
                float(contrast_limits[0]),
                float(contrast_limits[1]),
            ]
            layer_name = (
                f"{source_component} (p={position_index}, c={channel_index})"
                if channel_count > 1
                else f"{source_component} (p={position_index})"
            )

            image_kwargs: Dict[str, Any] = {
                "multiscale": is_multiscale,
                "name": layer_name,
                "scale": scale,
                "affine": channel_affine,
                "metadata": {
                    str(key): value for key, value in image_layer_metadata.items()
                },
                "colormap": str(colormaps[channel_index]),
                "opacity": float(channel_opacity),
                "blending": "additive",
                "contrast_limits": (
                    float(contrast_limits[0]),
                    float(contrast_limits[1]),
                ),
            }
            if volume_rendering_enabled:
                image_kwargs["rendering"] = "attenuated_mip"
            viewer.add_image(image_data, **image_kwargs)
            if not axis_labels_applied:
                try:
                    if int(viewer.dims.ndim) == len(axis_labels_tuple):
                        viewer.dims.axis_labels = axis_labels_tuple
                        axis_labels_applied = True
                except Exception:
                    pass

        points = np.asarray(
            points_by_position.get(
                position_index,
                np.empty((0, 5), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if points.shape[0] <= 0:
            continue

        point_properties = point_properties_by_position.get(position_index, {})
        points_layer_metadata = dict(points_metadata)
        points_layer_metadata["position_index"] = int(position_index)
        layer_name = (
            f"Particle Detections (p={position_index})"
            if multi_position
            else "Particle Detections"
        )
        viewer.add_points(
            points.astype(np.float32, copy=False),
            name=layer_name,
            properties={str(k): np.asarray(v) for k, v in point_properties.items()},
            scale=scale,
            affine=affine_base,
            metadata={str(key): value for key, value in points_layer_metadata.items()},
            size=7.0,
            opacity=0.5,
            face_color="transparent",
            border_color="white",
            border_width=0.2,
            blending="translucent",
        )
    napari.run()


def _save_visualization_metadata(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    source_components: Sequence[str],
    position_index: int,
    selected_positions: Sequence[int],
    show_all_positions: bool,
    parameters: Mapping[str, Any],
    overlay_points_count: int,
    launch_mode: str,
    viewer_pid: Optional[int],
    run_id: Optional[str] = None,
) -> str:
    """Persist visualization metadata in ``results/visualization/latest``.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    source_component : str
        Base source component rendered by napari.
    source_components : sequence of str
        Ordered source levels used for multiscale loading.
    position_index : int
        Reference ``p``-axis index.
    selected_positions : sequence of int
        Rendered position indices.
    show_all_positions : bool
        Whether all positions were rendered.
    parameters : mapping[str, Any]
        Effective visualization parameters.
    overlay_points_count : int
        Number of overlaid points.
    launch_mode : str
        Effective launch mode.
    viewer_pid : int, optional
        Viewer process ID when launched as subprocess.
    run_id : str, optional
        Optional provenance run identifier.

    Returns
    -------
    str
        Component path for the latest visualization metadata group.
    """
    component = "results/visualization/latest"
    root = zarr.open_group(str(zarr_path), mode="a")
    results_group = root.require_group("results")
    visualization_group = results_group.require_group("visualization")
    if "latest" in visualization_group:
        del visualization_group["latest"]
    latest_group = visualization_group.create_group("latest")

    payload: Dict[str, Any] = {
        "source_component": str(source_component),
        "source_components": [str(item) for item in source_components],
        "position_index": int(position_index),
        "selected_positions": [int(value) for value in selected_positions],
        "show_all_positions": bool(show_all_positions),
        "overlay_points_count": int(overlay_points_count),
        "launch_mode": str(launch_mode),
        "parameters": {str(k): v for k, v in dict(parameters).items()},
        "storage_policy": "latest_only",
        "run_id": run_id,
    }
    if viewer_pid is not None:
        payload["viewer_pid"] = int(viewer_pid)
    latest_group.attrs.update(payload)

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="visualization",
        component=component,
        run_id=run_id,
        metadata=payload,
    )
    return component


def run_visualization_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> VisualizationSummary:
    """Run napari visualization on canonical ClearEx analysis data.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Visualization parameters.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.
    run_id : str, optional
        Optional provenance run identifier.

    Returns
    -------
    VisualizationSummary
        Summary including source selection and viewer-launch metadata.

    Raises
    ------
    ValueError
        If source component is missing or selected position is out of bounds.
    """
    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_visualization_parameters(parameters)
    source_component = str(normalized.get("input_source", "data")).strip() or "data"

    root = zarr.open_group(str(zarr_path), mode="r")
    source_components = _resolve_multiscale_components(
        root=root,
        source_component=source_component,
        use_multiscale=bool(normalized.get("use_multiscale", True)),
    )

    source_array = root[source_component]
    source_shape = tuple(int(value) for value in source_array.shape)
    position_count = int(source_shape[1])
    show_all_positions = bool(normalized.get("show_all_positions", False))
    if show_all_positions:
        selected_positions: tuple[int, ...] = tuple(range(position_count))
    else:
        position_index = int(normalized["position_index"])
        if position_index >= position_count:
            raise ValueError(
                f"visualization position_index={position_index} is out of bounds "
                f"for position axis size {source_shape[1]}."
            )
        selected_positions = (position_index,)
    reference_position_index = int(selected_positions[0])

    points_by_position: Dict[int, np.ndarray] = {}
    point_properties_by_position: Dict[int, Dict[str, np.ndarray]] = {}
    if bool(normalized.get("overlay_particle_detections", True)):
        for position_index in selected_positions:
            overlay_points, overlay_properties = _load_particle_overlay_points(
                root=root,
                detection_component=str(normalized["particle_detection_component"]),
                position_index=int(position_index),
            )
            points_by_position[int(position_index)] = overlay_points
            point_properties_by_position[int(position_index)] = overlay_properties
    total_overlay_points = int(
        sum(
            int(np.asarray(points).shape[0])
            for points in points_by_position.values()
        )
    )
    point_property_names = tuple(
        sorted(
            {
                str(name)
                for properties in point_properties_by_position.values()
                for name in properties.keys()
            }
        )
    )

    napari_payload = _build_napari_layer_payload(
        zarr_path=zarr_path,
        root=root,
        source_component=source_component,
        source_components=source_components,
        position_index=reference_position_index,
        parameters=normalized,
        overlay_points_count=total_overlay_points,
        point_property_names=point_property_names,
    )
    position_affines_tczyx, stage_rows = _resolve_position_affines_tczyx(
        root_attrs=dict(root.attrs),
        selected_positions=selected_positions,
        scale_tczyx=napari_payload.scale_tczyx,
    )
    napari_payload.image_metadata["position_index"] = int(reference_position_index)
    napari_payload.image_metadata["selected_positions"] = [
        int(value) for value in selected_positions
    ]
    napari_payload.image_metadata["show_all_positions"] = bool(show_all_positions)
    napari_payload.image_metadata["position_affines_tczyx"] = {
        str(index): np.asarray(matrix, dtype=np.float64).tolist()
        for index, matrix in position_affines_tczyx.items()
    }
    napari_payload.image_metadata["stage_positions_xyztheta"] = [
        {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "theta": float(row["theta"]),
        }
        for row in stage_rows
    ]
    napari_payload.points_metadata["position_index"] = int(reference_position_index)
    napari_payload.points_metadata["selected_positions"] = [
        int(value) for value in selected_positions
    ]
    napari_payload.points_metadata["show_all_positions"] = bool(show_all_positions)

    _emit(20, "Preparing napari visualization layers")
    effective_launch_mode = _resolve_effective_launch_mode(
        str(normalized.get("launch_mode", "auto"))
    )
    viewer_pid: Optional[int] = None
    if effective_launch_mode == "subprocess":
        _emit(65, "Launching napari in a separate process")
        viewer_pid = _launch_napari_subprocess(
            zarr_path=zarr_path,
            normalized_parameters={
                **normalized,
                "launch_mode": "in_process",
            },
        )
    else:
        _emit(65, "Opening napari viewer")
        _launch_napari_viewer(
            zarr_path=zarr_path,
            source_components=source_components,
            source_component=source_component,
            selected_positions=selected_positions,
            points_by_position=points_by_position,
            point_properties_by_position=point_properties_by_position,
            position_affines_tczyx=position_affines_tczyx,
            axis_labels=napari_payload.axis_labels_tczyx,
            scale_tczyx=napari_payload.scale_tczyx,
            image_metadata=napari_payload.image_metadata,
            points_metadata=napari_payload.points_metadata,
        )

    _emit(90, "Writing visualization metadata")
    component = _save_visualization_metadata(
        zarr_path=zarr_path,
        source_component=source_component,
        source_components=source_components,
        position_index=reference_position_index,
        selected_positions=selected_positions,
        show_all_positions=show_all_positions,
        parameters=normalized,
        overlay_points_count=total_overlay_points,
        launch_mode=effective_launch_mode,
        viewer_pid=viewer_pid,
        run_id=run_id,
    )
    _emit(100, "Visualization workflow complete")
    return VisualizationSummary(
        component=component,
        source_component=source_component,
        source_components=tuple(str(item) for item in source_components),
        position_index=reference_position_index,
        selected_positions=tuple(int(value) for value in selected_positions),
        show_all_positions=bool(show_all_positions),
        overlay_points_count=total_overlay_points,
        launch_mode=effective_launch_mode,
        viewer_pid=viewer_pid,
    )


def _create_subprocess_parser() -> argparse.ArgumentParser:
    """Create CLI parser for subprocess napari-launch entrypoint.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for module entrypoint.
    """
    parser = argparse.ArgumentParser(
        prog="python -m clearex.visualization.pipeline",
    )
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--parameters-json", default="{}")
    return parser


def _run_subprocess_entrypoint(argv: Optional[Sequence[str]] = None) -> int:
    """Run subprocess entrypoint to open napari from JSON parameters.

    Parameters
    ----------
    argv : sequence of str, optional
        Explicit CLI argument list.

    Returns
    -------
    int
        Process exit code.
    """
    parser = _create_subprocess_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        parameters_raw = json.loads(str(args.parameters_json))
    except json.JSONDecodeError:
        parameters_raw = {}
    if not isinstance(parameters_raw, dict):
        parameters_raw = {}
    parameters_raw["launch_mode"] = "in_process"
    run_visualization_analysis(
        zarr_path=str(args.zarr_path),
        parameters=parameters_raw,
        progress_callback=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_subprocess_entrypoint())
