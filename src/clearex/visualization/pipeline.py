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

"""Napari visualization workflow for canonical ClearEx analysis stores."""

from __future__ import annotations

# Standard Library Imports
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

# Third Party Imports
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import zarr

# Local Imports
from clearex.io.experiment import load_navigate_experiment
from clearex.io.provenance import register_latest_output_reference
from clearex.workflow import (
    SpatialCalibrationConfig,
    format_spatial_calibration,
    spatial_calibration_from_dict,
    spatial_calibration_to_dict,
)

ProgressCallback = Callable[[int, str], None]
_AXIS_LABELS_TCZYX = ("t", "c", "z", "y", "x")
_TPCZYX_TO_TCZYX = (0, 2, 3, 4, 5)
_DEFAULT_PYRAMID_FACTORS_TPCZYX = (
    (1,),
    (1,),
    (1,),
    (1, 2, 4, 8),
    (1, 2, 4, 8),
    (1, 2, 4, 8),
)
_VOLUME_LAYER_MULTISCALE_POLICIES = frozenset({"inherit", "require", "off"})
_MAX_LAYER_DISPLAY_VOXELS = 256_000_000
_DISPLAY_PYRAMID_LEVELS_ATTR = "display_pyramid_levels"
_DISPLAY_PYRAMID_FACTORS_ATTR = "display_pyramid_factors_tpczyx"
_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "display_pyramid_levels_by_component"
_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR = "visualization_pyramid_levels"
_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR = "visualization_pyramid_factors_tpczyx"
_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "visualization_pyramid_levels_by_component"
_DISPLAY_CONTRAST_LIMITS_ATTR = "display_contrast_limits_by_channel"
_DISPLAY_CONTRAST_PERCENTILES_ATTR = "display_contrast_percentiles"
_DISPLAY_CONTRAST_LEVEL_SOURCE_ATTR = "display_contrast_source_component"
_DISPLAY_CONTRAST_SAMPLE_TARGET_VOXELS = 2_000_000
_DISPLAY_PYRAMID_BUILD_PROGRESS_TASK_STEP = 1_024
_DISPLAY_PYRAMID_BUILD_PROGRESS_MIN_INTERVAL_SECONDS = 1.5
_SOFTWARE_RENDERER_HINTS = (
    "llvmpipe",
    "softpipe",
    "swiftshader",
    "software rasterizer",
    "gdi generic",
)
_GPU_VENDOR_HINTS = (
    "nvidia",
    "amd",
    "ati",
    "intel",
    "apple",
    "asahi",
    "mesa dri",
    "radeon",
    "iris",
    "adreno",
    "mali",
    "powervr",
    "geforce",
    "quadro",
    "tesla",
    "rtx",
)
_LOGGER = logging.getLogger(__name__)


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
    keyframe_manifest_path : str, optional
        JSON path used to persist interactive keyframe selections.
    keyframe_count : int
        Number of captured keyframes written to the manifest.
    renderer : dict[str, Any], optional
        OpenGL renderer probe payload captured from napari context.
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
    keyframe_manifest_path: Optional[str] = None
    keyframe_count: int = 0
    renderer: Optional[Dict[str, Any]] = None
    viewer_ndisplay_requested: int = 3
    viewer_ndisplay_effective: int = 3
    display_mode_fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class DisplayPyramidSummary:
    """Summary metadata for one display-pyramid preparation run.

    Attributes
    ----------
    component : str
        Latest output metadata group path for the task.
    source_component : str
        Source component used to prepare the display pyramid.
    source_components : tuple[str, ...]
        Ordered multiscale components registered for later visualization.
    channel_count : int
        Number of source channels summarized for display contrast.
    contrast_limits_by_channel : tuple[tuple[float, float], ...]
        Stored per-channel contrast limits derived from the fixed display
        percentiles.
    reused_existing_levels : bool
        Whether the task reused existing levels instead of writing new ones.
    """

    component: str
    source_component: str
    source_components: tuple[str, ...]
    channel_count: int
    contrast_limits_by_channel: tuple[tuple[float, float], ...]
    reused_existing_levels: bool


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


@dataclass(frozen=True)
class ResolvedVolumeLayer:
    """Resolved visualization layer configuration for one source component.

    Attributes
    ----------
    component : str
        Base component path rendered by this layer.
    source_components : tuple[str, ...]
        Ordered component paths for multiscale rendering, including base level.
    layer_type : str
        Napari layer type: ``image`` or ``labels``.
    channels : tuple[int, ...]
        Requested channel indices. Empty means all available channels.
    visible : bool, optional
        Explicit visibility override, or ``None`` to use napari defaults.
    opacity : float, optional
        Explicit opacity override in ``[0, 1]``, or ``None`` for defaults.
    blending : str
        Optional napari blending mode override.
    colormap : str
        Optional colormap override for image layers.
    rendering : str
        Optional rendering-style override for image layers.
    name : str
        Optional layer base-name override.
    multiscale_policy : str
        Layer multiscale policy: ``inherit``, ``require``, or ``off``.
    multiscale_status : str
        Effective multiscale outcome: ``off``, ``existing``, or
        ``single_scale``.
    """

    component: str
    source_components: tuple[str, ...]
    layer_type: str
    channels: tuple[int, ...]
    visible: Optional[bool]
    opacity: Optional[float]
    blending: str
    colormap: str
    rendering: str
    name: str
    multiscale_policy: str
    multiscale_status: str


def _decode_gl_string(value: Any) -> str:
    """Decode OpenGL vendor/renderer/version strings into plain text."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="ignore").strip()
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _probe_napari_opengl_renderer(viewer: Any) -> Dict[str, Any]:
    """Probe active napari OpenGL renderer metadata.

    Parameters
    ----------
    viewer : Any
        Napari viewer instance.

    Returns
    -------
    dict[str, Any]
        Renderer metadata with keys:
        ``vendor``, ``renderer``, ``version``, ``software_renderer``,
        ``gpu_renderer``, ``gpu_vendor_hint``, and optional ``error``.
    """
    vendor = ""
    renderer = ""
    version = ""
    error_text = ""
    try:
        canvas = getattr(
            getattr(getattr(viewer, "window", None), "qt_viewer", None),
            "canvas",
            None,
        )
        if canvas is None:
            canvas = getattr(
                getattr(getattr(viewer, "window", None), "_qt_viewer", None),
                "canvas",
                None,
            )
        if canvas is not None:
            context = getattr(canvas, "context", None)
            if context is not None and callable(getattr(context, "set_current", None)):
                context.set_current()
        from vispy.gloo import gl

        vendor = _decode_gl_string(gl.glGetParameter(gl.GL_VENDOR))
        renderer = _decode_gl_string(gl.glGetParameter(gl.GL_RENDERER))
        version = _decode_gl_string(gl.glGetParameter(gl.GL_VERSION))
    except Exception as exc:
        error_text = str(exc)

    combined = f"{vendor} {renderer}".strip().lower()
    software_renderer = bool(
        combined and any(hint in combined for hint in _SOFTWARE_RENDERER_HINTS)
    )
    gpu_vendor_hint = any(hint in combined for hint in _GPU_VENDOR_HINTS)
    gpu_renderer = bool(combined) and not software_renderer and gpu_vendor_hint

    payload: Dict[str, Any] = {
        "vendor": vendor,
        "renderer": renderer,
        "version": version,
        "software_renderer": bool(software_renderer),
        "gpu_renderer": bool(gpu_renderer),
        "gpu_vendor_hint": bool(gpu_vendor_hint),
    }
    if error_text:
        payload["error"] = error_text
    return payload


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
        return {str(key): _sanitize_metadata_value(item) for key, item in value.items()}
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
            (
                (fov_x / img_x)
                if fov_x is not None and img_x is not None and img_x > 0
                else None
            ),
            (
                (fov_y / img_y)
                if fov_y is not None and img_y is not None and img_y > 0
                else None
            ),
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
        lateral_from_pixel_zoom = float(
            pixel_size / zoom * ((binning_x + binning_y) / 2.0)
        )

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
    volume_layers: Sequence[ResolvedVolumeLayer],
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
    volume_layers : sequence[ResolvedVolumeLayer]
        Resolved volume-layer specifications. First row is treated as primary
        source for metadata compatibility fields.
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
    if len(volume_layers) <= 0:
        raise ValueError("Visualization requires at least one resolved volume layer.")

    primary_layer = volume_layers[0]
    source_component = str(primary_layer.component)
    source_components = tuple(str(item) for item in primary_layer.source_components)
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
    volume_layers_payload = _serialize_resolved_volume_layers(volume_layers)

    image_metadata_raw: Dict[str, Any] = {
        "schema": root_attrs.get("schema"),
        "store_path": str(Path(zarr_path).expanduser().resolve()),
        "axis_labels_tczyx": list(_AXIS_LABELS_TCZYX),
        "scale_tczyx": [float(value) for value in scale_tczyx],
        "source_component": str(source_component),
        "source_components": [str(item) for item in source_components],
        "volume_layers": volume_layers_payload,
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
        "display_pyramid_lookup": {
            "source_attr": _DISPLAY_PYRAMID_LEVELS_ATTR,
            "root_attr_map": _DISPLAY_PYRAMID_ROOT_MAP_ATTR,
            "legacy_source_attr": _LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR,
            "legacy_root_attr_map": _LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR,
        },
        "display_contrast_metadata_reference": {
            "limits_attr": _DISPLAY_CONTRAST_LIMITS_ATTR,
            "percentiles_attr": _DISPLAY_CONTRAST_PERCENTILES_ATTR,
        },
        "data_pyramid_levels": root_attrs.get("data_pyramid_levels"),
        "data_pyramid_factors_tpczyx": root_attrs.get("data_pyramid_factors_tpczyx"),
        "multiscale_levels": multiscale_levels,
        "primary_multiscale_status": str(primary_layer.multiscale_status),
        "source_array_attrs": source_attrs,
        "root_attrs": root_attrs,
        "visualization_parameters": dict(parameters),
    }
    points_metadata_raw: Dict[str, Any] = {
        "coordinate_axes_tczyx": list(_AXIS_LABELS_TCZYX),
        "scale_tczyx": [float(value) for value in scale_tczyx],
        "source_component": str(source_component),
        "volume_layers": volume_layers_payload,
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
    normalized["show_all_positions"] = bool(normalized.get("show_all_positions", False))
    normalized["position_index"] = max(0, int(normalized.get("position_index", 0)))
    normalized["use_multiscale"] = bool(normalized.get("use_multiscale", True))
    normalized["use_3d_view"] = bool(normalized.get("use_3d_view", True))
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
    normalized["require_gpu_rendering"] = bool(
        normalized.get("require_gpu_rendering", True)
    )
    normalized["capture_keyframes"] = bool(normalized.get("capture_keyframes", True))
    normalized["keyframe_manifest_path"] = str(
        normalized.get("keyframe_manifest_path", "")
    ).strip()
    normalized["keyframe_layer_overrides"] = _normalize_keyframe_layer_overrides(
        normalized.get("keyframe_layer_overrides", [])
    )
    normalized["volume_layers"] = _normalize_visualization_volume_layers(
        normalized.get("volume_layers", [])
    )
    if not normalized["volume_layers"]:
        default_source = str(normalized.get("input_source", "data")).strip() or "data"
        normalized["volume_layers"] = [
            {
                "component": default_source,
                "name": "",
                "layer_type": "image",
                "channels": [],
                "visible": None,
                "opacity": None,
                "blending": "",
                "colormap": "",
                "rendering": "",
                "multiscale_policy": (
                    "inherit" if bool(normalized.get("use_multiscale", True)) else "off"
                ),
            }
        ]

    launch_mode = str(normalized.get("launch_mode", "auto")).strip().lower() or "auto"
    if launch_mode not in {"auto", "in_process", "subprocess"}:
        raise ValueError(
            "visualization launch_mode must be one of auto, in_process, subprocess."
        )
    normalized["launch_mode"] = launch_mode
    return normalized


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


def _resolve_keyframe_manifest_path(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
) -> Optional[Path]:
    """Resolve keyframe manifest JSON path from runtime parameters.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path used as the default naming anchor.
    parameters : mapping[str, Any]
        Effective visualization parameters.

    Returns
    -------
    pathlib.Path, optional
        Absolute path where keyframes should be written. Returns ``None`` when
        keyframe capture is disabled.

    Raises
    ------
    None
        Invalid or missing optional values fall back to defaults.
    """
    capture_keyframes = bool(parameters.get("capture_keyframes", True))
    if not capture_keyframes:
        return None

    explicit_path = str(parameters.get("keyframe_manifest_path", "")).strip()
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    store_path = Path(zarr_path).expanduser().resolve()
    return Path(f"{store_path}.visualization_keyframes.json")


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    """Coerce a value to ``True``/``False``/``None``.

    Parameters
    ----------
    value : Any
        Candidate boolean value.

    Returns
    -------
    bool, optional
        Parsed boolean, or ``None`` when the value is unspecified.

    Raises
    ------
    None
        Invalid values return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if not text or text == "auto":
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_keyframe_layer_overrides(value: Any) -> list[Dict[str, Any]]:
    """Normalize keyframe-layer override rows from runtime parameters.

    Parameters
    ----------
    value : Any
        Candidate list-like value from visualization parameters.

    Returns
    -------
    list[dict[str, Any]]
        Normalized rows with keys ``layer_name``, ``visible``, ``colormap``,
        ``rendering``, and ``annotation``.

    Raises
    ------
    None
        Invalid rows are skipped.
    """
    if not isinstance(value, (list, tuple)):
        return []

    rows: list[Dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, Mapping):
            continue
        layer_name = str(raw_row.get("layer_name", raw_row.get("layer", ""))).strip()
        visible = _coerce_optional_bool(raw_row.get("visible"))
        colormap = str(raw_row.get("colormap", raw_row.get("lut", ""))).strip()
        rendering = str(raw_row.get("rendering", "")).strip()
        annotation = str(raw_row.get("annotation", "")).strip()
        if not any(
            (
                layer_name,
                colormap,
                rendering,
                annotation,
                visible is not None,
            )
        ):
            continue
        rows.append(
            {
                "layer_name": layer_name,
                "visible": visible,
                "colormap": colormap,
                "rendering": rendering,
                "annotation": annotation,
            }
        )
    return rows


def _coerce_optional_unit_interval_float(value: Any) -> Optional[float]:
    """Coerce optional opacity values into finite ``[0, 1]`` floats.

    Parameters
    ----------
    value : Any
        Candidate numeric opacity value.

    Returns
    -------
    float, optional
        Parsed opacity value in ``[0, 1]`` or ``None`` when unspecified.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(min(1.0, max(0.0, parsed)))


def _normalize_visualization_channels(value: Any) -> list[int]:
    """Normalize one layer's channel-selection field.

    Parameters
    ----------
    value : Any
        Candidate channel selection value.

    Returns
    -------
    list[int]
        Unique sorted non-negative channel indices. Empty means all channels.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"all", "auto"}:
            return []
        items = [part.strip() for part in text.split(",")]
    elif isinstance(value, (tuple, list)):
        items = list(value)
    else:
        items = [value]

    parsed: list[int] = []
    for item in items:
        if item is None:
            continue
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        parsed.append(int(index))
    return sorted(set(parsed))


def _normalize_visualization_volume_layers(value: Any) -> list[Dict[str, Any]]:
    """Normalize visualization layer rows used for volume overlays.

    Parameters
    ----------
    value : Any
        Candidate list-like rows from runtime parameters.

    Returns
    -------
    list[dict[str, Any]]
        Normalized layer rows.
    """
    if not isinstance(value, (tuple, list)):
        return []

    rows: list[Dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, Mapping):
            continue
        component = str(
            raw_row.get("component", raw_row.get("source_component", ""))
        ).strip()
        if not component:
            continue
        layer_type = str(raw_row.get("layer_type", "image")).strip().lower() or "image"
        if layer_type not in {"image", "labels"}:
            layer_type = "image"
        multiscale_policy = (
            str(raw_row.get("multiscale_policy", "inherit")).strip().lower()
            or "inherit"
        )
        if multiscale_policy == "auto_build":
            multiscale_policy = "inherit"
        if multiscale_policy not in _VOLUME_LAYER_MULTISCALE_POLICIES:
            multiscale_policy = "inherit"
        blending = str(raw_row.get("blending", "")).strip().lower()
        if blending in {"auto", "default"}:
            blending = ""
        colormap = str(raw_row.get("colormap", raw_row.get("lut", ""))).strip()
        if colormap.strip().lower() in {"auto", "default"}:
            colormap = ""
        rendering = str(raw_row.get("rendering", "")).strip().lower()
        if rendering in {"auto", "default"}:
            rendering = ""
        rows.append(
            {
                "component": component,
                "name": str(raw_row.get("name", "")).strip(),
                "layer_type": layer_type,
                "channels": _normalize_visualization_channels(raw_row.get("channels")),
                "visible": _coerce_optional_bool(raw_row.get("visible")),
                "opacity": _coerce_optional_unit_interval_float(raw_row.get("opacity")),
                "blending": blending,
                "colormap": colormap,
                "rendering": rendering,
                "multiscale_policy": multiscale_policy,
            }
        )
    return rows


def _serialize_colormap(colormap: Any) -> Optional[Dict[str, Any]]:
    """Serialize napari colormap metadata for reproducible manifests.

    Parameters
    ----------
    colormap : Any
        Napari colormap object.

    Returns
    -------
    dict[str, Any], optional
        Serializable colormap payload when available.

    Raises
    ------
    None
        Missing attributes are ignored.
    """
    if colormap is None:
        return None

    payload: Dict[str, Any] = {
        "name": str(getattr(colormap, "name", colormap)),
    }
    controls = getattr(colormap, "controls", None)
    colors = getattr(colormap, "colors", None)
    interpolation = getattr(colormap, "interpolation", None)
    if controls is not None:
        payload["controls"] = _sanitize_metadata_value(controls)
    if colors is not None:
        payload["colors"] = _sanitize_metadata_value(colors)
    if interpolation is not None:
        payload["interpolation"] = str(interpolation)
    return _sanitize_metadata_value(payload)


def _serialize_layer_attribute(value: Any) -> Any:
    """Serialize a layer attribute while bounding manifest growth.

    Parameters
    ----------
    value : Any
        Candidate attribute value.

    Returns
    -------
    Any
        Serializable value or a compact summary for large arrays/sequences.

    Raises
    ------
    None
        Invalid values are converted through metadata sanitization.
    """
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if int(array.size) > 256:
            return {
                "truncated": True,
                "shape": [int(dim) for dim in array.shape],
                "dtype": str(array.dtype),
            }
        return _sanitize_metadata_value(array)
    if isinstance(value, (list, tuple)) and len(value) > 256:
        return {
            "truncated": True,
            "length": int(len(value)),
        }
    return _sanitize_metadata_value(value)


def _serialize_layer_configuration(
    layer: Any,
    *,
    order_index: int,
    selected_layer_names: set[str],
    active_layer_name: Optional[str],
    override_by_layer_name: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Serialize a napari layer into movie-relevant display settings.

    Parameters
    ----------
    layer : Any
        Napari layer instance.
    order_index : int
        Layer index in viewer draw order.
    selected_layer_names : set[str]
        Names currently selected in the viewer layer list.
    active_layer_name : str, optional
        Active/primary selected layer name.
    override_by_layer_name : mapping[str, mapping[str, Any]]
        Optional row overrides keyed by layer name.

    Returns
    -------
    dict[str, Any]
        Layer display payload containing visibility, opacity, blending, and
        image-specific properties when available.

    Raises
    ------
    None
        Missing attributes are ignored and omitted from the payload.
    """
    layer_name = str(getattr(layer, "name", ""))
    payload: Dict[str, Any] = {
        "name": layer_name,
        "type": str(type(layer).__name__),
        "order_index": int(max(0, order_index)),
        "selected": bool(layer_name in selected_layer_names),
        "active": bool(layer_name == active_layer_name),
        "visible": bool(getattr(layer, "visible", True)),
        "opacity": float(_safe_float(getattr(layer, "opacity", 1.0), default=1.0)),
        "blending": str(getattr(layer, "blending", "opaque")),
    }

    annotation_row = override_by_layer_name.get(layer_name, {})
    annotation = str(annotation_row.get("annotation", "")).strip()
    if annotation:
        payload["annotation"] = annotation

    visible_override = _coerce_optional_bool(annotation_row.get("visible"))
    if visible_override is not None:
        payload["visible_override"] = bool(visible_override)
    colormap_override = str(annotation_row.get("colormap", "")).strip()
    if colormap_override:
        payload["colormap_override"] = colormap_override
    rendering_override = str(annotation_row.get("rendering", "")).strip()
    if rendering_override:
        payload["rendering_override"] = rendering_override

    colormap_payload = _serialize_colormap(getattr(layer, "colormap", None))
    if colormap_payload is not None:
        payload["colormap"] = colormap_payload

    contrast_limits = getattr(layer, "contrast_limits", None)
    if isinstance(contrast_limits, (tuple, list)) and len(contrast_limits) >= 2:
        payload["contrast_limits"] = [
            float(_safe_float(contrast_limits[0], default=0.0)),
            float(_safe_float(contrast_limits[1], default=1.0)),
        ]

    for attr_name in (
        "gamma",
        "attenuation",
        "iso_threshold",
        "edge_width",
        "border_width",
    ):
        if hasattr(layer, attr_name):
            payload[attr_name] = float(
                _safe_float(getattr(layer, attr_name, 0.0), default=0.0)
            )
    for attr_name in (
        "rendering",
        "depiction",
        "interpolation",
        "interpolation2d",
        "interpolation3d",
        "symbol",
        "face_color",
        "edge_color",
        "border_color",
    ):
        if hasattr(layer, attr_name):
            payload[attr_name] = _serialize_layer_attribute(
                getattr(layer, attr_name, None)
            )
    for attr_name in ("size", "scale", "translate", "rotate", "shear"):
        if hasattr(layer, attr_name):
            payload[attr_name] = _serialize_layer_attribute(
                getattr(layer, attr_name, None)
            )

    affine_value = getattr(layer, "affine", None)
    if affine_value is not None:
        affine_matrix = getattr(affine_value, "affine_matrix", None)
        if affine_matrix is not None:
            payload["affine_matrix"] = _serialize_layer_attribute(affine_matrix)
        else:
            payload["affine"] = _serialize_layer_attribute(affine_value)
    return payload


def _capture_viewer_keyframe(
    *,
    viewer: Any,
    keyframe_index: int,
    layer_overrides: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Capture camera/dim/layer state from a live napari viewer.

    Parameters
    ----------
    viewer : Any
        Napari viewer instance.
    keyframe_index : int
        Zero-based keyframe index.
    layer_overrides : sequence of mapping[str, Any]
        Optional layer-override rows from visualization parameters.

    Returns
    -------
    dict[str, Any]
        Serializable keyframe payload.

    Raises
    ------
    None
        Attribute lookups are best-effort and fall back to defaults.
    """
    camera = getattr(viewer, "camera", None)
    dims = getattr(viewer, "dims", None)
    layer_list = getattr(viewer, "layers", None)
    layers = tuple(layer_list) if layer_list is not None else tuple()
    selection = getattr(layer_list, "selection", None)

    angles_raw = tuple(getattr(camera, "angles", (0.0, 0.0, 0.0)))
    center_raw = tuple(getattr(camera, "center", (0.0, 0.0, 0.0)))
    current_step_raw = tuple(getattr(dims, "current_step", tuple()))
    axis_labels_raw = tuple(getattr(dims, "axis_labels", tuple()))
    dims_order_raw = tuple(getattr(dims, "order", tuple()))

    selected_layer_names: set[str] = set()
    active_layer_name: Optional[str] = None
    if selection is not None:
        try:
            selected_layer_names = {
                str(getattr(layer, "name", "")) for layer in tuple(selection)
            }
        except Exception:
            selected_layer_names = set()
        try:
            active_layer = getattr(selection, "active", None)
            if active_layer is not None:
                active_layer_name = str(getattr(active_layer, "name", ""))
        except Exception:
            active_layer_name = None

    override_by_layer_name: Dict[str, Mapping[str, Any]] = {}
    for row in _normalize_keyframe_layer_overrides(layer_overrides):
        name = str(row.get("layer_name", "")).strip()
        if name:
            override_by_layer_name[name] = row

    keyframe: Dict[str, Any] = {
        "index": int(max(0, keyframe_index)),
        "captured_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "camera": {
            "angles": [
                float(_safe_float(value, default=0.0)) for value in angles_raw[:3]
            ],
            "zoom": float(_safe_float(getattr(camera, "zoom", 1.0), default=1.0)),
            "center": [
                float(_safe_float(value, default=0.0)) for value in center_raw[:3]
            ],
            "perspective": float(
                _safe_float(getattr(camera, "perspective", 0.0), default=0.0)
            ),
            "field_of_view": float(
                _safe_float(getattr(camera, "fov", 0.0), default=0.0)
            ),
        },
        "dims": {
            "current_step": [
                int(round(_safe_float(v, default=0.0))) for v in current_step_raw
            ],
            "axis_labels": [str(label) for label in axis_labels_raw],
            "ndisplay": int(
                max(2, round(_safe_float(getattr(dims, "ndisplay", 3), default=3.0)))
            ),
            "order": [int(round(_safe_float(v, default=0.0))) for v in dims_order_raw],
        },
        "viewer": {
            "theme": str(getattr(viewer, "theme", "")),
            "title": str(getattr(viewer, "title", "")),
            "layer_order": [str(getattr(layer, "name", "")) for layer in layers],
            "selected_layers": sorted(selected_layer_names),
            "active_layer": active_layer_name,
        },
        "layers": [
            _serialize_layer_configuration(
                layer,
                order_index=index,
                selected_layer_names=selected_layer_names,
                active_layer_name=active_layer_name,
                override_by_layer_name=override_by_layer_name,
            )
            for index, layer in enumerate(layers)
        ],
    }
    return _sanitize_metadata_value(keyframe)


def _write_keyframe_manifest(
    *,
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write the interactive keyframe manifest as JSON.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Destination JSON path.
    payload : mapping[str, Any]
        Manifest payload containing keyframe snapshots.

    Returns
    -------
    None
        Writes the JSON file to disk.

    Raises
    ------
    OSError
        If output directories/files cannot be written.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_sanitize_metadata_value(dict(payload)), indent=2),
        encoding="utf-8",
    )


def _coerce_level_factors_level_major(
    value: Any,
) -> Optional[tuple[tuple[int, int, int, int, int, int], ...]]:
    """Parse absolute pyramid factors from level-major payloads.

    Parameters
    ----------
    value : Any
        Candidate payload where each row is one absolute factor tuple in
        ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...], optional
        Parsed level-major factors, or ``None`` when parsing fails.
    """
    if not isinstance(value, (tuple, list)):
        return None
    parsed_levels: list[tuple[int, int, int, int, int, int]] = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 6:
            return None
        parsed_row: list[int] = []
        for item in row:
            try:
                factor = int(item)
            except (TypeError, ValueError):
                return None
            if factor <= 0:
                return None
            parsed_row.append(int(factor))
        parsed_levels.append(
            (
                int(parsed_row[0]),
                int(parsed_row[1]),
                int(parsed_row[2]),
                int(parsed_row[3]),
                int(parsed_row[4]),
                int(parsed_row[5]),
            )
        )
    if not parsed_levels:
        return None
    if any(int(value) != 1 for value in parsed_levels[0]):
        return None
    return tuple(parsed_levels)


def _coerce_level_factors_axis_major(
    value: Any,
) -> Optional[tuple[tuple[int, int, int, int, int, int], ...]]:
    """Parse absolute pyramid factors from axis-major payloads.

    Parameters
    ----------
    value : Any
        Candidate payload where each axis provides one list of factors.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...], optional
        Parsed absolute level factors, or ``None`` when parsing fails.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        return None

    axis_levels: list[list[int]] = []
    for axis_payload in value:
        if not isinstance(axis_payload, (tuple, list)) or len(axis_payload) <= 0:
            return None
        parsed_axis: list[int] = []
        for item in axis_payload:
            try:
                factor = int(item)
            except (TypeError, ValueError):
                return None
            if factor <= 0:
                return None
            parsed_axis.append(int(factor))
        if int(parsed_axis[0]) != 1:
            return None
        axis_levels.append(parsed_axis)

    level_count = max(len(levels) for levels in axis_levels)
    for axis_index in range(len(axis_levels)):
        axis = axis_levels[axis_index]
        if len(axis) >= level_count:
            continue
        axis_levels[axis_index] = axis + ([int(axis[-1])] * (level_count - len(axis)))

    levels: list[tuple[int, int, int, int, int, int]] = []
    for level_index in range(level_count):
        levels.append(
            (
                int(axis_levels[0][level_index]),
                int(axis_levels[1][level_index]),
                int(axis_levels[2][level_index]),
                int(axis_levels[3][level_index]),
                int(axis_levels[4][level_index]),
                int(axis_levels[5][level_index]),
            )
        )
    if any(int(value) != 1 for value in levels[0]):
        return None
    return tuple(levels)


def _resolve_visualization_pyramid_factors_tpczyx(
    *,
    root_attrs: Mapping[str, Any],
    source_attrs: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Resolve absolute level factors used by display-pyramid preparation.

    Parameters
    ----------
    root_attrs : mapping[str, Any]
        Root-group attrs.
    source_attrs : mapping[str, Any]
        Source-array attrs.

    Returns
    -------
    tuple[tuple[int, int, int, int, int, int], ...]
        Absolute level factors in canonical order.
    """
    candidates = [
        source_attrs.get(_DISPLAY_PYRAMID_FACTORS_ATTR),
        source_attrs.get(_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR),
        source_attrs.get("visualization_pyramid_factors_tpczyx"),
        source_attrs.get("pyramid_factors_tpczyx"),
        source_attrs.get("resolution_pyramid_factors_tpczyx"),
        root_attrs.get("resolution_pyramid_factors_tpczyx"),
        _DEFAULT_PYRAMID_FACTORS_TPCZYX,
    ]
    for candidate in candidates:
        parsed_level_major = _coerce_level_factors_level_major(candidate)
        if parsed_level_major is not None:
            return parsed_level_major
        parsed_axis_major = _coerce_level_factors_axis_major(candidate)
        if parsed_axis_major is not None:
            return parsed_axis_major
    return ((1, 1, 1, 1, 1, 1),)


def _downsample_tpczyx_by_stride(
    array: da.Array,
    factors_tpczyx: tuple[int, int, int, int, int, int],
) -> da.Array:
    """Create a nearest-neighbor strided downsampled view.

    Parameters
    ----------
    array : dask.array.Array
        Canonical ``(t, p, c, z, y, x)`` array.
    factors_tpczyx : tuple[int, int, int, int, int, int]
        Stride factors in canonical order.

    Returns
    -------
    dask.array.Array
        Strided view suitable for multiscale levels.
    """
    return array[
        tuple(slice(None, None, max(1, int(factor))) for factor in factors_tpczyx)
    ]


def _downsample_tczyx_by_stride(
    array: da.Array,
    factors_zyx: tuple[int, int, int],
) -> da.Array:
    """Create a strided ``(t, c, z, y, x)`` downsampled view."""
    z_factor = max(1, int(factors_zyx[0]))
    y_factor = max(1, int(factors_zyx[1]))
    x_factor = max(1, int(factors_zyx[2]))
    return array[:, :, ::z_factor, ::y_factor, ::x_factor]


def _volume_voxel_count_tczyx(shape_tczyx: Sequence[int]) -> int:
    """Return voxel count for ``(z, y, x)`` axes of a ``(t, c, z, y, x)`` shape."""
    if len(tuple(shape_tczyx)) < 5:
        return 0
    return max(
        0,
        int(shape_tczyx[-3]) * int(shape_tczyx[-2]) * int(shape_tczyx[-1]),
    )


def _resolve_effective_viewer_ndisplay(
    *,
    root: zarr.hierarchy.Group,
    volume_layers: Sequence[ResolvedVolumeLayer],
    requested_ndisplay: int,
    max_display_voxels: int = _MAX_LAYER_DISPLAY_VOXELS,
) -> tuple[int, Optional[str]]:
    """Resolve effective napari display mode with strict 3D safety fallback."""
    requested = 3 if int(requested_ndisplay) >= 3 else 2
    if requested != 3:
        return 2, None

    for layer in volume_layers:
        if str(layer.layer_type) != "image":
            continue
        source_component = str(layer.component).strip() or "data"
        try:
            source_array = root[source_component]
        except Exception:
            continue
        shape_tpczyx = tuple(int(value) for value in tuple(source_array.shape))
        if len(shape_tpczyx) != 6:
            continue
        voxel_count = max(
            0,
            int(shape_tpczyx[3]) * int(shape_tpczyx[4]) * int(shape_tpczyx[5]),
        )
        if voxel_count <= int(max_display_voxels):
            continue
        reason = (
            "3D rendering requested, but image layer "
            f"'{source_component}' has {voxel_count:,} z/y/x voxels, exceeding the "
            f"ClearEx 3D render limit of {int(max_display_voxels):,}. "
            "Falling back to 2D without launch-time downsampling."
        )
        return 2, reason
    return 3, None


def _resolve_display_level_arrays_for_voxel_budget(
    *,
    level_arrays: Sequence[da.Array],
    layer_scale_tczyx: Sequence[float],
    max_display_voxels: int = _MAX_LAYER_DISPLAY_VOXELS,
) -> tuple[
    tuple[da.Array, ...],
    tuple[float, float, float, float, float],
    tuple[float, float, float],
    str,
]:
    """Resolve display arrays/scales that fit a 3D texture voxel budget.

    Parameters
    ----------
    level_arrays : sequence of dask.array.Array
        Layer arrays in ``(t, c, z, y, x)`` order, finest to coarsest.
    layer_scale_tczyx : sequence of float
        Base layer scale in ``(t, c, z, y, x)`` order.
    max_display_voxels : int, default=256_000_000
        Target voxel budget for one 3D texture upload.

    Returns
    -------
    tuple[tuple[dask.array.Array, ...], tuple[float, float, float, float, float], tuple[float, float, float], str]
        ``(resolved_arrays, resolved_scale_tczyx, display_factors_zyx, strategy)``.
        Strategy values include ``"none"``, ``"coarse_level"``, and
        ``"coarse_level+stride"``/``"stride"``.
    """
    if len(level_arrays) <= 0:
        normalized_scale = _normalize_scale_tczyx(layer_scale_tczyx)
        return tuple(), normalized_scale, (1.0, 1.0, 1.0), "none"

    resolved_arrays = tuple(level_arrays)
    normalized_scale = _normalize_scale_tczyx(layer_scale_tczyx)
    display_factors_zyx = [1.0, 1.0, 1.0]
    strategy = "none"
    budget = max(1, int(max_display_voxels))

    finest_shape = tuple(int(value) for value in tuple(resolved_arrays[0].shape))
    finest_voxels = _volume_voxel_count_tczyx(finest_shape)
    if finest_voxels > budget and len(resolved_arrays) > 1:
        selected_index = len(resolved_arrays) - 1
        for index, candidate in enumerate(resolved_arrays):
            candidate_shape = tuple(int(value) for value in tuple(candidate.shape))
            if _volume_voxel_count_tczyx(candidate_shape) <= budget:
                selected_index = int(index)
                break
        if selected_index > 0:
            resolved_arrays = tuple(resolved_arrays[selected_index:])
            selected_shape = tuple(
                int(value) for value in tuple(resolved_arrays[0].shape)
            )
            for axis_index, (base_dim, selected_dim) in enumerate(
                zip(finest_shape[-3:], selected_shape[-3:], strict=False)
            ):
                if int(base_dim) <= 0 or int(selected_dim) <= 0:
                    factor = 1.0
                else:
                    factor = max(1.0, float(base_dim) / float(selected_dim))
                display_factors_zyx[axis_index] *= float(factor)
            normalized_scale = _normalize_scale_tczyx(
                (
                    float(normalized_scale[0]),
                    float(normalized_scale[1]),
                    float(normalized_scale[2]) * float(display_factors_zyx[0]),
                    float(normalized_scale[3]) * float(display_factors_zyx[1]),
                    float(normalized_scale[4]) * float(display_factors_zyx[2]),
                )
            )
            strategy = "coarse_level"

    current_shape = tuple(int(value) for value in tuple(resolved_arrays[0].shape))
    current_voxels = _volume_voxel_count_tczyx(current_shape)
    if current_voxels > budget:
        stride = max(
            1,
            int(math.ceil((float(current_voxels) / float(budget)) ** (1.0 / 3.0))),
        )
        if stride > 1:
            resolved_arrays = tuple(
                _downsample_tczyx_by_stride(array, (stride, stride, stride))
                for array in resolved_arrays
            )
            display_factors_zyx = [
                float(display_factors_zyx[0]) * float(stride),
                float(display_factors_zyx[1]) * float(stride),
                float(display_factors_zyx[2]) * float(stride),
            ]
            normalized_scale = _normalize_scale_tczyx(
                (
                    float(normalized_scale[0]),
                    float(normalized_scale[1]),
                    float(normalized_scale[2]) * float(stride),
                    float(normalized_scale[3]) * float(stride),
                    float(normalized_scale[4]) * float(stride),
                )
            )
            strategy = "stride" if strategy == "none" else f"{strategy}+stride"

    return (
        resolved_arrays,
        normalized_scale,
        (
            float(display_factors_zyx[0]),
            float(display_factors_zyx[1]),
            float(display_factors_zyx[2]),
        ),
        strategy,
    )


def _resolve_level_chunks_tpczyx(
    *,
    source_chunks: Optional[Sequence[int]],
    level_shape: Sequence[int],
    level_array: Optional[da.Array] = None,
) -> tuple[int, int, int, int, int, int]:
    """Resolve practical chunking for one generated multiscale level.

    Parameters
    ----------
    source_chunks : sequence[int], optional
        Source-array chunk shape.
    level_shape : sequence[int]
        Level array shape.
    level_array : dask.array.Array, optional
        Generated level array. When provided, its current chunk geometry is
        preferred to avoid introducing costly rechunk/shuffle traffic before
        persistence.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Chunk shape in canonical order.
    """
    normalized_shape = tuple(max(1, int(size)) for size in level_shape)
    if level_array is not None:
        chunksize = getattr(level_array, "chunksize", None)
        if (
            isinstance(chunksize, tuple)
            and len(chunksize) == 6
            and all(int(size) > 0 for size in chunksize)
        ):
            return (
                int(max(1, min(normalized_shape[0], int(chunksize[0])))),
                int(max(1, min(normalized_shape[1], int(chunksize[1])))),
                int(max(1, min(normalized_shape[2], int(chunksize[2])))),
                int(max(1, min(normalized_shape[3], int(chunksize[3])))),
                int(max(1, min(normalized_shape[4], int(chunksize[4])))),
                int(max(1, min(normalized_shape[5], int(chunksize[5])))),
            )
    if source_chunks is None or len(tuple(source_chunks)) != 6:
        return (
            int(normalized_shape[0]),
            int(normalized_shape[1]),
            int(normalized_shape[2]),
            int(normalized_shape[3]),
            int(normalized_shape[4]),
            int(normalized_shape[5]),
        )
    chunks = tuple(int(size) for size in source_chunks)
    return (
        int(max(1, min(normalized_shape[0], chunks[0]))),
        int(max(1, min(normalized_shape[1], chunks[1]))),
        int(max(1, min(normalized_shape[2], chunks[2]))),
        int(max(1, min(normalized_shape[3], chunks[3]))),
        int(max(1, min(normalized_shape[4], chunks[4]))),
        int(max(1, min(normalized_shape[5], chunks[5]))),
    )


def _component_matches_shape_chunks(
    *,
    root: zarr.hierarchy.Group,
    component: str,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    chunks_tpczyx: tuple[int, int, int, int, int, int],
) -> bool:
    """Return whether an existing component matches shape/chunks exactly."""
    try:
        array = root[str(component)]
    except Exception:
        return False
    array_shape = tuple(int(size) for size in tuple(getattr(array, "shape", tuple())))
    if array_shape != tuple(int(v) for v in shape_tpczyx):
        return False
    array_chunks_raw = getattr(array, "chunks", None)
    if array_chunks_raw is None:
        return False
    array_chunks = tuple(int(size) for size in tuple(array_chunks_raw))
    return array_chunks == tuple(int(v) for v in chunks_tpczyx)


def _estimate_chunk_region_count_tpczyx(
    *,
    shape_tpczyx: Sequence[int],
    chunks_tpczyx: Sequence[int],
) -> int:
    """Estimate chunk-region count for one canonical ``(t,p,c,z,y,x)`` array.

    Parameters
    ----------
    shape_tpczyx : sequence[int]
        Array shape in canonical axis order.
    chunks_tpczyx : sequence[int]
        Chunk shape in canonical axis order.

    Returns
    -------
    int
        Estimated chunk-region count. Returns ``0`` for invalid payloads.
    """
    if len(tuple(shape_tpczyx)) != 6 or len(tuple(chunks_tpczyx)) != 6:
        return 0
    region_counts = [
        int(max(1, math.ceil(float(max(1, int(size))) / float(max(1, int(chunk))))))
        for size, chunk in zip(shape_tpczyx, chunks_tpczyx, strict=False)
    ]
    return int(math.prod(region_counts))


def _write_display_pyramid_level_with_progress(
    *,
    level_array: da.Array,
    zarr_path: Union[str, Path],
    level_component: str,
    source_component: str,
    level_index: int,
    total_levels: int,
    absolute_factors_tpczyx: Sequence[int],
    level_shape_tpczyx: Sequence[int],
    level_chunks_tpczyx: Sequence[int],
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 100,
) -> None:
    """Write one display-pyramid level and emit incremental progress updates.

    Parameters
    ----------
    level_array : dask.array.Array
        Prepared downsampled level data.
    zarr_path : str or pathlib.Path
        Analysis-store path.
    level_component : str
        Target level component path.
    source_component : str
        Source component path used to derive this level.
    level_index : int
        One-based level index currently being written.
    total_levels : int
        Total number of generated levels.
    absolute_factors_tpczyx : sequence[int]
        Absolute level factors in canonical order.
    level_shape_tpczyx : sequence[int]
        Target level shape in canonical order.
    level_chunks_tpczyx : sequence[int]
        Target level chunks in canonical order.
    progress_callback : callable, optional
        Callback receiving ``(percent, message)`` updates.
    progress_start : int, default=0
        Start percent for this level write stage.
    progress_end : int, default=100
        End percent for this level write stage.

    Returns
    -------
    None
        Writes the level into the configured Zarr/N5 store.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized_shape = tuple(int(max(1, int(v))) for v in level_shape_tpczyx)
    normalized_chunks = tuple(int(max(1, int(v))) for v in level_chunks_tpczyx)
    estimated_chunks = _estimate_chunk_region_count_tpczyx(
        shape_tpczyx=normalized_shape,
        chunks_tpczyx=normalized_chunks,
    )

    write_graph = da.to_zarr(
        level_array,
        str(zarr_path),
        component=level_component,
        overwrite=True,
        compute=False,
    )
    total_tasks = max(1, int(len(getattr(write_graph, "dask", {}))))
    progress_start_int = int(progress_start)
    progress_end_int = int(max(progress_start_int, int(progress_end)))
    progress_span = max(1, progress_end_int - progress_start_int)

    _LOGGER.info(
        "[display_pyramid] writing level %d/%d component=%s source=%s "
        "shape=%s chunks=%s estimated_chunks=%d tasks=%d factors_tpczyx=%s",
        int(level_index),
        int(total_levels),
        str(level_component),
        str(source_component),
        normalized_shape,
        normalized_chunks,
        int(estimated_chunks),
        int(total_tasks),
        [int(value) for value in absolute_factors_tpczyx],
    )
    _emit(
        progress_start_int,
        "Writing pyramid level "
        f"{int(level_index)}/{int(total_levels)} "
        f"(estimated_chunks={int(estimated_chunks):,})",
    )

    if progress_callback is None or total_tasks <= 1:
        da.compute(write_graph)
        _emit(
            progress_end_int,
            f"Wrote pyramid level {int(level_index)}/{int(total_levels)}",
        )
        return

    class _TaskProgressCallback(Callback):
        """Throttle task-level updates for UI-friendly progress reporting."""

        def __init__(self) -> None:
            super().__init__()
            self._completed = 0
            self._last_emitted_completed = 0
            self._last_emitted_time = 0.0

        def _posttask(
            self,
            key: object,
            result: object,
            dsk: object,
            state: object,
            worker_id: object,
        ) -> None:
            del key, result, dsk, state, worker_id
            self._completed += 1
            now = float(time.monotonic())
            due_by_count = (
                int(self._completed) - int(self._last_emitted_completed)
            ) >= int(_DISPLAY_PYRAMID_BUILD_PROGRESS_TASK_STEP)
            due_by_time = (float(now) - float(self._last_emitted_time)) >= float(
                _DISPLAY_PYRAMID_BUILD_PROGRESS_MIN_INTERVAL_SECONDS
            )
            is_final = int(self._completed) >= int(total_tasks)
            if not is_final and not due_by_count and not due_by_time:
                return

            fraction = max(
                0.0,
                min(1.0, float(self._completed) / float(max(1, int(total_tasks)))),
            )
            mapped_percent = progress_start_int + int(round(fraction * progress_span))
            _emit(
                mapped_percent,
                "Writing pyramid level "
                f"{int(level_index)}/{int(total_levels)} "
                f"(tasks={int(self._completed):,}/{int(total_tasks):,})",
            )
            self._last_emitted_completed = int(self._completed)
            self._last_emitted_time = float(now)

    with _TaskProgressCallback():
        da.compute(write_graph)

    _emit(
        progress_end_int,
        f"Wrote pyramid level {int(level_index)}/{int(total_levels)}",
    )


def _display_pyramid_level_component(
    *,
    source_component: str,
    level: int,
) -> str:
    """Return canonical component path for one generated display-pyramid level.

    Parameters
    ----------
    source_component : str
        Base source component path.
    level : int
        Pyramid level index (must be >= 1).

    Returns
    -------
    str
        Target component path for this level.

    Raises
    ------
    ValueError
        If ``level`` is less than 1.

    Notes
    -----
    Raw data keeps canonical root-level ``data_pyramid`` naming to remain
    compatible with existing analysis readers. Derived components are written
    alongside their source at ``<source_component>_pyramid/level_<n>``.
    """
    level_index = int(level)
    if level_index < 1:
        raise ValueError("display-pyramid level index must be >= 1.")
    component = str(source_component).strip() or "data"
    if component == "data":
        return f"data_pyramid/level_{level_index}"
    return f"{component}_pyramid/level_{level_index}"


def _is_source_pyramid_layout_compatible(
    *,
    source_component: str,
    source_components: Sequence[str],
) -> bool:
    """Return whether a discovered multiscale set follows canonical layout.

    Parameters
    ----------
    source_component : str
        Base source component path.
    source_components : sequence[str]
        Ordered component paths discovered for multiscale rendering.

    Returns
    -------
    bool
        ``True`` when all levels follow canonical source-adjacent pyramid
        component naming.
    """
    components = [str(item).strip() for item in source_components]
    if not components:
        return False

    base_component = str(source_component).strip() or "data"
    if components[0] != base_component:
        return False

    for level_index, component in enumerate(components[1:], start=1):
        expected = _display_pyramid_level_component(
            source_component=base_component,
            level=level_index,
        )
        if str(component).strip() != expected:
            return False
    return True


def _collect_existing_multiscale_components(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
) -> tuple[str, ...]:
    """Collect existing display-pyramid component paths for one source component.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened Zarr root group.
    source_component : str
        Base source component.

    Returns
    -------
    tuple[str, ...]
        Ordered existing level components. Always includes base component.

    Raises
    ------
    ValueError
        If source component is missing or not canonical 6D.
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

    candidates: list[str] = [str(source_component)]
    source_attrs = dict(source_array.attrs)
    display_levels = source_attrs.get(_DISPLAY_PYRAMID_LEVELS_ATTR)
    if isinstance(display_levels, (tuple, list)):
        candidates.extend(str(item) for item in display_levels)
    source_levels = source_attrs.get("pyramid_levels")
    if isinstance(source_levels, (tuple, list)):
        candidates.extend(str(item) for item in source_levels)
    viz_levels = source_attrs.get(_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR)
    if isinstance(viz_levels, (tuple, list)):
        candidates.extend(str(item) for item in viz_levels)

    root_attrs = dict(root.attrs)
    root_level_map = root_attrs.get(_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if isinstance(root_level_map, Mapping):
        mapped = root_level_map.get(str(source_component))
        if isinstance(mapped, (tuple, list)):
            candidates.extend(str(item) for item in mapped)
    root_level_map = root_attrs.get(_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if isinstance(root_level_map, Mapping):
        mapped = root_level_map.get(str(source_component))
        if isinstance(mapped, (tuple, list)):
            candidates.extend(str(item) for item in mapped)
    if str(source_component) == "data":
        root_levels = root_attrs.get("data_pyramid_levels")
        if isinstance(root_levels, (tuple, list)):
            candidates.extend(str(item) for item in root_levels)

    ordered: list[str] = []
    for component in candidates:
        text = str(component).strip()
        if not text or text in ordered:
            continue
        try:
            array = root[text]
        except Exception:
            continue
        if len(tuple(array.shape)) != 6:
            continue
        ordered.append(text)

    return tuple(ordered) if ordered else (str(source_component),)


def _build_visualization_multiscale_components(
    *,
    zarr_path: Union[str, Path],
    root: zarr.hierarchy.Group,
    source_component: str,
    level_factors_tpczyx: tuple[tuple[int, int, int, int, int, int], ...],
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 30,
    progress_end: int = 68,
) -> tuple[str, ...]:
    """Materialize reusable display-pyramid levels for one source component.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Analysis store path.
    root : zarr.hierarchy.Group
        Opened root group (write-capable).
    source_component : str
        Base component path.
    level_factors_tpczyx : tuple[tuple[int, int, int, int, int, int], ...]
        Absolute level factors including base level.
    progress_callback : callable, optional
        Callback receiving ``(percent, message)`` updates while levels are
        generated.
    progress_start : int, default=30
        Start percent for level-generation progress.
    progress_end : int, default=68
        End percent for level-generation progress.

    Returns
    -------
    tuple[str, ...]
        Ordered components including generated levels.
    """
    source_array = root[str(source_component)]
    source_chunks = (
        tuple(source_array.chunks) if source_array.chunks is not None else None
    )
    source_dtype = np.dtype(source_array.dtype)
    level_paths: list[str] = [str(source_component)]
    factor_payload: list[list[int]] = [
        [int(value) for value in level_factors_tpczyx[0]]
    ]
    prior_component = str(source_component)
    prior_factors = tuple(int(value) for value in level_factors_tpczyx[0])
    total_generated_levels = max(1, int(len(level_factors_tpczyx) - 1))
    progress_start_int = int(progress_start)
    progress_end_int = int(max(progress_start_int, int(progress_end)))
    progress_span = max(0, int(progress_end_int - progress_start_int))

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    for level_index, absolute_factors in enumerate(level_factors_tpczyx[1:], start=1):
        level_progress_start = progress_start_int + int(
            ((int(level_index) - 1) / float(total_generated_levels))
            * float(progress_span)
        )
        level_progress_end = progress_start_int + int(
            (int(level_index) / float(total_generated_levels)) * float(progress_span)
        )
        all_relative = all(
            int(current) % int(previous) == 0
            for current, previous in zip(absolute_factors, prior_factors, strict=False)
        )
        if all_relative:
            relative_factors = (
                int(absolute_factors[0] // prior_factors[0]),
                int(absolute_factors[1] // prior_factors[1]),
                int(absolute_factors[2] // prior_factors[2]),
                int(absolute_factors[3] // prior_factors[3]),
                int(absolute_factors[4] // prior_factors[4]),
                int(absolute_factors[5] // prior_factors[5]),
            )
            source_level_component = str(prior_component)
            downsample_factors = relative_factors
        else:
            source_level_component = str(source_component)
            downsample_factors = (
                int(absolute_factors[0]),
                int(absolute_factors[1]),
                int(absolute_factors[2]),
                int(absolute_factors[3]),
                int(absolute_factors[4]),
                int(absolute_factors[5]),
            )

        level_component = _display_pyramid_level_component(
            source_component=str(source_component),
            level=int(level_index),
        )
        source_level = da.from_zarr(str(zarr_path), component=source_level_component)
        downsampled = _downsample_tpczyx_by_stride(source_level, downsample_factors)
        level_shape = tuple(int(size) for size in tuple(downsampled.shape))
        level_chunks = _resolve_level_chunks_tpczyx(
            source_chunks=source_chunks,
            level_shape=level_shape,
            level_array=downsampled,
        )

        if not _component_matches_shape_chunks(
            root=root,
            component=level_component,
            shape_tpczyx=(
                int(level_shape[0]),
                int(level_shape[1]),
                int(level_shape[2]),
                int(level_shape[3]),
                int(level_shape[4]),
                int(level_shape[5]),
            ),
            chunks_tpczyx=level_chunks,
        ):
            root.create_dataset(
                name=level_component,
                shape=level_shape,
                chunks=level_chunks,
                dtype=source_dtype.name,
                overwrite=True,
            )
            _write_display_pyramid_level_with_progress(
                level_array=downsampled,
                zarr_path=zarr_path,
                level_component=level_component,
                source_component=source_level_component,
                level_index=int(level_index),
                total_levels=int(total_generated_levels),
                absolute_factors_tpczyx=absolute_factors,
                level_shape_tpczyx=level_shape,
                level_chunks_tpczyx=level_chunks,
                progress_callback=progress_callback,
                progress_start=int(level_progress_start),
                progress_end=int(level_progress_end),
            )
        else:
            estimated_chunks = _estimate_chunk_region_count_tpczyx(
                shape_tpczyx=level_shape,
                chunks_tpczyx=level_chunks,
            )
            _LOGGER.info(
                "[display_pyramid] reusing existing level %d/%d component=%s "
                "shape=%s chunks=%s estimated_chunks=%d factors_tpczyx=%s",
                int(level_index),
                int(total_generated_levels),
                str(level_component),
                tuple(int(v) for v in level_shape),
                tuple(int(v) for v in level_chunks),
                int(estimated_chunks),
                [int(value) for value in absolute_factors],
            )
            _emit(
                int(level_progress_end),
                f"Reusing existing pyramid level {int(level_index)}/{int(total_generated_levels)}",
            )

        root[level_component].attrs.update(
            {
                "axes": ["t", "p", "c", "z", "y", "x"],
                "pyramid_level": int(level_index),
                "downsample_factors_tpczyx": [int(value) for value in absolute_factors],
                "chunk_shape_tpczyx": [int(value) for value in level_chunks],
                "source_component": str(source_level_component),
                "generated_by": "display_pyramid",
            }
        )

        level_paths.append(level_component)
        factor_payload.append([int(value) for value in absolute_factors])
        prior_component = level_component
        prior_factors = tuple(int(value) for value in absolute_factors)

    root[str(source_component)].attrs[_DISPLAY_PYRAMID_LEVELS_ATTR] = [
        str(item) for item in level_paths
    ]
    root[str(source_component)].attrs[_DISPLAY_PYRAMID_FACTORS_ATTR] = [
        list(row) for row in factor_payload
    ]
    root[str(source_component)].attrs[_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR] = [
        str(item) for item in level_paths
    ]
    root[str(source_component)].attrs[_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR] = [
        list(row) for row in factor_payload
    ]

    component_map = root.attrs.get(_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if not isinstance(component_map, dict):
        component_map = {}
    component_map[str(source_component)] = [str(item) for item in level_paths]
    root.attrs[_DISPLAY_PYRAMID_ROOT_MAP_ATTR] = _sanitize_metadata_value(component_map)
    legacy_component_map = root.attrs.get(_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if not isinstance(legacy_component_map, dict):
        legacy_component_map = {}
    legacy_component_map[str(source_component)] = [str(item) for item in level_paths]
    root.attrs[_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR] = _sanitize_metadata_value(
        legacy_component_map
    )
    return tuple(str(item) for item in level_paths)


def _resolve_multiscale_components(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
    use_multiscale: bool,
    multiscale_policy: str,
) -> tuple[tuple[str, ...], str]:
    """Resolve prepared source multiscale levels without viewer-time mutation.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened root group.
    source_component : str
        Base source component path.
    use_multiscale : bool
        Global visualization multiscale toggle.
    multiscale_policy : str
        Layer policy (``inherit``, ``require``, or ``off``).

    Returns
    -------
    tuple[tuple[str, ...], str]
        Ordered component paths and effective multiscale status.

    Raises
    ------
    ValueError
        If source component is invalid, or ``require`` is requested without
        available prepared multiscale levels.
    """
    policy = str(multiscale_policy).strip().lower() or "inherit"
    if policy == "auto_build":
        policy = "inherit"
    if policy not in _VOLUME_LAYER_MULTISCALE_POLICIES:
        policy = "inherit"

    existing = _collect_existing_multiscale_components(
        root=root,
        source_component=str(source_component),
    )

    if policy == "off":
        return (str(source_component),), "off"

    if policy == "inherit" and not bool(use_multiscale):
        return (str(source_component),), "off"

    if len(existing) > 1:
        return tuple(str(item) for item in existing), "existing"

    if policy == "require":
        raise ValueError(
            f"Visualization layer '{source_component}' requires multiscale levels, "
            "but no pyramid was found."
        )

    return (str(source_component),), "single_scale"


def _resolve_volume_layers(
    *,
    root: zarr.hierarchy.Group,
    parameters: Mapping[str, Any],
) -> list[ResolvedVolumeLayer]:
    """Resolve normalized runtime layer rows into concrete volume layers.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened root group.
    parameters : mapping[str, Any]
        Normalized visualization parameters.

    Returns
    -------
    list[ResolvedVolumeLayer]
        Resolved layer rows ready for napari rendering.
    """
    rows = _normalize_visualization_volume_layers(parameters.get("volume_layers", []))
    if not rows:
        default_source = str(parameters.get("input_source", "data")).strip() or "data"
        rows = [
            {
                "component": default_source,
                "name": "",
                "layer_type": "image",
                "channels": [],
                "visible": None,
                "opacity": None,
                "blending": "",
                "colormap": "",
                "rendering": "",
                "multiscale_policy": (
                    "inherit" if bool(parameters.get("use_multiscale", True)) else "off"
                ),
            }
        ]

    resolved: list[ResolvedVolumeLayer] = []
    use_multiscale = bool(parameters.get("use_multiscale", True))
    for row in rows:
        component = str(row.get("component", "")).strip()
        if not component:
            continue
        source_components, multiscale_status = _resolve_multiscale_components(
            root=root,
            source_component=component,
            use_multiscale=use_multiscale,
            multiscale_policy=str(row.get("multiscale_policy", "inherit")),
        )
        resolved.append(
            ResolvedVolumeLayer(
                component=component,
                source_components=tuple(str(item) for item in source_components),
                layer_type=str(row.get("layer_type", "image")).strip().lower()
                or "image",
                channels=tuple(
                    int(index)
                    for index in _normalize_visualization_channels(row.get("channels"))
                ),
                visible=_coerce_optional_bool(row.get("visible")),
                opacity=_coerce_optional_unit_interval_float(row.get("opacity")),
                blending=str(row.get("blending", "")).strip().lower(),
                colormap=str(row.get("colormap", "")).strip(),
                rendering=str(row.get("rendering", "")).strip().lower(),
                name=str(row.get("name", "")).strip(),
                multiscale_policy=(
                    str(row.get("multiscale_policy", "inherit")).strip().lower()
                    or "inherit"
                ),
                multiscale_status=str(multiscale_status),
            )
        )
    return resolved


def _serialize_resolved_volume_layers(
    volume_layers: Sequence[ResolvedVolumeLayer],
) -> list[Dict[str, Any]]:
    """Serialize resolved volume layers for metadata/provenance payloads."""
    return [
        {
            "component": str(layer.component),
            "source_components": [str(item) for item in layer.source_components],
            "layer_type": str(layer.layer_type),
            "channels": [int(value) for value in layer.channels],
            "visible": layer.visible,
            "opacity": layer.opacity,
            "blending": str(layer.blending),
            "colormap": str(layer.colormap),
            "rendering": str(layer.rendering),
            "name": str(layer.name),
            "multiscale_policy": str(layer.multiscale_policy),
            "multiscale_status": str(layer.multiscale_status),
        }
        for layer in volume_layers
    ]


def _coerce_contrast_limit_pair(value: Any) -> Optional[tuple[float, float]]:
    """Parse one contrast-limit pair into finite ascending floats."""
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high <= low:
        high = low + 1.0
    return (float(low), float(high))


def _coerce_contrast_limits_by_channel(
    value: Any,
) -> Optional[tuple[tuple[float, float], ...]]:
    """Parse per-channel contrast-limit metadata."""
    if not isinstance(value, (tuple, list)):
        return None
    parsed: list[tuple[float, float]] = []
    for row in value:
        pair = _coerce_contrast_limit_pair(row)
        if pair is None:
            return None
        parsed.append(pair)
    return tuple(parsed)


def _default_contrast_limits_for_dtype(dtype: Any) -> tuple[float, float]:
    """Return a no-compute contrast fallback for one dtype."""
    normalized = np.dtype(dtype)
    if np.issubdtype(normalized, np.bool_):
        return (0.0, 1.0)
    if np.issubdtype(normalized, np.integer):
        info = np.iinfo(normalized)
        low = float(info.min)
        high = float(info.max)
        if high <= low:
            high = low + 1.0
        return (low, high)
    return (0.0, 1.0)


def _load_display_contrast_limits_by_channel(
    *,
    root: Optional[zarr.hierarchy.Group],
    source_component: str,
) -> Optional[tuple[tuple[float, float], ...]]:
    """Load stored per-channel display contrast metadata when available."""
    if root is None:
        return None
    try:
        attrs = dict(root[str(source_component)].attrs)
    except Exception:
        return None
    return _coerce_contrast_limits_by_channel(attrs.get(_DISPLAY_CONTRAST_LIMITS_ATTR))


def _save_display_contrast_metadata(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
    contrast_limits_by_channel: Sequence[Sequence[float]],
    contrast_source_component: Optional[str] = None,
    percentiles: tuple[float, float] = (1.0, 95.0),
) -> None:
    """Persist per-channel display contrast metadata on the source component.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Opened root group.
    source_component : str
        Source component that stores display contrast metadata.
    contrast_limits_by_channel : sequence[sequence[float]]
        Per-channel display limits.
    contrast_source_component : str, optional
        Component used to estimate percentiles. Defaults to
        ``source_component``.
    percentiles : tuple[float, float], default=(1.0, 95.0)
        Percentiles used for the computed limits.
    """
    source_attrs = root[str(source_component)].attrs
    source_attrs[_DISPLAY_CONTRAST_LIMITS_ATTR] = [
        [float(row[0]), float(row[1])] for row in contrast_limits_by_channel
    ]
    source_attrs[_DISPLAY_CONTRAST_PERCENTILES_ATTR] = [
        float(percentiles[0]),
        float(percentiles[1]),
    ]
    source_attrs[_DISPLAY_CONTRAST_LEVEL_SOURCE_ATTR] = str(
        contrast_source_component or source_component
    )


def _sample_channel_for_display_contrast(
    *,
    channel_data_tpzyx: da.Array,
    target_voxels: int,
) -> da.Array:
    """Sample one channel lazily to bound contrast-estimation compute cost.

    Parameters
    ----------
    channel_data_tpzyx : dask.array.Array
        Channel array in ``(t, p, z, y, x)`` order.
    target_voxels : int
        Upper-bound target for sampled voxel count.

    Returns
    -------
    dask.array.Array
        Lazily sliced sample preserving the original chunk graph.
    """
    shape_tpzyx = tuple(int(value) for value in tuple(channel_data_tpzyx.shape))
    if len(shape_tpzyx) != 5:
        return channel_data_tpzyx

    target = max(1, int(target_voxels))
    total_voxels = max(1, int(np.prod(shape_tpzyx, dtype=np.int64)))
    if total_voxels <= target:
        return channel_data_tpzyx

    # Favor spatial downsampling first to retain time/position diversity.
    spatial_stride = int(
        max(
            1,
            math.ceil((float(total_voxels) / float(target)) ** (1.0 / 3.0)),
        )
    )
    sampled = channel_data_tpzyx[
        :,
        :,
        ::spatial_stride,
        ::spatial_stride,
        ::spatial_stride,
    ]
    sampled_voxels = max(1, int(np.prod(sampled.shape, dtype=np.int64)))
    if sampled_voxels <= target:
        return sampled

    tp_stride = int(max(1, math.ceil(math.sqrt(float(sampled_voxels) / float(target)))))
    return sampled[::tp_stride, ::tp_stride, :, :, :]


def _compute_display_contrast_limits_by_channel(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    sampling_component: Optional[str] = None,
    channel_count: int,
    low_percentile: float = 1.0,
    high_percentile: float = 95.0,
    sampling_target_voxels: int = _DISPLAY_CONTRAST_SAMPLE_TARGET_VOXELS,
) -> tuple[tuple[float, float], ...]:
    """Compute fixed display percentiles for each channel of one source array.

    Notes
    -----
    To avoid expensive global reshapes and percentile shuffles, this computes
    percentiles on a lazily strided sample from one component (typically the
    coarsest prepared display-pyramid level).
    """
    source_component_text = str(source_component).strip() or "data"
    sampling_component_text = str(sampling_component or "").strip()
    contrast_component = sampling_component_text or source_component_text
    source = da.from_zarr(str(zarr_path), component=contrast_component)
    source_shape = tuple(int(value) for value in tuple(source.shape))
    available_channels = int(source_shape[2]) if len(source_shape) >= 3 else 0
    contrast_limits: list[tuple[float, float]] = []
    target_channels = max(0, int(channel_count))
    if available_channels <= 0:
        fallback = _default_contrast_limits_for_dtype(source.dtype)
        return tuple(fallback for _ in range(target_channels))

    for channel_index in range(max(0, int(channel_count))):
        clamped_channel_index = int(min(channel_index, available_channels - 1))
        channel_data = source[:, :, clamped_channel_index, :, :, :]
        sampled_channel = _sample_channel_for_display_contrast(
            channel_data_tpzyx=channel_data,
            target_voxels=sampling_target_voxels,
        )
        try:
            sample_np = np.asarray(sampled_channel.compute(), dtype=np.float32)
            if sample_np.size <= 0:
                pair = None
            else:
                percentiles = np.percentile(
                    sample_np,
                    [float(low_percentile), float(high_percentile)],
                )
                pair = _coerce_contrast_limit_pair(percentiles)
        except Exception:
            pair = None
        if pair is None:
            pair = _default_contrast_limits_for_dtype(source.dtype)
        contrast_limits.append(pair)
    return tuple(contrast_limits)


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
        Parsed rows with ``x``, ``y``, ``z``, ``theta``, and ``f`` values.
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
                "f": _value("F", 4),
            }
        )
    return parsed_rows


def _load_spatial_calibration(
    root_attrs: Mapping[str, Any],
) -> SpatialCalibrationConfig:
    """Load store-level spatial calibration from root attrs.

    Parameters
    ----------
    root_attrs : mapping[str, Any]
        Root Zarr attributes.

    Returns
    -------
    SpatialCalibrationConfig
        Parsed store calibration. Missing attrs resolve to identity.
    """
    return spatial_calibration_from_dict(root_attrs.get("spatial_calibration"))


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
    delta_world_x: float,
    delta_world_y: float,
    delta_world_z: float,
    delta_theta_deg: float,
    scale_tczyx: Sequence[float],
) -> np.ndarray:
    """Build a 5D napari affine matrix for one position offset.

    Parameters
    ----------
    delta_world_x : float
        World X translation delta relative to reference position.
    delta_world_y : float
        World Y translation delta relative to reference position.
    delta_world_z : float
        World Z translation delta relative to reference position.
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
    affine[2, 5] = float(delta_world_z)
    affine[3, 5] = float(delta_world_y)
    affine[4, 5] = float(delta_world_x)
    return affine


def _resolve_world_axis_delta(
    *,
    row: Mapping[str, float],
    reference: Mapping[str, float],
    binding: str,
) -> float:
    """Resolve one world-axis translation delta from stage coordinates.

    Parameters
    ----------
    row : mapping[str, float]
        Current multiposition row.
    reference : mapping[str, float]
        Reference multiposition row.
    binding : str
        Canonical spatial-calibration binding for the world axis.

    Returns
    -------
    float
        Translation delta in stage/world units.
    """
    if binding == "none":
        return 0.0
    sign = -1.0 if binding.startswith("-") else 1.0
    source_axis = binding[1:]
    return sign * float(row[source_axis] - reference[source_axis])


def _resolve_position_affines_tczyx(
    *,
    root_attrs: Mapping[str, Any],
    selected_positions: Sequence[int],
    scale_tczyx: Sequence[float],
) -> tuple[dict[int, np.ndarray], list[dict[str, float]], SpatialCalibrationConfig]:
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
    tuple[dict[int, numpy.ndarray], list[dict[str, float]], SpatialCalibrationConfig]
        Mapping of position index to affine matrix, parsed stage rows, and the
        effective store calibration.
    """
    affines: dict[int, np.ndarray] = {
        int(index): np.eye(6, dtype=np.float64) for index in selected_positions
    }
    spatial_calibration = _load_spatial_calibration(root_attrs)
    stage_rows = _load_multiposition_stage_rows(root_attrs)
    if not stage_rows:
        return affines, [], spatial_calibration

    reference = stage_rows[0]
    stage_axis_map = spatial_calibration.stage_axis_map_by_world_axis()
    for position_index in selected_positions:
        idx = int(position_index)
        if idx < 0 or idx >= len(stage_rows):
            continue
        row = stage_rows[idx]
        affines[idx] = _build_position_affine_tczyx(
            delta_world_x=_resolve_world_axis_delta(
                row=row,
                reference=reference,
                binding=stage_axis_map["x"],
            ),
            delta_world_y=_resolve_world_axis_delta(
                row=row,
                reference=reference,
                binding=stage_axis_map["y"],
            ),
            delta_world_z=_resolve_world_axis_delta(
                row=row,
                reference=reference,
                binding=stage_axis_map["z"],
            ),
            delta_theta_deg=float(row["theta"] - reference["theta"]),
            scale_tczyx=scale_tczyx,
        )
    return affines, stage_rows, spatial_calibration


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
    volume_layers: Sequence[ResolvedVolumeLayer],
    selected_positions: Sequence[int],
    points_by_position: Mapping[int, np.ndarray],
    point_properties_by_position: Mapping[int, Mapping[str, np.ndarray]],
    position_affines_tczyx: Mapping[int, np.ndarray],
    axis_labels: Sequence[str],
    scale_tczyx: Sequence[float],
    image_metadata: Mapping[str, Any],
    points_metadata: Mapping[str, Any],
    require_gpu_rendering: bool,
    capture_keyframes: bool,
    keyframe_manifest_path: Optional[Union[str, Path]],
    keyframe_layer_overrides: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Open a napari viewer and render image/labels + optional overlays.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    volume_layers : sequence[ResolvedVolumeLayer]
        Resolved source layers to render in order.
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
    require_gpu_rendering : bool
        Whether to fail when napari does not report a GPU-backed OpenGL
        renderer.
    capture_keyframes : bool
        Whether interactive keyframe capture hotkeys are enabled.
    keyframe_manifest_path : str or pathlib.Path, optional
        Destination JSON path used for writing keyframe selections.
    keyframe_layer_overrides : sequence of mapping[str, Any]
        Optional per-layer override rows, including annotation text.

    Returns
    -------
    dict[str, Any]
        Viewer-capture summary with ``keyframe_manifest_path`` and
        ``keyframe_count`` entries.

    Raises
    ------
    ImportError
        If napari is unavailable.
    RuntimeError
        If ``require_gpu_rendering`` is enabled and the active OpenGL
        renderer is not confirmed as GPU-backed.
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

    scale = tuple(float(value) for value in scale_tczyx)
    scale_root: Optional[zarr.hierarchy.Group] = None
    scale_root_attrs: Dict[str, Any] = {}
    scale_source_experiment_raw: Dict[str, Any] = {}
    try:
        scale_root = zarr.open_group(str(zarr_path), mode="r")
        scale_root_attrs = dict(scale_root.attrs)
        scale_source_experiment_raw = _load_source_experiment_raw(scale_root_attrs)
    except Exception:
        scale_root = None
    layer_scale_cache: dict[str, tuple[float, float, float, float, float]] = {}
    layer_display_contrast_cache: dict[str, tuple[tuple[float, float], ...]] = {}

    def _resolve_layer_scale_tczyx(
        component: str,
    ) -> tuple[float, float, float, float, float]:
        """Resolve per-layer ``(t, c, z, y, x)`` scale for napari rendering."""
        key = str(component).strip() or "data"
        cached = layer_scale_cache.get(key)
        if cached is not None:
            return cached
        if scale_root is None:
            layer_scale_cache[key] = scale
            return scale

        source_attrs: Dict[str, Any]
        source_shape: Optional[tuple[int, int, int, int, int, int]]
        try:
            source_array = scale_root[key]
            source_attrs = dict(source_array.attrs)
            source_shape = tuple(int(value) for value in tuple(source_array.shape))
        except Exception:
            layer_scale_cache[key] = scale
            return scale

        resolved_scale = _extract_scale_tczyx_from_attrs(
            root_attrs={},
            source_attrs=source_attrs,
        )
        if resolved_scale is None and key == "data":
            resolved_scale = _extract_scale_tczyx_from_attrs(
                root_attrs=scale_root_attrs,
                source_attrs=source_attrs,
            )
        if resolved_scale is None and key == "data":
            resolved_scale = _extract_scale_tczyx_from_navigate_raw(
                scale_source_experiment_raw
            )

        if resolved_scale is None:
            base_scale = scale
            base_shape: Optional[tuple[int, int, int, int, int, int]] = None
            try:
                base_array = scale_root["data"]
                base_shape = tuple(int(value) for value in tuple(base_array.shape))
                base_scale = (
                    _extract_scale_tczyx_from_attrs(
                        root_attrs=scale_root_attrs,
                        source_attrs=dict(base_array.attrs),
                    )
                    or _extract_scale_tczyx_from_navigate_raw(
                        scale_source_experiment_raw
                    )
                    or scale
                )
            except Exception:
                base_shape = None

            if (
                base_shape is not None
                and source_shape is not None
                and len(base_shape) == 6
                and len(source_shape) == 6
                and all(int(size) > 0 for size in source_shape[3:])
                and all(int(size) > 0 for size in base_shape[3:])
            ):
                factors: list[float] = []
                for base_dim, source_dim in zip(
                    base_shape[3:],
                    source_shape[3:],
                    strict=False,
                ):
                    if int(source_dim) <= 0 or int(base_dim) <= 0:
                        factors.append(1.0)
                        continue
                    ratio = float(base_dim) / float(source_dim)
                    rounded = int(round(ratio))
                    if rounded >= 1 and abs(ratio - float(rounded)) <= 1e-6:
                        factors.append(float(rounded))
                    else:
                        factors.append(1.0)
                resolved_scale = (
                    float(base_scale[0]),
                    float(base_scale[1]),
                    float(base_scale[2]) * float(factors[0]),
                    float(base_scale[3]) * float(factors[1]),
                    float(base_scale[4]) * float(factors[2]),
                )

        if resolved_scale is None:
            resolved_scale = scale
        normalized = _normalize_scale_tczyx(
            tuple(float(value) for value in resolved_scale)
        )
        layer_scale_cache[key] = normalized
        return normalized

    def _resolve_layer_contrast_limits_by_channel(
        component: str,
        channel_count: int,
        dtype: Any,
    ) -> tuple[tuple[float, float], ...]:
        """Resolve display contrast metadata without viewer-time reductions."""
        key = str(component).strip() or "data"
        cached = layer_display_contrast_cache.get(key)
        if cached is None:
            cached = (
                _load_display_contrast_limits_by_channel(
                    root=scale_root,
                    source_component=key,
                )
                or tuple()
            )
            layer_display_contrast_cache[key] = cached
        if len(cached) >= int(channel_count):
            return tuple(cached[: int(channel_count)])
        fallback = _default_contrast_limits_for_dtype(dtype)
        return tuple(
            cached[index] if index < len(cached) else fallback
            for index in range(max(0, int(channel_count)))
        )

    requested_ndisplay = int(
        max(
            2,
            round(
                _safe_float(
                    image_metadata.get(
                        "viewer_ndisplay_requested",
                        image_metadata.get("viewer_ndisplay", 3),
                    ),
                    default=3.0,
                )
            ),
        )
    )
    effective_ndisplay = int(
        max(
            2,
            round(
                _safe_float(
                    image_metadata.get(
                        "viewer_ndisplay_effective",
                        image_metadata.get("viewer_ndisplay", requested_ndisplay),
                    ),
                    default=float(requested_ndisplay),
                )
            ),
        )
    )
    viewer = napari.Viewer(
        ndisplay=(3 if int(effective_ndisplay) >= 3 else 2),
        show=True,
    )
    axis_labels_tuple = tuple(str(label) for label in axis_labels)
    axis_labels_applied = False
    renderer_info = _probe_napari_opengl_renderer(viewer)
    renderer_summary = (
        f"{renderer_info.get('vendor', '')} {renderer_info.get('renderer', '')}".strip()
        or "unknown"
    )
    print(f"[visualization] OpenGL renderer: {renderer_summary}")
    software_renderer = bool(renderer_info.get("software_renderer", False))
    gpu_renderer = bool(renderer_info.get("gpu_renderer", False))
    if bool(require_gpu_rendering) and not gpu_renderer:
        if software_renderer:
            reason = (
                "Napari OpenGL renderer appears to be software-based "
                f"('{renderer_summary}')"
            )
        else:
            probe_error = str(renderer_info.get("error", "")).strip()
            if probe_error:
                reason = (
                    "Napari OpenGL renderer probe did not confirm GPU rendering "
                    f"('{renderer_summary}'; error: {probe_error})"
                )
            else:
                reason = (
                    "Napari OpenGL renderer could not be confirmed as GPU-backed "
                    f"('{renderer_summary}')"
                )
        raise RuntimeError(
            f"{reason}. Disable visualization parameter "
            "'require_gpu_rendering' to override."
        )

    manifest_path: Optional[Path] = None
    if bool(capture_keyframes):
        if keyframe_manifest_path is not None and str(keyframe_manifest_path).strip():
            manifest_path = Path(keyframe_manifest_path).expanduser().resolve()
        else:
            manifest_path = _resolve_keyframe_manifest_path(
                zarr_path=zarr_path,
                parameters={"capture_keyframes": True},
            )
    keyframes: list[Dict[str, Any]] = []
    normalized_layer_overrides = _normalize_keyframe_layer_overrides(
        keyframe_layer_overrides
    )
    if len(volume_layers) <= 0:
        raise ValueError("Visualization requires at least one resolved volume layer.")
    primary_layer = volume_layers[0]
    primary_source_component = str(primary_layer.component)
    primary_source_components = tuple(
        str(item) for item in primary_layer.source_components
    )
    serialized_volume_layers = _serialize_resolved_volume_layers(volume_layers)

    selected = tuple(int(index) for index in selected_positions)
    multi_position = len(selected) > 1
    for position_index in selected:
        affine_base = np.asarray(
            position_affines_tczyx.get(position_index, np.eye(6, dtype=np.float64)),
            dtype=np.float64,
        )
        for layer in volume_layers:
            layer_scale = _resolve_layer_scale_tczyx(str(layer.component))
            rendered_source_components = (
                tuple(str(item) for item in layer.source_components)
                if int(effective_ndisplay) < 3
                else (str(layer.component),)
            )
            level_arrays = [
                da.from_zarr(str(zarr_path), component=str(component))[
                    :, position_index, :, :, :, :
                ]
                for component in rendered_source_components
            ]
            if not level_arrays:
                continue
            display_level_arrays = tuple(level_arrays)
            display_scale = tuple(float(value) for value in layer_scale)
            display_factors_zyx = (1.0, 1.0, 1.0)
            display_strategy = "none"
            if not display_level_arrays:
                continue
            is_multiscale = (
                int(effective_ndisplay) < 3 and len(display_level_arrays) > 1
            )
            channel_count = max(1, int(display_level_arrays[0].shape[1]))
            requested_channels = tuple(
                int(index)
                for index in layer.channels
                if 0 <= int(index) < channel_count
            )
            channel_indices = requested_channels or tuple(range(channel_count))
            channel_opacity_default = _default_channel_opacity(len(channel_indices))
            effective_opacity = (
                float(layer.opacity)
                if layer.opacity is not None
                else float(channel_opacity_default)
            )
            effective_visibility = (
                bool(layer.visible) if layer.visible is not None else True
            )
            base_name = str(layer.name).strip() or str(layer.component)
            effective_blending = str(layer.blending).strip().lower()
            if not effective_blending:
                if str(layer.layer_type) == "labels":
                    effective_blending = "translucent"
                elif multi_position:
                    effective_blending = "translucent"
                else:
                    effective_blending = "additive"
            default_rendering = str(layer.rendering).strip().lower() or "attenuated_mip"
            colormaps = _channel_colormap_cycle(channel_count)
            contrast_limits_by_channel = _resolve_layer_contrast_limits_by_channel(
                str(layer.component),
                channel_count,
                display_level_arrays[0].dtype,
            )
            for channel_index in channel_indices:
                layer_data: Any
                if is_multiscale:
                    layer_data = [
                        level[:, channel_index : channel_index + 1, :, :, :]
                        for level in display_level_arrays
                    ]
                else:
                    layer_data = display_level_arrays[0][
                        :, channel_index : channel_index + 1, :, :, :
                    ]

                layer_affine = np.asarray(affine_base, dtype=np.float64).copy()
                layer_name = (
                    f"{base_name} (p={position_index}, c={channel_index})"
                    if (len(channel_indices) > 1 or channel_count > 1)
                    else f"{base_name} (p={position_index})"
                )
                layer_metadata = dict(image_metadata)
                layer_metadata["position_index"] = int(position_index)
                layer_metadata["channel_index"] = int(channel_index)
                layer_metadata["source_component"] = str(layer.component)
                layer_metadata["source_components"] = [
                    str(item) for item in rendered_source_components
                ]
                layer_metadata["available_source_components"] = [
                    str(item) for item in layer.source_components
                ]
                layer_metadata["layer_type"] = str(layer.layer_type)
                layer_metadata["multiscale_policy"] = str(layer.multiscale_policy)
                layer_metadata["multiscale_status"] = str(layer.multiscale_status)
                layer_metadata["volume_layers"] = serialized_volume_layers
                layer_metadata["scale_tczyx"] = [
                    float(value) for value in display_scale
                ]
                layer_metadata["display_downsample_strategy"] = str(display_strategy)
                layer_metadata["display_downsample_factors_zyx"] = [
                    float(display_factors_zyx[0]),
                    float(display_factors_zyx[1]),
                    float(display_factors_zyx[2]),
                ]

                if str(layer.layer_type) == "labels":
                    viewer.add_labels(
                        layer_data,
                        multiscale=is_multiscale,
                        name=layer_name,
                        scale=display_scale,
                        affine=layer_affine,
                        metadata={
                            str(key): value for key, value in layer_metadata.items()
                        },
                        opacity=float(
                            layer.opacity if layer.opacity is not None else 0.7
                        ),
                        blending=effective_blending,
                        visible=effective_visibility,
                    )
                else:
                    contrast_limits = contrast_limits_by_channel[int(channel_index)]
                    selected_colormap = str(layer.colormap).strip() or str(
                        colormaps[int(channel_index) % max(1, len(colormaps))]
                    )
                    layer_metadata["channel_colormap"] = selected_colormap
                    layer_metadata["contrast_limits"] = [
                        float(contrast_limits[0]),
                        float(contrast_limits[1]),
                    ]
                    viewer.add_image(
                        layer_data,
                        multiscale=is_multiscale,
                        name=layer_name,
                        scale=display_scale,
                        affine=layer_affine,
                        metadata={
                            str(key): value for key, value in layer_metadata.items()
                        },
                        colormap=selected_colormap,
                        opacity=float(effective_opacity),
                        blending=effective_blending,
                        projection_mode="max",
                        rendering=default_rendering,
                        contrast_limits=(
                            float(contrast_limits[0]),
                            float(contrast_limits[1]),
                        ),
                        visible=effective_visibility,
                    )

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

    def _notify(message: str) -> None:
        """Emit a keyframe-status message in logs and the napari UI.

        Parameters
        ----------
        message : str
            Status text.

        Returns
        -------
        None
            Viewer/logging side effects only.

        Raises
        ------
        None
            Notification failures are ignored.
        """
        text = str(message)
        print(f"[visualization] {text}")
        try:
            viewer.status = text
        except Exception:
            pass
        try:
            from napari.utils.notifications import show_info

            show_info(text)
        except Exception:
            pass

    def _persist_keyframes() -> None:
        """Persist the current keyframe manifest to disk.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Writes JSON payload when keyframe capture is enabled.

        Raises
        ------
        None
            Write failures are reported as viewer status and ignored.
        """
        if manifest_path is None:
            return
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "zarr_path": str(Path(zarr_path).expanduser().resolve()),
            "source_component": str(primary_source_component),
            "source_components": [str(item) for item in primary_source_components],
            "volume_layers": serialized_volume_layers,
            "renderer": dict(renderer_info),
            "selected_positions": [int(value) for value in selected],
            "axis_labels_tczyx": [str(label) for label in axis_labels_tuple],
            "scale_tczyx": [float(value) for value in scale],
            "viewer_type": (
                "3d"
                if int(
                    max(
                        2,
                        round(
                            _safe_float(
                                getattr(viewer.dims, "ndisplay", 3), default=3.0
                            )
                        ),
                    )
                )
                == 3
                else "2d"
            ),
            "keyframe_layer_overrides": list(normalized_layer_overrides),
            "keyframes": list(keyframes),
        }
        try:
            _write_keyframe_manifest(manifest_path=manifest_path, payload=payload)
        except OSError as exc:
            _notify(f"Keyframe manifest write failed: {exc}")

    if manifest_path is not None:
        _persist_keyframes()

        def _capture_keyframe(_viewer: Any = None) -> None:
            """Capture and persist one viewer keyframe.

            Parameters
            ----------
            _viewer : Any, optional
                Ignored napari callback argument.

            Returns
            -------
            None
                Updates in-memory and on-disk keyframe state.
            """
            del _viewer
            keyframe = _capture_viewer_keyframe(
                viewer=viewer,
                keyframe_index=len(keyframes),
                layer_overrides=normalized_layer_overrides,
            )
            keyframes.append(keyframe)
            _persist_keyframes()
            _notify(
                f"Captured keyframe {int(keyframe['index']) + 1} at {manifest_path}."
            )

        def _pop_keyframe(_viewer: Any = None) -> None:
            """Remove the most recent keyframe selection.

            Parameters
            ----------
            _viewer : Any, optional
                Ignored napari callback argument.

            Returns
            -------
            None
                Mutates in-memory and on-disk keyframe state.
            """
            del _viewer
            if not keyframes:
                _notify("No keyframes to remove.")
                return
            keyframes.pop()
            _persist_keyframes()
            _notify(f"Removed latest keyframe. Remaining: {len(keyframes)}.")

        bind_key = getattr(viewer, "bind_key", None)
        bound_hotkeys = 0
        if callable(bind_key):
            try:
                bind_key("k", _capture_keyframe, overwrite=True)
                bound_hotkeys += 1
            except TypeError:
                try:
                    bind_key("k")(_capture_keyframe)
                    bound_hotkeys += 1
                except Exception:
                    pass
            except Exception:
                pass

            try:
                bind_key("Shift-K", _pop_keyframe, overwrite=True)
                bound_hotkeys += 1
            except TypeError:
                try:
                    bind_key("Shift-K")(_pop_keyframe)
                    bound_hotkeys += 1
                except Exception:
                    pass
            except Exception:
                pass

        try:
            close_event = getattr(getattr(viewer, "events", None), "close", None)
            if close_event is not None and callable(
                getattr(close_event, "connect", None)
            ):
                close_event.connect(lambda _event=None: _persist_keyframes())
        except Exception:
            pass

        if bound_hotkeys > 0:
            _notify(
                "Keyframe capture enabled. Press K to capture and Shift-K to remove "
                f"the latest keyframe. Manifest: {manifest_path}."
            )
        else:
            _notify(
                "Keyframe capture manifest is enabled, but hotkeys could not be "
                f"registered. Manifest: {manifest_path}."
            )

    napari.run()
    _persist_keyframes()
    return {
        "keyframe_manifest_path": (
            str(manifest_path) if manifest_path is not None else None
        ),
        "keyframe_count": int(len(keyframes)),
        "renderer": dict(renderer_info),
    }


def _save_display_pyramid_metadata(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    source_components: Sequence[str],
    contrast_limits_by_channel: Sequence[Sequence[float]],
    contrast_source_component: str,
    reused_existing_levels: bool,
    run_id: Optional[str] = None,
) -> str:
    """Persist display-pyramid metadata in ``results/display_pyramid/latest``."""
    component = "results/display_pyramid/latest"
    root = zarr.open_group(str(zarr_path), mode="a")
    results_group = root.require_group("results")
    display_pyramid_group = results_group.require_group("display_pyramid")
    if "latest" in display_pyramid_group:
        del display_pyramid_group["latest"]
    latest_group = display_pyramid_group.create_group("latest")

    payload: Dict[str, Any] = {
        "source_component": str(source_component),
        "source_components": [str(item) for item in source_components],
        "display_pyramid_lookup": {
            "source_attr": _DISPLAY_PYRAMID_LEVELS_ATTR,
            "root_attr_map": _DISPLAY_PYRAMID_ROOT_MAP_ATTR,
            "legacy_source_attr": _LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR,
            "legacy_root_attr_map": _LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR,
        },
        "display_contrast_metadata_reference": {
            "limits_attr": _DISPLAY_CONTRAST_LIMITS_ATTR,
            "percentiles_attr": _DISPLAY_CONTRAST_PERCENTILES_ATTR,
        },
        "display_contrast_percentiles": [1.0, 95.0],
        "display_contrast_limits_by_channel": [
            [float(row[0]), float(row[1])] for row in contrast_limits_by_channel
        ],
        "display_contrast_source_component": str(contrast_source_component),
        "reused_existing_levels": bool(reused_existing_levels),
        "storage_policy": "latest_only",
        "run_id": run_id,
    }
    latest_group.attrs.update(payload)
    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="display_pyramid",
        component=component,
        run_id=run_id,
        metadata=payload,
    )
    return component


def run_display_pyramid_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional[Any] = None,
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> DisplayPyramidSummary:
    """Prepare reusable display pyramids and stored display contrast metadata."""

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    del client
    source_component = str(parameters.get("input_source", "data")).strip() or "data"
    force_rerun = bool(parameters.get("force_rerun", False))
    root = zarr.open_group(str(zarr_path), mode="a")
    try:
        source_array = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"Display pyramid input component '{source_component}' was not found."
        ) from exc
    if len(tuple(source_array.shape)) != 6:
        raise ValueError(
            "Display pyramid preparation requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    source_attrs = dict(source_array.attrs)
    root_attrs = dict(root.attrs)
    channel_count = max(1, int(tuple(source_array.shape)[2]))
    existing_components = _collect_existing_multiscale_components(
        root=root,
        source_component=source_component,
    )
    has_multiscale_levels = bool(len(existing_components) > 1)
    layout_compatible = _is_source_pyramid_layout_compatible(
        source_component=source_component,
        source_components=existing_components,
    )
    reused_existing_levels = bool(
        has_multiscale_levels and not force_rerun and layout_compatible
    )
    should_rebuild_for_layout = bool(
        has_multiscale_levels and not force_rerun and not layout_compatible
    )
    if reused_existing_levels:
        _emit(30, f"Reusing existing display pyramid for {source_component}")
        source_components = tuple(str(item) for item in existing_components)
    else:
        if should_rebuild_for_layout:
            _emit(
                15,
                "Existing display pyramid uses legacy layout; rebuilding source-adjacent levels.",
            )
        _emit(30, f"Preparing display pyramid for {source_component}")
        level_factors = _resolve_visualization_pyramid_factors_tpczyx(
            root_attrs=root_attrs,
            source_attrs=source_attrs,
        )
        source_components = _build_visualization_multiscale_components(
            zarr_path=zarr_path,
            root=root,
            source_component=source_component,
            level_factors_tpczyx=level_factors,
            progress_callback=_emit,
            progress_start=32,
            progress_end=68,
        )

    level_factors = _resolve_visualization_pyramid_factors_tpczyx(
        root_attrs=root_attrs,
        source_attrs=source_attrs,
    )
    persisted_factors = [
        list(row) for row in level_factors[: max(1, len(tuple(source_components)))]
    ]
    root[str(source_component)].attrs[_DISPLAY_PYRAMID_LEVELS_ATTR] = [
        str(item) for item in source_components
    ]
    root[str(source_component)].attrs[_DISPLAY_PYRAMID_FACTORS_ATTR] = list(
        persisted_factors
    )
    root[str(source_component)].attrs[_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR] = [
        str(item) for item in source_components
    ]
    root[str(source_component)].attrs[_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR] = list(
        persisted_factors
    )
    component_map = root.attrs.get(_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if not isinstance(component_map, dict):
        component_map = {}
    component_map[str(source_component)] = [str(item) for item in source_components]
    root.attrs[_DISPLAY_PYRAMID_ROOT_MAP_ATTR] = _sanitize_metadata_value(component_map)
    legacy_component_map = root.attrs.get(_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if not isinstance(legacy_component_map, dict):
        legacy_component_map = {}
    legacy_component_map[str(source_component)] = [
        str(item) for item in source_components
    ]
    root.attrs[_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR] = _sanitize_metadata_value(
        legacy_component_map
    )

    contrast_source_component = (
        str(source_components[-1]) if source_components else source_component
    )
    _emit(
        70,
        "Computing display contrast metadata for "
        f"{source_component} (sampling={contrast_source_component})",
    )
    contrast_limits_by_channel = _compute_display_contrast_limits_by_channel(
        zarr_path=zarr_path,
        source_component=source_component,
        sampling_component=contrast_source_component,
        channel_count=channel_count,
    )
    _save_display_contrast_metadata(
        root=root,
        source_component=source_component,
        contrast_limits_by_channel=contrast_limits_by_channel,
        contrast_source_component=contrast_source_component,
    )
    _emit(90, "Writing display pyramid metadata")
    component = _save_display_pyramid_metadata(
        zarr_path=zarr_path,
        source_component=source_component,
        source_components=source_components,
        contrast_limits_by_channel=contrast_limits_by_channel,
        contrast_source_component=contrast_source_component,
        reused_existing_levels=reused_existing_levels,
        run_id=run_id,
    )
    _emit(100, "Display pyramid preparation complete")
    return DisplayPyramidSummary(
        component=component,
        source_component=source_component,
        source_components=tuple(str(item) for item in source_components),
        channel_count=int(channel_count),
        contrast_limits_by_channel=tuple(
            (float(row[0]), float(row[1])) for row in contrast_limits_by_channel
        ),
        reused_existing_levels=bool(reused_existing_levels),
    )


def _save_visualization_metadata(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    source_components: Sequence[str],
    volume_layers: Sequence[ResolvedVolumeLayer],
    position_index: int,
    selected_positions: Sequence[int],
    show_all_positions: bool,
    spatial_calibration: SpatialCalibrationConfig,
    parameters: Mapping[str, Any],
    overlay_points_count: int,
    renderer: Optional[Mapping[str, Any]],
    launch_mode: str,
    viewer_pid: Optional[int],
    viewer_ndisplay_requested: int,
    viewer_ndisplay_effective: int,
    display_mode_fallback_reason: Optional[str],
    keyframe_manifest_path: Optional[str],
    keyframe_count: int,
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
    volume_layers : sequence[ResolvedVolumeLayer]
        Resolved volume layers rendered for this run.
    position_index : int
        Reference ``p``-axis index.
    selected_positions : sequence of int
        Rendered position indices.
    show_all_positions : bool
        Whether all positions were rendered.
    spatial_calibration : SpatialCalibrationConfig
        Effective store-level stage-to-world axis mapping used for placement.
    parameters : mapping[str, Any]
        Effective visualization parameters.
    overlay_points_count : int
        Number of overlaid points.
    renderer : mapping[str, Any], optional
        OpenGL renderer probe payload.
    launch_mode : str
        Effective launch mode.
    viewer_pid : int, optional
        Viewer process ID when launched as subprocess.
    viewer_ndisplay_requested : int
        Requested napari display mode (2 or 3).
    viewer_ndisplay_effective : int
        Effective napari display mode after ClearEx fallback policy.
    display_mode_fallback_reason : str, optional
        Explanation when requested 3D rendering is forced to 2D.
    keyframe_manifest_path : str, optional
        JSON path where interactive keyframe selections were written.
    keyframe_count : int
        Number of captured keyframes.
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
        "volume_layers": _serialize_resolved_volume_layers(volume_layers),
        "position_index": int(position_index),
        "selected_positions": [int(value) for value in selected_positions],
        "show_all_positions": bool(show_all_positions),
        "spatial_calibration": spatial_calibration_to_dict(spatial_calibration),
        "spatial_calibration_text": format_spatial_calibration(spatial_calibration),
        "overlay_points_count": int(overlay_points_count),
        "renderer": _sanitize_metadata_value(dict(renderer or {})),
        "launch_mode": str(launch_mode),
        "viewer_ndisplay_requested": int(max(2, int(viewer_ndisplay_requested))),
        "viewer_ndisplay_effective": int(max(2, int(viewer_ndisplay_effective))),
        "display_pyramid_lookup": {
            "source_attr": _DISPLAY_PYRAMID_LEVELS_ATTR,
            "root_attr_map": _DISPLAY_PYRAMID_ROOT_MAP_ATTR,
            "legacy_source_attr": _LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR,
            "legacy_root_attr_map": _LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR,
        },
        "display_contrast_metadata_reference": {
            "limits_attr": _DISPLAY_CONTRAST_LIMITS_ATTR,
            "percentiles_attr": _DISPLAY_CONTRAST_PERCENTILES_ATTR,
        },
        "parameters": {str(k): v for k, v in dict(parameters).items()},
        "storage_policy": "latest_only",
        "run_id": run_id,
        "capture_keyframes": bool(parameters.get("capture_keyframes", True)),
        "keyframe_count": int(max(0, keyframe_count)),
        "keyframe_layer_overrides": _normalize_keyframe_layer_overrides(
            parameters.get("keyframe_layer_overrides", [])
        ),
    }
    if viewer_pid is not None:
        payload["viewer_pid"] = int(viewer_pid)
    if (
        display_mode_fallback_reason is not None
        and str(display_mode_fallback_reason).strip()
    ):
        payload["display_mode_fallback_reason"] = str(display_mode_fallback_reason)
    if keyframe_manifest_path is not None and str(keyframe_manifest_path).strip():
        payload["keyframe_manifest_path"] = str(keyframe_manifest_path)
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
    root = zarr.open_group(str(zarr_path), mode="a")
    volume_layers = _resolve_volume_layers(
        root=root,
        parameters=normalized,
    )
    if len(volume_layers) <= 0:
        raise ValueError("Visualization requires at least one valid volume layer.")

    source_component = str(volume_layers[0].component)
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
        sum(int(np.asarray(points).shape[0]) for points in points_by_position.values())
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

    viewer_ndisplay_requested = 3 if bool(normalized.get("use_3d_view", True)) else 2
    viewer_ndisplay_effective, display_mode_fallback_reason = (
        _resolve_effective_viewer_ndisplay(
            root=root,
            volume_layers=volume_layers,
            requested_ndisplay=viewer_ndisplay_requested,
        )
    )
    if display_mode_fallback_reason is not None:
        _emit(15, display_mode_fallback_reason)

    launch_volume_layers: list[ResolvedVolumeLayer] = []
    for layer in volume_layers:
        rendered_source_components = (
            tuple(str(item) for item in layer.source_components)
            if int(viewer_ndisplay_effective) < 3
            else (str(layer.component),)
        )
        launch_volume_layers.append(
            ResolvedVolumeLayer(
                component=str(layer.component),
                source_components=rendered_source_components,
                layer_type=str(layer.layer_type),
                channels=tuple(int(value) for value in layer.channels),
                visible=layer.visible,
                opacity=layer.opacity,
                blending=str(layer.blending),
                colormap=str(layer.colormap),
                rendering=str(layer.rendering),
                name=str(layer.name),
                multiscale_policy=str(layer.multiscale_policy),
                multiscale_status=str(layer.multiscale_status),
            )
        )
    primary_layer = launch_volume_layers[0]
    source_components = tuple(str(item) for item in primary_layer.source_components)

    napari_payload = _build_napari_layer_payload(
        zarr_path=zarr_path,
        root=root,
        volume_layers=launch_volume_layers,
        position_index=reference_position_index,
        parameters=normalized,
        overlay_points_count=total_overlay_points,
        point_property_names=point_property_names,
    )
    (
        position_affines_tczyx,
        stage_rows,
        spatial_calibration,
    ) = _resolve_position_affines_tczyx(
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
    stage_position_rows = [
        {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "theta": float(row["theta"]),
            "f": float(row["f"]),
        }
        for row in stage_rows
    ]
    napari_payload.image_metadata["stage_positions_xyztheta"] = stage_position_rows
    napari_payload.image_metadata["stage_positions_xyzthetaf"] = stage_position_rows
    napari_payload.image_metadata["spatial_calibration"] = spatial_calibration_to_dict(
        spatial_calibration
    )
    napari_payload.image_metadata["spatial_calibration_text"] = (
        format_spatial_calibration(spatial_calibration)
    )
    napari_payload.points_metadata["position_index"] = int(reference_position_index)
    napari_payload.points_metadata["selected_positions"] = [
        int(value) for value in selected_positions
    ]
    napari_payload.points_metadata["show_all_positions"] = bool(show_all_positions)
    napari_payload.image_metadata["viewer_ndisplay_requested"] = int(
        viewer_ndisplay_requested
    )
    napari_payload.image_metadata["viewer_ndisplay_effective"] = int(
        viewer_ndisplay_effective
    )
    napari_payload.image_metadata["viewer_ndisplay"] = int(viewer_ndisplay_effective)
    napari_payload.points_metadata["viewer_ndisplay_requested"] = int(
        viewer_ndisplay_requested
    )
    napari_payload.points_metadata["viewer_ndisplay_effective"] = int(
        viewer_ndisplay_effective
    )
    napari_payload.points_metadata["viewer_ndisplay"] = int(viewer_ndisplay_effective)
    if display_mode_fallback_reason is not None:
        napari_payload.image_metadata["display_mode_fallback_reason"] = str(
            display_mode_fallback_reason
        )
        napari_payload.points_metadata["display_mode_fallback_reason"] = str(
            display_mode_fallback_reason
        )

    _emit(20, "Preparing napari visualization layers")
    effective_launch_mode = _resolve_effective_launch_mode(
        str(normalized.get("launch_mode", "auto"))
    )
    keyframe_manifest = _resolve_keyframe_manifest_path(
        zarr_path=zarr_path,
        parameters=normalized,
    )
    keyframe_manifest_path: Optional[str] = (
        str(keyframe_manifest) if keyframe_manifest is not None else None
    )
    keyframe_count = 0
    viewer_pid: Optional[int] = None
    renderer_info: Dict[str, Any] = {}
    if effective_launch_mode == "subprocess":
        _emit(65, "Launching napari in a separate process")
        subprocess_parameters: Dict[str, Any] = {
            **normalized,
            "launch_mode": "in_process",
        }
        if keyframe_manifest_path is not None:
            subprocess_parameters["keyframe_manifest_path"] = keyframe_manifest_path
        viewer_pid = _launch_napari_subprocess(
            zarr_path=zarr_path,
            normalized_parameters=subprocess_parameters,
        )
    else:
        _emit(65, "Opening napari viewer")
        viewer_capture = _launch_napari_viewer(
            zarr_path=zarr_path,
            volume_layers=launch_volume_layers,
            selected_positions=selected_positions,
            points_by_position=points_by_position,
            point_properties_by_position=point_properties_by_position,
            position_affines_tczyx=position_affines_tczyx,
            axis_labels=napari_payload.axis_labels_tczyx,
            scale_tczyx=napari_payload.scale_tczyx,
            image_metadata=napari_payload.image_metadata,
            points_metadata=napari_payload.points_metadata,
            require_gpu_rendering=bool(normalized.get("require_gpu_rendering", True)),
            capture_keyframes=bool(normalized.get("capture_keyframes", True)),
            keyframe_manifest_path=keyframe_manifest_path,
            keyframe_layer_overrides=normalized.get("keyframe_layer_overrides", []),
        )
        if isinstance(viewer_capture, Mapping):
            keyframe_manifest_candidate = viewer_capture.get("keyframe_manifest_path")
            if (
                keyframe_manifest_candidate is not None
                and str(keyframe_manifest_candidate).strip()
            ):
                keyframe_manifest_path = str(keyframe_manifest_candidate)
            try:
                keyframe_count = max(
                    0,
                    int(viewer_capture.get("keyframe_count", 0)),
                )
            except (TypeError, ValueError):
                keyframe_count = 0
            renderer_candidate = viewer_capture.get("renderer")
            if isinstance(renderer_candidate, Mapping):
                renderer_info = {
                    str(key): value for key, value in dict(renderer_candidate).items()
                }

    _emit(90, "Writing visualization metadata")
    component = _save_visualization_metadata(
        zarr_path=zarr_path,
        source_component=source_component,
        source_components=source_components,
        volume_layers=launch_volume_layers,
        position_index=reference_position_index,
        selected_positions=selected_positions,
        show_all_positions=show_all_positions,
        spatial_calibration=spatial_calibration,
        parameters=normalized,
        overlay_points_count=total_overlay_points,
        renderer=renderer_info,
        launch_mode=effective_launch_mode,
        viewer_pid=viewer_pid,
        viewer_ndisplay_requested=viewer_ndisplay_requested,
        viewer_ndisplay_effective=viewer_ndisplay_effective,
        display_mode_fallback_reason=display_mode_fallback_reason,
        keyframe_manifest_path=keyframe_manifest_path,
        keyframe_count=keyframe_count,
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
        keyframe_manifest_path=keyframe_manifest_path,
        keyframe_count=int(keyframe_count),
        renderer=dict(renderer_info),
        viewer_ndisplay_requested=int(viewer_ndisplay_requested),
        viewer_ndisplay_effective=int(viewer_ndisplay_effective),
        display_mode_fallback_reason=display_mode_fallback_reason,
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
