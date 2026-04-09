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
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

# Third Party Imports
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp
import zarr

# Local Imports
from clearex.io.experiment import load_navigate_experiment
from clearex.io.ome_store import (
    analysis_auxiliary_root,
    ensure_group,
    load_store_metadata,
    resolve_voxel_size_um_zyx_with_source,
)
from clearex.io.provenance import register_latest_output_reference
from clearex.visualization.export import (
    compile_png_frames_to_mp4,
    compile_png_frames_to_prores,
    verify_png_frame_directory,
)
from clearex.workflow import (
    SpatialCalibrationConfig,
    format_spatial_calibration,
    resolve_analysis_input_component,
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
_DISPLAY_PYRAMID_BUILD_COMPLETE_ATTR = "display_pyramid_build_complete"
_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR = "visualization_pyramid_levels"
_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR = "visualization_pyramid_factors_tpczyx"
_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "visualization_pyramid_levels_by_component"
_DISPLAY_CONTRAST_LIMITS_ATTR = "display_contrast_limits_by_channel"
_DISPLAY_CONTRAST_PERCENTILES_ATTR = "display_contrast_percentiles"
_DISPLAY_CONTRAST_LEVEL_SOURCE_ATTR = "display_contrast_source_component"
_DISPLAY_CONTRAST_SAMPLE_TARGET_VOXELS = 2_000_000
_DISPLAY_PYRAMID_BUILD_PROGRESS_TASK_STEP = 1_024
_DISPLAY_PYRAMID_BUILD_PROGRESS_MIN_INTERVAL_SECONDS = 1.5
_MAX_CAPTURED_OVERLAY_LAYER_ELEMENTS = 500_000
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
_ZARR_METADATA_FILE_NAMES = frozenset(
    {"zarr.json", ".zattrs", ".zgroup", ".zmetadata", ".zarray"}
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
class RenderMovieSummary:
    """Summary metadata for one render-movie run."""

    component: str
    visualization_component: str
    keyframe_manifest_path: str
    render_manifest_path: str
    output_directory: str
    rendered_levels: tuple[int, ...]
    frame_count: int
    fps: int


@dataclass(frozen=True)
class CompileMovieSummary:
    """Summary metadata for one compile-movie run."""

    component: str
    render_component: str
    render_manifest_path: str
    output_directory: str
    rendered_level: int
    output_format: str
    compiled_files: tuple[str, ...]
    fps: int


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


@dataclass(frozen=True)
class PreparedVisualizationScene:
    """Prepared visualization context reusable across viewer entrypoints."""

    normalized_parameters: Dict[str, Any]
    volume_layers: tuple[ResolvedVolumeLayer, ...]
    selected_positions: tuple[int, ...]
    reference_position_index: int
    points_by_position: Dict[int, np.ndarray]
    point_properties_by_position: Dict[int, Dict[str, np.ndarray]]
    total_overlay_points: int
    source_component: str
    source_components: tuple[str, ...]
    viewer_ndisplay_requested: int
    viewer_ndisplay_effective: int
    display_mode_fallback_reason: Optional[str]
    napari_payload: NapariLayerPayload
    position_affines_tczyx: Dict[int, np.ndarray]
    spatial_calibration: SpatialCalibrationConfig


@dataclass(frozen=True)
class BuiltNapariScene:
    """Live napari viewer scene returned by the shared scene builder."""

    viewer: Any
    renderer_info: Dict[str, Any]
    manifest_path: Optional[Path]
    primary_source_component: str
    primary_source_components: tuple[str, ...]
    serialized_volume_layers: list[Dict[str, Any]]
    axis_labels_tczyx: tuple[str, ...]
    scale_tczyx: tuple[float, ...]


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


def _close_zarr_store(container: Any) -> None:
    """Close a Zarr store handle when the backing store supports it."""
    store = getattr(container, "store", None)
    close_method = getattr(store, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            return


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
    store_metadata : mapping[str, Any], optional
        ClearEx namespaced metadata attrs.
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
    *,
    store_metadata: Optional[Mapping[str, Any]] = None,
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
    source_experiment = None
    if isinstance(store_metadata, Mapping):
        candidate = store_metadata.get("source_experiment")
        if isinstance(candidate, str) and candidate.strip():
            source_experiment = candidate
    if source_experiment is None:
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
    store_metadata = load_store_metadata(root)
    source_data_path = (
        store_metadata.get("source_data_path")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("source_data_path")
    source_data_component = (
        store_metadata.get("source_data_component")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("source_data_component")
    source_experiment = (
        store_metadata.get("source_experiment")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("source_experiment")
    navigate_experiment = (
        store_metadata.get("navigate_experiment")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("navigate_experiment")
    configured_chunks_tpczyx = (
        store_metadata.get("configured_chunks_tpczyx")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("configured_chunks_tpczyx")
    data_pyramid_levels = (
        store_metadata.get("data_pyramid_levels")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("data_pyramid_levels")
    data_pyramid_factors_tpczyx = (
        store_metadata.get("data_pyramid_factors_tpczyx")
        if isinstance(store_metadata, Mapping)
        else None
    ) or root_attrs.get("data_pyramid_factors_tpczyx")
    source_array = root[str(source_component)]
    source_attrs = dict(source_array.attrs)
    source_experiment_raw = _load_source_experiment_raw(
        root_attrs,
        store_metadata=store_metadata,
    )
    resolved_voxel_size_um_zyx, voxel_size_resolution_source = (
        resolve_voxel_size_um_zyx_with_source(
            root,
            source_component=source_component,
        )
    )
    scale_tczyx = (
        _extract_scale_tczyx_from_attrs(
            root_attrs=root_attrs,
            source_attrs=source_attrs,
        )
        or _extract_scale_tczyx_from_navigate_raw(source_experiment_raw)
        or (
            1.0,
            1.0,
            float(resolved_voxel_size_um_zyx[0]),
            float(resolved_voxel_size_um_zyx[1]),
            float(resolved_voxel_size_um_zyx[2]),
        )
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
        "voxel_size_um_zyx": [float(value) for value in resolved_voxel_size_um_zyx],
        "voxel_size_resolution_source": str(voxel_size_resolution_source),
        "source_component": str(source_component),
        "source_components": [str(item) for item in source_components],
        "volume_layers": volume_layers_payload,
        "position_index": int(position_index),
        "source_data_path": source_data_path,
        "source_data_component": source_data_component,
        "source_data_axes": root_attrs.get("source_data_axes"),
        "source_experiment": source_experiment,
        "navigate_experiment": navigate_experiment,
        "source_experiment_metadata": source_experiment_raw,
        "configured_chunks_tpczyx": configured_chunks_tpczyx,
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
        "data_pyramid_levels": data_pyramid_levels,
        "data_pyramid_factors_tpczyx": data_pyramid_factors_tpczyx,
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
                f"{analysis_auxiliary_root('particle_detection')}/detections",
            )
        ),
        "source_data_path": source_data_path,
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
                f"{analysis_auxiliary_root('particle_detection')}/detections",
            )
        ).strip()
        or f"{analysis_auxiliary_root('particle_detection')}/detections"
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


def _normalize_render_movie_parameters(
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize render-movie runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate render-movie parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized render-movie parameters.

    Raises
    ------
    ValueError
        If timing, output-size, camera-effect, or overlay values are invalid.
    """
    normalized = dict(parameters)
    normalized["input_source"] = (
        str(normalized.get("input_source", "visualization")).strip() or "visualization"
    )
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["keyframe_manifest_path"] = str(
        normalized.get("keyframe_manifest_path", "")
    ).strip()
    raw_levels = normalized.get("resolution_levels", [0])
    if not isinstance(raw_levels, (tuple, list)):
        raw_levels = [raw_levels]
    levels = sorted(
        set(max(0, int(value)) for value in raw_levels if str(value).strip() != "")
    )
    normalized["resolution_levels"] = levels or [0]
    render_size_xy = normalized.get("render_size_xy", [1920, 1080])
    if not isinstance(render_size_xy, (tuple, list)) or len(render_size_xy) != 2:
        raise ValueError(
            "render_movie render_size_xy must define two integers in (x, y) order."
        )
    normalized["render_size_xy"] = [
        max(1, int(render_size_xy[0])),
        max(1, int(render_size_xy[1])),
    ]
    normalized["fps"] = max(1, int(normalized.get("fps", 24)))
    normalized["default_transition_frames"] = max(
        0, int(normalized.get("default_transition_frames", 48))
    )
    raw_transition_frames = normalized.get("transition_frames_by_gap", [])
    if not isinstance(raw_transition_frames, (tuple, list)):
        raw_transition_frames = [raw_transition_frames]
    normalized["transition_frames_by_gap"] = [
        max(0, int(value))
        for value in raw_transition_frames
        if str(value).strip() != ""
    ]
    normalized["hold_frames"] = max(0, int(normalized.get("hold_frames", 12)))
    interpolation_mode = (
        str(normalized.get("interpolation_mode", "ease_in_out")).strip().lower()
        or "ease_in_out"
    )
    if interpolation_mode not in {"linear", "ease_in_out"}:
        raise ValueError(
            "render_movie interpolation_mode must be one of linear, ease_in_out."
        )
    normalized["interpolation_mode"] = interpolation_mode
    camera_effect = (
        str(normalized.get("camera_effect", "none")).strip().lower() or "none"
    )
    if camera_effect not in {"none", "orbit", "flythrough", "zoom_fx"}:
        raise ValueError(
            "render_movie camera_effect must be one of none, orbit, flythrough, zoom_fx."
        )
    normalized["camera_effect"] = camera_effect
    normalized["orbit_degrees"] = max(0.0, float(normalized.get("orbit_degrees", 45.0)))
    normalized["flythrough_distance_factor"] = max(
        0.0, float(normalized.get("flythrough_distance_factor", 0.10))
    )
    normalized["zoom_effect_factor"] = max(
        0.0, float(normalized.get("zoom_effect_factor", 0.15))
    )
    normalized["overlay_title"] = str(normalized.get("overlay_title", "")).strip()
    normalized["overlay_subtitle"] = str(normalized.get("overlay_subtitle", "")).strip()
    frame_text_mode = (
        str(normalized.get("overlay_frame_text_mode", "none")).strip().lower() or "none"
    )
    if frame_text_mode not in {"none", "frame_number", "keyframe_annotations"}:
        raise ValueError(
            "render_movie overlay_frame_text_mode must be one of none, "
            "frame_number, keyframe_annotations."
        )
    normalized["overlay_frame_text_mode"] = frame_text_mode
    normalized["overlay_scalebar"] = bool(normalized.get("overlay_scalebar", False))
    normalized["overlay_scalebar_length_um"] = max(
        np.finfo(np.float32).eps,
        float(normalized.get("overlay_scalebar_length_um", 50.0)),
    )
    scalebar_position = (
        str(normalized.get("overlay_scalebar_position", "bottom_left")).strip().lower()
        or "bottom_left"
    )
    if scalebar_position not in {
        "bottom_left",
        "bottom_right",
        "top_left",
        "top_right",
    }:
        raise ValueError(
            "render_movie overlay_scalebar_position must be one of "
            "bottom_left, bottom_right, top_left, top_right."
        )
    normalized["overlay_scalebar_position"] = scalebar_position
    normalized["output_directory"] = str(normalized.get("output_directory", "")).strip()
    launch_mode = str(normalized.get("launch_mode", "auto")).strip().lower() or "auto"
    if launch_mode not in {"auto", "in_process", "subprocess"}:
        raise ValueError(
            "render_movie launch_mode must be one of auto, in_process, subprocess."
        )
    normalized["launch_mode"] = launch_mode
    return normalized


def _normalize_compile_movie_parameters(
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize compile-movie runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate compile-movie parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized compile-movie parameters.

    Raises
    ------
    ValueError
        If codec, FPS, level-selection, or resize values are invalid.
    """
    normalized = dict(parameters)
    normalized["input_source"] = (
        str(normalized.get("input_source", "render_movie")).strip() or "render_movie"
    )
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["render_manifest_path"] = str(
        normalized.get("render_manifest_path", "")
    ).strip()
    normalized["rendered_level"] = max(0, int(normalized.get("rendered_level", 0)))
    output_format = str(normalized.get("output_format", "mp4")).strip().lower() or "mp4"
    if output_format not in {"mp4", "prores", "both"}:
        raise ValueError(
            "compile_movie output_format must be one of mp4, prores, both."
        )
    normalized["output_format"] = output_format
    normalized["fps"] = max(1, int(normalized.get("fps", 24)))
    mp4_crf = int(normalized.get("mp4_crf", 18))
    if mp4_crf < 0 or mp4_crf > 51:
        raise ValueError("compile_movie mp4_crf must be in [0, 51].")
    normalized["mp4_crf"] = mp4_crf
    normalized["mp4_preset"] = (
        str(normalized.get("mp4_preset", "slow")).strip().lower() or "slow"
    )
    normalized["prores_profile"] = max(
        0, min(5, int(normalized.get("prores_profile", 3)))
    )
    normalized["pixel_format"] = str(normalized.get("pixel_format", "")).strip()
    resize_xy = normalized.get("resize_xy", [])
    if resize_xy in (None, "", [], ()):
        normalized["resize_xy"] = []
    else:
        if not isinstance(resize_xy, (tuple, list)) or len(resize_xy) != 2:
            raise ValueError(
                "compile_movie resize_xy must define two integers in (x, y) order."
            )
        normalized["resize_xy"] = [
            max(1, int(resize_xy[0])),
            max(1, int(resize_xy[1])),
        ]
    normalized["output_directory"] = str(normalized.get("output_directory", "")).strip()
    normalized["output_stem"] = str(normalized.get("output_stem", "")).strip()
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

    def _qt_application_is_active() -> bool:
        """Return whether a Qt application instance is currently active."""
        qt_widgets_module = sys.modules.get("PyQt6.QtWidgets")
        application_cls = (
            getattr(qt_widgets_module, "QApplication", None)
            if qt_widgets_module is not None
            else None
        )
        if application_cls is None:
            try:
                from PyQt6.QtWidgets import QApplication as _QApplication
            except Exception:
                return False
            application_cls = _QApplication
        instance_getter = getattr(application_cls, "instance", None)
        if not callable(instance_getter):
            return False
        try:
            return instance_getter() is not None
        except Exception:
            return False

    mode = str(requested_mode).strip().lower() or "auto"
    if mode == "auto":
        if _qt_application_is_active():
            # Keep GUI workflows non-blocking while napari remains open.
            return "subprocess"
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
    return (
        store_path / analysis_auxiliary_root("visualization") / "keyframes.json"
    ).resolve()


def _legacy_keyframe_manifest_path(zarr_path: Union[str, Path]) -> Path:
    """Return the legacy sidecar keyframe-manifest path.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.

    Returns
    -------
    pathlib.Path
        Legacy sidecar keyframe-manifest path used by older ClearEx versions.
    """
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
        "tail_length",
        "head_length",
        "tail_width",
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


def _serialize_reconstructable_overlay_layer(layer: Any) -> Optional[Dict[str, Any]]:
    """Serialize a non-image napari overlay layer that can be rebuilt later.

    Parameters
    ----------
    layer : Any
        Live napari layer instance.

    Returns
    -------
    dict[str, Any], optional
        Serializable overlay payload for supported layer types, otherwise
        ``None``.
    """
    layer_type = str(type(layer).__name__).strip().lower()
    if layer_type not in {"points", "tracks"}:
        return None
    data = np.asarray(getattr(layer, "data", np.empty((0,), dtype=np.float32)))
    if int(data.size) > _MAX_CAPTURED_OVERLAY_LAYER_ELEMENTS:
        return None

    payload = _serialize_layer_configuration(
        layer,
        order_index=0,
        selected_layer_names=set(),
        active_layer_name=None,
        override_by_layer_name={},
    )
    payload["data"] = _sanitize_metadata_value(data)

    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, Mapping):
        payload["metadata"] = {
            str(key): _sanitize_metadata_value(value) for key, value in metadata.items()
        }
    properties = getattr(layer, "properties", None)
    if isinstance(properties, Mapping):
        payload["properties"] = {
            str(key): _sanitize_metadata_value(value)
            for key, value in properties.items()
        }
    if layer_type == "tracks":
        graph = getattr(layer, "graph", None)
        if graph is not None:
            payload["graph"] = _sanitize_metadata_value(graph)
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
        "id": f"keyframe_{int(max(0, keyframe_index)):04d}",
        "index": int(max(0, keyframe_index)),
        "order_index": int(max(0, keyframe_index)),
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


def _read_json_mapping(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON object from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Source JSON path.

    Returns
    -------
    dict[str, Any]
        Parsed mapping payload.

    Raises
    ------
    ValueError
        If the JSON document is not an object.
    """
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return {str(key): value for key, value in payload.items()}


def _write_json_mapping(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    """Write one JSON object to disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination JSON path.
    payload : mapping[str, Any]
        Mapping payload to serialize.

    Returns
    -------
    None
        Writes the JSON file for its side effects only.
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_metadata_value(dict(payload)), indent=2),
        encoding="utf-8",
    )


def _resolve_movie_output_base_directory(
    *,
    zarr_path: Union[str, Path],
    output_directory: str,
    suffix: str,
) -> Path:
    """Resolve a movie-render or movie-compile output base directory.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    output_directory : str
        Optional explicit output directory.
    suffix : str
        Auto-generated directory suffix when ``output_directory`` is empty.

    Returns
    -------
    pathlib.Path
        Resolved output base directory.
    """
    store_path = Path(zarr_path).expanduser().resolve()
    candidate = str(output_directory).strip()
    if candidate:
        resolved = Path(candidate).expanduser()
        if not resolved.is_absolute():
            resolved = (store_path.parent / resolved).resolve()
        return resolved
    return (store_path / analysis_auxiliary_root(str(suffix))).resolve().parent


def _legacy_movie_output_base_directory(
    *,
    zarr_path: Union[str, Path],
    suffix: str,
) -> Path:
    """Return the legacy sidecar movie-output directory root.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    suffix : str
        Sidecar suffix previously used for movie artifacts.

    Returns
    -------
    pathlib.Path
        Legacy output base directory outside the store.
    """
    store_path = Path(zarr_path).expanduser().resolve()
    return (store_path.parent / f"{store_path.name}_{suffix}").resolve()


def _prepare_movie_output_directory(base_directory: Path) -> Path:
    """Create a clean ``latest`` movie-output directory.

    Parameters
    ----------
    base_directory : pathlib.Path
        Parent movie-output directory.

    Returns
    -------
    pathlib.Path
        Prepared ``latest`` directory path.
    """
    latest_directory = (
        base_directory
        if str(base_directory.name).strip().lower() == "latest"
        else (base_directory / "latest")
    ).resolve()
    latest_directory.mkdir(parents=True, exist_ok=True)
    for child in tuple(latest_directory.iterdir()):
        if child.name in _ZARR_METADATA_FILE_NAMES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return latest_directory


def _clear_group_attrs(group: zarr.Group) -> None:
    """Remove all attributes from one Zarr group in place.

    Parameters
    ----------
    group : zarr.Group
        Group whose attributes should be cleared.

    Returns
    -------
    None
        Mutates ``group.attrs`` in place.
    """
    for key in tuple(getattr(group, "attrs", {}).keys()):
        try:
            del group.attrs[key]
        except Exception:
            continue


def _ease_fraction(alpha: float, mode: str) -> float:
    """Map linear interpolation fractions through an easing curve."""
    fraction = max(0.0, min(1.0, float(alpha)))
    if str(mode).strip().lower() == "linear":
        return fraction
    return float(fraction * fraction * (3.0 - (2.0 * fraction)))


def _interpolate_scalar(start: float, end: float, alpha: float) -> float:
    """Linearly interpolate one scalar."""
    return float((1.0 - float(alpha)) * float(start) + float(alpha) * float(end))


def _interpolate_vector(
    start: Sequence[float],
    end: Sequence[float],
    alpha: float,
) -> tuple[float, ...]:
    """Linearly interpolate one numeric vector."""
    return tuple(
        _interpolate_scalar(float(a), float(b), float(alpha))
        for a, b in zip(tuple(start), tuple(end), strict=False)
    )


def _slerp_camera_angles(
    start_angles: Sequence[float],
    end_angles: Sequence[float],
    fractions: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Interpolate napari camera angles using quaternion SLERP."""
    if len(tuple(fractions)) <= 0:
        return tuple()
    start_rotation = Rotation.from_euler("yzx", tuple(start_angles), degrees=True)
    end_rotation = Rotation.from_euler("yzx", tuple(end_angles), degrees=True)
    slerp = Slerp(
        np.asarray([0.0, 1.0], dtype=np.float64),
        Rotation.concatenate((start_rotation, end_rotation)),
    )
    interpolated = slerp(np.asarray([float(v) for v in fractions], dtype=np.float64))
    angle_rows = interpolated.as_euler("yzx", degrees=True)
    return tuple(
        (
            float(row[0]),
            float(row[1]),
            float(row[2]),
        )
        for row in np.asarray(angle_rows, dtype=np.float64)
    )


def _ensure_rgba_frame(frame: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Coerce screenshot data into RGBA uint8."""
    rgba = np.asarray(frame)
    if rgba.dtype != np.uint8:
        if rgba.size > 0 and float(np.nanmax(rgba)) <= 1.0:
            rgba = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgba = np.clip(rgba, 0.0, 255.0).astype(np.uint8)
    if rgba.ndim == 2:
        rgba = np.stack([rgba, rgba, rgba], axis=-1)
    if rgba.shape[-1] == 3:
        alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=-1)
    if rgba.shape[-1] != 4:
        raise ValueError(f"Unexpected screenshot shape: {rgba.shape}")
    return rgba


def _frame_has_visible_rgb_content(frame: npt.ArrayLike) -> bool:
    """Return whether a captured frame contains any visible RGB signal.

    Parameters
    ----------
    frame : numpy.ndarray
        Captured frame data in any napari/PIL-compatible image shape.

    Returns
    -------
    bool
        ``True`` when at least one RGB value is non-zero after RGBA coercion.
    """
    rgba = _ensure_rgba_frame(frame)
    if rgba.size <= 0:
        return False
    return bool(np.any(rgba[..., :3] > 0))


def _load_overlay_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font suitable for movie overlays."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _movie_overlay_rgba(
    frame_rgba_u8: npt.ArrayLike,
    *,
    title: str,
    subtitle: str,
    frame_text: str,
    scalebar_enabled: bool,
    scalebar_length_um: float,
    scalebar_position: str,
    pixel_size_x_um: Optional[float],
) -> npt.NDArray[np.uint8]:
    """Draw title/subtitle/frame-text/scalebar overlays on one frame."""
    rgba = _ensure_rgba_frame(frame_rgba_u8)
    base = Image.fromarray(rgba, mode="RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width_px = int(rgba.shape[1])
    height_px = int(rgba.shape[0])
    margin = max(18, min(width_px, height_px) // 40)
    title_font = _load_overlay_font(max(20, height_px // 28))
    subtitle_font = _load_overlay_font(max(16, height_px // 38))
    meta_font = _load_overlay_font(max(14, height_px // 48))

    title_text = str(title).strip()
    subtitle_text = str(subtitle).strip()
    frame_meta_text = str(frame_text).strip()
    if title_text or subtitle_text:
        text_lines = [text for text in (title_text, subtitle_text) if text]
        line_boxes = [
            draw.textbbox(
                (0, 0), text, font=(title_font if idx == 0 else subtitle_font)
            )
            for idx, text in enumerate(text_lines)
        ]
        box_width = max(max(0, bbox[2] - bbox[0]) for bbox in line_boxes)
        box_height = sum(max(0, bbox[3] - bbox[1]) for bbox in line_boxes) + (
            max(0, len(text_lines) - 1) * max(6, height_px // 180)
        )
        draw.rounded_rectangle(
            (
                margin - 10,
                margin - 10,
                margin + box_width + 10,
                margin + box_height + 10,
            ),
            radius=12,
            fill=(0, 0, 0, 150),
        )
        cursor_y = margin
        for idx, text in enumerate(text_lines):
            font = title_font if idx == 0 else subtitle_font
            fill = (255, 255, 255, 255) if idx == 0 else (220, 230, 255, 255)
            draw.text((margin, cursor_y), text, font=font, fill=fill)
            bbox = draw.textbbox((margin, cursor_y), text, font=font)
            cursor_y = int(bbox[3]) + max(6, height_px // 180)

    if frame_meta_text:
        bbox = draw.textbbox((0, 0), frame_meta_text, font=meta_font)
        text_width = max(0, bbox[2] - bbox[0])
        text_height = max(0, bbox[3] - bbox[1])
        x0 = width_px - margin - text_width - 10
        y0 = margin
        draw.rounded_rectangle(
            (x0, y0, x0 + text_width + 20, y0 + text_height + 16),
            radius=10,
            fill=(0, 0, 0, 140),
        )
        draw.text(
            (x0 + 10, y0 + 8),
            frame_meta_text,
            font=meta_font,
            fill=(255, 255, 255, 255),
        )

    if scalebar_enabled and pixel_size_x_um is not None and pixel_size_x_um > 0:
        bar_width = max(
            8, int(round(float(scalebar_length_um) / float(pixel_size_x_um)))
        )
        bar_width = min(bar_width, max(8, width_px - (2 * margin)))
        bar_height = max(4, height_px // 180)
        label = f"{float(scalebar_length_um):g} um"
        label_bbox = draw.textbbox((0, 0), label, font=meta_font)
        label_width = max(0, label_bbox[2] - label_bbox[0])
        label_height = max(0, label_bbox[3] - label_bbox[1])
        total_width = max(bar_width, label_width)
        total_height = bar_height + label_height + 10
        if scalebar_position == "bottom_right":
            x0 = width_px - margin - total_width
            y0 = height_px - margin - total_height
        elif scalebar_position == "top_left":
            x0 = margin
            y0 = margin
        elif scalebar_position == "top_right":
            x0 = width_px - margin - total_width
            y0 = margin
        else:
            x0 = margin
            y0 = height_px - margin - total_height
        draw.rounded_rectangle(
            (x0 - 10, y0 - 8, x0 + total_width + 10, y0 + total_height + 8),
            radius=10,
            fill=(0, 0, 0, 140),
        )
        draw.rectangle(
            (
                x0,
                y0 + label_height + 10,
                x0 + bar_width,
                y0 + label_height + 10 + bar_height,
            ),
            fill=(255, 255, 255, 255),
        )
        draw.text((x0, y0), label, font=meta_font, fill=(255, 255, 255, 255))

    base.alpha_composite(overlay)
    return np.asarray(base)


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


def _display_pyramid_root_component(*, source_component: str) -> str:
    """Return the pyramid-root group path for one source component."""
    component = str(source_component).strip() or "data"
    if component == "data":
        return "data_pyramid"
    return f"{component}_pyramid"


def _component_sequence_matches(
    payload: object,
    *,
    expected: Sequence[str],
) -> bool:
    """Return whether metadata payload matches an ordered component sequence."""
    if not isinstance(payload, (tuple, list)):
        return False
    normalized = tuple(str(item).strip() for item in payload if str(item).strip())
    return normalized == tuple(str(item).strip() for item in expected)


def _display_pyramid_level_metadata_is_complete(
    *,
    root: zarr.hierarchy.Group,
    component: str,
    level_index: int,
) -> bool:
    """Return whether one existing level looks fully materialized."""
    try:
        array = root[str(component)]
    except Exception:
        return False

    if len(tuple(getattr(array, "shape", tuple()))) != 6:
        return False

    attrs = dict(array.attrs)
    try:
        if int(attrs.get("pyramid_level", -1)) != int(level_index):
            return False
    except Exception:
        return False

    axes = attrs.get("axes")
    if list(axes) != ["t", "p", "c", "z", "y", "x"]:
        return False

    factors = attrs.get("downsample_factors_tpczyx")
    if not isinstance(factors, (tuple, list)) or len(factors) != 6:
        return False
    try:
        if any(int(value) < 1 for value in factors):
            return False
    except Exception:
        return False

    chunk_shape = attrs.get("chunk_shape_tpczyx")
    if not isinstance(chunk_shape, (tuple, list)) or len(chunk_shape) != 6:
        return False
    try:
        chunk_shape_tpczyx = tuple(int(value) for value in chunk_shape)
    except Exception:
        return False
    array_chunks = getattr(array, "chunks", None)
    if array_chunks is None:
        return False
    if tuple(int(value) for value in tuple(array_chunks)) != chunk_shape_tpczyx:
        return False

    source_component = str(attrs.get("source_component", "")).strip()
    return bool(source_component)


def _source_has_multiscale_lookup_metadata(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
    source_components: Sequence[str],
) -> bool:
    """Return whether lookup metadata references the discovered components."""
    expected = tuple(str(item).strip() for item in source_components)
    source_attrs = dict(root[str(source_component)].attrs)
    source_match = any(
        _component_sequence_matches(source_attrs.get(attr_name), expected=expected)
        for attr_name in (
            _DISPLAY_PYRAMID_LEVELS_ATTR,
            _LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR,
            "pyramid_levels",
        )
    )
    if source_match:
        return True

    root_attrs = dict(root.attrs)
    for attr_name in (
        _DISPLAY_PYRAMID_ROOT_MAP_ATTR,
        _LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR,
    ):
        payload = root_attrs.get(attr_name)
        if not isinstance(payload, Mapping):
            continue
        if _component_sequence_matches(
            payload.get(str(source_component)),
            expected=expected,
        ):
            return True

    if str(source_component).strip() == "data":
        return _component_sequence_matches(
            root_attrs.get("data_pyramid_levels"),
            expected=expected,
        )
    return False


def _display_pyramid_state_is_complete(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
) -> bool:
    """Return whether an existing pyramid can be safely reused as-is."""
    source_components = _collect_existing_multiscale_components(
        root=root,
        source_component=str(source_component),
    )
    if len(source_components) <= 1:
        return False
    if not _is_source_pyramid_layout_compatible(
        source_component=str(source_component),
        source_components=source_components,
    ):
        return False
    if not _source_has_multiscale_lookup_metadata(
        root=root,
        source_component=str(source_component),
        source_components=source_components,
    ):
        return False

    for level_index, component in enumerate(source_components[1:], start=1):
        if not _display_pyramid_level_metadata_is_complete(
            root=root,
            component=str(component),
            level_index=int(level_index),
        ):
            return False

    build_complete = root[str(source_component)].attrs.get(
        _DISPLAY_PYRAMID_BUILD_COMPLETE_ATTR
    )
    if build_complete is None:
        return True
    return bool(build_complete)


def _clear_display_pyramid_retry_metadata(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
) -> None:
    """Clear display-pyramid lookup metadata for one source component."""
    source_attrs = root[str(source_component)].attrs
    for attr_name in (
        _DISPLAY_PYRAMID_LEVELS_ATTR,
        _DISPLAY_PYRAMID_FACTORS_ATTR,
        _DISPLAY_PYRAMID_BUILD_COMPLETE_ATTR,
        _LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR,
        _LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR,
    ):
        if attr_name in source_attrs:
            del source_attrs[attr_name]

    for attr_name in (
        _DISPLAY_PYRAMID_ROOT_MAP_ATTR,
        _LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR,
    ):
        payload = root.attrs.get(attr_name)
        if not isinstance(payload, Mapping):
            continue
        updated = {
            str(key): value
            for key, value in dict(payload).items()
            if str(key).strip() != str(source_component).strip()
        }
        if updated:
            root.attrs[attr_name] = _sanitize_metadata_value(updated)
        elif attr_name in root.attrs:
            del root.attrs[attr_name]


def _cleanup_incomplete_display_pyramid_state(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
) -> bool:
    """Delete an incomplete source-specific pyramid subtree before retry."""
    pyramid_root = _display_pyramid_root_component(source_component=source_component)
    if pyramid_root not in root:
        return False
    if _display_pyramid_state_is_complete(
        root=root,
        source_component=str(source_component),
    ):
        return False

    _LOGGER.warning(
        "[display_pyramid] deleting incomplete pyramid subtree for source=%s root=%s",
        str(source_component),
        str(pyramid_root),
    )
    del root[pyramid_root]
    _clear_display_pyramid_retry_metadata(
        root=root,
        source_component=str(source_component),
    )
    return True


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
    root[str(source_component)].attrs[_DISPLAY_PYRAMID_BUILD_COMPLETE_ATTR] = True
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


def _deserialize_resolved_volume_layers(
    payload: Any,
) -> tuple[ResolvedVolumeLayer, ...]:
    """Deserialize metadata payloads into resolved volume layers."""
    if not isinstance(payload, (tuple, list)):
        return tuple()
    resolved: list[ResolvedVolumeLayer] = []
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        component = str(row.get("component", "")).strip()
        source_components_raw = row.get("source_components", [])
        source_components = (
            tuple(
                str(item).strip() for item in source_components_raw if str(item).strip()
            )
            if isinstance(source_components_raw, (tuple, list))
            else tuple()
        )
        if not component:
            continue
        if not source_components:
            source_components = (component,)
        resolved.append(
            ResolvedVolumeLayer(
                component=component,
                source_components=tuple(source_components),
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
                multiscale_status=(
                    str(row.get("multiscale_status", "single_scale")).strip().lower()
                    or "single_scale"
                ),
            )
        )
    return tuple(resolved)


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
    *,
    store_metadata: Optional[Mapping[str, Any]] = None,
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
    if isinstance(store_metadata, Mapping):
        payload = store_metadata.get("spatial_calibration")
        if payload is not None:
            return spatial_calibration_from_dict(payload)
    return spatial_calibration_from_dict(root_attrs.get("spatial_calibration"))


def _load_multiposition_stage_rows(
    root_attrs: Mapping[str, Any],
    *,
    store_metadata: Optional[Mapping[str, Any]] = None,
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
    if isinstance(store_metadata, Mapping):
        stage_rows = store_metadata.get("stage_rows")
        parsed_stage_rows = _parse_multiposition_stage_rows(stage_rows)
        if parsed_stage_rows:
            return parsed_stage_rows

    source_experiment = None
    if isinstance(store_metadata, Mapping):
        candidate = store_metadata.get("source_experiment")
        if isinstance(candidate, str) and candidate.strip():
            source_experiment = candidate
    if source_experiment is None:
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
    store_metadata: Optional[Mapping[str, Any]] = None,
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
    spatial_calibration = _load_spatial_calibration(
        root_attrs,
        store_metadata=store_metadata,
    )
    stage_rows = _load_multiposition_stage_rows(
        root_attrs,
        store_metadata=store_metadata,
    )
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


def _build_visualization_subprocess_command(
    *,
    zarr_path: Union[str, Path],
    normalized_parameters: Mapping[str, Any],
    mode: str,
    run_id: Optional[str] = None,
) -> list[str]:
    """Build a module-entrypoint command for visualization subprocess work.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    normalized_parameters : mapping[str, Any]
        Normalized visualization or movie-render parameters.
    mode : str
        Subprocess entrypoint mode.
    run_id : str, optional
        Provenance run identifier when available.

    Returns
    -------
    list[str]
        Argument vector suitable for ``subprocess`` execution.
    """
    payload = json.dumps(dict(normalized_parameters), separators=(",", ":"))
    command = [
        sys.executable,
        "-m",
        "clearex.visualization.pipeline",
        "--mode",
        str(mode),
        "--zarr-path",
        str(zarr_path),
        "--parameters-json",
        payload,
    ]
    if run_id is not None and str(run_id).strip():
        command.extend(["--run-id", str(run_id).strip()])
    return command


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
    command = _build_visualization_subprocess_command(
        zarr_path=zarr_path,
        normalized_parameters=normalized_parameters,
        mode="visualization",
    )
    process = subprocess.Popen(command)
    return int(process.pid)


def _run_render_movie_subprocess(
    *,
    zarr_path: Union[str, Path],
    normalized_parameters: Mapping[str, Any],
    run_id: Optional[str] = None,
) -> None:
    """Run visible movie rendering in a dedicated subprocess and wait.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    normalized_parameters : mapping[str, Any]
        Normalized render-movie parameters for the child process.
    run_id : str, optional
        Provenance run identifier when available.

    Returns
    -------
    None
        Blocks until the child render completes successfully.

    Raises
    ------
    RuntimeError
        If the render subprocess exits unsuccessfully.
    """
    command = _build_visualization_subprocess_command(
        zarr_path=zarr_path,
        normalized_parameters=normalized_parameters,
        mode="render_movie",
        run_id=run_id,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    stderr = str(completed.stderr or "").strip()
    stdout = str(completed.stdout or "").strip()
    details = stderr or stdout
    message = (
        "render_movie subprocess failed " f"with exit code {int(completed.returncode)}."
    )
    if details:
        message = f"{message} {details}"
    raise RuntimeError(message)


def _build_napari_viewer_scene(
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
    movie_level_index: Optional[int] = None,
    show: bool = True,
) -> BuiltNapariScene:
    """Build a live napari viewer scene for interactive or scripted use."""
    import napari

    def _channel_colormap_cycle(channel_count: int) -> tuple[str, ...]:
        if channel_count <= 0:
            return tuple()
        base = ("green", "magenta", "bop orange")
        extras = ("cyan", "yellow", "blue", "red", "gray", "turquoise", "hotpink")
        palette: list[str] = []
        for index in range(channel_count):
            if index < len(base):
                palette.append(base[index])
            else:
                palette.append(extras[(index - len(base)) % len(extras)])
        return tuple(palette)

    def _default_channel_opacity(channel_count: int) -> float:
        if channel_count <= 1:
            return 1.0
        return float(max(0.55, 1.0 - (0.1 * (channel_count - 1))))

    scale = tuple(float(value) for value in scale_tczyx)
    scale_root: Optional[zarr.hierarchy.Group] = None
    scale_root_attrs: Dict[str, Any] = {}
    scale_store_metadata: Dict[str, Any] = {}
    scale_source_experiment_raw: Optional[Dict[str, Any]] = None
    try:
        scale_root = zarr.open_group(str(zarr_path), mode="r")
        scale_root_attrs = dict(scale_root.attrs)
        scale_store_metadata = load_store_metadata(scale_root)
        scale_source_experiment_raw = _load_source_experiment_raw(
            scale_root_attrs,
            store_metadata=scale_store_metadata,
        )
    except Exception:
        scale_root = None
    layer_scale_cache: dict[str, tuple[float, float, float, float, float]] = {}
    layer_display_contrast_cache: dict[str, tuple[tuple[float, float], ...]] = {}

    def _resolve_layer_scale_tczyx(
        component: str,
    ) -> tuple[float, float, float, float, float]:
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
        resolver_base_scale_hint: Optional[tuple[float, float, float, float, float]] = (
            None
        )
        if resolved_scale is None:
            resolved_voxel_size_um_zyx, voxel_resolution_source = (
                resolve_voxel_size_um_zyx_with_source(
                    scale_root,
                    source_component=key,
                )
            )
            if str(voxel_resolution_source) != "default":
                resolver_scale = (
                    float(scale[0]),
                    float(scale[1]),
                    float(resolved_voxel_size_um_zyx[0]),
                    float(resolved_voxel_size_um_zyx[1]),
                    float(resolved_voxel_size_um_zyx[2]),
                )
                resolution_source = str(voxel_resolution_source).strip().lower()
                component_specific = resolution_source.startswith(
                    "component:"
                ) or resolution_source.startswith("component_navigate:")
                if key == "data" or component_specific:
                    resolved_scale = resolver_scale
                else:
                    resolver_base_scale_hint = resolver_scale
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
            base_scale = (
                resolver_base_scale_hint
                if resolver_base_scale_hint is not None
                else scale
            )
            base_shape: Optional[tuple[int, int, int, int, int, int]] = None
            try:
                base_array = scale_root["data"]
                base_shape = tuple(int(value) for value in tuple(base_array.shape))
                base_scale = (
                    _extract_scale_tczyx_from_attrs(
                        root_attrs=scale_root_attrs,
                        source_attrs=dict(base_array.attrs),
                    )
                    or resolver_base_scale_hint
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
                    factors.append(
                        float(rounded)
                        if rounded >= 1 and abs(ratio - float(rounded)) <= 1e-6
                        else 1.0
                    )
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
        show=bool(show),
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
            if movie_level_index is None:
                rendered_source_components = (
                    tuple(str(item) for item in layer.source_components)
                    if int(effective_ndisplay) < 3
                    else (str(layer.component),)
                )
            else:
                available_levels = tuple(str(item) for item in layer.source_components)
                level_component = available_levels[
                    min(
                        max(0, int(movie_level_index)),
                        max(0, len(available_levels) - 1),
                    )
                ]
                rendered_source_components = (str(level_component),)

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
            is_multiscale = (
                movie_level_index is None
                and int(effective_ndisplay) < 3
                and len(display_level_arrays) > 1
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
            points_by_position.get(position_index, np.empty((0, 5), dtype=np.float32)),
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

    return BuiltNapariScene(
        viewer=viewer,
        renderer_info=dict(renderer_info),
        manifest_path=manifest_path,
        primary_source_component=str(primary_source_component),
        primary_source_components=tuple(
            str(item) for item in primary_source_components
        ),
        serialized_volume_layers=list(serialized_volume_layers),
        axis_labels_tczyx=tuple(axis_labels_tuple),
        scale_tczyx=tuple(float(value) for value in scale),
    )


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

    built_scene = _build_napari_viewer_scene(
        zarr_path=zarr_path,
        volume_layers=volume_layers,
        selected_positions=selected_positions,
        points_by_position=points_by_position,
        point_properties_by_position=point_properties_by_position,
        position_affines_tczyx=position_affines_tczyx,
        axis_labels=axis_labels,
        scale_tczyx=scale_tczyx,
        image_metadata=image_metadata,
        points_metadata=points_metadata,
        require_gpu_rendering=require_gpu_rendering,
        capture_keyframes=capture_keyframes,
        keyframe_manifest_path=keyframe_manifest_path,
        movie_level_index=None,
        show=True,
    )
    viewer = built_scene.viewer
    renderer_info = dict(built_scene.renderer_info)
    manifest_path = built_scene.manifest_path
    primary_source_component = str(built_scene.primary_source_component)
    primary_source_components = tuple(
        str(item) for item in built_scene.primary_source_components
    )
    serialized_volume_layers = list(built_scene.serialized_volume_layers)
    axis_labels_tuple = tuple(str(label) for label in built_scene.axis_labels_tczyx)
    scale = tuple(float(value) for value in built_scene.scale_tczyx)
    keyframes: list[Dict[str, Any]] = []
    normalized_layer_overrides = _normalize_keyframe_layer_overrides(
        keyframe_layer_overrides
    )
    selected = tuple(int(index) for index in selected_positions)

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
        layers = tuple(getattr(viewer, "layers", tuple()))
        payload: Dict[str, Any] = {
            "schema_version": 2,
            "zarr_path": str(Path(zarr_path).expanduser().resolve()),
            "source_component": str(primary_source_component),
            "source_components": [str(item) for item in primary_source_components],
            "volume_layers": serialized_volume_layers,
            "captured_overlay_layers": [
                overlay_payload
                for overlay_payload in (
                    _serialize_reconstructable_overlay_layer(layer) for layer in layers
                )
                if overlay_payload is not None
            ],
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
    root = zarr.open_group(str(zarr_path), mode="a")
    component = analysis_auxiliary_root("display_pyramid")
    if component in root:
        del root[component]
    latest_group = root.require_group(component)

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

    cleaned_incomplete_pyramid = _cleanup_incomplete_display_pyramid_state(
        root=root,
        source_component=source_component,
    )
    if cleaned_incomplete_pyramid:
        _emit(
            15,
            "Found incomplete display pyramid state; rebuilding from source.",
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
        if _DISPLAY_PYRAMID_BUILD_COMPLETE_ATTR in root[str(source_component)].attrs:
            del root[str(source_component)].attrs[_DISPLAY_PYRAMID_BUILD_COMPLETE_ATTR]
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
    root = zarr.open_group(str(zarr_path), mode="a")
    component = analysis_auxiliary_root("visualization")
    latest_group = ensure_group(root, component)
    _clear_group_attrs(latest_group)

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


def _prepare_visualization_scene(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
) -> PreparedVisualizationScene:
    """Prepare shared visualization scene inputs without launching napari."""

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_visualization_parameters(parameters)
    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        volume_layers = tuple(
            _resolve_volume_layers(
                root=root,
                parameters=normalized,
            )
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

        viewer_ndisplay_requested = (
            3 if bool(normalized.get("use_3d_view", True)) else 2
        )
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
            store_metadata=load_store_metadata(root),
            selected_positions=selected_positions,
            scale_tczyx=napari_payload.scale_tczyx,
        )
    finally:
        _close_zarr_store(root)

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

    return PreparedVisualizationScene(
        normalized_parameters={str(key): value for key, value in normalized.items()},
        volume_layers=tuple(launch_volume_layers),
        selected_positions=tuple(int(value) for value in selected_positions),
        reference_position_index=int(reference_position_index),
        points_by_position=dict(points_by_position),
        point_properties_by_position={
            int(key): dict(value) for key, value in point_properties_by_position.items()
        },
        total_overlay_points=int(total_overlay_points),
        source_component=str(source_component),
        source_components=tuple(str(item) for item in source_components),
        viewer_ndisplay_requested=int(viewer_ndisplay_requested),
        viewer_ndisplay_effective=int(viewer_ndisplay_effective),
        display_mode_fallback_reason=display_mode_fallback_reason,
        napari_payload=napari_payload,
        position_affines_tczyx={
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in position_affines_tczyx.items()
        },
        spatial_calibration=spatial_calibration,
    )


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

    scene = _prepare_visualization_scene(
        zarr_path=zarr_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )
    normalized = dict(scene.normalized_parameters)
    source_component = str(scene.source_component)
    source_components = tuple(str(item) for item in scene.source_components)
    reference_position_index = int(scene.reference_position_index)
    selected_positions = tuple(int(value) for value in scene.selected_positions)
    show_all_positions = bool(normalized.get("show_all_positions", False))
    total_overlay_points = int(scene.total_overlay_points)
    viewer_ndisplay_requested = int(scene.viewer_ndisplay_requested)
    viewer_ndisplay_effective = int(scene.viewer_ndisplay_effective)
    display_mode_fallback_reason = scene.display_mode_fallback_reason
    launch_volume_layers = tuple(scene.volume_layers)
    napari_payload = scene.napari_payload
    points_by_position = dict(scene.points_by_position)
    point_properties_by_position = {
        int(key): dict(value)
        for key, value in scene.point_properties_by_position.items()
    }
    position_affines_tczyx = {
        int(key): np.asarray(value, dtype=np.float64)
        for key, value in scene.position_affines_tczyx.items()
    }
    spatial_calibration = scene.spatial_calibration

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


def _read_latest_analysis_metadata(
    *,
    zarr_path: Union[str, Path],
    component: str,
    operation_name: str,
) -> Dict[str, Any]:
    """Read one latest-analysis metadata group as a plain dict.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    component : str
        Latest metadata component path.
    operation_name : str
        Operation name used in error messages.

    Returns
    -------
    dict[str, Any]
        Latest metadata attributes as a plain mapping.

    Raises
    ------
    ValueError
        If the requested component is missing.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        group = root[str(component)]
    except Exception as exc:
        raise ValueError(
            f"{operation_name} requires metadata component '{component}', "
            "but it was not found."
        ) from exc
    return {str(key): value for key, value in dict(group.attrs).items()}


def _resolve_render_movie_keyframe_manifest_path(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    visualization_metadata: Mapping[str, Any],
) -> str:
    """Resolve the keyframe manifest path consumed by ``render_movie``.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    parameters : mapping[str, Any]
        Normalized render-movie parameters.
    visualization_metadata : mapping[str, Any]
        Latest visualization metadata payload.

    Returns
    -------
    str
        Resolved JSON manifest path, or an empty string when no candidate can
        be inferred.
    """
    explicit_path = str(parameters.get("keyframe_manifest_path", "")).strip()
    if explicit_path:
        return str(Path(explicit_path).expanduser().resolve())

    fallback_missing_candidate: Optional[str] = None
    metadata_candidates: list[str] = []
    metadata_path = str(
        visualization_metadata.get("keyframe_manifest_path", "")
    ).strip()
    if metadata_path:
        metadata_candidates.append(metadata_path)
    metadata_parameters = visualization_metadata.get("parameters", {})
    if isinstance(metadata_parameters, Mapping):
        manifest_from_parameters = str(
            metadata_parameters.get("keyframe_manifest_path", "")
        ).strip()
        if manifest_from_parameters:
            metadata_candidates.append(manifest_from_parameters)
    for candidate in metadata_candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved.exists():
            return str(resolved)
        if fallback_missing_candidate is None:
            fallback_missing_candidate = str(resolved)

    default_manifest = _resolve_keyframe_manifest_path(
        zarr_path=zarr_path,
        parameters={"capture_keyframes": True},
    )
    if default_manifest is not None:
        resolved_default = default_manifest.expanduser().resolve()
        if resolved_default.exists():
            return str(resolved_default)

    legacy_default = _legacy_keyframe_manifest_path(zarr_path)
    if legacy_default.exists():
        return str(legacy_default.resolve())

    return fallback_missing_candidate or ""


def _resolve_compile_movie_render_manifest_path(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    render_metadata: Mapping[str, Any],
) -> str:
    """Resolve the render manifest path consumed by ``compile_movie``.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    parameters : mapping[str, Any]
        Normalized compile-movie parameters.
    render_metadata : mapping[str, Any]
        Latest render-movie metadata payload.

    Returns
    -------
    str
        Resolved JSON manifest path, or an empty string when no candidate can
        be inferred.
    """
    explicit_path = str(parameters.get("render_manifest_path", "")).strip()
    if explicit_path:
        return str(Path(explicit_path).expanduser().resolve())

    fallback_missing_candidate: Optional[str] = None
    metadata_candidates: list[str] = []
    metadata_path = str(render_metadata.get("render_manifest_path", "")).strip()
    if metadata_path:
        metadata_candidates.append(metadata_path)
    metadata_parameters = render_metadata.get("parameters", {})
    if isinstance(metadata_parameters, Mapping):
        manifest_from_parameters = str(
            metadata_parameters.get("render_manifest_path", "")
        ).strip()
        if manifest_from_parameters:
            metadata_candidates.append(manifest_from_parameters)
    output_directory = str(render_metadata.get("output_directory", "")).strip()
    if output_directory:
        metadata_candidates.append(
            str((Path(output_directory).expanduser() / "render_manifest.json"))
        )
    for candidate in metadata_candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved.exists():
            return str(resolved)
        if fallback_missing_candidate is None:
            fallback_missing_candidate = str(resolved)
    legacy_manifest = (
        _legacy_movie_output_base_directory(
            zarr_path=zarr_path,
            suffix="render_movie",
        )
        / "latest"
        / "render_manifest.json"
    ).resolve()
    if legacy_manifest.exists():
        return str(legacy_manifest)
    return fallback_missing_candidate or ""


def _normalize_captured_keyframes(value: Any) -> tuple[Dict[str, Any], ...]:
    """Normalize keyframe payloads loaded from disk.

    Parameters
    ----------
    value : Any
        Candidate keyframe sequence loaded from JSON.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Normalized keyframes sorted by capture order.
    """
    if not isinstance(value, (tuple, list)):
        return tuple()
    normalized: list[Dict[str, Any]] = []
    for order_index, row in enumerate(value):
        if not isinstance(row, Mapping):
            continue
        keyframe = {str(key): val for key, val in dict(row).items()}
        keyframe_index = int(
            max(0, int(keyframe.get("index", keyframe.get("order_index", order_index))))
        )
        keyframe["index"] = int(keyframe_index)
        keyframe["order_index"] = int(
            max(0, int(keyframe.get("order_index", keyframe_index)))
        )
        keyframe_id = str(keyframe.get("id", "")).strip()
        if not keyframe_id:
            keyframe_id = f"keyframe_{int(keyframe_index):04d}"
        keyframe["id"] = keyframe_id
        normalized.append(keyframe)
    normalized.sort(key=lambda item: (int(item["order_index"]), int(item["index"])))
    return tuple(normalized)


def _resolve_transition_frame_counts(
    *,
    keyframe_count: int,
    default_transition_frames: int,
    transition_frames_by_gap: Sequence[int],
) -> tuple[int, ...]:
    """Resolve one transition-frame count per gap between captured keyframes.

    Parameters
    ----------
    keyframe_count : int
        Number of captured keyframes.
    default_transition_frames : int
        Fallback transition count for gaps without explicit overrides.
    transition_frames_by_gap : sequence of int
        Optional per-gap transition counts.

    Returns
    -------
    tuple[int, ...]
        One non-negative transition count per gap.
    """
    gap_count = max(0, int(keyframe_count) - 1)
    resolved: list[int] = []
    for gap_index in range(gap_count):
        if gap_index < len(transition_frames_by_gap):
            resolved.append(max(0, int(transition_frames_by_gap[gap_index])))
        else:
            resolved.append(max(0, int(default_transition_frames)))
    return tuple(resolved)


def _build_movie_frame_specs(
    *,
    keyframes: Sequence[Mapping[str, Any]],
    hold_frames: int,
    transition_frames_by_gap: Sequence[int],
) -> tuple[Dict[str, Any], ...]:
    """Build ordered per-frame render specifications from captured keyframes.

    Parameters
    ----------
    keyframes : sequence of mapping[str, Any]
        Ordered captured keyframes.
    hold_frames : int
        Number of duplicate still frames to emit at each keyframe.
    transition_frames_by_gap : sequence of int
        Interpolated frame count for each inter-keyframe gap.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Ordered render specifications spanning keyframes, holds, and
        transitions.
    """
    specs: list[Dict[str, Any]] = []
    for keyframe_index in range(len(keyframes)):
        specs.append(
            {
                "kind": "keyframe",
                "keyframe_index": int(keyframe_index),
            }
        )
        for hold_index in range(max(0, int(hold_frames))):
            specs.append(
                {
                    "kind": "hold",
                    "keyframe_index": int(keyframe_index),
                    "hold_index": int(hold_index),
                }
            )
        if keyframe_index >= len(keyframes) - 1:
            continue
        gap_frame_count = int(transition_frames_by_gap[keyframe_index])
        for gap_frame_index in range(gap_frame_count):
            specs.append(
                {
                    "kind": "transition",
                    "gap_index": int(keyframe_index),
                    "start_index": int(keyframe_index),
                    "end_index": int(keyframe_index + 1),
                    "alpha": float(gap_frame_index + 1) / float(gap_frame_count + 1),
                }
            )
    return tuple(specs)


def _collect_keyframe_annotations(keyframe: Mapping[str, Any]) -> str:
    """Collect unique layer annotation strings from one captured keyframe."""
    layers = keyframe.get("layers", [])
    if not isinstance(layers, (tuple, list)):
        return ""
    annotations: list[str] = []
    for row in layers:
        if not isinstance(row, Mapping):
            continue
        text = str(row.get("annotation", "")).strip()
        if text and text not in annotations:
            annotations.append(text)
    return " | ".join(annotations)


def _frame_text_for_movie_spec(
    *,
    spec: Mapping[str, Any],
    keyframes: Sequence[Mapping[str, Any]],
    mode: str,
    frame_index: int,
    total_frames: int,
) -> str:
    """Resolve per-frame overlay text for one rendered movie frame."""
    frame_mode = str(mode).strip().lower() or "none"
    if frame_mode == "frame_number":
        return f"Frame {int(frame_index) + 1}/{int(total_frames)}"
    if frame_mode != "keyframe_annotations":
        return ""
    kind = str(spec.get("kind", "")).strip().lower()
    if kind == "transition":
        alpha = float(spec.get("alpha", 0.0))
        keyframe_index = (
            int(spec.get("end_index", 0))
            if alpha >= 0.5
            else int(spec.get("start_index", 0))
        )
    else:
        keyframe_index = int(spec.get("keyframe_index", 0))
    if keyframe_index < 0 or keyframe_index >= len(keyframes):
        return ""
    return _collect_keyframe_annotations(keyframes[keyframe_index])


def _payload_numeric_sequence(value: Any) -> Optional[tuple[float, ...]]:
    """Parse a small numeric sequence payload."""
    if isinstance(value, Mapping):
        return None
    if not isinstance(value, (tuple, list, np.ndarray)):
        return None
    parsed: list[float] = []
    for item in value:
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            return None
    return tuple(parsed)


def _payload_contrast_limits(
    payload: Mapping[str, Any],
) -> Optional[tuple[float, float]]:
    """Parse contrast-limit payloads from a captured layer row."""
    value = payload.get("contrast_limits")
    if value is None:
        return None
    sequence = _payload_numeric_sequence(value)
    if sequence is None or len(sequence) < 2:
        return None
    low = float(sequence[0])
    high = float(sequence[1])
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high <= low:
        high = low + 1.0
    return (float(low), float(high))


def _restore_overlay_layer_properties(value: Any) -> Dict[str, np.ndarray]:
    """Normalize serialized napari property mappings back into numpy arrays.

    Parameters
    ----------
    value : Any
        Candidate serialized property mapping.

    Returns
    -------
    dict[str, numpy.ndarray]
        Properties ready for napari layer construction.
    """
    if not isinstance(value, Mapping):
        return {}
    return {str(key): np.asarray(item) for key, item in value.items()}


def _apply_layer_snapshot(layer: Any, payload: Mapping[str, Any]) -> None:
    """Apply one captured layer payload to a live napari layer."""
    visible_value = payload.get("visible_override", payload.get("visible"))
    if visible_value is not None and hasattr(layer, "visible"):
        try:
            layer.visible = bool(visible_value)
        except Exception:
            pass
    if "opacity" in payload and hasattr(layer, "opacity"):
        try:
            layer.opacity = float(_safe_float(payload.get("opacity"), default=1.0))
        except Exception:
            pass
    contrast_limits = _payload_contrast_limits(payload)
    if contrast_limits is not None and hasattr(layer, "contrast_limits"):
        try:
            layer.contrast_limits = contrast_limits
        except Exception:
            pass
    for attr_name in (
        "gamma",
        "attenuation",
        "iso_threshold",
        "edge_width",
        "border_width",
        "tail_length",
        "head_length",
        "tail_width",
    ):
        if attr_name in payload and hasattr(layer, attr_name):
            try:
                setattr(
                    layer,
                    attr_name,
                    float(_safe_float(payload.get(attr_name), default=0.0)),
                )
            except Exception:
                pass
    for attr_name in ("scale", "translate", "rotate", "shear", "size"):
        if attr_name not in payload or not hasattr(layer, attr_name):
            continue
        value = payload.get(attr_name)
        if isinstance(value, Mapping):
            continue
        try:
            setattr(layer, attr_name, _sanitize_metadata_value(value))
        except Exception:
            pass
    for attr_name in (
        "blending",
        "depiction",
        "interpolation",
        "interpolation2d",
        "interpolation3d",
        "symbol",
        "face_color",
        "edge_color",
        "border_color",
    ):
        value = payload.get(attr_name)
        if value in (None, "") or not hasattr(layer, attr_name):
            continue
        try:
            setattr(layer, attr_name, _sanitize_metadata_value(value))
        except Exception:
            pass
    rendering_value = str(
        payload.get("rendering_override", payload.get("rendering", ""))
    ).strip()
    if rendering_value and hasattr(layer, "rendering"):
        try:
            layer.rendering = rendering_value
        except Exception:
            pass
    colormap_value = payload.get("colormap_override", "")
    if not colormap_value:
        colormap_payload = payload.get("colormap")
        if isinstance(colormap_payload, Mapping):
            colormap_value = str(colormap_payload.get("name", "")).strip()
        elif colormap_payload not in (None, ""):
            colormap_value = str(colormap_payload).strip()
    if colormap_value and hasattr(layer, "colormap"):
        try:
            layer.colormap = str(colormap_value)
        except Exception:
            pass


def _apply_interpolated_layer_state(
    *,
    layer: Any,
    start_payload: Optional[Mapping[str, Any]],
    end_payload: Optional[Mapping[str, Any]],
    alpha: float,
) -> None:
    """Apply interpolated layer state between two captured keyframes."""
    if start_payload is None and end_payload is None:
        return
    if start_payload is None:
        _apply_layer_snapshot(layer, end_payload or {})
        return
    if end_payload is None:
        _apply_layer_snapshot(layer, start_payload)
        return

    snap_to_end = float(alpha) >= 0.5
    interpolated_payload: Dict[str, Any] = {}

    for attr_name in ("opacity", "gamma", "attenuation", "iso_threshold"):
        if attr_name in start_payload and attr_name in end_payload:
            interpolated_payload[attr_name] = _interpolate_scalar(
                _safe_float(start_payload.get(attr_name), default=0.0),
                _safe_float(end_payload.get(attr_name), default=0.0),
                alpha,
            )
        elif attr_name in start_payload or attr_name in end_payload:
            source = end_payload if snap_to_end else start_payload
            if attr_name in source:
                interpolated_payload[attr_name] = source.get(attr_name)
    for attr_name in ("tail_length", "head_length", "tail_width"):
        if attr_name in start_payload and attr_name in end_payload:
            interpolated_payload[attr_name] = _interpolate_scalar(
                _safe_float(start_payload.get(attr_name), default=0.0),
                _safe_float(end_payload.get(attr_name), default=0.0),
                alpha,
            )
        else:
            source = end_payload if snap_to_end else start_payload
            if attr_name in source:
                interpolated_payload[attr_name] = source.get(attr_name)

    for attr_name in ("scale", "translate", "rotate", "shear", "size"):
        start_vector = _payload_numeric_sequence(start_payload.get(attr_name))
        end_vector = _payload_numeric_sequence(end_payload.get(attr_name))
        if (
            start_vector is not None
            and end_vector is not None
            and len(start_vector) == len(end_vector)
        ):
            interpolated_payload[attr_name] = list(
                _interpolate_vector(start_vector, end_vector, alpha)
            )
        else:
            source = end_payload if snap_to_end else start_payload
            if attr_name in source:
                interpolated_payload[attr_name] = source.get(attr_name)

    start_limits = _payload_contrast_limits(start_payload)
    end_limits = _payload_contrast_limits(end_payload)
    if start_limits is not None and end_limits is not None:
        interpolated_payload["contrast_limits"] = list(
            _interpolate_vector(start_limits, end_limits, alpha)
        )
    else:
        source = end_payload if snap_to_end else start_payload
        if "contrast_limits" in source:
            interpolated_payload["contrast_limits"] = source.get("contrast_limits")

    for attr_name in (
        "visible",
        "visible_override",
        "blending",
        "rendering",
        "rendering_override",
        "depiction",
        "interpolation",
        "interpolation2d",
        "interpolation3d",
        "symbol",
        "face_color",
        "edge_color",
        "border_color",
        "colormap",
        "colormap_override",
    ):
        source = end_payload if snap_to_end else start_payload
        if attr_name in source:
            interpolated_payload[attr_name] = source.get(attr_name)

    _apply_layer_snapshot(layer, interpolated_payload)


def _process_pending_qt_events() -> None:
    """Flush pending Qt events between scripted viewer updates."""
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass
    time.sleep(0.01)


def _resize_movie_viewer_canvas(
    *,
    viewer: Any,
    render_size_xy: Sequence[int],
) -> None:
    """Best-effort resize of the napari canvas before movie screenshots.

    Parameters
    ----------
    viewer : Any
        Live napari viewer instance.
    render_size_xy : sequence of int
        Requested movie frame size in ``(x, y)`` order.

    Returns
    -------
    None
        Applies resize side effects to the viewer when the underlying Qt
        widgets are available.
    """
    if len(render_size_xy) < 2:
        return
    width = max(1, int(render_size_xy[0]))
    height = max(1, int(render_size_xy[1]))
    try:
        window = getattr(viewer, "window", None)
        qt_viewer = getattr(window, "qt_viewer", None)
        canvas = getattr(qt_viewer, "canvas", None)
        native_canvas = getattr(canvas, "native", None)
        if native_canvas is not None and hasattr(native_canvas, "resize"):
            native_canvas.resize(width, height)
        if window is not None and hasattr(window, "resize"):
            window.resize(width + 200, height + 50)
    except Exception:
        pass
    _process_pending_qt_events()


def _set_napari_async_slice_loading(enabled: bool) -> Optional[bool]:
    """Set napari async slice loading and return the previous value.

    Parameters
    ----------
    enabled : bool
        Desired async slice-loading state.

    Returns
    -------
    bool, optional
        Previous async state when napari settings are available, otherwise
        ``None``.
    """
    try:
        from napari.settings import get_settings

        settings = get_settings()
        experimental = getattr(settings, "experimental", None)
        if experimental is None or not hasattr(experimental, "async_"):
            return None
        previous = bool(getattr(experimental, "async_"))
        setattr(experimental, "async_", bool(enabled))
        return previous
    except Exception:
        return None


def _refresh_movie_viewer_layers(viewer: Any) -> None:
    """Synchronously refresh napari layers before capturing a movie frame.

    Parameters
    ----------
    viewer : Any
        Live napari viewer instance.

    Returns
    -------
    None
        Forces layer refresh side effects when supported by the active napari
        layer classes.
    """
    for layer in tuple(getattr(viewer, "layers", tuple())):
        refresh = getattr(layer, "refresh", None)
        if not callable(refresh):
            continue
        try:
            refresh(
                thumbnail=False,
                data_displayed=True,
                highlight=False,
                extent=False,
                force=True,
            )
        except TypeError:
            try:
                refresh(force=True)
            except TypeError:
                try:
                    refresh()
                except Exception:
                    pass
        except Exception:
            pass
    _process_pending_qt_events()
    time.sleep(0.05)
    _process_pending_qt_events()


def _build_movie_render_viewer_scene(
    *,
    zarr_path: Union[str, Path],
    scene: PreparedVisualizationScene,
    movie_level_index: int,
) -> tuple[BuiltNapariScene, bool]:
    """Build a visible napari viewer for movie capture.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    scene : PreparedVisualizationScene
        Prepared visualization scene shared with interactive visualization.
    movie_level_index : int
        Requested movie resolution level.

    Returns
    -------
    tuple[BuiltNapariScene, bool]
        The built viewer scene and a flag indicating that the movie viewer is
        visible.
    """
    common_kwargs: Dict[str, Any] = {
        "zarr_path": zarr_path,
        "volume_layers": scene.volume_layers,
        "selected_positions": scene.selected_positions,
        "points_by_position": scene.points_by_position,
        "point_properties_by_position": scene.point_properties_by_position,
        "position_affines_tczyx": scene.position_affines_tczyx,
        "axis_labels": scene.napari_payload.axis_labels_tczyx,
        "scale_tczyx": scene.napari_payload.scale_tczyx,
        "image_metadata": scene.napari_payload.image_metadata,
        "points_metadata": scene.napari_payload.points_metadata,
        "require_gpu_rendering": False,
        "capture_keyframes": False,
        "keyframe_manifest_path": None,
        "movie_level_index": int(movie_level_index),
    }
    built_scene = _build_napari_viewer_scene(
        **common_kwargs,
        show=True,
    )
    return built_scene, True


def _restore_captured_overlay_layers(
    viewer: Any,
    captured_layers: Any,
) -> None:
    """Rebuild serialized overlay layers before scripted movie rendering.

    Parameters
    ----------
    viewer : Any
        Live napari viewer instance.
    captured_layers : Any
        Serialized overlay-layer payloads from the keyframe manifest.

    Returns
    -------
    None
        Adds restorable overlay layers to the viewer for side effects only.
    """
    if not isinstance(captured_layers, (tuple, list)):
        return

    existing_names = {
        str(getattr(layer, "name", "")).strip()
        for layer in tuple(getattr(viewer, "layers", tuple()))
        if str(getattr(layer, "name", "")).strip()
    }
    for row in captured_layers:
        if not isinstance(row, Mapping):
            continue
        layer_name = str(row.get("name", "")).strip()
        if not layer_name or layer_name in existing_names:
            continue
        layer_type = str(row.get("type", "")).strip().lower()
        layer_data = np.asarray(row.get("data", []))
        metadata = (
            {
                str(key): _sanitize_metadata_value(value)
                for key, value in row.get("metadata", {}).items()
            }
            if isinstance(row.get("metadata"), Mapping)
            else {}
        )
        properties = _restore_overlay_layer_properties(row.get("properties"))
        try:
            if layer_type == "points":
                layer = viewer.add_points(
                    layer_data,
                    name=layer_name,
                    metadata=metadata,
                    properties=properties,
                )
            elif layer_type == "tracks":
                kwargs: Dict[str, Any] = {
                    "name": layer_name,
                    "metadata": metadata,
                    "properties": properties,
                }
                if "graph" in row:
                    kwargs["graph"] = _sanitize_metadata_value(row.get("graph"))
                layer = viewer.add_tracks(layer_data, **kwargs)
            else:
                continue
        except Exception:
            continue
        _apply_layer_snapshot(layer, row)
        existing_names.add(layer_name)


def _apply_viewer_snapshot(viewer: Any, keyframe: Mapping[str, Any]) -> None:
    """Apply one captured keyframe exactly to a live napari viewer."""
    dims_payload = keyframe.get("dims", {})
    if isinstance(dims_payload, Mapping):
        requested_ndisplay = int(
            max(2, int(_safe_float(dims_payload.get("ndisplay", 3), default=3.0)))
        )
        try:
            viewer.dims.ndisplay = 3 if requested_ndisplay >= 3 else 2
        except Exception:
            pass
        current_step_value = dims_payload.get("current_step")
        current_step = _payload_numeric_sequence(current_step_value)
        if current_step is not None:
            existing = tuple(getattr(viewer.dims, "current_step", tuple()))
            padded = list(int(round(value)) for value in existing)
            for index, value in enumerate(current_step[: len(padded)]):
                padded[index] = int(round(float(value)))
            try:
                viewer.dims.current_step = tuple(padded)
            except Exception:
                pass
        dims_order = _payload_numeric_sequence(dims_payload.get("order"))
        if dims_order is not None:
            try:
                viewer.dims.order = tuple(int(round(value)) for value in dims_order)
            except Exception:
                pass

    camera_payload = keyframe.get("camera", {})
    if isinstance(camera_payload, Mapping):
        angles = _payload_numeric_sequence(camera_payload.get("angles"))
        if angles is not None and len(angles) >= 3 and hasattr(viewer.camera, "angles"):
            try:
                viewer.camera.angles = tuple(float(value) for value in angles[:3])
            except Exception:
                pass
        center = _payload_numeric_sequence(camera_payload.get("center"))
        if center is not None and len(center) >= 3 and hasattr(viewer.camera, "center"):
            try:
                viewer.camera.center = tuple(float(value) for value in center[:3])
            except Exception:
                pass
        if "zoom" in camera_payload and hasattr(viewer.camera, "zoom"):
            try:
                viewer.camera.zoom = float(
                    _safe_float(camera_payload.get("zoom"), default=1.0)
                )
            except Exception:
                pass
        if "perspective" in camera_payload and hasattr(viewer.camera, "perspective"):
            try:
                viewer.camera.perspective = float(
                    _safe_float(camera_payload.get("perspective"), default=0.0)
                )
            except Exception:
                pass
        field_of_view = camera_payload.get(
            "field_of_view",
            camera_payload.get("fov"),
        )
        if field_of_view is not None and hasattr(viewer.camera, "fov"):
            try:
                viewer.camera.fov = float(_safe_float(field_of_view, default=0.0))
            except Exception:
                pass

    layer_map = {
        str(getattr(layer, "name", "")): layer
        for layer in tuple(getattr(viewer, "layers", tuple()))
    }
    layers_payload = keyframe.get("layers", [])
    if isinstance(layers_payload, (tuple, list)):
        for payload in layers_payload:
            if not isinstance(payload, Mapping):
                continue
            layer_name = str(payload.get("name", "")).strip()
            if not layer_name:
                continue
            layer = layer_map.get(layer_name)
            if layer is None:
                continue
            _apply_layer_snapshot(layer, payload)


def _apply_viewer_interpolated_state(
    *,
    viewer: Any,
    start_keyframe: Mapping[str, Any],
    end_keyframe: Mapping[str, Any],
    alpha: float,
    parameters: Mapping[str, Any],
) -> None:
    """Apply interpolated viewer state between two captured keyframes."""
    start_dims = start_keyframe.get("dims", {})
    end_dims = end_keyframe.get("dims", {})
    start_ndisplay = (
        int(max(2, int(_safe_float(start_dims.get("ndisplay", 3), default=3.0))))
        if isinstance(start_dims, Mapping)
        else 3
    )
    end_ndisplay = (
        int(max(2, int(_safe_float(end_dims.get("ndisplay", 3), default=3.0))))
        if isinstance(end_dims, Mapping)
        else start_ndisplay
    )
    target_ndisplay = (
        end_ndisplay
        if float(alpha) >= 0.5 and end_ndisplay != start_ndisplay
        else start_ndisplay
    )
    try:
        viewer.dims.ndisplay = 3 if target_ndisplay >= 3 else 2
    except Exception:
        pass

    if isinstance(start_dims, Mapping) and isinstance(end_dims, Mapping):
        start_step = _payload_numeric_sequence(start_dims.get("current_step"))
        end_step = _payload_numeric_sequence(end_dims.get("current_step"))
        existing = list(tuple(getattr(viewer.dims, "current_step", tuple())))
        if (
            start_step is not None
            and end_step is not None
            and len(start_step) == len(end_step)
            and len(existing) >= len(start_step)
        ):
            current_step = list(existing)
            for index, value in enumerate(
                _interpolate_vector(start_step, end_step, alpha)
            ):
                current_step[index] = int(round(float(value)))
            try:
                viewer.dims.current_step = tuple(current_step)
            except Exception:
                pass
        else:
            source = end_dims if float(alpha) >= 0.5 else start_dims
            source_step = _payload_numeric_sequence(source.get("current_step"))
            if source_step is not None and len(existing) >= len(source_step):
                current_step = list(existing)
                for index, value in enumerate(source_step):
                    current_step[index] = int(round(float(value)))
                try:
                    viewer.dims.current_step = tuple(current_step)
                except Exception:
                    pass
        dims_order = (
            _payload_numeric_sequence(end_dims.get("order"))
            if float(alpha) >= 0.5
            else _payload_numeric_sequence(start_dims.get("order"))
        )
        if dims_order is not None:
            try:
                viewer.dims.order = tuple(int(round(value)) for value in dims_order)
            except Exception:
                pass

    start_camera = start_keyframe.get("camera", {})
    end_camera = end_keyframe.get("camera", {})
    if isinstance(start_camera, Mapping) and isinstance(end_camera, Mapping):
        start_angles = _payload_numeric_sequence(start_camera.get("angles"))
        end_angles = _payload_numeric_sequence(end_camera.get("angles"))
        if (
            target_ndisplay >= 3
            and start_ndisplay >= 3
            and end_ndisplay >= 3
            and start_angles is not None
            and end_angles is not None
            and len(start_angles) >= 3
            and len(end_angles) >= 3
        ):
            angles = _slerp_camera_angles(
                start_angles[:3],
                end_angles[:3],
                [float(alpha)],
            )[0]
        else:
            source_angles = end_angles if float(alpha) >= 0.5 else start_angles
            if source_angles is None:
                source_angles = start_angles or end_angles
            angles = tuple(float(value) for value in tuple(source_angles or ())[:3])
        if len(angles) >= 3:
            if str(parameters.get("camera_effect", "none")).strip().lower() == "orbit":
                angles = (
                    float(angles[0]),
                    float(angles[1])
                    + (
                        float(parameters.get("orbit_degrees", 45.0))
                        * (float(alpha) - 0.5)
                    ),
                    float(angles[2]),
                )
            try:
                viewer.camera.angles = tuple(float(value) for value in angles[:3])
            except Exception:
                pass

        start_center = _payload_numeric_sequence(start_camera.get("center"))
        end_center = _payload_numeric_sequence(end_camera.get("center"))
        if (
            start_center is not None
            and end_center is not None
            and len(start_center) >= 3
            and len(end_center) >= 3
        ):
            center = np.asarray(
                _interpolate_vector(start_center[:3], end_center[:3], alpha),
                dtype=np.float64,
            )
            if (
                str(parameters.get("camera_effect", "none")).strip().lower()
                == "flythrough"
            ):
                delta = np.asarray(end_center[:3], dtype=np.float64) - np.asarray(
                    start_center[:3],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(delta))
                if norm > 0.0:
                    center = center + (
                        (delta / norm)
                        * norm
                        * float(parameters.get("flythrough_distance_factor", 0.10))
                        * math.sin(math.pi * float(alpha))
                    )
            try:
                viewer.camera.center = tuple(float(value) for value in center[:3])
            except Exception:
                pass

        start_zoom = float(_safe_float(start_camera.get("zoom"), default=1.0))
        end_zoom = float(_safe_float(end_camera.get("zoom"), default=start_zoom))
        zoom = _interpolate_scalar(start_zoom, end_zoom, alpha)
        if str(parameters.get("camera_effect", "none")).strip().lower() == "zoom_fx":
            zoom = float(
                zoom
                * (
                    1.0
                    + (
                        float(parameters.get("zoom_effect_factor", 0.15))
                        * math.sin(math.pi * float(alpha))
                    )
                )
            )
        try:
            viewer.camera.zoom = float(zoom)
        except Exception:
            pass

        for attr_name, camera_key in (
            ("perspective", "perspective"),
            ("fov", "field_of_view"),
        ):
            if not hasattr(viewer.camera, attr_name):
                continue
            start_value = float(_safe_float(start_camera.get(camera_key), default=0.0))
            end_value = float(
                _safe_float(end_camera.get(camera_key), default=start_value)
            )
            try:
                setattr(
                    viewer.camera,
                    attr_name,
                    _interpolate_scalar(start_value, end_value, alpha),
                )
            except Exception:
                pass

    start_layers_raw = start_keyframe.get("layers", [])
    end_layers_raw = end_keyframe.get("layers", [])
    start_layers = {
        str(row.get("name", "")).strip(): row
        for row in start_layers_raw
        if isinstance(row, Mapping) and str(row.get("name", "")).strip()
    }
    end_layers = {
        str(row.get("name", "")).strip(): row
        for row in end_layers_raw
        if isinstance(row, Mapping) and str(row.get("name", "")).strip()
    }
    for layer in tuple(getattr(viewer, "layers", tuple())):
        layer_name = str(getattr(layer, "name", "")).strip()
        if not layer_name:
            continue
        _apply_interpolated_layer_state(
            layer=layer,
            start_payload=start_layers.get(layer_name),
            end_payload=end_layers.get(layer_name),
            alpha=float(alpha),
        )


def _apply_movie_frame_state(
    *,
    viewer: Any,
    spec: Mapping[str, Any],
    keyframes: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> None:
    """Apply one movie-frame specification to a scripted napari viewer."""
    kind = str(spec.get("kind", "")).strip().lower()
    if kind in {"keyframe", "hold"}:
        keyframe_index = int(spec.get("keyframe_index", 0))
        if 0 <= keyframe_index < len(keyframes):
            _apply_viewer_snapshot(viewer, keyframes[keyframe_index])
        return
    if kind != "transition":
        raise ValueError(f"Unsupported movie frame kind: {kind}")
    start_index = int(spec.get("start_index", 0))
    end_index = int(spec.get("end_index", 0))
    if start_index < 0 or end_index < 0:
        return
    if start_index >= len(keyframes) or end_index >= len(keyframes):
        return
    alpha = _ease_fraction(
        float(spec.get("alpha", 0.0)),
        str(parameters.get("interpolation_mode", "ease_in_out")),
    )
    _apply_viewer_interpolated_state(
        viewer=viewer,
        start_keyframe=keyframes[start_index],
        end_keyframe=keyframes[end_index],
        alpha=float(alpha),
        parameters=parameters,
    )


def _viewer_pixel_size_x_um(viewer: Any) -> Optional[float]:
    """Estimate the current rendered x-axis pixel size from live layer scale.

    Parameters
    ----------
    viewer : Any
        Live napari viewer instance.

    Returns
    -------
    float, optional
        Rendered x-axis pixel size in microns when available.
    """
    for layer in tuple(getattr(viewer, "layers", tuple())):
        scale_value = _payload_numeric_sequence(getattr(layer, "scale", None))
        if scale_value is None or len(scale_value) <= 0:
            continue
        return float(scale_value[-1])
    return None


def _resolve_default_movie_stem(zarr_path: Union[str, Path]) -> str:
    """Resolve a stable movie filename stem from the current store name.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.

    Returns
    -------
    str
        Default movie filename stem.
    """
    store_name = Path(zarr_path).expanduser().resolve().name
    if store_name.endswith(".ome.zarr"):
        store_name = store_name[: -len(".ome.zarr")]
    elif store_name.endswith(".zarr"):
        store_name = store_name[: -len(".zarr")]
    return str(store_name).strip() or "clearex_movie"


def _save_render_movie_metadata(
    *,
    zarr_path: Union[str, Path],
    visualization_component: str,
    keyframe_manifest_path: str,
    render_manifest_path: str,
    output_directory: str,
    rendered_levels: Sequence[int],
    frame_count: int,
    fps: int,
    frame_directories_by_level: Mapping[str, str],
    parameters: Mapping[str, Any],
    source_component: str,
    source_components: Sequence[str],
    run_id: Optional[str] = None,
) -> str:
    """Persist render-movie metadata in ``clearex/results/render_movie/latest``."""
    root = zarr.open_group(str(zarr_path), mode="a")
    component = analysis_auxiliary_root("render_movie")
    latest_group = ensure_group(root, component)
    _clear_group_attrs(latest_group)
    payload: Dict[str, Any] = {
        "visualization_component": str(visualization_component),
        "keyframe_manifest_path": str(keyframe_manifest_path),
        "render_manifest_path": str(render_manifest_path),
        "output_directory": str(output_directory),
        "rendered_levels": [int(value) for value in rendered_levels],
        "frame_count": int(max(0, frame_count)),
        "fps": int(max(1, fps)),
        "frame_directories_by_level": {
            str(key): str(value) for key, value in frame_directories_by_level.items()
        },
        "parameters": {str(key): value for key, value in dict(parameters).items()},
        "source_component": str(source_component),
        "source_components": [str(item) for item in source_components],
        "storage_policy": "latest_only",
        "run_id": run_id,
    }
    latest_group.attrs.update(_sanitize_metadata_value(payload))
    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="render_movie",
        component=component,
        run_id=run_id,
        metadata=payload,
    )
    return component


def _save_compile_movie_metadata(
    *,
    zarr_path: Union[str, Path],
    render_component: str,
    render_manifest_path: str,
    output_directory: str,
    rendered_level: int,
    output_format: str,
    compiled_files: Sequence[str],
    fps: int,
    parameters: Mapping[str, Any],
    run_id: Optional[str] = None,
) -> str:
    """Persist compile-movie metadata in ``clearex/results/compile_movie/latest``."""
    root = zarr.open_group(str(zarr_path), mode="a")
    component = analysis_auxiliary_root("compile_movie")
    latest_group = ensure_group(root, component)
    _clear_group_attrs(latest_group)
    payload: Dict[str, Any] = {
        "render_component": str(render_component),
        "render_manifest_path": str(render_manifest_path),
        "output_directory": str(output_directory),
        "rendered_level": int(max(0, rendered_level)),
        "output_format": str(output_format),
        "compiled_files": [str(path) for path in compiled_files],
        "fps": int(max(1, fps)),
        "parameters": {str(key): value for key, value in dict(parameters).items()},
        "storage_policy": "latest_only",
        "run_id": run_id,
    }
    latest_group.attrs.update(_sanitize_metadata_value(payload))
    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="compile_movie",
        component=component,
        run_id=run_id,
        metadata=payload,
    )
    return component


def _render_movie_summary_from_latest_metadata(
    *,
    zarr_path: Union[str, Path],
) -> RenderMovieSummary:
    """Build a render-movie summary from the latest persisted metadata.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.

    Returns
    -------
    RenderMovieSummary
        Parsed latest render-movie summary.
    """
    metadata = _read_latest_analysis_metadata(
        zarr_path=zarr_path,
        component=analysis_auxiliary_root("render_movie"),
        operation_name="render_movie",
    )
    rendered_levels_raw = metadata.get("rendered_levels", [])
    if not isinstance(rendered_levels_raw, (tuple, list)):
        rendered_levels_raw = [rendered_levels_raw]
    return RenderMovieSummary(
        component=str(
            metadata.get("component", analysis_auxiliary_root("render_movie"))
        ).strip()
        or analysis_auxiliary_root("render_movie"),
        visualization_component=str(
            metadata.get(
                "visualization_component",
                analysis_auxiliary_root("visualization"),
            )
        ).strip()
        or analysis_auxiliary_root("visualization"),
        keyframe_manifest_path=str(metadata.get("keyframe_manifest_path", "")).strip(),
        render_manifest_path=str(metadata.get("render_manifest_path", "")).strip(),
        output_directory=str(metadata.get("output_directory", "")).strip(),
        rendered_levels=tuple(int(value) for value in rendered_levels_raw),
        frame_count=max(0, int(metadata.get("frame_count", 0))),
        fps=max(1, int(metadata.get("fps", 24))),
    )


def run_render_movie_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> RenderMovieSummary:
    """Render smooth frame sequences from captured napari keyframes.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    parameters : mapping[str, Any]
        Render-movie parameter mapping.
    progress_callback : callable, optional
        Optional callback receiving ``(percent, message)`` progress updates.
    run_id : str, optional
        Provenance run identifier when available.

    Returns
    -------
    RenderMovieSummary
        Summary of rendered frame sets and latest metadata paths.

    Raises
    ------
    ValueError
        If required visualization metadata or keyframes are missing.
    RuntimeError
        If a render subprocess fails.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_render_movie_parameters(parameters)
    effective_launch_mode = _resolve_effective_launch_mode(
        str(normalized.get("launch_mode", "auto"))
    )
    if effective_launch_mode == "subprocess":
        _emit(5, "Launching visible napari movie renderer in a separate process")
        subprocess_parameters: Dict[str, Any] = {
            **normalized,
            "launch_mode": "in_process",
        }
        _run_render_movie_subprocess(
            zarr_path=zarr_path,
            normalized_parameters=subprocess_parameters,
            run_id=run_id,
        )
        _emit(95, "Collecting rendered movie metadata from subprocess")
        _emit(100, "Render-movie workflow complete")
        return _render_movie_summary_from_latest_metadata(zarr_path=zarr_path)
    return _run_render_movie_analysis_in_process(
        zarr_path=zarr_path,
        parameters=normalized,
        progress_callback=progress_callback,
        run_id=run_id,
    )


def _run_render_movie_analysis_in_process(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> RenderMovieSummary:
    """Render smooth frame sequences from captured napari keyframes.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    parameters : mapping[str, Any]
        Render-movie parameter mapping.
    progress_callback : callable, optional
        Optional callback receiving ``(percent, message)`` progress updates.
    run_id : str, optional
        Provenance run identifier when available.

    Returns
    -------
    RenderMovieSummary
        Summary of rendered frame sets and latest metadata paths.

    Raises
    ------
    ValueError
        If required visualization metadata or keyframes are missing.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = dict(parameters)
    visualization_component = resolve_analysis_input_component(
        str(
            normalized.get("input_source", analysis_auxiliary_root("visualization"))
        ).strip()
        or analysis_auxiliary_root("visualization")
    )
    visualization_metadata = _read_latest_analysis_metadata(
        zarr_path=zarr_path,
        component=visualization_component,
        operation_name="render_movie",
    )
    visualization_parameters = visualization_metadata.get("parameters", {})
    if not isinstance(visualization_parameters, Mapping):
        visualization_parameters = {}

    keyframe_manifest_candidate = _resolve_render_movie_keyframe_manifest_path(
        zarr_path=zarr_path,
        parameters=normalized,
        visualization_metadata=visualization_metadata,
    )
    if not keyframe_manifest_candidate:
        raise ValueError(
            "render_movie requires a keyframe manifest path. Capture keyframes "
            "in visualization first or provide keyframe_manifest_path explicitly."
        )
    keyframe_manifest_path = str(
        Path(keyframe_manifest_candidate).expanduser().resolve()
    )
    keyframe_manifest = _read_json_mapping(keyframe_manifest_path)
    keyframes = _normalize_captured_keyframes(keyframe_manifest.get("keyframes", []))
    if len(keyframes) <= 0:
        raise ValueError(
            f"render_movie found no keyframes in '{keyframe_manifest_path}'."
        )

    scene_parameters: Dict[str, Any] = dict(visualization_parameters)
    if not scene_parameters:
        scene_parameters = {
            "input_source": str(
                visualization_metadata.get(
                    "source_component",
                    keyframe_manifest.get("source_component", "data"),
                )
            ).strip()
            or "data",
            "volume_layers": keyframe_manifest.get(
                "volume_layers",
                visualization_metadata.get("volume_layers", []),
            ),
            "position_index": int(visualization_metadata.get("position_index", 0)),
            "show_all_positions": bool(
                visualization_metadata.get("show_all_positions", False)
            ),
            "use_multiscale": True,
            "use_3d_view": bool(
                int(visualization_metadata.get("viewer_ndisplay_effective", 3)) >= 3
            ),
            "overlay_particle_detections": bool(
                int(visualization_metadata.get("overlay_points_count", 0)) > 0
            ),
        }
    scene_parameters["capture_keyframes"] = False
    scene_parameters["require_gpu_rendering"] = False
    scene_parameters["launch_mode"] = "in_process"

    _emit(5, "Preparing render-movie source scene")
    scene = _prepare_visualization_scene(
        zarr_path=zarr_path,
        parameters=scene_parameters,
        progress_callback=None,
    )

    transition_frames = _resolve_transition_frame_counts(
        keyframe_count=len(keyframes),
        default_transition_frames=int(normalized.get("default_transition_frames", 48)),
        transition_frames_by_gap=tuple(
            int(value) for value in normalized.get("transition_frames_by_gap", [])
        ),
    )
    frame_specs = _build_movie_frame_specs(
        keyframes=keyframes,
        hold_frames=int(normalized.get("hold_frames", 12)),
        transition_frames_by_gap=transition_frames,
    )
    if len(frame_specs) <= 0:
        raise ValueError("render_movie did not produce any frame specifications.")

    output_base_directory = _resolve_movie_output_base_directory(
        zarr_path=zarr_path,
        output_directory=str(normalized.get("output_directory", "")),
        suffix="render_movie",
    )
    output_directory = _prepare_movie_output_directory(output_base_directory)
    render_manifest_path = (output_directory / "render_manifest.json").resolve()

    rendered_levels = tuple(
        int(value) for value in normalized.get("resolution_levels", [0])
    )
    render_size_xy = tuple(
        int(value) for value in normalized.get("render_size_xy", [1920, 1080])
    )
    frame_directories_by_level: Dict[str, str] = {}
    level_manifest_rows: list[Dict[str, Any]] = []
    total_levels = max(1, len(rendered_levels))
    total_frames = max(1, len(frame_specs))
    previous_async_slice_loading = _set_napari_async_slice_loading(False)

    try:
        for level_offset, level_index in enumerate(rendered_levels):
            level_progress_start = 10 + int(
                (float(level_offset) / float(total_levels)) * 75.0
            )
            level_progress_end = 10 + int(
                (float(level_offset + 1) / float(total_levels)) * 75.0
            )
            frames_directory = (
                output_directory / f"level_{int(level_index):02d}_frames"
            ).resolve()
            frames_directory.mkdir(parents=True, exist_ok=True)
            frame_directories_by_level[str(level_index)] = str(frames_directory)

            _emit(
                level_progress_start,
                f"Rendering movie frames for resolution level {int(level_index)}",
            )
            built_scene, using_visible_viewer = _build_movie_render_viewer_scene(
                zarr_path=zarr_path,
                scene=scene,
                movie_level_index=int(level_index),
            )
            viewer = built_scene.viewer
            if using_visible_viewer:
                _emit(
                    level_progress_start,
                    "Using visible napari viewer for movie capture",
                )
            try:

                def _prepare_viewer_for_capture(active_viewer: Any) -> None:
                    """Restore overlays and resize one movie-capture viewer."""
                    _restore_captured_overlay_layers(
                        active_viewer,
                        keyframe_manifest.get("captured_overlay_layers", []),
                    )
                    _resize_movie_viewer_canvas(
                        viewer=active_viewer,
                        render_size_xy=render_size_xy,
                    )

                def _capture_frame_rgba(
                    *,
                    active_viewer: Any,
                    spec: Mapping[str, Any],
                    frame_index: int,
                ) -> npt.NDArray[np.uint8]:
                    """Apply one frame specification and capture the screenshot."""
                    _apply_movie_frame_state(
                        viewer=active_viewer,
                        spec=spec,
                        keyframes=keyframes,
                        parameters=normalized,
                    )
                    _refresh_movie_viewer_layers(active_viewer)
                    screenshot = active_viewer.screenshot(
                        canvas_only=True,
                        flash=False,
                        size=(int(render_size_xy[1]), int(render_size_xy[0])),
                    )
                    frame_text = _frame_text_for_movie_spec(
                        spec=spec,
                        keyframes=keyframes,
                        mode=str(normalized.get("overlay_frame_text_mode", "none")),
                        frame_index=int(frame_index),
                        total_frames=int(total_frames),
                    )
                    return _movie_overlay_rgba(
                        _ensure_rgba_frame(np.asarray(screenshot)),
                        title=str(normalized.get("overlay_title", "")),
                        subtitle=str(normalized.get("overlay_subtitle", "")),
                        frame_text=frame_text,
                        scalebar_enabled=bool(
                            normalized.get("overlay_scalebar", False)
                        ),
                        scalebar_length_um=float(
                            normalized.get("overlay_scalebar_length_um", 50.0)
                        ),
                        scalebar_position=str(
                            normalized.get("overlay_scalebar_position", "bottom_left")
                        ),
                        pixel_size_x_um=_viewer_pixel_size_x_um(active_viewer),
                    )

                _prepare_viewer_for_capture(viewer)
                first_frame_rgba = _capture_frame_rgba(
                    active_viewer=viewer,
                    spec=frame_specs[0],
                    frame_index=0,
                )
                if not _frame_has_visible_rgb_content(first_frame_rgba):
                    raise RuntimeError(
                        "render_movie captured an empty napari probe frame from "
                        "the visible viewer. Check the saved visualization "
                        "keyframes and source layer visibility."
                    )

                for frame_index, spec in enumerate(frame_specs):
                    if int(frame_index) == 0:
                        rgba = np.asarray(first_frame_rgba, dtype=np.uint8)
                    else:
                        rgba = _capture_frame_rgba(
                            active_viewer=viewer,
                            spec=spec,
                            frame_index=int(frame_index),
                        )
                    frame_path = frames_directory / f"frame_{int(frame_index):06d}.png"
                    Image.fromarray(rgba, mode="RGBA").save(frame_path)
                    mapped = level_progress_start + int(
                        (
                            (float(frame_index + 1) / float(total_frames))
                            * max(
                                1,
                                level_progress_end - level_progress_start,
                            )
                        )
                    )
                    _emit(
                        mapped,
                        f"Rendered frame {int(frame_index) + 1}/{int(total_frames)} "
                        f"for level {int(level_index)}",
                    )
            finally:
                try:
                    viewer.close()
                except Exception:
                    pass
                _process_pending_qt_events()

            level_manifest_rows.append(
                {
                    "requested_level": int(level_index),
                    "frame_directory": str(frames_directory),
                    "frame_pattern": "frame_%06d.png",
                    "frame_count": int(total_frames),
                    "render_size_xy": [int(render_size_xy[0]), int(render_size_xy[1])],
                }
            )
    finally:
        if previous_async_slice_loading is not None:
            _set_napari_async_slice_loading(previous_async_slice_loading)

    render_manifest_payload: Dict[str, Any] = {
        "schema_version": 1,
        "zarr_path": str(Path(zarr_path).expanduser().resolve()),
        "visualization_component": str(visualization_component),
        "keyframe_manifest_path": str(keyframe_manifest_path),
        "source_component": str(scene.source_component),
        "source_components": [str(item) for item in scene.source_components],
        "selected_positions": [int(value) for value in scene.selected_positions],
        "fps": int(normalized.get("fps", 24)),
        "render_size_xy": [int(render_size_xy[0]), int(render_size_xy[1])],
        "rendered_levels": [int(value) for value in rendered_levels],
        "default_transition_frames": int(
            normalized.get("default_transition_frames", 48)
        ),
        "transition_frames_by_gap": [int(value) for value in transition_frames],
        "hold_frames": int(normalized.get("hold_frames", 12)),
        "interpolation_mode": str(normalized.get("interpolation_mode", "ease_in_out")),
        "camera_effect": str(normalized.get("camera_effect", "none")),
        "camera_effect_parameters": {
            "orbit_degrees": float(normalized.get("orbit_degrees", 45.0)),
            "flythrough_distance_factor": float(
                normalized.get("flythrough_distance_factor", 0.10)
            ),
            "zoom_effect_factor": float(normalized.get("zoom_effect_factor", 0.15)),
        },
        "overlay": {
            "title": str(normalized.get("overlay_title", "")),
            "subtitle": str(normalized.get("overlay_subtitle", "")),
            "frame_text_mode": str(normalized.get("overlay_frame_text_mode", "none")),
            "scalebar_enabled": bool(normalized.get("overlay_scalebar", False)),
            "scalebar_length_um": float(
                normalized.get("overlay_scalebar_length_um", 50.0)
            ),
            "scalebar_position": str(
                normalized.get("overlay_scalebar_position", "bottom_left")
            ),
        },
        "frame_count": int(total_frames),
        "keyframe_ids": [str(keyframe.get("id", "")) for keyframe in keyframes],
        "levels": level_manifest_rows,
        "parameters": {str(key): value for key, value in normalized.items()},
    }
    _write_json_mapping(render_manifest_path, render_manifest_payload)

    component = _save_render_movie_metadata(
        zarr_path=zarr_path,
        visualization_component=visualization_component,
        keyframe_manifest_path=keyframe_manifest_path,
        render_manifest_path=str(render_manifest_path),
        output_directory=str(output_directory),
        rendered_levels=rendered_levels,
        frame_count=int(total_frames),
        fps=int(normalized.get("fps", 24)),
        frame_directories_by_level=frame_directories_by_level,
        parameters=normalized,
        source_component=str(scene.source_component),
        source_components=tuple(str(item) for item in scene.source_components),
        run_id=run_id,
    )
    _emit(100, "Render-movie workflow complete")
    return RenderMovieSummary(
        component=component,
        visualization_component=str(visualization_component),
        keyframe_manifest_path=str(keyframe_manifest_path),
        render_manifest_path=str(render_manifest_path),
        output_directory=str(output_directory),
        rendered_levels=tuple(int(value) for value in rendered_levels),
        frame_count=int(total_frames),
        fps=int(normalized.get("fps", 24)),
    )


def run_compile_movie_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> CompileMovieSummary:
    """Compile rendered PNG frames into movie files with ffmpeg.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    parameters : mapping[str, Any]
        Compile-movie parameter mapping.
    progress_callback : callable, optional
        Optional callback receiving ``(percent, message)`` progress updates.
    run_id : str, optional
        Provenance run identifier when available.

    Returns
    -------
    CompileMovieSummary
        Summary of encoded movie outputs and latest metadata paths.

    Raises
    ------
    ValueError
        If the render manifest or selected frame set is missing or invalid.
    RuntimeError
        If ``ffmpeg`` fails through the compile helpers.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_compile_movie_parameters(parameters)
    render_component = resolve_analysis_input_component(
        str(
            normalized.get("input_source", analysis_auxiliary_root("render_movie"))
        ).strip()
        or analysis_auxiliary_root("render_movie")
    )
    render_metadata = _read_latest_analysis_metadata(
        zarr_path=zarr_path,
        component=render_component,
        operation_name="compile_movie",
    )
    render_manifest_candidate = _resolve_compile_movie_render_manifest_path(
        zarr_path=zarr_path,
        parameters=normalized,
        render_metadata=render_metadata,
    )
    if not render_manifest_candidate:
        raise ValueError(
            "compile_movie requires a render manifest. Run render_movie first "
            "or provide render_manifest_path explicitly."
        )
    render_manifest_path = str(Path(render_manifest_candidate).expanduser().resolve())
    render_manifest = _read_json_mapping(render_manifest_path)
    level_rows = render_manifest.get("levels", [])
    if not isinstance(level_rows, (tuple, list)):
        raise ValueError(
            f"compile_movie manifest '{render_manifest_path}' is missing its levels."
        )
    selected_level = int(normalized.get("rendered_level", 0))
    selected_row: Optional[Mapping[str, object]] = None
    for row in level_rows:
        if not isinstance(row, Mapping):
            continue
        requested_level_value = row.get("requested_level", -1)
        if isinstance(requested_level_value, bool):
            continue
        if isinstance(requested_level_value, int):
            requested_level = int(requested_level_value)
        elif isinstance(requested_level_value, float):
            if not float(requested_level_value).is_integer():
                continue
            requested_level = int(requested_level_value)
        elif isinstance(requested_level_value, str):
            try:
                requested_level = int(requested_level_value)
            except ValueError:
                continue
        else:
            continue
        if requested_level == selected_level:
            selected_row = row
            break
    if selected_row is None:
        raise ValueError(
            f"compile_movie could not find rendered level {selected_level} in "
            f"'{render_manifest_path}'."
        )
    frames_directory = Path(
        str(selected_row.get("frame_directory", "")).strip()
    ).expanduser()
    if not frames_directory.exists():
        raise ValueError(
            f"compile_movie frame directory '{frames_directory}' does not exist."
        )

    _emit(10, "Validating rendered PNG frame directory")
    verify_png_frame_directory(frames_directory)

    output_base_directory = _resolve_movie_output_base_directory(
        zarr_path=zarr_path,
        output_directory=str(normalized.get("output_directory", "")),
        suffix="compile_movie",
    )
    output_directory = _prepare_movie_output_directory(output_base_directory)
    output_stem = str(normalized.get("output_stem", "")).strip()
    if not output_stem:
        output_stem = _resolve_default_movie_stem(zarr_path)
    fps = int(normalized.get("fps", render_manifest.get("fps", 24)))
    resize_xy = normalized.get("resize_xy", [])
    resize = (
        tuple(int(value) for value in resize_xy)
        if isinstance(resize_xy, list) and len(resize_xy) == 2
        else None
    )
    output_format = str(normalized.get("output_format", "mp4")).strip().lower() or "mp4"
    pixel_format = str(normalized.get("pixel_format", "")).strip() or None
    compiled_files: list[str] = []

    if output_format in {"mp4", "both"}:
        _emit(40, "Compiling H.264 MP4 movie")
        mp4_path = (
            output_directory / f"{output_stem}_level_{int(selected_level):02d}.mp4"
        ).resolve()
        compile_png_frames_to_mp4(
            frames_directory=frames_directory,
            output_path=mp4_path,
            fps=fps,
            crf=int(normalized.get("mp4_crf", 18)),
            preset=str(normalized.get("mp4_preset", "slow")),
            pixel_format=pixel_format,
            resize_xy=resize,
        )
        compiled_files.append(str(mp4_path))

    if output_format in {"prores", "both"}:
        _emit(75, "Compiling ProRes MOV movie")
        mov_path = (
            output_directory / f"{output_stem}_level_{int(selected_level):02d}.mov"
        ).resolve()
        compile_png_frames_to_prores(
            frames_directory=frames_directory,
            output_path=mov_path,
            fps=fps,
            profile=int(normalized.get("prores_profile", 3)),
            pixel_format=pixel_format,
            resize_xy=resize,
        )
        compiled_files.append(str(mov_path))

    component = _save_compile_movie_metadata(
        zarr_path=zarr_path,
        render_component=str(render_component),
        render_manifest_path=str(render_manifest_path),
        output_directory=str(output_directory),
        rendered_level=int(selected_level),
        output_format=str(output_format),
        compiled_files=tuple(str(path) for path in compiled_files),
        fps=int(fps),
        parameters=normalized,
        run_id=run_id,
    )
    _emit(100, "Compile-movie workflow complete")
    return CompileMovieSummary(
        component=component,
        render_component=str(render_component),
        render_manifest_path=str(render_manifest_path),
        output_directory=str(output_directory),
        rendered_level=int(selected_level),
        output_format=str(output_format),
        compiled_files=tuple(str(path) for path in compiled_files),
        fps=int(fps),
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
    parser.add_argument(
        "--mode",
        choices=("visualization", "render_movie"),
        default="visualization",
    )
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--run-id", default="")
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
    if str(args.mode).strip().lower() == "render_movie":
        run_render_movie_analysis(
            zarr_path=str(args.zarr_path),
            parameters=parameters_raw,
            progress_callback=None,
            run_id=str(args.run_id).strip() or None,
        )
        return 0
    run_visualization_analysis(
        zarr_path=str(args.zarr_path),
        parameters=parameters_raw,
        progress_callback=None,
        run_id=str(args.run_id).strip() or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_subprocess_entrypoint())
