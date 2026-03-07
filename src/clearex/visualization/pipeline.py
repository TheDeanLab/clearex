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
from pathlib import Path
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
        Selected position index from the ``p`` axis.
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

    lateral_size = _first_positive_float(
        (
            microscope_camera_mapping.get("pixel_size"),
            camera_mapping.get("pixel_size"),
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
    position_index: int,
    points: np.ndarray,
    point_properties: Mapping[str, np.ndarray],
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
    position_index : int
        Position index selected from the ``p`` axis.
    points : numpy.ndarray
        Overlay points in ``(t, c, z, y, x)`` order.
    point_properties : mapping[str, numpy.ndarray]
        Optional per-point property arrays.
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

    level_arrays = [
        da.from_zarr(str(zarr_path), component=component)[:, position_index, :, :, :, :]
        for component in source_components
    ]
    image_data: Any
    is_multiscale = len(level_arrays) > 1
    if is_multiscale:
        image_data = level_arrays
    else:
        image_data = level_arrays[0]

    scale = tuple(float(value) for value in scale_tczyx)
    viewer = napari.Viewer(ndisplay=3, show=True)
    try:
        viewer.dims.axis_labels = tuple(str(label) for label in axis_labels)
    except Exception:
        pass

    viewer.add_image(
        image_data,
        multiscale=is_multiscale,
        name=f"{source_component} (p={position_index})",
        scale=scale,
        metadata={str(key): value for key, value in image_metadata.items()},
    )
    if points.shape[0] > 0:
        viewer.add_points(
            points.astype(np.float32, copy=False),
            name="Particle Detections",
            properties={str(k): np.asarray(v) for k, v in point_properties.items()},
            scale=scale,
            metadata={str(key): value for key, value in points_metadata.items()},
            size=7.0,
            face_color="transparent",
            border_color="yellow",
            border_width=0.2,
        )
    napari.run()


def _save_visualization_metadata(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    source_components: Sequence[str],
    position_index: int,
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
        Selected ``p``-axis index.
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
    position_index = int(normalized["position_index"])
    if position_index >= int(source_shape[1]):
        raise ValueError(
            f"visualization position_index={position_index} is out of bounds "
            f"for position axis size {source_shape[1]}."
        )

    overlay_points = np.empty((0, 5), dtype=np.float32)
    overlay_properties: Dict[str, np.ndarray] = {}
    if bool(normalized.get("overlay_particle_detections", True)):
        overlay_points, overlay_properties = _load_particle_overlay_points(
            root=root,
            detection_component=str(normalized["particle_detection_component"]),
            position_index=position_index,
        )

    napari_payload = _build_napari_layer_payload(
        zarr_path=zarr_path,
        root=root,
        source_component=source_component,
        source_components=source_components,
        position_index=position_index,
        parameters=normalized,
        overlay_points_count=int(overlay_points.shape[0]),
        point_property_names=tuple(str(name) for name in overlay_properties.keys()),
    )

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
            position_index=position_index,
            points=overlay_points,
            point_properties=overlay_properties,
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
        position_index=position_index,
        parameters=normalized,
        overlay_points_count=int(overlay_points.shape[0]),
        launch_mode=effective_launch_mode,
        viewer_pid=viewer_pid,
        run_id=run_id,
    )
    _emit(100, "Visualization workflow complete")
    return VisualizationSummary(
        component=component,
        source_component=source_component,
        source_components=tuple(str(item) for item in source_components),
        position_index=position_index,
        overlay_points_count=int(overlay_points.shape[0]),
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
