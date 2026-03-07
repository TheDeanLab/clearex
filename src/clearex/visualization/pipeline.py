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
from clearex.io.provenance import register_latest_output_reference


ProgressCallback = Callable[[int, str], None]


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

    viewer = napari.Viewer(ndisplay=3, show=True)
    viewer.add_image(
        image_data,
        multiscale=is_multiscale,
        name=f"{source_component} (p={position_index})",
    )
    if points.shape[0] > 0:
        viewer.add_points(
            points.astype(np.float32, copy=False),
            name="Particle Detections",
            properties={str(k): np.asarray(v) for k, v in point_properties.items()},
            size=7.0,
            face_color="transparent",
            edge_color="yellow",
            edge_width=0.25,
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
