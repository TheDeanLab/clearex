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

from __future__ import annotations

# Standard Library Imports
from contextlib import ExitStack
from dataclasses import replace
import json
import logging
import math
import os
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Local Imports
from .spacing import (
    apply_compact_row_spacing,
    apply_dialog_grid_spacing,
    apply_footer_row_spacing,
    apply_form_spacing,
    apply_help_stack_spacing,
    apply_metadata_grid_spacing,
    apply_popup_root_spacing,
    apply_row_spacing,
    apply_stack_spacing,
    apply_window_root_spacing,
)
from clearex.io.experiment import (
    ExperimentDataResolutionError,
    NavigateExperiment,
    create_dask_client,
    has_complete_canonical_data_store,
    infer_zyx_shape,
    is_navigate_experiment_file,
    load_navigate_experiment,
    materialize_experiment_data_store,
    resolve_data_store_path,
    resolve_experiment_data_path,
)
from clearex.io.provenance import is_zarr_store_path, summarize_analysis_history
from clearex.io.read import ImageInfo, ImageOpener
from clearex.workflow import (
    DASK_BACKEND_LOCAL_CLUSTER,
    DASK_BACKEND_MODE_LABELS,
    DASK_BACKEND_SLURM_CLUSTER,
    DASK_BACKEND_SLURM_RUNNER,
    DEFAULT_ZARR_CHUNKS_PTCZYX,
    DEFAULT_ZARR_PYRAMID_PTCZYX,
    DEFAULT_SLURM_CLUSTER_JOB_EXTRA_DIRECTIVES,
    DaskBackendConfig,
    LocalClusterRecommendation,
    LocalClusterConfig,
    PTCZYX_AXES,
    SlurmClusterConfig,
    SlurmRunnerConfig,
    WorkflowConfig,
    ZarrSaveConfig,
    dask_backend_from_dict,
    dask_backend_to_dict,
    default_analysis_operation_parameters,
    format_dask_backend_summary,
    format_local_cluster_recommendation_summary,
    format_pyramid_levels,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    normalize_analysis_operation_parameters,
    parse_pyramid_levels,
    recommend_local_cluster_config,
)

# Third Party Imports
import zarr

try:
    from PyQt6.QtCore import (
        QEvent,
        QEventLoop,
        QMimeData,
        QObject,
        QSize,
        QThread,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QDragEnterEvent,
        QDragMoveEvent,
        QDropEvent,
        QIcon,
        QImage,
        QImageReader,
        QPixmap,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QHeaderView,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class GuiUnavailableError(RuntimeError):
    """Raised when the PyQt GUI cannot be launched in this environment."""


_GUI_ASSET_DIRECTORY = Path(__file__).resolve().parent
_GUI_SPLASH_IMAGE = "ClearEx_full.png"
_GUI_HEADER_IMAGE = "ClearEx_full_header.png"
_GUI_APP_ICON = "icon.png"
_CLEAREX_SETTINGS_DIR_NAME = ".clearex"
_CLEAREX_DASK_BACKEND_SETTINGS_FILE = "dask_backend_settings.json"


def _resolve_gui_asset_path(filename: str) -> Path:
    """Resolve a GUI asset path in the package asset directory.

    Parameters
    ----------
    filename : str
        Asset filename.

    Returns
    -------
    pathlib.Path
        Resolved asset path.
    """
    return (_GUI_ASSET_DIRECTORY / filename).resolve()


def _display_is_available() -> bool:
    """Check whether the runtime environment likely supports GUI display.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        ``True`` when a display server is likely available, otherwise
        ``False``.
    """
    if os.name != "posix":
        return True
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _resolve_clearex_settings_directory() -> Path:
    """Resolve the default per-user ClearEx settings directory.

    Parameters
    ----------
    None

    Returns
    -------
    pathlib.Path
        Absolute path to ``~/.clearex``.
    """
    return (Path.home() / _CLEAREX_SETTINGS_DIR_NAME).expanduser().resolve()


def _resolve_dask_backend_settings_path(
    settings_directory: Optional[Path] = None,
) -> Path:
    """Resolve the user settings JSON path for persisted backend config.

    Parameters
    ----------
    settings_directory : pathlib.Path, optional
        Settings directory override. Defaults to ``~/.clearex``.

    Returns
    -------
    pathlib.Path
        Full JSON path used for Dask backend persistence.
    """
    directory = (
        settings_directory
        if settings_directory is not None
        else _resolve_clearex_settings_directory()
    )
    return directory / _CLEAREX_DASK_BACKEND_SETTINGS_FILE


def _ensure_clearex_settings_directory(
    settings_directory: Optional[Path] = None,
) -> Path:
    """Create the ClearEx user settings directory when missing.

    Parameters
    ----------
    settings_directory : pathlib.Path, optional
        Directory path override. Defaults to ``~/.clearex``.

    Returns
    -------
    pathlib.Path
        The resolved settings directory path.

    Raises
    ------
    None
        Directory creation errors are logged and handled internally.
    """
    directory = (
        settings_directory
        if settings_directory is not None
        else _resolve_clearex_settings_directory()
    )
    target = directory.expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to create ClearEx settings directory %s: %s",
            target,
            exc,
        )
    return target


def _load_last_used_dask_backend_config(
    settings_path: Optional[Path] = None,
) -> Optional[DaskBackendConfig]:
    """Load the last-used Dask backend configuration from JSON.

    Parameters
    ----------
    settings_path : pathlib.Path, optional
        JSON file path override.

    Returns
    -------
    DaskBackendConfig, optional
        Persisted backend configuration, or ``None`` when settings are missing
        or unreadable.

    Raises
    ------
    None
        Read and parse errors are logged and handled internally.
    """
    path = (
        settings_path
        if settings_path is not None
        else _resolve_dask_backend_settings_path()
    )
    resolved = path.expanduser()
    if not resolved.exists():
        return None

    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to read Dask backend settings %s: %s",
            resolved,
            exc,
        )
        return None

    if not raw_text.strip():
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logging.getLogger(__name__).warning(
            "Failed to decode Dask backend settings %s: %s",
            resolved,
            exc,
        )
        return None

    if isinstance(payload, dict) and not payload:
        return None

    return dask_backend_from_dict(payload)


def _save_last_used_dask_backend_config(
    config: DaskBackendConfig,
    settings_path: Optional[Path] = None,
) -> bool:
    """Persist the most recently used Dask backend configuration.

    Parameters
    ----------
    config : DaskBackendConfig
        Backend configuration to persist.
    settings_path : pathlib.Path, optional
        JSON file path override.

    Returns
    -------
    bool
        ``True`` when settings are written successfully, otherwise ``False``.

    Raises
    ------
    None
        Write errors are logged and handled internally.
    """
    path = (
        settings_path
        if settings_path is not None
        else _resolve_dask_backend_settings_path()
    )
    resolved = path.expanduser()
    _ensure_clearex_settings_directory(resolved.parent)

    try:
        payload = dask_backend_to_dict(config)
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        resolved.write_text(f"{serialized}\n", encoding="utf-8")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to persist Dask backend settings %s: %s",
            resolved,
            exc,
        )
        return False
    return True


def _should_apply_persisted_dask_backend(initial: Optional[WorkflowConfig]) -> bool:
    """Return whether persisted backend settings should override GUI defaults.

    Parameters
    ----------
    initial : WorkflowConfig, optional
        Initial workflow provided by caller.

    Returns
    -------
    bool
        ``True`` when caller did not provide a custom backend and persisted
        settings can safely populate the GUI defaults.
    """
    if initial is None:
        return True
    return initial.dask_backend == DaskBackendConfig()


def _zarr_component_exists_in_root(root: Any, component: str) -> bool:
    """Return whether a component path resolves inside an open Zarr root.

    Parameters
    ----------
    root : Any
        Open Zarr group-like object.
    component : str
        Candidate component path.

    Returns
    -------
    bool
        ``True`` when the component exists, otherwise ``False``.

    Notes
    -----
    Existence checks are intentionally exception-tolerant because Zarr backends
    may surface missing keys and transient I/O issues through different error
    types.
    """
    path = str(component).strip()
    if not path:
        return False
    try:
        root[path]
    except Exception:
        return False
    return True


def _discover_available_operation_output_components(
    *,
    store_path: Optional[str],
    operation_output_components: Mapping[str, str],
) -> Dict[str, str]:
    """Discover operation outputs currently present in a canonical Zarr store.

    Parameters
    ----------
    store_path : str, optional
        Candidate Zarr/N5 store path.
    operation_output_components : mapping[str, str]
        Mapping from operation key to expected output component path.

    Returns
    -------
    dict[str, str]
        Operation-to-component mapping for outputs that currently exist.

    Notes
    -----
    This helper is used only for GUI option hints; failures return an empty
    mapping so runtime behavior remains unchanged.
    """
    candidate = str(store_path or "").strip()
    if not candidate or not is_zarr_store_path(candidate):
        return {}

    try:
        root = zarr.open_group(candidate, mode="r")
    except Exception:
        return {}

    available: Dict[str, str] = {}
    for operation_name, component in operation_output_components.items():
        if _zarr_component_exists_in_root(root, component):
            available[str(operation_name)] = str(component)
    return available


def _build_input_source_options(
    *,
    operation_name: str,
    selected_order: Sequence[str],
    operation_key_order: Sequence[str],
    operation_labels: Mapping[str, str],
    operation_output_components: Mapping[str, str],
    available_store_output_components: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, str]]:
    """Build GUI input-source options for one analysis operation.

    Parameters
    ----------
    operation_name : str
        Operation key for which options are generated.
    selected_order : sequence of str
        Operations selected in current execution order.
    operation_key_order : sequence of str
        Canonical operation order used for deterministic display ordering.
    operation_labels : mapping[str, str]
        Human-readable labels per operation.
    operation_output_components : mapping[str, str]
        Expected output component per operation.
    available_store_output_components : mapping[str, str], optional
        Existing outputs discovered in the selected store from prior runs.

    Returns
    -------
    list[tuple[str, str]]
        Ordered ``(value, label)`` options used by source-selection combo boxes.
    """
    options: list[tuple[str, str]] = [("data", "Raw data (data)")]
    option_values = {"data"}

    for upstream_name in selected_order:
        if upstream_name == operation_name:
            break
        component = operation_output_components.get(upstream_name)
        if not component or upstream_name in option_values:
            continue
        label = operation_labels.get(
            upstream_name, upstream_name.replace("_", " ").title()
        )
        options.append((upstream_name, f"{label} output ({component})"))
        option_values.add(upstream_name)

    available = available_store_output_components or {}
    for upstream_name in operation_key_order:
        if upstream_name == operation_name or upstream_name in option_values:
            continue
        component = str(available.get(upstream_name, "")).strip()
        if not component:
            continue
        label = operation_labels.get(
            upstream_name, upstream_name.replace("_", " ").title()
        )
        options.append((upstream_name, f"{label} output ({component}) [existing]"))
        option_values.add(upstream_name)

    return options


def _particle_overlay_available_for_visualization(
    *,
    selected_order: Sequence[str],
    has_particle_detection_history: bool = False,
) -> bool:
    """Return whether particle-overlay controls should be available.

    Parameters
    ----------
    selected_order : sequence[str]
        Currently selected operations in execution order.
    has_particle_detection_history : bool, default=False
        Whether provenance indicates a completed particle-detection run.

    Returns
    -------
    bool
        ``True`` when overlay can be supported by either:
        a current particle-detection run before visualization or successful
        particle-detection provenance history.
    """
    ordered = [str(name).strip() for name in selected_order]
    if "particle_detection" in ordered and "visualization" in ordered:
        if ordered.index("particle_detection") < ordered.index("visualization"):
            return True

    return bool(has_particle_detection_history)


def _shear_degrees_to_coefficient(angle_degrees: float) -> float:
    """Convert a shear angle in degrees to a dimensionless shear coefficient.

    Parameters
    ----------
    angle_degrees : float
        Shear angle in degrees.

    Returns
    -------
    float
        Shear coefficient computed as ``tan(angle_degrees)``.
    """
    return float(math.tan(math.radians(float(angle_degrees))))


def _shear_coefficient_to_degrees(coefficient: float) -> float:
    """Convert a shear coefficient to an equivalent angle in degrees.

    Parameters
    ----------
    coefficient : float
        Dimensionless shear coefficient.

    Returns
    -------
    float
        Shear angle in degrees computed as ``atan(coefficient)``.
    """
    return float(math.degrees(math.atan(float(coefficient))))


def _format_optional_value(value: Optional[Any]) -> str:
    """Format optional metadata values for display labels.

    Parameters
    ----------
    value : Any, optional
        Metadata value to display.

    Returns
    -------
    str
        Human-readable value string. Missing values return ``"n/a"``.
    """
    if value is None:
        return "n/a"
    if isinstance(value, tuple):
        return " x ".join(str(v) for v in value)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _coerce_positive_float(value: Any) -> Optional[float]:
    """Parse a value into a finite positive float.

    Parameters
    ----------
    value : Any
        Candidate numeric value.

    Returns
    -------
    float, optional
        Parsed finite positive float, otherwise ``None``.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return float(parsed)


def _extract_pixel_size_um_zyx_from_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[tuple[float, float, float]]:
    """Extract physical voxel size from metadata mappings.

    Parameters
    ----------
    metadata : mapping[str, Any], optional
        Metadata dictionary from image readers.

    Returns
    -------
    tuple[float, float, float], optional
        Voxel size in ``(z, y, x)`` microns when available.
    """
    if not metadata:
        return None

    candidate_keys = (
        "voxel_size_um_zyx",
        "pixel_size_um_zyx",
        "physical_pixel_size_zyx",
        "pixel_size_zyx",
    )
    for key in candidate_keys:
        raw_value = metadata.get(key)
        if not isinstance(raw_value, (tuple, list)) or len(raw_value) < 3:
            continue
        z_um = _coerce_positive_float(raw_value[0])
        y_um = _coerce_positive_float(raw_value[1])
        x_um = _coerce_positive_float(raw_value[2])
        if z_um is None or y_um is None or x_um is None:
            continue
        return (z_um, y_um, x_um)

    navigate = metadata.get("navigate_experiment")
    if isinstance(navigate, Mapping):
        xy_um = _coerce_positive_float(navigate.get("xy_pixel_size_um"))
        z_um = _coerce_positive_float(navigate.get("z_step_um"))
        if xy_um is not None and z_um is not None:
            return (z_um, xy_um, xy_um)

    xy_um = _coerce_positive_float(metadata.get("xy_pixel_size_um"))
    z_um = _coerce_positive_float(metadata.get("z_step_um"))
    if xy_um is not None and z_um is not None:
        return (z_um, xy_um, xy_um)
    return None


def _extract_pixel_size_um_zyx(info: ImageInfo) -> Optional[tuple[float, float, float]]:
    """Extract physical voxel size in microns from ``ImageInfo``.

    Parameters
    ----------
    info : ImageInfo
        Image metadata returned by an image reader.

    Returns
    -------
    tuple[float, float, float], optional
        Voxel size in ``(z, y, x)`` order when available.
    """
    raw_pixel_size = getattr(info, "pixel_size", None)
    if isinstance(raw_pixel_size, (tuple, list)) and len(raw_pixel_size) >= 3:
        z_um = _coerce_positive_float(raw_pixel_size[0])
        y_um = _coerce_positive_float(raw_pixel_size[1])
        x_um = _coerce_positive_float(raw_pixel_size[2])
        if z_um is not None and y_um is not None and x_um is not None:
            return (z_um, y_um, x_um)
    return _extract_pixel_size_um_zyx_from_metadata(info.metadata)


def _format_pixel_size_um_zyx(
    pixel_size_um_zyx: Optional[tuple[float, float, float]],
) -> str:
    """Format physical voxel size for GUI labels.

    Parameters
    ----------
    pixel_size_um_zyx : tuple[float, float, float], optional
        Voxel size in ``(z, y, x)`` microns.

    Returns
    -------
    str
        Display string for metadata panel.
    """
    if pixel_size_um_zyx is None:
        return "n/a"
    z_um, y_um, x_um = pixel_size_um_zyx
    return f"z={z_um:.6g}, y={y_um:.6g}, x={x_um:.6g}"


def _extract_axis_map(info: ImageInfo) -> Dict[str, int]:
    """Create an axis-to-size map from ``ImageInfo``.

    Parameters
    ----------
    info : ImageInfo
        Image metadata object returned by :class:`clearex.io.read.ImageOpener`.

    Returns
    -------
    dict[str, int]
        Lowercase axis names mapped to integer sizes. Returns an empty mapping
        if axes are missing or do not match shape dimensionality.
    """
    if not info.axes or len(info.axes) != len(info.shape):
        return {}
    return {
        axis.lower(): int(size)
        for axis, size in zip(info.axes, info.shape, strict=False)
        if isinstance(axis, str)
    }


def _metadata_count(
    metadata: Optional[Dict[str, Any]], keys: tuple[str, ...]
) -> Optional[int]:
    """Extract a count-like metadata value using candidate keys.

    Parameters
    ----------
    metadata : dict[str, Any], optional
        Metadata dictionary from :class:`clearex.io.read.ImageInfo`.
    keys : tuple of str
        Candidate lowercase keys to test in priority order.

    Returns
    -------
    int, optional
        Parsed count value when available, otherwise ``None``.
    """
    if not metadata:
        return None
    for key, value in metadata.items():
        if str(key).lower() not in keys:
            continue
        if isinstance(value, int):
            return value
        try:
            return len(value)
        except TypeError:
            return None
    return None


def summarize_image_info(info: ImageInfo) -> Dict[str, str]:
    """Build GUI-friendly metadata label strings.

    Parameters
    ----------
    info : ImageInfo
        Image metadata returned by an image reader.

    Returns
    -------
    dict[str, str]
        Flattened text fields for GUI presentation.
    """
    axis_map = _extract_axis_map(info)
    channels = axis_map.get("c")
    time_points = axis_map.get("t")
    positions = axis_map.get("p") or axis_map.get("s")

    if channels is None:
        channels = _metadata_count(info.metadata, ("channels", "channel", "c"))
    if positions is None:
        positions = _metadata_count(
            info.metadata, ("positions", "position", "fields", "field", "s", "p")
        )

    x_size = axis_map.get("x")
    y_size = axis_map.get("y")
    z_size = axis_map.get("z")
    if x_size is not None and y_size is not None:
        if z_size is not None:
            image_size = f"{x_size} x {y_size} x {z_size}"
        else:
            image_size = f"{x_size} x {y_size}"
    else:
        fallback_spatial = info.shape[-3:] if len(info.shape) >= 3 else info.shape
        image_size = " x ".join(str(v) for v in fallback_spatial)

    metadata_keys = "n/a"
    if info.metadata:
        key_names = [str(key) for key in info.metadata.keys()]
        metadata_keys = ", ".join(sorted(key_names)[:8])
        if len(key_names) > 8:
            metadata_keys += ", ..."

    pixel_size_um_zyx = _extract_pixel_size_um_zyx(info)

    return {
        "path": str(info.path),
        "shape": _format_optional_value(info.shape),
        "dtype": _format_optional_value(info.dtype),
        "axes": _format_optional_value(info.axes),
        "channels": _format_optional_value(channels),
        "positions": _format_optional_value(positions),
        "image_size": image_size,
        "time_points": _format_optional_value(time_points),
        "pixel_size": _format_pixel_size_um_zyx(pixel_size_um_zyx),
        "metadata_keys": metadata_keys,
    }


def _apply_experiment_overrides(
    summary: Dict[str, str],
    experiment_path: Path,
    resolved_data_path: Path,
    experiment: NavigateExperiment,
) -> Dict[str, str]:
    """Overlay Navigate experiment metadata onto GUI summary fields.

    Parameters
    ----------
    summary : dict[str, str]
        Base summary generated from image metadata.
    experiment_path : pathlib.Path
        Selected ``experiment.yml`` path.
    resolved_data_path : pathlib.Path
        Data path resolved from experiment metadata.
    experiment : Any
        Parsed experiment object with channel/time/position attributes.

    Returns
    -------
    dict[str, str]
        Updated summary dictionary.
    """
    out = dict(summary)
    out["path"] = f"{experiment_path} -> {resolved_data_path}"
    out["channels"] = str(experiment.channel_count)
    out["positions"] = str(experiment.multiposition_count)
    out["time_points"] = str(experiment.timepoints)
    xy_um = _coerce_positive_float(experiment.xy_pixel_size_um)
    z_um = _coerce_positive_float(experiment.z_step_um)
    if xy_um is not None and z_um is not None:
        out["pixel_size"] = _format_pixel_size_um_zyx((z_um, xy_um, xy_um))
    out["metadata_keys"] = (
        f"{summary.get('metadata_keys', 'n/a')} | file_type={experiment.file_type}"
    )
    return out


def _format_zarr_save_summary(config: ZarrSaveConfig) -> str:
    """Build a compact GUI summary string for Zarr save settings.

    Parameters
    ----------
    config : ZarrSaveConfig
        Zarr save configuration selected in the dialog.

    Returns
    -------
    str
        Single-line summary including chunk sizes and pyramid factors.
    """
    chunks_text = format_zarr_chunks_ptczyx(config.chunks_ptczyx)
    pyramid_text = format_zarr_pyramid_ptczyx(config.pyramid_ptczyx)
    return f"Chunks: {chunks_text} | Pyramid: {pyramid_text}"


def _dask_mode_help_text(mode: str) -> str:
    """Return operator-focused guidance text for a Dask backend mode.

    Parameters
    ----------
    mode : str
        Dask backend mode key.

    Returns
    -------
    str
        Short descriptive help text.
    """
    if mode == DASK_BACKEND_LOCAL_CLUSTER:
        return (
            "LocalCluster runs scheduler/workers on this machine. "
            "Best for laptop/workstation development and single-node runs."
        )
    if mode == DASK_BACKEND_SLURM_RUNNER:
        return (
            "SLURMRunner attaches to an existing scheduler file. "
            "Use when your cluster launcher already created a scheduler endpoint."
        )
    return (
        "SLURMCluster submits worker jobs to Slurm directly from ClearEx. "
        "Best for scalable multi-node execution from one configuration dialog."
    )


def _popup_dialog_stylesheet() -> str:
    """Return shared stylesheet for configuration popup dialogs.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Qt stylesheet string for dark-themed popup dialogs.
    """
    return """
        QDialog {
            background-color: #0c1118;
            color: #e6edf3;
            font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
            font-size: 13px;
        }
        QLabel {
            color: #d9e2f1;
        }
        QGroupBox {
            border: 1px solid #2a3442;
            border-radius: 10px;
            margin-top: 16px;
            padding: 14px;
            background-color: #111925;
            font-weight: 600;
            color: #9cc6ff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #9cc6ff;
        }
        QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {
            background-color: #0b1320;
            border: 1px solid #2b3f58;
            border-radius: 8px;
            min-height: 30px;
            padding: 7px 10px;
            color: #e6edf3;
            selection-background-color: #2f81f7;
        }
        QComboBox {
            padding-right: 24px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 22px;
            border-left: 1px solid #2b3f58;
            background-color: #162538;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #0b1320;
            color: #e6edf3;
            border: 1px solid #2b3f58;
            selection-background-color: #2f81f7;
            selection-color: #f8fbff;
            outline: 0;
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #2f81f7;
            color: #f8fbff;
        }
        QTableWidget {
            background-color: #0b1320;
            color: #e6edf3;
            border: 1px solid #2b3f58;
            border-radius: 8px;
            gridline-color: #2a3442;
            selection-background-color: #2f81f7;
            selection-color: #f8fbff;
        }
        QTableWidget::item {
            color: #e6edf3;
        }
        QHeaderView::section {
            background-color: #162538;
            color: #9cc6ff;
            border: 1px solid #2b3f58;
            padding: 6px 8px;
            font-weight: 600;
        }
        QTableCornerButton::section {
            background-color: #162538;
            border: 1px solid #2b3f58;
        }
        QLineEdit::placeholder, QPlainTextEdit {
            color: #a6b7d0;
        }
        QPushButton {
            background-color: #1a2635;
            border: 1px solid #2f4460;
            border-radius: 8px;
            padding: 9px 14px;
            color: #dbe9ff;
        }
        QPushButton:hover {
            background-color: #22354c;
        }
        QPushButton:pressed {
            background-color: #182639;
        }
        QPushButton#runButton {
            background-color: #2f81f7;
            border-color: #2f81f7;
            color: #f8fbff;
            font-weight: 700;
        }
        QPushButton#runButton:hover {
            background-color: #1f6cd8;
        }
    """


def _active_log_file_path() -> Optional[Path]:
    """Return the current root logger file destination when available.

    Parameters
    ----------
    None

    Returns
    -------
    pathlib.Path, optional
        Active log file path for the root logger, when a file handler exists.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        base_filename = getattr(handler, "baseFilename", None)
        if not base_filename:
            continue
        try:
            return Path(str(base_filename)).expanduser().resolve()
        except Exception:
            return Path(str(base_filename))
    return None


def _show_themed_error_dialog(
    parent: Optional["QWidget"],
    title: str,
    message: str,
    *,
    summary: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    """Show a styled error dialog with optional traceback details.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget for the modal dialog.
    title : str
        Window title.
    message : str
        Primary error text shown to the operator.
    summary : str, optional
        Short exception summary displayed beneath the main message.
    details : str, optional
        Detailed traceback or diagnostic text exposed behind the standard
        ``Show Details`` affordance.

    Returns
    -------
    None
        Modal dialog side effects only.
    """
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setWindowTitle(str(title))
    dialog.setText(str(message))

    informative_lines: list[str] = []
    if summary:
        informative_lines.append(str(summary))
    log_path = _active_log_file_path()
    if log_path is not None:
        informative_lines.append(f"Log file: {log_path}")
    if informative_lines:
        dialog.setInformativeText("\n".join(informative_lines))
    if details:
        dialog.setDetailedText(str(details))

    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.setStyleSheet(
        _popup_dialog_stylesheet()
        + """
        QMessageBox {
            background-color: #0c1118;
            color: #e6edf3;
        }
        QTextEdit, QPlainTextEdit {
            background-color: #0b1320;
            border: 1px solid #2b3f58;
            border-radius: 8px;
            padding: 6px 8px;
            color: #e6edf3;
            selection-background-color: #2f81f7;
        }
        """
    )
    dialog.exec()


def _configure_dask_backend_client(
    backend: DaskBackendConfig,
    *,
    exit_stack: ExitStack,
) -> Optional[Any]:
    """Create and register a Dask client for a configured backend mode.

    Parameters
    ----------
    backend : DaskBackendConfig
        User-selected Dask backend configuration.
    exit_stack : contextlib.ExitStack
        Exit stack used to register client/cluster cleanup callbacks.

    Returns
    -------
    Any, optional
        Dask client-like object when backend initialization succeeds.

    Raises
    ------
    ValueError
        If backend settings are incomplete for the selected mode.
    Exception
        Propagates backend connection or cluster startup failures.
    """
    if backend.mode == DASK_BACKEND_LOCAL_CLUSTER:
        client = create_dask_client(
            n_workers=backend.local_cluster.n_workers,
            threads_per_worker=backend.local_cluster.threads_per_worker,
            processes=False,
            memory_limit=backend.local_cluster.memory_limit,
            local_directory=backend.local_cluster.local_directory,
        )
        exit_stack.callback(client.close)
        return client

    if backend.mode == DASK_BACKEND_SLURM_RUNNER:
        scheduler_file = backend.slurm_runner.scheduler_file
        if not scheduler_file:
            raise ValueError("SLURMRunner backend requires a scheduler file path.")

        from dask.distributed import Client
        from dask_jobqueue.slurm import SLURMRunner

        runner = exit_stack.enter_context(SLURMRunner(scheduler_file=scheduler_file))
        client = exit_stack.enter_context(Client(runner))

        wait_for_workers = backend.slurm_runner.wait_for_workers
        if wait_for_workers is None:
            runner_workers = getattr(runner, "n_workers", None)
            if isinstance(runner_workers, int) and runner_workers > 0:
                wait_for_workers = runner_workers
        if wait_for_workers is not None:
            client.wait_for_workers(wait_for_workers)
        return client

    if backend.mode == DASK_BACKEND_SLURM_CLUSTER:
        from dask.distributed import Client
        from dask_jobqueue import SLURMCluster

        cluster_cfg = backend.slurm_cluster
        extra_directives = [
            directive.strip()
            for directive in cluster_cfg.job_extra_directives
            if directive.strip()
        ]
        if cluster_cfg.mail_user:
            extra_directives = [
                directive
                for directive in extra_directives
                if not directive.startswith("--mail-user=")
            ]
            extra_directives.append(f"--mail-user={cluster_cfg.mail_user}")

        cluster_kwargs = {
            "cores": cluster_cfg.cores,
            "processes": cluster_cfg.processes,
            "memory": cluster_cfg.memory,
            "local_directory": cluster_cfg.local_directory,
            "interface": cluster_cfg.interface,
            "walltime": cluster_cfg.walltime,
            "job_name": cluster_cfg.job_name,
            "queue": cluster_cfg.queue,
            "death_timeout": cluster_cfg.death_timeout,
            "job_extra_directives": extra_directives,
            "scheduler_options": {
                "dashboard_address": cluster_cfg.dashboard_address,
                "interface": cluster_cfg.scheduler_interface,
                "idle_timeout": cluster_cfg.idle_timeout,
                "allowed_failures": cluster_cfg.allowed_failures,
            },
        }
        cluster = SLURMCluster(**cluster_kwargs)
        exit_stack.callback(cluster.close)
        cluster.scale(jobs=cluster_cfg.workers)

        client = Client(cluster)
        exit_stack.callback(client.close)
        client.wait_for_workers(cluster_cfg.workers)
        return client

    raise ValueError(f"Unsupported Dask backend mode: {backend.mode}")


if HAS_PYQT6:

    def _apply_application_icon(app: QApplication) -> None:
        """Apply packaged app icon for window manager and task switcher display.

        Parameters
        ----------
        app : QApplication
            Active Qt application instance.

        Returns
        -------
        None
            Application icon is updated in-place when the packaged icon exists.
        """
        icon_path = _resolve_gui_asset_path(_GUI_APP_ICON)
        if not icon_path.exists():
            return
        icon = QIcon(str(icon_path))
        if icon.isNull():
            return
        app.setWindowIcon(icon)

    def _load_branding_pixmap(
        filename: str, *, max_decode_pixels: int = 24_000_000
    ) -> Optional[QPixmap]:
        """Load a GUI branding image from the packaged asset directory.

        Parameters
        ----------
        filename : str
            Asset filename.
        max_decode_pixels : int, default=24_000_000
            Maximum source pixel count decoded in memory. Larger images are
            decoded through :class:`QImageReader` with an internal scaled size
            to avoid high-memory image allocation failures.

        Returns
        -------
        QPixmap, optional
        Loaded pixmap when the file exists and can be decoded; otherwise
            ``None``.
        """
        asset_path = _resolve_gui_asset_path(filename)
        if not asset_path.exists():
            return None
        reader = QImageReader(str(asset_path))
        source_size = reader.size()
        required_limit_mb = 1024
        if source_size.isValid():
            pixel_count = int(source_size.width()) * int(source_size.height())
            required_limit_mb = max(
                512,
                int(((pixel_count * 4) / (1024 * 1024)) + 32),
            )
            if pixel_count > max(1, int(max_decode_pixels)):
                scale = (int(max_decode_pixels) / float(pixel_count)) ** 0.5
                reader.setScaledSize(
                    QSize(
                        max(1, int(source_size.width() * scale)),
                        max(1, int(source_size.height() * scale)),
                    )
                )
        previous_limit_mb = int(QImageReader.allocationLimit())
        next_limit_mb = max(previous_limit_mb, required_limit_mb)
        if next_limit_mb != previous_limit_mb:
            QImageReader.setAllocationLimit(next_limit_mb)
        try:
            image = reader.read()
        finally:
            if next_limit_mb != previous_limit_mb:
                QImageReader.setAllocationLimit(previous_limit_mb)
        if image.isNull():
            return None
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return None
        return pixmap

    def _create_scaled_branding_label(
        *,
        filename: str,
        max_width: int,
        max_height: int,
        object_name: str,
    ) -> Optional[QLabel]:
        """Create a centered QLabel containing a smoothly scaled branding image.

        Parameters
        ----------
        filename : str
            Branding asset filename.
        max_width : int
            Maximum rendered label width in pixels.
        max_height : int
            Maximum rendered label height in pixels.
        object_name : str
            Qt object name for stylesheet targeting.

        Returns
        -------
        QLabel, optional
            Prepared image label, or ``None`` when the asset cannot be loaded.
        """
        pixmap = _load_branding_pixmap(filename)
        if pixmap is None:
            return None
        scaled = pixmap.scaled(
            max(1, int(max_width)),
            max(1, int(max_height)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label = QLabel()
        label.setObjectName(object_name)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label.setPixmap(scaled)
        label.setMinimumHeight(scaled.height())
        label.setMaximumHeight(max_height)
        return label

    def _create_startup_splash() -> Optional[QSplashScreen]:
        """Create startup splash screen widget from packaged branding image.

        Parameters
        ----------
        None

        Returns
        -------
        QSplashScreen, optional
            Configured splash screen or ``None`` when asset loading fails.
        """
        pixmap = _load_branding_pixmap(_GUI_SPLASH_IMAGE)
        if pixmap is None:
            return None
        scaled = pixmap.scaled(
            420,
            420,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        splash = QSplashScreen(scaled)
        splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        return splash

    def _show_startup_splash(app: QApplication) -> None:
        """Display startup splash image briefly before opening setup dialog.

        Parameters
        ----------
        app : QApplication
            Active Qt application instance.

        Returns
        -------
        None
            Splash is shown and dismissed in-place.
        """
        splash = _create_startup_splash()
        if splash is None:
            return
        splash.show()
        app.processEvents()
        event_loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(event_loop.quit)
        timer.start(800)
        event_loop.exec()
        splash.close()
        app.processEvents()

    class ZarrSaveConfigDialog(QDialog):
        """Dialog for configuring analysis-store Zarr chunking and pyramid factors.

        Parameters
        ----------
        initial : ZarrSaveConfig
            Initial Zarr save configuration used to populate controls.
        parent : QDialog, optional
            Parent dialog widget.

        Attributes
        ----------
        result_config : ZarrSaveConfig, optional
            Selected configuration when dialog is accepted.
        """

        def __init__(
            self,
            initial: ZarrSaveConfig,
            parent: Optional[QDialog] = None,
        ) -> None:
            """Initialize dialog widgets with the provided configuration.

            Parameters
            ----------
            initial : ZarrSaveConfig
                Initial Zarr save configuration.
            parent : QDialog, optional
                Parent dialog widget.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__(parent)
            self.setWindowTitle("Zarr Save Settings")
            self.setMinimumWidth(860)
            self.setMinimumHeight(660)
            self.result_config: Optional[ZarrSaveConfig] = None
            self._chunk_inputs: Dict[str, QSpinBox] = {}
            self._pyramid_inputs: Dict[str, QLineEdit] = {}

            self._build_ui()
            self._hydrate(initial)
            self.setStyleSheet(_popup_dialog_stylesheet())

        def _build_ui(self) -> None:
            """Construct dialog controls and wire signals.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widgets are created and connected in-place.
            """
            root = QVBoxLayout(self)
            apply_popup_root_spacing(root)

            description = QLabel(
                "Configure Zarr save chunk sizes and downsampling pyramid in "
                "(p, t, c, z, y, x) axis order."
            )
            description.setWordWrap(True)
            root.addWidget(description)

            chunk_group = QGroupBox("Chunk Size")
            chunk_layout = QGridLayout(chunk_group)
            apply_dialog_grid_spacing(chunk_layout)

            for row, axis_name in enumerate(PTCZYX_AXES):
                axis_label = QLabel(axis_name.upper())
                spinbox = QSpinBox()
                spinbox.setRange(1, 65536)
                spinbox.setSingleStep(1)
                chunk_layout.addWidget(axis_label, row, 0)
                chunk_layout.addWidget(spinbox, row, 1)
                self._chunk_inputs[axis_name] = spinbox

            root.addWidget(chunk_group)

            pyramid_group = QGroupBox("Resolution Pyramid Factors")
            pyramid_layout = QGridLayout(pyramid_group)
            apply_dialog_grid_spacing(pyramid_layout)

            for row, axis_name in enumerate(PTCZYX_AXES):
                axis_label = QLabel(axis_name.upper())
                levels_input = QLineEdit()
                levels_input.setPlaceholderText("1,2,4,8")
                levels_input.setToolTip(
                    "Comma-separated positive factors. Each axis must start with 1."
                )
                pyramid_layout.addWidget(axis_label, row, 0)
                pyramid_layout.addWidget(levels_input, row, 1)
                self._pyramid_inputs[axis_name] = levels_input

            root.addWidget(pyramid_group)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            self._defaults_button = QPushButton("Reset Defaults")
            self._cancel_button = QPushButton("Cancel")
            self._apply_button = QPushButton("Apply")
            self._apply_button.setObjectName("runButton")
            footer.addWidget(self._defaults_button)
            footer.addStretch(1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._apply_button)
            root.addLayout(footer)

            self._defaults_button.clicked.connect(self._on_reset_defaults)
            self._cancel_button.clicked.connect(self.reject)
            self._apply_button.clicked.connect(self._on_apply)

        def _hydrate(self, initial: ZarrSaveConfig) -> None:
            """Populate form controls from an initial Zarr save configuration.

            Parameters
            ----------
            initial : ZarrSaveConfig
                Initial values for chunk sizes and pyramid factors.

            Returns
            -------
            None
                Widget values are updated in-place.
            """
            for axis_name, chunk_size in zip(
                PTCZYX_AXES, initial.chunks_ptczyx, strict=False
            ):
                self._chunk_inputs[axis_name].setValue(int(chunk_size))

            for axis_name, factors in zip(
                PTCZYX_AXES, initial.pyramid_ptczyx, strict=False
            ):
                self._pyramid_inputs[axis_name].setText(format_pyramid_levels(factors))

        def _on_reset_defaults(self) -> None:
            """Restore default chunking and pyramid settings.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Form controls are reset in-place.
            """
            self._hydrate(
                ZarrSaveConfig(
                    chunks_ptczyx=DEFAULT_ZARR_CHUNKS_PTCZYX,
                    pyramid_ptczyx=DEFAULT_ZARR_PYRAMID_PTCZYX,
                )
            )

        def _on_apply(self) -> None:
            """Validate user input and return selected Zarr configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Stores the selected :class:`ZarrSaveConfig` and accepts the
                dialog when validation succeeds.

            Raises
            ------
            None
                Validation issues are handled via GUI warnings.
            """
            chunks_ptczyx = tuple(
                self._chunk_inputs[axis_name].value() for axis_name in PTCZYX_AXES
            )

            pyramid_levels: list[Tuple[int, ...]] = []
            for axis_name in PTCZYX_AXES:
                text = self._pyramid_inputs[axis_name].text()
                try:
                    parsed = parse_pyramid_levels(text, axis_name=axis_name)
                except ValueError as exc:
                    QMessageBox.warning(self, "Invalid Pyramid Factors", str(exc))
                    return
                pyramid_levels.append(parsed)

            try:
                self.result_config = ZarrSaveConfig(
                    chunks_ptczyx=(
                        chunks_ptczyx[0],
                        chunks_ptczyx[1],
                        chunks_ptczyx[2],
                        chunks_ptczyx[3],
                        chunks_ptczyx[4],
                        chunks_ptczyx[5],
                    ),
                    pyramid_ptczyx=(
                        pyramid_levels[0],
                        pyramid_levels[1],
                        pyramid_levels[2],
                        pyramid_levels[3],
                        pyramid_levels[4],
                        pyramid_levels[5],
                    ),
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Zarr Settings", str(exc))
                return

            self.accept()

    class DaskBackendConfigDialog(QDialog):
        """Dialog for configuring Dask backend execution mode and parameters.

        Parameters
        ----------
        initial : DaskBackendConfig
            Initial backend configuration used to populate controls.
        parent : QDialog, optional
            Parent dialog widget.

        Attributes
        ----------
        result_config : DaskBackendConfig, optional
            Selected configuration when dialog is accepted.
        """

        def __init__(
            self,
            initial: DaskBackendConfig,
            recommendation_shape_tpczyx: Optional[
                Tuple[int, int, int, int, int, int]
            ] = None,
            recommendation_chunks_tpczyx: Optional[
                Tuple[int, int, int, int, int, int]
            ] = None,
            recommendation_dtype_itemsize: Optional[int] = None,
            parent: Optional[QDialog] = None,
        ) -> None:
            """Initialize dialog controls and hydrate from initial config.

            Parameters
            ----------
            initial : DaskBackendConfig
                Initial backend settings.
            parent : QDialog, optional
                Parent dialog widget.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__(parent)
            self.setWindowTitle("Dask Backend Settings")
            self.setMinimumWidth(940)
            self.setMinimumHeight(840)
            self.result_config: Optional[DaskBackendConfig] = None
            self._mode_index: Dict[str, int] = {}
            self._recommendation_shape_tpczyx = recommendation_shape_tpczyx
            self._recommendation_chunks_tpczyx = recommendation_chunks_tpczyx
            self._recommendation_dtype_itemsize = recommendation_dtype_itemsize
            self._latest_local_recommendation: Optional[LocalClusterRecommendation] = (
                None
            )

            self._build_ui()
            self._hydrate(initial)
            self.setStyleSheet(_popup_dialog_stylesheet())

        def _build_ui(self) -> None:
            """Build all backend configuration widgets and wire signals.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widgets are created and connected in-place.
            """
            root = QVBoxLayout(self)
            apply_popup_root_spacing(root)

            overview = QLabel(
                "Choose how ClearEx orchestrates Dask workers for loading and analysis. "
                "Mode guidance is shown below."
            )
            overview.setWordWrap(True)
            root.addWidget(overview)

            mode_row = QHBoxLayout()
            apply_row_spacing(mode_row)
            mode_label = QLabel("Backend mode:")
            self._mode_combo = QComboBox()
            mode_specs = (
                (
                    DASK_BACKEND_LOCAL_CLUSTER,
                    DASK_BACKEND_MODE_LABELS[DASK_BACKEND_LOCAL_CLUSTER],
                ),
                (
                    DASK_BACKEND_SLURM_RUNNER,
                    DASK_BACKEND_MODE_LABELS[DASK_BACKEND_SLURM_RUNNER],
                ),
                (
                    DASK_BACKEND_SLURM_CLUSTER,
                    DASK_BACKEND_MODE_LABELS[DASK_BACKEND_SLURM_CLUSTER],
                ),
            )
            for idx, (mode_key, mode_text) in enumerate(mode_specs):
                self._mode_combo.addItem(mode_text, mode_key)
                self._mode_index[mode_key] = idx
            mode_row.addWidget(mode_label)
            mode_row.addWidget(self._mode_combo, 1)
            root.addLayout(mode_row)

            self._mode_help_label = QLabel("")
            self._mode_help_label.setWordWrap(True)
            self._mode_help_label.setObjectName("metadataFieldValue")
            root.addWidget(self._mode_help_label)

            self._mode_stack = QStackedWidget()
            self._mode_stack.addWidget(self._build_local_cluster_page())
            self._mode_stack.addWidget(self._build_slurm_runner_page())
            self._mode_stack.addWidget(self._build_slurm_cluster_page())
            root.addWidget(self._mode_stack, 1)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            self._defaults_button = QPushButton("Reset Defaults")
            self._cancel_button = QPushButton("Cancel")
            self._apply_button = QPushButton("Apply")
            self._apply_button.setObjectName("runButton")
            footer.addWidget(self._defaults_button)
            footer.addStretch(1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._apply_button)
            root.addLayout(footer)

            self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            self._defaults_button.clicked.connect(self._on_reset_defaults)
            self._cancel_button.clicked.connect(self.reject)
            self._apply_button.clicked.connect(self._on_apply)

        def _build_local_cluster_page(self) -> QWidget:
            """Build page for LocalCluster options.

            Parameters
            ----------
            None

            Returns
            -------
            QWidget
                Constructed LocalCluster page widget.
            """
            page = QWidget()
            form = QFormLayout(page)
            apply_form_spacing(form)
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )

            self._local_workers_input = QLineEdit()
            self._local_workers_input.setPlaceholderText("blank = auto")
            form.addRow("Workers", self._local_workers_input)

            self._local_threads_spin = QSpinBox()
            self._local_threads_spin.setRange(1, 1024)
            form.addRow("Threads per worker", self._local_threads_spin)

            self._local_memory_input = QLineEdit()
            self._local_memory_input.setPlaceholderText("auto")
            form.addRow("Memory limit", self._local_memory_input)

            self._local_directory_input = QLineEdit()
            self._local_directory_browse = QPushButton("Browse")
            self._local_directory_browse.clicked.connect(
                lambda: self._browse_directory(self._local_directory_input)
            )
            local_dir_row = QHBoxLayout()
            apply_row_spacing(local_dir_row)
            local_dir_row.addWidget(self._local_directory_input, 1)
            local_dir_row.addWidget(self._local_directory_browse)
            local_dir_widget = QWidget()
            local_dir_widget.setLayout(local_dir_row)
            form.addRow("Local directory", local_dir_widget)

            self._local_recommend_button = QPushButton("Recommend Settings")
            self._local_recommend_button.clicked.connect(
                self._on_recommend_local_cluster
            )
            form.addRow("", self._local_recommend_button)

            self._local_recommendation_label = QLabel("")
            self._local_recommendation_label.setWordWrap(True)
            self._local_recommendation_label.setObjectName("metadataFieldValue")
            form.addRow("", self._local_recommendation_label)
            return page

        def _build_slurm_runner_page(self) -> QWidget:
            """Build page for SLURMRunner options.

            Parameters
            ----------
            None

            Returns
            -------
            QWidget
                Constructed SLURMRunner page widget.
            """
            page = QWidget()
            form = QFormLayout(page)
            apply_form_spacing(form)
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )

            self._runner_scheduler_file_input = QLineEdit()
            self._runner_scheduler_file_input.setPlaceholderText(
                "Path to scheduler file"
            )
            self._runner_scheduler_file_browse = QPushButton("Browse")
            self._runner_scheduler_file_browse.clicked.connect(
                self._on_browse_scheduler_file
            )
            scheduler_row = QHBoxLayout()
            apply_row_spacing(scheduler_row)
            scheduler_row.addWidget(self._runner_scheduler_file_input, 1)
            scheduler_row.addWidget(self._runner_scheduler_file_browse)
            scheduler_widget = QWidget()
            scheduler_widget.setLayout(scheduler_row)
            form.addRow("Scheduler file", scheduler_widget)

            self._runner_wait_workers_spin = QSpinBox()
            self._runner_wait_workers_spin.setRange(0, 100000)
            self._runner_wait_workers_spin.setSpecialValueText("auto")
            form.addRow("Wait for workers", self._runner_wait_workers_spin)
            return page

        def _build_slurm_cluster_page(self) -> QWidget:
            """Build page for SLURMCluster options.

            Parameters
            ----------
            None

            Returns
            -------
            QWidget
                Constructed SLURMCluster page widget.
            """
            page = QWidget()
            root = QVBoxLayout(page)
            apply_stack_spacing(root)

            worker_group = QGroupBox("Worker Job Settings")
            worker_form = QFormLayout(worker_group)
            apply_form_spacing(worker_form)
            worker_form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )

            self._cluster_workers_spin = QSpinBox()
            self._cluster_workers_spin.setRange(1, 100000)
            worker_form.addRow("Workers (jobs)", self._cluster_workers_spin)

            self._cluster_cores_spin = QSpinBox()
            self._cluster_cores_spin.setRange(1, 1024)
            worker_form.addRow("Cores", self._cluster_cores_spin)

            self._cluster_processes_spin = QSpinBox()
            self._cluster_processes_spin.setRange(1, 256)
            worker_form.addRow("Processes", self._cluster_processes_spin)

            self._cluster_memory_input = QLineEdit()
            worker_form.addRow("Memory", self._cluster_memory_input)

            self._cluster_local_directory_input = QLineEdit()
            self._cluster_local_directory_browse = QPushButton("Browse")
            self._cluster_local_directory_browse.clicked.connect(
                lambda: self._browse_directory(self._cluster_local_directory_input)
            )
            cluster_local_dir_row = QHBoxLayout()
            apply_row_spacing(cluster_local_dir_row)
            cluster_local_dir_row.addWidget(self._cluster_local_directory_input, 1)
            cluster_local_dir_row.addWidget(self._cluster_local_directory_browse)
            cluster_local_dir_widget = QWidget()
            cluster_local_dir_widget.setLayout(cluster_local_dir_row)
            worker_form.addRow("Local directory", cluster_local_dir_widget)

            self._cluster_interface_input = QLineEdit()
            worker_form.addRow("Interface", self._cluster_interface_input)

            self._cluster_walltime_input = QLineEdit()
            worker_form.addRow("Walltime", self._cluster_walltime_input)

            self._cluster_job_name_input = QLineEdit()
            worker_form.addRow("Job name", self._cluster_job_name_input)

            self._cluster_queue_input = QLineEdit()
            worker_form.addRow("Queue / partition", self._cluster_queue_input)

            self._cluster_death_timeout_input = QLineEdit()
            worker_form.addRow("Death timeout", self._cluster_death_timeout_input)

            self._cluster_mail_user_input = QLineEdit()
            self._cluster_mail_user_input.setPlaceholderText("name@institution.edu")
            worker_form.addRow("Mail user", self._cluster_mail_user_input)

            self._cluster_directives_input = QPlainTextEdit()
            self._cluster_directives_input.setPlaceholderText(
                "One Slurm directive per line"
            )
            self._cluster_directives_input.setMinimumHeight(110)
            worker_form.addRow("Extra directives", self._cluster_directives_input)

            scheduler_group = QGroupBox("Scheduler Options")
            scheduler_form = QFormLayout(scheduler_group)
            apply_form_spacing(scheduler_form)
            scheduler_form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )

            self._cluster_dashboard_input = QLineEdit()
            scheduler_form.addRow("Dashboard address", self._cluster_dashboard_input)

            self._cluster_scheduler_interface_input = QLineEdit()
            scheduler_form.addRow("Interface", self._cluster_scheduler_interface_input)

            self._cluster_idle_timeout_input = QLineEdit()
            scheduler_form.addRow("Idle timeout", self._cluster_idle_timeout_input)

            self._cluster_allowed_failures_spin = QSpinBox()
            self._cluster_allowed_failures_spin.setRange(1, 100000)
            scheduler_form.addRow(
                "Allowed failures", self._cluster_allowed_failures_spin
            )

            root.addWidget(worker_group)
            root.addWidget(scheduler_group)
            return page

        def _browse_directory(self, target_input: QLineEdit) -> None:
            """Prompt for a directory path and set the target field.

            Parameters
            ----------
            target_input : QLineEdit
                Line edit that receives the selected path.

            Returns
            -------
            None
                Target input is updated in-place.
            """
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Directory",
                str(Path.cwd()),
            )
            if directory:
                target_input.setText(directory)

        def _on_browse_scheduler_file(self) -> None:
            """Prompt for a scheduler file path for SLURMRunner mode.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Scheduler file field is updated in-place.
            """
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Dask Scheduler File",
                str(Path.cwd()),
                "All Files (*)",
            )
            if file_path:
                self._runner_scheduler_file_input.setText(file_path)

        def _on_mode_changed(self, _: int) -> None:
            """Update mode-specific page and help text after mode selection.

            Parameters
            ----------
            _ : int
                Unused combo-box index provided by Qt.

            Returns
            -------
            None
                Stack page and help text are updated in-place.
            """
            mode = str(self._mode_combo.currentData())
            mode_index = self._mode_index.get(mode, 0)
            self._mode_stack.setCurrentIndex(mode_index)
            self._mode_help_label.setText(_dask_mode_help_text(mode))

        def _hydrate(self, initial: DaskBackendConfig) -> None:
            """Populate all controls from an initial backend configuration.

            Parameters
            ----------
            initial : DaskBackendConfig
                Initial backend values.

            Returns
            -------
            None
                Widget values are updated in-place.
            """
            mode_index = self._mode_index.get(initial.mode, 0)
            self._mode_combo.setCurrentIndex(mode_index)

            local_cfg = initial.local_cluster
            self._local_workers_input.setText(
                "" if local_cfg.n_workers is None else str(local_cfg.n_workers)
            )
            self._local_threads_spin.setValue(local_cfg.threads_per_worker)
            self._local_memory_input.setText(local_cfg.memory_limit)
            self._local_directory_input.setText(local_cfg.local_directory or "")
            self._update_local_recommendation_label()

            runner_cfg = initial.slurm_runner
            self._runner_scheduler_file_input.setText(runner_cfg.scheduler_file or "")
            self._runner_wait_workers_spin.setValue(runner_cfg.wait_for_workers or 0)

            cluster_cfg = initial.slurm_cluster
            self._cluster_workers_spin.setValue(cluster_cfg.workers)
            self._cluster_cores_spin.setValue(cluster_cfg.cores)
            self._cluster_processes_spin.setValue(cluster_cfg.processes)
            self._cluster_memory_input.setText(cluster_cfg.memory)
            self._cluster_local_directory_input.setText(
                cluster_cfg.local_directory or ""
            )
            self._cluster_interface_input.setText(cluster_cfg.interface)
            self._cluster_walltime_input.setText(cluster_cfg.walltime)
            self._cluster_job_name_input.setText(cluster_cfg.job_name)
            self._cluster_queue_input.setText(cluster_cfg.queue)
            self._cluster_death_timeout_input.setText(cluster_cfg.death_timeout)
            self._cluster_mail_user_input.setText(cluster_cfg.mail_user or "")
            self._cluster_directives_input.setPlainText(
                "\n".join(cluster_cfg.job_extra_directives)
            )
            self._cluster_dashboard_input.setText(cluster_cfg.dashboard_address)
            self._cluster_scheduler_interface_input.setText(
                cluster_cfg.scheduler_interface
            )
            self._cluster_idle_timeout_input.setText(cluster_cfg.idle_timeout)
            self._cluster_allowed_failures_spin.setValue(cluster_cfg.allowed_failures)
            self._on_mode_changed(mode_index)

        def _on_reset_defaults(self) -> None:
            """Reset all controls to default backend settings.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Form controls are reset in-place.
            """
            self._hydrate(DaskBackendConfig())

        def _update_local_recommendation_label(
            self,
            recommendation: Optional[LocalClusterRecommendation] = None,
        ) -> None:
            """Update helper text for LocalCluster recommendations.

            Parameters
            ----------
            recommendation : LocalClusterRecommendation, optional
                Latest computed recommendation. When omitted, default guidance
                text is shown.

            Returns
            -------
            None
                Label text is updated in-place.
            """
            if recommendation is not None:
                self._latest_local_recommendation = recommendation
            recommendation = self._latest_local_recommendation
            if recommendation is None:
                if self._recommendation_shape_tpczyx is None:
                    self._local_recommendation_label.setText(
                        "Recommend Settings uses detected CPUs/RAM and current "
                        "chunk settings. Load metadata first for dataset-aware tuning."
                    )
                else:
                    self._local_recommendation_label.setText(
                        "Recommend Settings uses detected CPUs/RAM plus the "
                        "current dataset and chunk configuration."
                    )
                return
            self._local_recommendation_label.setText(
                format_local_cluster_recommendation_summary(recommendation)
            )

        def _on_recommend_local_cluster(self) -> None:
            """Populate LocalCluster fields from an automatic recommendation.

            Parameters
            ----------
            None

            Returns
            -------
            None
                LocalCluster inputs are updated in-place.
            """
            recommendation = recommend_local_cluster_config(
                shape_tpczyx=self._recommendation_shape_tpczyx,
                chunks_tpczyx=self._recommendation_chunks_tpczyx,
                dtype_itemsize=self._recommendation_dtype_itemsize,
            )
            config = recommendation.config
            self._local_workers_input.setText(
                "" if config.n_workers is None else str(config.n_workers)
            )
            self._local_threads_spin.setValue(int(config.threads_per_worker))
            self._local_memory_input.setText(str(config.memory_limit))
            self._update_local_recommendation_label(recommendation)

        def _parse_optional_positive_int(
            self,
            text: str,
            *,
            field_name: str,
        ) -> Optional[int]:
            """Parse optional positive integers from line-edit text.

            Parameters
            ----------
            text : str
                Input text.
            field_name : str
                Field name used in validation messages.

            Returns
            -------
            int, optional
                Parsed integer or ``None`` if empty.

            Raises
            ------
            ValueError
                If text is not a positive integer.
            """
            stripped = text.strip()
            if not stripped:
                return None
            try:
                value = int(stripped)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an integer.") from exc
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")
            return value

        def _on_apply(self) -> None:
            """Validate user input and return selected backend configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Stores :class:`DaskBackendConfig` and accepts dialog on success.

            Raises
            ------
            None
                Validation errors are handled via GUI warnings.
            """
            try:
                local_cfg = LocalClusterConfig(
                    n_workers=self._parse_optional_positive_int(
                        self._local_workers_input.text(),
                        field_name="Local workers",
                    ),
                    threads_per_worker=self._local_threads_spin.value(),
                    memory_limit=self._local_memory_input.text().strip() or "auto",
                    local_directory=self._local_directory_input.text().strip() or None,
                )

                runner_cfg = SlurmRunnerConfig(
                    scheduler_file=(
                        self._runner_scheduler_file_input.text().strip() or None
                    ),
                    wait_for_workers=(
                        None
                        if self._runner_wait_workers_spin.value() == 0
                        else self._runner_wait_workers_spin.value()
                    ),
                )

                raw_directives = [
                    line.strip()
                    for line in self._cluster_directives_input.toPlainText().splitlines()
                    if line.strip()
                ]
                directives = (
                    tuple(raw_directives)
                    if raw_directives
                    else DEFAULT_SLURM_CLUSTER_JOB_EXTRA_DIRECTIVES
                )

                mail_user = self._cluster_mail_user_input.text().strip()
                cluster_cfg = SlurmClusterConfig(
                    workers=self._cluster_workers_spin.value(),
                    cores=self._cluster_cores_spin.value(),
                    processes=self._cluster_processes_spin.value(),
                    memory=self._cluster_memory_input.text().strip(),
                    local_directory=(
                        self._cluster_local_directory_input.text().strip() or None
                    ),
                    interface=self._cluster_interface_input.text().strip(),
                    walltime=self._cluster_walltime_input.text().strip(),
                    job_name=self._cluster_job_name_input.text().strip(),
                    queue=self._cluster_queue_input.text().strip(),
                    death_timeout=self._cluster_death_timeout_input.text().strip(),
                    mail_user=mail_user or None,
                    job_extra_directives=directives,
                    dashboard_address=self._cluster_dashboard_input.text().strip(),
                    scheduler_interface=self._cluster_scheduler_interface_input.text().strip(),
                    idle_timeout=self._cluster_idle_timeout_input.text().strip(),
                    allowed_failures=self._cluster_allowed_failures_spin.value(),
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Dask Backend Settings", str(exc))
                return

            mode = str(self._mode_combo.currentData())
            if mode == DASK_BACKEND_SLURM_RUNNER and runner_cfg.scheduler_file is None:
                QMessageBox.warning(
                    self,
                    "Missing Scheduler File",
                    "SLURMRunner mode requires a scheduler file path.",
                )
                return
            if mode == DASK_BACKEND_SLURM_CLUSTER:
                if cluster_cfg.mail_user is None:
                    QMessageBox.warning(
                        self,
                        "Missing Email",
                        "SLURMCluster mode requires a mail user email address.",
                    )
                    return
                if "@" not in cluster_cfg.mail_user:
                    QMessageBox.warning(
                        self,
                        "Invalid Email",
                        "Mail user must look like a valid email address.",
                    )
                    return

            try:
                self.result_config = DaskBackendConfig(
                    mode=mode,
                    local_cluster=local_cfg,
                    slurm_runner=runner_cfg,
                    slurm_cluster=cluster_cfg,
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Dask Backend Settings", str(exc))
                return

            self.accept()

    class DataStoreMaterializationWorker(QThread):
        """Background worker that materializes canonical store data.

        Parameters
        ----------
        experiment : NavigateExperiment
            Parsed Navigate experiment metadata.
        source_data_path : pathlib.Path
            Source acquisition data path.
        dask_backend : DaskBackendConfig
            Backend configuration used for Dask execution.
        zarr_save : ZarrSaveConfig
            Zarr chunk and pyramid configuration.

        Attributes
        ----------
        progress_changed : pyqtSignal
            Signal with ``(percent, message)`` progress payload.
        succeeded : pyqtSignal
            Signal with resulting store path string.
        failed : pyqtSignal
            Signal with error text.
        """

        progress_changed = pyqtSignal(int, str)
        succeeded = pyqtSignal(str)
        failed = pyqtSignal(str, str)

        def __init__(
            self,
            *,
            experiment: NavigateExperiment,
            source_data_path: Path,
            dask_backend: DaskBackendConfig,
            zarr_save: ZarrSaveConfig,
        ) -> None:
            """Initialize worker state.

            Parameters
            ----------
            experiment : NavigateExperiment
                Parsed Navigate experiment metadata.
            source_data_path : pathlib.Path
                Source acquisition data path.
            dask_backend : DaskBackendConfig
                Backend configuration used for Dask execution.
            zarr_save : ZarrSaveConfig
                Zarr chunk and pyramid configuration.

            Returns
            -------
            None
                Worker is initialized in-place.
            """
            super().__init__()
            self._experiment = experiment
            self._source_data_path = source_data_path
            self._dask_backend = dask_backend
            self._zarr_save = zarr_save

        def _emit_progress(self, percent: int, message: str) -> None:
            """Emit stage progress updates from worker thread.

            Parameters
            ----------
            percent : int
                Progress percentage value.
            message : str
                Human-readable stage text.

            Returns
            -------
            None
                Signal side effects only.
            """
            self.progress_changed.emit(int(percent), str(message))

        def run(self) -> None:
            """Execute materialization workflow in the background.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Emits success/failure signals on completion.

            Raises
            ------
            None
                Exceptions are converted to ``failed`` signals.
            """
            try:
                with ExitStack() as exit_stack:
                    client = _configure_dask_backend_client(
                        self._dask_backend,
                        exit_stack=exit_stack,
                    )
                    result = materialize_experiment_data_store(
                        experiment=self._experiment,
                        source_path=self._source_data_path,
                        chunks=self._zarr_save.chunks_tpczyx(),
                        pyramid_factors=self._zarr_save.pyramid_tpczyx(),
                        client=client,
                        progress_callback=self._emit_progress,
                    )
                self.succeeded.emit(str(result.store_path))
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Canonical store materialization failed for %s.",
                    self._source_data_path,
                )
                self.failed.emit(
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )

    class MaterializationProgressDialog(QDialog):
        """Modal progress dialog for canonical store creation.

        Parameters
        ----------
        parent : QDialog, optional
            Parent window.
        """

        def __init__(self, parent: Optional[QDialog] = None) -> None:
            """Initialize progress dialog widgets.

            Parameters
            ----------
            parent : QDialog, optional
                Parent window.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__(parent)
            self.setWindowTitle("Preparing Data Store")
            self.setMinimumWidth(620)
            self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

            root = QVBoxLayout(self)
            apply_window_root_spacing(root)

            title = QLabel("Building `data_store.zarr`")
            title.setObjectName("progressTitle")
            root.addWidget(title)

            self._message_label = QLabel("Starting materialization...")
            self._message_label.setObjectName("progressMessage")
            self._message_label.setWordWrap(True)
            root.addWidget(self._message_label)

            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setTextVisible(True)
            self._progress.setFormat("%p%")
            root.addWidget(self._progress)

            self.setStyleSheet(
                """
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
                }
                QLabel#progressTitle {
                    font-size: 18px;
                    font-weight: 700;
                    color: #f0f5ff;
                }
                QLabel#progressMessage {
                    color: #a6b7d0;
                }
                QProgressBar {
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                    background-color: #111925;
                    text-align: center;
                    height: 22px;
                    color: #e6edf3;
                }
                QProgressBar::chunk {
                    border-radius: 7px;
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #1f7fdc, stop:1 #33c3a5
                    );
                }
                """
            )

        def update_progress(self, percent: int, message: str) -> None:
            """Update progress bar and stage text.

            Parameters
            ----------
            percent : int
                Progress percentage value.
            message : str
                Human-readable stage text.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            self._progress.setValue(max(0, min(100, int(percent))))
            self._message_label.setText(str(message))

    class AnalysisExecutionWorker(QThread):
        """Background worker that executes selected analysis workflows.

        Parameters
        ----------
        workflow : WorkflowConfig
            Workflow configuration selected in GUI.
        run_callback : callable
            Callback with signature ``(workflow, progress_callback)`` that
            executes the workflow.

        Attributes
        ----------
        progress_changed : pyqtSignal
            Signal with ``(percent, message)`` progress payload.
        succeeded : pyqtSignal
            Signal emitted when execution completes successfully.
        failed : pyqtSignal
            Signal carrying an error message when execution fails.
        """

        progress_changed = pyqtSignal(int, str)
        succeeded = pyqtSignal()
        failed = pyqtSignal(str, str)

        def __init__(
            self,
            *,
            workflow: WorkflowConfig,
            run_callback: Callable[
                [WorkflowConfig, Callable[[int, str], None]],
                None,
            ],
        ) -> None:
            """Initialize worker state.

            Parameters
            ----------
            workflow : WorkflowConfig
                Workflow configuration selected in GUI.
            run_callback : callable
                Workflow execution callback.

            Returns
            -------
            None
                Worker is initialized in-place.
            """
            super().__init__()
            self._workflow = workflow
            self._run_callback = run_callback

        def _emit_progress(self, percent: int, message: str) -> None:
            """Emit progress updates from worker thread.

            Parameters
            ----------
            percent : int
                Progress percentage.
            message : str
                Human-readable status text.

            Returns
            -------
            None
                Signal side effects only.
            """
            self.progress_changed.emit(int(percent), str(message))

        def run(self) -> None:
            """Execute the configured workflow in the background.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Emits success/failure signals.
            """
            try:
                self._emit_progress(1, "Starting analysis workflow...")
                self._run_callback(self._workflow, self._emit_progress)
                self._emit_progress(100, "Analysis workflow completed.")
                self.succeeded.emit()
            except Exception as exc:
                self.failed.emit(
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )

    class AnalysisExecutionProgressDialog(QDialog):
        """Modal progress dialog for analysis execution.

        Parameters
        ----------
        parent : QDialog, optional
            Parent window.
        """

        def __init__(self, parent: Optional[QDialog] = None) -> None:
            """Initialize analysis progress dialog widgets.

            Parameters
            ----------
            parent : QDialog, optional
                Parent window.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__(parent)
            self.setWindowTitle("Running Analysis")
            self.setMinimumWidth(640)
            self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

            root = QVBoxLayout(self)
            apply_window_root_spacing(root)

            title = QLabel("Executing selected analysis routines")
            title.setObjectName("progressTitle")
            root.addWidget(title)

            self._message_label = QLabel("Preparing analysis...")
            self._message_label.setObjectName("progressMessage")
            self._message_label.setWordWrap(True)
            root.addWidget(self._message_label)

            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setTextVisible(True)
            self._progress.setFormat("%p%")
            root.addWidget(self._progress)

            self.setStyleSheet(
                """
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
                }
                QLabel#progressTitle {
                    font-size: 18px;
                    font-weight: 700;
                    color: #f0f5ff;
                }
                QLabel#progressMessage {
                    color: #a6b7d0;
                }
                QProgressBar {
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                    background-color: #111925;
                    text-align: center;
                    height: 22px;
                    color: #e6edf3;
                }
                QProgressBar::chunk {
                    border-radius: 7px;
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #2f81f7, stop:1 #33c3a5
                    );
                }
                """
            )

        def update_progress(self, percent: int, message: str) -> None:
            """Update progress bar and stage message.

            Parameters
            ----------
            percent : int
                Progress percentage value.
            message : str
                Human-readable stage text.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            self._progress.setValue(max(0, min(100, int(percent))))
            self._message_label.setText(str(message))

    class ClearExSetupDialog(QDialog):
        """First-step GUI dialog for experiment setup and store readiness."""

        def __init__(self, initial: WorkflowConfig) -> None:
            """Initialize setup window state and widgets.

            Parameters
            ----------
            initial : WorkflowConfig
                Initial workflow values for pre-population.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__()
            self.setWindowTitle("ClearEx")
            self.setMinimumWidth(1100)
            self.setMinimumHeight(780)
            self.setAcceptDrops(True)

            self._opener = ImageOpener()
            self.result_config: Optional[WorkflowConfig] = None
            self._metadata_labels: Dict[str, QLabel] = {}
            self._dask_backend_config: DaskBackendConfig = initial.dask_backend
            self._zarr_save_config: ZarrSaveConfig = initial.zarr_save
            self._chunks = initial.chunks
            self._loaded_experiment: Optional[NavigateExperiment] = None
            self._loaded_experiment_path: Optional[Path] = None
            self._loaded_image_info: Optional[ImageInfo] = None
            self._loaded_source_data_path: Optional[Path] = None
            self._source_data_directory_override: Optional[Path] = None
            self._materialization_worker: Optional[DataStoreMaterializationWorker] = (
                None
            )

            self._build_ui()
            self._apply_theme()
            self._hydrate(initial)

        def _build_ui(self) -> None:
            """Build setup window widgets and connect actions.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widgets are created and connected in-place.
            """
            root = QVBoxLayout(self)
            apply_window_root_spacing(root)

            header = QFrame()
            header.setObjectName("headerCard")
            header_layout = QVBoxLayout(header)
            apply_stack_spacing(header_layout)
            header_image = _create_scaled_branding_label(
                filename=_GUI_HEADER_IMAGE,
                max_width=840,
                max_height=150,
                object_name="brandHeaderImage",
            )
            if header_image is not None:
                header_layout.addWidget(header_image)
            root.addWidget(header)

            data_group = QGroupBox("Navigate Experiment")
            data_layout = QVBoxLayout(data_group)
            apply_stack_spacing(data_layout)

            path_row = QHBoxLayout()
            apply_row_spacing(path_row)
            self._path_input = QLineEdit()
            self._path_input.setPlaceholderText(
                "Select Navigate experiment.yml or experiment.yaml"
            )
            self._path_input.setToolTip(
                "Drop Navigate experiment.yml/experiment.yaml here."
            )
            self._path_input.setAcceptDrops(True)
            self._path_input.installEventFilter(self)
            self._browse_file_button = QPushButton("Browse Experiment")
            self._load_button = QPushButton("Load Metadata")
            path_row.addWidget(self._path_input, 1)
            path_row.addWidget(self._browse_file_button)
            path_row.addWidget(self._load_button)
            data_layout.addLayout(path_row)

            dask_backend_row = QHBoxLayout()
            apply_row_spacing(dask_backend_row)
            dask_backend_label = QLabel("Dask backend:")
            dask_backend_label.setObjectName("metadataFieldLabel")
            self._dask_backend_summary = QLabel("n/a")
            self._dask_backend_summary.setObjectName("metadataFieldValue")
            self._dask_backend_summary.setWordWrap(True)
            self._dask_backend_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._dask_backend_button = QPushButton("Edit Dask Backend")
            dask_backend_row.addWidget(dask_backend_label)
            dask_backend_row.addWidget(self._dask_backend_summary, 1)
            dask_backend_row.addWidget(self._dask_backend_button)
            data_layout.addLayout(dask_backend_row)

            zarr_row = QHBoxLayout()
            apply_row_spacing(zarr_row)
            zarr_label = QLabel("Zarr save config:")
            zarr_label.setObjectName("metadataFieldLabel")
            self._zarr_config_summary = QLabel("n/a")
            self._zarr_config_summary.setObjectName("zarrConfigSummary")
            self._zarr_config_summary.setWordWrap(True)
            self._zarr_config_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._zarr_config_button = QPushButton("Edit Zarr Settings")
            zarr_row.addWidget(zarr_label)
            zarr_row.addWidget(self._zarr_config_summary, 1)
            zarr_row.addWidget(self._zarr_config_button)
            data_layout.addLayout(zarr_row)

            root.addWidget(data_group)

            metadata_group = QGroupBox("Image Metadata")
            metadata_layout = QGridLayout(metadata_group)
            apply_metadata_grid_spacing(metadata_layout)

            metadata_items = (
                ("path", "Path"),
                ("shape", "Shape"),
                ("dtype", "Data type"),
                ("axes", "Axes"),
                ("channels", "Channels"),
                ("positions", "Positions"),
                ("image_size", "Image size"),
                ("time_points", "Time points"),
                ("pixel_size", "Pixel size (um)"),
                ("metadata_keys", "Metadata keys"),
            )
            for idx, (key, label) in enumerate(metadata_items):
                row = idx // 2
                pair_col = (idx % 2) * 2

                field_label = QLabel(f"{label}:")
                field_label.setObjectName("metadataFieldLabel")

                value_label = QLabel("n/a")
                value_label.setObjectName("metadataFieldValue")
                value_label.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                value_label.setWordWrap(True)
                self._metadata_labels[key] = value_label
                metadata_layout.addWidget(
                    field_label, row, pair_col, alignment=Qt.AlignmentFlag.AlignTop
                )
                metadata_layout.addWidget(value_label, row, pair_col + 1)

            metadata_layout.setColumnStretch(1, 1)
            metadata_layout.setColumnStretch(3, 1)
            root.addWidget(metadata_group)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            self._status_label = QLabel("Ready")
            self._status_label.setObjectName("statusLabel")
            self._cancel_button = QPushButton("Cancel")
            self._next_button = QPushButton("Next")
            self._next_button.setObjectName("runButton")
            footer.addWidget(self._status_label, 1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._next_button)
            root.addLayout(footer)

            self._browse_file_button.clicked.connect(self._on_browse_file)
            self._load_button.clicked.connect(self._on_load_metadata)
            self._path_input.textChanged.connect(self._on_experiment_path_changed)
            self._dask_backend_button.clicked.connect(self._on_edit_dask_backend)
            self._zarr_config_button.clicked.connect(self._on_edit_zarr_settings)
            self._cancel_button.clicked.connect(self.reject)
            self._next_button.clicked.connect(self._on_next)

        def _hydrate(self, initial: WorkflowConfig) -> None:
            """Populate setup window fields from initial workflow config.

            Parameters
            ----------
            initial : WorkflowConfig
                Initial workflow values for the setup window.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            self._path_input.setText(initial.file or "")
            self._refresh_dask_backend_summary()
            self._refresh_zarr_save_summary()

        def _refresh_dask_backend_summary(self) -> None:
            """Refresh setup summary text for Dask backend configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary labels are updated in-place.
            """
            summary = format_dask_backend_summary(self._dask_backend_config)
            self._dask_backend_summary.setText(summary)
            self._dask_backend_summary.setToolTip(summary)

        def _refresh_zarr_save_summary(self) -> None:
            """Refresh setup summary text for Zarr save configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary labels are updated in-place.
            """
            summary = _format_zarr_save_summary(self._zarr_save_config)
            self._zarr_config_summary.setText(summary)
            self._zarr_config_summary.setToolTip(summary)

        def _on_edit_dask_backend(self) -> None:
            """Open backend dialog and apply selected configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Selected backend values are stored in-place.
            """
            dialog = DaskBackendConfigDialog(
                initial=self._dask_backend_config,
                recommendation_shape_tpczyx=self._current_local_cluster_shape_tpczyx(),
                recommendation_chunks_tpczyx=self._zarr_save_config.chunks_tpczyx(),
                recommendation_dtype_itemsize=self._current_dtype_itemsize(),
                parent=self,
            )
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.result_config is None:
                return
            self._dask_backend_config = dialog.result_config
            self._refresh_dask_backend_summary()
            self._set_status("Updated Dask backend settings.")

        def _on_edit_zarr_settings(self) -> None:
            """Open Zarr settings dialog and apply selected configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Selected Zarr settings are stored in-place.
            """
            dialog = ZarrSaveConfigDialog(initial=self._zarr_save_config, parent=self)
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.result_config is None:
                return
            self._zarr_save_config = dialog.result_config
            self._refresh_zarr_save_summary()
            self._set_status("Updated Zarr save settings.")

        def _apply_theme(self) -> None:
            """Apply stylesheet-based theme for setup window.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Styles are applied in-place.
            """
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
                    font-size: 13px;
                }
                QLabel {
                    color: #d9e2f1;
                }
                #headerCard {
                    background-color: #f0f2ef;
                    border: 1px solid #f0f2ef;
                    border-radius: 12px;
                    padding: 10px;
                }
                QLabel#brandHeaderImage {
                    padding: 0;
                    margin-bottom: 2px;
                    background: transparent;
                }
                QLabel#title {
                    font-size: 24px;
                    font-weight: 700;
                    color: #f0f5ff;
                }
                QLabel#subtitle {
                    font-size: 13px;
                    color: #a6b7d0;
                }
                QGroupBox {
                    border: 1px solid #2a3442;
                    border-radius: 10px;
                    margin-top: 16px;
                    padding: 14px;
                    background-color: #111925;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 6px;
                    color: #9cc6ff;
                }
                QLabel#metadataFieldLabel {
                    color: #9cc6ff;
                    font-weight: 600;
                }
                QLabel#metadataFieldValue {
                    color: #d9e2f1;
                }
                QLabel#zarrConfigSummary {
                    color: #d9e2f1;
                }
                QLineEdit {
                    background-color: #0b1320;
                    border: 1px solid #2b3f58;
                    border-radius: 8px;
                    min-height: 30px;
                    padding: 5px 10px;
                    color: #e6edf3;
                    selection-background-color: #2f81f7;
                }
                QCheckBox {
                    spacing: 8px;
                    color: #d9e2f1;
                }
                QPushButton {
                    background-color: #1a2635;
                    border: 1px solid #2f4460;
                    border-radius: 8px;
                    padding: 9px 14px;
                    color: #dbe9ff;
                }
                QPushButton:hover {
                    background-color: #22354c;
                }
                QPushButton:pressed {
                    background-color: #182639;
                }
                QPushButton#runButton {
                    background-color: #2f81f7;
                    border-color: #2f81f7;
                    color: #f8fbff;
                    font-weight: 700;
                }
                QPushButton#runButton:hover {
                    background-color: #1f6cd8;
                }
                QLabel#statusLabel {
                    color: #9ab0ca;
                }
                """
            )

        def _set_status(self, text: str) -> None:
            """Set setup footer status text.

            Parameters
            ----------
            text : str
                Status message text.

            Returns
            -------
            None
                Label text is updated in-place.
            """
            self._status_label.setText(text)

        def _resolve_dropped_experiment_path(
            self,
            mime_data: QMimeData,
        ) -> Optional[Path]:
            """Resolve a dropped Navigate experiment descriptor path.

            Parameters
            ----------
            mime_data : QMimeData
                Drag/drop payload to inspect.

            Returns
            -------
            pathlib.Path, optional
                Resolved local path to ``experiment.yml`` or
                ``experiment.yaml`` when available, otherwise ``None``.
            """
            if not mime_data.hasUrls():
                return None

            for url in mime_data.urls():
                if not url.isLocalFile():
                    continue

                candidate_path = Path(url.toLocalFile()).expanduser()
                if candidate_path.is_dir():
                    for name in ("experiment.yml", "experiment.yaml"):
                        experiment_path = candidate_path / name
                        if experiment_path.exists():
                            return experiment_path.resolve()
                    continue

                if candidate_path.exists() and is_navigate_experiment_file(
                    candidate_path
                ):
                    return candidate_path.resolve()
            return None

        def _can_accept_experiment_drop(self, mime_data: QMimeData) -> bool:
            """Determine whether dropped payload contains an experiment path.

            Parameters
            ----------
            mime_data : QMimeData
                Drag/drop payload to inspect.

            Returns
            -------
            bool
                ``True`` when a supported local Navigate experiment path is
                present, otherwise ``False``.
            """
            return self._resolve_dropped_experiment_path(mime_data) is not None

        def _apply_experiment_drop(self, mime_data: QMimeData) -> bool:
            """Apply dropped Navigate experiment path to the setup input field.

            Parameters
            ----------
            mime_data : QMimeData
                Drag/drop payload to inspect.

            Returns
            -------
            bool
                ``True`` when a supported path was applied, otherwise ``False``.
            """
            experiment_path = self._resolve_dropped_experiment_path(mime_data)
            if experiment_path is None:
                return False

            self._path_input.setText(str(experiment_path))
            self._set_status(
                "Experiment path set from drag-and-drop. Click Load Metadata or Next."
            )
            return True

        def _clear_loaded_experiment_context(self) -> None:
            """Clear cached experiment metadata and source-path override.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Cached setup state is reset in-place.
            """
            self._loaded_experiment = None
            self._loaded_experiment_path = None
            self._loaded_image_info = None
            self._loaded_source_data_path = None
            self._source_data_directory_override = None

        def _on_experiment_path_changed(self, _text: str) -> None:
            """Reset cached experiment state when the selected path changes.

            Parameters
            ----------
            _text : str
                Current text content. Unused.

            Returns
            -------
            None
                Cached setup state is reset in-place.
            """
            self._clear_loaded_experiment_context()

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:
            """Handle drag/drop events routed through the path input widget.

            Parameters
            ----------
            watched : QObject
                Widget emitting the event.
            event : QEvent
                Qt event object to evaluate.

            Returns
            -------
            bool
                ``True`` when the event was handled, otherwise base class
                event filter result.
            """
            if watched is self._path_input:
                if isinstance(event, QDragEnterEvent):
                    if self._can_accept_experiment_drop(event.mimeData()):
                        event.acceptProposedAction()
                        return True
                    event.ignore()
                    return True
                if isinstance(event, QDragMoveEvent):
                    if self._can_accept_experiment_drop(event.mimeData()):
                        event.acceptProposedAction()
                        return True
                    event.ignore()
                    return True
                if isinstance(event, QDropEvent):
                    if self._apply_experiment_drop(event.mimeData()):
                        event.acceptProposedAction()
                        return True
                    event.ignore()
                    return True
            return super().eventFilter(watched, event)

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            """Accept dialog-level drags containing Navigate experiment files.

            Parameters
            ----------
            event : QDragEnterEvent
                Incoming drag-enter event.

            Returns
            -------
            None
                Event acceptance is updated in-place.
            """
            if self._can_accept_experiment_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def dragMoveEvent(self, event: QDragMoveEvent) -> None:
            """Accept dialog-level drag-move events for valid experiment files.

            Parameters
            ----------
            event : QDragMoveEvent
                Incoming drag-move event.

            Returns
            -------
            None
                Event acceptance is updated in-place.
            """
            if self._can_accept_experiment_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:
            """Handle dialog-level file drops for experiment path selection.

            Parameters
            ----------
            event : QDropEvent
                Incoming drop event.

            Returns
            -------
            None
                Path input and event acceptance are updated in-place.
            """
            if self._apply_experiment_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def _on_browse_file(self) -> None:
            """Open file picker for Navigate experiment descriptor.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Selected path is written into the input field.
            """
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Navigate experiment.yml",
                str(Path.cwd()),
                "Navigate Experiment (experiment.yml experiment.yaml *.yml *.yaml)",
            )
            if file_path:
                self._path_input.setText(file_path)

        def _prompt_for_source_data_directory(
            self,
            *,
            experiment: NavigateExperiment,
            error: ExperimentDataResolutionError,
        ) -> Optional[Path]:
            """Prompt for a replacement acquisition directory after lookup failure.

            Parameters
            ----------
            experiment : NavigateExperiment
                Parsed experiment metadata.
            error : ExperimentDataResolutionError
                Initial resolution failure details.

            Returns
            -------
            pathlib.Path, optional
                Resolved source data path when the user selected a valid
                replacement directory, otherwise ``None``.
            """
            QMessageBox.warning(
                self,
                "Source Data Not Found",
                "ClearEx could not locate the acquisition data from the "
                "experiment metadata.\n\n"
                f"{error}\n\n"
                "Select the directory containing the moved acquisition data.",
            )

            start_directory = str(experiment.path.parent)
            while True:
                selected_directory = QFileDialog.getExistingDirectory(
                    self,
                    "Select Navigate Acquisition Directory",
                    start_directory,
                )
                if not selected_directory:
                    return None

                override_directory = Path(selected_directory).expanduser().resolve()
                try:
                    source_data_path = resolve_experiment_data_path(
                        experiment,
                        search_directory=override_directory,
                    )
                except ExperimentDataResolutionError as inner_error:
                    QMessageBox.warning(
                        self,
                        "Source Data Not Found",
                        f"{inner_error}\n\nPlease choose a different directory.",
                    )
                    start_directory = str(override_directory)
                    continue

                self._source_data_directory_override = override_directory
                return source_data_path

        def _load_experiment_context(
            self,
            *,
            path_text: str,
        ) -> tuple[Path, NavigateExperiment, Path, ImageInfo]:
            """Load experiment and source metadata for setup validation.

            Parameters
            ----------
            path_text : str
                User-entered experiment path text.

            Returns
            -------
            tuple[pathlib.Path, NavigateExperiment, pathlib.Path, ImageInfo]
                Experiment path, parsed experiment, resolved source path, and
                source metadata.

            Raises
            ------
            ValueError
                If path is missing or not an experiment descriptor.
            FileNotFoundError
                If the selected path does not exist.
            Exception
                Propagates parse/read failures from experiment or image I/O.
            """
            if not path_text:
                raise ValueError("Select a Navigate experiment.yml path first.")
            selected_path = Path(path_text).expanduser()
            if not selected_path.exists():
                raise FileNotFoundError(f"Path does not exist: {selected_path}")
            if not is_navigate_experiment_file(selected_path):
                raise ValueError(
                    "This setup window requires Navigate experiment.yml or "
                    "experiment.yaml."
                )

            experiment_path = selected_path.resolve()
            experiment = load_navigate_experiment(experiment_path)
            try:
                source_data_path = resolve_experiment_data_path(
                    experiment,
                    search_directory=self._source_data_directory_override,
                )
            except ExperimentDataResolutionError as exc:
                source_data_path = self._prompt_for_source_data_directory(
                    experiment=experiment,
                    error=exc,
                )
                if source_data_path is None:
                    raise
            _, info = self._opener.open(
                path=str(source_data_path),
                prefer_dask=True,
                chunks=self._chunks,
            )
            return experiment_path, experiment, source_data_path, info

        def _on_load_metadata(self) -> None:
            """Load and display source metadata from selected experiment.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Metadata display fields and internal setup state are updated.
            """
            try:
                experiment_path, experiment, source_data_path, info = (
                    self._load_experiment_context(
                        path_text=self._path_input.text().strip()
                    )
                )
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Failed to load experiment metadata from %s.",
                    self._path_input.text().strip(),
                )
                _show_themed_error_dialog(
                    self,
                    "Metadata Load Failed",
                    "Failed to load experiment metadata.",
                    summary=f"{type(exc).__name__}: {exc}",
                    details=traceback.format_exc(),
                )
                self._set_status("Failed to load metadata.")
                return

            summary = summarize_image_info(info)
            summary = _apply_experiment_overrides(
                summary=summary,
                experiment_path=experiment_path,
                resolved_data_path=source_data_path,
                experiment=experiment,
            )

            for key, value in summary.items():
                self._metadata_labels[key].setText(value)

            self._loaded_experiment = experiment
            self._loaded_experiment_path = experiment_path
            self._loaded_image_info = info
            self._loaded_source_data_path = source_data_path

            target_store = resolve_data_store_path(experiment, source_data_path)
            self._set_status(f"Metadata loaded. Target store: {target_store}")

        def _current_local_cluster_shape_tpczyx(
            self,
        ) -> Optional[Tuple[int, int, int, int, int, int]]:
            """Return current canonical shape estimate for LocalCluster tuning.

            Parameters
            ----------
            None

            Returns
            -------
            tuple[int, int, int, int, int, int], optional
                Canonical ``(t, p, c, z, y, x)`` shape estimate when metadata
                has been loaded.
            """
            if self._loaded_experiment is None:
                return None
            z_size, y_size, x_size = infer_zyx_shape(
                self._loaded_experiment,
                self._loaded_image_info,
            )
            return (
                int(self._loaded_experiment.timepoints),
                int(self._loaded_experiment.multiposition_count),
                int(self._loaded_experiment.channel_count),
                int(z_size),
                int(y_size),
                int(x_size),
            )

        def _current_dtype_itemsize(self) -> Optional[int]:
            """Return source bytes-per-voxel for LocalCluster tuning.

            Parameters
            ----------
            None

            Returns
            -------
            int, optional
                Source dtype itemsize when metadata has been loaded.
            """
            if self._loaded_image_info is None or self._loaded_image_info.dtype is None:
                return None
            try:
                return max(1, int(getattr(self._loaded_image_info.dtype, "itemsize")))
            except Exception:
                return None

        def _accept_with_store_path(self, store_path: Path) -> None:
            """Finalize setup dialog with prepared store path configuration.

            Parameters
            ----------
            store_path : pathlib.Path
                Prepared canonical store path.

            Returns
            -------
            None
                Stores setup workflow config and accepts this dialog.
            """
            self.result_config = WorkflowConfig(
                file=str(store_path),
                prefer_dask=True,
                dask_backend=self._dask_backend_config,
                chunks=self._chunks,
                flatfield=False,
                deconvolution=False,
                shear_transform=False,
                particle_detection=False,
                registration=False,
                visualization=False,
                mip_export=False,
                zarr_save=self._zarr_save_config,
            )
            _save_last_used_dask_backend_config(self._dask_backend_config)
            self.accept()

        def _on_next(self) -> None:
            """Advance to analysis-selection step after store readiness checks.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Proceeds only when canonical store is confirmed ready.
            """
            path_text = self._path_input.text().strip()
            needs_reload = True
            if self._loaded_experiment_path is not None:
                needs_reload = str(self._loaded_experiment_path) != str(
                    Path(path_text).expanduser().resolve()
                )

            if (
                needs_reload
                or self._loaded_experiment is None
                or self._loaded_source_data_path is None
            ):
                self._on_load_metadata()
                if (
                    self._loaded_experiment is None
                    or self._loaded_source_data_path is None
                ):
                    return

            experiment = self._loaded_experiment
            source_data_path = self._loaded_source_data_path
            target_store = resolve_data_store_path(experiment, source_data_path)

            if target_store.exists() and has_complete_canonical_data_store(
                target_store,
                expected_chunks_tpczyx=self._zarr_save_config.chunks_tpczyx(),
                expected_pyramid_factors=self._zarr_save_config.pyramid_tpczyx(),
            ):
                self._set_status(
                    "Found existing data store. Opening analysis selection."
                )
                self._accept_with_store_path(target_store)
                return
            if target_store.exists():
                self._set_status(
                    "Existing store found, but canonical 6D data is missing or "
                    "incompatible. Rebuilding canonical store."
                )

            progress_dialog = MaterializationProgressDialog(parent=self)
            failure_payload: dict[str, str] = {}
            success_paths: list[Path] = []

            worker = DataStoreMaterializationWorker(
                experiment=experiment,
                source_data_path=source_data_path,
                dask_backend=self._dask_backend_config,
                zarr_save=self._zarr_save_config,
            )
            self._materialization_worker = worker
            worker.progress_changed.connect(progress_dialog.update_progress)
            worker.succeeded.connect(lambda store: success_paths.append(Path(store)))
            worker.succeeded.connect(lambda _: progress_dialog.accept())
            worker.failed.connect(
                lambda summary, details: failure_payload.update(
                    {"summary": str(summary), "details": str(details)}
                )
            )
            worker.failed.connect(lambda *_: progress_dialog.reject())

            worker.start()
            progress_dialog.exec()
            worker.wait()
            self._materialization_worker = None

            if failure_payload:
                _show_themed_error_dialog(
                    self,
                    "Store Creation Failed",
                    "Failed to create canonical data store.",
                    summary=failure_payload.get("summary"),
                    details=failure_payload.get("details"),
                )
                self._set_status("Store creation failed.")
                return

            if not success_paths:
                self._set_status("Store creation was cancelled.")
                return

            store_path = success_paths[0]
            self._set_status(
                "Created canonical data store. Opening analysis selection."
            )
            self._accept_with_store_path(store_path)

    class AnalysisSelectionDialog(QDialog):
        """Second-step GUI dialog for selecting and sequencing analysis operations."""

        _OPERATION_KEYS: tuple[str, ...] = (
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "visualization",
            "mip_export",
        )
        _PROVENANCE_HISTORY_OPERATIONS: tuple[str, ...] = (
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "mip_export",
        )
        _OPERATION_LABELS: Dict[str, str] = {
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shearing",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "visualization": "Visualization",
            "mip_export": "MIP Export",
        }
        _OPERATION_TABS: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                "Preprocessing",
                ("flatfield", "deconvolution", "shear_transform"),
            ),
            ("Segmentation", ("particle_detection", "usegment3d")),
            ("Postprocessing", ("registration",)),
            ("Export", ("visualization", "mip_export")),
        )
        _OPERATION_OUTPUT_COMPONENTS: Dict[str, str] = {
            "flatfield": "results/flatfield/latest/data",
            "deconvolution": "results/deconvolution/latest/data",
            "shear_transform": "results/shear_transform/latest/data",
            "usegment3d": "results/usegment3d/latest/data",
            "registration": "results/registration/latest/data",
            "mip_export": "results/mip_export/latest",
        }
        _PARTICLE_DETECTION_OVERLAY_COMPONENT = (
            "results/particle_detection/latest/detections"
        )
        _DEFAULT_USEGMENT3D_PARAMETERS: Dict[str, Any] = {
            "execution_order": 5,
            "input_source": "data",
            "force_rerun": False,
            "chunk_basis": "3d",
            "detect_2d_per_slice": False,
            "use_map_overlap": False,
            "overlap_zyx": [12, 24, 24],
            "memory_overhead_factor": 3.0,
            "channel_indices": [0],
            "channel_index": 0,
            "use_views": ["xy", "xz", "yz"],
            "input_resolution_level": 0,
            "output_reference_space": "level0",
            "save_native_labels": False,
            "gpu": True,
            "require_gpu": False,
            "preprocess_factor": 1.0,
            "preprocess_voxel_res_zyx": [1.0, 1.0, 1.0],
            "preprocess_do_bg_correction": True,
            "preprocess_bg_ds": 16,
            "preprocess_bg_sigma": 5.0,
            "preprocess_normalize_min": 2.0,
            "preprocess_normalize_max": 99.8,
            "cellpose_model_name": "cyto",
            "cellpose_channels": "grayscale",
            "cellpose_hist_norm": False,
            "cellpose_ksize": 15,
            "cellpose_use_auto_diameter": False,
            "cellpose_best_diameter": None,
            "cellpose_diameter_range": [10.0, 120.0, 2.5],
            "cellpose_use_edge": True,
            "cellpose_model_invert": False,
            "cellpose_test_slice": None,
            "aggregation_gradient_decay": 0.0,
            "aggregation_n_iter": 200,
            "aggregation_momenta": 0.98,
            "aggregation_prob_threshold": None,
            "aggregation_threshold_n_levels": 3,
            "aggregation_threshold_level": 1,
            "aggregation_min_prob_threshold": 0.0,
            "aggregation_connected_min_area": 5,
            "aggregation_connected_smooth_sigma": 1.0,
            "aggregation_connected_thresh_factor": 0.0,
            "aggregation_binary_fill_holes": False,
            "aggregation_tile_mode": False,
            "aggregation_tile_shape_zyx": [128, 256, 256],
            "aggregation_tile_overlap_ratio": 0.25,
            "n_cpu": None,
            "postprocess_enable": True,
            "postprocess_min_size": 200,
            "postprocess_do_flow_remove": True,
            "postprocess_flow_threshold": 0.85,
            "postprocess_dtform_method": "cellpose_improve",
            "postprocess_edt_fixed_point_percentile": 0.01,
            "output_dtype": "uint32",
        }
        _PARAMETER_HINTS: Dict[str, str] = {
            "input_source": (
                "Input source controls which dataset this operation reads from. "
                "Use raw data, or choose an upstream operation output."
            ),
            "get_darkfield": (
                "Estimate and remove a darkfield component in addition to the "
                "flatfield profile. BaSiCPy warns this can be unstable on some datasets."
            ),
            "smoothness_flatfield": (
                "Controls the flatfield smoothness regularization passed to BaSiCPy."
            ),
            "working_size": (
                "Downsample size used by BaSiCPy during profile fitting. Larger "
                "values use more detail but increase runtime and memory."
            ),
            "fit_tile_shape_yx": (
                "Spatial tile size for chunk-wise flatfield profile fitting in (y, x). "
                "Larger tiles reduce seams but increase per-task memory."
            ),
            "blend_tiles": (
                "Blend overlapping tile estimates when stitching flatfield/darkfield "
                "profiles. Disable to use strict core-crop stitching."
            ),
            "is_timelapse": (
                "Apply BaSiCPy baseline subtraction for time-varying illumination."
            ),
            "psf_mode": (
                "Choose whether to use measured PSFs from files or generate synthetic "
                "PSFs from optical parameters."
            ),
            "measured_psf_paths": (
                "Measured PSF paths in channel order. Provide a single path to reuse "
                "for all channels, or comma-separated paths per channel."
            ),
            "measured_psf_xy_um": (
                "Measured PSF XY pixel size in microns. Single value or comma-separated "
                "values per channel."
            ),
            "measured_psf_z_um": (
                "Measured PSF Z pixel size in microns. Single value or comma-separated "
                "values per channel."
            ),
            "synthetic_excitation_nm": (
                "Synthetic illumination wavelength in nanometers. In light-sheet mode, "
                "this is the illumination laser wavelength."
            ),
            "synthetic_emission_nm": (
                "Synthetic PSF emission wavelength in nanometers. Single value or "
                "comma-separated values per channel."
            ),
            "synthetic_microscopy_mode": (
                "Select the synthetic PSF model: widefield, confocal, or light-sheet."
            ),
            "synthetic_illumination_wavelength_nm": (
                "Light-sheet illumination wavelength in nanometers. Single value or "
                "comma-separated values per channel."
            ),
            "synthetic_illumination_numerical_aperture": (
                "Light-sheet illumination NA values. Single value or comma-separated "
                "values per channel."
            ),
            "synthetic_detection_numerical_aperture": (
                "Synthetic detection objective NA. Single value or comma-separated "
                "values per channel."
            ),
            "synthetic_numerical_aperture": (
                "Synthetic PSF numerical aperture. Single value or comma-separated "
                "values per channel."
            ),
            "synthetic_preview": (
                "Render a synthetic PSF preview image using the current synthetic "
                "parameters and voxel sizes."
            ),
            "data_xy_pixel_um": (
                "Input data XY pixel size in microns. Defaults from store metadata "
                "when available."
            ),
            "data_z_pixel_um": (
                "Input data Z step in microns. Defaults from store metadata when available."
            ),
            "hann_window_bounds": (
                "OTF Hann window lower and upper bounds. These control frequency-domain "
                "windowing before Richardson-Lucy updates."
            ),
            "wiener_alpha": (
                "Wiener regularization alpha used by OWM deconvolution. Higher values "
                "increase regularization."
            ),
            "background": (
                "Background offset removed before deconvolution. Use values in source "
                "intensity units."
            ),
            "decon_iterations": (
                "Number of Richardson-Lucy iterations for each 3D volume."
            ),
            "large_file": (
                "Enable block/batch processing for large volumes in PyPetaKit5D."
            ),
            "block_size_zyx": (
                "Block size in (z, y, x) when large-file mode is enabled."
            ),
            "batch_size_zyx": (
                "Batch size in (z, y, x) when large-file mode is enabled."
            ),
            "shear_xy": (
                "XY shear angle in degrees where X changes as a function of Y. "
                "Converted internally to coefficient tan(angle): "
                "x' = x + tan(shear_xy_deg) * y."
            ),
            "shear_xz": (
                "XZ shear angle in degrees where X changes as a function of Z. "
                "Converted internally to coefficient tan(angle): "
                "x' = x + tan(shear_xz_deg) * z."
            ),
            "shear_yz": (
                "YZ shear angle in degrees where Y changes as a function of Z. "
                "Converted internally to coefficient tan(angle): "
                "y' = y + tan(shear_yz_deg) * z."
            ),
            "rotation_deg_xyz": (
                "Rotation angles in degrees around X, Y, and Z after shearing. "
                "Positive values follow right-hand-rule axes in physical space."
            ),
            "auto_rotate_from_shear": (
                "Automatically derive rotation angles from shear values (after "
                "degree-to-coefficient conversion) to "
                "reduce empty-space growth after deskew."
            ),
            "auto_estimate_shear_yz": (
                "Estimate shear_yz_deg from left/right X-extreme source slabs before "
                "running the full transform."
            ),
            "auto_estimate_extreme_fraction_x": (
                "Fraction of X used at each extreme slab for auto-estimation. "
                "Higher values improve robustness but increase pre-pass I/O."
            ),
            "auto_estimate_zy_stride": (
                "Downsampling stride applied to Z and Y during auto-estimation."
            ),
            "auto_estimate_signal_fraction": (
                "Foreground threshold fraction within local intensity range for "
                "envelope fitting during auto-estimation."
            ),
            "auto_estimate_t_index": (
                "Time index sampled for shear auto-estimation."
            ),
            "auto_estimate_p_index": (
                "Position index sampled for shear auto-estimation."
            ),
            "auto_estimate_c_index": (
                "Channel index sampled for shear auto-estimation."
            ),
            "interpolation": (
                "ANTsPy interpolation mode used during affine resampling."
            ),
            "fill_value": ("Fill value for non-overlapping output regions."),
            "output_dtype": (
                "Output dtype used for results/shear_transform/latest/data."
            ),
            "roi_padding_zyx": (
                "Extra source-ROI padding in (z, y, x) voxels per output tile."
            ),
            "cpus_per_task": (
                "CPU count passed to each PyPetaKit5D job. Keep low when many tasks run "
                "in parallel on process workers."
            ),
            "channel_index": "Channel index selects the channel axis index used for detection.",
            "bg_sigma": (
                "Background sigma controls Gaussian smoothing used in preprocess "
                "to estimate and remove low-frequency background."
            ),
            "fwhm_px": (
                "fwhm_px defines expected particle diameter in pixels and sets "
                "the LoG detection scale neighborhood."
            ),
            "sigma_min_factor": (
                "Lower scale factor for blob search relative to fwhm_px."
            ),
            "sigma_max_factor": (
                "Upper scale factor for blob search relative to fwhm_px."
            ),
            "threshold": (
                "Minimum normalized response required to keep a candidate particle."
            ),
            "overlap": (
                "Allowed overlap ratio between nearby blob candidates before "
                "suppression is applied."
            ),
            "exclude_border": (
                "Excludes detections this many pixels from image borders to "
                "reduce edge artifacts."
            ),
            "use_map_overlap": (
                "Enable chunk overlap margins to reduce seam artifacts across chunk edges."
            ),
            "overlap_zyx": (
                "Overlap margins in (z, y, x) voxels when chunk overlap is enabled."
            ),
            "eliminate_insignificant_particles": (
                "Post-filter candidates by statistical significance."
            ),
            "remove_close_particles": (
                "Removes particles that are too close to stronger neighbors."
            ),
            "min_distance_sigma": (
                "Minimum allowed separation in units of detected sigma for close-particle removal."
            ),
            "usegment3d_channel": (
                "Select one or more source channels for uSegment3D segmentation."
            ),
            "usegment3d_views": (
                "Comma-separated list of orthogonal views to evaluate (for example "
                "xy,xz,yz)."
            ),
            "usegment3d_resolution_level": (
                "Input pyramid level used for segmentation (0 = full resolution). "
                "Higher levels reduce memory/runtime."
            ),
            "usegment3d_output_reference_space": (
                "Choose whether final labels are stored at level 0 (original "
                "resolution) or native selected input level."
            ),
            "usegment3d_save_native_labels": (
                "When segmenting from a downsampled level and writing level-0 labels, "
                "also save native-level labels to results/usegment3d/latest/data_native."
            ),
            "usegment3d_gpu": ("Enable GPU-accelerated components when available."),
            "usegment3d_require_gpu": (
                "Fail this operation when no compatible GPU is available."
            ),
            "usegment3d_preprocess_min": (
                "Lower intensity bound used during uSegment3D preprocessing normalization."
            ),
            "usegment3d_preprocess_max": (
                "Upper intensity bound used during uSegment3D preprocessing normalization."
            ),
            "usegment3d_preprocess_bg_correction": (
                "Enable low-frequency background correction before segmentation."
            ),
            "usegment3d_preprocess_factor": (
                "Intensity scaling factor applied after preprocess normalization."
            ),
            "usegment3d_cellpose_model": (
                "Cellpose model identifier used for intermediate mask inference."
            ),
            "usegment3d_cellpose_channels": (
                "Cellpose channel interpretation mode ('grayscale' or 'color')."
            ),
            "usegment3d_cellpose_diameter": (
                "Cellpose object diameter in pixels. Use 0 to allow automatic sizing."
            ),
            "usegment3d_cellpose_edge": (
                "Enable Cellpose edge handling configuration for border-touching objects."
            ),
            "usegment3d_aggregation_decay": (
                "Temporal/view aggregation decay factor used when fusing uSegment3D scores."
            ),
            "usegment3d_aggregation_iterations": (
                "Number of aggregation refinement iterations for uSegment3D fusion."
            ),
            "usegment3d_aggregation_momentum": (
                "Momentum term applied during iterative aggregation updates."
            ),
            "usegment3d_tile_mode": (
                "Tile execution mode for segmentation. Use auto/manual to process by "
                "tiles, or none for full-volume processing."
            ),
            "usegment3d_tile_shape": (
                "Tile shape in (z, y, x) voxels when tile mode is enabled."
            ),
            "usegment3d_tile_overlap": (
                "Tile overlap in (z, y, x) voxels when tile mode is enabled."
            ),
            "usegment3d_postprocess": (
                "Enable uSegment3D postprocessing cleanup filters."
            ),
            "usegment3d_postprocess_min_size": (
                "Minimum connected-component size retained during postprocessing."
            ),
            "usegment3d_postprocess_flow_threshold": (
                "Cellpose flow-confidence threshold used in postprocessing."
            ),
            "usegment3d_postprocess_dtform": (
                "Distance-transform formulation used in final postprocessing."
            ),
            "execution_order": (
                "Execution order controls when this operation runs relative to other selected operations."
            ),
            "position_index": (
                "Position index selects which multiposition volume is rendered in napari. "
                "Single-position datasets should remain at 0."
            ),
            "show_all_positions": (
                "Render every position in the acquisition using stage transforms "
                "from multi_positions.yml."
            ),
            "use_multiscale": (
                "When enabled, napari loads pyramid levels as a multiscale image "
                "for faster navigation across zoom levels."
            ),
            "overlay_particle_detections": (
                "Overlay particle detections as a napari points layer when "
                "particle detections already exist in the store or are selected "
                "to run before visualization."
            ),
            "require_gpu_rendering": (
                "Fail visualization launch when napari reports a software OpenGL "
                "renderer (for example llvmpipe/swiftshader) or when GPU "
                "rendering cannot be confirmed."
            ),
            "volume_layers": (
                "Configure image/labels volume overlays for napari. "
                "Each row can select source component, type, channels, display "
                "settings, and multiscale policy (inherit/require/auto_build/off)."
            ),
            "keyframe_layer_overrides": (
                "Optional per-layer keyframe overrides used in the manifest. "
                "Each row can set layer visibility, LUT/colormap name, rendering "
                "style, and an annotation label for movie legends."
            ),
            "mip_position_mode": (
                "Choose whether each projection file contains all positions for a "
                "(time, channel) pair, or writes separate files per position."
            ),
            "mip_export_format": (
                "Choose export file format. OME-TIFF outputs are written as uint16 and "
                "include physical pixel-size calibration metadata; Zarr exports keep "
                "projection dtype."
            ),
            "mip_output_directory": (
                "Optional export directory path. Leave empty to write under an "
                "auto-generated sibling directory next to the analysis store."
            ),
        }

        def __init__(self, initial: WorkflowConfig) -> None:
            """Initialize analysis-selection window.

            Parameters
            ----------
            initial : WorkflowConfig
                Workflow values from setup step.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__()
            self.setWindowTitle("ClearEx Analysis")
            self.setMinimumWidth(1280)
            self.setMinimumHeight(760)

            self._base_config = initial
            self._dask_backend_config: DaskBackendConfig = initial.dask_backend
            self.result_config: Optional[WorkflowConfig] = None
            self._status_label: Optional[QLabel] = None
            self._dask_backend_summary_label: Optional[QLabel] = None
            self._dask_backend_button: Optional[QPushButton] = None
            self._dask_dashboard_button: Optional[QPushButton] = None
            self._parameter_help_default = (
                "Hover over a parameter to see a detailed explanation."
            )

            self._operation_defaults = default_analysis_operation_parameters()
            self._flatfield_defaults = dict(
                self._operation_defaults.get("flatfield", {})
            )
            self._decon_defaults = dict(
                self._operation_defaults.get("deconvolution", {})
            )
            self._shear_defaults = dict(
                self._operation_defaults.get("shear_transform", {})
            )
            self._particle_defaults = dict(
                self._operation_defaults.get("particle_detection", {})
            )
            self._usegment3d_defaults = dict(
                self._operation_defaults.get(
                    "usegment3d",
                    self._DEFAULT_USEGMENT3D_PARAMETERS,
                )
            )
            self._operation_defaults["usegment3d"] = dict(self._usegment3d_defaults)
            self._visualization_defaults = dict(
                self._operation_defaults.get("visualization", {})
            )
            self._visualization_volume_layers: list[Dict[str, Any]] = (
                self._normalize_visualization_volume_layers(
                    self._visualization_defaults.get("volume_layers", [])
                )
            )
            self._visualization_keyframe_layer_overrides: list[Dict[str, Any]] = (
                self._normalize_visualization_layer_overrides(
                    self._visualization_defaults.get("keyframe_layer_overrides", [])
                )
            )
            self._mip_export_defaults = dict(
                self._operation_defaults.get("mip_export", {})
            )

            self._operation_checkboxes: Dict[str, QCheckBox] = {}
            self._operation_order_spins: Dict[str, QSpinBox] = {}
            self._operation_config_buttons: Dict[str, QPushButton] = {}
            self._operation_input_combos: Dict[str, QComboBox] = {}
            self._operation_force_rerun_checkboxes: Dict[str, QCheckBox] = {}
            self._operation_history_labels: Dict[str, QLabel] = {}
            self._operation_history_cache: Dict[str, Dict[str, Any]] = {}
            self._operation_panel_indices: Dict[str, int] = {}
            self._parameter_help_map: Dict[QObject, str] = {}
            self._active_config_operation: Optional[str] = None

            self._operation_panel_stack: Optional[QStackedWidget] = None
            self._operation_panel_scroll: Optional[QScrollArea] = None
            self._parameter_help_label: Optional[QLabel] = None
            self._store_label: Optional[QLabel] = None
            self._decon_measured_section: Optional[QFrame] = None
            self._decon_synthetic_section: Optional[QFrame] = None
            self._decon_light_sheet_section: Optional[QFrame] = None
            self._visualization_volume_layers_button: Optional[QPushButton] = None
            self._visualization_volume_layers_summary_label: Optional[QLabel] = None
            self._visualization_layer_table_button: Optional[QPushButton] = None
            self._visualization_layer_table_summary_label: Optional[QLabel] = None

            self._build_ui()
            self._apply_theme()
            self._hydrate(initial)

        def _build_ui(self) -> None:
            """Build analysis window controls and connect actions.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widgets are created and connected in-place.
            """
            root = QVBoxLayout(self)
            apply_window_root_spacing(root)

            header = QFrame()
            header.setObjectName("headerCard")
            header_layout = QVBoxLayout(header)
            apply_stack_spacing(header_layout)
            header_image = _create_scaled_branding_label(
                filename=_GUI_HEADER_IMAGE,
                max_width=920,
                max_height=160,
                object_name="brandHeaderImage",
            )
            if header_image is not None:
                header_layout.addWidget(header_image)
            root.addWidget(header)

            content_row = QHBoxLayout()
            apply_stack_spacing(content_row)
            root.addLayout(content_row, 1)

            left_column = QVBoxLayout()
            apply_stack_spacing(left_column)

            title = QLabel("Select Analysis Methods")
            title.setObjectName("title")
            left_column.addWidget(title)

            subtitle = QLabel(
                "Enable routines, configure each operation, and define execution sequence."
            )
            subtitle.setObjectName("subtitle")
            subtitle.setWordWrap(True)
            left_column.addWidget(subtitle)

            store_row = QHBoxLayout()
            apply_row_spacing(store_row)
            label = QLabel("Data store:")
            label.setObjectName("metadataFieldLabel")
            self._store_label = QLabel("n/a")
            self._store_label.setObjectName("metadataFieldValue")
            self._store_label.setWordWrap(True)
            self._store_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            store_row.addWidget(label)
            store_row.addWidget(self._store_label, 1)
            left_column.addLayout(store_row)

            analysis_tabs = QTabWidget()
            analysis_tabs.setObjectName("analysisTabs")
            analysis_tabs.setMinimumWidth(500)
            for tab_name, tab_operations in self._OPERATION_TABS:
                tab_widget = QWidget()
                tab_layout = QVBoxLayout(tab_widget)
                apply_stack_spacing(tab_layout)

                operations_group = QGroupBox(f"{tab_name} Operations")
                operations_layout = QVBoxLayout(operations_group)
                apply_stack_spacing(operations_layout)
                for operation_name in tab_operations:
                    self._build_operation_selection_row(
                        parent_layout=operations_layout,
                        operation_name=operation_name,
                    )
                operations_layout.addStretch(1)
                tab_layout.addWidget(operations_group, 1)
                analysis_tabs.addTab(tab_widget, tab_name)

            analysis_hint = QLabel(
                "Use Configure to edit one operation at a time. "
                "Only selected operations can be configured."
            )
            analysis_hint.setObjectName("statusLabel")
            analysis_hint.setWordWrap(True)
            left_column.addWidget(analysis_tabs, 1)
            left_column.addWidget(analysis_hint)
            content_row.addLayout(left_column, 1)

            parameters_group = QGroupBox("Operation Parameters")
            parameters_layout = QVBoxLayout(parameters_group)
            apply_stack_spacing(parameters_layout)

            self._operation_panel_stack = QStackedWidget()
            self._operation_panel_stack.addWidget(self._build_no_selection_panel())
            for operation_name in self._OPERATION_KEYS:
                panel = self._build_operation_panel(operation_name)
                panel_index = self._operation_panel_stack.addWidget(panel)
                self._operation_panel_indices[operation_name] = panel_index
            self._operation_panel_scroll = QScrollArea()
            self._operation_panel_scroll.setWidgetResizable(True)
            self._operation_panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._operation_panel_scroll.setWidget(self._operation_panel_stack)
            parameters_layout.addWidget(self._operation_panel_scroll, 1)

            help_card = QFrame()
            help_card.setObjectName("helpCard")
            help_layout = QVBoxLayout(help_card)
            apply_help_stack_spacing(help_layout)
            help_title = QLabel("Parameter Help")
            help_title.setObjectName("helpTitle")
            help_layout.addWidget(help_title)
            self._parameter_help_label = QLabel(self._parameter_help_default)
            self._parameter_help_label.setObjectName("helpBody")
            self._parameter_help_label.setWordWrap(True)
            help_layout.addWidget(self._parameter_help_label)
            parameters_layout.addWidget(help_card)

            content_row.addWidget(parameters_group, 1)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            status_stack = QVBoxLayout()
            apply_help_stack_spacing(status_stack)
            self._status_label = QLabel(
                "Configure selected analysis routines and click Run."
            )
            self._status_label.setObjectName("statusLabel")
            self._status_label.setWordWrap(True)
            status_stack.addWidget(self._status_label)
            self._dask_backend_summary_label = QLabel("Dask backend: n/a")
            self._dask_backend_summary_label.setObjectName("statusLabel")
            self._dask_backend_summary_label.setWordWrap(True)
            self._dask_backend_summary_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            status_stack.addWidget(self._dask_backend_summary_label)
            footer.addLayout(status_stack, 1)
            self._dask_backend_button = QPushButton("Edit Dask Backend")
            self._dask_dashboard_button = QPushButton("Open Dask Dashboard")
            self._cancel_button = QPushButton("Cancel")
            self._run_button = QPushButton("Run")
            self._run_button.setObjectName("runButton")
            footer.addWidget(self._dask_backend_button)
            footer.addWidget(self._dask_dashboard_button)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._run_button)
            root.addLayout(footer)

            self._dask_backend_button.clicked.connect(self._on_edit_dask_backend)
            self._dask_dashboard_button.clicked.connect(self._on_open_dask_dashboard)
            self._cancel_button.clicked.connect(self.reject)
            self._run_button.clicked.connect(self._on_run)
            self._decon_psf_mode_combo.currentIndexChanged.connect(
                self._on_deconvolution_psf_mode_changed
            )
            self._decon_synth_mode_combo.currentIndexChanged.connect(
                self._set_deconvolution_parameter_enabled_state
            )
            self._decon_preview_psf_button.clicked.connect(
                self._on_preview_synthetic_psf
            )
            self._decon_large_file_checkbox.toggled.connect(
                self._set_deconvolution_parameter_enabled_state
            )
            self._flatfield_use_overlap_checkbox.toggled.connect(
                self._set_flatfield_parameter_enabled_state
            )
            self._particle_use_overlap_checkbox.toggled.connect(
                self._set_particle_parameter_enabled_state
            )
            self._particle_remove_close_checkbox.toggled.connect(
                self._set_particle_parameter_enabled_state
            )
            self._usegment3d_require_gpu_checkbox.toggled.connect(
                self._set_usegment3d_parameter_enabled_state
            )
            self._usegment3d_resolution_level_spin.valueChanged.connect(
                self._set_usegment3d_parameter_enabled_state
            )
            self._usegment3d_output_reference_combo.currentIndexChanged.connect(
                self._set_usegment3d_parameter_enabled_state
            )
            self._usegment3d_tile_mode_combo.currentIndexChanged.connect(
                self._set_usegment3d_parameter_enabled_state
            )
            self._usegment3d_postprocess_checkbox.toggled.connect(
                self._set_usegment3d_parameter_enabled_state
            )
            self._visualization_show_all_positions_checkbox.toggled.connect(
                self._set_visualization_position_selector_state
            )

        def _build_operation_selection_row(
            self,
            *,
            parent_layout: QVBoxLayout,
            operation_name: str,
        ) -> None:
            """Create one operation-selection row and attach it to a parent layout.

            Parameters
            ----------
            parent_layout : QVBoxLayout
                Layout receiving the operation controls.
            operation_name : str
                Operation key rendered in the row.

            Returns
            -------
            None
                Widgets are created, connected, and stored in-place.

            Raises
            ------
            None
                All validation is handled internally.
            """
            row = QHBoxLayout()
            apply_compact_row_spacing(row)

            checkbox = QCheckBox(self._OPERATION_LABELS[operation_name])
            checkbox.setObjectName("operationCheckbox")
            row.addWidget(checkbox, 1)
            self._operation_checkboxes[operation_name] = checkbox

            order_label = QLabel("Order")
            order_label.setObjectName("statusLabel")
            row.addWidget(order_label)

            order_spin = QSpinBox()
            order_spin.setRange(1, len(self._OPERATION_KEYS))
            order_spin.setMinimumWidth(76)
            row.addWidget(order_spin)
            self._operation_order_spins[operation_name] = order_spin
            self._register_parameter_hint(
                order_spin, self._PARAMETER_HINTS["execution_order"]
            )

            configure_button = QPushButton("Configure")
            configure_button.setCheckable(True)
            configure_button.setObjectName("configureButton")
            row.addWidget(configure_button)
            self._operation_config_buttons[operation_name] = configure_button

            if operation_name in self._PROVENANCE_HISTORY_OPERATIONS:
                force_rerun_checkbox = QCheckBox("Force rerun")
                force_rerun_checkbox.setVisible(False)
                force_rerun_checkbox.setEnabled(False)
                force_rerun_checkbox.setToolTip(
                    "Run this routine even when a matching completed run exists "
                    "in provenance."
                )
                row.addWidget(force_rerun_checkbox)
                self._operation_force_rerun_checkboxes[operation_name] = (
                    force_rerun_checkbox
                )

            parent_layout.addLayout(row)

            if operation_name in self._PROVENANCE_HISTORY_OPERATIONS:
                history_label = QLabel("Provenance: no successful run recorded yet.")
                history_label.setObjectName("statusLabel")
                history_label.setWordWrap(True)
                history_label.setContentsMargins(24, 0, 0, 0)
                parent_layout.addWidget(history_label)
                self._operation_history_labels[operation_name] = history_label

            checkbox.toggled.connect(self._on_operation_selection_changed)
            order_spin.valueChanged.connect(self._on_operation_order_changed)
            configure_button.clicked.connect(
                lambda _checked=False, op=operation_name: (
                    self._show_operation_configuration(op)
                )
            )

        def _build_no_selection_panel(self) -> QWidget:
            """Create the default panel shown before Configure is selected.

            Parameters
            ----------
            None

            Returns
            -------
            QWidget
                Placeholder panel widget.
            """
            widget = QWidget()
            layout = QVBoxLayout(widget)
            apply_stack_spacing(layout)
            text = QLabel(
                "Select an analysis routine on the left, then click Configure to open "
                "its operation parameters."
            )
            text.setWordWrap(True)
            text.setObjectName("parameterHint")
            layout.addWidget(text)
            layout.addStretch(1)
            return widget

        def _build_operation_panel(self, operation_name: str) -> QWidget:
            """Build the parameter panel for one operation.

            Parameters
            ----------
            operation_name : str
                Analysis operation key.

            Returns
            -------
            QWidget
                Parameter panel for the operation.
            """
            panel = QWidget()
            layout = QVBoxLayout(panel)
            apply_stack_spacing(layout)

            profile = QLabel(
                f"{self._OPERATION_LABELS[operation_name]} Configuration\n"
                f"Output component: {self._operation_output_component(operation_name)}"
            )
            profile.setObjectName("parameterHint")
            profile.setWordWrap(True)
            layout.addWidget(profile)

            form = QFormLayout()
            form.setLabelAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
            apply_form_spacing(form)

            input_combo = QComboBox()
            form.addRow("Input source", input_combo)
            self._operation_input_combos[operation_name] = input_combo
            self._register_parameter_hint(
                input_combo, self._PARAMETER_HINTS["input_source"]
            )

            if operation_name == "deconvolution":
                self._build_deconvolution_parameter_rows(form)
            elif operation_name == "flatfield":
                self._build_flatfield_parameter_rows(form)
            elif operation_name == "shear_transform":
                self._build_shear_parameter_rows(form)
            elif operation_name == "particle_detection":
                self._build_particle_parameter_rows(form)
            elif operation_name == "usegment3d":
                self._build_usegment3d_parameter_rows(form)
            elif operation_name == "visualization":
                self._build_visualization_parameter_rows(form)
            elif operation_name == "mip_export":
                self._build_mip_export_parameter_rows(form)
            else:
                stub = QLabel(
                    "Advanced parameters for this operation are not exposed yet. "
                    "Execution order and input source are fully configurable."
                )
                stub.setWordWrap(True)
                stub.setObjectName("statusLabel")
                form.addRow("Details", stub)

            layout.addLayout(form)
            layout.addStretch(1)
            return panel

        def _build_parameter_section_card(
            self, title: str
        ) -> tuple[QFrame, QFormLayout]:
            """Create a themed section card containing a form layout.

            Parameters
            ----------
            title : str
                Section title text.

            Returns
            -------
            tuple[QFrame, QFormLayout]
                Created section card and inner form layout.
            """
            section = QFrame()
            section.setObjectName("operationSection")
            section_layout = QVBoxLayout(section)
            apply_stack_spacing(section_layout)

            title_label = QLabel(str(title))
            title_label.setObjectName("operationSectionTitle")
            section_layout.addWidget(title_label)

            section_form = QFormLayout()
            section_form.setLabelAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            section_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
            apply_form_spacing(section_form)
            section_layout.addLayout(section_form)
            return section, section_form

        def _build_flatfield_parameter_rows(self, form: QFormLayout) -> None:
            """Add flatfield-correction parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Form receiving flatfield widgets.

            Returns
            -------
            None
                Widgets are created and stored on ``self``.
            """
            self._flatfield_darkfield_checkbox = QCheckBox("Estimate darkfield profile")
            form.addRow("", self._flatfield_darkfield_checkbox)
            self._register_parameter_hint(
                self._flatfield_darkfield_checkbox,
                self._PARAMETER_HINTS["get_darkfield"],
            )

            self._flatfield_smoothness_spin = QDoubleSpinBox()
            self._flatfield_smoothness_spin.setRange(0.0, 100.0)
            self._flatfield_smoothness_spin.setDecimals(3)
            self._flatfield_smoothness_spin.setSingleStep(0.1)
            form.addRow("Smoothness", self._flatfield_smoothness_spin)
            self._register_parameter_hint(
                self._flatfield_smoothness_spin,
                self._PARAMETER_HINTS["smoothness_flatfield"],
            )

            self._flatfield_working_size_spin = QSpinBox()
            self._flatfield_working_size_spin.setRange(1, 4096)
            form.addRow("Working size", self._flatfield_working_size_spin)
            self._register_parameter_hint(
                self._flatfield_working_size_spin,
                self._PARAMETER_HINTS["working_size"],
            )

            self._flatfield_tile_y_spin = QSpinBox()
            self._flatfield_tile_y_spin.setRange(1, 16384)
            form.addRow("Fit tile Y", self._flatfield_tile_y_spin)
            self._register_parameter_hint(
                self._flatfield_tile_y_spin,
                self._PARAMETER_HINTS["fit_tile_shape_yx"],
            )

            self._flatfield_tile_x_spin = QSpinBox()
            self._flatfield_tile_x_spin.setRange(1, 16384)
            form.addRow("Fit tile X", self._flatfield_tile_x_spin)
            self._register_parameter_hint(
                self._flatfield_tile_x_spin,
                self._PARAMETER_HINTS["fit_tile_shape_yx"],
            )

            self._flatfield_timelapse_checkbox = QCheckBox(
                "Apply timelapse baseline correction"
            )
            form.addRow("", self._flatfield_timelapse_checkbox)
            self._register_parameter_hint(
                self._flatfield_timelapse_checkbox,
                self._PARAMETER_HINTS["is_timelapse"],
            )

            self._flatfield_use_overlap_checkbox = QCheckBox("Use map_overlap margins")
            form.addRow("", self._flatfield_use_overlap_checkbox)
            self._register_parameter_hint(
                self._flatfield_use_overlap_checkbox,
                self._PARAMETER_HINTS["use_map_overlap"],
            )

            self._flatfield_blend_tiles_checkbox = QCheckBox("Blend Tiles")
            form.addRow("", self._flatfield_blend_tiles_checkbox)
            self._register_parameter_hint(
                self._flatfield_blend_tiles_checkbox,
                self._PARAMETER_HINTS["blend_tiles"],
            )

            self._flatfield_overlap_z_spin = QSpinBox()
            self._flatfield_overlap_z_spin.setRange(0, 4096)
            form.addRow("Overlap Z", self._flatfield_overlap_z_spin)
            self._register_parameter_hint(
                self._flatfield_overlap_z_spin,
                self._PARAMETER_HINTS["overlap_zyx"],
            )

            self._flatfield_overlap_y_spin = QSpinBox()
            self._flatfield_overlap_y_spin.setRange(0, 4096)
            form.addRow("Overlap Y", self._flatfield_overlap_y_spin)
            self._register_parameter_hint(
                self._flatfield_overlap_y_spin,
                self._PARAMETER_HINTS["overlap_zyx"],
            )

            self._flatfield_overlap_x_spin = QSpinBox()
            self._flatfield_overlap_x_spin.setRange(0, 4096)
            form.addRow("Overlap X", self._flatfield_overlap_x_spin)
            self._register_parameter_hint(
                self._flatfield_overlap_x_spin,
                self._PARAMETER_HINTS["overlap_zyx"],
            )

        def _build_deconvolution_parameter_rows(self, form: QFormLayout) -> None:
            """Add deconvolution parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving deconvolution controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            self._decon_psf_mode_combo = QComboBox()
            self._decon_psf_mode_combo.addItem("Measured PSF", "measured")
            self._decon_psf_mode_combo.addItem("Synthetic PSF", "synthetic")
            form.addRow("PSF mode", self._decon_psf_mode_combo)
            self._register_parameter_hint(
                self._decon_psf_mode_combo, self._PARAMETER_HINTS["psf_mode"]
            )

            self._decon_measured_section, measured_form = (
                self._build_parameter_section_card("Measured PSF Settings")
            )
            self._decon_measured_psf_paths_input = QLineEdit()
            self._decon_measured_psf_paths_input.setPlaceholderText(
                "/path/to/ch0_psf.tif,/path/to/ch1_psf.tif"
            )
            measured_form.addRow(
                "Measured PSF path(s)",
                self._decon_measured_psf_paths_input,
            )
            self._register_parameter_hint(
                self._decon_measured_psf_paths_input,
                self._PARAMETER_HINTS["measured_psf_paths"],
            )

            self._decon_measured_psf_xy_input = QLineEdit()
            self._decon_measured_psf_xy_input.setPlaceholderText("0.108,0.108")
            measured_form.addRow(
                "Measured PSF XY size (um)",
                self._decon_measured_psf_xy_input,
            )
            self._register_parameter_hint(
                self._decon_measured_psf_xy_input,
                self._PARAMETER_HINTS["measured_psf_xy_um"],
            )

            self._decon_measured_psf_z_input = QLineEdit()
            self._decon_measured_psf_z_input.setPlaceholderText("0.5,0.5")
            measured_form.addRow(
                "Measured PSF Z size (um)",
                self._decon_measured_psf_z_input,
            )
            self._register_parameter_hint(
                self._decon_measured_psf_z_input,
                self._PARAMETER_HINTS["measured_psf_z_um"],
            )
            form.addRow(self._decon_measured_section)

            self._decon_synthetic_section, synthetic_form = (
                self._build_parameter_section_card("Synthetic PSF Settings")
            )
            self._decon_synth_mode_combo = QComboBox()
            self._decon_synth_mode_combo.addItem("Widefield", "widefield")
            self._decon_synth_mode_combo.addItem("Confocal", "confocal")
            self._decon_synth_mode_combo.addItem("Light-Sheet", "light_sheet")
            synthetic_form.addRow("Synthetic microscopy", self._decon_synth_mode_combo)
            self._register_parameter_hint(
                self._decon_synth_mode_combo,
                self._PARAMETER_HINTS["synthetic_microscopy_mode"],
            )

            self._decon_synth_emission_input = QLineEdit()
            self._decon_synth_emission_input.setPlaceholderText("520,610")
            synthetic_form.addRow(
                "Detection emission (nm)",
                self._decon_synth_emission_input,
            )
            self._register_parameter_hint(
                self._decon_synth_emission_input,
                self._PARAMETER_HINTS["synthetic_emission_nm"],
            )

            self._decon_synth_na_input = QLineEdit()
            self._decon_synth_na_input.setPlaceholderText("0.7,0.7")
            synthetic_form.addRow("Detection NA", self._decon_synth_na_input)
            self._register_parameter_hint(
                self._decon_synth_na_input,
                self._PARAMETER_HINTS["synthetic_detection_numerical_aperture"],
            )

            self._decon_light_sheet_section, light_sheet_form = (
                self._build_parameter_section_card("Light-Sheet Optical Settings")
            )
            self._decon_synth_excitation_input = QLineEdit()
            self._decon_synth_excitation_input.setPlaceholderText("488,561")
            light_sheet_form.addRow(
                "Illumination wavelength (nm)",
                self._decon_synth_excitation_input,
            )
            self._register_parameter_hint(
                self._decon_synth_excitation_input,
                self._PARAMETER_HINTS["synthetic_illumination_wavelength_nm"],
            )

            self._decon_synth_illum_na_input = QLineEdit()
            self._decon_synth_illum_na_input.setPlaceholderText("0.2,0.2")
            light_sheet_form.addRow(
                "Illumination NA",
                self._decon_synth_illum_na_input,
            )
            self._register_parameter_hint(
                self._decon_synth_illum_na_input,
                self._PARAMETER_HINTS["synthetic_illumination_numerical_aperture"],
            )
            synthetic_form.addRow(self._decon_light_sheet_section)

            self._decon_preview_psf_button = QPushButton("Preview Synthetic PSF")
            synthetic_form.addRow("Synthetic preview", self._decon_preview_psf_button)
            self._register_parameter_hint(
                self._decon_preview_psf_button,
                self._PARAMETER_HINTS["synthetic_preview"],
            )
            form.addRow(self._decon_synthetic_section)

            self._decon_data_xy_spin = QDoubleSpinBox()
            self._decon_data_xy_spin.setDecimals(5)
            self._decon_data_xy_spin.setRange(0.0, 1_000_000.0)
            self._decon_data_xy_spin.setSingleStep(0.01)
            form.addRow("Data XY size (um)", self._decon_data_xy_spin)
            self._register_parameter_hint(
                self._decon_data_xy_spin, self._PARAMETER_HINTS["data_xy_pixel_um"]
            )

            self._decon_data_z_spin = QDoubleSpinBox()
            self._decon_data_z_spin.setDecimals(5)
            self._decon_data_z_spin.setRange(0.0, 1_000_000.0)
            self._decon_data_z_spin.setSingleStep(0.01)
            form.addRow("Data Z size (um)", self._decon_data_z_spin)
            self._register_parameter_hint(
                self._decon_data_z_spin, self._PARAMETER_HINTS["data_z_pixel_um"]
            )

            hann_row = QHBoxLayout()
            apply_compact_row_spacing(hann_row)
            self._decon_hann_low_spin = QDoubleSpinBox()
            self._decon_hann_low_spin.setDecimals(3)
            self._decon_hann_low_spin.setRange(0.001, 10.0)
            self._decon_hann_low_spin.setSingleStep(0.05)
            self._decon_hann_high_spin = QDoubleSpinBox()
            self._decon_hann_high_spin.setDecimals(3)
            self._decon_hann_high_spin.setRange(0.001, 10.0)
            self._decon_hann_high_spin.setSingleStep(0.05)
            hann_row.addWidget(QLabel("low"))
            hann_row.addWidget(self._decon_hann_low_spin)
            hann_row.addWidget(QLabel("high"))
            hann_row.addWidget(self._decon_hann_high_spin)
            hann_widget = QWidget()
            hann_widget.setLayout(hann_row)
            form.addRow("OTF Hann window", hann_widget)
            self._register_parameter_hint(
                self._decon_hann_low_spin, self._PARAMETER_HINTS["hann_window_bounds"]
            )
            self._register_parameter_hint(
                self._decon_hann_high_spin, self._PARAMETER_HINTS["hann_window_bounds"]
            )

            self._decon_wiener_spin = QDoubleSpinBox()
            self._decon_wiener_spin.setDecimals(6)
            self._decon_wiener_spin.setRange(0.0, 1000.0)
            self._decon_wiener_spin.setSingleStep(0.001)
            form.addRow("Wiener regularization", self._decon_wiener_spin)
            self._register_parameter_hint(
                self._decon_wiener_spin, self._PARAMETER_HINTS["wiener_alpha"]
            )

            self._decon_background_spin = QDoubleSpinBox()
            self._decon_background_spin.setDecimals(3)
            self._decon_background_spin.setRange(0.0, 1_000_000_000.0)
            self._decon_background_spin.setSingleStep(1.0)
            form.addRow("Background offset", self._decon_background_spin)
            self._register_parameter_hint(
                self._decon_background_spin, self._PARAMETER_HINTS["background"]
            )

            self._decon_iterations_spin = QSpinBox()
            self._decon_iterations_spin.setRange(1, 10_000)
            form.addRow("RL iterations", self._decon_iterations_spin)
            self._register_parameter_hint(
                self._decon_iterations_spin, self._PARAMETER_HINTS["decon_iterations"]
            )

            self._decon_large_file_checkbox = QCheckBox("Enable large-file mode")
            form.addRow("Large-file mode", self._decon_large_file_checkbox)
            self._register_parameter_hint(
                self._decon_large_file_checkbox, self._PARAMETER_HINTS["large_file"]
            )

            block_row = QHBoxLayout()
            apply_compact_row_spacing(block_row)
            self._decon_block_z_spin = QSpinBox()
            self._decon_block_z_spin.setRange(1, 1_000_000)
            self._decon_block_y_spin = QSpinBox()
            self._decon_block_y_spin.setRange(1, 1_000_000)
            self._decon_block_x_spin = QSpinBox()
            self._decon_block_x_spin.setRange(1, 1_000_000)
            block_row.addWidget(QLabel("z"))
            block_row.addWidget(self._decon_block_z_spin)
            block_row.addWidget(QLabel("y"))
            block_row.addWidget(self._decon_block_y_spin)
            block_row.addWidget(QLabel("x"))
            block_row.addWidget(self._decon_block_x_spin)
            block_widget = QWidget()
            block_widget.setLayout(block_row)
            form.addRow("Block size (z,y,x)", block_widget)
            self._register_parameter_hint(
                self._decon_block_z_spin, self._PARAMETER_HINTS["block_size_zyx"]
            )
            self._register_parameter_hint(
                self._decon_block_y_spin, self._PARAMETER_HINTS["block_size_zyx"]
            )
            self._register_parameter_hint(
                self._decon_block_x_spin, self._PARAMETER_HINTS["block_size_zyx"]
            )

            batch_row = QHBoxLayout()
            apply_compact_row_spacing(batch_row)
            self._decon_batch_z_spin = QSpinBox()
            self._decon_batch_z_spin.setRange(1, 1_000_000)
            self._decon_batch_y_spin = QSpinBox()
            self._decon_batch_y_spin.setRange(1, 1_000_000)
            self._decon_batch_x_spin = QSpinBox()
            self._decon_batch_x_spin.setRange(1, 1_000_000)
            batch_row.addWidget(QLabel("z"))
            batch_row.addWidget(self._decon_batch_z_spin)
            batch_row.addWidget(QLabel("y"))
            batch_row.addWidget(self._decon_batch_y_spin)
            batch_row.addWidget(QLabel("x"))
            batch_row.addWidget(self._decon_batch_x_spin)
            batch_widget = QWidget()
            batch_widget.setLayout(batch_row)
            form.addRow("Batch size (z,y,x)", batch_widget)
            self._register_parameter_hint(
                self._decon_batch_z_spin, self._PARAMETER_HINTS["batch_size_zyx"]
            )
            self._register_parameter_hint(
                self._decon_batch_y_spin, self._PARAMETER_HINTS["batch_size_zyx"]
            )
            self._register_parameter_hint(
                self._decon_batch_x_spin, self._PARAMETER_HINTS["batch_size_zyx"]
            )

            self._decon_cpus_spin = QSpinBox()
            self._decon_cpus_spin.setRange(1, 1024)
            form.addRow("CPUs per task", self._decon_cpus_spin)
            self._register_parameter_hint(
                self._decon_cpus_spin, self._PARAMETER_HINTS["cpus_per_task"]
            )

        def _build_shear_parameter_rows(self, form: QFormLayout) -> None:
            """Add shear-transform parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving shear-transform controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            self._shear_xy_spin = QDoubleSpinBox()
            self._shear_xy_spin.setDecimals(3)
            self._shear_xy_spin.setRange(-89.9, 89.9)
            self._shear_xy_spin.setSingleStep(0.1)
            form.addRow("shear_xy_deg", self._shear_xy_spin)
            self._register_parameter_hint(
                self._shear_xy_spin, self._PARAMETER_HINTS["shear_xy"]
            )

            self._shear_xz_spin = QDoubleSpinBox()
            self._shear_xz_spin.setDecimals(3)
            self._shear_xz_spin.setRange(-89.9, 89.9)
            self._shear_xz_spin.setSingleStep(0.1)
            form.addRow("shear_xz_deg", self._shear_xz_spin)
            self._register_parameter_hint(
                self._shear_xz_spin, self._PARAMETER_HINTS["shear_xz"]
            )

            self._shear_yz_spin = QDoubleSpinBox()
            self._shear_yz_spin.setDecimals(3)
            self._shear_yz_spin.setRange(-89.9, 89.9)
            self._shear_yz_spin.setSingleStep(0.1)
            form.addRow("shear_yz_deg", self._shear_yz_spin)
            self._register_parameter_hint(
                self._shear_yz_spin, self._PARAMETER_HINTS["shear_yz"]
            )

            self._shear_auto_rotate_checkbox = QCheckBox(
                "Derive rotation from shear angles"
            )
            form.addRow("Auto rotate", self._shear_auto_rotate_checkbox)
            self._register_parameter_hint(
                self._shear_auto_rotate_checkbox,
                self._PARAMETER_HINTS["auto_rotate_from_shear"],
            )

            self._shear_auto_estimate_checkbox = QCheckBox(
                "Estimate shear_yz from X extremes"
            )
            form.addRow("Auto estimate shear_yz", self._shear_auto_estimate_checkbox)
            self._register_parameter_hint(
                self._shear_auto_estimate_checkbox,
                self._PARAMETER_HINTS["auto_estimate_shear_yz"],
            )

            self._shear_auto_estimate_fraction_spin = QDoubleSpinBox()
            self._shear_auto_estimate_fraction_spin.setDecimals(4)
            self._shear_auto_estimate_fraction_spin.setRange(0.0001, 0.5)
            self._shear_auto_estimate_fraction_spin.setSingleStep(0.001)
            form.addRow(
                "auto_estimate_extreme_fraction_x",
                self._shear_auto_estimate_fraction_spin,
            )
            self._register_parameter_hint(
                self._shear_auto_estimate_fraction_spin,
                self._PARAMETER_HINTS["auto_estimate_extreme_fraction_x"],
            )

            self._shear_auto_estimate_stride_spin = QSpinBox()
            self._shear_auto_estimate_stride_spin.setRange(1, 64)
            self._shear_auto_estimate_stride_spin.setSingleStep(1)
            form.addRow("auto_estimate_zy_stride", self._shear_auto_estimate_stride_spin)
            self._register_parameter_hint(
                self._shear_auto_estimate_stride_spin,
                self._PARAMETER_HINTS["auto_estimate_zy_stride"],
            )

            self._shear_auto_estimate_signal_fraction_spin = QDoubleSpinBox()
            self._shear_auto_estimate_signal_fraction_spin.setDecimals(3)
            self._shear_auto_estimate_signal_fraction_spin.setRange(0.0, 1.0)
            self._shear_auto_estimate_signal_fraction_spin.setSingleStep(0.01)
            form.addRow(
                "auto_estimate_signal_fraction",
                self._shear_auto_estimate_signal_fraction_spin,
            )
            self._register_parameter_hint(
                self._shear_auto_estimate_signal_fraction_spin,
                self._PARAMETER_HINTS["auto_estimate_signal_fraction"],
            )

            self._shear_auto_estimate_t_spin = QSpinBox()
            self._shear_auto_estimate_t_spin.setRange(0, 9999)
            self._shear_auto_estimate_t_spin.setSingleStep(1)
            form.addRow("auto_estimate_t_index", self._shear_auto_estimate_t_spin)
            self._register_parameter_hint(
                self._shear_auto_estimate_t_spin,
                self._PARAMETER_HINTS["auto_estimate_t_index"],
            )

            self._shear_auto_estimate_p_spin = QSpinBox()
            self._shear_auto_estimate_p_spin.setRange(0, 9999)
            self._shear_auto_estimate_p_spin.setSingleStep(1)
            form.addRow("auto_estimate_p_index", self._shear_auto_estimate_p_spin)
            self._register_parameter_hint(
                self._shear_auto_estimate_p_spin,
                self._PARAMETER_HINTS["auto_estimate_p_index"],
            )

            self._shear_auto_estimate_c_spin = QSpinBox()
            self._shear_auto_estimate_c_spin.setRange(0, 9999)
            self._shear_auto_estimate_c_spin.setSingleStep(1)
            form.addRow("auto_estimate_c_index", self._shear_auto_estimate_c_spin)
            self._register_parameter_hint(
                self._shear_auto_estimate_c_spin,
                self._PARAMETER_HINTS["auto_estimate_c_index"],
            )

            self._shear_rotation_x_spin = QDoubleSpinBox()
            self._shear_rotation_x_spin.setDecimals(4)
            self._shear_rotation_x_spin.setRange(-360.0, 360.0)
            self._shear_rotation_x_spin.setSingleStep(0.5)
            form.addRow("rotation_deg_x", self._shear_rotation_x_spin)
            self._register_parameter_hint(
                self._shear_rotation_x_spin, self._PARAMETER_HINTS["rotation_deg_xyz"]
            )

            self._shear_rotation_y_spin = QDoubleSpinBox()
            self._shear_rotation_y_spin.setDecimals(4)
            self._shear_rotation_y_spin.setRange(-360.0, 360.0)
            self._shear_rotation_y_spin.setSingleStep(0.5)
            form.addRow("rotation_deg_y", self._shear_rotation_y_spin)
            self._register_parameter_hint(
                self._shear_rotation_y_spin, self._PARAMETER_HINTS["rotation_deg_xyz"]
            )

            self._shear_rotation_z_spin = QDoubleSpinBox()
            self._shear_rotation_z_spin.setDecimals(4)
            self._shear_rotation_z_spin.setRange(-360.0, 360.0)
            self._shear_rotation_z_spin.setSingleStep(0.5)
            form.addRow("rotation_deg_z", self._shear_rotation_z_spin)
            self._register_parameter_hint(
                self._shear_rotation_z_spin, self._PARAMETER_HINTS["rotation_deg_xyz"]
            )

            self._shear_interpolation_combo = QComboBox()
            self._shear_interpolation_combo.addItem("Linear", "linear")
            self._shear_interpolation_combo.addItem(
                "Nearest Neighbor", "nearestneighbor"
            )
            self._shear_interpolation_combo.addItem("B-spline", "bspline")
            form.addRow("interpolation", self._shear_interpolation_combo)
            self._register_parameter_hint(
                self._shear_interpolation_combo, self._PARAMETER_HINTS["interpolation"]
            )

            self._shear_fill_value_spin = QDoubleSpinBox()
            self._shear_fill_value_spin.setDecimals(4)
            self._shear_fill_value_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            self._shear_fill_value_spin.setSingleStep(1.0)
            form.addRow("fill_value", self._shear_fill_value_spin)
            self._register_parameter_hint(
                self._shear_fill_value_spin, self._PARAMETER_HINTS["fill_value"]
            )

            self._shear_output_dtype_combo = QComboBox()
            self._shear_output_dtype_combo.addItem("float32", "float32")
            self._shear_output_dtype_combo.addItem("float64", "float64")
            self._shear_output_dtype_combo.addItem("uint16", "uint16")
            self._shear_output_dtype_combo.addItem("uint8", "uint8")
            form.addRow("output_dtype", self._shear_output_dtype_combo)
            self._register_parameter_hint(
                self._shear_output_dtype_combo, self._PARAMETER_HINTS["output_dtype"]
            )

            padding_row = QHBoxLayout()
            apply_compact_row_spacing(padding_row)
            self._shear_padding_z_spin = QSpinBox()
            self._shear_padding_z_spin.setRange(0, 10_000)
            self._shear_padding_y_spin = QSpinBox()
            self._shear_padding_y_spin.setRange(0, 10_000)
            self._shear_padding_x_spin = QSpinBox()
            self._shear_padding_x_spin.setRange(0, 10_000)
            padding_row.addWidget(QLabel("z"))
            padding_row.addWidget(self._shear_padding_z_spin)
            padding_row.addWidget(QLabel("y"))
            padding_row.addWidget(self._shear_padding_y_spin)
            padding_row.addWidget(QLabel("x"))
            padding_row.addWidget(self._shear_padding_x_spin)
            padding_widget = QWidget()
            padding_widget.setLayout(padding_row)
            form.addRow("roi_padding_zyx", padding_widget)
            self._register_parameter_hint(
                self._shear_padding_z_spin, self._PARAMETER_HINTS["roi_padding_zyx"]
            )
            self._register_parameter_hint(
                self._shear_padding_y_spin, self._PARAMETER_HINTS["roi_padding_zyx"]
            )
            self._register_parameter_hint(
                self._shear_padding_x_spin, self._PARAMETER_HINTS["roi_padding_zyx"]
            )

        def _build_particle_parameter_rows(self, form: QFormLayout) -> None:
            """Add particle-detection parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving particle controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            self._particle_channel_spin = QSpinBox()
            self._particle_channel_spin.setMinimum(0)
            self._particle_channel_spin.setMaximum(0)
            form.addRow("Channel Index", self._particle_channel_spin)
            self._register_parameter_hint(
                self._particle_channel_spin, self._PARAMETER_HINTS["channel_index"]
            )

            self._particle_bg_sigma_spin = QDoubleSpinBox()
            self._particle_bg_sigma_spin.setDecimals(2)
            self._particle_bg_sigma_spin.setRange(0.0, 1_000_000.0)
            self._particle_bg_sigma_spin.setSingleStep(1.0)
            form.addRow("Preprocess bg_sigma", self._particle_bg_sigma_spin)
            self._register_parameter_hint(
                self._particle_bg_sigma_spin, self._PARAMETER_HINTS["bg_sigma"]
            )

            self._particle_fwhm_spin = QDoubleSpinBox()
            self._particle_fwhm_spin.setDecimals(3)
            self._particle_fwhm_spin.setRange(0.001, 1_000_000.0)
            self._particle_fwhm_spin.setSingleStep(0.25)
            form.addRow("fwhm_px", self._particle_fwhm_spin)
            self._register_parameter_hint(
                self._particle_fwhm_spin, self._PARAMETER_HINTS["fwhm_px"]
            )

            self._particle_sigma_min_spin = QDoubleSpinBox()
            self._particle_sigma_min_spin.setDecimals(3)
            self._particle_sigma_min_spin.setRange(0.001, 1000.0)
            self._particle_sigma_min_spin.setSingleStep(0.1)
            form.addRow("sigma_min_factor", self._particle_sigma_min_spin)
            self._register_parameter_hint(
                self._particle_sigma_min_spin, self._PARAMETER_HINTS["sigma_min_factor"]
            )

            self._particle_sigma_max_spin = QDoubleSpinBox()
            self._particle_sigma_max_spin.setDecimals(3)
            self._particle_sigma_max_spin.setRange(0.001, 1000.0)
            self._particle_sigma_max_spin.setSingleStep(0.1)
            form.addRow("sigma_max_factor", self._particle_sigma_max_spin)
            self._register_parameter_hint(
                self._particle_sigma_max_spin, self._PARAMETER_HINTS["sigma_max_factor"]
            )

            self._particle_threshold_spin = QDoubleSpinBox()
            self._particle_threshold_spin.setDecimals(4)
            self._particle_threshold_spin.setRange(0.0, 1_000_000.0)
            self._particle_threshold_spin.setSingleStep(0.01)
            form.addRow("threshold", self._particle_threshold_spin)
            self._register_parameter_hint(
                self._particle_threshold_spin, self._PARAMETER_HINTS["threshold"]
            )

            self._particle_overlap_spin = QDoubleSpinBox()
            self._particle_overlap_spin.setDecimals(3)
            self._particle_overlap_spin.setRange(0.0, 1.0)
            self._particle_overlap_spin.setSingleStep(0.05)
            form.addRow("overlap", self._particle_overlap_spin)
            self._register_parameter_hint(
                self._particle_overlap_spin, self._PARAMETER_HINTS["overlap"]
            )

            self._particle_exclude_border_spin = QSpinBox()
            self._particle_exclude_border_spin.setRange(0, 1_000_000)
            form.addRow("exclude_border", self._particle_exclude_border_spin)
            self._register_parameter_hint(
                self._particle_exclude_border_spin,
                self._PARAMETER_HINTS["exclude_border"],
            )

            self._particle_use_overlap_checkbox = QCheckBox("Use map_overlap margins")
            form.addRow("Chunk overlap", self._particle_use_overlap_checkbox)
            self._register_parameter_hint(
                self._particle_use_overlap_checkbox,
                self._PARAMETER_HINTS["use_map_overlap"],
            )

            overlap_row = QHBoxLayout()
            apply_compact_row_spacing(overlap_row)
            self._particle_overlap_z_spin = QSpinBox()
            self._particle_overlap_z_spin.setRange(0, 1_000_000)
            self._particle_overlap_y_spin = QSpinBox()
            self._particle_overlap_y_spin.setRange(0, 1_000_000)
            self._particle_overlap_x_spin = QSpinBox()
            self._particle_overlap_x_spin.setRange(0, 1_000_000)
            overlap_row.addWidget(QLabel("z"))
            overlap_row.addWidget(self._particle_overlap_z_spin)
            overlap_row.addWidget(QLabel("y"))
            overlap_row.addWidget(self._particle_overlap_y_spin)
            overlap_row.addWidget(QLabel("x"))
            overlap_row.addWidget(self._particle_overlap_x_spin)
            overlap_widget = QWidget()
            overlap_widget.setLayout(overlap_row)
            form.addRow("overlap_zyx", overlap_widget)
            self._register_parameter_hint(
                self._particle_overlap_z_spin, self._PARAMETER_HINTS["overlap_zyx"]
            )
            self._register_parameter_hint(
                self._particle_overlap_y_spin, self._PARAMETER_HINTS["overlap_zyx"]
            )
            self._register_parameter_hint(
                self._particle_overlap_x_spin, self._PARAMETER_HINTS["overlap_zyx"]
            )

            self._particle_eliminate_checkbox = QCheckBox(
                "Eliminate statistically insignificant particles"
            )
            form.addRow("Post-filter 1", self._particle_eliminate_checkbox)
            self._register_parameter_hint(
                self._particle_eliminate_checkbox,
                self._PARAMETER_HINTS["eliminate_insignificant_particles"],
            )

            self._particle_remove_close_checkbox = QCheckBox(
                "Remove particles that are too close"
            )
            form.addRow("Post-filter 2", self._particle_remove_close_checkbox)
            self._register_parameter_hint(
                self._particle_remove_close_checkbox,
                self._PARAMETER_HINTS["remove_close_particles"],
            )

            self._particle_min_distance_spin = QDoubleSpinBox()
            self._particle_min_distance_spin.setDecimals(3)
            self._particle_min_distance_spin.setRange(0.0, 1_000_000.0)
            self._particle_min_distance_spin.setSingleStep(0.5)
            form.addRow("min_distance_sigma", self._particle_min_distance_spin)
            self._register_parameter_hint(
                self._particle_min_distance_spin,
                self._PARAMETER_HINTS["min_distance_sigma"],
            )

        def _build_usegment3d_parameter_rows(self, form: QFormLayout) -> None:
            """Add uSegment3D parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving uSegment3D controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            runtime_section, runtime_form = self._build_parameter_section_card(
                "uSegment3D Runtime"
            )
            self._usegment3d_channel_scroll = QScrollArea()
            self._usegment3d_channel_scroll.setWidgetResizable(True)
            self._usegment3d_channel_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._usegment3d_channel_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._usegment3d_channel_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._usegment3d_channel_container = QWidget()
            self._usegment3d_channel_layout = QHBoxLayout()
            apply_compact_row_spacing(self._usegment3d_channel_layout)
            self._usegment3d_channel_layout.setContentsMargins(0, 0, 0, 0)
            self._usegment3d_channel_container.setLayout(
                self._usegment3d_channel_layout
            )
            self._usegment3d_channel_scroll.setWidget(
                self._usegment3d_channel_container
            )
            self._usegment3d_channel_checkboxes: list[QCheckBox] = []
            runtime_form.addRow("channels", self._usegment3d_channel_scroll)
            self._register_parameter_hint(
                self._usegment3d_channel_scroll,
                self._PARAMETER_HINTS["usegment3d_channel"],
            )
            self._rebuild_usegment3d_channel_checkboxes(
                channel_count=1,
                selected_channels=[0],
            )

            self._usegment3d_views_input = QLineEdit()
            self._usegment3d_views_input.setPlaceholderText("xy,xz,yz")
            runtime_form.addRow("views", self._usegment3d_views_input)
            self._register_parameter_hint(
                self._usegment3d_views_input,
                self._PARAMETER_HINTS["usegment3d_views"],
            )

            self._usegment3d_resolution_level_spin = QSpinBox()
            self._usegment3d_resolution_level_spin.setRange(0, 0)
            runtime_form.addRow(
                "resolution level",
                self._usegment3d_resolution_level_spin,
            )
            self._register_parameter_hint(
                self._usegment3d_resolution_level_spin,
                self._PARAMETER_HINTS["usegment3d_resolution_level"],
            )

            self._usegment3d_output_reference_combo = QComboBox()
            self._usegment3d_output_reference_combo.addItem(
                "Level 0 (original)",
                "level0",
            )
            self._usegment3d_output_reference_combo.addItem(
                "Native selected level",
                "native_level",
            )
            runtime_form.addRow(
                "output space",
                self._usegment3d_output_reference_combo,
            )
            self._register_parameter_hint(
                self._usegment3d_output_reference_combo,
                self._PARAMETER_HINTS["usegment3d_output_reference_space"],
            )

            self._usegment3d_save_native_labels_checkbox = QCheckBox(
                "Save native-level labels"
            )
            runtime_form.addRow(
                "native labels",
                self._usegment3d_save_native_labels_checkbox,
            )
            self._register_parameter_hint(
                self._usegment3d_save_native_labels_checkbox,
                self._PARAMETER_HINTS["usegment3d_save_native_labels"],
            )

            self._usegment3d_gpu_checkbox = QCheckBox("Use GPU")
            runtime_form.addRow("GPU", self._usegment3d_gpu_checkbox)
            self._register_parameter_hint(
                self._usegment3d_gpu_checkbox,
                self._PARAMETER_HINTS["usegment3d_gpu"],
            )

            self._usegment3d_require_gpu_checkbox = QCheckBox("Require GPU")
            runtime_form.addRow("Require GPU", self._usegment3d_require_gpu_checkbox)
            self._register_parameter_hint(
                self._usegment3d_require_gpu_checkbox,
                self._PARAMETER_HINTS["usegment3d_require_gpu"],
            )
            form.addRow(runtime_section)

            preprocess_section, preprocess_form = self._build_parameter_section_card(
                "Preprocess"
            )
            self._usegment3d_preprocess_min_spin = QDoubleSpinBox()
            self._usegment3d_preprocess_min_spin.setDecimals(4)
            self._usegment3d_preprocess_min_spin.setRange(
                -1_000_000_000.0, 1_000_000_000.0
            )
            self._usegment3d_preprocess_min_spin.setSingleStep(0.1)
            preprocess_form.addRow("min", self._usegment3d_preprocess_min_spin)
            self._register_parameter_hint(
                self._usegment3d_preprocess_min_spin,
                self._PARAMETER_HINTS["usegment3d_preprocess_min"],
            )

            self._usegment3d_preprocess_max_spin = QDoubleSpinBox()
            self._usegment3d_preprocess_max_spin.setDecimals(4)
            self._usegment3d_preprocess_max_spin.setRange(
                -1_000_000_000.0, 1_000_000_000.0
            )
            self._usegment3d_preprocess_max_spin.setSingleStep(0.1)
            preprocess_form.addRow("max", self._usegment3d_preprocess_max_spin)
            self._register_parameter_hint(
                self._usegment3d_preprocess_max_spin,
                self._PARAMETER_HINTS["usegment3d_preprocess_max"],
            )

            self._usegment3d_preprocess_bg_correction_checkbox = QCheckBox(
                "Enable background correction"
            )
            preprocess_form.addRow(
                "bg correction",
                self._usegment3d_preprocess_bg_correction_checkbox,
            )
            self._register_parameter_hint(
                self._usegment3d_preprocess_bg_correction_checkbox,
                self._PARAMETER_HINTS["usegment3d_preprocess_bg_correction"],
            )

            self._usegment3d_preprocess_factor_spin = QDoubleSpinBox()
            self._usegment3d_preprocess_factor_spin.setDecimals(4)
            self._usegment3d_preprocess_factor_spin.setRange(0.0001, 1_000_000.0)
            self._usegment3d_preprocess_factor_spin.setSingleStep(0.1)
            preprocess_form.addRow("factor", self._usegment3d_preprocess_factor_spin)
            self._register_parameter_hint(
                self._usegment3d_preprocess_factor_spin,
                self._PARAMETER_HINTS["usegment3d_preprocess_factor"],
            )
            form.addRow(preprocess_section)

            cellpose_section, cellpose_form = self._build_parameter_section_card(
                "Cellpose"
            )
            self._usegment3d_cellpose_model_combo = QComboBox()
            self._usegment3d_cellpose_model_combo.addItem("cyto3", "cyto3")
            self._usegment3d_cellpose_model_combo.addItem("cyto2", "cyto2")
            self._usegment3d_cellpose_model_combo.addItem("nuclei", "nuclei")
            self._usegment3d_cellpose_model_combo.addItem("cyto", "cyto")
            self._usegment3d_cellpose_model_combo.addItem("custom", "custom")
            cellpose_form.addRow("model", self._usegment3d_cellpose_model_combo)
            self._register_parameter_hint(
                self._usegment3d_cellpose_model_combo,
                self._PARAMETER_HINTS["usegment3d_cellpose_model"],
            )

            self._usegment3d_cellpose_channels_input = QComboBox()
            self._usegment3d_cellpose_channels_input.addItem("Grayscale", "grayscale")
            self._usegment3d_cellpose_channels_input.addItem("Color", "color")
            cellpose_form.addRow("channels", self._usegment3d_cellpose_channels_input)
            self._register_parameter_hint(
                self._usegment3d_cellpose_channels_input,
                self._PARAMETER_HINTS["usegment3d_cellpose_channels"],
            )

            self._usegment3d_cellpose_diameter_spin = QDoubleSpinBox()
            self._usegment3d_cellpose_diameter_spin.setDecimals(3)
            self._usegment3d_cellpose_diameter_spin.setRange(0.0, 1_000_000.0)
            self._usegment3d_cellpose_diameter_spin.setSingleStep(1.0)
            cellpose_form.addRow("diameter", self._usegment3d_cellpose_diameter_spin)
            self._register_parameter_hint(
                self._usegment3d_cellpose_diameter_spin,
                self._PARAMETER_HINTS["usegment3d_cellpose_diameter"],
            )

            self._usegment3d_cellpose_edge_checkbox = QCheckBox("Enable edge handling")
            cellpose_form.addRow("edge", self._usegment3d_cellpose_edge_checkbox)
            self._register_parameter_hint(
                self._usegment3d_cellpose_edge_checkbox,
                self._PARAMETER_HINTS["usegment3d_cellpose_edge"],
            )
            form.addRow(cellpose_section)

            aggregation_section, aggregation_form = self._build_parameter_section_card(
                "Aggregation"
            )
            self._usegment3d_aggregation_decay_spin = QDoubleSpinBox()
            self._usegment3d_aggregation_decay_spin.setDecimals(4)
            self._usegment3d_aggregation_decay_spin.setRange(0.0, 1.0)
            self._usegment3d_aggregation_decay_spin.setSingleStep(0.01)
            aggregation_form.addRow("decay", self._usegment3d_aggregation_decay_spin)
            self._register_parameter_hint(
                self._usegment3d_aggregation_decay_spin,
                self._PARAMETER_HINTS["usegment3d_aggregation_decay"],
            )

            self._usegment3d_aggregation_iterations_spin = QSpinBox()
            self._usegment3d_aggregation_iterations_spin.setRange(1, 10_000)
            aggregation_form.addRow(
                "iterations",
                self._usegment3d_aggregation_iterations_spin,
            )
            self._register_parameter_hint(
                self._usegment3d_aggregation_iterations_spin,
                self._PARAMETER_HINTS["usegment3d_aggregation_iterations"],
            )

            self._usegment3d_aggregation_momentum_spin = QDoubleSpinBox()
            self._usegment3d_aggregation_momentum_spin.setDecimals(4)
            self._usegment3d_aggregation_momentum_spin.setRange(0.0, 1.0)
            self._usegment3d_aggregation_momentum_spin.setSingleStep(0.01)
            aggregation_form.addRow(
                "momentum",
                self._usegment3d_aggregation_momentum_spin,
            )
            self._register_parameter_hint(
                self._usegment3d_aggregation_momentum_spin,
                self._PARAMETER_HINTS["usegment3d_aggregation_momentum"],
            )
            form.addRow(aggregation_section)

            tile_section, tile_form = self._build_parameter_section_card("Tiling")
            self._usegment3d_tile_mode_combo = QComboBox()
            self._usegment3d_tile_mode_combo.addItem("Auto", "auto")
            self._usegment3d_tile_mode_combo.addItem("Manual", "manual")
            self._usegment3d_tile_mode_combo.addItem("None", "none")
            tile_form.addRow("mode", self._usegment3d_tile_mode_combo)
            self._register_parameter_hint(
                self._usegment3d_tile_mode_combo,
                self._PARAMETER_HINTS["usegment3d_tile_mode"],
            )

            tile_shape_row = QHBoxLayout()
            apply_compact_row_spacing(tile_shape_row)
            self._usegment3d_tile_z_spin = QSpinBox()
            self._usegment3d_tile_z_spin.setRange(1, 1_000_000)
            self._usegment3d_tile_y_spin = QSpinBox()
            self._usegment3d_tile_y_spin.setRange(1, 1_000_000)
            self._usegment3d_tile_x_spin = QSpinBox()
            self._usegment3d_tile_x_spin.setRange(1, 1_000_000)
            tile_shape_row.addWidget(QLabel("z"))
            tile_shape_row.addWidget(self._usegment3d_tile_z_spin)
            tile_shape_row.addWidget(QLabel("y"))
            tile_shape_row.addWidget(self._usegment3d_tile_y_spin)
            tile_shape_row.addWidget(QLabel("x"))
            tile_shape_row.addWidget(self._usegment3d_tile_x_spin)
            tile_shape_widget = QWidget()
            tile_shape_widget.setLayout(tile_shape_row)
            tile_form.addRow("shape", tile_shape_widget)
            self._register_parameter_hint(
                self._usegment3d_tile_z_spin,
                self._PARAMETER_HINTS["usegment3d_tile_shape"],
            )
            self._register_parameter_hint(
                self._usegment3d_tile_y_spin,
                self._PARAMETER_HINTS["usegment3d_tile_shape"],
            )
            self._register_parameter_hint(
                self._usegment3d_tile_x_spin,
                self._PARAMETER_HINTS["usegment3d_tile_shape"],
            )

            tile_overlap_row = QHBoxLayout()
            apply_compact_row_spacing(tile_overlap_row)
            self._usegment3d_tile_overlap_z_spin = QSpinBox()
            self._usegment3d_tile_overlap_z_spin.setRange(0, 1_000_000)
            self._usegment3d_tile_overlap_y_spin = QSpinBox()
            self._usegment3d_tile_overlap_y_spin.setRange(0, 1_000_000)
            self._usegment3d_tile_overlap_x_spin = QSpinBox()
            self._usegment3d_tile_overlap_x_spin.setRange(0, 1_000_000)
            tile_overlap_row.addWidget(QLabel("z"))
            tile_overlap_row.addWidget(self._usegment3d_tile_overlap_z_spin)
            tile_overlap_row.addWidget(QLabel("y"))
            tile_overlap_row.addWidget(self._usegment3d_tile_overlap_y_spin)
            tile_overlap_row.addWidget(QLabel("x"))
            tile_overlap_row.addWidget(self._usegment3d_tile_overlap_x_spin)
            tile_overlap_widget = QWidget()
            tile_overlap_widget.setLayout(tile_overlap_row)
            tile_form.addRow("overlap", tile_overlap_widget)
            self._register_parameter_hint(
                self._usegment3d_tile_overlap_z_spin,
                self._PARAMETER_HINTS["usegment3d_tile_overlap"],
            )
            self._register_parameter_hint(
                self._usegment3d_tile_overlap_y_spin,
                self._PARAMETER_HINTS["usegment3d_tile_overlap"],
            )
            self._register_parameter_hint(
                self._usegment3d_tile_overlap_x_spin,
                self._PARAMETER_HINTS["usegment3d_tile_overlap"],
            )
            form.addRow(tile_section)

            postprocess_section, postprocess_form = self._build_parameter_section_card(
                "Postprocess"
            )
            self._usegment3d_postprocess_checkbox = QCheckBox("Enable postprocessing")
            postprocess_form.addRow("enabled", self._usegment3d_postprocess_checkbox)
            self._register_parameter_hint(
                self._usegment3d_postprocess_checkbox,
                self._PARAMETER_HINTS["usegment3d_postprocess"],
            )

            self._usegment3d_postprocess_min_size_spin = QSpinBox()
            self._usegment3d_postprocess_min_size_spin.setRange(0, 1_000_000_000)
            postprocess_form.addRow(
                "min size",
                self._usegment3d_postprocess_min_size_spin,
            )
            self._register_parameter_hint(
                self._usegment3d_postprocess_min_size_spin,
                self._PARAMETER_HINTS["usegment3d_postprocess_min_size"],
            )

            self._usegment3d_postprocess_flow_threshold_spin = QDoubleSpinBox()
            self._usegment3d_postprocess_flow_threshold_spin.setDecimals(4)
            self._usegment3d_postprocess_flow_threshold_spin.setRange(0.0001, 10.0)
            self._usegment3d_postprocess_flow_threshold_spin.setSingleStep(0.05)
            postprocess_form.addRow(
                "flow threshold",
                self._usegment3d_postprocess_flow_threshold_spin,
            )
            self._register_parameter_hint(
                self._usegment3d_postprocess_flow_threshold_spin,
                self._PARAMETER_HINTS["usegment3d_postprocess_flow_threshold"],
            )

            self._usegment3d_postprocess_dtform_combo = QComboBox()
            self._usegment3d_postprocess_dtform_combo.addItem(
                "Cellpose Improve",
                "cellpose_improve",
            )
            self._usegment3d_postprocess_dtform_combo.addItem("EDT", "edt")
            self._usegment3d_postprocess_dtform_combo.addItem("FMM", "fmm")
            self._usegment3d_postprocess_dtform_combo.addItem(
                "FMM Skeleton", "fmm_skel"
            )
            self._usegment3d_postprocess_dtform_combo.addItem(
                "Cellpose Skeleton",
                "cellpose_skel",
            )
            self._usegment3d_postprocess_dtform_combo.addItem("None (Disable)", "none")
            postprocess_form.addRow(
                "dtform",
                self._usegment3d_postprocess_dtform_combo,
            )
            self._register_parameter_hint(
                self._usegment3d_postprocess_dtform_combo,
                self._PARAMETER_HINTS["usegment3d_postprocess_dtform"],
            )
            form.addRow(postprocess_section)

        def _rebuild_usegment3d_channel_checkboxes(
            self,
            *,
            channel_count: int,
            selected_channels: Optional[Sequence[int]] = None,
        ) -> None:
            """Rebuild uSegment3D channel checkboxes for the detected channel count.

            Parameters
            ----------
            channel_count : int
                Number of source channels available in the selected dataset.
            selected_channels : sequence[int], optional
                Channel indices to mark selected.

            Returns
            -------
            None
                Checkbox widgets are replaced in-place.
            """
            count = max(1, int(channel_count))
            selected_values = (
                [] if selected_channels is None else list(selected_channels)
            )
            selected_set: set[int] = set()
            for value in selected_values:
                parsed = max(0, min(count - 1, int(value)))
                selected_set.add(parsed)
            if not selected_set:
                selected_set = {0}

            while self._usegment3d_channel_layout.count():
                item = self._usegment3d_channel_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._usegment3d_channel_checkboxes = []

            for channel_index in range(count):
                checkbox = QCheckBox(f"C{channel_index}")
                checkbox.setChecked(channel_index in selected_set)
                self._usegment3d_channel_layout.addWidget(checkbox)
                self._register_parameter_hint(
                    checkbox,
                    self._PARAMETER_HINTS["usegment3d_channel"],
                )
                self._usegment3d_channel_checkboxes.append(checkbox)
            self._usegment3d_channel_layout.addStretch(1)

        def _selected_usegment3d_channel_indices(self) -> list[int]:
            """Return selected uSegment3D channel indices.

            Parameters
            ----------
            None

            Returns
            -------
            list[int]
                Selected channel indices in display order. Ensures at least one
                selected channel when checkboxes are available.
            """
            selected = [
                int(index)
                for index, checkbox in enumerate(self._usegment3d_channel_checkboxes)
                if checkbox.isChecked()
            ]
            if not selected:
                if self._usegment3d_channel_checkboxes:
                    self._usegment3d_channel_checkboxes[0].setChecked(True)
                    return [0]
                return [0]
            return selected

        def _build_visualization_parameter_rows(self, form: QFormLayout) -> None:
            """Add visualization parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving visualization controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            self._visualization_show_all_positions_checkbox = QCheckBox(
                "Show all positions"
            )
            form.addRow(
                "Multi-position",
                self._visualization_show_all_positions_checkbox,
            )
            self._register_parameter_hint(
                self._visualization_show_all_positions_checkbox,
                self._PARAMETER_HINTS["show_all_positions"],
            )

            self._visualization_position_spin = QSpinBox()
            self._visualization_position_spin.setRange(0, 0)
            self._visualization_position_label = QLabel("Position index")
            form.addRow(
                self._visualization_position_label,
                self._visualization_position_spin,
            )
            self._register_parameter_hint(
                self._visualization_position_spin,
                self._PARAMETER_HINTS["position_index"],
            )

            self._visualization_multiscale_checkbox = QCheckBox(
                "Load as multiscale pyramid"
            )
            form.addRow("Multiscale", self._visualization_multiscale_checkbox)
            self._register_parameter_hint(
                self._visualization_multiscale_checkbox,
                self._PARAMETER_HINTS["use_multiscale"],
            )

            self._visualization_require_gpu_checkbox = QCheckBox(
                "Require GPU OpenGL renderer"
            )
            form.addRow("GPU rendering", self._visualization_require_gpu_checkbox)
            self._register_parameter_hint(
                self._visualization_require_gpu_checkbox,
                self._PARAMETER_HINTS["require_gpu_rendering"],
            )

            self._visualization_volume_layers_button = QPushButton("Volume Layers...")
            self._visualization_volume_layers_button.clicked.connect(
                self._open_visualization_volume_layers_dialog
            )
            form.addRow("Volume layers", self._visualization_volume_layers_button)
            self._register_parameter_hint(
                self._visualization_volume_layers_button,
                self._PARAMETER_HINTS["volume_layers"],
            )

            self._visualization_volume_layers_summary_label = QLabel()
            self._visualization_volume_layers_summary_label.setWordWrap(True)
            form.addRow("", self._visualization_volume_layers_summary_label)
            self._refresh_visualization_volume_layers_summary()

            self._visualization_overlay_points_checkbox = QCheckBox(
                "Overlay particle detections"
            )
            form.addRow("Particle overlay", self._visualization_overlay_points_checkbox)
            self._register_parameter_hint(
                self._visualization_overlay_points_checkbox,
                self._PARAMETER_HINTS["overlay_particle_detections"],
            )

            self._visualization_layer_table_button = QPushButton("Layer/View Table...")
            self._visualization_layer_table_button.clicked.connect(
                self._open_visualization_layer_table_dialog
            )
            form.addRow("Keyframe table", self._visualization_layer_table_button)
            self._register_parameter_hint(
                self._visualization_layer_table_button,
                self._PARAMETER_HINTS["keyframe_layer_overrides"],
            )

            self._visualization_layer_table_summary_label = QLabel()
            self._visualization_layer_table_summary_label.setWordWrap(True)
            form.addRow("", self._visualization_layer_table_summary_label)
            self._refresh_visualization_layer_override_summary()

        def _coerce_optional_bool(self, value: Any) -> Optional[bool]:
            """Coerce a value into ``True``/``False``/``None``.

            Parameters
            ----------
            value : Any
                Candidate boolean-like value.

            Returns
            -------
            bool, optional
                Parsed boolean or ``None`` when unspecified.

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

        def _coerce_optional_unit_interval_float(self, value: Any) -> Optional[float]:
            """Coerce optional opacity values into finite ``[0, 1]`` floats.

            Parameters
            ----------
            value : Any
                Candidate opacity value.

            Returns
            -------
            float, optional
                Parsed opacity value or ``None`` when unspecified.
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
            if not math.isfinite(parsed):
                return None
            return float(min(1.0, max(0.0, parsed)))

        def _normalize_visualization_channels(self, value: Any) -> list[int]:
            """Normalize one layer's channel selection.

            Parameters
            ----------
            value : Any
                Candidate channel-selection value.

            Returns
            -------
            list[int]
                Unique sorted non-negative channel indices. Empty means all.
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

        def _default_visualization_multiscale_policy(self) -> str:
            """Return default per-layer multiscale policy from current controls."""
            checkbox = getattr(self, "_visualization_multiscale_checkbox", None)
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                return "inherit"
            return "off"

        def _default_visualization_volume_layers(self) -> list[Dict[str, Any]]:
            """Return default volume-layer rows derived from current input source."""
            combo = self._operation_input_combos.get("visualization")
            component = "data"
            if combo is not None:
                data = combo.currentData()
                component = (
                    str(data).strip() if data is not None else "data"
                ) or "data"
            return [
                {
                    "component": component,
                    "name": "",
                    "layer_type": "image",
                    "channels": [],
                    "visible": None,
                    "opacity": None,
                    "blending": "",
                    "colormap": "",
                    "rendering": "",
                    "multiscale_policy": self._default_visualization_multiscale_policy(),
                }
            ]

        def _normalize_visualization_volume_layers(
            self,
            value: Any,
        ) -> list[Dict[str, Any]]:
            """Normalize visualization volume-layer table rows.

            Parameters
            ----------
            value : Any
                Candidate list-like row payload.

            Returns
            -------
            list[dict[str, Any]]
                Normalized rows for visualization runtime configuration.
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
                layer_type = (
                    str(raw_row.get("layer_type", "image")).strip().lower() or "image"
                )
                if layer_type not in {"image", "labels"}:
                    layer_type = "image"
                multiscale_policy = (
                    str(raw_row.get("multiscale_policy", "inherit")).strip().lower()
                    or "inherit"
                )
                if multiscale_policy not in {"inherit", "require", "auto_build", "off"}:
                    multiscale_policy = "inherit"
                rows.append(
                    {
                        "component": component,
                        "name": str(raw_row.get("name", "")).strip(),
                        "layer_type": layer_type,
                        "channels": self._normalize_visualization_channels(
                            raw_row.get("channels")
                        ),
                        "visible": self._coerce_optional_bool(raw_row.get("visible")),
                        "opacity": self._coerce_optional_unit_interval_float(
                            raw_row.get("opacity")
                        ),
                        "blending": str(raw_row.get("blending", "")).strip().lower(),
                        "colormap": str(
                            raw_row.get("colormap", raw_row.get("lut", ""))
                        ).strip(),
                        "rendering": str(raw_row.get("rendering", "")).strip().lower(),
                        "multiscale_policy": multiscale_policy,
                    }
                )
            return rows

        def _sync_visualization_input_source_from_volume_layers(self) -> None:
            """Align visualization input-source combo with first volume layer."""
            if not self._visualization_volume_layers:
                return
            combo = self._operation_input_combos.get("visualization")
            if combo is None:
                return
            component = (
                str(
                    self._visualization_volume_layers[0].get("component", "data")
                ).strip()
                or "data"
            )
            combo.blockSignals(True)
            combo_index = combo.findData(component)
            if combo_index < 0:
                combo.addItem(f"Custom component ({component})", component)
                combo_index = combo.count() - 1
            combo.setCurrentIndex(combo_index)
            combo.blockSignals(False)

        def _refresh_visualization_volume_layers_summary(self) -> None:
            """Refresh summary text for visualization volume-layer rows."""
            label = self._visualization_volume_layers_summary_label
            if label is None:
                return
            rows = list(self._visualization_volume_layers)
            if not rows:
                label.setText(
                    "No volume layers configured; primary input source will be used."
                )
                return
            type_counts: Dict[str, int] = {"image": 0, "labels": 0}
            for row in rows:
                layer_type = str(row.get("layer_type", "image")).strip().lower()
                if layer_type not in type_counts:
                    layer_type = "image"
                type_counts[layer_type] += 1
            label.setText(
                f"{len(rows)} layer(s): {type_counts['image']} image, "
                f"{type_counts['labels']} labels."
            )
            self._sync_visualization_input_source_from_volume_layers()

        def _open_visualization_volume_layers_dialog(self) -> None:
            """Open popup editor for visualization volume-layer rows."""
            dialog = QDialog(self)
            dialog.setWindowTitle("Visualization Volume Layers")
            dialog.setMinimumWidth(1160)
            dialog.setMinimumHeight(460)
            dialog.setModal(True)
            dialog.setStyleSheet(_popup_dialog_stylesheet())

            root = QVBoxLayout(dialog)
            apply_popup_root_spacing(root)

            helper = QLabel(
                "Configure image/labels overlays for napari. "
                "Use channels as comma-separated indices (leave blank for all)."
            )
            helper.setWordWrap(True)
            root.addWidget(helper)

            table = QTableWidget(dialog)
            table.setColumnCount(10)
            table.setHorizontalHeaderLabels(
                (
                    "Name",
                    "Component",
                    "Type",
                    "Channels",
                    "Visible",
                    "Opacity",
                    "Blending",
                    "Colormap",
                    "Rendering",
                    "Multiscale",
                )
            )
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(9, QHeaderView.ResizeMode.ResizeToContents)
            table.verticalHeader().setVisible(False)
            root.addWidget(table, stretch=1)

            def _make_type_combo(current_value: str) -> QComboBox:
                combo = QComboBox(table)
                combo.addItem("Image", "image")
                combo.addItem("Labels", "labels")
                index = combo.findData(str(current_value).strip().lower() or "image")
                combo.setCurrentIndex(index if index >= 0 else 0)
                return combo

            def _make_visible_combo(current_value: Optional[bool]) -> QComboBox:
                combo = QComboBox(table)
                combo.addItem("Auto", None)
                combo.addItem("True", True)
                combo.addItem("False", False)
                index = combo.findData(current_value)
                combo.setCurrentIndex(index if index >= 0 else 0)
                return combo

            def _make_multiscale_combo(current_value: str) -> QComboBox:
                combo = QComboBox(table)
                combo.addItem("Inherit", "inherit")
                combo.addItem("Require", "require")
                combo.addItem("Auto build", "auto_build")
                combo.addItem("Off", "off")
                index = combo.findData(str(current_value).strip().lower() or "inherit")
                combo.setCurrentIndex(index if index >= 0 else 0)
                return combo

            def _append_row(row_data: Optional[Mapping[str, Any]] = None) -> None:
                values = dict(row_data) if isinstance(row_data, Mapping) else {}
                row_index = table.rowCount()
                table.insertRow(row_index)

                channels = self._normalize_visualization_channels(
                    values.get("channels")
                )
                channels_text = ", ".join(str(value) for value in channels)
                table.setItem(
                    row_index,
                    0,
                    QTableWidgetItem(str(values.get("name", "")).strip()),
                )
                table.setItem(
                    row_index,
                    1,
                    QTableWidgetItem(
                        str(
                            values.get("component", values.get("source_component", ""))
                        ).strip()
                    ),
                )
                table.setCellWidget(
                    row_index,
                    2,
                    _make_type_combo(str(values.get("layer_type", "image"))),
                )
                table.setItem(row_index, 3, QTableWidgetItem(channels_text))
                table.setCellWidget(
                    row_index,
                    4,
                    _make_visible_combo(
                        self._coerce_optional_bool(values.get("visible"))
                    ),
                )
                opacity_value = self._coerce_optional_unit_interval_float(
                    values.get("opacity")
                )
                table.setItem(
                    row_index,
                    5,
                    QTableWidgetItem(
                        "" if opacity_value is None else f"{float(opacity_value):.3f}"
                    ),
                )
                table.setItem(
                    row_index,
                    6,
                    QTableWidgetItem(str(values.get("blending", "")).strip()),
                )
                table.setItem(
                    row_index,
                    7,
                    QTableWidgetItem(str(values.get("colormap", "")).strip()),
                )
                table.setItem(
                    row_index,
                    8,
                    QTableWidgetItem(str(values.get("rendering", "")).strip()),
                )
                table.setCellWidget(
                    row_index,
                    9,
                    _make_multiscale_combo(
                        str(values.get("multiscale_policy", "inherit"))
                    ),
                )

            rows_seed = (
                list(self._visualization_volume_layers)
                if self._visualization_volume_layers
                else self._default_visualization_volume_layers()
            )
            for row in rows_seed:
                _append_row(row)
            if table.rowCount() == 0:
                _append_row()

            controls = QHBoxLayout()
            add_row_button = QPushButton("Add Row", dialog)
            remove_row_button = QPushButton("Remove Selected", dialog)
            controls.addWidget(add_row_button)
            controls.addWidget(remove_row_button)
            controls.addStretch(1)
            root.addLayout(controls)

            add_row_button.clicked.connect(lambda: _append_row())

            def _remove_selected_rows() -> None:
                selected_indexes = table.selectionModel().selectedRows()
                if not selected_indexes:
                    return
                for index in sorted(
                    (idx.row() for idx in selected_indexes),
                    reverse=True,
                ):
                    table.removeRow(int(index))
                if table.rowCount() == 0:
                    _append_row()

            remove_row_button.clicked.connect(_remove_selected_rows)

            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel,
                Qt.Orientation.Horizontal,
                dialog,
            )
            root.addWidget(button_box)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            rows: list[Dict[str, Any]] = []
            for row_index in range(table.rowCount()):
                name_item = table.item(row_index, 0)
                component_item = table.item(row_index, 1)
                channels_item = table.item(row_index, 3)
                opacity_item = table.item(row_index, 5)
                blending_item = table.item(row_index, 6)
                colormap_item = table.item(row_index, 7)
                rendering_item = table.item(row_index, 8)
                type_widget = table.cellWidget(row_index, 2)
                visible_widget = table.cellWidget(row_index, 4)
                multiscale_widget = table.cellWidget(row_index, 9)

                component = (
                    str(component_item.text()).strip()
                    if isinstance(component_item, QTableWidgetItem)
                    else ""
                )
                if not component:
                    continue
                row_payload: Dict[str, Any] = {
                    "name": (
                        str(name_item.text()).strip()
                        if isinstance(name_item, QTableWidgetItem)
                        else ""
                    ),
                    "component": component,
                    "layer_type": (
                        str(type_widget.currentData() or "image").strip().lower()
                        if isinstance(type_widget, QComboBox)
                        else "image"
                    ),
                    "channels": self._normalize_visualization_channels(
                        channels_item.text()
                        if isinstance(channels_item, QTableWidgetItem)
                        else ""
                    ),
                    "visible": (
                        self._coerce_optional_bool(visible_widget.currentData())
                        if isinstance(visible_widget, QComboBox)
                        else None
                    ),
                    "opacity": self._coerce_optional_unit_interval_float(
                        opacity_item.text()
                        if isinstance(opacity_item, QTableWidgetItem)
                        else ""
                    ),
                    "blending": (
                        str(blending_item.text()).strip().lower()
                        if isinstance(blending_item, QTableWidgetItem)
                        else ""
                    ),
                    "colormap": (
                        str(colormap_item.text()).strip()
                        if isinstance(colormap_item, QTableWidgetItem)
                        else ""
                    ),
                    "rendering": (
                        str(rendering_item.text()).strip().lower()
                        if isinstance(rendering_item, QTableWidgetItem)
                        else ""
                    ),
                    "multiscale_policy": (
                        str(multiscale_widget.currentData() or "inherit")
                        .strip()
                        .lower()
                        if isinstance(multiscale_widget, QComboBox)
                        else "inherit"
                    ),
                }
                rows.append(row_payload)

            self._visualization_volume_layers = (
                self._normalize_visualization_volume_layers(rows)
            )
            if not self._visualization_volume_layers:
                self._visualization_volume_layers = (
                    self._default_visualization_volume_layers()
                )
            self._refresh_visualization_volume_layers_summary()

        def _normalize_visualization_layer_overrides(
            self,
            value: Any,
        ) -> list[Dict[str, Any]]:
            """Normalize visualization keyframe table rows.

            Parameters
            ----------
            value : Any
                Candidate list-like value.

            Returns
            -------
            list[dict[str, Any]]
                Normalized rows containing ``layer_name``, ``visible``,
                ``colormap``, ``rendering``, and ``annotation``.

            Raises
            ------
            None
                Invalid rows are skipped.
            """
            if not isinstance(value, (tuple, list)):
                return []

            rows: list[Dict[str, Any]] = []
            for raw_row in value:
                if not isinstance(raw_row, Mapping):
                    continue
                layer_name = str(
                    raw_row.get("layer_name", raw_row.get("layer", ""))
                ).strip()
                visible = self._coerce_optional_bool(raw_row.get("visible"))
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

        def _refresh_visualization_layer_override_summary(self) -> None:
            """Refresh summary text for visualization keyframe table rows.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary label text is updated in-place.
            """
            label = self._visualization_layer_table_summary_label
            if label is None:
                return
            rows = list(self._visualization_keyframe_layer_overrides)
            if not rows:
                label.setText("No keyframe layer overrides configured.")
                return
            annotation_count = sum(
                1 for row in rows if str(row.get("annotation", "")).strip()
            )
            label.setText(
                f"{len(rows)} override row(s), {annotation_count} annotation(s)."
            )

        def _open_visualization_layer_table_dialog(self) -> None:
            """Open popup editor for visualization keyframe layer overrides.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Updates in-memory override rows when accepted.
            """
            dialog = QDialog(self)
            dialog.setWindowTitle("Visualization Keyframe Table")
            dialog.setMinimumWidth(900)
            dialog.setMinimumHeight(420)
            dialog.setModal(True)
            dialog.setStyleSheet(_popup_dialog_stylesheet())

            root = QVBoxLayout(dialog)
            apply_popup_root_spacing(root)

            helper = QLabel(
                "Optional overrides recorded in keyframe manifests. "
                "Use layer names as they appear in napari."
            )
            helper.setWordWrap(True)
            root.addWidget(helper)

            table = QTableWidget(dialog)
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(
                ("Layer", "Visible", "LUT/Colormap", "Rendering", "Annotation")
            )
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            table.verticalHeader().setVisible(False)
            root.addWidget(table, stretch=1)

            def _make_visible_combo(current_value: Optional[bool]) -> QComboBox:
                combo = QComboBox(table)
                combo.addItem("Auto", None)
                combo.addItem("True", True)
                combo.addItem("False", False)
                index = combo.findData(current_value)
                if index < 0:
                    index = 0
                combo.setCurrentIndex(index)
                return combo

            def _append_row(row_data: Optional[Mapping[str, Any]] = None) -> None:
                values = dict(row_data) if isinstance(row_data, Mapping) else {}
                row_index = table.rowCount()
                table.insertRow(row_index)

                layer_name = str(
                    values.get("layer_name", values.get("layer", ""))
                ).strip()
                colormap = str(values.get("colormap", values.get("lut", ""))).strip()
                rendering = str(values.get("rendering", "")).strip()
                annotation = str(values.get("annotation", "")).strip()
                visible = self._coerce_optional_bool(values.get("visible"))

                table.setItem(row_index, 0, QTableWidgetItem(layer_name))
                table.setCellWidget(row_index, 1, _make_visible_combo(visible))
                table.setItem(row_index, 2, QTableWidgetItem(colormap))
                table.setItem(row_index, 3, QTableWidgetItem(rendering))
                table.setItem(row_index, 4, QTableWidgetItem(annotation))

            for row in self._visualization_keyframe_layer_overrides:
                _append_row(row)
            if table.rowCount() == 0:
                _append_row()

            controls = QHBoxLayout()
            add_row_button = QPushButton("Add Row", dialog)
            remove_row_button = QPushButton("Remove Selected", dialog)
            controls.addWidget(add_row_button)
            controls.addWidget(remove_row_button)
            controls.addStretch(1)
            root.addLayout(controls)

            add_row_button.clicked.connect(lambda: _append_row())

            def _remove_selected_rows() -> None:
                selected_indexes = table.selectionModel().selectedRows()
                if not selected_indexes:
                    return
                for index in sorted(
                    (idx.row() for idx in selected_indexes),
                    reverse=True,
                ):
                    table.removeRow(int(index))
                if table.rowCount() == 0:
                    _append_row()

            remove_row_button.clicked.connect(_remove_selected_rows)

            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel,
                Qt.Orientation.Horizontal,
                dialog,
            )
            root.addWidget(button_box)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            rows: list[Dict[str, Any]] = []
            for row_index in range(table.rowCount()):
                layer_item = table.item(row_index, 0)
                colormap_item = table.item(row_index, 2)
                rendering_item = table.item(row_index, 3)
                annotation_item = table.item(row_index, 4)
                visible_widget = table.cellWidget(row_index, 1)
                visible_value: Optional[bool] = None
                if isinstance(visible_widget, QComboBox):
                    visible_value = self._coerce_optional_bool(
                        visible_widget.currentData()
                    )

                row_payload: Dict[str, Any] = {
                    "layer_name": (
                        str(layer_item.text()).strip()
                        if isinstance(layer_item, QTableWidgetItem)
                        else ""
                    ),
                    "visible": visible_value,
                    "colormap": (
                        str(colormap_item.text()).strip()
                        if isinstance(colormap_item, QTableWidgetItem)
                        else ""
                    ),
                    "rendering": (
                        str(rendering_item.text()).strip()
                        if isinstance(rendering_item, QTableWidgetItem)
                        else ""
                    ),
                    "annotation": (
                        str(annotation_item.text()).strip()
                        if isinstance(annotation_item, QTableWidgetItem)
                        else ""
                    ),
                }
                if not any(
                    (
                        row_payload["layer_name"],
                        row_payload["colormap"],
                        row_payload["rendering"],
                        row_payload["annotation"],
                        row_payload["visible"] is not None,
                    )
                ):
                    continue
                rows.append(row_payload)

            self._visualization_keyframe_layer_overrides = (
                self._normalize_visualization_layer_overrides(rows)
            )
            self._refresh_visualization_layer_override_summary()

        def _build_mip_export_parameter_rows(self, form: QFormLayout) -> None:
            """Add MIP-export parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving MIP-export controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            self._mip_position_mode_combo = QComboBox()
            self._mip_position_mode_combo.addItem(
                "Multi-position stack per time/channel",
                "multi_position",
            )
            self._mip_position_mode_combo.addItem(
                "Per-position files",
                "per_position",
            )
            form.addRow("Position mode", self._mip_position_mode_combo)
            self._register_parameter_hint(
                self._mip_position_mode_combo,
                self._PARAMETER_HINTS["mip_position_mode"],
            )

            self._mip_export_format_combo = QComboBox()
            self._mip_export_format_combo.addItem(
                "OME-TIFF (uint16 + calibration)",
                "ome-tiff",
            )
            self._mip_export_format_combo.addItem("Zarr", "zarr")
            form.addRow("Export format", self._mip_export_format_combo)
            self._register_parameter_hint(
                self._mip_export_format_combo,
                self._PARAMETER_HINTS["mip_export_format"],
            )

            self._mip_output_directory_input = QLineEdit()
            self._mip_output_directory_input.setPlaceholderText("Auto")
            form.addRow("Output directory", self._mip_output_directory_input)
            self._register_parameter_hint(
                self._mip_output_directory_input,
                self._PARAMETER_HINTS["mip_output_directory"],
            )

        def _register_parameter_hint(self, widget: QWidget, message: str) -> None:
            """Register hover/focus help text for a widget.

            Parameters
            ----------
            widget : QWidget
                Input widget to register.
            message : str
                Help text shown in the parameter help panel.

            Returns
            -------
            None
                Hint mapping is stored in-place.
            """
            widget.setToolTip(message)
            widget.installEventFilter(self)
            self._parameter_help_map[widget] = str(message)

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:
            """Handle hover/focus transitions for parameter-help updates.

            Parameters
            ----------
            watched : QObject
                Widget emitting the event.
            event : QEvent
                Qt event object.

            Returns
            -------
            bool
                Returns base class event filter result.
            """
            message = self._parameter_help_map.get(watched)
            if message:
                event_type = event.type()
                if event_type in (QEvent.Type.Enter, QEvent.Type.FocusIn):
                    self._set_parameter_help(message)
                elif event_type in (QEvent.Type.Leave, QEvent.Type.FocusOut):
                    focus_widget = self.focusWidget()
                    if (
                        focus_widget is not None
                        and focus_widget in self._parameter_help_map
                    ):
                        self._set_parameter_help(self._parameter_help_map[focus_widget])
                    else:
                        self._set_parameter_help(self._parameter_help_default)
            return super().eventFilter(watched, event)

        def _show_operation_configuration(self, operation_name: str) -> None:
            """Display parameter controls for one operation.

            Parameters
            ----------
            operation_name : str
                Operation key to display.

            Returns
            -------
            None
                Visible parameter panel is updated in-place.
            """
            checkbox = self._operation_checkboxes[operation_name]
            if not checkbox.isChecked():
                self._set_status(
                    f"Enable {self._OPERATION_LABELS[operation_name]} before configuring."
                )
                return

            if self._operation_panel_stack is None:
                return

            self._active_config_operation = operation_name
            panel_index = int(self._operation_panel_indices[operation_name])
            self._operation_panel_stack.setCurrentIndex(panel_index)
            for key, button in self._operation_config_buttons.items():
                button.setChecked(key == operation_name)

            self._set_status(
                f"Configuring {self._OPERATION_LABELS[operation_name]} parameters."
            )

        def _operation_output_component(self, operation_name: str) -> str:
            """Return expected latest output component path for an operation.

            Parameters
            ----------
            operation_name : str
                Operation key.

            Returns
            -------
            str
                Component path used for upstream-input labels.
            """
            if operation_name == "particle_detection":
                return self._PARTICLE_DETECTION_OVERLAY_COMPONENT
            if operation_name == "visualization":
                return "results/visualization/latest"
            return self._OPERATION_OUTPUT_COMPONENTS.get(
                operation_name,
                f"results/{operation_name}/latest/data",
            )

        def _selected_operations_in_sequence(self) -> list[str]:
            """Return selected operation keys sorted by execution order.

            Parameters
            ----------
            None

            Returns
            -------
            list[str]
                Selected operations sorted by order spin value and declaration order.
            """
            selected = [
                operation_name
                for operation_name in self._OPERATION_KEYS
                if self._operation_checkboxes[operation_name].isChecked()
            ]
            tie_break = {
                operation_name: idx
                for idx, operation_name in enumerate(self._OPERATION_KEYS)
            }
            return sorted(
                selected,
                key=lambda name: (
                    int(self._operation_order_spins[name].value()),
                    int(tie_break[name]),
                ),
            )

        def _input_options_for_operation(
            self,
            operation_name: str,
            available_store_outputs: Optional[Mapping[str, str]] = None,
        ) -> list[tuple[str, str]]:
            """Build input-source options for a specific operation.

            Parameters
            ----------
            operation_name : str
                Operation key for which options are generated.
            available_store_outputs : mapping[str, str], optional
                Existing operation outputs discovered in the selected store.

            Returns
            -------
            list[tuple[str, str]]
                List of ``(value, label)`` input-source options.
            """
            return _build_input_source_options(
                operation_name=operation_name,
                selected_order=self._selected_operations_in_sequence(),
                operation_key_order=self._OPERATION_KEYS,
                operation_labels=self._OPERATION_LABELS,
                operation_output_components=self._OPERATION_OUTPUT_COMPONENTS,
                available_store_output_components=available_store_outputs,
            )

        def _refresh_input_source_options(self) -> None:
            """Refresh all input-source combo box options.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Combo options are rebuilt in-place.
            """
            available_store_outputs = _discover_available_operation_output_components(
                store_path=str(self._base_config.file or "").strip(),
                operation_output_components=self._OPERATION_OUTPUT_COMPONENTS,
            )

            for operation_name, combo in self._operation_input_combos.items():
                current_data = combo.currentData()
                current_source = (
                    str(current_data).strip() if current_data is not None else "data"
                ) or "data"
                options = self._input_options_for_operation(
                    operation_name,
                    available_store_outputs=available_store_outputs,
                )
                option_values = [value for value, _ in options]

                combo.blockSignals(True)
                combo.clear()
                selected_index = 0
                for idx, (value, label) in enumerate(options):
                    combo.addItem(label, value)
                    if value == current_source:
                        selected_index = idx

                if current_source not in option_values:
                    combo.addItem(
                        f"Custom component ({current_source})",
                        current_source,
                    )
                    selected_index = combo.count() - 1

                combo.setCurrentIndex(selected_index)
                combo.setEnabled(self._operation_checkboxes[operation_name].isChecked())
                combo.blockSignals(False)

        def _has_completed_particle_detection_history(self) -> bool:
            """Return whether provenance has a successful particle-detection run.

            Parameters
            ----------
            None

            Returns
            -------
            bool
                ``True`` when provenance indicates particle detection completed.
            """
            history = self._operation_history_cache.get("particle_detection", {})
            return bool(history.get("has_successful_run", False))

        def _is_particle_overlay_available(self) -> bool:
            """Return whether visualization particle overlay can be enabled.

            Parameters
            ----------
            None

            Returns
            -------
            bool
                ``True`` when current run order or existing outputs/history make
                particle overlay available.
            """
            return _particle_overlay_available_for_visualization(
                selected_order=self._selected_operations_in_sequence(),
                has_particle_detection_history=self._has_completed_particle_detection_history(),
            )

        def _refresh_operation_provenance_statuses(self) -> None:
            """Update per-operation provenance history labels and force-rerun controls.

            Parameters
            ----------
            None

            Returns
            -------
            None
                History labels and force-rerun visibility are updated in-place.
            """
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                for operation_name in self._PROVENANCE_HISTORY_OPERATIONS:
                    history_label = self._operation_history_labels.get(operation_name)
                    force_checkbox = self._operation_force_rerun_checkboxes.get(
                        operation_name
                    )
                    if history_label is not None:
                        history_label.setText(
                            "Provenance: unavailable (current input is not a Zarr/N5 store)."
                        )
                    if force_checkbox is not None:
                        force_checkbox.setVisible(False)
                        force_checkbox.setChecked(False)
                        force_checkbox.setEnabled(False)
                return

            for operation_name in self._PROVENANCE_HISTORY_OPERATIONS:
                history_label = self._operation_history_labels.get(operation_name)
                force_checkbox = self._operation_force_rerun_checkboxes.get(
                    operation_name
                )
                if history_label is None:
                    continue

                try:
                    operation_params = self._collect_operation_parameters(
                        operation_name
                    )
                    normalized = normalize_analysis_operation_parameters(
                        {operation_name: operation_params}
                    )
                    compare_params = dict(
                        normalized.get(operation_name, operation_params)
                    )
                    history = summarize_analysis_history(
                        store_path,
                        operation_name,
                        parameters=compare_params,
                    )
                except Exception as exc:
                    history = {
                        "has_successful_run": False,
                        "latest_success_run_id": None,
                        "latest_success_ended_utc": None,
                        "matches_parameters": False,
                        "matching_run_id": None,
                        "matching_ended_utc": None,
                    }
                    history_label.setText(f"Provenance: unavailable ({exc}).")
                    if force_checkbox is not None:
                        force_checkbox.setVisible(False)
                        force_checkbox.setChecked(False)
                        force_checkbox.setEnabled(False)
                    continue

                self._operation_history_cache[operation_name] = dict(history)
                has_successful_run = bool(history.get("has_successful_run", False))
                matches_parameters = bool(history.get("matches_parameters", False))
                latest_run_id = str(history.get("latest_success_run_id") or "")
                latest_ended = str(history.get("latest_success_ended_utc") or "")
                matching_run_id = str(history.get("matching_run_id") or "")
                matching_ended = str(history.get("matching_ended_utc") or "")

                if not has_successful_run:
                    history_label.setText("Provenance: no successful run recorded yet.")
                elif matches_parameters:
                    summary_bits = (
                        [f"run_id={matching_run_id}"] if matching_run_id else []
                    )
                    if matching_ended:
                        summary_bits.append(f"ended={matching_ended}")
                    summary_text = (
                        ", ".join(summary_bits) if summary_bits else "latest run"
                    )
                    history_label.setText(
                        "Provenance: matching successful run exists "
                        f"({summary_text}); routine will be skipped unless Force rerun is enabled."
                    )
                else:
                    summary_bits = [f"run_id={latest_run_id}"] if latest_run_id else []
                    if latest_ended:
                        summary_bits.append(f"ended={latest_ended}")
                    summary_text = (
                        ", ".join(summary_bits) if summary_bits else "latest run"
                    )
                    history_label.setText(
                        "Provenance: successful run exists with different parameters "
                        f"({summary_text})."
                    )

                if force_checkbox is not None:
                    force_checkbox.setVisible(has_successful_run)
                    if not has_successful_run:
                        force_checkbox.setChecked(False)
                    force_checkbox.setEnabled(
                        has_successful_run
                        and self._operation_checkboxes[operation_name].isChecked()
                    )

        def _set_status(self, text: str) -> None:
            """Update the analysis status label text.

            Parameters
            ----------
            text : str
                Status message.

            Returns
            -------
            None
                Label text is updated in-place.
            """
            if self._status_label is not None:
                self._status_label.setText(str(text))

        def _set_parameter_help(self, text: str) -> None:
            """Update verbose parameter-help text.

            Parameters
            ----------
            text : str
                Help text to display.

            Returns
            -------
            None
                Help label is updated in-place.
            """
            if self._parameter_help_label is not None:
                self._parameter_help_label.setText(str(text))

        def _refresh_dask_backend_summary(self) -> None:
            """Refresh footer summary text for active Dask backend settings.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Footer summary text and tooltip are updated in-place.

            Raises
            ------
            None
                Errors are handled internally.
            """
            if self._dask_backend_summary_label is None:
                return
            summary = format_dask_backend_summary(self._dask_backend_config)
            text = f"Dask backend: {summary}"
            self._dask_backend_summary_label.setText(text)
            self._dask_backend_summary_label.setToolTip(text)
            self._refresh_dask_dashboard_button_state()

        def _analysis_store_shape_tpczyx(
            self,
        ) -> Optional[Tuple[int, int, int, int, int, int]]:
            """Estimate canonical store shape for LocalCluster recommendations.

            Parameters
            ----------
            None

            Returns
            -------
            tuple[int, int, int, int, int, int], optional
                Canonical ``(t, p, c, z, y, x)`` shape when available.

            Raises
            ------
            None
                Store lookup failures are handled internally.
            """
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return None
            try:
                root = zarr.open_group(store_path, mode="r")
                if "data" not in root:
                    return None
                shape = tuple(getattr(root["data"], "shape", ()))
            except Exception:
                return None
            if len(shape) != 6:
                return None
            try:
                return (
                    int(shape[0]),
                    int(shape[1]),
                    int(shape[2]),
                    int(shape[3]),
                    int(shape[4]),
                    int(shape[5]),
                )
            except Exception:
                return None

        def _analysis_store_dtype_itemsize(self) -> Optional[int]:
            """Return bytes-per-voxel for recommendation-aware backend tuning.

            Parameters
            ----------
            None

            Returns
            -------
            int, optional
                Data dtype itemsize in bytes when the canonical store is available.

            Raises
            ------
            None
                Store lookup failures are handled internally.
            """
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return None
            try:
                root = zarr.open_group(store_path, mode="r")
                if "data" not in root:
                    return None
                dtype = getattr(root["data"], "dtype", None)
                itemsize = int(getattr(dtype, "itemsize"))
                return max(1, itemsize)
            except Exception:
                return None

        def _on_edit_dask_backend(self) -> None:
            """Open backend settings dialog and apply selected configuration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Selected backend values are stored in-place.

            Raises
            ------
            None
                Validation and persistence errors are handled internally.
            """
            dialog = DaskBackendConfigDialog(
                initial=self._dask_backend_config,
                recommendation_shape_tpczyx=self._analysis_store_shape_tpczyx(),
                recommendation_chunks_tpczyx=self._base_config.zarr_save.chunks_tpczyx(),
                recommendation_dtype_itemsize=self._analysis_store_dtype_itemsize(),
                parent=self,
            )
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.result_config is None:
                return
            self._dask_backend_config = dialog.result_config
            _save_last_used_dask_backend_config(self._dask_backend_config)
            self._refresh_dask_backend_summary()
            self._set_status("Updated Dask backend settings.")

        @staticmethod
        def _normalize_dashboard_url(
            raw_url: str,
            *,
            default_host: str = "127.0.0.1",
        ) -> Optional[str]:
            """Normalize dashboard address text into an HTTP/HTTPS URL.

            Parameters
            ----------
            raw_url : str
                Raw dashboard address text.
            default_host : str, default="127.0.0.1"
                Hostname used when only a port is supplied.

            Returns
            -------
            str, optional
                Normalized dashboard URL, or ``None`` when parsing fails.

            Raises
            ------
            None
                Invalid inputs are handled internally.
            """
            stripped = str(raw_url).strip()
            if not stripped:
                return None

            if stripped.startswith("tcp://") or stripped.startswith("tls://"):
                parsed_tcp = urlparse(stripped)
                host = parsed_tcp.hostname
                port = parsed_tcp.port
                if host is None:
                    return None
                stripped = f"http://{host}:{port}" if port else f"http://{host}"
            elif stripped.startswith(":"):
                stripped = f"http://{default_host}{stripped}"
            elif stripped.isdigit():
                stripped = f"http://{default_host}:{stripped}"
            elif "://" not in stripped:
                stripped = f"http://{stripped}"

            parsed = urlparse(stripped)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return None

            path = parsed.path
            if not path or path == "/":
                path = "/status"

            normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
            if parsed.query:
                normalized = f"{normalized}?{parsed.query}"
            if parsed.fragment:
                normalized = f"{normalized}#{parsed.fragment}"
            return normalized

        def _dashboard_url_from_scheduler_file(
            self,
            scheduler_file: str,
        ) -> Optional[str]:
            """Resolve dashboard URL from a Dask scheduler file payload.

            Parameters
            ----------
            scheduler_file : str
                Path to the scheduler JSON file.

            Returns
            -------
            str, optional
                Dashboard URL when host and port can be inferred.

            Raises
            ------
            None
                Read/parse failures are handled internally.
            """
            path = Path(str(scheduler_file)).expanduser()
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None

            services = payload.get("services")
            dashboard_service = None
            if isinstance(services, dict):
                dashboard_service = services.get("dashboard")
            if isinstance(dashboard_service, str):
                direct_url = self._normalize_dashboard_url(dashboard_service)
                if direct_url is not None:
                    return direct_url

            try:
                dashboard_port = (
                    int(dashboard_service) if dashboard_service is not None else None
                )
            except (TypeError, ValueError):
                dashboard_port = None

            address = str(payload.get("address") or "").strip()
            parsed_address = urlparse(address) if address else None
            host = (
                (parsed_address.hostname if parsed_address is not None else None)
                or str(payload.get("host") or "").strip()
                or "127.0.0.1"
            )

            if dashboard_port is None:
                return None
            return self._normalize_dashboard_url(
                f"{host}:{dashboard_port}",
                default_host=host,
            )

        def _resolve_dask_dashboard_url(self) -> Optional[str]:
            """Resolve dashboard URL for the currently selected backend mode.

            Parameters
            ----------
            None

            Returns
            -------
            str, optional
                Resolved dashboard URL when it can be inferred from settings.

            Raises
            ------
            None
                Parsing failures are handled internally.
            """
            mode = str(self._dask_backend_config.mode).strip().lower()
            if mode == DASK_BACKEND_LOCAL_CLUSTER:
                return self._normalize_dashboard_url("127.0.0.1:8787")
            if mode == DASK_BACKEND_SLURM_CLUSTER:
                return self._normalize_dashboard_url(
                    self._dask_backend_config.slurm_cluster.dashboard_address,
                )
            if mode == DASK_BACKEND_SLURM_RUNNER:
                scheduler_file = self._dask_backend_config.slurm_runner.scheduler_file
                if not scheduler_file:
                    return None
                return self._dashboard_url_from_scheduler_file(scheduler_file)
            return None

        def _refresh_dask_dashboard_button_state(self) -> None:
            """Refresh dashboard-button enabled state and tooltip text.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Button state and tooltip are updated in-place.

            Raises
            ------
            None
                Errors are handled internally.
            """
            if self._dask_dashboard_button is None:
                return

            dashboard_url = self._resolve_dask_dashboard_url()
            if dashboard_url is None:
                self._dask_dashboard_button.setEnabled(False)
                self._dask_dashboard_button.setToolTip(
                    "Dashboard URL unavailable for the current backend settings."
                )
                return

            self._dask_dashboard_button.setEnabled(True)
            self._dask_dashboard_button.setToolTip(f"Open {dashboard_url}")

        def _on_open_dask_dashboard(self) -> None:
            """Open the configured Dask dashboard URL in a web browser.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Browser-launch side effects only.

            Raises
            ------
            None
                Launch errors are handled via GUI warnings.
            """
            dashboard_url = self._resolve_dask_dashboard_url()
            if dashboard_url is None:
                QMessageBox.information(
                    self,
                    "Dashboard URL Unavailable",
                    "Could not determine a dashboard URL from current Dask backend settings.",
                )
                self._set_status(
                    "Dashboard URL unavailable for current backend settings."
                )
                return

            try:
                opened = bool(webbrowser.open_new_tab(dashboard_url))
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Dashboard Launch Failed",
                    f"Could not open dashboard URL.\n\n{exc}",
                )
                self._set_status("Failed to launch the Dask dashboard browser tab.")
                return

            if not opened:
                QMessageBox.warning(
                    self,
                    "Dashboard Launch Failed",
                    f"Browser did not confirm opening:\n{dashboard_url}",
                )
                self._set_status("Browser did not confirm Dask dashboard launch.")
                return

            self._set_status(f"Opened Dask dashboard: {dashboard_url}")

        def _on_operation_selection_changed(self) -> None:
            """React to operation selection checkbox changes.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget state and status text are updated in-place.
            """
            for operation_name in self._OPERATION_KEYS:
                enabled = self._operation_checkboxes[operation_name].isChecked()
                self._operation_order_spins[operation_name].setEnabled(enabled)
                self._operation_config_buttons[operation_name].setEnabled(enabled)
                self._operation_input_combos[operation_name].setEnabled(enabled)
                force_checkbox = self._operation_force_rerun_checkboxes.get(
                    operation_name
                )
                if force_checkbox is not None:
                    has_success = bool(
                        self._operation_history_cache.get(operation_name, {}).get(
                            "has_successful_run", False
                        )
                    )
                    force_checkbox.setEnabled(enabled and has_success)

            self._refresh_input_source_options()
            self._set_flatfield_parameter_enabled_state()
            self._set_deconvolution_parameter_enabled_state()
            self._set_shear_parameter_enabled_state()
            self._set_particle_parameter_enabled_state()
            self._set_usegment3d_parameter_enabled_state()
            self._set_visualization_parameter_enabled_state()
            self._set_mip_export_parameter_enabled_state()

            if (
                self._active_config_operation is not None
                and not self._operation_checkboxes[
                    self._active_config_operation
                ].isChecked()
                and self._operation_panel_stack is not None
            ):
                self._active_config_operation = None
                self._operation_panel_stack.setCurrentIndex(0)
                for button in self._operation_config_buttons.values():
                    button.setChecked(False)

            sequence = self._selected_operations_in_sequence()
            if not sequence:
                self._set_status("Select at least one analysis routine.")
                return

            sequence_text = " -> ".join(
                self._OPERATION_LABELS[name] for name in sequence
            )
            self._set_status(f"Current execution sequence: {sequence_text}")

        def _on_operation_order_changed(self) -> None:
            """Update dependent options when operation order values change.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Input options and status text are refreshed in-place.
            """
            self._refresh_input_source_options()
            self._set_visualization_parameter_enabled_state()
            sequence = self._selected_operations_in_sequence()
            if sequence:
                sequence_text = " -> ".join(
                    self._OPERATION_LABELS[name] for name in sequence
                )
                self._set_status(f"Current execution sequence: {sequence_text}")

        def _on_deconvolution_psf_mode_changed(self, _: int) -> None:
            """Update deconvolution PSF-parameter widget states.

            Parameters
            ----------
            _ : int
                Unused combo-box index.

            Returns
            -------
            None
                Deconvolution widget state is updated in-place.
            """
            self._set_deconvolution_parameter_enabled_state()

        def _on_preview_synthetic_psf(self) -> None:
            """Render and display a synthetic PSF preview dialog.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Opens a modal preview dialog on success.
            """
            try:
                params = self._collect_deconvolution_parameters()
                if str(params.get("psf_mode", "measured")) != "synthetic":
                    QMessageBox.information(
                        self,
                        "Synthetic Preview Unavailable",
                        "Switch PSF mode to Synthetic to preview synthetic PSFs.",
                    )
                    return
                from clearex.deconvolution.pipeline import (
                    generate_synthetic_psf_preview,
                )

                preview_png, metadata = generate_synthetic_psf_preview(
                    parameters=params
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Synthetic Preview Failed",
                    f"Could not generate synthetic PSF preview.\n\n{exc}",
                )
                return

            image = QImage.fromData(preview_png, "PNG")
            if image.isNull():
                QMessageBox.warning(
                    self,
                    "Synthetic Preview Failed",
                    "Generated preview image data is not a valid PNG.",
                )
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("Synthetic PSF Preview")
            dialog.setMinimumWidth(840)
            dialog.setMinimumHeight(700)
            dialog.setStyleSheet(
                """
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                }
                QLabel#previewTitle {
                    color: #9cc6ff;
                    font-size: 15px;
                    font-weight: 700;
                }
                QLabel#previewDetails {
                    color: #c8daf3;
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                    padding: 8px;
                }
                QLabel#previewImage {
                    background-color: #111925;
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                }
                QPushButton {
                    background-color: #1a2635;
                    border: 1px solid #2f4460;
                    border-radius: 8px;
                    padding: 8px 14px;
                    color: #dbe9ff;
                }
                QPushButton:hover {
                    background-color: #22354c;
                }
                """
            )
            layout = QVBoxLayout(dialog)
            apply_popup_root_spacing(layout)

            title = QLabel("Vectorial Synthetic PSF Preview")
            title.setObjectName("previewTitle")
            layout.addWidget(title)

            details = QLabel(
                ", ".join(f"{key}={value}" for key, value in sorted(metadata.items()))
            )
            details.setObjectName("previewDetails")
            details.setWordWrap(True)
            layout.addWidget(details)

            image_label = QLabel()
            image_label.setObjectName("previewImage")
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    780,
                    560,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            layout.addWidget(image_label, 1)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            footer.addStretch(1)
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            footer.addWidget(close_button)
            layout.addLayout(footer)
            dialog.exec()

        def _set_deconvolution_parameter_enabled_state(self) -> None:
            """Enable or disable deconvolution widgets based on mode/selection.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            decon_enabled = self._operation_checkboxes["deconvolution"].isChecked()
            psf_mode = str(self._decon_psf_mode_combo.currentData() or "measured")
            measured_mode = psf_mode == "measured"
            synthetic_mode = decon_enabled and not measured_mode
            synthetic_microscopy_mode = str(
                self._decon_synth_mode_combo.currentData() or "widefield"
            )
            light_sheet_mode = synthetic_microscopy_mode == "light_sheet"
            large_file_enabled = (
                decon_enabled and self._decon_large_file_checkbox.isChecked()
            )

            general_widgets = (
                self._decon_psf_mode_combo,
                self._decon_synth_mode_combo,
                self._decon_data_xy_spin,
                self._decon_data_z_spin,
                self._decon_hann_low_spin,
                self._decon_hann_high_spin,
                self._decon_wiener_spin,
                self._decon_background_spin,
                self._decon_iterations_spin,
                self._decon_large_file_checkbox,
                self._decon_cpus_spin,
            )
            for widget in general_widgets:
                widget.setEnabled(decon_enabled)

            measured_widgets = (
                self._decon_measured_psf_paths_input,
                self._decon_measured_psf_xy_input,
                self._decon_measured_psf_z_input,
            )
            for widget in measured_widgets:
                widget.setEnabled(decon_enabled and measured_mode)
            if self._decon_measured_section is not None:
                self._decon_measured_section.setVisible(decon_enabled and measured_mode)

            synthetic_widgets = (
                self._decon_synth_excitation_input,
                self._decon_synth_illum_na_input,
                self._decon_synth_emission_input,
                self._decon_synth_na_input,
                self._decon_preview_psf_button,
            )
            for widget in synthetic_widgets:
                widget.setEnabled(synthetic_mode)
            if self._decon_synthetic_section is not None:
                self._decon_synthetic_section.setVisible(synthetic_mode)

            light_sheet_widgets = (
                self._decon_synth_excitation_input,
                self._decon_synth_illum_na_input,
            )
            for widget in light_sheet_widgets:
                widget.setEnabled(synthetic_mode and light_sheet_mode)
            if self._decon_light_sheet_section is not None:
                self._decon_light_sheet_section.setVisible(
                    synthetic_mode and light_sheet_mode
                )

            block_widgets = (
                self._decon_block_z_spin,
                self._decon_block_y_spin,
                self._decon_block_x_spin,
                self._decon_batch_z_spin,
                self._decon_batch_y_spin,
                self._decon_batch_x_spin,
            )
            for widget in block_widgets:
                widget.setEnabled(large_file_enabled)

        def _set_shear_parameter_enabled_state(self) -> None:
            """Enable/disable shear-transform widgets based on selection.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            shear_enabled = self._operation_checkboxes["shear_transform"].isChecked()
            widgets = (
                self._shear_xy_spin,
                self._shear_xz_spin,
                self._shear_yz_spin,
                self._shear_auto_rotate_checkbox,
                self._shear_auto_estimate_checkbox,
                self._shear_auto_estimate_fraction_spin,
                self._shear_auto_estimate_stride_spin,
                self._shear_auto_estimate_signal_fraction_spin,
                self._shear_auto_estimate_t_spin,
                self._shear_auto_estimate_p_spin,
                self._shear_auto_estimate_c_spin,
                self._shear_rotation_x_spin,
                self._shear_rotation_y_spin,
                self._shear_rotation_z_spin,
                self._shear_interpolation_combo,
                self._shear_fill_value_spin,
                self._shear_output_dtype_combo,
                self._shear_padding_z_spin,
                self._shear_padding_y_spin,
                self._shear_padding_x_spin,
            )
            for widget in widgets:
                widget.setEnabled(shear_enabled)

        def _set_particle_parameter_enabled_state(self) -> None:
            """Enable/disable particle widgets based on selection and sub-options.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            particle_enabled = self._operation_checkboxes[
                "particle_detection"
            ].isChecked()
            overlap_enabled = (
                particle_enabled and self._particle_use_overlap_checkbox.isChecked()
            )
            close_enabled = (
                particle_enabled and self._particle_remove_close_checkbox.isChecked()
            )

            widgets = (
                self._particle_channel_spin,
                self._particle_bg_sigma_spin,
                self._particle_fwhm_spin,
                self._particle_sigma_min_spin,
                self._particle_sigma_max_spin,
                self._particle_threshold_spin,
                self._particle_overlap_spin,
                self._particle_exclude_border_spin,
                self._particle_use_overlap_checkbox,
                self._particle_eliminate_checkbox,
                self._particle_remove_close_checkbox,
            )
            for widget in widgets:
                widget.setEnabled(particle_enabled)

            self._particle_overlap_z_spin.setEnabled(overlap_enabled)
            self._particle_overlap_y_spin.setEnabled(overlap_enabled)
            self._particle_overlap_x_spin.setEnabled(overlap_enabled)
            self._particle_min_distance_spin.setEnabled(close_enabled)

        def _set_usegment3d_parameter_enabled_state(self) -> None:
            """Enable/disable uSegment3D widgets based on selection and sub-options.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            usegment_enabled = self._operation_checkboxes["usegment3d"].isChecked()
            require_gpu = bool(self._usegment3d_require_gpu_checkbox.isChecked())
            if require_gpu and not self._usegment3d_gpu_checkbox.isChecked():
                self._usegment3d_gpu_checkbox.setChecked(True)

            widgets = (
                self._usegment3d_channel_scroll,
                self._usegment3d_views_input,
                self._usegment3d_resolution_level_spin,
                self._usegment3d_output_reference_combo,
                self._usegment3d_save_native_labels_checkbox,
                self._usegment3d_require_gpu_checkbox,
                self._usegment3d_preprocess_min_spin,
                self._usegment3d_preprocess_max_spin,
                self._usegment3d_preprocess_bg_correction_checkbox,
                self._usegment3d_preprocess_factor_spin,
                self._usegment3d_cellpose_model_combo,
                self._usegment3d_cellpose_channels_input,
                self._usegment3d_cellpose_diameter_spin,
                self._usegment3d_cellpose_edge_checkbox,
                self._usegment3d_aggregation_decay_spin,
                self._usegment3d_aggregation_iterations_spin,
                self._usegment3d_aggregation_momentum_spin,
                self._usegment3d_tile_mode_combo,
                self._usegment3d_postprocess_checkbox,
            )
            for widget in widgets:
                widget.setEnabled(usegment_enabled)

            self._usegment3d_gpu_checkbox.setEnabled(
                usegment_enabled and not require_gpu
            )

            resolution_level = int(self._usegment3d_resolution_level_spin.value())
            output_reference_space = (
                str(self._usegment3d_output_reference_combo.currentData() or "level0")
                .strip()
                .lower()
                or "level0"
            )
            output_space_enabled = usegment_enabled and resolution_level > 0
            self._usegment3d_output_reference_combo.setEnabled(output_space_enabled)
            save_native_enabled = (
                usegment_enabled
                and resolution_level > 0
                and output_reference_space == "level0"
            )
            self._usegment3d_save_native_labels_checkbox.setEnabled(save_native_enabled)
            if not save_native_enabled:
                self._usegment3d_save_native_labels_checkbox.setChecked(False)

            tile_mode = str(self._usegment3d_tile_mode_combo.currentData() or "auto")
            tile_mode = tile_mode.strip().lower() or "auto"
            tile_widgets_enabled = usegment_enabled and tile_mode != "none"
            self._usegment3d_tile_z_spin.setEnabled(tile_widgets_enabled)
            self._usegment3d_tile_y_spin.setEnabled(tile_widgets_enabled)
            self._usegment3d_tile_x_spin.setEnabled(tile_widgets_enabled)
            self._usegment3d_tile_overlap_z_spin.setEnabled(tile_widgets_enabled)
            self._usegment3d_tile_overlap_y_spin.setEnabled(tile_widgets_enabled)
            self._usegment3d_tile_overlap_x_spin.setEnabled(tile_widgets_enabled)

            postprocess_enabled = (
                usegment_enabled and self._usegment3d_postprocess_checkbox.isChecked()
            )
            self._usegment3d_postprocess_min_size_spin.setEnabled(postprocess_enabled)
            self._usegment3d_postprocess_flow_threshold_spin.setEnabled(
                postprocess_enabled
            )
            self._usegment3d_postprocess_dtform_combo.setEnabled(postprocess_enabled)

        def _set_flatfield_parameter_enabled_state(self) -> None:
            """Enable/disable flatfield widgets based on selection and overlap mode.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            flatfield_enabled = self._operation_checkboxes["flatfield"].isChecked()
            overlap_enabled = (
                flatfield_enabled and self._flatfield_use_overlap_checkbox.isChecked()
            )

            widgets = (
                self._flatfield_darkfield_checkbox,
                self._flatfield_smoothness_spin,
                self._flatfield_working_size_spin,
                self._flatfield_tile_y_spin,
                self._flatfield_tile_x_spin,
                self._flatfield_timelapse_checkbox,
                self._flatfield_use_overlap_checkbox,
            )
            for widget in widgets:
                widget.setEnabled(flatfield_enabled)

            self._flatfield_blend_tiles_checkbox.setEnabled(overlap_enabled)
            self._flatfield_overlap_z_spin.setEnabled(overlap_enabled)
            self._flatfield_overlap_y_spin.setEnabled(overlap_enabled)
            self._flatfield_overlap_x_spin.setEnabled(overlap_enabled)

        def _set_visualization_parameter_enabled_state(self) -> None:
            """Enable/disable visualization widgets based on operation selection.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            visualization_enabled = self._operation_checkboxes[
                "visualization"
            ].isChecked()
            overlay_available = self._is_particle_overlay_available()
            self._visualization_show_all_positions_checkbox.setEnabled(
                visualization_enabled
            )
            show_all_positions = bool(
                self._visualization_show_all_positions_checkbox.isChecked()
            )
            self._visualization_position_spin.setEnabled(
                visualization_enabled and not show_all_positions
            )
            self._visualization_position_label.setEnabled(
                visualization_enabled and not show_all_positions
            )
            self._visualization_multiscale_checkbox.setEnabled(visualization_enabled)
            self._visualization_require_gpu_checkbox.setEnabled(visualization_enabled)
            if self._visualization_volume_layers_button is not None:
                self._visualization_volume_layers_button.setEnabled(
                    visualization_enabled
                )
            if self._visualization_volume_layers_summary_label is not None:
                self._visualization_volume_layers_summary_label.setEnabled(
                    visualization_enabled
                )
            if not overlay_available:
                self._visualization_overlay_points_checkbox.setChecked(False)
            self._visualization_overlay_points_checkbox.setEnabled(
                visualization_enabled and overlay_available
            )
            if self._visualization_layer_table_button is not None:
                self._visualization_layer_table_button.setEnabled(visualization_enabled)
            if self._visualization_layer_table_summary_label is not None:
                self._visualization_layer_table_summary_label.setEnabled(
                    visualization_enabled
                )
            overlay_hint = self._PARAMETER_HINTS["overlay_particle_detections"]
            if not overlay_available:
                overlay_hint = (
                    "Particle overlay is unavailable until particle detection has "
                    "completed in this store or is selected to run before visualization."
                )
            self._visualization_overlay_points_checkbox.setToolTip(overlay_hint)
            self._parameter_help_map[self._visualization_overlay_points_checkbox] = (
                overlay_hint
            )
            self._set_visualization_position_selector_state()

        def _set_visualization_position_selector_state(self) -> None:
            """Show/hide position-index controls for multi-position rendering.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Position-index widget visibility is updated in-place.
            """
            show_all_positions = bool(
                self._visualization_show_all_positions_checkbox.isChecked()
            )
            self._visualization_position_label.setVisible(not show_all_positions)
            self._visualization_position_spin.setVisible(not show_all_positions)

        def _set_mip_export_parameter_enabled_state(self) -> None:
            """Enable/disable MIP-export widgets based on operation selection.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            mip_enabled = self._operation_checkboxes["mip_export"].isChecked()
            widgets = (
                self._mip_position_mode_combo,
                self._mip_export_format_combo,
                self._mip_output_directory_input,
            )
            for widget in widgets:
                widget.setEnabled(mip_enabled)

        def _hydrate(self, initial: WorkflowConfig) -> None:
            """Populate analysis selections from initial workflow values.

            Parameters
            ----------
            initial : WorkflowConfig
                Workflow values from setup step.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            normalized_parameters = normalize_analysis_operation_parameters(
                initial.analysis_parameters
            )
            flatfield_params = dict(
                normalized_parameters.get("flatfield", self._flatfield_defaults)
            )
            decon_params = dict(
                normalized_parameters.get("deconvolution", self._decon_defaults)
            )
            shear_params = dict(
                normalized_parameters.get("shear_transform", self._shear_defaults)
            )
            particle_params = dict(
                normalized_parameters.get("particle_detection", self._particle_defaults)
            )
            usegment3d_params = dict(
                normalized_parameters.get("usegment3d", self._usegment3d_defaults)
            )
            visualization_params = dict(
                normalized_parameters.get("visualization", self._visualization_defaults)
            )
            mip_export_params = dict(
                normalized_parameters.get("mip_export", self._mip_export_defaults)
            )

            if self._store_label is not None:
                self._store_label.setText(initial.file or "n/a")
            self._refresh_dask_backend_summary()

            checkbox_defaults = {
                "flatfield": bool(initial.flatfield),
                "deconvolution": bool(initial.deconvolution),
                "shear_transform": bool(initial.shear_transform),
                "particle_detection": bool(initial.particle_detection),
                "usegment3d": bool(getattr(initial, "usegment3d", False)),
                "registration": bool(initial.registration),
                "visualization": bool(initial.visualization),
                "mip_export": bool(initial.mip_export),
            }

            for idx, operation_name in enumerate(self._OPERATION_KEYS):
                operation_params = dict(
                    normalized_parameters.get(
                        operation_name,
                        self._operation_defaults.get(operation_name, {}),
                    )
                )
                self._operation_checkboxes[operation_name].setChecked(
                    checkbox_defaults[operation_name]
                )
                self._operation_order_spins[operation_name].setValue(
                    int(operation_params.get("execution_order", idx + 1))
                )
                force_checkbox = self._operation_force_rerun_checkboxes.get(
                    operation_name
                )
                if force_checkbox is not None:
                    force_checkbox.setChecked(
                        bool(operation_params.get("force_rerun", False))
                    )

            self._refresh_input_source_options()
            for operation_name in self._OPERATION_KEYS:
                requested_source = (
                    str(
                        normalized_parameters.get(operation_name, {}).get(
                            "input_source", "data"
                        )
                    ).strip()
                    or "data"
                )
                combo = self._operation_input_combos[operation_name]
                combo.blockSignals(True)
                combo_index = combo.findData(requested_source)
                if combo_index < 0:
                    combo.addItem(
                        f"Custom component ({requested_source})",
                        requested_source,
                    )
                    combo_index = combo.count() - 1
                combo.setCurrentIndex(combo_index)
                combo.blockSignals(False)

            channel_count = 1
            position_count = 1
            max_usegment3d_resolution_level = 0
            store_xy_um: Optional[float] = None
            store_z_um: Optional[float] = None
            if initial.file and is_zarr_store_path(initial.file):
                try:
                    root = zarr.open_group(str(initial.file), mode="r")
                    if "data" in root and len(tuple(root["data"].shape)) == 6:
                        position_count = max(1, int(root["data"].shape[1]))
                        channel_count = max(1, int(root["data"].shape[2]))
                        voxel_size = root["data"].attrs.get("voxel_size_um_zyx")
                        if (
                            isinstance(voxel_size, (list, tuple))
                            and len(voxel_size) >= 3
                        ):
                            store_z_um = float(voxel_size[0])
                            store_xy_um = float(voxel_size[2])
                    if store_xy_um is None or store_z_um is None:
                        root_voxel_size = root.attrs.get("voxel_size_um_zyx")
                        if (
                            isinstance(root_voxel_size, (list, tuple))
                            and len(root_voxel_size) >= 3
                        ):
                            store_z_um = float(root_voxel_size[0])
                            store_xy_um = float(root_voxel_size[2])
                    if store_xy_um is None or store_z_um is None:
                        navigate = root.attrs.get("navigate_experiment")
                        if isinstance(navigate, dict):
                            xy_value = navigate.get("xy_pixel_size_um")
                            z_value = navigate.get("z_step_um")
                            if xy_value is not None and z_value is not None:
                                store_xy_um = float(xy_value)
                                store_z_um = float(z_value)
                    if "data_pyramid" in root:
                        data_pyramid_group = root["data_pyramid"]
                        for key in data_pyramid_group.group_keys():
                            token = str(key).strip().lower()
                            if not token.startswith("level_"):
                                continue
                            try:
                                parsed_level = int(token.split("_", maxsplit=1)[1])
                            except Exception:
                                continue
                            max_usegment3d_resolution_level = max(
                                max_usegment3d_resolution_level,
                                max(0, int(parsed_level)),
                            )
                    factors = root.attrs.get("data_pyramid_factors_tpczyx")
                    if isinstance(factors, (tuple, list)):
                        max_usegment3d_resolution_level = max(
                            max_usegment3d_resolution_level,
                            max(0, int(len(factors) - 1)),
                        )
                except Exception:
                    position_count = 1
                    channel_count = 1
                    max_usegment3d_resolution_level = 0
                    store_xy_um = None
                    store_z_um = None
            self._visualization_position_spin.setMaximum(position_count - 1)
            self._particle_channel_spin.setMaximum(channel_count - 1)
            self._usegment3d_resolution_level_spin.setMaximum(
                max(0, int(max_usegment3d_resolution_level))
            )
            requested_usegment_channels = usegment3d_params.get("channel_indices", [])
            if isinstance(requested_usegment_channels, str):
                requested_channel_values: list[Any] = self._split_csv_values(
                    requested_usegment_channels
                )
            elif isinstance(requested_usegment_channels, (tuple, list)):
                requested_channel_values = list(requested_usegment_channels)
            else:
                requested_channel_values = []
            if not requested_channel_values:
                requested_channel_values = [usegment3d_params.get("channel_index", 0)]
            selected_usegment_channels: list[int] = []
            selected_usegment_seen: set[int] = set()
            for value in requested_channel_values:
                try:
                    parsed_channel = int(float(value))
                except (TypeError, ValueError):
                    continue
                clamped_channel = max(0, min(channel_count - 1, parsed_channel))
                if clamped_channel in selected_usegment_seen:
                    continue
                selected_usegment_seen.add(clamped_channel)
                selected_usegment_channels.append(clamped_channel)
            self._rebuild_usegment3d_channel_checkboxes(
                channel_count=channel_count,
                selected_channels=selected_usegment_channels,
            )

            self._flatfield_darkfield_checkbox.setChecked(
                bool(flatfield_params.get("get_darkfield", False))
            )
            self._flatfield_smoothness_spin.setValue(
                float(flatfield_params.get("smoothness_flatfield", 1.0))
            )
            self._flatfield_working_size_spin.setValue(
                max(1, int(flatfield_params.get("working_size", 128)))
            )
            flatfield_fit_tile_shape = flatfield_params.get(
                "fit_tile_shape_yx", [256, 256]
            )
            if (
                not isinstance(flatfield_fit_tile_shape, (tuple, list))
                or len(flatfield_fit_tile_shape) != 2
            ):
                flatfield_fit_tile_shape = [256, 256]
            self._flatfield_tile_y_spin.setValue(
                max(1, int(flatfield_fit_tile_shape[0]))
            )
            self._flatfield_tile_x_spin.setValue(
                max(1, int(flatfield_fit_tile_shape[1]))
            )
            self._flatfield_timelapse_checkbox.setChecked(
                bool(flatfield_params.get("is_timelapse", False))
            )
            self._flatfield_use_overlap_checkbox.setChecked(
                bool(flatfield_params.get("use_map_overlap", True))
            )
            self._flatfield_blend_tiles_checkbox.setChecked(
                bool(flatfield_params.get("blend_tiles", False))
            )
            flatfield_overlap_zyx = flatfield_params.get("overlap_zyx", [0, 32, 32])
            if (
                not isinstance(flatfield_overlap_zyx, (tuple, list))
                or len(flatfield_overlap_zyx) != 3
            ):
                flatfield_overlap_zyx = [0, 32, 32]
            self._flatfield_overlap_z_spin.setValue(int(flatfield_overlap_zyx[0]))
            self._flatfield_overlap_y_spin.setValue(int(flatfield_overlap_zyx[1]))
            self._flatfield_overlap_x_spin.setValue(int(flatfield_overlap_zyx[2]))

            decon_mode = str(decon_params.get("psf_mode", "measured")).strip().lower()
            decon_mode_index = self._decon_psf_mode_combo.findData(decon_mode)
            if decon_mode_index < 0:
                decon_mode_index = self._decon_psf_mode_combo.findData("measured")
            if decon_mode_index < 0:
                decon_mode_index = 0
            self._decon_psf_mode_combo.setCurrentIndex(decon_mode_index)
            self._decon_measured_psf_paths_input.setText(
                ",".join(
                    str(value) for value in decon_params.get("measured_psf_paths", [])
                )
            )
            self._decon_measured_psf_xy_input.setText(
                ",".join(
                    str(float(value))
                    for value in decon_params.get("measured_psf_xy_um", [])
                )
            )
            self._decon_measured_psf_z_input.setText(
                ",".join(
                    str(float(value))
                    for value in decon_params.get("measured_psf_z_um", [])
                )
            )
            synth_mode = (
                str(
                    decon_params.get(
                        "synthetic_microscopy_mode",
                        self._decon_defaults.get(
                            "synthetic_microscopy_mode", "widefield"
                        ),
                    )
                )
                .strip()
                .lower()
                .replace("-", "_")
            )
            if synth_mode == "lightsheet":
                synth_mode = "light_sheet"
            synth_mode_index = self._decon_synth_mode_combo.findData(synth_mode)
            if synth_mode_index < 0:
                synth_mode_index = self._decon_synth_mode_combo.findData("widefield")
            if synth_mode_index < 0:
                synth_mode_index = 0
            self._decon_synth_mode_combo.setCurrentIndex(synth_mode_index)
            self._decon_synth_excitation_input.setText(
                ",".join(
                    str(float(value))
                    for value in decon_params.get(
                        "synthetic_illumination_wavelength_nm",
                        decon_params.get("synthetic_excitation_nm", [488.0]),
                    )
                )
            )
            self._decon_synth_illum_na_input.setText(
                ",".join(
                    str(float(value))
                    for value in decon_params.get(
                        "synthetic_illumination_numerical_aperture",
                        [0.2],
                    )
                )
            )
            self._decon_synth_emission_input.setText(
                ",".join(
                    str(float(value))
                    for value in decon_params.get("synthetic_emission_nm", [520.0])
                )
            )
            self._decon_synth_na_input.setText(
                ",".join(
                    str(float(value))
                    for value in decon_params.get(
                        "synthetic_detection_numerical_aperture",
                        decon_params.get("synthetic_numerical_aperture", [0.7]),
                    )
                )
            )
            data_xy_um = float(decon_params.get("data_xy_pixel_um", 0.0))
            data_z_um = float(decon_params.get("data_z_pixel_um", 0.0))
            if data_xy_um <= 0 and store_xy_um is not None:
                data_xy_um = float(store_xy_um)
            if data_z_um <= 0 and store_z_um is not None:
                data_z_um = float(store_z_um)
            self._decon_data_xy_spin.setValue(max(0.0, data_xy_um))
            self._decon_data_z_spin.setValue(max(0.0, data_z_um))
            hann_bounds = decon_params.get("hann_window_bounds", [0.8, 1.0])
            if not isinstance(hann_bounds, (tuple, list)) or len(hann_bounds) != 2:
                hann_bounds = [0.8, 1.0]
            self._decon_hann_low_spin.setValue(float(hann_bounds[0]))
            self._decon_hann_high_spin.setValue(float(hann_bounds[1]))
            self._decon_wiener_spin.setValue(
                float(decon_params.get("wiener_alpha", 0.005))
            )
            self._decon_background_spin.setValue(
                float(decon_params.get("background", 110.0))
            )
            self._decon_iterations_spin.setValue(
                max(1, int(decon_params.get("decon_iterations", 2)))
            )
            self._decon_large_file_checkbox.setChecked(
                bool(decon_params.get("large_file", False))
            )
            block_size = decon_params.get("block_size_zyx", [256, 256, 256])
            if not isinstance(block_size, (tuple, list)) or len(block_size) != 3:
                block_size = [256, 256, 256]
            self._decon_block_z_spin.setValue(max(1, int(block_size[0])))
            self._decon_block_y_spin.setValue(max(1, int(block_size[1])))
            self._decon_block_x_spin.setValue(max(1, int(block_size[2])))
            batch_size = decon_params.get("batch_size_zyx", [1024, 1024, 1024])
            if not isinstance(batch_size, (tuple, list)) or len(batch_size) != 3:
                batch_size = [1024, 1024, 1024]
            self._decon_batch_z_spin.setValue(max(1, int(batch_size[0])))
            self._decon_batch_y_spin.setValue(max(1, int(batch_size[1])))
            self._decon_batch_x_spin.setValue(max(1, int(batch_size[2])))
            self._decon_cpus_spin.setValue(
                max(1, int(decon_params.get("cpus_per_task", 2)))
            )

            self._shear_xy_spin.setValue(
                float(
                    shear_params.get(
                        "shear_xy_deg",
                        _shear_coefficient_to_degrees(
                            float(shear_params.get("shear_xy", 0.0))
                        ),
                    )
                )
            )
            self._shear_xz_spin.setValue(
                float(
                    shear_params.get(
                        "shear_xz_deg",
                        _shear_coefficient_to_degrees(
                            float(shear_params.get("shear_xz", 0.0))
                        ),
                    )
                )
            )
            self._shear_yz_spin.setValue(
                float(
                    shear_params.get(
                        "shear_yz_deg",
                        _shear_coefficient_to_degrees(
                            float(shear_params.get("shear_yz", 0.0))
                        ),
                    )
                )
            )
            self._shear_auto_rotate_checkbox.setChecked(
                bool(shear_params.get("auto_rotate_from_shear", False))
            )
            self._shear_auto_estimate_checkbox.setChecked(
                bool(shear_params.get("auto_estimate_shear_yz", False))
            )
            self._shear_auto_estimate_fraction_spin.setValue(
                float(shear_params.get("auto_estimate_extreme_fraction_x", 0.03))
            )
            self._shear_auto_estimate_stride_spin.setValue(
                max(1, int(shear_params.get("auto_estimate_zy_stride", 4)))
            )
            self._shear_auto_estimate_signal_fraction_spin.setValue(
                float(shear_params.get("auto_estimate_signal_fraction", 0.10))
            )
            self._shear_auto_estimate_t_spin.setValue(
                max(0, int(shear_params.get("auto_estimate_t_index", 0)))
            )
            self._shear_auto_estimate_p_spin.setValue(
                max(0, int(shear_params.get("auto_estimate_p_index", 0)))
            )
            self._shear_auto_estimate_c_spin.setValue(
                max(0, int(shear_params.get("auto_estimate_c_index", 0)))
            )
            self._shear_rotation_x_spin.setValue(
                float(shear_params.get("rotation_deg_x", 0.0))
            )
            self._shear_rotation_y_spin.setValue(
                float(shear_params.get("rotation_deg_y", 0.0))
            )
            self._shear_rotation_z_spin.setValue(
                float(shear_params.get("rotation_deg_z", 0.0))
            )
            interpolation_value = (
                str(shear_params.get("interpolation", "linear")).strip().lower()
                or "linear"
            )
            interpolation_index = self._shear_interpolation_combo.findData(
                interpolation_value
            )
            if interpolation_index < 0:
                interpolation_index = self._shear_interpolation_combo.findData("linear")
            if interpolation_index < 0:
                interpolation_index = 0
            self._shear_interpolation_combo.setCurrentIndex(interpolation_index)
            self._shear_fill_value_spin.setValue(
                float(shear_params.get("fill_value", 0.0))
            )
            dtype_value = (
                str(shear_params.get("output_dtype", "float32")).strip().lower()
                or "float32"
            )
            dtype_index = self._shear_output_dtype_combo.findData(dtype_value)
            if dtype_index < 0:
                self._shear_output_dtype_combo.addItem(dtype_value, dtype_value)
                dtype_index = self._shear_output_dtype_combo.count() - 1
            self._shear_output_dtype_combo.setCurrentIndex(dtype_index)
            padding_zyx = shear_params.get("roi_padding_zyx", [2, 2, 2])
            if not isinstance(padding_zyx, (tuple, list)) or len(padding_zyx) != 3:
                padding_zyx = [2, 2, 2]
            self._shear_padding_z_spin.setValue(max(0, int(padding_zyx[0])))
            self._shear_padding_y_spin.setValue(max(0, int(padding_zyx[1])))
            self._shear_padding_x_spin.setValue(max(0, int(padding_zyx[2])))

            self._particle_channel_spin.setValue(
                max(
                    0,
                    min(
                        int(self._particle_channel_spin.maximum()),
                        int(particle_params.get("channel_index", 0)),
                    ),
                )
            )
            self._particle_bg_sigma_spin.setValue(
                float(particle_params.get("bg_sigma", 20.0))
            )
            self._particle_fwhm_spin.setValue(
                float(particle_params.get("fwhm_px", 3.0))
            )
            self._particle_sigma_min_spin.setValue(
                float(particle_params.get("sigma_min_factor", 1.0))
            )
            self._particle_sigma_max_spin.setValue(
                float(particle_params.get("sigma_max_factor", 3.0))
            )
            self._particle_threshold_spin.setValue(
                float(particle_params.get("threshold", 0.1))
            )
            self._particle_overlap_spin.setValue(
                float(particle_params.get("overlap", 0.5))
            )
            self._particle_exclude_border_spin.setValue(
                int(particle_params.get("exclude_border", 5))
            )
            self._particle_use_overlap_checkbox.setChecked(
                bool(particle_params.get("use_map_overlap", False))
            )
            overlap_zyx = particle_params.get("overlap_zyx", [0, 0, 0])
            if not isinstance(overlap_zyx, (tuple, list)) or len(overlap_zyx) != 3:
                overlap_zyx = [0, 0, 0]
            self._particle_overlap_z_spin.setValue(int(overlap_zyx[0]))
            self._particle_overlap_y_spin.setValue(int(overlap_zyx[1]))
            self._particle_overlap_x_spin.setValue(int(overlap_zyx[2]))
            self._particle_eliminate_checkbox.setChecked(
                bool(particle_params.get("eliminate_insignificant_particles", False))
            )
            self._particle_remove_close_checkbox.setChecked(
                bool(particle_params.get("remove_close_particles", False))
            )
            self._particle_min_distance_spin.setValue(
                float(particle_params.get("min_distance_sigma", 10.0))
            )

            views_value = usegment3d_params.get(
                "use_views",
                self._usegment3d_defaults.get("use_views", ["xy", "xz", "yz"]),
            )
            if isinstance(views_value, (tuple, list)):
                views_text = ",".join(
                    str(item).strip() for item in views_value if str(item).strip()
                )
            else:
                views_text = str(views_value or "").strip()
            self._usegment3d_views_input.setText(views_text)
            input_resolution_level = max(
                0,
                int(
                    usegment3d_params.get(
                        "input_resolution_level",
                        usegment3d_params.get("resolution_level", 0),
                    )
                ),
            )
            self._usegment3d_resolution_level_spin.setValue(
                min(
                    int(self._usegment3d_resolution_level_spin.maximum()),
                    int(input_resolution_level),
                )
            )
            output_reference_space = (
                str(usegment3d_params.get("output_reference_space", "level0"))
                .strip()
                .lower()
                or "level0"
            )
            if output_reference_space not in {"level0", "native_level"}:
                output_reference_space = "level0"
            output_space_index = self._usegment3d_output_reference_combo.findData(
                output_reference_space
            )
            if output_space_index < 0:
                output_space_index = self._usegment3d_output_reference_combo.findData(
                    "level0"
                )
            if output_space_index < 0:
                output_space_index = 0
            self._usegment3d_output_reference_combo.setCurrentIndex(output_space_index)
            self._usegment3d_save_native_labels_checkbox.setChecked(
                bool(usegment3d_params.get("save_native_labels", False))
            )
            self._usegment3d_gpu_checkbox.setChecked(
                bool(
                    usegment3d_params.get(
                        "gpu", usegment3d_params.get("use_gpu", False)
                    )
                )
            )
            self._usegment3d_require_gpu_checkbox.setChecked(
                bool(usegment3d_params.get("require_gpu", False))
            )
            self._usegment3d_preprocess_min_spin.setValue(
                float(
                    usegment3d_params.get(
                        "preprocess_normalize_min",
                        usegment3d_params.get("preprocess_min", 2.0),
                    )
                )
            )
            self._usegment3d_preprocess_max_spin.setValue(
                float(
                    usegment3d_params.get(
                        "preprocess_normalize_max",
                        usegment3d_params.get("preprocess_max", 99.8),
                    )
                )
            )
            self._usegment3d_preprocess_bg_correction_checkbox.setChecked(
                bool(
                    usegment3d_params.get(
                        "preprocess_do_bg_correction",
                        usegment3d_params.get(
                            "preprocess_bg_correction",
                            usegment3d_params.get("bg_correction", True),
                        ),
                    )
                )
            )
            self._usegment3d_preprocess_factor_spin.setValue(
                float(usegment3d_params.get("preprocess_factor", 1.0))
            )
            model_value = (
                str(
                    usegment3d_params.get(
                        "cellpose_model_name",
                        usegment3d_params.get("cellpose_model", "cyto"),
                    )
                ).strip()
                or "cyto"
            )
            model_index = self._usegment3d_cellpose_model_combo.findData(model_value)
            if model_index < 0:
                self._usegment3d_cellpose_model_combo.addItem(model_value, model_value)
                model_index = self._usegment3d_cellpose_model_combo.count() - 1
            self._usegment3d_cellpose_model_combo.setCurrentIndex(model_index)
            cellpose_channels_value = (
                str(usegment3d_params.get("cellpose_channels", "grayscale"))
                .strip()
                .lower()
            )
            if cellpose_channels_value not in {"grayscale", "color"}:
                cellpose_channels_value = "grayscale"
            channels_index = self._usegment3d_cellpose_channels_input.findData(
                cellpose_channels_value
            )
            if channels_index < 0:
                channels_index = self._usegment3d_cellpose_channels_input.findData(
                    "grayscale"
                )
            if channels_index < 0:
                channels_index = 0
            self._usegment3d_cellpose_channels_input.setCurrentIndex(channels_index)
            self._usegment3d_cellpose_diameter_spin.setValue(
                float(
                    usegment3d_params.get(
                        "cellpose_best_diameter",
                        usegment3d_params.get("cellpose_diameter", 0.0),
                    )
                    or 0.0
                )
            )
            self._usegment3d_cellpose_edge_checkbox.setChecked(
                bool(
                    usegment3d_params.get(
                        "cellpose_use_edge",
                        usegment3d_params.get("cellpose_edge", True),
                    )
                )
            )
            self._usegment3d_aggregation_decay_spin.setValue(
                float(
                    usegment3d_params.get(
                        "aggregation_gradient_decay",
                        usegment3d_params.get("aggregation_decay", 0.0),
                    )
                )
            )
            self._usegment3d_aggregation_iterations_spin.setValue(
                max(
                    1,
                    int(
                        usegment3d_params.get(
                            "aggregation_n_iter",
                            usegment3d_params.get("aggregation_iterations", 200),
                        )
                    ),
                )
            )
            self._usegment3d_aggregation_momentum_spin.setValue(
                float(
                    usegment3d_params.get(
                        "aggregation_momenta",
                        usegment3d_params.get("aggregation_momentum", 0.98),
                    )
                )
            )
            tile_mode_value = (
                str(usegment3d_params.get("tile_mode", "")).strip().lower()
            )
            if tile_mode_value not in {"auto", "manual", "none"}:
                tile_mode_value = (
                    "auto"
                    if bool(usegment3d_params.get("aggregation_tile_mode", False))
                    else "none"
                )
            tile_mode_index = self._usegment3d_tile_mode_combo.findData(tile_mode_value)
            if tile_mode_index < 0:
                tile_mode_index = self._usegment3d_tile_mode_combo.findData("auto")
            if tile_mode_index < 0:
                tile_mode_index = 0
            self._usegment3d_tile_mode_combo.setCurrentIndex(tile_mode_index)
            tile_shape = usegment3d_params.get(
                "aggregation_tile_shape_zyx",
                usegment3d_params.get(
                    "tile_shape_zyx",
                    usegment3d_params.get("tile_shape", [128, 256, 256]),
                ),
            )
            if not isinstance(tile_shape, (tuple, list)) or len(tile_shape) != 3:
                tile_shape = [128, 256, 256]
            tile_shape_values = [
                max(1, int(tile_shape[0])),
                max(1, int(tile_shape[1])),
                max(1, int(tile_shape[2])),
            ]
            self._usegment3d_tile_z_spin.setValue(tile_shape_values[0])
            self._usegment3d_tile_y_spin.setValue(tile_shape_values[1])
            self._usegment3d_tile_x_spin.setValue(tile_shape_values[2])

            tile_overlap_ratio = float(
                usegment3d_params.get("aggregation_tile_overlap_ratio", 0.25)
            )
            tile_overlap_ratio = max(0.0, min(tile_overlap_ratio, 0.99))
            self._usegment3d_tile_overlap_z_spin.setValue(
                max(0, int(round(tile_shape_values[0] * tile_overlap_ratio)))
            )
            self._usegment3d_tile_overlap_y_spin.setValue(
                max(0, int(round(tile_shape_values[1] * tile_overlap_ratio)))
            )
            self._usegment3d_tile_overlap_x_spin.setValue(
                max(0, int(round(tile_shape_values[2] * tile_overlap_ratio)))
            )

            postprocess_enabled = bool(
                usegment3d_params.get(
                    "postprocess_enable",
                    usegment3d_params.get(
                        "postprocess",
                        usegment3d_params.get("postprocess_enabled", True),
                    ),
                )
            )
            self._usegment3d_postprocess_checkbox.setChecked(postprocess_enabled)
            self._usegment3d_postprocess_min_size_spin.setValue(
                max(0, int(usegment3d_params.get("postprocess_min_size", 200)))
            )
            self._usegment3d_postprocess_flow_threshold_spin.setValue(
                float(
                    usegment3d_params.get(
                        "postprocess_flow_threshold",
                        usegment3d_params.get("flow_threshold", 0.85),
                    )
                )
            )
            do_flow_remove = bool(
                usegment3d_params.get("postprocess_do_flow_remove", postprocess_enabled)
            )
            dtform_value = (
                str(
                    usegment3d_params.get(
                        "postprocess_dtform_method",
                        usegment3d_params.get("postprocess_dtform", "cellpose_improve"),
                    )
                )
                .strip()
                .lower()
                or "cellpose_improve"
            )
            if not do_flow_remove:
                dtform_value = "none"
            dtform_index = self._usegment3d_postprocess_dtform_combo.findData(
                dtform_value
            )
            if dtform_index < 0:
                self._usegment3d_postprocess_dtform_combo.addItem(
                    dtform_value,
                    dtform_value,
                )
                dtform_index = self._usegment3d_postprocess_dtform_combo.count() - 1
            self._usegment3d_postprocess_dtform_combo.setCurrentIndex(dtform_index)

            self._visualization_show_all_positions_checkbox.setChecked(
                bool(visualization_params.get("show_all_positions", False))
            )
            self._visualization_position_spin.setValue(
                max(
                    0,
                    min(
                        int(self._visualization_position_spin.maximum()),
                        int(visualization_params.get("position_index", 0)),
                    ),
                )
            )
            self._visualization_multiscale_checkbox.setChecked(
                bool(visualization_params.get("use_multiscale", True))
            )
            self._visualization_require_gpu_checkbox.setChecked(
                bool(visualization_params.get("require_gpu_rendering", True))
            )
            self._visualization_overlay_points_checkbox.setChecked(
                bool(visualization_params.get("overlay_particle_detections", True))
            )
            self._visualization_volume_layers = (
                self._normalize_visualization_volume_layers(
                    visualization_params.get(
                        "volume_layers",
                        self._visualization_defaults.get("volume_layers", []),
                    )
                )
            )
            if not self._visualization_volume_layers:
                self._visualization_volume_layers = (
                    self._default_visualization_volume_layers()
                )
            self._refresh_visualization_volume_layers_summary()
            self._visualization_keyframe_layer_overrides = (
                self._normalize_visualization_layer_overrides(
                    visualization_params.get(
                        "keyframe_layer_overrides",
                        self._visualization_defaults.get(
                            "keyframe_layer_overrides",
                            [],
                        ),
                    )
                )
            )
            self._refresh_visualization_layer_override_summary()
            self._set_visualization_position_selector_state()
            mip_mode = (
                str(mip_export_params.get("position_mode", "multi_position")).strip()
                or "multi_position"
            )
            mip_mode_index = self._mip_position_mode_combo.findData(mip_mode)
            if mip_mode_index < 0:
                mip_mode_index = self._mip_position_mode_combo.findData(
                    "multi_position"
                )
            if mip_mode_index < 0:
                mip_mode_index = 0
            self._mip_position_mode_combo.setCurrentIndex(mip_mode_index)

            mip_format = (
                str(mip_export_params.get("export_format", "ome-tiff")).strip()
                or "ome-tiff"
            )
            if mip_format.lower() in {"tiff", "ome_tiff", "ome.tiff"}:
                mip_format = "ome-tiff"
            mip_format_index = self._mip_export_format_combo.findData(mip_format)
            if mip_format_index < 0:
                mip_format_index = self._mip_export_format_combo.findData("ome-tiff")
            if mip_format_index < 0:
                mip_format_index = 0
            self._mip_export_format_combo.setCurrentIndex(mip_format_index)
            self._mip_output_directory_input.setText(
                str(mip_export_params.get("output_directory", "") or "").strip()
            )

            self._refresh_operation_provenance_statuses()
            self._on_operation_selection_changed()
            if self._operation_panel_stack is not None:
                self._operation_panel_stack.setCurrentIndex(0)
            for button in self._operation_config_buttons.values():
                button.setChecked(False)
            self._active_config_operation = None
            self._set_parameter_help(self._parameter_help_default)

        def _apply_theme(self) -> None:
            """Apply stylesheet-based theme for analysis window.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Styles are applied in-place.
            """
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
                    font-size: 13px;
                }
                QLabel {
                    color: #d9e2f1;
                }
                #headerCard {
                    background-color: #f0f2ef;
                    border: 1px solid #f0f2ef;
                    border-radius: 12px;
                    padding: 10px;
                }
                QLabel#brandHeaderImage {
                    padding: 0;
                    margin-bottom: 2px;
                    background: transparent;
                }
                QLabel#title {
                    font-size: 20px;
                    font-weight: 700;
                    color: #f0f5ff;
                }
                QLabel#subtitle {
                    font-size: 13px;
                    color: #a6b7d0;
                }
                QLabel#parameterHint {
                    color: #c8daf3;
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                    padding: 8px;
                }
                QFrame#helpCard {
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                    padding: 8px;
                }
                QLabel#helpTitle {
                    color: #9cc6ff;
                    font-weight: 700;
                }
                QLabel#helpBody {
                    color: #d9e2f1;
                }
                QScrollArea {
                    border: none;
                    background: transparent;
                }
                QFrame#operationSection {
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                    border-radius: 10px;
                    padding: 8px;
                }
                QLabel#operationSectionTitle {
                    color: #9cc6ff;
                    font-weight: 700;
                    font-size: 12px;
                }
                QTabWidget#analysisTabs::pane {
                    border: 1px solid #2a3442;
                    border-radius: 10px;
                    background-color: #111925;
                    top: -1px;
                }
                QTabWidget#analysisTabs QTabBar::tab {
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                    border-bottom: none;
                    padding: 8px 12px;
                    margin-right: 4px;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    color: #9ab0ca;
                }
                QTabWidget#analysisTabs QTabBar::tab:selected {
                    color: #f0f5ff;
                    background-color: #182637;
                    border-color: #2f81f7;
                }
                QGroupBox {
                    border: 1px solid #2a3442;
                    border-radius: 10px;
                    margin-top: 16px;
                    padding: 14px;
                    background-color: #111925;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 6px;
                    color: #9cc6ff;
                }
                QLabel#metadataFieldLabel {
                    color: #9cc6ff;
                    font-weight: 600;
                }
                QLabel#metadataFieldValue {
                    color: #d9e2f1;
                }
                QCheckBox {
                    spacing: 8px;
                    color: #d9e2f1;
                }
                QLabel#statusLabel {
                    color: #9ab0ca;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    min-height: 30px;
                    border: 1px solid #2b3f58;
                    border-radius: 8px;
                    background-color: #101a29;
                    color: #e6edf3;
                    padding: 5px 10px;
                    selection-background-color: #2f81f7;
                    selection-color: #f8fbff;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                    border-color: #2f81f7;
                }
                QComboBox QAbstractItemView {
                    background-color: #101a29;
                    color: #e6edf3;
                    selection-background-color: #2f81f7;
                    selection-color: #f8fbff;
                    border: 1px solid #2b3f58;
                    outline: 0;
                }
                QTableWidget {
                    background-color: #101a29;
                    color: #e6edf3;
                    border: 1px solid #2b3f58;
                    border-radius: 8px;
                    gridline-color: #2a3442;
                    selection-background-color: #2f81f7;
                    selection-color: #f8fbff;
                }
                QTableWidget::item {
                    color: #e6edf3;
                }
                QHeaderView::section {
                    background-color: #162538;
                    color: #9cc6ff;
                    border: 1px solid #2b3f58;
                    padding: 6px 8px;
                    font-weight: 600;
                }
                QTableCornerButton::section {
                    background-color: #162538;
                    border: 1px solid #2b3f58;
                }
                QToolTip {
                    color: #e6edf3;
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                }
                QPushButton {
                    background-color: #1a2635;
                    border: 1px solid #2f4460;
                    border-radius: 8px;
                    padding: 9px 14px;
                    color: #dbe9ff;
                }
                QPushButton:hover {
                    background-color: #22354c;
                }
                QPushButton:pressed {
                    background-color: #182639;
                }
                QPushButton#configureButton:checked {
                    background-color: #2f81f7;
                    border-color: #2f81f7;
                    color: #f8fbff;
                }
                QPushButton#runButton {
                    background-color: #2f81f7;
                    border-color: #2f81f7;
                    color: #f8fbff;
                    font-weight: 700;
                }
                QPushButton#runButton:hover {
                    background-color: #1f6cd8;
                }
                """
            )

        @staticmethod
        def _split_csv_values(text: str) -> list[str]:
            """Split comma-separated text values.

            Parameters
            ----------
            text : str
                Input CSV-like text.

            Returns
            -------
            list[str]
                Non-empty stripped tokens.
            """
            return [
                token.strip()
                for token in str(text).replace("\n", ",").split(",")
                if token.strip()
            ]

        def _collect_deconvolution_parameters(self) -> Dict[str, Any]:
            """Collect deconvolution parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Deconvolution parameter mapping.
            """
            psf_mode = str(self._decon_psf_mode_combo.currentData() or "measured")
            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._decon_defaults.get("memory_overhead_factor", 2.0)
                ),
                "psf_mode": psf_mode,
                "synthetic_microscopy_mode": str(
                    self._decon_synth_mode_combo.currentData() or "widefield"
                ),
                "channel_indices": [],
                "measured_psf_paths": self._split_csv_values(
                    self._decon_measured_psf_paths_input.text()
                ),
                "measured_psf_xy_um": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_measured_psf_xy_input.text()
                    )
                ],
                "measured_psf_z_um": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_measured_psf_z_input.text()
                    )
                ],
                "synthetic_excitation_nm": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_synth_excitation_input.text()
                    )
                ],
                "synthetic_illumination_wavelength_nm": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_synth_excitation_input.text()
                    )
                ],
                "synthetic_emission_nm": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_synth_emission_input.text()
                    )
                ],
                "synthetic_illumination_numerical_aperture": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_synth_illum_na_input.text()
                    )
                ],
                "synthetic_numerical_aperture": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_synth_na_input.text()
                    )
                ],
                "synthetic_detection_numerical_aperture": [
                    float(value)
                    for value in self._split_csv_values(
                        self._decon_synth_na_input.text()
                    )
                ],
                "data_xy_pixel_um": float(self._decon_data_xy_spin.value()),
                "data_z_pixel_um": float(self._decon_data_z_spin.value()),
                "hann_window_bounds": [
                    float(self._decon_hann_low_spin.value()),
                    float(self._decon_hann_high_spin.value()),
                ],
                "wiener_alpha": float(self._decon_wiener_spin.value()),
                "background": float(self._decon_background_spin.value()),
                "decon_iterations": int(self._decon_iterations_spin.value()),
                "otf_cum_thresh": float(
                    self._decon_defaults.get("otf_cum_thresh", 0.6)
                ),
                "large_file": bool(self._decon_large_file_checkbox.isChecked()),
                "block_size_zyx": [
                    int(self._decon_block_z_spin.value()),
                    int(self._decon_block_y_spin.value()),
                    int(self._decon_block_x_spin.value()),
                ],
                "batch_size_zyx": [
                    int(self._decon_batch_z_spin.value()),
                    int(self._decon_batch_y_spin.value()),
                    int(self._decon_batch_x_spin.value()),
                ],
                "save_16bit": bool(self._decon_defaults.get("save_16bit", True)),
                "save_zarr": bool(self._decon_defaults.get("save_zarr", False)),
                "gpu_job": bool(self._decon_defaults.get("gpu_job", False)),
                "debug": bool(self._decon_defaults.get("debug", False)),
                "cpus_per_task": int(self._decon_cpus_spin.value()),
                "mcc_mode": bool(self._decon_defaults.get("mcc_mode", True)),
                "config_file": str(self._decon_defaults.get("config_file", "")),
                "gpu_config_file": str(self._decon_defaults.get("gpu_config_file", "")),
            }

        def _collect_flatfield_parameters(self) -> Dict[str, Any]:
            """Collect flatfield-correction parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Flatfield-correction parameter mapping.
            """
            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": bool(
                    self._flatfield_use_overlap_checkbox.isChecked()
                ),
                "overlap_zyx": [
                    int(self._flatfield_overlap_z_spin.value()),
                    int(self._flatfield_overlap_y_spin.value()),
                    int(self._flatfield_overlap_x_spin.value()),
                ],
                "memory_overhead_factor": float(
                    self._flatfield_defaults.get("memory_overhead_factor", 2.0)
                ),
                "fit_mode": "tiled",
                "fit_tile_shape_yx": [
                    int(self._flatfield_tile_y_spin.value()),
                    int(self._flatfield_tile_x_spin.value()),
                ],
                "blend_tiles": bool(
                    self._flatfield_blend_tiles_checkbox.isChecked()
                    and self._flatfield_use_overlap_checkbox.isChecked()
                ),
                "get_darkfield": bool(self._flatfield_darkfield_checkbox.isChecked()),
                "smoothness_flatfield": float(self._flatfield_smoothness_spin.value()),
                "working_size": int(self._flatfield_working_size_spin.value()),
                "is_timelapse": bool(self._flatfield_timelapse_checkbox.isChecked()),
            }

        def _collect_shear_parameters(self) -> Dict[str, Any]:
            """Collect shear-transform parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Shear-transform parameter mapping.
            """
            interpolation = str(
                self._shear_interpolation_combo.currentData() or "linear"
            ).strip()
            output_dtype = str(
                self._shear_output_dtype_combo.currentData() or "float32"
            ).strip()
            shear_xy_deg = float(self._shear_xy_spin.value())
            shear_xz_deg = float(self._shear_xz_spin.value())
            shear_yz_deg = float(self._shear_yz_spin.value())
            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._shear_defaults.get("memory_overhead_factor", 2.0)
                ),
                "shear_xy_deg": float(shear_xy_deg),
                "shear_xz_deg": float(shear_xz_deg),
                "shear_yz_deg": float(shear_yz_deg),
                "shear_xy": _shear_degrees_to_coefficient(shear_xy_deg),
                "shear_xz": _shear_degrees_to_coefficient(shear_xz_deg),
                "shear_yz": _shear_degrees_to_coefficient(shear_yz_deg),
                "rotation_deg_x": float(self._shear_rotation_x_spin.value()),
                "rotation_deg_y": float(self._shear_rotation_y_spin.value()),
                "rotation_deg_z": float(self._shear_rotation_z_spin.value()),
                "auto_rotate_from_shear": bool(
                    self._shear_auto_rotate_checkbox.isChecked()
                ),
                "auto_estimate_shear_yz": bool(
                    self._shear_auto_estimate_checkbox.isChecked()
                ),
                "auto_estimate_extreme_fraction_x": float(
                    self._shear_auto_estimate_fraction_spin.value()
                ),
                "auto_estimate_zy_stride": int(
                    self._shear_auto_estimate_stride_spin.value()
                ),
                "auto_estimate_signal_fraction": float(
                    self._shear_auto_estimate_signal_fraction_spin.value()
                ),
                "auto_estimate_t_index": int(self._shear_auto_estimate_t_spin.value()),
                "auto_estimate_p_index": int(self._shear_auto_estimate_p_spin.value()),
                "auto_estimate_c_index": int(self._shear_auto_estimate_c_spin.value()),
                "interpolation": interpolation.lower() or "linear",
                "fill_value": float(self._shear_fill_value_spin.value()),
                "output_dtype": output_dtype.lower() or "float32",
                "roi_padding_zyx": [
                    int(self._shear_padding_z_spin.value()),
                    int(self._shear_padding_y_spin.value()),
                    int(self._shear_padding_x_spin.value()),
                ],
            }

        def _collect_particle_parameters(self) -> Dict[str, Any]:
            """Collect particle-detection parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Particle-detection parameter mapping.
            """
            return {
                "channel_index": int(self._particle_channel_spin.value()),
                "chunk_basis": "3d",
                "detect_2d_per_slice": True,
                "use_map_overlap": bool(
                    self._particle_use_overlap_checkbox.isChecked()
                ),
                "overlap_zyx": [
                    int(self._particle_overlap_z_spin.value()),
                    int(self._particle_overlap_y_spin.value()),
                    int(self._particle_overlap_x_spin.value()),
                ],
                "memory_overhead_factor": float(
                    self._particle_defaults.get("memory_overhead_factor", 1.5)
                ),
                "bg_sigma": float(self._particle_bg_sigma_spin.value()),
                "fwhm_px": float(self._particle_fwhm_spin.value()),
                "sigma_min_factor": float(self._particle_sigma_min_spin.value()),
                "sigma_max_factor": float(self._particle_sigma_max_spin.value()),
                "threshold": float(self._particle_threshold_spin.value()),
                "overlap": float(self._particle_overlap_spin.value()),
                "exclude_border": int(self._particle_exclude_border_spin.value()),
                "eliminate_insignificant_particles": bool(
                    self._particle_eliminate_checkbox.isChecked()
                ),
                "remove_close_particles": bool(
                    self._particle_remove_close_checkbox.isChecked()
                ),
                "min_distance_sigma": float(self._particle_min_distance_spin.value()),
            }

        def _collect_usegment3d_parameters(self) -> Dict[str, Any]:
            """Collect uSegment3D parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                uSegment3D parameter mapping.

            Raises
            ------
            ValueError
                If required parameter values are invalid.
            """
            views = self._split_csv_values(self._usegment3d_views_input.text())
            if not views:
                default_views = self._usegment3d_defaults.get(
                    "use_views", ["xy", "xz", "yz"]
                )
                if isinstance(default_views, (tuple, list)):
                    views = [
                        str(item).strip() for item in default_views if str(item).strip()
                    ]
                else:
                    default_text = str(default_views or "").strip()
                    views = [default_text] if default_text else ["xy", "xz", "yz"]
            seen_views: set[str] = set()
            normalized_views: list[str] = []
            for item in views:
                token = str(item).strip().lower()
                if token not in {"xy", "xz", "yz"} or token in seen_views:
                    continue
                normalized_views.append(token)
                seen_views.add(token)
            if not normalized_views:
                raise ValueError(
                    "uSegment3D views must include at least one of xy, xz, yz."
                )

            selected_channel_indices = self._selected_usegment3d_channel_indices()
            channel = int(selected_channel_indices[0])
            input_resolution_level = int(self._usegment3d_resolution_level_spin.value())
            output_reference_space = (
                str(self._usegment3d_output_reference_combo.currentData() or "level0")
                .strip()
                .lower()
                or "level0"
            )
            if input_resolution_level <= 0:
                output_reference_space = "level0"
            save_native_labels = bool(
                self._usegment3d_save_native_labels_checkbox.isChecked()
                and input_resolution_level > 0
                and output_reference_space == "level0"
            )
            gpu_enabled = bool(
                self._usegment3d_gpu_checkbox.isChecked()
                or self._usegment3d_require_gpu_checkbox.isChecked()
            )
            postprocess_enabled = bool(
                self._usegment3d_postprocess_checkbox.isChecked()
            )
            tile_mode = str(self._usegment3d_tile_mode_combo.currentData() or "auto")
            tile_mode = tile_mode.strip().lower() or "auto"
            tile_shape_zyx = [
                int(self._usegment3d_tile_z_spin.value()),
                int(self._usegment3d_tile_y_spin.value()),
                int(self._usegment3d_tile_x_spin.value()),
            ]
            tile_overlap_zyx = [
                int(self._usegment3d_tile_overlap_z_spin.value()),
                int(self._usegment3d_tile_overlap_y_spin.value()),
                int(self._usegment3d_tile_overlap_x_spin.value()),
            ]
            tile_ratio_candidates = [
                (
                    float(max(0, tile_overlap_zyx[idx]))
                    / float(max(1, tile_shape_zyx[idx]))
                )
                for idx in range(3)
            ]
            tile_overlap_ratio = (
                max(tile_ratio_candidates) if tile_ratio_candidates else 0.0
            )
            tile_overlap_ratio = max(0.0, min(tile_overlap_ratio, 0.99))
            flow_threshold = float(
                self._usegment3d_postprocess_flow_threshold_spin.value()
            )
            dtform_value = str(
                self._usegment3d_postprocess_dtform_combo.currentData()
                or "cellpose_improve"
            ).strip()
            dtform_value = dtform_value.lower() or "cellpose_improve"
            do_flow_remove = bool(postprocess_enabled)
            dtform_method = dtform_value
            if dtform_value == "none":
                do_flow_remove = False
                dtform_method = "edt"

            best_diameter = float(self._usegment3d_cellpose_diameter_spin.value())
            if best_diameter <= 0:
                best_diameter_value: Optional[float] = None
                use_auto_diameter = True
            else:
                best_diameter_value = best_diameter
                use_auto_diameter = False

            return {
                "channel_indices": [int(value) for value in selected_channel_indices],
                "channel_index": channel,
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": False,
                "overlap_zyx": list(
                    self._usegment3d_defaults.get("overlap_zyx", [0, 0, 0])
                ),
                "memory_overhead_factor": float(
                    self._usegment3d_defaults.get("memory_overhead_factor", 3.0)
                ),
                "use_views": list(normalized_views),
                "input_resolution_level": int(input_resolution_level),
                "output_reference_space": str(output_reference_space),
                "save_native_labels": bool(save_native_labels),
                "gpu": gpu_enabled,
                "require_gpu": bool(self._usegment3d_require_gpu_checkbox.isChecked()),
                "preprocess_normalize_min": float(
                    self._usegment3d_preprocess_min_spin.value()
                ),
                "preprocess_normalize_max": float(
                    self._usegment3d_preprocess_max_spin.value()
                ),
                "preprocess_do_bg_correction": bool(
                    self._usegment3d_preprocess_bg_correction_checkbox.isChecked()
                ),
                "preprocess_factor": float(
                    self._usegment3d_preprocess_factor_spin.value()
                ),
                "cellpose_model_name": str(
                    self._usegment3d_cellpose_model_combo.currentData() or "cyto"
                ).strip()
                or "cyto",
                "cellpose_channels": str(
                    self._usegment3d_cellpose_channels_input.currentData()
                    or "grayscale"
                ).strip()
                or "grayscale",
                "cellpose_best_diameter": best_diameter_value,
                "cellpose_use_auto_diameter": use_auto_diameter,
                "cellpose_use_edge": bool(
                    self._usegment3d_cellpose_edge_checkbox.isChecked()
                ),
                "aggregation_gradient_decay": float(
                    self._usegment3d_aggregation_decay_spin.value()
                ),
                "aggregation_n_iter": int(
                    self._usegment3d_aggregation_iterations_spin.value()
                ),
                "aggregation_momenta": float(
                    self._usegment3d_aggregation_momentum_spin.value()
                ),
                "tile_mode": tile_mode,
                "aggregation_tile_mode": tile_mode != "none",
                "aggregation_tile_shape_zyx": list(tile_shape_zyx),
                "aggregation_tile_overlap_ratio": float(tile_overlap_ratio),
                "postprocess_enable": postprocess_enabled,
                "postprocess_min_size": int(
                    self._usegment3d_postprocess_min_size_spin.value()
                ),
                "postprocess_do_flow_remove": do_flow_remove,
                "postprocess_flow_threshold": flow_threshold,
                "postprocess_dtform_method": dtform_method,
            }

        def _collect_visualization_parameters(self) -> Dict[str, Any]:
            """Collect visualization parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Visualization parameter mapping.
            """
            overlay_enabled = self._is_particle_overlay_available()
            volume_layers = self._normalize_visualization_volume_layers(
                self._visualization_volume_layers
            )
            if not volume_layers:
                volume_layers = self._default_visualization_volume_layers()
            primary_component = (
                str(volume_layers[0].get("component", "data")).strip() or "data"
            )
            return {
                "input_source": primary_component,
                "chunk_basis": "2d",
                "detect_2d_per_slice": True,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._visualization_defaults.get("memory_overhead_factor", 1.0)
                ),
                "show_all_positions": bool(
                    self._visualization_show_all_positions_checkbox.isChecked()
                ),
                "position_index": int(self._visualization_position_spin.value()),
                "use_multiscale": bool(
                    self._visualization_multiscale_checkbox.isChecked()
                ),
                "require_gpu_rendering": bool(
                    self._visualization_require_gpu_checkbox.isChecked()
                ),
                "overlay_particle_detections": bool(
                    self._visualization_overlay_points_checkbox.isChecked()
                )
                and overlay_enabled,
                "particle_detection_component": str(
                    self._visualization_defaults.get(
                        "particle_detection_component",
                        self._PARTICLE_DETECTION_OVERLAY_COMPONENT,
                    )
                ),
                "launch_mode": str(
                    self._visualization_defaults.get("launch_mode", "auto")
                ),
                "capture_keyframes": bool(
                    self._visualization_defaults.get("capture_keyframes", True)
                ),
                "keyframe_manifest_path": str(
                    self._visualization_defaults.get("keyframe_manifest_path", "") or ""
                ).strip(),
                "keyframe_layer_overrides": list(
                    self._visualization_keyframe_layer_overrides
                ),
                "volume_layers": list(volume_layers),
            }

        def _collect_mip_export_parameters(self) -> Dict[str, Any]:
            """Collect MIP-export parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                MIP-export parameter mapping.
            """
            position_mode = str(
                self._mip_position_mode_combo.currentData() or "multi_position"
            ).strip()
            export_format = str(
                self._mip_export_format_combo.currentData() or "ome-tiff"
            ).strip()
            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._mip_export_defaults.get("memory_overhead_factor", 1.0)
                ),
                "position_mode": position_mode.lower() or "multi_position",
                "export_format": export_format.lower() or "ome-tiff",
                "output_directory": str(
                    self._mip_output_directory_input.text()
                ).strip(),
            }

        def _collect_operation_parameters(self, operation_name: str) -> Dict[str, Any]:
            """Collect parameter mapping for one operation.

            Parameters
            ----------
            operation_name : str
                Operation key.

            Returns
            -------
            dict[str, Any]
                Collected operation parameter mapping.
            """
            defaults = dict(self._operation_defaults.get(operation_name, {}))
            defaults["execution_order"] = int(
                self._operation_order_spins[operation_name].value()
            )
            combo_value = self._operation_input_combos[operation_name].currentData()
            defaults["input_source"] = (
                str(combo_value).strip() if combo_value is not None else "data"
            ) or "data"
            force_checkbox = self._operation_force_rerun_checkboxes.get(operation_name)
            defaults["force_rerun"] = (
                bool(force_checkbox.isChecked())
                if force_checkbox is not None
                else False
            )
            if operation_name == "flatfield":
                defaults.update(self._collect_flatfield_parameters())
            elif operation_name == "deconvolution":
                defaults.update(self._collect_deconvolution_parameters())
            elif operation_name == "shear_transform":
                defaults.update(self._collect_shear_parameters())
            elif operation_name == "particle_detection":
                defaults.update(self._collect_particle_parameters())
            elif operation_name == "usegment3d":
                defaults.update(self._collect_usegment3d_parameters())
            elif operation_name == "visualization":
                defaults.update(self._collect_visualization_parameters())
            elif operation_name == "mip_export":
                defaults.update(self._collect_mip_export_parameters())
            return defaults

        def _on_run(self) -> None:
            """Finalize selected analysis flags and close dialog.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Stores selected workflow and accepts this dialog.
            """
            self._refresh_operation_provenance_statuses()
            selected_flags = {
                operation_name: self._operation_checkboxes[operation_name].isChecked()
                for operation_name in self._OPERATION_KEYS
            }
            if not any(selected_flags.values()):
                QMessageBox.warning(
                    self,
                    "No Analysis Selected",
                    "Select at least one analysis routine before running.",
                )
                self._set_status("Select at least one analysis routine.")
                return

            analysis_parameters = normalize_analysis_operation_parameters(
                self._base_config.analysis_parameters
            )
            for operation_name in self._OPERATION_KEYS:
                try:
                    analysis_parameters[operation_name] = (
                        self._collect_operation_parameters(operation_name)
                    )
                except ValueError as exc:
                    QMessageBox.warning(
                        self,
                        "Invalid Operation Parameters",
                        f"Invalid values for {self._OPERATION_LABELS[operation_name]}.\n\n{exc}",
                    )
                    self._set_status(
                        f"Invalid {self._OPERATION_LABELS[operation_name]} parameters."
                    )
                    return
            analysis_parameters = normalize_analysis_operation_parameters(
                analysis_parameters
            )

            if selected_flags["deconvolution"]:
                decon_params = dict(analysis_parameters.get("deconvolution", {}))
                psf_mode = str(decon_params.get("psf_mode", "measured"))
                if psf_mode == "measured":
                    if not decon_params.get("measured_psf_paths"):
                        QMessageBox.warning(
                            self,
                            "Missing PSF Paths",
                            "Measured PSF mode requires at least one PSF path.",
                        )
                        self._set_status(
                            "Provide at least one measured PSF path for deconvolution."
                        )
                        return
                    if not decon_params.get(
                        "measured_psf_xy_um"
                    ) or not decon_params.get("measured_psf_z_um"):
                        QMessageBox.warning(
                            self,
                            "Missing PSF Pixel Size",
                            "Measured PSF mode requires PSF XY and Z pixel sizes.",
                        )
                        self._set_status(
                            "Provide PSF XY and Z pixel sizes for measured PSF mode."
                        )
                        return
                else:
                    microscopy_mode = str(
                        decon_params.get("synthetic_microscopy_mode", "widefield")
                    ).strip()
                    if not decon_params.get(
                        "synthetic_emission_nm"
                    ) or not decon_params.get("synthetic_detection_numerical_aperture"):
                        QMessageBox.warning(
                            self,
                            "Missing Synthetic PSF Parameters",
                            "Synthetic PSF mode requires detection emission and detection NA values.",
                        )
                        self._set_status(
                            "Provide synthetic detection emission and NA values."
                        )
                        return
                    if microscopy_mode == "light_sheet" and (
                        not decon_params.get("synthetic_illumination_wavelength_nm")
                        or not decon_params.get(
                            "synthetic_illumination_numerical_aperture"
                        )
                    ):
                        QMessageBox.warning(
                            self,
                            "Missing Light-Sheet Parameters",
                            "Light-sheet synthetic PSF mode requires illumination "
                            "wavelength and illumination NA values.",
                        )
                        self._set_status(
                            "Provide light-sheet illumination wavelength and NA values."
                        )
                        return

            workflow_kwargs: Dict[str, Any] = {
                "file": self._base_config.file,
                "prefer_dask": self._base_config.prefer_dask,
                "dask_backend": self._dask_backend_config,
                "chunks": self._base_config.chunks,
                "flatfield": selected_flags["flatfield"],
                "deconvolution": selected_flags["deconvolution"],
                "shear_transform": selected_flags["shear_transform"],
                "particle_detection": selected_flags["particle_detection"],
                "registration": selected_flags["registration"],
                "visualization": selected_flags["visualization"],
                "mip_export": selected_flags["mip_export"],
                "zarr_save": self._base_config.zarr_save,
                "analysis_parameters": analysis_parameters,
            }
            dataclass_fields = getattr(WorkflowConfig, "__dataclass_fields__", {})
            if "usegment3d" in dataclass_fields:
                workflow_kwargs["usegment3d"] = selected_flags["usegment3d"]
            self.result_config = WorkflowConfig(**workflow_kwargs)
            _save_last_used_dask_backend_config(self._dask_backend_config)
            sequence = self._selected_operations_in_sequence()
            sequence_text = " -> ".join(
                self._OPERATION_LABELS[name] for name in sequence
            )
            self._set_status(f"Launching selected analysis routines: {sequence_text}")
            self.accept()


def launch_gui(
    initial: Optional[WorkflowConfig] = None,
    run_callback: Optional[
        Callable[[WorkflowConfig, Callable[[int, str], None]], None]
    ] = None,
) -> Optional[WorkflowConfig]:
    """Launch the PyQt GUI for workflow selection/execution.

    Parameters
    ----------
    initial : WorkflowConfig, optional
        Initial workflow values to pre-populate GUI controls.
    run_callback : callable, optional
        Optional execution callback. When provided, the GUI runs in iterative
        mode: after each analysis run, the analysis-selection dialog is shown
        again so the user can select the next task. Signature must be
        ``(workflow, progress_callback)``.

    Returns
    -------
    WorkflowConfig, optional
        In single-run mode (no ``run_callback``), returns the selected workflow.
        In iterative mode, returns ``None`` when the user explicitly exits.

    Raises
    ------
    GuiUnavailableError
        If PyQt6 is not installed or no display server is available.

    Notes
    -----
    The launch path ensures ``~/.clearex`` exists and attempts to pre-populate
    GUI backend controls from the last persisted backend settings JSON.
    """
    if not HAS_PYQT6:
        raise GuiUnavailableError(
            "PyQt6 is not installed. Install it with `pip install PyQt6` or use `--headless`."
        )
    if not _display_is_available():
        raise GuiUnavailableError(
            "No display detected. Use `--headless` or run with a graphical display."
        )

    settings_directory = _ensure_clearex_settings_directory()
    settings_path = _resolve_dask_backend_settings_path(settings_directory)
    effective_initial = initial or WorkflowConfig()
    persisted_backend = _load_last_used_dask_backend_config(settings_path=settings_path)
    if persisted_backend is not None and _should_apply_persisted_dask_backend(initial):
        effective_initial = replace(effective_initial, dask_backend=persisted_backend)

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    _apply_application_icon(app)

    _show_startup_splash(app)

    setup_dialog = ClearExSetupDialog(initial=effective_initial)
    setup_result = setup_dialog.exec()
    if (
        setup_result != QDialog.DialogCode.Accepted
        or setup_dialog.result_config is None
    ):
        selected = None
    else:
        if run_callback is None:
            analysis_dialog = AnalysisSelectionDialog(
                initial=setup_dialog.result_config
            )
            analysis_result = analysis_dialog.exec()
            selected = (
                analysis_dialog.result_config
                if analysis_result == QDialog.DialogCode.Accepted
                else None
            )
        else:
            current = setup_dialog.result_config
            selected = None
            while True:
                analysis_dialog = AnalysisSelectionDialog(initial=current)
                analysis_result = analysis_dialog.exec()
                if (
                    analysis_result != QDialog.DialogCode.Accepted
                    or analysis_dialog.result_config is None
                ):
                    selected = None
                    break
                current = analysis_dialog.result_config
                completed = run_workflow_with_progress(
                    workflow=current,
                    run_callback=run_callback,
                )
                if not completed:
                    continue
                current = _reset_analysis_selection_for_next_run(current)

    if owns_app:
        app.quit()
    return selected


def _reset_analysis_selection_for_next_run(workflow: WorkflowConfig) -> WorkflowConfig:
    """Return a workflow copy with analysis selections reset for next GUI pass.

    Parameters
    ----------
    workflow : WorkflowConfig
        Most-recently executed workflow configuration.

    Returns
    -------
    WorkflowConfig
        Updated workflow with all analysis flags disabled and ``force_rerun``
        reset to ``False`` for provenance-aware operations.
    """
    analysis_parameters = normalize_analysis_operation_parameters(
        workflow.analysis_parameters
    )
    for operation_name in (
        "flatfield",
        "deconvolution",
        "shear_transform",
        "particle_detection",
        "usegment3d",
        "registration",
        "mip_export",
    ):
        params = dict(analysis_parameters.get(operation_name, {}))
        params["force_rerun"] = False
        analysis_parameters[operation_name] = params

    workflow_kwargs: Dict[str, Any] = {
        "file": workflow.file,
        "prefer_dask": workflow.prefer_dask,
        "dask_backend": workflow.dask_backend,
        "chunks": workflow.chunks,
        "flatfield": False,
        "deconvolution": False,
        "shear_transform": False,
        "particle_detection": False,
        "registration": False,
        "visualization": False,
        "mip_export": False,
        "zarr_save": workflow.zarr_save,
        "analysis_parameters": analysis_parameters,
    }
    dataclass_fields = getattr(WorkflowConfig, "__dataclass_fields__", {})
    if "usegment3d" in dataclass_fields:
        workflow_kwargs["usegment3d"] = False
    return WorkflowConfig(**workflow_kwargs)


def run_workflow_with_progress(
    *,
    workflow: WorkflowConfig,
    run_callback: Callable[[WorkflowConfig, Callable[[int, str], None]], None],
) -> bool:
    """Execute a workflow while showing a modal GUI progress dialog.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration to execute.
    run_callback : callable
        Callback that performs workflow execution. Signature must be
        ``(workflow, progress_callback)``.

    Returns
    -------
    bool
        ``True`` when execution succeeds, ``False`` when it fails.

    Raises
    ------
    GuiUnavailableError
        If PyQt6 or a display server is unavailable.

    Notes
    -----
    ``SLURMCluster``/``SLURMRunner`` backend startup can require Python signal
    handlers, which must be installed from the main interpreter thread.
    For these backend modes, execution is run synchronously on the GUI thread
    with explicit ``processEvents`` updates.
    """
    if not HAS_PYQT6:
        raise GuiUnavailableError(
            "PyQt6 is not installed. Install it with `pip install PyQt6` or use `--headless`."
        )
    if not _display_is_available():
        raise GuiUnavailableError(
            "No display detected. Use `--headless` or run with a graphical display."
        )

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    _apply_application_icon(app)

    progress_dialog = AnalysisExecutionProgressDialog(parent=None)
    failure_payload: dict[str, str] = {}
    completed = {"ok": False}
    use_main_thread_execution = workflow.dask_backend.mode in {
        DASK_BACKEND_SLURM_RUNNER,
        DASK_BACKEND_SLURM_CLUSTER,
    }
    if use_main_thread_execution:
        progress_dialog.show()
        app.processEvents()

        def _progress_with_events(percent: int, message: str) -> None:
            """Update modal progress UI and flush GUI events.

            Parameters
            ----------
            percent : int
                Progress percentage.
            message : str
                Human-readable status text.

            Returns
            -------
            None
                Progress UI side effects only.
            """
            progress_dialog.update_progress(int(percent), str(message))
            app.processEvents()

        try:
            _progress_with_events(1, "Starting analysis workflow...")
            run_callback(workflow, _progress_with_events)
            _progress_with_events(100, "Analysis workflow completed.")
            completed["ok"] = True
            progress_dialog.accept()
        except Exception as exc:
            failure_payload.update(
                {
                    "summary": f"{type(exc).__name__}: {exc}",
                    "details": traceback.format_exc(),
                }
            )
            progress_dialog.reject()
        finally:
            progress_dialog.close()
    else:
        worker = AnalysisExecutionWorker(
            workflow=workflow,
            run_callback=run_callback,
        )
        worker.progress_changed.connect(progress_dialog.update_progress)
        worker.succeeded.connect(lambda: completed.__setitem__("ok", True))
        worker.succeeded.connect(progress_dialog.accept)
        worker.failed.connect(
            lambda summary, details: failure_payload.update(
                {"summary": str(summary), "details": str(details)}
            )
        )
        worker.failed.connect(lambda *_: progress_dialog.reject())

        worker.start()
        progress_dialog.exec()
        worker.wait()

    if failure_payload:
        _show_themed_error_dialog(
            progress_dialog,
            "Analysis Failed",
            "Analysis execution failed.",
            summary=failure_payload.get("summary"),
            details=failure_payload.get("details"),
        )
        if owns_app:
            app.quit()
        return False

    if owns_app:
        app.quit()
    return bool(completed["ok"])
