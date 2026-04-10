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

from __future__ import annotations

# Standard Library Imports
from contextlib import ExitStack
from dataclasses import dataclass, replace
import json
import logging
import math
import os
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

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
from .dask_dashboard_proxy import DashboardRelayManager
from clearex.io.ome_store import (
    analysis_auxiliary_root,
    resolve_voxel_size_um_zyx_with_source,
    source_cache_component,
)
from clearex.io.experiment import (
    ExperimentDataResolutionError,
    NavigateExperiment,
    create_dask_client,
    has_complete_canonical_data_store,
    infer_zyx_shape,
    is_navigate_experiment_file,
    load_navigate_experiment,
    load_navigate_experiment_source_image_info,
    load_store_spatial_calibration,
    materialize_experiment_data_store,
    resolve_data_store_path,
    resolve_experiment_data_path,
    save_store_spatial_calibration,
)
from clearex.io.provenance import (
    is_zarr_store_path,
    load_latest_analysis_gui_state,
    load_latest_completed_workflow_state,
    persist_latest_analysis_gui_state,
    summarize_analysis_history,
)
from clearex.io.read import ImageInfo, ImageOpener
from clearex.workflow import (
    ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS,
    AnalysisInputDependencyIssue,
    AnalysisTarget,
    DASK_BACKEND_LOCAL_CLUSTER,
    DASK_BACKEND_MODE_LABELS,
    DASK_BACKEND_SLURM_CLUSTER,
    DASK_BACKEND_SLURM_RUNNER,
    AnalysisInputReference,
    DEFAULT_ZARR_CHUNKS_PTCZYX,
    DEFAULT_ZARR_PYRAMID_PTCZYX,
    DEFAULT_SLURM_CLUSTER_JOB_EXTRA_DIRECTIVES,
    DaskBackendConfig,
    LocalClusterRecommendation,
    LocalClusterConfig,
    PTCZYX_AXES,
    SPATIAL_CALIBRATION_WORLD_AXES,
    SpatialCalibrationConfig,
    SlurmClusterConfig,
    SlurmRunnerConfig,
    WorkflowConfig,
    WorkflowExecutionCancelled,
    ZarrSaveConfig,
    analysis_chainable_output_component,
    analysis_operation_for_output_component,
    collect_analysis_input_references,
    dask_backend_from_dict,
    dask_backend_to_dict,
    default_analysis_operation_parameters,
    format_dask_backend_summary,
    format_spatial_calibration,
    format_local_cluster_recommendation_summary,
    format_pyramid_levels,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    normalize_spatial_calibration,
    normalize_analysis_operation_parameters,
    parse_pyramid_levels,
    recommend_local_cluster_config,
    resolve_analysis_input_component,
    spatial_calibration_to_dict,
    validate_analysis_input_references,
    zarr_save_from_dict,
    zarr_save_to_dict,
)

# Third Party Imports
import zarr

try:
    from PyQt6.QtCore import (
        QEvent,
        QEventLoop,
        QMimeData,
        QObject,
        QPoint,
        QSize,
        QThread,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QCloseEvent,
        QDragEnterEvent,
        QDragMoveEvent,
        QDropEvent,
        QIcon,
        QImage,
        QImageReader,
        QKeyEvent,
        QPixmap,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView,
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
        QListWidget,
        QListWidgetItem,
        QMenu,
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
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


DaskClientLifecycleCallback = Callable[[str, str, str, object], None]


class GuiRunCallback(Protocol):
    def __call__(
        self,
        workflow: WorkflowConfig,
        progress_callback: Callable[[int, str], None],
        dask_client_lifecycle_callback: Optional[DaskClientLifecycleCallback] = None,
    ) -> None: ...


class GuiUnavailableError(RuntimeError):
    """Raised when the PyQt GUI cannot be launched in this environment."""


_GUI_ASSET_DIRECTORY = Path(__file__).resolve().parent
_GUI_SPLASH_IMAGE = "ClearEx_full.png"
_GUI_HEADER_IMAGE = "ClearEx_full_header.png"
_GUI_APP_ICON = "icon.png"
_CLEAREX_SETTINGS_DIR_NAME = ".clearex"
_CLEAREX_DASK_BACKEND_SETTINGS_FILE = "dask_backend_settings.json"
_CLEAREX_ZARR_SAVE_SETTINGS_FILE = "zarr_save_settings.json"
_CLEAREX_EXPERIMENT_LIST_FORMAT = "clearex-experiment-list/v1"
_CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX = ".clearex-experiment-list.json"
_SETUP_DIALOG_MINIMUM_SIZE = (1240, 920)
_SETUP_DIALOG_PREFERRED_SIZE = (1520, 1120)
_SPATIAL_CALIBRATION_DIALOG_MINIMUM_SIZE = (620, 420)
_SPATIAL_CALIBRATION_DIALOG_PREFERRED_SIZE = (720, 520)
_ZARR_SAVE_DIALOG_MINIMUM_SIZE = (860, 660)
_ZARR_SAVE_DIALOG_PREFERRED_SIZE = (940, 760)
_DASK_BACKEND_DIALOG_MINIMUM_SIZE = (940, 840)
_DASK_BACKEND_DIALOG_PREFERRED_SIZE = (1120, 980)
_MATERIALIZATION_PROGRESS_DIALOG_MINIMUM_SIZE = (620, 180)
_MATERIALIZATION_PROGRESS_DIALOG_PREFERRED_SIZE = (720, 220)
_ANALYSIS_PROGRESS_DIALOG_MINIMUM_SIZE = (640, 180)
_ANALYSIS_PROGRESS_DIALOG_PREFERRED_SIZE = (720, 220)
_ANALYSIS_DIALOG_MINIMUM_SIZE = (1440, 860)
_ANALYSIS_DIALOG_PREFERRED_SIZE = (1520, 940)
_VOLUME_LAYERS_DIALOG_MINIMUM_SIZE = (1240, 460)
_VOLUME_LAYERS_DIALOG_PREFERRED_SIZE = (1460, 720)
_LAYER_OVERRIDES_DIALOG_MINIMUM_SIZE = (900, 420)
_LAYER_OVERRIDES_DIALOG_PREFERRED_SIZE = (1080, 620)
_SYNTHETIC_PSF_PREVIEW_DIALOG_MINIMUM_SIZE = (840, 700)
_SYNTHETIC_PSF_PREVIEW_DIALOG_PREFERRED_SIZE = (980, 820)
_DIALOG_SCREEN_MARGIN_PX = 72


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


def _apply_application_icon(app: Any) -> None:
    """Fallback no-op when PyQt6 is unavailable at import time.

    Parameters
    ----------
    app : Any
        Placeholder application object.

    Returns
    -------
    None
        No-op fallback used until the PyQt implementation overrides it.
    """
    del app


def _show_startup_splash(app: Any) -> None:
    """Fallback no-op when PyQt6 is unavailable at import time.

    Parameters
    ----------
    app : Any
        Placeholder application object.

    Returns
    -------
    None
        No-op fallback used until the PyQt implementation overrides it.
    """
    del app


def _primary_screen_available_size() -> Optional[tuple[int, int]]:
    """Return the available size of the primary screen for the active app.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[int, int], optional
        Width and height of the current primary screen's available geometry.

    Notes
    -----
    Returns ``None`` when Qt is unavailable or when there is no active
    application/screen context.
    """
    if not HAS_PYQT6:
        return None

    app = QApplication.instance()
    if app is None:
        return None

    primary_screen_getter = getattr(app, "primaryScreen", None)
    if not callable(primary_screen_getter):
        return None

    screen = primary_screen_getter()
    if screen is None:
        return None

    geometry = screen.availableGeometry()
    return (int(geometry.width()), int(geometry.height()))


def _resolve_initial_dialog_dimensions(
    minimum_size: tuple[int, int],
    preferred_size: tuple[int, int],
    available_size: Optional[tuple[int, int]] = None,
    margin_px: int = _DIALOG_SCREEN_MARGIN_PX,
    content_size_hint: Optional[tuple[int, int]] = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve minimum and startup dialog sizes for the current screen.

    Parameters
    ----------
    minimum_size : tuple[int, int]
        Preferred minimum width/height for the dialog.
    preferred_size : tuple[int, int]
        Target startup width/height for the dialog.
    available_size : tuple[int, int], optional
        Current screen available width/height. When omitted, the provided
        minimum and preferred sizes are returned unchanged.
    margin_px : int, default=_DIALOG_SCREEN_MARGIN_PX
        Margin reserved around the dialog so it does not open flush against the
        screen edges.
    content_size_hint : tuple[int, int], optional
        Actual built-content width/height requirement. When provided, the
        minimum and preferred sizes are expanded to fit the content before
        screen clamping is applied.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        Resolved ``(minimum_width, minimum_height)`` and startup
        ``(width, height)``.
    """
    minimum_width, minimum_height = (int(minimum_size[0]), int(minimum_size[1]))
    preferred_width, preferred_height = (
        int(preferred_size[0]),
        int(preferred_size[1]),
    )
    if content_size_hint is not None:
        content_width = max(0, int(content_size_hint[0]))
        content_height = max(0, int(content_size_hint[1]))
        minimum_width = max(minimum_width, content_width)
        minimum_height = max(minimum_height, content_height)
        preferred_width = max(preferred_width, content_width)
        preferred_height = max(preferred_height, content_height)
    if available_size is None:
        return (minimum_width, minimum_height), (preferred_width, preferred_height)

    available_width = max(0, int(available_size[0]))
    available_height = max(0, int(available_size[1]))
    usable_width = (
        available_width if available_width <= margin_px else available_width - margin_px
    )
    usable_height = (
        available_height
        if available_height <= margin_px
        else available_height - margin_px
    )

    startup_width = (
        preferred_width if usable_width <= 0 else min(preferred_width, usable_width)
    )
    startup_height = (
        preferred_height if usable_height <= 0 else min(preferred_height, usable_height)
    )
    resolved_minimum_width = min(minimum_width, startup_width)
    resolved_minimum_height = min(minimum_height, startup_height)
    return (
        resolved_minimum_width,
        resolved_minimum_height,
    ), (
        startup_width,
        startup_height,
    )


def _apply_initial_dialog_geometry(
    dialog: "QDialog",
    *,
    minimum_size: tuple[int, int],
    preferred_size: tuple[int, int],
    available_size: Optional[tuple[int, int]] = None,
    margin_px: int = _DIALOG_SCREEN_MARGIN_PX,
    content_size_hint: Optional[tuple[int, int]] = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Apply screen-aware minimum and startup size to a dialog.

    Parameters
    ----------
    dialog : QDialog
        Dialog receiving the resolved geometry.
    minimum_size : tuple[int, int]
        Preferred minimum width/height for the dialog.
    preferred_size : tuple[int, int]
        Target startup width/height for the dialog.
    available_size : tuple[int, int], optional
        Screen-available width/height. When omitted, the active primary screen
        is queried.
    margin_px : int, default=_DIALOG_SCREEN_MARGIN_PX
        Margin reserved around the dialog so it does not open flush against the
        screen edges.
    content_size_hint : tuple[int, int], optional
        Actual built-content width/height requirement used to expand the dialog
        target size before screen clamping.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        Applied minimum and startup size tuples.
    """
    resolved_available_size = (
        _primary_screen_available_size() if available_size is None else available_size
    )
    minimum_dimensions, startup_dimensions = _resolve_initial_dialog_dimensions(
        minimum_size=minimum_size,
        preferred_size=preferred_size,
        available_size=resolved_available_size,
        margin_px=margin_px,
        content_size_hint=content_size_hint,
    )
    dialog.setMinimumWidth(minimum_dimensions[0])
    dialog.setMinimumHeight(minimum_dimensions[1])
    dialog.resize(startup_dimensions[0], startup_dimensions[1])
    return minimum_dimensions, startup_dimensions


def _deduplicate_resolved_paths(paths: Sequence[Path]) -> list[Path]:
    """Return first-seen unique resolved paths.

    Parameters
    ----------
    paths : sequence[pathlib.Path]
        Candidate paths to deduplicate.

    Returns
    -------
    list[pathlib.Path]
        Resolved paths in first-seen order with duplicates removed.
    """
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        resolved = Path(raw_path).expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(resolved)
    return deduplicated


def _discover_navigate_experiment_files(search_root: Path) -> list[Path]:
    """Recursively discover Navigate experiment descriptors in a directory.

    Parameters
    ----------
    search_root : pathlib.Path
        Directory to search recursively.

    Returns
    -------
    list[pathlib.Path]
        Sorted absolute experiment-descriptor paths.

    Raises
    ------
    FileNotFoundError
        If ``search_root`` does not exist.
    NotADirectoryError
        If ``search_root`` is not a directory.
    """
    root = Path(search_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    matches = list(root.rglob("experiment.yml")) + list(root.rglob("experiment.yaml"))
    discovered = _deduplicate_resolved_paths(matches)
    return sorted(discovered, key=lambda path: str(path).lower())


def _is_saved_experiment_list_path(path: Path) -> bool:
    """Return whether a path matches the ClearEx experiment-list format name.

    Parameters
    ----------
    path : pathlib.Path
        Candidate file path.

    Returns
    -------
    bool
        ``True`` when the filename uses the ClearEx experiment-list suffix.
    """
    return str(Path(path).name).endswith(_CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX)


def _save_experiment_list_file(
    list_path: Path,
    experiment_paths: Sequence[Path],
) -> Path:
    """Persist a sequence of experiment descriptors to a JSON list file.

    Parameters
    ----------
    list_path : pathlib.Path
        Output JSON list path.
    experiment_paths : sequence[pathlib.Path]
        Ordered experiment-descriptor paths to persist.

    Returns
    -------
    pathlib.Path
        Resolved output list path.

    Raises
    ------
    ValueError
        If no experiment paths are provided or a path is not a Navigate
        experiment descriptor.
    FileNotFoundError
        If an experiment path does not exist.
    """
    resolved_list_path = Path(list_path).expanduser().resolve()
    resolved_experiments = _deduplicate_resolved_paths(experiment_paths)
    if not resolved_experiments:
        raise ValueError("Select at least one experiment.yml file to save a list.")

    serialized_paths: list[str] = []
    base_directory = resolved_list_path.parent
    for experiment_path in resolved_experiments:
        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Experiment path does not exist: {experiment_path}"
            )
        if not is_navigate_experiment_file(experiment_path):
            raise ValueError(
                f"Expected Navigate experiment.yml/experiment.yaml, got: {experiment_path}"
            )
        relative_path = os.path.relpath(
            str(experiment_path),
            start=str(base_directory),
        )
        serialized_paths.append(str(relative_path))

    payload = {
        "format": _CLEAREX_EXPERIMENT_LIST_FORMAT,
        "experiments": serialized_paths,
    }
    resolved_list_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_list_path.write_text(
        f"{json.dumps(payload, indent=2)}\n",
        encoding="utf-8",
    )
    return resolved_list_path


def _load_experiment_list_file(list_path: Path) -> list[Path]:
    """Load a saved ClearEx experiment list from disk.

    Parameters
    ----------
    list_path : pathlib.Path
        Saved list JSON path.

    Returns
    -------
    list[pathlib.Path]
        Ordered resolved experiment-descriptor paths from the saved list.

    Raises
    ------
    FileNotFoundError
        If ``list_path`` does not exist.
    ValueError
        If the file does not contain a valid ClearEx experiment list or any
        listed path is not a Navigate experiment descriptor.
    """
    resolved_list_path = Path(list_path).expanduser().resolve()
    if not resolved_list_path.exists():
        raise FileNotFoundError(
            f"Experiment list file does not exist: {resolved_list_path}"
        )

    try:
        payload = json.loads(resolved_list_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Experiment list is not valid JSON: {resolved_list_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Experiment list must be a JSON object: {resolved_list_path}")
    if payload.get("format") != _CLEAREX_EXPERIMENT_LIST_FORMAT:
        raise ValueError(
            "Unsupported experiment list format. "
            f"Expected {_CLEAREX_EXPERIMENT_LIST_FORMAT}."
        )

    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(
            f"Experiment list does not contain any saved experiments: {resolved_list_path}"
        )

    experiments: list[Path] = []
    base_directory = resolved_list_path.parent
    for raw_experiment in raw_experiments:
        if not isinstance(raw_experiment, str) or not raw_experiment.strip():
            raise ValueError(
                f"Experiment list contains an invalid entry: {raw_experiment!r}"
            )
        candidate = Path(raw_experiment).expanduser()
        if not candidate.is_absolute():
            candidate = (base_directory / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise ValueError(f"Experiment list entry does not exist: {candidate}")
        if not is_navigate_experiment_file(candidate):
            raise ValueError(
                "Experiment list entries must point to Navigate "
                f"experiment.yml/experiment.yaml files: {candidate}"
            )
        experiments.append(candidate)

    return _deduplicate_resolved_paths(experiments)


def _collect_experiment_paths_from_input_path(path: Path) -> list[Path]:
    """Collect experiment descriptors from a file, directory, or saved list.

    Parameters
    ----------
    path : pathlib.Path
        Input path that may refer to one experiment descriptor, a directory to
        scan recursively, or a saved ClearEx experiment list.

    Returns
    -------
    list[pathlib.Path]
        Ordered resolved experiment-descriptor paths discovered from the input.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a saved list file is invalid.
    """
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if candidate.is_dir():
        return _discover_navigate_experiment_files(candidate)

    resolved = candidate.resolve()
    if _is_saved_experiment_list_path(resolved):
        return _load_experiment_list_file(resolved)
    if is_navigate_experiment_file(resolved):
        return [resolved]
    return []


def _analysis_targets_from_store_requests(
    requests: Sequence[ExperimentStorePreparationRequest],
) -> tuple[AnalysisTarget, ...]:
    """Build ordered analysis targets from prepared experiment-store requests.

    Parameters
    ----------
    requests : sequence[ExperimentStorePreparationRequest]
        Prepared experiment/store requests from the setup dialog.

    Returns
    -------
    tuple[AnalysisTarget, ...]
        Ordered experiment/store target list for the analysis dialog.
    """
    return tuple(
        AnalysisTarget(
            experiment_path=str(Path(request.experiment_path).expanduser().resolve()),
            store_path=str(Path(request.target_store).expanduser().resolve()),
        )
        for request in requests
    )


def _analysis_targets_for_workflow(
    workflow: WorkflowConfig,
) -> tuple[AnalysisTarget, ...]:
    """Return available analysis targets for a workflow.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration to inspect.

    Returns
    -------
    tuple[AnalysisTarget, ...]
        Configured analysis targets, or a single fallback entry derived from
        ``workflow.file`` when no explicit experiment list is available.
    """
    if workflow.analysis_targets:
        return tuple(workflow.analysis_targets)
    file_path = str(workflow.file or "").strip()
    if not file_path:
        return tuple()
    return (
        AnalysisTarget(
            experiment_path=file_path,
            store_path=file_path,
        ),
    )


def _resolve_selected_analysis_target(
    workflow: WorkflowConfig,
    *,
    targets: Optional[Sequence[AnalysisTarget]] = None,
) -> Optional[AnalysisTarget]:
    """Resolve the active analysis target for a workflow.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration carrying experiment selection state.
    targets : sequence[AnalysisTarget], optional
        Precomputed targets. Defaults to the workflow-derived targets.

    Returns
    -------
    AnalysisTarget, optional
        Selected target, or ``None`` when no file/targets are available.
    """
    available_targets = (
        tuple(targets)
        if targets is not None
        else _analysis_targets_for_workflow(workflow)
    )
    if not available_targets:
        return None
    selected_experiment_path = str(
        workflow.analysis_selected_experiment_path or ""
    ).strip()
    if selected_experiment_path:
        for target in available_targets:
            if target.experiment_path == selected_experiment_path:
                return target
    current_file = str(workflow.file or "").strip()
    if current_file:
        for target in available_targets:
            if target.store_path == current_file:
                return target
    return available_targets[0]


def _analysis_target_display_text(target: AnalysisTarget) -> str:
    """Return operator-facing combo text for one analysis target.

    Parameters
    ----------
    target : AnalysisTarget
        Target to format.

    Returns
    -------
    str
        Experiment descriptor path shown in the analysis scope combo.
    """
    return str(target.experiment_path)


def _analysis_scope_summary_text(
    *,
    targets: Sequence[AnalysisTarget],
    selected_target: Optional[AnalysisTarget],
    apply_to_all: bool,
) -> str:
    """Build descriptive text for the analysis-scope panel.

    Parameters
    ----------
    targets : sequence[AnalysisTarget]
        Available analysis targets.
    selected_target : AnalysisTarget, optional
        Currently selected experiment/store target.
    apply_to_all : bool
        Whether the workflow is in batch-analysis mode.

    Returns
    -------
    str
        Human-readable scope summary.
    """
    if not targets:
        return "Analysis will run for the current data store."
    if apply_to_all and len(targets) > 1:
        reference_text = (
            f" Reference experiment: {selected_target.experiment_path}."
            if selected_target is not None
            else ""
        )
        return (
            f"Selected operations will run for all {len(targets)} loaded experiments."
            f"{reference_text}"
        )
    if selected_target is None:
        return "Analysis will run for the selected experiment."
    return (
        "Selected operations will run for the chosen experiment: "
        f"{selected_target.experiment_path}"
    )


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


_dashboard_relay_manager_singleton: Optional[DashboardRelayManager] = None


def get_dashboard_relay_manager() -> DashboardRelayManager:
    """Return the shared GUI dashboard relay manager singleton."""
    global _dashboard_relay_manager_singleton
    if _dashboard_relay_manager_singleton is None:
        _dashboard_relay_manager_singleton = DashboardRelayManager()
    return _dashboard_relay_manager_singleton


def _shutdown_dashboard_relay_manager() -> None:
    """Shut down the shared GUI dashboard relay manager."""
    global _dashboard_relay_manager_singleton
    manager = _dashboard_relay_manager_singleton
    if manager is None:
        return
    try:
        manager.shutdown()
    finally:
        _dashboard_relay_manager_singleton = None


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


def _resolve_zarr_save_settings_path(
    settings_directory: Optional[Path] = None,
) -> Path:
    """Resolve the user settings JSON path for persisted Zarr save config.

    Parameters
    ----------
    settings_directory : pathlib.Path, optional
        Settings directory override. Defaults to ``~/.clearex``.

    Returns
    -------
    pathlib.Path
        Full JSON path used for Zarr save configuration persistence.
    """
    directory = (
        settings_directory
        if settings_directory is not None
        else _resolve_clearex_settings_directory()
    )
    return directory / _CLEAREX_ZARR_SAVE_SETTINGS_FILE


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


def _load_last_used_zarr_save_config(
    settings_path: Optional[Path] = None,
) -> Optional[ZarrSaveConfig]:
    """Load the last-used Zarr save configuration from JSON.

    Parameters
    ----------
    settings_path : pathlib.Path, optional
        JSON file path override.

    Returns
    -------
    ZarrSaveConfig, optional
        Persisted Zarr save configuration, or ``None`` when settings are
        missing or unreadable.

    Raises
    ------
    None
        Read and parse errors are logged and handled internally.
    """
    path = (
        settings_path
        if settings_path is not None
        else _resolve_zarr_save_settings_path()
    )
    resolved = path.expanduser()
    if not resolved.exists():
        return None

    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to read Zarr save settings %s: %s",
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
            "Failed to decode Zarr save settings %s: %s",
            resolved,
            exc,
        )
        return None

    if isinstance(payload, dict) and not payload:
        return None

    try:
        return zarr_save_from_dict(payload)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to parse Zarr save settings %s: %s",
            resolved,
            exc,
        )
        return None


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


def _save_last_used_zarr_save_config(
    config: ZarrSaveConfig,
    settings_path: Optional[Path] = None,
) -> bool:
    """Persist the most recently used Zarr save configuration.

    Parameters
    ----------
    config : ZarrSaveConfig
        Zarr save configuration to persist.
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
        else _resolve_zarr_save_settings_path()
    )
    resolved = path.expanduser()
    _ensure_clearex_settings_directory(resolved.parent)

    try:
        payload = zarr_save_to_dict(config)
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        resolved.write_text(f"{serialized}\n", encoding="utf-8")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to persist Zarr save settings %s: %s",
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


def _should_apply_persisted_zarr_save(initial: Optional[WorkflowConfig]) -> bool:
    """Return whether persisted Zarr save settings should override defaults.

    Parameters
    ----------
    initial : WorkflowConfig, optional
        Initial workflow provided by caller.

    Returns
    -------
    bool
        ``True`` when caller did not provide custom Zarr save settings and
        persisted values can safely populate the GUI defaults.
    """
    if initial is None:
        return True
    return initial.zarr_save == ZarrSaveConfig()


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
        return bool(path in root)
    except Exception:
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


def _parse_source_component_level(
    *,
    component: str,
    full_resolution_source: str,
) -> Optional[int]:
    """Parse one component path into a pyramid resolution level.

    Parameters
    ----------
    component : str
        Candidate component token or full component path.
    full_resolution_source : str
        Full-resolution source component path associated with the token.

    Returns
    -------
    int, optional
        Parsed non-negative level when the component represents a pyramid level;
        otherwise ``None``.
    """
    token = str(component).strip()
    if not token:
        return None

    suffix_text: Optional[str] = None
    if token.startswith("level_"):
        suffix_text = token.split("_", maxsplit=1)[1]
    elif full_resolution_source == "data" and token.startswith("data_pyramid/level_"):
        suffix_text = token.split("data_pyramid/level_", maxsplit=1)[1]
    else:
        prefix = f"{full_resolution_source}_pyramid/level_"
        if token.startswith(prefix):
            suffix_text = token.split(prefix, maxsplit=1)[1]

    if suffix_text is None:
        return None
    suffix_text = str(suffix_text).split("/", maxsplit=1)[0]
    try:
        parsed = int(suffix_text)
    except Exception:
        return None
    if parsed < 0:
        return None
    return parsed


def _analysis_store_runtime_source_component(root: Any) -> Optional[str]:
    """Return the canonical 6D runtime-source component when available."""
    for candidate in (source_cache_component(), "data"):
        if _zarr_component_exists_in_root(root, candidate):
            return str(candidate)
    return None


def _analysis_store_runtime_source_shape_tpczyx(
    root: Any,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return canonical runtime-source shape in ``(t, p, c, z, y, x)`` order."""
    component = _analysis_store_runtime_source_component(root)
    if component is None:
        return None
    try:
        shape = tuple(getattr(root[component], "shape", ()))
    except Exception:
        return None
    if len(shape) != 6:
        return None
    try:
        return tuple(int(value) for value in shape)  # type: ignore[return-value]
    except Exception:
        return None


def _analysis_store_runtime_source_dtype_itemsize(root: Any) -> Optional[int]:
    """Return bytes-per-voxel for the canonical runtime-source component."""
    component = _analysis_store_runtime_source_component(root)
    if component is None:
        return None
    try:
        dtype = getattr(root[component], "dtype", None)
        itemsize = int(getattr(dtype, "itemsize"))
    except Exception:
        return None
    return max(1, itemsize)


def _full_resolution_source_component(source_component: str) -> str:
    """Return the full-resolution source component for a requested path.

    Parameters
    ----------
    source_component : str
        Requested source component path.

    Returns
    -------
    str
        Corresponding full-resolution component path.
    """
    requested = str(source_component).strip() or "data"
    if requested.startswith("data_pyramid/level_"):
        return "data"
    if "_pyramid/level_" in requested:
        return requested.split("_pyramid/level_", maxsplit=1)[0]
    return requested


def _component_channel_count(
    *,
    root: Any,
    component: str,
) -> Optional[int]:
    """Return channel count for a 6D-like source component when available.

    Parameters
    ----------
    root : Any
        Open Zarr root group.
    component : str
        Candidate component path.

    Returns
    -------
    int, optional
        Positive channel count, or ``None`` if unavailable.
    """
    candidate = str(component).strip()
    if not candidate or not _zarr_component_exists_in_root(root, candidate):
        return None
    try:
        source = root[candidate]
    except Exception:
        return None

    shape = getattr(source, "shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) < 3:
        return None
    try:
        count = int(shape[2])
    except Exception:
        return None
    if count <= 0:
        return None
    return int(count)


def _discover_component_resolution_levels(
    *,
    root: Any,
    source_component: str,
) -> tuple[int, ...]:
    """Discover available resolution levels for one source component.

    Parameters
    ----------
    root : Any
        Open Zarr root group.
    source_component : str
        Resolved source component path requested by the user.

    Returns
    -------
    tuple[int, ...]
        Sorted unique resolution levels available for the requested source.

    Notes
    -----
    This helper is GUI-facing and intentionally tolerant to missing metadata.
    Resolution level ``0`` is always retained as a safe fallback.
    """
    requested = str(source_component).strip() or "data"
    full_resolution_source = _full_resolution_source_component(requested)

    available_levels: set[int] = set()
    if _zarr_component_exists_in_root(root, full_resolution_source):
        available_levels.add(0)

    direct_level = _parse_source_component_level(
        component=requested,
        full_resolution_source=full_resolution_source,
    )
    if direct_level is not None and _zarr_component_exists_in_root(root, requested):
        available_levels.add(int(direct_level))

    pyramid_group_component = (
        "data_pyramid"
        if full_resolution_source == "data"
        else f"{full_resolution_source}_pyramid"
    )
    if _zarr_component_exists_in_root(root, pyramid_group_component):
        try:
            pyramid_group = root[pyramid_group_component]
        except Exception:
            pyramid_group = None
        if pyramid_group is not None:
            member_keys: set[str] = set()
            try:
                member_keys.update(
                    str(key).strip() for key in pyramid_group.group_keys()
                )
            except Exception:
                pass
            try:
                member_keys.update(
                    str(key).strip() for key in pyramid_group.array_keys()
                )
            except Exception:
                pass
            for key in member_keys:
                parsed_level = _parse_source_component_level(
                    component=key,
                    full_resolution_source=full_resolution_source,
                )
                if parsed_level is None:
                    continue
                component = f"{pyramid_group_component}/{key}"
                if _zarr_component_exists_in_root(root, component):
                    available_levels.add(int(parsed_level))

    metadata_components: list[str] = []
    if full_resolution_source == "data":
        try:
            data_levels = root.attrs.get("data_pyramid_levels")
        except Exception:
            data_levels = None
        if isinstance(data_levels, (tuple, list)):
            metadata_components.extend(str(value).strip() for value in data_levels)
    try:
        source_node = root[full_resolution_source]
    except Exception:
        source_node = None
    if source_node is not None:
        try:
            pyramid_levels = source_node.attrs.get("pyramid_levels")
        except Exception:
            pyramid_levels = None
        if isinstance(pyramid_levels, (tuple, list)):
            metadata_components.extend(str(value).strip() for value in pyramid_levels)
    for component in metadata_components:
        if not component:
            continue
        parsed_level = _parse_source_component_level(
            component=component,
            full_resolution_source=full_resolution_source,
        )
        if parsed_level is None:
            continue
        if _zarr_component_exists_in_root(root, component):
            available_levels.add(int(parsed_level))

    if not available_levels:
        available_levels.add(0)
    return tuple(sorted(available_levels))


def _discover_component_channels(
    *,
    root: Any,
    source_component: str,
) -> tuple[int, ...]:
    """Discover available channel indices for one source component.

    Parameters
    ----------
    root : Any
        Open Zarr root group.
    source_component : str
        Resolved source component path requested by the user.

    Returns
    -------
    tuple[int, ...]
        Sorted channel indices available for the requested source.

    Notes
    -----
    This helper is GUI-facing and intentionally tolerant to missing metadata.
    Channel index ``0`` is always retained as a safe fallback.
    """
    requested = str(source_component).strip() or "data"
    full_resolution_source = _full_resolution_source_component(requested)

    channel_count = _component_channel_count(
        root=root, component=full_resolution_source
    )
    if channel_count is None:
        channel_count = _component_channel_count(root=root, component=requested)
    if channel_count is None:
        channel_count = _component_channel_count(root=root, component="data")

    if channel_count is None:
        return (0,)
    return tuple(int(index) for index in range(int(channel_count)))


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
    if operation_name in {"render_movie", "compile_movie"}:
        options = []
        option_values: set[str] = set()
    else:
        options = [("data", "Raw data (data)")]
        option_values = {"data"}
    available = available_store_output_components or {}

    for upstream_name in selected_order:
        if upstream_name == operation_name:
            break
        if upstream_name in option_values:
            continue
        component = analysis_chainable_output_component(upstream_name)
        allow_non_chainable = (
            (operation_name == "fusion" and upstream_name == "registration")
            or (operation_name == "render_movie" and upstream_name == "visualization")
            or (operation_name == "compile_movie" and upstream_name == "render_movie")
        )
        if allow_non_chainable and component is None:
            component = operation_output_components.get(upstream_name)
        if component is None or upstream_name == "data":
            continue
        label = operation_labels.get(
            upstream_name, upstream_name.replace("_", " ").title()
        )
        options.append((upstream_name, f"{label} output ({component}) [scheduled]"))
        option_values.add(upstream_name)

    for upstream_name in operation_key_order:
        if upstream_name == operation_name or upstream_name in option_values:
            continue
        component = str(available.get(upstream_name, "")).strip()
        if not component:
            continue
        allow_non_chainable = (
            (operation_name == "fusion" and upstream_name == "registration")
            or (operation_name == "render_movie" and upstream_name == "visualization")
            or (operation_name == "compile_movie" and upstream_name == "render_movie")
        )
        if (
            analysis_chainable_output_component(upstream_name) is None
            and not allow_non_chainable
        ):
            continue
        label = operation_labels.get(
            upstream_name, upstream_name.replace("_", " ").title()
        )
        options.append((upstream_name, f"{label} output ({component}) [existing]"))
        option_values.add(upstream_name)

    return options


def _build_visualization_volume_layer_component_options(
    *,
    selected_order: Sequence[str],
    operation_key_order: Sequence[str],
    operation_labels: Mapping[str, str],
    available_store_output_components: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, str]]:
    """Build component options for visualization volume layers.

    Parameters
    ----------
    selected_order : sequence[str]
        Currently selected operations in execution order.
    operation_key_order : sequence of str
        Canonical operation display order.
    operation_labels : mapping[str, str]
        Human-readable labels per operation.
    available_store_output_components : mapping[str, str], optional
        Existing output components discovered in the selected store.

    Returns
    -------
    list[tuple[str, str]]
        Ordered ``(component_path, label)`` options. Includes raw ``data``
        plus scheduled and existing chainable outputs.
    """
    options: list[tuple[str, str]] = [("data", "Raw data (data)")]
    option_components = {"data"}
    available = available_store_output_components or {}

    for operation_name in selected_order:
        if operation_name == "visualization":
            break
        component = analysis_chainable_output_component(operation_name)
        if component is None or component in option_components:
            continue
        label = operation_labels.get(
            operation_name,
            operation_name.replace("_", " ").title(),
        )
        options.append((component, f"{label} output ({component}) [scheduled]"))
        option_components.add(component)

    for operation_name in operation_key_order:
        component = str(available.get(operation_name, "")).strip()
        if not component or component in option_components:
            continue
        if analysis_chainable_output_component(operation_name) is None:
            continue
        label = operation_labels.get(
            operation_name,
            operation_name.replace("_", " ").title(),
        )
        options.append((component, f"{label} output ({component}) [existing]"))
        option_components.add(component)
    return options


_FALLBACK_VISUALIZATION_BLENDINGS: tuple[str, ...] = (
    "translucent",
    "translucent_no_depth",
    "additive",
    "minimum",
    "opaque",
    "multiplicative",
)
_FALLBACK_VISUALIZATION_RENDERINGS: tuple[str, ...] = (
    "translucent",
    "additive",
    "iso",
    "mip",
    "minip",
    "attenuated_mip",
    "average",
)
_FALLBACK_VISUALIZATION_COLORMAPS: tuple[str, ...] = (
    "gray",
    "green",
    "magenta",
    "cyan",
    "yellow",
    "red",
    "blue",
    "bop orange",
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "turbo",
)


def _deduplicate_non_empty_text(values: Sequence[Any]) -> list[str]:
    """Return first-seen unique non-empty string values.

    Parameters
    ----------
    values : sequence[Any]
        Candidate values.

    Returns
    -------
    list[str]
        Unique non-empty string values in original order.
    """
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _available_visualization_blending_options() -> list[str]:
    """Return selectable napari blending modes for visualization layers."""
    try:
        from napari.layers.base._base_constants import Blending

        values = [str(getattr(item, "value", item)).strip() for item in Blending]
        normalized = _deduplicate_non_empty_text(values)
        if normalized:
            return normalized
    except Exception:
        pass
    return list(_FALLBACK_VISUALIZATION_BLENDINGS)


def _available_visualization_rendering_options() -> list[str]:
    """Return selectable napari image-rendering modes for visualization layers."""
    try:
        from napari.layers.image._image_constants import ImageRendering

        values = [str(getattr(item, "value", item)).strip() for item in ImageRendering]
        normalized = _deduplicate_non_empty_text(values)
        if normalized:
            return normalized
    except Exception:
        pass
    return list(_FALLBACK_VISUALIZATION_RENDERINGS)


def _available_visualization_colormap_options() -> list[str]:
    """Return selectable napari colormap names for visualization layers."""
    try:
        from napari.utils.colormaps import AVAILABLE_COLORMAPS

        keys = sorted(str(key).strip() for key in AVAILABLE_COLORMAPS.keys())
        normalized = _deduplicate_non_empty_text(keys)
        if normalized:
            return normalized
    except Exception:
        pass
    return list(_FALLBACK_VISUALIZATION_COLORMAPS)


def _particle_overlay_available_for_visualization(
    *,
    selected_order: Sequence[str],
    has_particle_detection_output: bool = False,
) -> bool:
    """Return whether particle-overlay controls should be available.

    Parameters
    ----------
    selected_order : sequence[str]
        Currently selected operations in execution order.
    has_particle_detection_output : bool, default=False
        Whether the current store already contains particle detections.

    Returns
    -------
    bool
        ``True`` when overlay can be supported by either:
        a current particle-detection run before visualization or successful
        particle-detection output already present in the store.
    """
    ordered = [str(name).strip() for name in selected_order]
    if "particle_detection" in ordered and "visualization" in ordered:
        if ordered.index("particle_detection") < ordered.index("visualization"):
            return True

    return bool(has_particle_detection_output)


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


def _path_signature(path: Path) -> Optional[tuple[int, int]]:
    """Return a lightweight `(mtime_ns, size)` signature for cache validation.

    Parameters
    ----------
    path : pathlib.Path
        Path to stat.

    Returns
    -------
    tuple[int, int], optional
        Signature tuple when stat succeeds, otherwise ``None``.
    """
    try:
        stat_result = path.stat()
    except OSError:
        return None
    return int(stat_result.st_mtime_ns), int(stat_result.st_size)


@dataclass(frozen=True)
class ExperimentMetadataCacheEntry:
    """Cache one resolved setup metadata context for an experiment path.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    source_data_path : pathlib.Path
        Resolved acquisition source path.
    image_info : ImageInfo
        Source metadata summary loaded for the setup panel.
    experiment_signature : tuple[int, int], optional
        Cached signature of the experiment descriptor file.
    source_signature : tuple[int, int], optional
        Cached signature of the resolved source path.
    override_directory : pathlib.Path, optional
        Source-directory override active at cache creation time.
    """

    experiment: NavigateExperiment
    source_data_path: Path
    image_info: ImageInfo
    experiment_signature: Optional[tuple[int, int]]
    source_signature: Optional[tuple[int, int]]
    override_directory: Optional[Path]


@dataclass(frozen=True)
class ExperimentStorePreparationRequest:
    """Describe one experiment store that may need canonical materialization.

    Parameters
    ----------
    experiment_path : pathlib.Path
        Navigate ``experiment.yml`` path represented in the setup list.
    experiment : NavigateExperiment
        Parsed experiment metadata used for canonical store preparation.
    source_data_path : pathlib.Path
        Resolved acquisition data path referenced by the experiment metadata.
    target_store : pathlib.Path
        Canonical ClearEx store path for this experiment.
    """

    experiment_path: Path
    experiment: NavigateExperiment
    source_data_path: Path
    target_store: Path


@dataclass(frozen=True)
class ExperimentStorePreparationResult:
    """Record the outcome of preparing one experiment's canonical store.

    Parameters
    ----------
    experiment_path : pathlib.Path
        Navigate ``experiment.yml`` path that was evaluated.
    store_path : pathlib.Path
        Canonical store path used or created for that experiment.
    skipped_existing : bool
        Whether an already-complete canonical store was reused without
        rematerialization.
    """

    experiment_path: Path
    store_path: Path
    skipped_existing: bool


def _has_reusable_canonical_store(store_path: Path) -> bool:
    """Return whether a canonical store already exists and is complete.

    Parameters
    ----------
    store_path : pathlib.Path
        Candidate canonical store path.

    Returns
    -------
    bool
        ``True`` when the store exists and canonical ingestion has completed.
    """
    resolved_store_path = Path(store_path).expanduser().resolve()
    if not resolved_store_path.exists():
        return False
    return has_complete_canonical_data_store(resolved_store_path)


def _plan_experiment_store_materialization(
    requests: Sequence[ExperimentStorePreparationRequest],
    *,
    selected_experiment_path: Path,
    rebuild_requested: bool = False,
) -> tuple[
    ExperimentStorePreparationRequest,
    list[ExperimentStorePreparationRequest],
    list[ExperimentStorePreparationRequest],
]:
    """Partition listed experiments into pending and already-ready stores.

    Parameters
    ----------
    requests : sequence[ExperimentStorePreparationRequest]
        Ordered experiment store requests represented in the setup list.
    selected_experiment_path : pathlib.Path
        Experiment currently selected in the GUI list.
    rebuild_requested : bool, default=False
        Whether all listed stores should be rebuilt explicitly even when a
        complete canonical store already exists.

    Returns
    -------
    tuple[ExperimentStorePreparationRequest, list[ExperimentStorePreparationRequest], list[ExperimentStorePreparationRequest]]
        Selected experiment request, pending requests that still require
        materialization, and ready requests whose stores can be reused.

    Raises
    ------
    ValueError
        If the selected experiment is not present in ``requests``.
    """
    resolved_selected_path = Path(selected_experiment_path).expanduser().resolve()
    selected_request: Optional[ExperimentStorePreparationRequest] = None
    pending_requests: list[ExperimentStorePreparationRequest] = []
    ready_requests: list[ExperimentStorePreparationRequest] = []

    for request in requests:
        if request.experiment_path == resolved_selected_path:
            selected_request = request
        if rebuild_requested:
            pending_requests.append(request)
        elif _has_reusable_canonical_store(request.target_store):
            ready_requests.append(request)
        else:
            pending_requests.append(request)

    if selected_request is None:
        raise ValueError("Selected experiment is not present in the experiment list.")
    return selected_request, pending_requests, ready_requests


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


def _spatial_calibration_binding_choices() -> tuple[tuple[str, str], ...]:
    """Return labeled spatial-calibration binding choices for GUI controls.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``(label, binding)`` pairs in UI display order.
    """
    return (
        ("+X stage", "+x"),
        ("-X stage", "-x"),
        ("+Y stage", "+y"),
        ("-Y stage", "-y"),
        ("+Z stage", "+z"),
        ("-Z stage", "-z"),
        ("+F focus", "+f"),
        ("-F focus", "-f"),
        ("Disabled", "none"),
    )


def _format_spatial_calibration_summary(config: SpatialCalibrationConfig) -> str:
    """Format a compact setup-dialog summary for spatial calibration.

    Parameters
    ----------
    config : SpatialCalibrationConfig
        Calibration to summarize.

    Returns
    -------
    str
        Human-readable summary for setup and analysis dialogs.
    """
    binding_labels = {
        binding: label for label, binding in _spatial_calibration_binding_choices()
    }
    lines = [
        f"Canonical: {format_spatial_calibration(config)}",
    ]
    for axis_name, binding in zip(
        SPATIAL_CALIBRATION_WORLD_AXES,
        config.stage_axis_map_zyx,
        strict=False,
    ):
        lines.append(
            f"World {axis_name.upper()}: " f"{binding_labels.get(binding, binding)}"
        )
    lines.append("Theta: Rotate Z/Y about world X")
    return "\n".join(lines)


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
        QScrollArea#popupDialogScroll {
            border: none;
            background: transparent;
        }
        QWidget#popupDialogContent {
            background: transparent;
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
        QComboBox QLineEdit {
            background: transparent;
            border: none;
            min-height: 0px;
            padding: 0px;
            margin: 0px;
            color: #e6edf3;
            selection-background-color: #2f81f7;
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
    dialog.setStyleSheet(_popup_dialog_stylesheet() + """
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
        """)
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

    class _CurrentPageStackedWidget(QStackedWidget):
        """QStackedWidget whose size hints follow only the active page."""

        def hasHeightForWidth(self) -> bool:
            """Return whether the active page supports height-for-width sizing.

            Parameters
            ----------
            None

            Returns
            -------
            bool
                ``True`` when the current page has height-for-width support.
            """
            current = self.currentWidget()
            if current is None:
                return super().hasHeightForWidth()
            return bool(current.hasHeightForWidth())

        def heightForWidth(self, width: int) -> int:
            """Return preferred height for the active page at the given width.

            Parameters
            ----------
            width : int
                Proposed widget width in pixels.

            Returns
            -------
            int
                Preferred height for the current page.
            """
            current = self.currentWidget()
            if current is None:
                return super().heightForWidth(int(width))
            if current.hasHeightForWidth():
                return int(current.heightForWidth(int(width)))
            return int(current.sizeHint().height())

        def sizeHint(self) -> QSize:
            """Return size hint for the current visible page.

            Parameters
            ----------
            None

            Returns
            -------
            QSize
                Size hint for the active page.
            """
            current = self.currentWidget()
            if current is None:
                return super().sizeHint()
            return current.sizeHint()

        def minimumSizeHint(self) -> QSize:
            """Return minimum size hint for the current visible page.

            Parameters
            ----------
            None

            Returns
            -------
            QSize
                Minimum size hint for the active page.
            """
            current = self.currentWidget()
            if current is None:
                return super().minimumSizeHint()
            return current.minimumSizeHint()

    def _configure_themed_scroll_area_surface(
        scroll_area: QScrollArea,
        *,
        scroll_object_name: str,
        viewport_object_name: str,
    ) -> None:
        """Assign themed surface object names to a rounded scroll area.

        Parameters
        ----------
        scroll_area : QScrollArea
            Scroll area whose viewport participates in a rounded dark surface.
        scroll_object_name : str
            Object name applied to ``scroll_area`` for stylesheet targeting.
        viewport_object_name : str
            Object name applied to the internal viewport widget.

        Returns
        -------
        None
            The scroll area is configured in-place.

        Notes
        -----
        Rounded dark cards need their scroll viewport explicitly styled to avoid
        exposing the platform-default viewport background in clipped corners.
        """
        scroll_area.setObjectName(scroll_object_name)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        viewport = scroll_area.viewport()
        viewport.setObjectName(viewport_object_name)
        viewport.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        viewport.setAutoFillBackground(True)

    class _TableRowSelectionFilter(QObject):
        """Select the owning table row when a cell widget is interacted with.

        Parameters
        ----------
        table : QTableWidget
            Table whose rows should track embedded widget interaction.

        Returns
        -------
        None
            Event filter is initialized in-place.
        """

        def __init__(self, table: QTableWidget) -> None:
            """Initialize the filter.

            Parameters
            ----------
            table : QTableWidget
                Table whose rows should be selected on widget interaction.

            Returns
            -------
            None
                Filter is initialized in-place.
            """
            super().__init__(table)
            self._table = table

        def eventFilter(self, watched: QObject, event: Optional[QEvent]) -> bool:
            """Select the corresponding table row on click or focus.

            Parameters
            ----------
            watched : QObject
                Widget receiving the event.
            event : QEvent, optional
                Qt event being filtered.

            Returns
            -------
            bool
                ``False`` to allow normal event processing to continue.
            """
            if (
                event is not None
                and isinstance(watched, QWidget)
                and event.type()
                in {
                    QEvent.Type.FocusIn,
                    QEvent.Type.MouseButtonPress,
                }
            ):
                cell_origin = watched.mapTo(self._table.viewport(), QPoint(0, 0))
                index = self._table.indexAt(cell_origin)
                if index.isValid():
                    self._table.selectRow(int(index.row()))
            return False

    def _install_table_row_selection_widget_hook(
        table: QTableWidget,
        widget: QWidget,
    ) -> QWidget:
        """Ensure an embedded table widget selects its row on interaction.

        Parameters
        ----------
        table : QTableWidget
            Table receiving row selection updates.
        widget : QWidget
            Embedded cell widget to monitor.

        Returns
        -------
        QWidget
            The same widget, configured in-place for convenient chaining.
        """
        selector = table.property("_clearexRowSelectionFilter")
        if not isinstance(selector, _TableRowSelectionFilter):
            selector = _TableRowSelectionFilter(table)
            table.setProperty("_clearexRowSelectionFilter", selector)
        widget.installEventFilter(selector)
        if isinstance(widget, QComboBox):
            line_edit = widget.lineEdit()
            if line_edit is not None:
                line_edit.installEventFilter(selector)
        return widget

    def _configure_fixed_height_button(
        button: QPushButton,
        *,
        minimum_height: int = 36,
    ) -> QPushButton:
        """Keep a dialog button visible under constrained-height layouts.

        Parameters
        ----------
        button : QPushButton
            Button to pin to a nonzero fixed-height policy.
        minimum_height : int, default=36
            Minimum pixel height retained when the dialog is height-clamped.

        Returns
        -------
        QPushButton
            The same button, configured in-place for convenient chaining.
        """
        button.setMinimumHeight(int(minimum_height))
        button.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        return button

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
            self.result_config: Optional[ZarrSaveConfig] = None
            self._chunk_inputs: Dict[str, QSpinBox] = {}
            self._pyramid_inputs: Dict[str, QLineEdit] = {}

            self._build_ui()
            self._hydrate(initial)
            self.setStyleSheet(_popup_dialog_stylesheet())
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_ZARR_SAVE_DIALOG_MINIMUM_SIZE,
                preferred_size=_ZARR_SAVE_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
            )

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
            outer_root = QVBoxLayout(self)
            outer_root.setContentsMargins(0, 0, 0, 0)
            outer_root.setSpacing(0)

            content_scroll = QScrollArea(self)
            content_scroll.setObjectName("popupDialogScroll")
            content_scroll.setWidgetResizable(True)
            content_scroll.setFrameShape(QFrame.Shape.NoFrame)
            content_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            outer_root.addWidget(content_scroll, 1)

            content_widget = QWidget()
            content_widget.setObjectName("popupDialogContent")
            content_scroll.setWidget(content_widget)

            root = QVBoxLayout(content_widget)
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
            self._defaults_button = _configure_fixed_height_button(
                QPushButton("Reset Defaults")
            )
            self._cancel_button = _configure_fixed_height_button(QPushButton("Cancel"))
            self._apply_button = _configure_fixed_height_button(QPushButton("Apply"))
            self._apply_button.setObjectName("runButton")
            footer.addWidget(self._defaults_button)
            footer.addStretch(1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._apply_button)
            root.addLayout(footer)

            self._defaults_button.clicked.connect(self._on_reset_defaults)
            self._cancel_button.clicked.connect(self.reject)
            self._apply_button.clicked.connect(self._on_apply)
            content_widget.setMinimumHeight(root.sizeHint().height())

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

    class SpatialCalibrationDialog(QDialog):
        """Dialog for configuring store-level stage-to-world axis bindings."""

        def __init__(
            self,
            initial: SpatialCalibrationConfig,
            parent: Optional[QDialog] = None,
        ) -> None:
            """Initialize spatial-calibration controls.

            Parameters
            ----------
            initial : SpatialCalibrationConfig
                Initial stage-to-world mapping.
            parent : QDialog, optional
                Parent dialog widget.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__(parent)
            self.setWindowTitle("Spatial Calibration")
            self.result_config: Optional[SpatialCalibrationConfig] = None
            self._binding_inputs: Dict[str, QComboBox] = {}

            self._build_ui()
            self._hydrate(initial)
            self.setStyleSheet(_popup_dialog_stylesheet())
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_SPATIAL_CALIBRATION_DIALOG_MINIMUM_SIZE,
                preferred_size=_SPATIAL_CALIBRATION_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
            )

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
            outer_root = QVBoxLayout(self)
            outer_root.setContentsMargins(0, 0, 0, 0)
            outer_root.setSpacing(0)

            content_scroll = QScrollArea(self)
            content_scroll.setObjectName("popupDialogScroll")
            content_scroll.setWidgetResizable(True)
            content_scroll.setFrameShape(QFrame.Shape.NoFrame)
            content_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            outer_root.addWidget(content_scroll, 1)

            content_widget = QWidget()
            content_widget.setObjectName("popupDialogContent")
            content_scroll.setWidget(content_widget)

            root = QVBoxLayout(content_widget)
            apply_popup_root_spacing(root)

            description = QLabel(
                "Map world Z/Y/X placement axes to Navigate multiposition stage "
                "coordinates. This affects spatial placement metadata only; "
                "canonical data remains unchanged."
            )
            description.setWordWrap(True)
            root.addWidget(description)

            bindings_group = QGroupBox("World Axis Mapping")
            bindings_layout = QFormLayout(bindings_group)
            apply_form_spacing(bindings_layout)

            for axis_name in SPATIAL_CALIBRATION_WORLD_AXES:
                combo = QComboBox()
                for label, binding in _spatial_calibration_binding_choices():
                    combo.addItem(label, binding)
                bindings_layout.addRow(f"World {axis_name.upper()}", combo)
                self._binding_inputs[axis_name] = combo

            root.addWidget(bindings_group)

            note = QLabel(
                "THETA remains a rotation of the Z/Y plane about world X for v1."
            )
            note.setWordWrap(True)
            root.addWidget(note)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            self._defaults_button = _configure_fixed_height_button(
                QPushButton("Reset Identity")
            )
            self._cancel_button = _configure_fixed_height_button(QPushButton("Cancel"))
            self._apply_button = _configure_fixed_height_button(QPushButton("Apply"))
            self._apply_button.setObjectName("runButton")
            footer.addWidget(self._defaults_button)
            footer.addStretch(1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._apply_button)
            root.addLayout(footer)

            self._defaults_button.clicked.connect(self._on_reset_defaults)
            self._cancel_button.clicked.connect(self.reject)
            self._apply_button.clicked.connect(self._on_apply)
            content_widget.setMinimumHeight(root.sizeHint().height())

        def _hydrate(self, initial: SpatialCalibrationConfig) -> None:
            """Populate combo boxes from an initial calibration.

            Parameters
            ----------
            initial : SpatialCalibrationConfig
                Calibration to display.

            Returns
            -------
            None
                Widget values are updated in-place.
            """
            for axis_name, binding in zip(
                SPATIAL_CALIBRATION_WORLD_AXES,
                initial.stage_axis_map_zyx,
                strict=False,
            ):
                combo = self._binding_inputs[axis_name]
                index = combo.findData(binding)
                combo.setCurrentIndex(index if index >= 0 else 0)

        def _on_reset_defaults(self) -> None:
            """Reset controls to the identity mapping."""
            self._hydrate(SpatialCalibrationConfig())

        def _on_apply(self) -> None:
            """Validate the selected bindings and accept the dialog."""
            try:
                self.result_config = SpatialCalibrationConfig(
                    stage_axis_map_zyx=tuple(
                        str(self._binding_inputs[axis_name].currentData())
                        for axis_name in SPATIAL_CALIBRATION_WORLD_AXES
                    )
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Spatial Calibration", str(exc))
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
            self.result_config: Optional[DaskBackendConfig] = None
            self._mode_index: Dict[str, int] = {}
            self._recommendation_shape_tpczyx = recommendation_shape_tpczyx
            self._recommendation_chunks_tpczyx = recommendation_chunks_tpczyx
            self._recommendation_dtype_itemsize = recommendation_dtype_itemsize
            self._latest_local_recommendation: Optional[LocalClusterRecommendation] = (
                None
            )
            self._parameter_help_map: Dict[QObject, str] = {}
            self._parameter_help_card: Optional[QFrame] = None
            self._parameter_help_label: Optional[QLabel] = None

            self._build_ui()
            self._hydrate(initial)
            self.setStyleSheet(_popup_dialog_stylesheet())
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_DASK_BACKEND_DIALOG_MINIMUM_SIZE,
                preferred_size=_DASK_BACKEND_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
            )

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
            outer_root = QVBoxLayout(self)
            outer_root.setContentsMargins(0, 0, 0, 0)
            outer_root.setSpacing(0)

            self._content_scroll = QScrollArea(self)
            self._content_scroll.setObjectName("popupDialogScroll")
            self._content_scroll.setWidgetResizable(True)
            self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._content_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            outer_root.addWidget(self._content_scroll, 1)

            content_widget = QWidget()
            content_widget.setObjectName("popupDialogContent")
            self._content_scroll.setWidget(content_widget)
            root = QVBoxLayout(content_widget)
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
            self._register_parameter_hint(
                self._mode_combo,
                self._dask_backend_help_texts()["mode"],
            )
            mode_row.addWidget(mode_label)
            mode_row.addWidget(self._mode_combo, 1)
            root.addLayout(mode_row)

            self._mode_help_label = QLabel("")
            self._mode_help_label.setWordWrap(True)
            self._mode_help_label.setObjectName("metadataFieldValue")
            self._mode_help_label.setMinimumHeight(
                max(28, int(self._mode_help_label.fontMetrics().height()) + 10)
            )
            root.addWidget(self._mode_help_label)

            self._mode_stack = QStackedWidget()
            self._mode_stack.addWidget(self._build_local_cluster_page())
            self._mode_stack.addWidget(self._build_slurm_runner_page())
            self._mode_stack.addWidget(self._build_slurm_cluster_page())
            root.addWidget(self._mode_stack, 1)

            self._parameter_help_card = QFrame(self)
            self._parameter_help_card.setObjectName("helpCard")
            help_layout = QVBoxLayout(self._parameter_help_card)
            apply_help_stack_spacing(help_layout)
            help_title = QLabel("Parameter Help")
            help_title.setObjectName("helpTitle")
            help_layout.addWidget(help_title)
            self._parameter_help_label = QLabel("")
            self._parameter_help_label.setObjectName("helpBody")
            self._parameter_help_label.setWordWrap(True)
            help_layout.addWidget(self._parameter_help_label)
            self._parameter_help_card.hide()
            outer_root.addWidget(self._parameter_help_card, 0)

            footer_frame = QFrame(self)
            footer_frame.setObjectName("analysisFooterCard")
            footer_frame.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
            footer_root = QVBoxLayout(footer_frame)
            footer_root.setContentsMargins(20, 0, 20, 20)
            footer_root.setSpacing(0)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            self._defaults_button = _configure_fixed_height_button(
                QPushButton("Reset Defaults")
            )
            self._cancel_button = _configure_fixed_height_button(QPushButton("Cancel"))
            self._apply_button = _configure_fixed_height_button(QPushButton("Apply"))
            self._apply_button.setObjectName("runButton")
            footer.addWidget(self._defaults_button)
            footer.addStretch(1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._apply_button)
            footer_root.addLayout(footer)
            outer_root.addWidget(footer_frame, 0)

            self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            self._defaults_button.clicked.connect(self._on_reset_defaults)
            self._cancel_button.clicked.connect(self.reject)
            self._apply_button.clicked.connect(self._on_apply)
            content_widget.setMinimumHeight(root.sizeHint().height())

        def _register_parameter_hint(self, widget: QWidget, message: str) -> None:
            """Register focus and hover help text for a dialog widget."""
            widget.setToolTip(message)
            widget.installEventFilter(self)
            self._parameter_help_map[widget] = str(message)
            viewport_getter = getattr(widget, "viewport", None)
            if callable(viewport_getter):
                viewport = viewport_getter()
                if isinstance(viewport, QWidget) and viewport is not widget:
                    viewport.installEventFilter(self)
                    self._parameter_help_map[viewport] = str(message)

        @staticmethod
        def _dask_backend_help_texts() -> Dict[str, str]:
            """Return plain-language help text for the backend dialog."""
            return {
                "mode": (
                    "Choose LocalCluster to run everything on this machine, "
                    "SLURMRunner to connect to a scheduler file that already exists, "
                    "or SLURMCluster to let ClearEx submit worker jobs to Slurm directly."
                ),
                "local_workers": (
                    "Leave this blank to let ClearEx choose a worker count automatically. "
                    "Set a number when you want to cap or pin local parallelism."
                ),
                "local_threads": (
                    "Choose how many threads each local worker should use. "
                    "One thread per worker is a safe default for most runs."
                ),
                "local_memory": (
                    "Set the maximum memory each local worker may use before Dask starts "
                    "spilling data to disk."
                ),
                "local_directory": (
                    "Pick a folder for local worker scratch files and spill files."
                ),
                "local_directory_browse": (
                    "Choose the folder where local workers should write scratch files."
                ),
                "local_recommend": (
                    "Fill the local worker settings with a suggested worker count, thread "
                    "count, and memory limit based on the current dataset."
                ),
                "runner_scheduler_file": (
                    "Select the scheduler file created by your Slurm launch script so "
                    "ClearEx can connect to the existing workers."
                ),
                "runner_scheduler_file_browse": (
                    "Choose the scheduler file for an already running Slurm-backed Dask cluster."
                ),
                "runner_wait_workers": (
                    "Wait for this many workers before ClearEx begins work. "
                    "Set it to auto to skip waiting."
                ),
                "cluster_workers": "Request this many Slurm worker jobs.",
                "cluster_cores": ("Set how many CPU cores each worker job should use."),
                "cluster_processes": (
                    "Choose how many worker processes each job should start."
                ),
                "cluster_memory": (
                    "Set the memory request for each worker job, such as 64GB."
                ),
                "cluster_local_directory": (
                    "Pick a scratch directory for worker spill files on the cluster."
                ),
                "cluster_local_directory_browse": (
                    "Choose the scratch directory that worker jobs should use for spill files."
                ),
                "cluster_interface": (
                    "Enter the network interface workers should use to connect, such as eth0 or ib0."
                ),
                "cluster_walltime": (
                    "Set the maximum run time for each worker job, such as 02:00:00."
                ),
                "cluster_job_name": "Set the Slurm job name shown in the queue.",
                "cluster_queue": (
                    "Choose the Slurm queue or partition that should receive the worker jobs."
                ),
                "cluster_death_timeout": (
                    "Set how long ClearEx should keep retrying before it gives up on a worker that stopped responding."
                ),
                "cluster_mail_user": (
                    "Enter an email address for Slurm job notifications."
                ),
                "cluster_directives": (
                    "Add extra Slurm directives here, one per line."
                ),
                "cluster_dashboard": (
                    "Set the address where the worker dashboard should listen, or leave it blank to let Dask choose."
                ),
                "cluster_scheduler_interface": (
                    "Choose the network interface the scheduler should use to talk to workers."
                ),
                "cluster_idle_timeout": (
                    "Set how long the cluster should stay alive with no work before shutting down."
                ),
                "cluster_allowed_failures": (
                    "Choose how many worker failures to tolerate before Slurm stops the cluster."
                ),
            }

        def _set_parameter_help(self, text: str) -> None:
            """Update the fixed parameter-help label text."""
            if self._parameter_help_label is not None:
                self._parameter_help_label.setText(str(text))

        def _show_parameter_help(self, text: str) -> None:
            """Show the fixed parameter-help card."""
            self._set_parameter_help(text)
            if self._parameter_help_card is not None:
                self._parameter_help_card.show()

        def _hide_parameter_help(self) -> None:
            """Hide the fixed parameter-help card."""
            if self._parameter_help_card is not None:
                self._parameter_help_card.hide()

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
            help_texts = self._dask_backend_help_texts()

            self._local_workers_input = QLineEdit()
            self._local_workers_input.setPlaceholderText("blank = auto")
            form.addRow("Workers", self._local_workers_input)
            self._register_parameter_hint(
                self._local_workers_input,
                help_texts["local_workers"],
            )

            self._local_threads_spin = QSpinBox()
            self._local_threads_spin.setRange(1, 1024)
            form.addRow("Threads per worker", self._local_threads_spin)
            self._register_parameter_hint(
                self._local_threads_spin,
                help_texts["local_threads"],
            )

            self._local_memory_input = QLineEdit()
            self._local_memory_input.setPlaceholderText("auto")
            form.addRow("Memory limit", self._local_memory_input)
            self._register_parameter_hint(
                self._local_memory_input,
                help_texts["local_memory"],
            )

            self._local_directory_input = QLineEdit()
            self._register_parameter_hint(
                self._local_directory_input,
                help_texts["local_directory"],
            )
            self._local_directory_browse = _configure_fixed_height_button(
                QPushButton("Browse")
            )
            self._register_parameter_hint(
                self._local_directory_browse,
                help_texts["local_directory_browse"],
            )
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

            self._local_recommend_button = _configure_fixed_height_button(
                QPushButton("Recommend Settings")
            )
            self._register_parameter_hint(
                self._local_recommend_button,
                help_texts["local_recommend"],
            )
            self._local_recommend_button.clicked.connect(
                self._on_recommend_local_cluster
            )
            form.addRow("", self._local_recommend_button)

            self._local_recommendation_label = QLabel("")
            self._local_recommendation_label.setWordWrap(True)
            self._local_recommendation_label.setObjectName("metadataFieldValue")
            self._local_recommendation_label.setMinimumHeight(
                max(
                    28,
                    int(self._local_recommendation_label.fontMetrics().height()) + 10,
                )
            )
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
            help_texts = self._dask_backend_help_texts()

            self._runner_scheduler_file_input = QLineEdit()
            self._runner_scheduler_file_input.setPlaceholderText(
                "Path to scheduler file"
            )
            self._register_parameter_hint(
                self._runner_scheduler_file_input,
                help_texts["runner_scheduler_file"],
            )
            self._runner_scheduler_file_browse = _configure_fixed_height_button(
                QPushButton("Browse")
            )
            self._register_parameter_hint(
                self._runner_scheduler_file_browse,
                help_texts["runner_scheduler_file_browse"],
            )
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
            self._register_parameter_hint(
                self._runner_wait_workers_spin,
                help_texts["runner_wait_workers"],
            )
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
            help_texts = self._dask_backend_help_texts()

            self._cluster_workers_spin = QSpinBox()
            self._cluster_workers_spin.setRange(1, 100000)
            worker_form.addRow("Workers (jobs)", self._cluster_workers_spin)
            self._register_parameter_hint(
                self._cluster_workers_spin,
                help_texts["cluster_workers"],
            )

            self._cluster_cores_spin = QSpinBox()
            self._cluster_cores_spin.setRange(1, 1024)
            worker_form.addRow("Cores", self._cluster_cores_spin)
            self._register_parameter_hint(
                self._cluster_cores_spin,
                help_texts["cluster_cores"],
            )

            self._cluster_processes_spin = QSpinBox()
            self._cluster_processes_spin.setRange(1, 256)
            worker_form.addRow("Processes", self._cluster_processes_spin)
            self._register_parameter_hint(
                self._cluster_processes_spin,
                help_texts["cluster_processes"],
            )

            self._cluster_memory_input = QLineEdit()
            worker_form.addRow("Memory", self._cluster_memory_input)
            self._register_parameter_hint(
                self._cluster_memory_input,
                help_texts["cluster_memory"],
            )

            self._cluster_local_directory_input = QLineEdit()
            self._register_parameter_hint(
                self._cluster_local_directory_input,
                help_texts["cluster_local_directory"],
            )
            self._cluster_local_directory_browse = _configure_fixed_height_button(
                QPushButton("Browse")
            )
            self._register_parameter_hint(
                self._cluster_local_directory_browse,
                help_texts["cluster_local_directory_browse"],
            )
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
            self._register_parameter_hint(
                self._cluster_interface_input,
                help_texts["cluster_interface"],
            )

            self._cluster_walltime_input = QLineEdit()
            worker_form.addRow("Walltime", self._cluster_walltime_input)
            self._register_parameter_hint(
                self._cluster_walltime_input,
                help_texts["cluster_walltime"],
            )

            self._cluster_job_name_input = QLineEdit()
            worker_form.addRow("Job name", self._cluster_job_name_input)
            self._register_parameter_hint(
                self._cluster_job_name_input,
                help_texts["cluster_job_name"],
            )

            self._cluster_queue_input = QLineEdit()
            worker_form.addRow("Queue / partition", self._cluster_queue_input)
            self._register_parameter_hint(
                self._cluster_queue_input,
                help_texts["cluster_queue"],
            )

            self._cluster_death_timeout_input = QLineEdit()
            worker_form.addRow("Death timeout", self._cluster_death_timeout_input)
            self._register_parameter_hint(
                self._cluster_death_timeout_input,
                help_texts["cluster_death_timeout"],
            )

            self._cluster_mail_user_input = QLineEdit()
            self._cluster_mail_user_input.setPlaceholderText("name@institution.edu")
            worker_form.addRow("Mail user", self._cluster_mail_user_input)
            self._register_parameter_hint(
                self._cluster_mail_user_input,
                help_texts["cluster_mail_user"],
            )

            self._cluster_directives_input = QPlainTextEdit()
            self._cluster_directives_input.setPlaceholderText(
                "One Slurm directive per line"
            )
            self._cluster_directives_input.setMinimumHeight(110)
            worker_form.addRow("Extra directives", self._cluster_directives_input)
            self._register_parameter_hint(
                self._cluster_directives_input,
                help_texts["cluster_directives"],
            )

            scheduler_group = QGroupBox("Scheduler Options")
            scheduler_form = QFormLayout(scheduler_group)
            apply_form_spacing(scheduler_form)
            scheduler_form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )

            self._cluster_dashboard_input = QLineEdit()
            scheduler_form.addRow("Dashboard address", self._cluster_dashboard_input)
            self._register_parameter_hint(
                self._cluster_dashboard_input,
                help_texts["cluster_dashboard"],
            )

            self._cluster_scheduler_interface_input = QLineEdit()
            scheduler_form.addRow("Interface", self._cluster_scheduler_interface_input)
            self._register_parameter_hint(
                self._cluster_scheduler_interface_input,
                help_texts["cluster_scheduler_interface"],
            )

            self._cluster_idle_timeout_input = QLineEdit()
            scheduler_form.addRow("Idle timeout", self._cluster_idle_timeout_input)
            self._register_parameter_hint(
                self._cluster_idle_timeout_input,
                help_texts["cluster_idle_timeout"],
            )

            self._cluster_allowed_failures_spin = QSpinBox()
            self._cluster_allowed_failures_spin.setRange(1, 100000)
            scheduler_form.addRow(
                "Allowed failures", self._cluster_allowed_failures_spin
            )
            self._register_parameter_hint(
                self._cluster_allowed_failures_spin,
                help_texts["cluster_allowed_failures"],
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

        def eventFilter(self, watched: QObject, event: Optional[QEvent]) -> bool:
            """Handle hover and focus transitions for registered parameter help."""
            message = self._parameter_help_map.get(watched)
            if message and event is not None:
                event_type = event.type()
                if event_type in (QEvent.Type.Enter, QEvent.Type.FocusIn):
                    self._show_parameter_help(message)
                elif event_type in (QEvent.Type.Leave, QEvent.Type.FocusOut):
                    focus_widget = self.focusWidget()
                    if (
                        focus_widget is not None
                        and focus_widget in self._parameter_help_map
                    ):
                        self._show_parameter_help(
                            self._parameter_help_map[focus_widget]
                        )
                    else:
                        self._hide_parameter_help()
            return super().eventFilter(watched, event)

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

    class BatchDataStoreMaterializationWorker(QThread):
        """Background worker that prepares canonical stores for many experiments.

        Parameters
        ----------
        requests : sequence[ExperimentStorePreparationRequest]
            Ordered experiment store requests that still require preparation.
        dask_backend : DaskBackendConfig
            Backend configuration used for Dask execution.
        zarr_save : ZarrSaveConfig
            Zarr chunk and pyramid configuration to enforce.

        Attributes
        ----------
        progress_changed : pyqtSignal
            Signal with ``(percent, message)`` progress payload.
        succeeded : pyqtSignal
            Signal with a list of :class:`ExperimentStorePreparationResult`
            objects for all prepared stores.
        failed : pyqtSignal
            Signal with error text.
        """

        progress_changed = pyqtSignal(int, str)
        succeeded = pyqtSignal(object)
        failed = pyqtSignal(str, str)

        def __init__(
            self,
            *,
            requests: Sequence[ExperimentStorePreparationRequest],
            dask_backend: DaskBackendConfig,
            zarr_save: ZarrSaveConfig,
            force_rebuild: bool = False,
        ) -> None:
            """Initialize worker state.

            Parameters
            ----------
            requests : sequence[ExperimentStorePreparationRequest]
                Ordered experiment store requests that still require
                preparation.
            dask_backend : DaskBackendConfig
                Backend configuration used for Dask execution.
            zarr_save : ZarrSaveConfig
                Zarr chunk and pyramid configuration to enforce.
            force_rebuild : bool, default=False
                Whether to rebuild stores even when a complete canonical store
                already exists.

            Returns
            -------
            None
                Worker is initialized in-place.
            """
            super().__init__()
            self._requests = list(requests)
            self._dask_backend = dask_backend
            self._zarr_save = zarr_save
            self._force_rebuild = bool(force_rebuild)

        def _emit_progress(self, percent: int, message: str) -> None:
            """Emit stage progress updates from the worker thread.

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

        def _prefixed_message(
            self,
            index: int,
            total: int,
            message: str,
        ) -> str:
            """Format a progress message with batch position context.

            Parameters
            ----------
            index : int
                Zero-based request index.
            total : int
                Total number of requests in the batch.
            message : str
                Stage text for the current request.

            Returns
            -------
            str
                Prefixed message string.
            """
            return f"[{index + 1}/{total}] {message}"

        def _overall_progress_percent(
            self,
            *,
            index: int,
            total: int,
            stage_percent: int,
        ) -> int:
            """Map per-request progress into one overall batch percentage.

            Parameters
            ----------
            index : int
                Zero-based request index.
            total : int
                Total number of requests in the batch.
            stage_percent : int
                Current request progress percentage.

            Returns
            -------
            int
                Overall batch progress percentage.
            """
            if total <= 0:
                return 100
            bounded_stage_percent = max(0, min(100, int(stage_percent)))
            batch_fraction = (float(index) + (bounded_stage_percent / 100.0)) / float(
                total
            )
            return max(0, min(100, int(round(batch_fraction * 100.0))))

        def run(self) -> None:
            """Execute batch canonical-store preparation in the background.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Emits success/failure signals on completion.
            """
            total = len(self._requests)
            results: list[ExperimentStorePreparationResult] = []
            if total <= 0:
                self.succeeded.emit(results)
                return

            try:
                with ExitStack() as exit_stack:
                    client = _configure_dask_backend_client(
                        self._dask_backend,
                        exit_stack=exit_stack,
                    )
                    for index, request in enumerate(self._requests):
                        experiment_label = str(request.experiment_path)

                        def _batch_progress_callback(
                            percent: int,
                            message: str,
                            *,
                            batch_index: int = index,
                            batch_total: int = total,
                        ) -> None:
                            overall = self._overall_progress_percent(
                                index=batch_index,
                                total=batch_total,
                                stage_percent=percent,
                            )
                            self._emit_progress(
                                overall,
                                self._prefixed_message(
                                    batch_index,
                                    batch_total,
                                    f"{experiment_label}: {message}",
                                ),
                            )

                        _batch_progress_callback(
                            1,
                            (
                                "Starting canonical store rebuild."
                                if self._force_rebuild
                                else "Starting canonical store preparation."
                            ),
                        )
                        result = materialize_experiment_data_store(
                            experiment=request.experiment,
                            source_path=request.source_data_path,
                            chunks=self._zarr_save.chunks_tpczyx(),
                            pyramid_factors=self._zarr_save.pyramid_tpczyx(),
                            client=client,
                            force_rebuild=self._force_rebuild,
                            progress_callback=_batch_progress_callback,
                        )
                        results.append(
                            ExperimentStorePreparationResult(
                                experiment_path=request.experiment_path,
                                store_path=Path(result.store_path)
                                .expanduser()
                                .resolve(),
                                skipped_existing=False,
                            )
                        )
                        self._emit_progress(
                            self._overall_progress_percent(
                                index=index,
                                total=total,
                                stage_percent=100,
                            ),
                            self._prefixed_message(
                                index,
                                total,
                                f"{experiment_label}: canonical store ready.",
                            ),
                        )
                self.succeeded.emit(results)
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Batch canonical store materialization failed."
                )
                self.failed.emit(
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )

    class MaterializationProgressDialog(QDialog):
        """Modal progress dialog for canonical store creation.

        Parameters
        ----------
        title_text : str, optional
            Title shown at the top of the dialog.
        initial_message : str, optional
            Initial progress message shown before updates arrive.
        parent : QDialog, optional
            Parent window.
        """

        def __init__(
            self,
            title_text: str = "Building `data_store.ome.zarr`",
            initial_message: str = "Starting materialization...",
            parent: Optional[QDialog] = None,
        ) -> None:
            """Initialize progress dialog widgets.

            Parameters
            ----------
            title_text : str, optional
                Title shown at the top of the dialog.
            initial_message : str, optional
                Initial progress message shown before updates arrive.
            parent : QDialog, optional
                Parent window.

            Returns
            -------
            None
                Dialog is initialized in-place.
            """
            super().__init__(parent)
            self.setWindowTitle("Preparing Data Store")
            self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

            root = QVBoxLayout(self)
            apply_window_root_spacing(root)

            title = QLabel(str(title_text))
            title.setObjectName("progressTitle")
            root.addWidget(title)

            self._message_label = QLabel(str(initial_message))
            self._message_label.setObjectName("progressMessage")
            self._message_label.setWordWrap(True)
            root.addWidget(self._message_label)

            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setTextVisible(True)
            self._progress.setFormat("%p%")
            root.addWidget(self._progress)

            self.setStyleSheet("""
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
                """)
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_MATERIALIZATION_PROGRESS_DIALOG_MINIMUM_SIZE,
                preferred_size=_MATERIALIZATION_PROGRESS_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
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
            Callback with signature ``(workflow, progress_callback,
            dask_client_lifecycle_callback=None)`` that executes the workflow.

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
        cancelled = pyqtSignal(str)
        failed = pyqtSignal(str, str)

        def __init__(
            self,
            *,
            workflow: WorkflowConfig,
            run_callback: GuiRunCallback,
            dask_client_lifecycle_callback: Optional[
                DaskClientLifecycleCallback
            ] = None,
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
            self._dask_client_lifecycle_callback = dask_client_lifecycle_callback
            self._cancel_requested = False

        def cancel(self) -> None:
            """Request cooperative cancellation of the running workflow.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Cancellation is recorded in-place and checked on progress
                callbacks.
            """
            self._cancel_requested = True
            self.requestInterruption()

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
            if self._cancel_requested or self.isInterruptionRequested():
                raise WorkflowExecutionCancelled("Analysis cancelled by user.")
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
                if self._dask_client_lifecycle_callback is None:
                    self._run_callback(self._workflow, self._emit_progress)
                else:
                    self._run_callback(
                        self._workflow,
                        self._emit_progress,
                        self._dask_client_lifecycle_callback,
                    )
                self._emit_progress(100, "Analysis workflow completed.")
                self.succeeded.emit()
            except WorkflowExecutionCancelled as exc:
                self.cancelled.emit(str(exc) or "Analysis cancelled by user.")
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

        cancel_requested = pyqtSignal()
        dashboard_requested = pyqtSignal()

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

            button_row = QHBoxLayout()
            apply_footer_row_spacing(button_row)
            button_row.addStretch(1)
            self._dashboard_button = QPushButton("Open Dask Dashboard")
            self._dashboard_button.setEnabled(False)
            self._dashboard_button.clicked.connect(
                lambda: self.dashboard_requested.emit()
            )
            button_row.addWidget(self._dashboard_button)
            self._stop_button = QPushButton("Stop Analysis")
            self._stop_button.clicked.connect(self._on_cancel_requested)
            button_row.addWidget(self._stop_button)
            root.addLayout(button_row)

            self.setStyleSheet("""
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
                QPushButton {
                    background-color: #1b2a3d;
                    border: 1px solid #2f4f78;
                    border-radius: 10px;
                    color: #dce8f8;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 10px 18px;
                }
                QPushButton:hover {
                    background-color: #24354b;
                    border-color: #4c6e99;
                }
                QPushButton:disabled {
                    color: #7f91ab;
                    border-color: #2a3442;
                    background-color: #16202d;
                }
                """)
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_ANALYSIS_PROGRESS_DIALOG_MINIMUM_SIZE,
                preferred_size=_ANALYSIS_PROGRESS_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
            )
            self.set_dashboard_available(False)

        def _on_cancel_requested(self) -> None:
            """Request cooperative cancellation of analysis execution.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Dialog state is updated in-place and a cancel signal is emitted.
            """
            self._stop_button.setEnabled(False)
            self._message_label.setText("Stopping after the current checkpoint...")
            self.cancel_requested.emit()

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

        def set_dashboard_available(
            self, available: bool, tooltip: Optional[str] = None
        ) -> None:
            """Update dashboard button availability and tooltip text."""
            self._dashboard_button.setEnabled(bool(available))
            if available:
                self._dashboard_button.setToolTip(
                    str(tooltip or "Open the local Dask dashboard relay.")
                )
            else:
                self._dashboard_button.setToolTip(
                    str(
                        tooltip
                        or "The Dask dashboard becomes available when an analysis client is active."
                    )
                )

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
            self._loaded_target_store_path: Optional[Path] = None
            self._experiment_list_file_path: Optional[Path] = None
            self._experiment_list_dirty = False
            self._source_data_directory_overrides: Dict[Path, Path] = {}
            self._experiment_metadata_cache: Dict[
                Path, ExperimentMetadataCacheEntry
            ] = {}
            self._spatial_calibration_drafts: Dict[Path, SpatialCalibrationConfig] = {}
            self._current_spatial_calibration: SpatialCalibrationConfig = (
                initial.spatial_calibration
            )
            self._materialization_worker: Optional[QThread] = None
            self._rebuild_store_checkbox: Optional[QCheckBox] = None

            self._build_ui()
            self._apply_theme()
            self._hydrate(initial)
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_SETUP_DIALOG_MINIMUM_SIZE,
                preferred_size=_SETUP_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
            )

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
            outer_root = QVBoxLayout(self)
            apply_window_root_spacing(outer_root)

            header = QFrame()
            header.setObjectName("headerCard")
            header_layout = QVBoxLayout(header)
            apply_stack_spacing(header_layout)
            header_image = _create_scaled_branding_label(
                filename=_GUI_HEADER_IMAGE,
                max_width=840,
                max_height=120,
                object_name="brandHeaderImage",
            )
            if header_image is not None:
                header_layout.addWidget(header_image)
            outer_root.addWidget(header)

            content_scroll = QScrollArea(self)
            content_scroll.setObjectName("setupDialogScroll")
            content_scroll.setWidgetResizable(True)
            content_scroll.setFrameShape(QFrame.Shape.NoFrame)
            content_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            outer_root.addWidget(content_scroll, 1)

            content_widget = QWidget()
            content_widget.setObjectName("setupDialogContent")
            content_scroll.setWidget(content_widget)

            root = QVBoxLayout(content_widget)
            apply_stack_spacing(root)

            data_group = QGroupBox("Navigate Experiment")
            data_layout = QVBoxLayout(data_group)
            apply_stack_spacing(data_layout)
            data_layout.setContentsMargins(12, 10, 12, 18)

            button_row = QHBoxLayout()
            apply_compact_row_spacing(button_row)
            self._load_experiment_button = QPushButton("Load Experiment")
            self._create_experiment_list_button = QPushButton("Create Experiment List")
            self._save_experiment_list_button = QPushButton("Save Experiment List")
            self._remove_experiment_button = QPushButton("Remove Selected")
            for button in (
                self._load_experiment_button,
                self._create_experiment_list_button,
                self._save_experiment_list_button,
                self._remove_experiment_button,
            ):
                button.setMinimumHeight(36)
                button.setSizePolicy(
                    QSizePolicy.Policy.Maximum,
                    QSizePolicy.Policy.Fixed,
                )
            button_row.addWidget(self._load_experiment_button)
            button_row.addWidget(self._create_experiment_list_button)
            button_row.addWidget(self._save_experiment_list_button)
            button_row.addWidget(self._remove_experiment_button)
            button_row.addStretch(1)
            data_layout.addLayout(button_row)

            self._experiment_list_hint = QLabel(
                "Load one experiment, scan a folder, or drag experiments, "
                "folders, and saved lists here. Selecting an item loads "
                "metadata automatically."
            )
            self._experiment_list_hint.setObjectName("experimentListHint")
            self._experiment_list_hint.setWordWrap(True)
            data_layout.addWidget(self._experiment_list_hint)

            self._experiment_list = QListWidget()
            self._experiment_list.setObjectName("experimentList")
            self._experiment_list.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
            self._experiment_list.setAlternatingRowColors(True)
            self._experiment_list.setAcceptDrops(True)
            self._experiment_list.setMinimumHeight(220)
            self._experiment_list.installEventFilter(self)
            self._experiment_list.viewport().installEventFilter(self)
            data_layout.addWidget(self._experiment_list)
            data_layout.addSpacing(2)

            self._experiment_list_status_label = QLabel("No experiments loaded.")
            self._experiment_list_status_label.setObjectName("experimentListStatus")
            self._experiment_list_status_label.setWordWrap(True)
            self._experiment_list_status_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._experiment_list_status_label.setContentsMargins(0, 4, 0, 4)
            self._experiment_list_status_label.setMinimumHeight(
                max(
                    28,
                    int(self._experiment_list_status_label.fontMetrics().height()) + 10,
                )
            )
            self._experiment_list_status_label.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
            data_layout.addWidget(self._experiment_list_status_label)

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

            zarr_group = QGroupBox("Zarr Save Config")
            zarr_layout = QGridLayout(zarr_group)
            apply_dialog_grid_spacing(zarr_layout)
            zarr_layout.setContentsMargins(10, 8, 10, 10)
            self._zarr_config_summary = QLabel("n/a")
            self._zarr_config_summary.setObjectName("zarrConfigSummary")
            self._zarr_config_summary.setWordWrap(True)
            self._zarr_config_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._zarr_config_button = QPushButton("Edit Zarr Settings")
            self._zarr_config_button.setSizePolicy(
                QSizePolicy.Policy.Maximum,
                QSizePolicy.Policy.Fixed,
            )
            zarr_layout.addWidget(
                self._zarr_config_summary,
                0,
                0,
                alignment=Qt.AlignmentFlag.AlignTop,
            )
            zarr_layout.addWidget(
                self._zarr_config_button,
                0,
                1,
                alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
            )
            zarr_layout.setColumnStretch(0, 1)
            root.addWidget(zarr_group)

            spatial_group = QGroupBox("Spatial Calibration")
            spatial_layout = QGridLayout(spatial_group)
            apply_dialog_grid_spacing(spatial_layout)
            spatial_layout.setContentsMargins(10, 8, 10, 10)
            self._spatial_calibration_summary = QLabel("n/a")
            self._spatial_calibration_summary.setObjectName("metadataFieldValue")
            self._spatial_calibration_summary.setWordWrap(True)
            self._spatial_calibration_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._spatial_calibration_button = QPushButton("Edit Spatial Calibration")
            self._spatial_calibration_button.setSizePolicy(
                QSizePolicy.Policy.Maximum,
                QSizePolicy.Policy.Fixed,
            )
            spatial_layout.addWidget(
                self._spatial_calibration_summary,
                0,
                0,
                alignment=Qt.AlignmentFlag.AlignTop,
            )
            spatial_layout.addWidget(
                self._spatial_calibration_button,
                0,
                1,
                alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
            )
            spatial_layout.setColumnStretch(0, 1)
            root.addWidget(spatial_group)

            dask_backend_group = QGroupBox("Dask Backend")
            dask_backend_layout = QGridLayout(dask_backend_group)
            apply_dialog_grid_spacing(dask_backend_layout)
            dask_backend_layout.setContentsMargins(10, 8, 10, 10)
            self._dask_backend_summary = QLabel("n/a")
            self._dask_backend_summary.setObjectName("metadataFieldValue")
            self._dask_backend_summary.setWordWrap(True)
            self._dask_backend_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._dask_backend_button = QPushButton("Edit Dask Backend")
            self._dask_backend_button.setSizePolicy(
                QSizePolicy.Policy.Maximum,
                QSizePolicy.Policy.Fixed,
            )
            dask_backend_layout.addWidget(
                self._dask_backend_summary,
                0,
                0,
                alignment=Qt.AlignmentFlag.AlignTop,
            )
            dask_backend_layout.addWidget(
                self._dask_backend_button,
                0,
                1,
                alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
            )
            dask_backend_layout.setColumnStretch(0, 1)
            root.addWidget(dask_backend_group)

            footer_frame = QFrame()
            footer_frame.setObjectName("setupFooterCard")
            footer_frame.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
            footer_root = QVBoxLayout(footer_frame)
            footer_root.setContentsMargins(0, 0, 0, 0)
            footer_root.setSpacing(0)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            footer_status = QVBoxLayout()
            apply_help_stack_spacing(footer_status)
            self._status_label = QLabel("Ready")
            self._status_label.setObjectName("statusLabel")
            self._status_label.setWordWrap(True)
            footer_status.addWidget(self._status_label)
            self._rebuild_store_checkbox = QCheckBox("Rebuild Canonical Store")
            self._rebuild_store_checkbox.setObjectName("statusLabel")
            self._rebuild_store_checkbox.setToolTip(
                "Rebuild every listed canonical store using the current Zarr "
                "chunking and pyramid settings."
            )
            footer_status.addWidget(self._rebuild_store_checkbox)
            self._cancel_button = QPushButton("Cancel")
            self._next_button = QPushButton("Next")
            self._next_button.setObjectName("runButton")
            footer.addLayout(footer_status, 1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._next_button)
            footer_root.addLayout(footer)
            outer_root.addWidget(footer_frame)

            self._load_experiment_button.clicked.connect(self._on_load_experiment)
            self._create_experiment_list_button.clicked.connect(
                self._on_create_experiment_list
            )
            self._save_experiment_list_button.clicked.connect(
                self._on_save_experiment_list
            )
            self._remove_experiment_button.clicked.connect(
                self._on_remove_selected_experiments
            )
            self._experiment_list.currentItemChanged.connect(
                self._on_current_experiment_changed
            )
            self._experiment_list.itemDoubleClicked.connect(
                self._on_experiment_item_activated
            )
            self._experiment_list.customContextMenuRequested.connect(
                self._show_experiment_list_context_menu
            )
            self._experiment_list.setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            self._dask_backend_button.clicked.connect(self._on_edit_dask_backend)
            self._zarr_config_button.clicked.connect(self._on_edit_zarr_settings)
            self._spatial_calibration_button.clicked.connect(
                self._on_edit_spatial_calibration
            )
            self._cancel_button.clicked.connect(self.reject)
            self._next_button.clicked.connect(self._on_next)
            self._rebuild_store_checkbox.toggled.connect(self._on_rebuild_store_toggled)
            content_widget.setMinimumHeight(root.sizeHint().height())

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
            self._refresh_dask_backend_summary()
            self._refresh_zarr_save_summary()
            self._refresh_spatial_calibration_summary()
            initial_file = str(initial.file or "").strip()
            if not initial_file:
                self._refresh_experiment_actions()
                return

            initial_path = Path(initial_file).expanduser()
            try:
                experiment_paths = _collect_experiment_paths_from_input_path(
                    initial_path
                )
            except Exception:
                logging.getLogger(__name__).debug(
                    "Ignoring non-experiment setup input during hydrate: %s",
                    initial_file,
                    exc_info=True,
                )
                self._refresh_experiment_actions()
                return
            if experiment_paths:
                if _is_saved_experiment_list_path(initial_path):
                    self._set_experiment_list_file_path(initial_path, dirty=False)
                self._set_experiment_list_paths(
                    experiment_paths,
                    current_path=experiment_paths[0],
                )
            self._refresh_experiment_actions()

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

        def _refresh_spatial_calibration_summary(self) -> None:
            """Refresh setup summary text for spatial calibration.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary label and button state are updated in-place.
            """
            summary = _format_spatial_calibration_summary(
                self._current_spatial_calibration
            )
            self._spatial_calibration_summary.setText(summary)
            self._spatial_calibration_summary.setToolTip(summary)
            has_selection = self._current_selected_experiment_path() is not None
            self._spatial_calibration_button.setEnabled(bool(has_selection))

        def _resolve_spatial_calibration_for_experiment(
            self,
            experiment_path: Path,
            *,
            target_store: Optional[Path] = None,
        ) -> SpatialCalibrationConfig:
            """Resolve draft/store/default calibration for one experiment.

            Parameters
            ----------
            experiment_path : pathlib.Path
                Selected experiment path.
            target_store : pathlib.Path, optional
                Prepared or target analysis-store path when already known.

            Returns
            -------
            SpatialCalibrationConfig
                Effective calibration for the experiment within the setup
                session.
            """
            resolved_experiment_path = Path(experiment_path).expanduser().resolve()
            draft = self._spatial_calibration_drafts.get(resolved_experiment_path)
            if draft is not None:
                return draft

            store_path = (
                Path(target_store).expanduser().resolve()
                if target_store is not None
                else None
            )
            if store_path is None:
                try:
                    request = self._resolve_store_preparation_request(
                        resolved_experiment_path
                    )
                except Exception:
                    return SpatialCalibrationConfig()
                store_path = request.target_store

            if is_zarr_store_path(store_path) and Path(store_path).exists():
                return load_store_spatial_calibration(store_path)
            return SpatialCalibrationConfig()

        def _set_current_spatial_calibration(
            self,
            *,
            experiment_path: Optional[Path],
            calibration: Optional[SpatialCalibrationConfig] = None,
            target_store: Optional[Path] = None,
        ) -> None:
            """Update the currently displayed calibration for setup.

            Parameters
            ----------
            experiment_path : pathlib.Path, optional
                Active experiment path.
            calibration : SpatialCalibrationConfig, optional
                Explicit calibration to display.
            target_store : pathlib.Path, optional
                Target store path used when loading from store attrs.

            Returns
            -------
            None
                Current setup calibration state is updated in-place.
            """
            if experiment_path is None:
                self._current_spatial_calibration = SpatialCalibrationConfig()
            else:
                resolved_experiment_path = Path(experiment_path).expanduser().resolve()
                self._current_spatial_calibration = (
                    calibration
                    if calibration is not None
                    else self._resolve_spatial_calibration_for_experiment(
                        resolved_experiment_path,
                        target_store=target_store,
                    )
                )
            self._refresh_spatial_calibration_summary()

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
            _save_last_used_zarr_save_config(self._zarr_save_config)
            self._set_status("Updated Zarr save settings.")

        def _on_edit_spatial_calibration(self) -> None:
            """Open spatial-calibration dialog for the selected experiment.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Stores the selected calibration as an in-session draft.
            """
            experiment_path = self._current_selected_experiment_path()
            if experiment_path is None:
                QMessageBox.information(
                    self,
                    "No Experiment Selected",
                    "Select an experiment before editing spatial calibration.",
                )
                return

            dialog = SpatialCalibrationDialog(
                initial=self._current_spatial_calibration,
                parent=self,
            )
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.result_config is None:
                return

            resolved_experiment_path = Path(experiment_path).expanduser().resolve()
            self._spatial_calibration_drafts[resolved_experiment_path] = (
                dialog.result_config
            )
            self._set_current_spatial_calibration(
                experiment_path=resolved_experiment_path,
                calibration=dialog.result_config,
            )
            self._set_status(
                "Updated spatial calibration draft for the selected experiment."
            )

        def _on_rebuild_store_toggled(self, checked: bool) -> None:
            """Update setup status when rebuild mode is toggled.

            Parameters
            ----------
            checked : bool
                Whether explicit canonical-store rebuild mode is enabled.

            Returns
            -------
            None
                Status text is updated in-place.
            """
            if checked:
                self._set_status(
                    "Rebuild mode enabled. Next will rebuild listed canonical stores "
                    "with the current Zarr settings."
                )
            else:
                self._set_status(
                    "Ready. Existing complete canonical stores will be reused."
                )

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
            self.setStyleSheet("""
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
                    font-size: 14px;
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
                    font-size: 14px;
                    color: #b8c8dd;
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
                QLabel#experimentListHint {
                    color: #9ab0ca;
                }
                QLabel#experimentListStatus {
                    color: #8ea4c0;
                }
                QScrollArea#setupDialogScroll {
                    border: none;
                    background: transparent;
                }
                QWidget#setupDialogContent {
                    background: transparent;
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
                QListWidget#experimentList {
                    background-color: #0b1320;
                    border: 1px solid #2b3f58;
                    border-radius: 10px;
                    padding: 6px;
                    alternate-background-color: #0f1724;
                    color: #e6edf3;
                    outline: none;
                }
                QListWidget#experimentList:focus {
                    border-color: #2f81f7;
                }
                QListWidget#experimentList::item {
                    border-radius: 6px;
                    padding: 8px 10px;
                }
                QListWidget#experimentList::item:alternate {
                    background-color: #0f1724;
                }
                QListWidget#experimentList::item:selected {
                    background-color: #1f6cd8;
                    color: #f8fbff;
                }
                QListWidget#experimentList::item:hover {
                    background-color: #172536;
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
                QPushButton:disabled {
                    background-color: #121a26;
                    border-color: #253447;
                    color: #6f8198;
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
                QMenu {
                    background-color: #111925;
                    border: 1px solid #2a3442;
                    border-radius: 8px;
                    padding: 6px;
                    color: #d9e2f1;
                }
                QMenu::item {
                    padding: 8px 20px;
                    border-radius: 6px;
                }
                QMenu::item:selected {
                    background-color: #1f6cd8;
                    color: #f8fbff;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2a3442;
                    margin: 6px 8px;
                }
                """)

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

        def _refresh_experiment_actions(self) -> None:
            """Refresh button enabled states for the experiment list.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Button state is updated in-place.
            """
            has_items = self._experiment_list.count() > 0
            has_selection = bool(self._experiment_list.selectedItems())
            self._save_experiment_list_button.setEnabled(has_items)
            self._remove_experiment_button.setEnabled(has_selection)
            self._next_button.setEnabled(has_items)
            self._refresh_experiment_list_status()

        def _set_experiment_list_file_path(
            self,
            list_path: Optional[Path],
            *,
            dirty: bool = False,
        ) -> None:
            """Store saved-list tracking state for the current experiment list.

            Parameters
            ----------
            list_path : pathlib.Path, optional
                Backing saved-list path when the current list originated from or
                was saved to disk.
            dirty : bool, default=False
                Whether the in-memory list has diverged from the saved file.

            Returns
            -------
            None
                Saved-list tracking state is updated in-place.
            """
            self._experiment_list_file_path = (
                Path(list_path).expanduser().resolve()
                if list_path is not None
                else None
            )
            self._experiment_list_dirty = bool(dirty) and (
                self._experiment_list_file_path is not None
            )
            self._refresh_experiment_list_status()

        def _refresh_experiment_list_status(self) -> None:
            """Refresh the summary line under the experiment list.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary label text and tooltip are updated in-place.
            """
            count = self._experiment_list.count()
            if count <= 0:
                text = "No experiments loaded."
            else:
                item_text = "experiment" if count == 1 else "experiments"
                current = self._current_selected_experiment_path()
                current_text = str(current) if current is not None else "none selected"
                if self._experiment_list_file_path is None:
                    source_text = "unsaved list"
                else:
                    state_suffix = " (modified)" if self._experiment_list_dirty else ""
                    source_text = f"{self._experiment_list_file_path}{state_suffix}"
                text = (
                    f"{count} {item_text} loaded. \n"
                    f"Current: {current_text}. \n"
                    f"List source: {source_text}."
                )
            self._experiment_list_status_label.setText(text)
            self._experiment_list_status_label.setToolTip(text)
            self._experiment_list_status_label.setWordWrap(True)

        def _reset_metadata_labels(self) -> None:
            """Reset metadata display labels to their empty state.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Metadata labels are updated in-place.
            """
            for label in self._metadata_labels.values():
                label.setText("n/a")
            self._set_current_spatial_calibration(experiment_path=None)

        def _experiment_path_from_item(
            self,
            item: Optional[QListWidgetItem],
        ) -> Optional[Path]:
            """Extract the experiment path stored on a list item.

            Parameters
            ----------
            item : QListWidgetItem, optional
                Experiment list item.

            Returns
            -------
            pathlib.Path, optional
                Stored experiment path when available, otherwise ``None``.
            """
            if item is None:
                return None
            raw_path = item.data(Qt.ItemDataRole.UserRole)
            text = str(raw_path).strip() if raw_path is not None else ""
            return Path(text).expanduser().resolve() if text else None

        def _experiment_list_paths(self) -> list[Path]:
            """Return all experiment paths currently listed in the setup UI.

            Parameters
            ----------
            None

            Returns
            -------
            list[pathlib.Path]
                Ordered resolved experiment paths.
            """
            experiment_paths: list[Path] = []
            for index in range(self._experiment_list.count()):
                path = self._experiment_path_from_item(
                    self._experiment_list.item(index)
                )
                if path is not None:
                    experiment_paths.append(path)
            return experiment_paths

        def _prune_experiment_metadata_cache(self, valid_paths: Sequence[Path]) -> None:
            """Drop cached metadata entries no longer present in the list.

            Parameters
            ----------
            valid_paths : sequence[pathlib.Path]
                Experiment paths that should remain cache-eligible.

            Returns
            -------
            None
                Cache is pruned in-place.
            """
            keep = {Path(path).expanduser().resolve() for path in valid_paths}
            for cached_path in list(self._experiment_metadata_cache):
                if cached_path not in keep:
                    self._experiment_metadata_cache.pop(cached_path, None)

        def _invalidate_experiment_metadata_cache_entry(
            self,
            experiment_path: Path,
        ) -> None:
            """Invalidate one cached experiment metadata entry.

            Parameters
            ----------
            experiment_path : pathlib.Path
                Experiment path whose cache entry should be removed.

            Returns
            -------
            None
                Cache is updated in-place.
            """
            resolved_path = Path(experiment_path).expanduser().resolve()
            self._experiment_metadata_cache.pop(resolved_path, None)

        def _cached_experiment_metadata_context(
            self,
            experiment_path: Path,
        ) -> Optional[tuple[Path, NavigateExperiment, Path, ImageInfo]]:
            """Return a valid cached metadata context when available.

            Parameters
            ----------
            experiment_path : pathlib.Path
                Experiment descriptor path to resolve.

            Returns
            -------
            tuple[pathlib.Path, NavigateExperiment, pathlib.Path, ImageInfo], optional
                Cached context tuple when signatures and overrides still match.
            """
            resolved_experiment_path = Path(experiment_path).expanduser().resolve()
            entry = self._experiment_metadata_cache.get(resolved_experiment_path)
            if entry is None:
                return None

            current_override = self._source_data_directory_overrides.get(
                resolved_experiment_path
            )
            if current_override != entry.override_directory:
                self._experiment_metadata_cache.pop(resolved_experiment_path, None)
                return None

            if _path_signature(resolved_experiment_path) != entry.experiment_signature:
                self._experiment_metadata_cache.pop(resolved_experiment_path, None)
                return None

            resolved_source_path = Path(entry.source_data_path).expanduser().resolve()
            if _path_signature(resolved_source_path) != entry.source_signature:
                self._experiment_metadata_cache.pop(resolved_experiment_path, None)
                return None

            return (
                resolved_experiment_path,
                entry.experiment,
                resolved_source_path,
                entry.image_info,
            )

        def _store_experiment_metadata_context(
            self,
            *,
            experiment_path: Path,
            experiment: NavigateExperiment,
            source_data_path: Path,
            info: ImageInfo,
        ) -> None:
            """Store one resolved metadata context in the in-memory cache.

            Parameters
            ----------
            experiment_path : pathlib.Path
                Experiment descriptor path.
            experiment : NavigateExperiment
                Parsed experiment metadata.
            source_data_path : pathlib.Path
                Resolved source path for the experiment.
            info : ImageInfo
                Source metadata summary.

            Returns
            -------
            None
                Cache is updated in-place.
            """
            resolved_experiment_path = Path(experiment_path).expanduser().resolve()
            resolved_source_path = Path(source_data_path).expanduser().resolve()
            self._experiment_metadata_cache[resolved_experiment_path] = (
                ExperimentMetadataCacheEntry(
                    experiment=experiment,
                    source_data_path=resolved_source_path,
                    image_info=info,
                    experiment_signature=_path_signature(resolved_experiment_path),
                    source_signature=_path_signature(resolved_source_path),
                    override_directory=self._source_data_directory_overrides.get(
                        resolved_experiment_path
                    ),
                )
            )

        def _current_selected_experiment_path(self) -> Optional[Path]:
            """Return the current experiment selection path.

            Parameters
            ----------
            None

            Returns
            -------
            pathlib.Path, optional
                Selected experiment path, otherwise ``None``.
            """
            return self._experiment_path_from_item(self._experiment_list.currentItem())

        def _set_experiment_list_paths(
            self,
            experiment_paths: Sequence[Path],
            *,
            current_path: Optional[Path] = None,
        ) -> None:
            """Replace the experiment list with a new ordered path set.

            Parameters
            ----------
            experiment_paths : sequence[pathlib.Path]
                Ordered experiment-descriptor paths to display.
            current_path : pathlib.Path, optional
                Path to select after the list is populated. Defaults to the
                first experiment when omitted.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            normalized_paths = _deduplicate_resolved_paths(experiment_paths)
            self._prune_experiment_metadata_cache(normalized_paths)
            prior_block_state = self._experiment_list.blockSignals(True)
            self._experiment_list.clear()
            for experiment_path in normalized_paths:
                item = QListWidgetItem(str(experiment_path))
                item.setData(Qt.ItemDataRole.UserRole, str(experiment_path))
                item.setToolTip(str(experiment_path))
                self._experiment_list.addItem(item)
            self._experiment_list.blockSignals(prior_block_state)

            if not normalized_paths:
                self._clear_loaded_experiment_context()
                self._reset_metadata_labels()
                self._set_experiment_list_file_path(None, dirty=False)
                self._refresh_experiment_actions()
                self._set_status("Ready")
                return

            target_path = (
                Path(current_path).expanduser().resolve()
                if current_path is not None
                else normalized_paths[0]
            )
            if target_path not in normalized_paths:
                target_path = normalized_paths[0]

            for index, experiment_path in enumerate(normalized_paths):
                if experiment_path == target_path:
                    self._experiment_list.setCurrentRow(index)
                    break
            self._refresh_experiment_actions()

        def _add_experiment_paths(
            self,
            experiment_paths: Sequence[Path],
            *,
            replace: bool = False,
            current_path: Optional[Path] = None,
        ) -> None:
            """Add experiments to the list or replace the list contents.

            Parameters
            ----------
            experiment_paths : sequence[pathlib.Path]
                Experiment paths to add.
            replace : bool, default=False
                When ``True``, replace the existing list contents.
            current_path : pathlib.Path, optional
                Path to select after the list update.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            previous_paths = self._experiment_list_paths()
            normalized_new_paths = _deduplicate_resolved_paths(experiment_paths)
            existing_paths = [] if replace else self._experiment_list_paths()
            combined_paths = _deduplicate_resolved_paths(
                [*existing_paths, *normalized_new_paths]
            )
            target_path = current_path
            if target_path is None and normalized_new_paths:
                target_path = normalized_new_paths[-1]
            if target_path is None:
                target_path = self._current_selected_experiment_path()
            self._set_experiment_list_paths(combined_paths, current_path=target_path)
            if (
                self._experiment_list_file_path is not None
                and combined_paths != previous_paths
            ):
                self._set_experiment_list_file_path(
                    self._experiment_list_file_path,
                    dirty=True,
                )

        def _resolve_dropped_experiment_paths(
            self,
            mime_data: QMimeData,
        ) -> tuple[list[Path], bool, Optional[Path]]:
            """Resolve experiment paths from a drag/drop payload.

            Parameters
            ----------
            mime_data : QMimeData
                Drag/drop payload to inspect.

            Returns
            -------
            tuple[list[pathlib.Path], bool, pathlib.Path or None]
                Resolved experiment paths, whether the list should be replaced,
                and the saved-list file path when one was dropped alone.
            """
            if not mime_data.hasUrls():
                return [], False, None

            local_paths: list[Path] = []
            for url in mime_data.urls():
                if not url.isLocalFile():
                    continue
                local_paths.append(Path(url.toLocalFile()).expanduser())

            replace_existing = False
            list_path: Optional[Path] = None
            if len(local_paths) == 1 and _is_saved_experiment_list_path(local_paths[0]):
                replace_existing = True
                list_path = local_paths[0].resolve()

            resolved_paths: list[Path] = []
            for candidate_path in local_paths:
                try:
                    resolved_paths.extend(
                        _collect_experiment_paths_from_input_path(candidate_path)
                    )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Failed to resolve dropped experiment input %s.",
                        candidate_path,
                        exc_info=True,
                    )
            return (
                _deduplicate_resolved_paths(resolved_paths),
                replace_existing,
                list_path,
            )

        def _can_accept_experiment_drop(self, mime_data: QMimeData) -> bool:
            """Determine whether dropped payload contains experiment inputs.

            Parameters
            ----------
            mime_data : QMimeData
                Drag/drop payload to inspect.

            Returns
            -------
            bool
                ``True`` when supported local experiment inputs are present.
            """
            experiment_paths, _replace_existing, _list_path = (
                self._resolve_dropped_experiment_paths(mime_data)
            )
            return bool(experiment_paths)

        def _apply_experiment_drop(self, mime_data: QMimeData) -> bool:
            """Apply dropped experiment inputs to the setup list.

            Parameters
            ----------
            mime_data : QMimeData
                Drag/drop payload to inspect.

            Returns
            -------
            bool
                ``True`` when a supported path was applied, otherwise ``False``.
            """
            experiment_paths, replace_existing, list_path = (
                self._resolve_dropped_experiment_paths(mime_data)
            )
            if not experiment_paths:
                return False

            current_path = (
                experiment_paths[0] if replace_existing else experiment_paths[-1]
            )
            self._add_experiment_paths(
                experiment_paths,
                replace=replace_existing,
                current_path=current_path,
            )
            if list_path is not None:
                self._set_experiment_list_file_path(list_path, dirty=False)
            return True

        def _clear_loaded_experiment_context(self) -> None:
            """Clear cached experiment metadata for the current selection.

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
            self._loaded_target_store_path = None

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:
            """Handle list keyboard removal and drag/drop routed through the list.

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
            if watched in {self._experiment_list, self._experiment_list.viewport()}:
                if (
                    watched is self._experiment_list
                    and isinstance(event, QKeyEvent)
                    and event.type() == QEvent.Type.KeyPress
                    and event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}
                ):
                    self._on_remove_selected_experiments()
                    event.accept()
                    return True
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
            """Accept dialog-level drags containing experiment inputs.

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
            """Accept dialog-level drag-move events for valid experiment inputs.

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
            """Handle dialog-level drops for experiment inputs.

            Parameters
            ----------
            event : QDropEvent
                Incoming drop event.

            Returns
            -------
            None
                Experiment list and event acceptance are updated in-place.
            """
            if self._apply_experiment_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def _on_load_experiment(self) -> None:
            """Open a file picker and add one experiment descriptor.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Experiment list is updated in-place.
            """
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Navigate experiment.yml",
                str(Path.cwd()),
                "Navigate Experiment (experiment.yml experiment.yaml *.yml *.yaml)",
            )
            if not file_path:
                return

            experiment_path = Path(file_path).expanduser().resolve()
            self._add_experiment_paths([experiment_path], current_path=experiment_path)

        def _on_create_experiment_list(self) -> None:
            """Open the experiment-list creation menu.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Runs the selected list-creation action.
            """
            menu = QMenu(self._create_experiment_list_button)
            scan_action = menu.addAction("Scan Folder...")
            load_action = menu.addAction("Load Saved List...")
            selected_action = menu.exec(
                self._create_experiment_list_button.mapToGlobal(
                    QPoint(0, self._create_experiment_list_button.height())
                )
            )
            if selected_action is scan_action:
                self._on_scan_experiment_folder()
            elif selected_action is load_action:
                self._on_load_saved_experiment_list()

        def _on_scan_experiment_folder(self) -> None:
            """Recursively discover experiments in a selected directory.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Experiment list is replaced in-place.
            """
            selected_directory = QFileDialog.getExistingDirectory(
                self,
                "Select Folder to Scan for experiment.yml",
                str(Path.cwd()),
            )
            if not selected_directory:
                return

            search_root = Path(selected_directory).expanduser().resolve()
            try:
                experiment_paths = _discover_navigate_experiment_files(search_root)
            except Exception as exc:
                _show_themed_error_dialog(
                    self,
                    "Experiment List Failed",
                    "Failed to create an experiment list from the selected folder.",
                    summary=f"{type(exc).__name__}: {exc}",
                    details=traceback.format_exc(),
                )
                self._set_status("Failed to scan for experiments.")
                return

            if not experiment_paths:
                QMessageBox.information(
                    self,
                    "No Experiments Found",
                    f"No Navigate experiment.yml files were found under {search_root}.",
                )
                self._set_status(
                    "No experiment.yml files found in the selected folder."
                )
                return

            self._set_experiment_list_file_path(None, dirty=False)
            self._set_experiment_list_paths(
                experiment_paths,
                current_path=experiment_paths[0],
            )

        def _on_load_saved_experiment_list(self) -> None:
            """Load a previously saved experiment list from disk.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Experiment list is replaced in-place.
            """
            file_filter = (
                "ClearEx Experiment List " f"(*{_CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX})"
            )
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Saved Experiment List",
                str(Path.cwd()),
                file_filter,
            )
            if not file_path:
                return

            list_path = Path(file_path).expanduser().resolve()
            try:
                experiment_paths = _load_experiment_list_file(list_path)
            except Exception as exc:
                _show_themed_error_dialog(
                    self,
                    "Experiment List Failed",
                    "Failed to load the saved experiment list.",
                    summary=f"{type(exc).__name__}: {exc}",
                    details=traceback.format_exc(),
                )
                self._set_status("Failed to load saved experiment list.")
                return

            self._set_experiment_list_file_path(list_path, dirty=False)
            self._set_experiment_list_paths(
                experiment_paths,
                current_path=experiment_paths[0],
            )

        def _on_save_experiment_list(self) -> None:
            """Save the current experiment list to disk.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Current experiment list is serialized to JSON.
            """
            experiment_paths = self._experiment_list_paths()
            if not experiment_paths:
                QMessageBox.information(
                    self,
                    "No Experiments",
                    "Add at least one experiment before saving a list.",
                )
                self._set_status("Add an experiment before saving a list.")
                return

            default_directory = (
                self._experiment_list_file_path.parent
                if self._experiment_list_file_path is not None
                else experiment_paths[0].parent
            )
            default_path = (
                default_directory / f"experiments{_CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX}"
            )
            file_filter = (
                "ClearEx Experiment List " f"(*{_CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX})"
            )
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Experiment List",
                str(default_path),
                file_filter,
            )
            if not file_path:
                return

            target_path = Path(file_path).expanduser()
            if not _is_saved_experiment_list_path(target_path):
                target_path = Path(
                    f"{target_path}{_CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX}"
                )
            try:
                saved_list_path = _save_experiment_list_file(
                    target_path, experiment_paths
                )
            except Exception as exc:
                _show_themed_error_dialog(
                    self,
                    "Experiment List Failed",
                    "Failed to save the experiment list.",
                    summary=f"{type(exc).__name__}: {exc}",
                    details=traceback.format_exc(),
                )
                self._set_status("Failed to save experiment list.")
                return

            self._set_experiment_list_file_path(saved_list_path, dirty=False)
            self._set_status(
                f"Saved experiment list to {self._experiment_list_file_path}."
            )

        def _on_remove_selected_experiments(self) -> None:
            """Remove the selected experiments from the setup list.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Experiment list and metadata state are updated in-place.
            """
            selected_items = self._experiment_list.selectedItems()
            if not selected_items:
                self._refresh_experiment_actions()
                return

            selected_paths = {
                path
                for path in (
                    self._experiment_path_from_item(item) for item in selected_items
                )
                if path is not None
            }
            remaining_paths = [
                path
                for path in self._experiment_list_paths()
                if path not in selected_paths
            ]
            current_path = self._current_selected_experiment_path()
            next_path = (
                current_path
                if current_path is not None and current_path in remaining_paths
                else (remaining_paths[0] if remaining_paths else None)
            )

            for removed_path in selected_paths:
                self._source_data_directory_overrides.pop(removed_path, None)
                self._invalidate_experiment_metadata_cache_entry(removed_path)
            if (
                self._experiment_list_file_path is not None
                and remaining_paths != self._experiment_list_paths()
            ):
                self._set_experiment_list_file_path(
                    self._experiment_list_file_path,
                    dirty=True,
                )
            self._set_experiment_list_paths(remaining_paths, current_path=next_path)
            if not remaining_paths:
                self._set_status("Removed selected experiments.")

        def _show_experiment_list_context_menu(self, position: QPoint) -> None:
            """Show the experiment-list context menu.

            Parameters
            ----------
            position : QPoint
                Click position in list-widget viewport coordinates.

            Returns
            -------
            None
                Executes one selected context-menu action.
            """
            menu = QMenu(self._experiment_list)
            item = self._experiment_list.itemAt(position)
            load_action = menu.addAction("Load Metadata")
            remove_action = menu.addAction("Remove Selected")
            load_action.setEnabled(item is not None)
            remove_action.setEnabled(bool(self._experiment_list.selectedItems()))
            selected_action = menu.exec(
                self._experiment_list.viewport().mapToGlobal(position)
            )
            if selected_action is load_action and item is not None:
                self._experiment_list.setCurrentItem(item)
                self._load_selected_experiment_metadata(force_reload=True)
            elif selected_action is remove_action:
                self._on_remove_selected_experiments()

        def _on_current_experiment_changed(
            self,
            current: Optional[QListWidgetItem],
            previous: Optional[QListWidgetItem],
        ) -> None:
            """Load metadata when the current experiment selection changes.

            Parameters
            ----------
            current : QListWidgetItem, optional
                Newly selected item.
            previous : QListWidgetItem, optional
                Previously selected item. Unused.

            Returns
            -------
            None
                Metadata state is updated in-place.
            """
            del previous
            self._refresh_experiment_actions()
            if current is None:
                if self._experiment_list.count() == 0:
                    self._clear_loaded_experiment_context()
                    self._reset_metadata_labels()
                    self._set_status("Ready")
                else:
                    self._set_current_spatial_calibration(experiment_path=None)
                return
            self._load_selected_experiment_metadata()

        def _on_experiment_item_activated(self, item: QListWidgetItem) -> None:
            """Reload metadata when a list item is double-clicked.

            Parameters
            ----------
            item : QListWidgetItem
                Activated list item.

            Returns
            -------
            None
                Metadata state is updated in-place.
            """
            self._experiment_list.setCurrentItem(item)
            self._load_selected_experiment_metadata(force_reload=True)

        def _load_selected_experiment_metadata(
            self,
            *,
            force_reload: bool = False,
        ) -> None:
            """Load metadata for the current experiment selection.

            Parameters
            ----------
            force_reload : bool, default=False
                When ``True``, bypass the cached experiment metadata state.

            Returns
            -------
            None
                Metadata state is updated in-place.
            """
            experiment_path = self._current_selected_experiment_path()
            if experiment_path is None:
                self._clear_loaded_experiment_context()
                self._reset_metadata_labels()
                self._set_status("Select an experiment to load metadata.")
                return
            self._load_metadata_for_experiment_path(
                experiment_path,
                force_reload=force_reload,
            )

        def _resolve_experiment_source_context(
            self,
            *,
            path: Path,
        ) -> tuple[Path, NavigateExperiment, Path]:
            """Resolve experiment metadata and source path for one experiment.

            Parameters
            ----------
            path : pathlib.Path
                Selected experiment path.

            Returns
            -------
            tuple[pathlib.Path, NavigateExperiment, pathlib.Path]
                Experiment path, parsed experiment, and resolved source path.

            Raises
            ------
            ValueError
                If path is missing or not an experiment descriptor.
            FileNotFoundError
                If the selected path does not exist.
            Exception
                Propagates parse/read failures from experiment metadata or
                acquisition path resolution.
            """
            selected_path = Path(path).expanduser()
            if not selected_path.exists():
                raise FileNotFoundError(f"Path does not exist: {selected_path}")
            if not is_navigate_experiment_file(selected_path):
                raise ValueError(
                    "This setup window requires Navigate experiment.yml or "
                    "experiment.yaml."
                )

            experiment_path = selected_path.resolve()
            experiment = load_navigate_experiment(experiment_path)
            override_directory = self._source_data_directory_overrides.get(
                experiment_path
            )
            try:
                source_data_path = resolve_experiment_data_path(
                    experiment,
                    search_directory=override_directory,
                )
            except ExperimentDataResolutionError as exc:
                source_data_path = self._prompt_for_source_data_directory(
                    experiment=experiment,
                    error=exc,
                )
                if source_data_path is None:
                    raise
            return experiment_path, experiment, source_data_path

        def _load_metadata_for_experiment_path(
            self,
            experiment_path: Path,
            *,
            force_reload: bool = False,
        ) -> None:
            """Load and display metadata for one experiment descriptor.

            Parameters
            ----------
            experiment_path : pathlib.Path
                Selected experiment-descriptor path.
            force_reload : bool, default=False
                When ``True``, bypass the cached metadata state.

            Returns
            -------
            None
                Metadata display fields and cached state are updated in-place.
            """
            resolved_experiment_path = Path(experiment_path).expanduser().resolve()
            if (
                not force_reload
                and self._loaded_experiment_path == resolved_experiment_path
                and self._loaded_experiment is not None
                and self._loaded_source_data_path is not None
                and self._loaded_image_info is not None
            ):
                return

            try:
                cached_context = (
                    None
                    if force_reload
                    else self._cached_experiment_metadata_context(
                        resolved_experiment_path
                    )
                )
                if cached_context is None:
                    loaded_path, experiment, source_data_path, info = (
                        self._load_experiment_context(path=resolved_experiment_path)
                    )
                    self._store_experiment_metadata_context(
                        experiment_path=loaded_path,
                        experiment=experiment,
                        source_data_path=source_data_path,
                        info=info,
                    )
                else:
                    loaded_path, experiment, source_data_path, info = cached_context
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Failed to load experiment metadata from %s.",
                    resolved_experiment_path,
                )
                self._clear_loaded_experiment_context()
                self._reset_metadata_labels()
                self._metadata_labels["path"].setText(str(resolved_experiment_path))
                self._set_current_spatial_calibration(
                    experiment_path=resolved_experiment_path
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
                experiment_path=loaded_path,
                resolved_data_path=source_data_path,
                experiment=experiment,
            )

            for key, value in summary.items():
                self._metadata_labels[key].setText(value)

            self._loaded_experiment = experiment
            self._loaded_experiment_path = loaded_path
            self._loaded_image_info = info
            self._loaded_source_data_path = source_data_path

            target_store = resolve_data_store_path(experiment, source_data_path)
            self._loaded_target_store_path = Path(target_store).expanduser().resolve()
            self._set_current_spatial_calibration(
                experiment_path=loaded_path,
                target_store=self._loaded_target_store_path,
            )
            self._set_status(f"Metadata loaded. \nTarget store: {target_store}")

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

                self._source_data_directory_overrides[
                    experiment.path.expanduser().resolve()
                ] = override_directory
                self._invalidate_experiment_metadata_cache_entry(experiment.path)
                return source_data_path

        def _load_experiment_context(
            self,
            *,
            path: Path,
        ) -> tuple[Path, NavigateExperiment, Path, ImageInfo]:
            """Load experiment and source metadata for setup validation.

            Parameters
            ----------
            path : pathlib.Path
                Selected experiment path.

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
            experiment_path, experiment, source_data_path = (
                self._resolve_experiment_source_context(path=path)
            )
            info = load_navigate_experiment_source_image_info(
                experiment=experiment,
                source_path=source_data_path,
                opener=self._opener,
                prefer_dask=True,
                chunks=self._chunks,
            )
            return experiment_path, experiment, source_data_path, info

        def _resolve_store_preparation_request(
            self,
            experiment_path: Path,
        ) -> ExperimentStorePreparationRequest:
            """Resolve canonical-store preparation inputs for one experiment.

            Parameters
            ----------
            experiment_path : pathlib.Path
                Experiment path to resolve.

            Returns
            -------
            ExperimentStorePreparationRequest
                Canonical-store preparation request for the experiment.
            """
            resolved_experiment_path = Path(experiment_path).expanduser().resolve()
            if (
                self._loaded_experiment_path == resolved_experiment_path
                and self._loaded_experiment is not None
                and self._loaded_source_data_path is not None
            ):
                experiment = self._loaded_experiment
                source_data_path = self._loaded_source_data_path
            else:
                cached_context = self._cached_experiment_metadata_context(
                    resolved_experiment_path
                )
                if cached_context is not None:
                    _loaded_path, experiment, source_data_path, _info = cached_context
                else:
                    _loaded_path, experiment, source_data_path = (
                        self._resolve_experiment_source_context(
                            path=resolved_experiment_path
                        )
                    )
            target_store = resolve_data_store_path(experiment, source_data_path)
            return ExperimentStorePreparationRequest(
                experiment_path=resolved_experiment_path,
                experiment=experiment,
                source_data_path=source_data_path,
                target_store=Path(target_store).expanduser().resolve(),
            )

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

        def _persist_spatial_calibration_for_requests(
            self,
            requests: Sequence[ExperimentStorePreparationRequest],
        ) -> Dict[Path, SpatialCalibrationConfig]:
            """Persist resolved spatial calibration for each prepared store.

            Parameters
            ----------
            requests : sequence[ExperimentStorePreparationRequest]
                Prepared or reused store requests.

            Returns
            -------
            dict[pathlib.Path, SpatialCalibrationConfig]
                Effective calibration written for each target store.
            """
            persisted: Dict[Path, SpatialCalibrationConfig] = {}
            for request in requests:
                resolved_store = Path(request.target_store).expanduser().resolve()
                calibration = self._resolve_spatial_calibration_for_experiment(
                    Path(request.experiment_path).expanduser().resolve(),
                    target_store=resolved_store,
                )
                persisted[resolved_store] = save_store_spatial_calibration(
                    resolved_store,
                    calibration,
                )
            return persisted

        def _accept_with_store_path(
            self,
            store_path: Path,
            *,
            analysis_targets: Sequence[AnalysisTarget],
            selected_experiment_path: Path,
            spatial_calibration: SpatialCalibrationConfig,
        ) -> None:
            """Finalize setup dialog with prepared store path configuration.

            Parameters
            ----------
            store_path : pathlib.Path
                Prepared canonical store path.
            analysis_targets : sequence[AnalysisTarget]
                Ordered experiment/store targets available for the analysis
                dialog.
            selected_experiment_path : pathlib.Path
                Navigate experiment descriptor currently selected in the setup
                list.
            spatial_calibration : SpatialCalibrationConfig
                Effective calibration written to the selected target store.

            Returns
            -------
            None
                Stores setup workflow config and accepts this dialog.
            """
            self.result_config = WorkflowConfig(
                file=str(store_path),
                analysis_targets=tuple(analysis_targets),
                analysis_selected_experiment_path=str(
                    Path(selected_experiment_path).expanduser().resolve()
                ),
                analysis_apply_to_all=False,
                prefer_dask=True,
                dask_backend=self._dask_backend_config,
                chunks=self._chunks,
                flatfield=False,
                deconvolution=False,
                shear_transform=False,
                particle_detection=False,
                registration=False,
                fusion=False,
                visualization=False,
                volume_export=False,
                mip_export=False,
                zarr_save=self._zarr_save_config,
                spatial_calibration=spatial_calibration,
                spatial_calibration_explicit=False,
            )
            _save_last_used_dask_backend_config(self._dask_backend_config)
            _save_last_used_zarr_save_config(self._zarr_save_config)
            self.accept()

        def _on_next(self) -> None:
            """Advance after batch store readiness checks for the experiment list.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Proceeds only when listed experiments have canonical stores.
            """
            selected_experiment_path = self._current_selected_experiment_path()
            if selected_experiment_path is None:
                QMessageBox.information(
                    self,
                    "No Experiment Selected",
                    "Select at least one experiment before continuing.",
                )
                self._set_status("Select an experiment before continuing.")
                return

            experiment_paths = self._experiment_list_paths()
            if not experiment_paths:
                QMessageBox.information(
                    self,
                    "No Experiments Loaded",
                    "Load at least one experiment before continuing.",
                )
                self._set_status("Load an experiment before continuing.")
                return

            requests: list[ExperimentStorePreparationRequest] = []
            for experiment_path in experiment_paths:
                try:
                    request = self._resolve_store_preparation_request(experiment_path)
                except Exception as exc:
                    logging.getLogger(__name__).exception(
                        "Failed to resolve experiment input from %s.",
                        experiment_path,
                    )
                    _show_themed_error_dialog(
                        self,
                        "Experiment Preparation Failed",
                        "Failed to prepare one experiment for batch ingestion.",
                        summary=f"{type(exc).__name__}: {exc}",
                        details=traceback.format_exc(),
                    )
                    self._set_status("Batch preparation failed before store creation.")
                    return
                requests.append(request)
            rebuild_requested = bool(
                self._rebuild_store_checkbox.isChecked()
                if self._rebuild_store_checkbox is not None
                else False
            )
            try:
                selected_request, pending_requests, ready_requests = (
                    _plan_experiment_store_materialization(
                        requests,
                        selected_experiment_path=selected_experiment_path,
                        rebuild_requested=rebuild_requested,
                    )
                )
            except ValueError:
                self._set_status("Selected experiment is no longer available.")
                return
            ready_count = len(ready_requests)
            analysis_targets = _analysis_targets_from_store_requests(requests)

            if not pending_requests:
                try:
                    persisted_calibrations = (
                        self._persist_spatial_calibration_for_requests(requests)
                    )
                except Exception as exc:
                    logging.getLogger(__name__).exception(
                        "Failed to persist spatial calibration for prepared stores."
                    )
                    _show_themed_error_dialog(
                        self,
                        "Spatial Calibration Failed",
                        "Failed to persist spatial calibration to the prepared analysis stores.",
                        summary=f"{type(exc).__name__}: {exc}",
                        details=traceback.format_exc(),
                    )
                    self._set_status("Failed to persist spatial calibration.")
                    return
                self._set_status(
                    "All listed data stores are ready. Opening analysis selection."
                )
                self._accept_with_store_path(
                    selected_request.target_store,
                    analysis_targets=analysis_targets,
                    selected_experiment_path=selected_request.experiment_path,
                    spatial_calibration=persisted_calibrations[
                        Path(selected_request.target_store).expanduser().resolve()
                    ],
                )
                return

            if rebuild_requested:
                self._set_status(
                    f"Rebuilding {len(pending_requests)} listed canonical data stores."
                )
            else:
                self._set_status(
                    f"Preparing {len(pending_requests)} of {len(requests)} listed data stores."
                )
            progress_dialog = MaterializationProgressDialog(
                title_text="Preparing Experiment Data Stores",
                initial_message=(
                    (
                        f"Rebuilding {len(pending_requests)} listed experiments with "
                        "the current Zarr chunking and pyramid settings."
                    )
                    if rebuild_requested
                    else (
                        f"Preparing {len(pending_requests)} of {len(requests)} listed "
                        "experiments. Existing canonical stores will be reused."
                    )
                ),
                parent=self,
            )
            failure_payload: dict[str, str] = {}
            success_results: list[ExperimentStorePreparationResult] = []

            worker = BatchDataStoreMaterializationWorker(
                requests=pending_requests,
                dask_backend=self._dask_backend_config,
                zarr_save=self._zarr_save_config,
                force_rebuild=rebuild_requested,
            )
            self._materialization_worker = worker
            worker.progress_changed.connect(progress_dialog.update_progress)
            worker.succeeded.connect(
                lambda results: success_results.extend(list(results))
            )
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

            if not success_results:
                self._set_status("Store creation was cancelled.")
                return

            prepared_count = len(success_results)
            self._set_status(
                (
                    "Rebuilt listed data stores. Opening analysis selection for the "
                    "selected experiment."
                )
                if rebuild_requested
                else (
                    "Prepared listed data stores. Opening analysis selection for the "
                    "selected experiment."
                )
            )
            logging.getLogger(__name__).info(
                (
                    "Rebuilt %s experiment data stores and reused %s existing stores."
                    if rebuild_requested
                    else "Prepared %s experiment data stores and reused %s existing stores."
                ),
                prepared_count,
                ready_count,
            )
            try:
                persisted_calibrations = self._persist_spatial_calibration_for_requests(
                    requests
                )
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Failed to persist spatial calibration for prepared stores."
                )
                _show_themed_error_dialog(
                    self,
                    "Spatial Calibration Failed",
                    "Failed to persist spatial calibration to the prepared analysis stores.",
                    summary=f"{type(exc).__name__}: {exc}",
                    details=traceback.format_exc(),
                )
                self._set_status("Failed to persist spatial calibration.")
                return
            self._accept_with_store_path(
                selected_request.target_store,
                analysis_targets=analysis_targets,
                selected_experiment_path=selected_request.experiment_path,
                spatial_calibration=persisted_calibrations[
                    Path(selected_request.target_store).expanduser().resolve()
                ],
            )

    class AnalysisSelectionDialog(QDialog):
        """Second-step GUI dialog for selecting and sequencing analysis operations."""

        _OPERATION_KEYS: tuple[str, ...] = (
            "flatfield",
            "deconvolution",
            "shear_transform",
            "registration",
            "fusion",
            "display_pyramid",
            "particle_detection",
            "usegment3d",
            "visualization",
            "render_movie",
            "compile_movie",
            "volume_export",
            "mip_export",
        )
        _PROVENANCE_HISTORY_OPERATIONS: tuple[str, ...] = (
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "fusion",
            "display_pyramid",
            "render_movie",
            "compile_movie",
            "volume_export",
            "mip_export",
        )
        _OPERATION_LABELS: Dict[str, str] = {
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shearing",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "fusion": "Fusion",
            "display_pyramid": "Pyramidal Downsampling",
            "visualization": "Napari",
            "render_movie": "Render Movie",
            "compile_movie": "Compile Movie",
            "volume_export": "Volume Export",
            "mip_export": "MIP Export",
        }
        _OPERATION_TABS: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                "Preprocessing",
                (
                    "flatfield",
                    "deconvolution",
                    "shear_transform",
                    "registration",
                    "fusion",
                    "display_pyramid",
                ),
            ),
            ("Segmentation", ("particle_detection", "usegment3d")),
            (
                "Visualization",
                (
                    "visualization",
                    "render_movie",
                    "compile_movie",
                    "volume_export",
                    "mip_export",
                ),
            ),
        )
        _OPERATION_OUTPUT_COMPONENTS: Dict[str, str] = {
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
            "shear_transform": "clearex/runtime_cache/results/shear_transform/latest/data",
            "usegment3d": "clearex/runtime_cache/results/usegment3d/latest/data",
            "registration": "clearex/results/registration/latest",
            "fusion": "clearex/runtime_cache/results/fusion/latest/data",
            "display_pyramid": "clearex/results/display_pyramid/latest",
            "visualization": "clearex/results/visualization/latest",
            "render_movie": "clearex/results/render_movie/latest",
            "compile_movie": "clearex/results/compile_movie/latest",
            "volume_export": "clearex/results/volume_export/latest",
            "mip_export": "clearex/results/mip_export/latest",
        }
        _PARTICLE_DETECTION_OVERLAY_COMPONENT = (
            "clearex/results/particle_detection/latest/detections"
        )
        _DEFAULT_USEGMENT3D_PARAMETERS: Dict[str, Any] = {
            "execution_order": 7,
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
        _DEFAULT_REGISTRATION_PARAMETERS: Dict[str, Any] = {
            "execution_order": 4,
            "input_source": "data",
            "force_rerun": False,
            "chunk_basis": "3d",
            "detect_2d_per_slice": False,
            "use_map_overlap": True,
            "pairwise_overlap_zyx": [8, 32, 32],
            "memory_overhead_factor": 2.5,
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "max_pairwise_voxels": 500000,
            "ants_iterations": [200, 100, 50, 25],
            "ants_sampling_rate": 0.20,
            "use_phase_correlation": False,
            "use_fft_initial_alignment": True,
        }
        _DEFAULT_FUSION_PARAMETERS: Dict[str, Any] = {
            "execution_order": 5,
            "input_source": "registration",
            "force_rerun": False,
            "chunk_basis": "3d",
            "detect_2d_per_slice": False,
            "use_map_overlap": True,
            "blend_overlap_zyx": [8, 32, 32],
            "memory_overhead_factor": 2.5,
            "blend_mode": "feather",
            "blend_exponent": 1.0,
            "gain_clip_range": [0.25, 4.0],
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
            "auto_estimate_t_index": ("Time index sampled for shear auto-estimation."),
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
                "Output dtype used for the shear_transform runtime-cache output."
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
            "registration_channel": (
                "Source channel used to estimate pairwise tile transforms. "
                "The solved transforms are then applied to all channels."
            ),
            "registration_type": (
                "Pairwise ANTsPy transform family for overlap registration: "
                "translation, rigid, or similarity."
            ),
            "registration_resolution_level": (
                "Input pyramid level used only for pairwise registration "
                "(0 = full resolution). Final fusion always uses full resolution."
            ),
            "registration_anchor_mode": (
                "Fix either the most central tile automatically or a manually "
                "selected anchor position during global optimization."
            ),
            "registration_anchor_position": (
                "Tile index held fixed when anchor mode is set to manual."
            ),
            "registration_pairwise_overlap_zyx": (
                "Overlap padding in (z, y, x) voxels around the estimated pairwise "
                "registration crop. Larger values make pairwise estimation more robust "
                "but increase per-worker memory."
            ),
            "fusion_blend_mode": (
                "Overlap fusion mode for the final stitched volume. Feather uses "
                "cosine edge ramps, center weighted emphasizes tile centers more "
                "strongly, content aware favors sharper regions, and "
                "gain-compensated feather estimates a per-edge intensity match "
                "before feather blending."
            ),
            "fusion_blend_overlap_zyx": (
                "Blend ramp width in (z, y, x) voxels for feathered fusion. "
                "Larger overlap widths generally hide seams better but can blur "
                "local intensity disagreements over a wider region."
            ),
            "fusion_blend_exponent": (
                "Blend-profile exponent. Values above 1.0 produce steeper "
                "edge falloff; values below 1.0 make the transition gentler."
            ),
            "fusion_gain_clip_range": (
                "Minimum and maximum allowed gain when estimating moving-to-"
                "fixed overlap intensity correction for gain-compensated feather."
            ),
            "registration_max_pairwise_voxels": (
                "Maximum voxel budget for pairwise overlap crops. Crops "
                "exceeding this budget are isotropically downsampled before "
                "ANTs estimation. Set to 0 to disable sub-sampling."
            ),
            "registration_use_phase_correlation": (
                "When enabled with translation-only registration, use FFT "
                "phase correlation instead of ANTs for faster pairwise "
                "estimation. Falls back to ANTs on failure."
            ),
            "registration_use_fft_initial_alignment": (
                "Use FFT phase correlation to pre-align the moving crop "
                "before ANTs optimization. This provides a better starting "
                "point and reduces the iterations ANTs needs to converge."
            ),
            "usegment3d_output_reference_space": (
                "Choose whether final labels are stored at level 0 (original "
                "resolution) or native selected input level."
            ),
            "usegment3d_save_native_labels": (
                "When segmenting from a downsampled level and writing level-0 labels, "
                "also save native-level labels as a ClearEx-owned auxiliary artifact."
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
            "display_pyramid": (
                "Prepare reusable pyramidal downsampling levels and stored "
                "per-channel 1/95 contrast limits for the selected source "
                "component."
            ),
            "use_multiscale": (
                "When enabled, napari uses existing display pyramids for 2D "
                "navigation. ClearEx no longer auto-builds or auto-downsamples "
                "viewer data at launch."
            ),
            "use_3d_view": (
                "Request napari 3D mode (ndisplay=3). Oversized image volumes "
                "trigger a warning and fall back to 2D instead of launch-time "
                "downsampling."
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
                "settings, and multiscale policy (inherit/require/off)."
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
                "include physical pixel-size calibration metadata; OME-Zarr exports "
                "keep projection dtype and write standalone .ome.zarr outputs with "
                "axes/scale metadata."
            ),
            "mip_output_directory": (
                "Optional export directory path. Leave empty to write under an "
                "auto-generated sibling directory next to the analysis store."
            ),
            "volume_export_scope": (
                "Choose whether volume export uses the current selection or "
                "exports all time, position, and channel indices."
            ),
            "volume_export_t_index": (
                "Time index used when exporting the current selection."
            ),
            "volume_export_p_index": (
                "Position index used when exporting the current selection."
            ),
            "volume_export_c_index": (
                "Channel index used when exporting the current selection."
            ),
            "volume_export_resolution_level": ("Pyramid resolution level to export."),
            "volume_export_export_format": (
                "Choose OME-Zarr for canonical multi-resolution output or "
                "OME-TIFF for TIFF export."
            ),
            "volume_export_tiff_file_layout": (
                "Choose whether OME-TIFF output is written as one file or as "
                "one file per volume."
            ),
            "render_movie_keyframe_manifest_path": (
                "Optional explicit keyframe manifest JSON. Leave empty to use the "
                "latest keyframes captured by the selected visualization result."
            ),
            "render_movie_resolution_levels": (
                "Comma-separated pyramid levels to render as frame sets. "
                "Use 0 for full resolution; higher integers select coarser levels."
            ),
            "render_movie_render_size_xy": (
                "Output screenshot size in pixels as width and height. "
                "Higher values improve exported frame detail but increase render time."
            ),
            "render_movie_default_transition_frames": (
                "Default number of interpolated frames inserted between consecutive "
                "keyframes when no per-gap override is provided."
            ),
            "render_movie_transition_frames_by_gap": (
                "Optional comma-separated per-gap transition counts. "
                "Gap 1 controls keyframe 1 to 2, Gap 2 controls 2 to 3, and so on."
            ),
            "render_movie_hold_frames": (
                "Extra still frames duplicated at each captured keyframe before "
                "the next transition begins."
            ),
            "render_movie_interpolation_mode": (
                "Camera and layer interpolation curve. Linear uses constant speed; "
                "Ease in/out slows at keyframes and accelerates mid-transition."
            ),
            "render_movie_camera_effect": (
                "Optional cinematic camera modifier layered on top of the keyframe "
                "path: orbit, flythrough, or zoom effect."
            ),
            "render_movie_overlay_text": (
                "Movie-only title/subtitle and frame text burned into rendered PNG "
                "frames without changing the interactive napari scene."
            ),
            "render_movie_overlay_scalebar": (
                "Draw a scalebar during movie rendering using the current rendered "
                "pixel size. Useful for publication-facing exports."
            ),
            "render_movie_output_directory": (
                "Optional frame-output directory. Leave empty to use a sibling "
                "render_movie/latest directory next to the analysis store."
            ),
            "compile_movie_render_manifest_path": (
                "Optional explicit render manifest JSON. Leave empty to compile the "
                "latest render_movie result for this store."
            ),
            "compile_movie_rendered_level": (
                "Rendered pyramid level to encode. This must match one of the frame "
                "sets produced by render_movie."
            ),
            "compile_movie_output_format": (
                "Choose review-friendly MP4, publication-friendly ProRes MOV, "
                "or both in one run."
            ),
            "compile_movie_resize_xy": (
                "Optional encode-time resize in pixels as width and height. "
                "Leave empty to preserve the rendered frame size."
            ),
            "compile_movie_output_directory": (
                "Optional movie-output directory. Leave empty to use a sibling "
                "compile_movie/latest directory next to the analysis store."
            ),
            "compile_movie_output_stem": (
                "Base filename stem used for encoded movie files."
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

            self._base_config = initial
            self._session_default_analysis_template = replace(initial)
            self._analysis_targets: tuple[AnalysisTarget, ...] = (
                _analysis_targets_for_workflow(initial)
            )
            self._active_analysis_target: Optional[AnalysisTarget] = None
            self._dask_backend_config: DaskBackendConfig = initial.dask_backend
            self.result_config: Optional[WorkflowConfig] = None
            self._analysis_scope_combo: Optional[QComboBox] = None
            self._analysis_apply_to_all_checkbox: Optional[QCheckBox] = None
            self._analysis_scope_summary_label: Optional[QLabel] = None
            self._analysis_state_source_label: Optional[QLabel] = None
            self._restore_latest_run_button: Optional[QPushButton] = None
            self._status_label: Optional[QLabel] = None
            self._dask_backend_summary_label: Optional[QLabel] = None
            self._dask_backend_button: Optional[QPushButton] = None
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
            self._registration_defaults = dict(
                self._operation_defaults.get(
                    "registration",
                    self._DEFAULT_REGISTRATION_PARAMETERS,
                )
            )
            self._operation_defaults["registration"] = dict(self._registration_defaults)
            self._fusion_defaults = dict(
                self._operation_defaults.get(
                    "fusion",
                    self._DEFAULT_FUSION_PARAMETERS,
                )
            )
            self._operation_defaults["fusion"] = dict(self._fusion_defaults)
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
            self._render_movie_defaults = dict(
                self._operation_defaults.get("render_movie", {})
            )
            self._compile_movie_defaults = dict(
                self._operation_defaults.get("compile_movie", {})
            )
            self._volume_export_defaults = dict(
                self._operation_defaults.get("volume_export", {})
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
            self._operation_history_copy_payloads: Dict[str, str] = {}
            self._operation_panel_indices: Dict[str, int] = {}
            self._parameter_help_map: Dict[QObject, str] = {}
            self._active_config_operation: Optional[str] = None

            self._operation_panel_stack: Optional[QStackedWidget] = None
            self._operation_panel_scroll: Optional[QScrollArea] = None
            self._analysis_tabs: Optional[QTabWidget] = None
            self._parameter_help_label: Optional[QLabel] = None
            self._store_label: Optional[QLabel] = None
            self._decon_measured_section: Optional[QFrame] = None
            self._decon_synthetic_section: Optional[QFrame] = None
            self._decon_light_sheet_section: Optional[QFrame] = None
            self._visualization_volume_layers_button: Optional[QPushButton] = None
            self._visualization_volume_layers_summary_label: Optional[QLabel] = None
            self._visualization_layer_table_button: Optional[QPushButton] = None
            self._visualization_layer_table_summary_label: Optional[QLabel] = None
            self._local_gpu_available = self._detect_local_gpu_available()

            self._build_ui()
            self._apply_theme()
            self._restore_analysis_state_for_target(
                _resolve_selected_analysis_target(
                    initial,
                    targets=self._analysis_targets,
                ),
                prefer_saved_gui_state=True,
            )
            _apply_initial_dialog_geometry(
                self,
                minimum_size=_ANALYSIS_DIALOG_MINIMUM_SIZE,
                preferred_size=_ANALYSIS_DIALOG_PREFERRED_SIZE,
                content_size_hint=(self.sizeHint().width(), self.sizeHint().height()),
            )

        def _build_analysis_scope_panel(self) -> QGroupBox:
            """Build the top-level experiment-scope controls.

            Parameters
            ----------
            None

            Returns
            -------
            QGroupBox
                Configured panel containing single-vs-batch analysis controls.
            """
            group = QGroupBox("Analysis Scope")
            layout = QVBoxLayout(group)
            apply_stack_spacing(layout)

            experiment_row = QHBoxLayout()
            apply_row_spacing(experiment_row)
            label = QLabel("Experiment:")
            label.setObjectName("metadataFieldLabel")
            experiment_row.addWidget(label)

            self._analysis_scope_combo = QComboBox()
            self._analysis_scope_combo.setObjectName("analysisScopeCombo")
            self._analysis_scope_combo.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            self._analysis_scope_combo.currentIndexChanged.connect(
                self._on_analysis_target_changed
            )
            experiment_row.addWidget(self._analysis_scope_combo, 1)
            layout.addLayout(experiment_row)

            self._analysis_apply_to_all_checkbox = QCheckBox(
                "Run selected operations on all loaded experiments"
            )
            self._analysis_apply_to_all_checkbox.toggled.connect(
                self._on_analysis_apply_to_all_changed
            )
            layout.addWidget(self._analysis_apply_to_all_checkbox)

            self._analysis_scope_summary_label = QLabel(
                "Select one experiment or enable batch mode."
            )
            self._analysis_scope_summary_label.setObjectName("statusLabel")
            self._analysis_scope_summary_label.setWordWrap(True)
            self._analysis_scope_summary_label.setMinimumHeight(
                max(
                    28,
                    int(self._analysis_scope_summary_label.fontMetrics().height()) + 10,
                )
            )
            layout.addWidget(self._analysis_scope_summary_label)

            restore_row = QHBoxLayout()
            apply_row_spacing(restore_row)
            self._analysis_state_source_label = QLabel(
                "Using default analysis parameters for this dataset."
            )
            self._analysis_state_source_label.setObjectName("statusLabel")
            self._analysis_state_source_label.setWordWrap(True)
            self._analysis_state_source_label.setMinimumHeight(
                max(
                    28,
                    int(self._analysis_state_source_label.fontMetrics().height()) + 10,
                )
            )
            restore_row.addWidget(self._analysis_state_source_label, 1)

            self._restore_latest_run_button = QPushButton(
                "Restore Latest Run Parameters"
            )
            self._restore_latest_run_button.setMinimumHeight(36)
            self._restore_latest_run_button.setSizePolicy(
                QSizePolicy.Policy.Maximum,
                QSizePolicy.Policy.Fixed,
            )
            self._restore_latest_run_button.clicked.connect(
                self._on_restore_latest_run_parameters
            )
            restore_row.addWidget(self._restore_latest_run_button)
            layout.addLayout(restore_row)
            return group

        def _populate_analysis_scope_targets(self, initial: WorkflowConfig) -> None:
            """Populate analysis-scope widgets from workflow state.

            Parameters
            ----------
            initial : WorkflowConfig
                Initial workflow carrying available targets and selection state.

            Returns
            -------
            None
                Scope controls are updated in-place.
            """
            selected_target = _resolve_selected_analysis_target(
                initial,
                targets=self._analysis_targets,
            )
            if self._analysis_scope_combo is not None:
                self._analysis_scope_combo.blockSignals(True)
                self._analysis_scope_combo.clear()
                selected_index = -1
                for index, target in enumerate(self._analysis_targets):
                    display_text = _analysis_target_display_text(target)
                    self._analysis_scope_combo.addItem(
                        display_text,
                        target.experiment_path,
                    )
                    self._analysis_scope_combo.setItemData(
                        index,
                        target.store_path,
                        Qt.ItemDataRole.ToolTipRole,
                    )
                    if (
                        selected_target is not None
                        and target.experiment_path == selected_target.experiment_path
                    ):
                        selected_index = index
                if selected_index >= 0:
                    self._analysis_scope_combo.setCurrentIndex(selected_index)
                self._analysis_scope_combo.setEnabled(bool(self._analysis_targets))
                self._analysis_scope_combo.blockSignals(False)
            if self._analysis_apply_to_all_checkbox is not None:
                self._analysis_apply_to_all_checkbox.blockSignals(True)
                self._analysis_apply_to_all_checkbox.setChecked(
                    bool(initial.analysis_apply_to_all)
                )
                self._analysis_apply_to_all_checkbox.setEnabled(
                    len(self._analysis_targets) > 1
                )
                self._analysis_apply_to_all_checkbox.blockSignals(False)
            self._set_active_analysis_target(selected_target, refresh_context=False)
            self._refresh_analysis_scope_summary()

        def _current_analysis_target(self) -> Optional[AnalysisTarget]:
            """Return the currently selected analysis target.

            Parameters
            ----------
            None

            Returns
            -------
            AnalysisTarget, optional
                Active experiment/store target.
            """
            if self._analysis_scope_combo is not None:
                selected_experiment_path = str(
                    self._analysis_scope_combo.currentData() or ""
                ).strip()
                if selected_experiment_path:
                    for target in self._analysis_targets:
                        if target.experiment_path == selected_experiment_path:
                            return target
            return self._active_analysis_target

        def _set_active_analysis_target(
            self,
            target: Optional[AnalysisTarget],
            *,
            refresh_context: bool,
        ) -> None:
            """Switch the analysis dialog to a specific experiment/store target.

            Parameters
            ----------
            target : AnalysisTarget, optional
                Target to activate.
            refresh_context : bool
                Whether to refresh provenance/input-source state immediately.

            Returns
            -------
            None
                Active analysis-store context is updated in-place.
            """
            if target is None:
                self._active_analysis_target = None
                if self._store_label is not None:
                    store_text = str(self._base_config.file or "").strip() or "n/a"
                    self._store_label.setText(store_text)
                    self._store_label.setToolTip(store_text)
                self._refresh_analysis_scope_summary()
                return

            self._active_analysis_target = target
            self._base_config.file = str(target.store_path)
            self._base_config.analysis_selected_experiment_path = str(
                target.experiment_path
            )
            if self._analysis_scope_combo is not None:
                self._analysis_scope_combo.setToolTip(str(target.experiment_path))
            if self._store_label is not None:
                self._store_label.setText(str(target.store_path))
                self._store_label.setToolTip(str(target.store_path))
            self._refresh_analysis_scope_summary()
            if refresh_context:
                self._refresh_operation_provenance_statuses()

        def _refresh_analysis_scope_summary(self) -> None:
            """Refresh descriptive text for the analysis-scope panel.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary label text is updated in-place.
            """
            if self._analysis_scope_summary_label is None:
                return
            apply_to_all = bool(
                self._analysis_apply_to_all_checkbox.isChecked()
                if self._analysis_apply_to_all_checkbox is not None
                else False
            )
            summary = _analysis_scope_summary_text(
                targets=self._analysis_targets,
                selected_target=self._current_analysis_target(),
                apply_to_all=apply_to_all,
            )
            self._analysis_scope_summary_label.setText(summary)
            self._analysis_scope_summary_label.setToolTip(summary)

        def _current_analysis_apply_to_all(self) -> bool:
            """Return whether batch-analysis mode is currently enabled.

            Parameters
            ----------
            None

            Returns
            -------
            bool
                ``True`` when batch-analysis mode is enabled.
            """
            if self._analysis_apply_to_all_checkbox is None:
                return bool(self._base_config.analysis_apply_to_all)
            return bool(self._analysis_apply_to_all_checkbox.isChecked())

        def _build_target_default_workflow(
            self,
            target: Optional[AnalysisTarget],
        ) -> WorkflowConfig:
            """Build the default workflow template for one analysis target.

            Parameters
            ----------
            target : AnalysisTarget, optional
                Experiment/store target being restored.

            Returns
            -------
            WorkflowConfig
                Target-aware default workflow template.
            """
            file_path = (
                str(target.store_path)
                if target is not None
                else str(self._base_config.file or "").strip() or None
            )
            selected_experiment_path = (
                str(target.experiment_path)
                if target is not None
                else (
                    str(
                        self._base_config.analysis_selected_experiment_path or ""
                    ).strip()
                    or None
                )
            )
            spatial_calibration = self._base_config.spatial_calibration
            if target is not None and is_zarr_store_path(target.store_path):
                try:
                    spatial_calibration = load_store_spatial_calibration(
                        target.store_path
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to load store spatial calibration for %s.",
                        target.store_path,
                    )
            return replace(
                self._session_default_analysis_template,
                file=file_path,
                analysis_targets=self._analysis_targets,
                analysis_selected_experiment_path=selected_experiment_path,
                analysis_apply_to_all=self._current_analysis_apply_to_all(),
                dask_backend=self._dask_backend_config,
                spatial_calibration=spatial_calibration,
                spatial_calibration_explicit=False,
            )

        def _analysis_state_workflow_from_payload(
            self,
            *,
            target: Optional[AnalysisTarget],
            payload: Mapping[str, Any],
        ) -> WorkflowConfig:
            """Merge persisted GUI/provenance payload into a target workflow.

            Parameters
            ----------
            target : AnalysisTarget, optional
                Experiment/store target being restored.
            payload : mapping[str, Any]
                Persisted workflow-like payload.

            Returns
            -------
            WorkflowConfig
                Restored workflow state for the active target.
            """
            default_workflow = self._build_target_default_workflow(target)
            selected_flags = {
                operation_name: bool(
                    payload.get(
                        operation_name, getattr(default_workflow, operation_name)
                    )
                )
                for operation_name in self._OPERATION_KEYS
            }
            analysis_parameters = normalize_analysis_operation_parameters(
                payload.get(
                    "analysis_parameters",
                    default_workflow.analysis_parameters,
                )
            )
            spatial_calibration_payload = payload.get(
                "spatial_calibration",
                default_workflow.spatial_calibration,
            )
            return replace(
                default_workflow,
                flatfield=selected_flags["flatfield"],
                deconvolution=selected_flags["deconvolution"],
                shear_transform=selected_flags["shear_transform"],
                particle_detection=selected_flags["particle_detection"],
                usegment3d=selected_flags["usegment3d"],
                registration=selected_flags["registration"],
                fusion=selected_flags["fusion"],
                visualization=selected_flags["visualization"],
                volume_export=selected_flags["volume_export"],
                mip_export=selected_flags["mip_export"],
                spatial_calibration=normalize_spatial_calibration(
                    spatial_calibration_payload
                ),
                spatial_calibration_explicit=bool(
                    payload.get(
                        "spatial_calibration_explicit",
                        default_workflow.spatial_calibration_explicit,
                    )
                ),
                analysis_parameters=analysis_parameters,
            )

        def _analysis_gui_state_payload_from_current_widgets(self) -> Dict[str, Any]:
            """Collect dataset-local GUI state from current analysis widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Persistable mapping of selected flags and analysis parameters.

            Raises
            ------
            ValueError
                If any current widget values are invalid.
            """
            payload: Dict[str, Any] = {
                operation_name: bool(
                    self._operation_checkboxes[operation_name].isChecked()
                )
                for operation_name in self._OPERATION_KEYS
            }
            analysis_parameters = normalize_analysis_operation_parameters(
                self._base_config.analysis_parameters
            )
            for operation_name in self._OPERATION_KEYS:
                analysis_parameters[operation_name] = (
                    self._collect_operation_parameters(operation_name)
                )
            payload["analysis_parameters"] = normalize_analysis_operation_parameters(
                analysis_parameters
            )
            target = self._current_analysis_target()
            if target is not None:
                payload["file"] = str(target.store_path)
                payload["analysis_selected_experiment_path"] = str(
                    target.experiment_path
                )
            payload["spatial_calibration"] = spatial_calibration_to_dict(
                self._base_config.spatial_calibration
            )
            payload["spatial_calibration_explicit"] = bool(
                self._base_config.spatial_calibration_explicit
            )
            analysis_selected_tab = self._current_analysis_tab_name()
            if analysis_selected_tab:
                payload["analysis_selected_tab"] = analysis_selected_tab
            return payload

        def _current_analysis_tab_name(self) -> str:
            """Return the visible Analysis Methods tab label.

            Parameters
            ----------
            None

            Returns
            -------
            str
                Current tab label, or ``""`` when the widget is unavailable.
            """
            if self._analysis_tabs is None:
                return ""
            current_index = int(self._analysis_tabs.currentIndex())
            if current_index < 0:
                return ""
            return str(self._analysis_tabs.tabText(current_index) or "").strip()

        def _restore_analysis_tab(self, tab_name: str) -> None:
            """Restore the selected Analysis Methods tab by label.

            Parameters
            ----------
            tab_name : str
                Previously persisted tab label.

            Returns
            -------
            None
                The tab widget selection is updated in-place when the label
                matches an existing tab.
            """
            if self._analysis_tabs is None:
                return
            normalized_name = str(tab_name or "").strip().casefold()
            if not normalized_name:
                return
            for tab_index in range(self._analysis_tabs.count()):
                current_name = (
                    str(self._analysis_tabs.tabText(tab_index) or "").strip().casefold()
                )
                if current_name == normalized_name:
                    self._analysis_tabs.setCurrentIndex(tab_index)
                    return

        def _persist_analysis_gui_state_for_target(
            self,
            target: Optional[AnalysisTarget],
        ) -> None:
            """Persist current GUI analysis state for one dataset.

            Parameters
            ----------
            target : AnalysisTarget, optional
                Dataset target to persist.

            Returns
            -------
            None
                Persistence is best-effort and handled in-place.
            """
            if target is None:
                return
            store_path = str(target.store_path).strip()
            if not store_path or not is_zarr_store_path(store_path):
                return
            try:
                payload = self._analysis_gui_state_payload_from_current_widgets()
            except ValueError as exc:
                logging.getLogger(__name__).debug(
                    "Skipping GUI-state persistence for %s because current analysis "
                    "parameters are invalid: %s",
                    store_path,
                    exc,
                )
                return
            try:
                persist_latest_analysis_gui_state(store_path, payload)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to persist analysis GUI state for %s.",
                    store_path,
                )

        def _set_analysis_state_source_text(
            self,
            text: str,
            *,
            has_latest_run: bool,
        ) -> None:
            """Update the analysis-state restore summary and button state.

            Parameters
            ----------
            text : str
                Human-readable source description.
            has_latest_run : bool
                Whether a completed-run restore target is available.

            Returns
            -------
            None
                Summary widgets are updated in-place.
            """
            if self._analysis_state_source_label is not None:
                self._analysis_state_source_label.setText(str(text))
                self._analysis_state_source_label.setToolTip(str(text))
            if self._restore_latest_run_button is not None:
                self._restore_latest_run_button.setEnabled(bool(has_latest_run))

        def _restore_analysis_state_for_target(
            self,
            target: Optional[AnalysisTarget],
            *,
            prefer_saved_gui_state: bool,
        ) -> None:
            """Restore GUI analysis state for the selected dataset target.

            Parameters
            ----------
            target : AnalysisTarget, optional
                Dataset target to restore.
            prefer_saved_gui_state : bool
                When ``True``, prefer unsaved GUI state over latest completed
                provenance workflow parameters.

            Returns
            -------
            None
                The dialog is rehydrated in-place.
            """
            default_workflow = self._build_target_default_workflow(target)
            restored_workflow = default_workflow
            source_text = "Using default analysis parameters for this dataset."
            latest_completed_state: Optional[Dict[str, Any]] = None
            saved_gui_state: Optional[Dict[str, Any]] = None

            store_path = (
                str(target.store_path).strip()
                if target is not None
                else str(default_workflow.file or "").strip()
            )
            if store_path and is_zarr_store_path(store_path):
                try:
                    latest_completed_state = load_latest_completed_workflow_state(
                        store_path
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to load latest completed workflow state for %s.",
                        store_path,
                    )
                try:
                    saved_gui_state = load_latest_analysis_gui_state(store_path)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to load saved analysis GUI state for %s.",
                        store_path,
                    )

            if (
                prefer_saved_gui_state
                and saved_gui_state is not None
                and isinstance(saved_gui_state.get("workflow"), Mapping)
            ):
                restored_workflow = self._analysis_state_workflow_from_payload(
                    target=target,
                    payload=saved_gui_state["workflow"],
                )
                source_text = "Restored unsaved GUI state from this dataset."
            elif latest_completed_state is not None and isinstance(
                latest_completed_state.get("workflow"), Mapping
            ):
                restored_workflow = self._analysis_state_workflow_from_payload(
                    target=target,
                    payload=latest_completed_state["workflow"],
                )
                source_text = (
                    "Loaded parameters from the latest completed run for this dataset."
                )
            elif store_path and not is_zarr_store_path(store_path):
                source_text = "Persistence is unavailable for this data source."

            self._hydrate(restored_workflow)
            restored_tab_name = ""
            if (
                prefer_saved_gui_state
                and saved_gui_state is not None
                and isinstance(saved_gui_state.get("workflow"), Mapping)
            ):
                restored_tab_name = str(
                    saved_gui_state["workflow"].get("analysis_selected_tab", "") or ""
                ).strip()
            elif latest_completed_state is not None and isinstance(
                latest_completed_state.get("workflow"), Mapping
            ):
                restored_tab_name = str(
                    latest_completed_state["workflow"].get(
                        "analysis_selected_tab",
                        "",
                    )
                    or ""
                ).strip()
                if (
                    not restored_tab_name
                    and saved_gui_state is not None
                    and isinstance(saved_gui_state.get("workflow"), Mapping)
                ):
                    restored_tab_name = str(
                        saved_gui_state["workflow"].get("analysis_selected_tab", "")
                        or ""
                    ).strip()
            self._restore_analysis_tab(restored_tab_name)
            self._base_config.analysis_parameters = (
                normalize_analysis_operation_parameters(
                    restored_workflow.analysis_parameters
                )
            )
            for operation_name in self._OPERATION_KEYS:
                setattr(
                    self._base_config,
                    operation_name,
                    bool(getattr(restored_workflow, operation_name)),
                )
            self._base_config.analysis_apply_to_all = bool(
                restored_workflow.analysis_apply_to_all
            )
            self._base_config.spatial_calibration = (
                restored_workflow.spatial_calibration
            )
            self._base_config.spatial_calibration_explicit = bool(
                restored_workflow.spatial_calibration_explicit
            )
            self._set_analysis_state_source_text(
                source_text,
                has_latest_run=latest_completed_state is not None,
            )

        def _on_restore_latest_run_parameters(self) -> None:
            """Restore the latest completed provenance workflow for the target.

            Parameters
            ----------
            None

            Returns
            -------
            None
                The selected dataset is rehydrated in-place.
            """
            target = self._current_analysis_target()
            if target is None:
                self._set_status("No analysis target is selected.")
                return
            self._restore_analysis_state_for_target(
                target,
                prefer_saved_gui_state=False,
            )
            self._set_status(
                "Restored parameters from the latest completed run for the selected dataset."
            )

        def _on_analysis_target_changed(self, index: int) -> None:
            """Handle experiment selection changes in the analysis-scope combo.

            Parameters
            ----------
            index : int
                Newly selected combo-box index.

            Returns
            -------
            None
                Active target and store-dependent status widgets are refreshed.
            """
            del index
            previous_target = self._active_analysis_target
            if previous_target is not None:
                self._persist_analysis_gui_state_for_target(previous_target)
            self._restore_analysis_state_for_target(
                self._current_analysis_target(),
                prefer_saved_gui_state=True,
            )

        def _on_analysis_apply_to_all_changed(self, checked: bool) -> None:
            """Refresh scope text when batch-analysis mode toggles.

            Parameters
            ----------
            checked : bool
                Whether batch-analysis mode is enabled.

            Returns
            -------
            None
                Scope text and status message are updated in-place.
            """
            self._base_config.analysis_apply_to_all = bool(checked)
            self._refresh_analysis_scope_summary()
            if checked and len(self._analysis_targets) > 1:
                self._set_status(
                    f"Batch analysis enabled for {len(self._analysis_targets)} experiments."
                )
            else:
                self._set_status("Configure selected analysis routines and click Run.")

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
            outer_root = QVBoxLayout(self)
            apply_window_root_spacing(outer_root)

            header = QFrame()
            header.setObjectName("headerCard")
            header_layout = QVBoxLayout(header)
            apply_stack_spacing(header_layout)
            header_image = _create_scaled_branding_label(
                filename=_GUI_HEADER_IMAGE,
                max_width=920,
                max_height=80,
                object_name="brandHeaderImage",
            )
            if header_image is not None:
                header_layout.addWidget(header_image)
            outer_root.addWidget(header)

            content_scroll = QScrollArea(self)
            content_scroll.setObjectName("analysisDialogScroll")
            content_scroll.setWidgetResizable(True)
            content_scroll.setFrameShape(QFrame.Shape.NoFrame)
            content_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            outer_root.addWidget(content_scroll, 1)

            content_widget = QWidget()
            content_widget.setObjectName("analysisDialogContent")
            content_scroll.setWidget(content_widget)

            root = QVBoxLayout(content_widget)
            apply_stack_spacing(root)

            root.addWidget(self._build_analysis_scope_panel())

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

            self._analysis_tabs = QTabWidget()
            self._analysis_tabs.setObjectName("analysisTabs")
            self._analysis_tabs.setMinimumWidth(500)
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
                self._analysis_tabs.addTab(tab_widget, tab_name)

            analysis_hint = QLabel(
                "Use Configure to edit one operation at a time. "
                "Only selected operations can be configured."
            )
            analysis_hint.setObjectName("statusLabel")
            analysis_hint.setWordWrap(True)
            analysis_hint.setMinimumHeight(
                max(28, int(analysis_hint.fontMetrics().height()) + 10)
            )
            left_column.addWidget(self._analysis_tabs, 1)
            left_column.addWidget(analysis_hint)
            content_row.addLayout(left_column, 1)

            parameters_group = QGroupBox("Operation Parameters")
            parameters_layout = QVBoxLayout(parameters_group)
            apply_stack_spacing(parameters_layout)

            self._operation_panel_stack = _CurrentPageStackedWidget()
            self._operation_panel_stack.setObjectName("operationPanelStack")
            self._operation_panel_stack.setAttribute(
                Qt.WidgetAttribute.WA_StyledBackground,
                True,
            )
            self._operation_panel_stack.addWidget(self._build_no_selection_panel())
            for operation_name in self._OPERATION_KEYS:
                panel = self._build_operation_panel(operation_name)
                panel_index = self._operation_panel_stack.addWidget(panel)
                self._operation_panel_indices[operation_name] = panel_index
            self._operation_panel_scroll = QScrollArea()
            _configure_themed_scroll_area_surface(
                self._operation_panel_scroll,
                scroll_object_name="operationPanelScroll",
                viewport_object_name="operationPanelViewport",
            )
            self._operation_panel_scroll.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
            )
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

            footer_frame = QFrame()
            footer_frame.setObjectName("analysisFooterCard")
            footer_frame.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
            footer_root = QVBoxLayout(footer_frame)
            footer_root.setContentsMargins(0, 0, 0, 0)
            footer_root.setSpacing(0)

            footer = QHBoxLayout()
            apply_footer_row_spacing(footer)
            status_stack = QVBoxLayout()
            apply_help_stack_spacing(status_stack)
            self._status_label = QLabel(
                "Configure selected analysis routines and click Run."
            )
            self._status_label.setObjectName("statusLabel")
            self._status_label.setWordWrap(True)
            self._status_label.setMinimumHeight(
                max(28, int(self._status_label.fontMetrics().height()) + 10)
            )
            status_stack.addWidget(self._status_label)
            self._dask_backend_summary_label = QLabel("Dask backend: n/a")
            self._dask_backend_summary_label.setObjectName("statusLabel")
            self._dask_backend_summary_label.setWordWrap(True)
            self._dask_backend_summary_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._dask_backend_summary_label.setMinimumHeight(
                max(
                    28,
                    int(self._dask_backend_summary_label.fontMetrics().height()) + 10,
                )
            )
            status_stack.addWidget(self._dask_backend_summary_label)
            footer.addLayout(status_stack, 1)
            self._dask_backend_button = QPushButton("Edit Dask Backend")
            self._cancel_button = QPushButton("Cancel")
            self._run_button = QPushButton("Run")
            self._run_button.setObjectName("runButton")
            for button in (
                self._dask_backend_button,
                self._cancel_button,
                self._run_button,
            ):
                button.setMinimumHeight(36)
                button.setSizePolicy(
                    QSizePolicy.Policy.Maximum,
                    QSizePolicy.Policy.Fixed,
                )
            footer.addWidget(self._dask_backend_button)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._run_button)
            footer_root.addLayout(footer)
            outer_root.addWidget(footer_frame)

            self._dask_backend_button.clicked.connect(self._on_edit_dask_backend)
            self._cancel_button.clicked.connect(self.reject)
            self._run_button.clicked.connect(self._on_run)
            content_widget.setMinimumHeight(root.sizeHint().height())
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
            self._registration_anchor_mode_combo.currentIndexChanged.connect(
                self._set_registration_parameter_enabled_state
            )
            self._fusion_blend_mode_combo.currentIndexChanged.connect(
                self._set_fusion_parameter_enabled_state
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
                history_label = QLabel("No Provenance Record")
                history_label.setObjectName("provenanceStatusLabel")
                history_label.setProperty("state", "missing")
                history_label.setContentsMargins(24, 0, 0, 0)
                history_label.setToolTip("No successful provenance run recorded yet.")
                history_label.setContextMenuPolicy(
                    Qt.ContextMenuPolicy.CustomContextMenu
                )
                history_label.customContextMenuRequested.connect(
                    lambda pos, op=operation_name: (
                        self._on_operation_history_context_menu(op, pos)
                    )
                )
                parent_layout.addWidget(history_label)
                self._operation_history_labels[operation_name] = history_label
                self._operation_history_copy_payloads[operation_name] = ""

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
            widget.setObjectName("operationPanel")
            widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            layout = QVBoxLayout(widget)
            apply_stack_spacing(layout)
            text = QLabel(
                "Select an analysis routine on the left, then click Configure to open "
                "its operation parameters."
            )
            text.setWordWrap(True)
            text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            text.setObjectName("parameterHint")
            layout.addWidget(text)
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
            panel.setObjectName("operationPanel")
            panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            layout = QVBoxLayout(panel)
            apply_stack_spacing(layout)

            profile = QLabel(
                f"{self._OPERATION_LABELS[operation_name]} Configuration\n"
                f"Output component: {self._operation_output_component(operation_name)}"
            )
            profile.setObjectName("parameterHint")
            profile.setWordWrap(True)
            profile.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
            )
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
            if operation_name == "visualization":
                input_combo.currentIndexChanged.connect(
                    self._on_visualization_input_source_changed
                )
            if operation_name == "registration":
                input_combo.currentIndexChanged.connect(
                    self._on_registration_input_source_changed
                )
            if operation_name == "volume_export":
                input_combo.currentIndexChanged.connect(
                    self._on_volume_export_input_source_changed
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
            elif operation_name == "registration":
                self._build_registration_parameter_rows(form)
            elif operation_name == "fusion":
                self._build_fusion_parameter_rows(form)
            elif operation_name == "visualization":
                self._build_visualization_parameter_rows(form)
            elif operation_name == "render_movie":
                self._build_render_movie_parameter_rows(form)
            elif operation_name == "compile_movie":
                self._build_compile_movie_parameter_rows(form)
            elif operation_name == "volume_export":
                self._build_volume_export_parameter_rows(form)
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
            form.addRow(
                "auto_estimate_zy_stride", self._shear_auto_estimate_stride_spin
            )
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
            _configure_themed_scroll_area_surface(
                self._usegment3d_channel_scroll,
                scroll_object_name="usegment3dChannelScroll",
                viewport_object_name="usegment3dChannelViewport",
            )
            self._usegment3d_channel_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._usegment3d_channel_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._usegment3d_channel_container = QWidget()
            self._usegment3d_channel_container.setObjectName(
                "usegment3dChannelContainer"
            )
            self._usegment3d_channel_container.setAttribute(
                Qt.WidgetAttribute.WA_StyledBackground,
                True,
            )
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

        def _build_registration_parameter_rows(self, form: QFormLayout) -> None:
            """Add registration parameter controls to a form.

            Parameters
            ----------
            form : QFormLayout
                Parent form layout receiving registration controls.

            Returns
            -------
            None
                Widgets are created and attached in-place.
            """
            pairwise_section, pairwise_form = self._build_parameter_section_card(
                "Pairwise Registration"
            )
            self._registration_channel_combo = QComboBox()
            self._registration_channel_combo.addItem("Channel 0", 0)
            pairwise_form.addRow("channel", self._registration_channel_combo)
            self._register_parameter_hint(
                self._registration_channel_combo,
                self._PARAMETER_HINTS["registration_channel"],
            )

            self._registration_type_combo = QComboBox()
            self._registration_type_combo.addItem("Translation", "translation")
            self._registration_type_combo.addItem("Rigid", "rigid")
            self._registration_type_combo.addItem("Similarity", "similarity")
            pairwise_form.addRow("type", self._registration_type_combo)
            self._register_parameter_hint(
                self._registration_type_combo,
                self._PARAMETER_HINTS["registration_type"],
            )

            self._registration_resolution_level_combo = QComboBox()
            self._registration_resolution_level_combo.addItem("Level 0", 0)
            pairwise_form.addRow(
                "resolution level",
                self._registration_resolution_level_combo,
            )
            self._register_parameter_hint(
                self._registration_resolution_level_combo,
                self._PARAMETER_HINTS["registration_resolution_level"],
            )
            overlap_row = QHBoxLayout()
            apply_compact_row_spacing(overlap_row)
            self._registration_pairwise_overlap_z_spin = QSpinBox()
            self._registration_pairwise_overlap_z_spin.setRange(0, 1_000_000)
            self._registration_pairwise_overlap_y_spin = QSpinBox()
            self._registration_pairwise_overlap_y_spin.setRange(0, 1_000_000)
            self._registration_pairwise_overlap_x_spin = QSpinBox()
            self._registration_pairwise_overlap_x_spin.setRange(0, 1_000_000)
            overlap_row.addWidget(QLabel("z"))
            overlap_row.addWidget(self._registration_pairwise_overlap_z_spin)
            overlap_row.addWidget(QLabel("y"))
            overlap_row.addWidget(self._registration_pairwise_overlap_y_spin)
            overlap_row.addWidget(QLabel("x"))
            overlap_row.addWidget(self._registration_pairwise_overlap_x_spin)
            overlap_widget = QWidget()
            overlap_widget.setLayout(overlap_row)
            pairwise_form.addRow("overlap pad", overlap_widget)
            self._register_parameter_hint(
                self._registration_pairwise_overlap_z_spin,
                self._PARAMETER_HINTS["registration_pairwise_overlap_zyx"],
            )
            self._register_parameter_hint(
                self._registration_pairwise_overlap_y_spin,
                self._PARAMETER_HINTS["registration_pairwise_overlap_zyx"],
            )
            self._register_parameter_hint(
                self._registration_pairwise_overlap_x_spin,
                self._PARAMETER_HINTS["registration_pairwise_overlap_zyx"],
            )
            form.addRow(pairwise_section)

            global_section, global_form = self._build_parameter_section_card(
                "Global Optimization"
            )
            self._registration_anchor_mode_combo = QComboBox()
            self._registration_anchor_mode_combo.addItem("Central tile", "central")
            self._registration_anchor_mode_combo.addItem("Manual tile", "manual")
            global_form.addRow("anchor mode", self._registration_anchor_mode_combo)
            self._register_parameter_hint(
                self._registration_anchor_mode_combo,
                self._PARAMETER_HINTS["registration_anchor_mode"],
            )

            self._registration_anchor_position_spin = QSpinBox()
            self._registration_anchor_position_spin.setRange(0, 0)
            self._registration_anchor_position_label = QLabel("anchor tile")
            global_form.addRow(
                self._registration_anchor_position_label,
                self._registration_anchor_position_spin,
            )
            self._register_parameter_hint(
                self._registration_anchor_position_spin,
                self._PARAMETER_HINTS["registration_anchor_position"],
            )
            form.addRow(global_section)

            perf_section, perf_form = self._build_parameter_section_card("Performance")
            self._registration_max_pairwise_voxels_spin = QSpinBox()
            self._registration_max_pairwise_voxels_spin.setRange(0, 100_000_000)
            self._registration_max_pairwise_voxels_spin.setSingleStep(50_000)
            perf_form.addRow(
                "max pairwise voxels",
                self._registration_max_pairwise_voxels_spin,
            )
            self._register_parameter_hint(
                self._registration_max_pairwise_voxels_spin,
                self._PARAMETER_HINTS["registration_max_pairwise_voxels"],
            )

            self._registration_use_phase_correlation_check = QCheckBox(
                "phase correlation (translation only)"
            )
            perf_form.addRow(self._registration_use_phase_correlation_check)
            self._register_parameter_hint(
                self._registration_use_phase_correlation_check,
                self._PARAMETER_HINTS["registration_use_phase_correlation"],
            )

            self._registration_use_fft_initial_alignment_check = QCheckBox(
                "FFT initial alignment"
            )
            self._registration_use_fft_initial_alignment_check.setChecked(True)
            perf_form.addRow(self._registration_use_fft_initial_alignment_check)
            self._register_parameter_hint(
                self._registration_use_fft_initial_alignment_check,
                self._PARAMETER_HINTS["registration_use_fft_initial_alignment"],
            )
            form.addRow(perf_section)

        def _build_fusion_parameter_rows(self, form: QFormLayout) -> None:
            """Add fusion parameter controls to a form."""
            blend_section, blend_form = self._build_parameter_section_card("Blending")
            overlap_row = QHBoxLayout()
            apply_compact_row_spacing(overlap_row)
            self._fusion_overlap_z_spin = QSpinBox()
            self._fusion_overlap_z_spin.setRange(0, 1_000_000)
            self._fusion_overlap_y_spin = QSpinBox()
            self._fusion_overlap_y_spin.setRange(0, 1_000_000)
            self._fusion_overlap_x_spin = QSpinBox()
            self._fusion_overlap_x_spin.setRange(0, 1_000_000)
            overlap_row.addWidget(QLabel("z"))
            overlap_row.addWidget(self._fusion_overlap_z_spin)
            overlap_row.addWidget(QLabel("y"))
            overlap_row.addWidget(self._fusion_overlap_y_spin)
            overlap_row.addWidget(QLabel("x"))
            overlap_row.addWidget(self._fusion_overlap_x_spin)
            overlap_widget = QWidget()
            overlap_widget.setLayout(overlap_row)
            blend_form.addRow("blend overlap", overlap_widget)
            self._register_parameter_hint(
                self._fusion_overlap_z_spin,
                self._PARAMETER_HINTS["fusion_blend_overlap_zyx"],
            )
            self._register_parameter_hint(
                self._fusion_overlap_y_spin,
                self._PARAMETER_HINTS["fusion_blend_overlap_zyx"],
            )
            self._register_parameter_hint(
                self._fusion_overlap_x_spin,
                self._PARAMETER_HINTS["fusion_blend_overlap_zyx"],
            )

            self._fusion_blend_mode_combo = QComboBox()
            self._fusion_blend_mode_combo.addItem("Feather", "feather")
            self._fusion_blend_mode_combo.addItem("Center weighted", "center_weighted")
            self._fusion_blend_mode_combo.addItem("Content aware", "content_aware")
            self._fusion_blend_mode_combo.addItem(
                "Gain-compensated feather",
                "gain_compensated_feather",
            )
            self._fusion_blend_mode_combo.addItem("Average", "average")
            blend_form.addRow("blend mode", self._fusion_blend_mode_combo)
            self._register_parameter_hint(
                self._fusion_blend_mode_combo,
                self._PARAMETER_HINTS["fusion_blend_mode"],
            )

            self._fusion_blend_exponent_spin = QDoubleSpinBox()
            self._fusion_blend_exponent_spin.setDecimals(2)
            self._fusion_blend_exponent_spin.setRange(0.10, 8.0)
            self._fusion_blend_exponent_spin.setSingleStep(0.10)
            self._fusion_blend_exponent_spin.setValue(1.0)
            blend_form.addRow("blend exponent", self._fusion_blend_exponent_spin)
            self._register_parameter_hint(
                self._fusion_blend_exponent_spin,
                self._PARAMETER_HINTS["fusion_blend_exponent"],
            )

            gain_clip_row = QHBoxLayout()
            apply_compact_row_spacing(gain_clip_row)
            self._fusion_gain_clip_min_spin = QDoubleSpinBox()
            self._fusion_gain_clip_min_spin.setDecimals(2)
            self._fusion_gain_clip_min_spin.setRange(0.01, 100.0)
            self._fusion_gain_clip_min_spin.setSingleStep(0.05)
            self._fusion_gain_clip_min_spin.setValue(0.25)
            self._fusion_gain_clip_max_spin = QDoubleSpinBox()
            self._fusion_gain_clip_max_spin.setDecimals(2)
            self._fusion_gain_clip_max_spin.setRange(0.01, 100.0)
            self._fusion_gain_clip_max_spin.setSingleStep(0.05)
            self._fusion_gain_clip_max_spin.setValue(4.0)
            gain_clip_row.addWidget(QLabel("min"))
            gain_clip_row.addWidget(self._fusion_gain_clip_min_spin)
            gain_clip_row.addWidget(QLabel("max"))
            gain_clip_row.addWidget(self._fusion_gain_clip_max_spin)
            gain_clip_widget = QWidget()
            gain_clip_widget.setLayout(gain_clip_row)
            blend_form.addRow("gain clip", gain_clip_widget)
            self._register_parameter_hint(
                self._fusion_gain_clip_min_spin,
                self._PARAMETER_HINTS["fusion_gain_clip_range"],
            )
            self._register_parameter_hint(
                self._fusion_gain_clip_max_spin,
                self._PARAMETER_HINTS["fusion_gain_clip_range"],
            )
            form.addRow(blend_section)

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

            self._visualization_3d_checkbox = QCheckBox("Launch in 3D mode")
            self._visualization_3d_checkbox.setChecked(True)
            form.addRow("Display mode", self._visualization_3d_checkbox)
            self._register_parameter_hint(
                self._visualization_3d_checkbox,
                self._PARAMETER_HINTS["use_3d_view"],
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

        def _build_render_movie_parameter_rows(self, form: QFormLayout) -> None:
            """Add render-movie parameter controls to a form."""
            self._render_movie_keyframe_manifest_input = QLineEdit()
            self._render_movie_keyframe_manifest_input.setPlaceholderText(
                "Latest visualization keyframes"
            )
            form.addRow(
                "Keyframe manifest",
                self._render_movie_keyframe_manifest_input,
            )
            self._register_parameter_hint(
                self._render_movie_keyframe_manifest_input,
                self._PARAMETER_HINTS["render_movie_keyframe_manifest_path"],
            )

            self._render_movie_resolution_levels_input = QLineEdit()
            self._render_movie_resolution_levels_input.setPlaceholderText("0")
            form.addRow(
                "Resolution levels",
                self._render_movie_resolution_levels_input,
            )
            self._register_parameter_hint(
                self._render_movie_resolution_levels_input,
                self._PARAMETER_HINTS["render_movie_resolution_levels"],
            )

            render_size_widget = QWidget()
            render_size_layout = QHBoxLayout(render_size_widget)
            render_size_layout.setContentsMargins(0, 0, 0, 0)
            render_size_layout.setSpacing(8)
            self._render_movie_width_spin = QSpinBox()
            self._render_movie_width_spin.setRange(64, 16384)
            self._render_movie_height_spin = QSpinBox()
            self._render_movie_height_spin.setRange(64, 16384)
            render_size_layout.addWidget(self._render_movie_width_spin)
            render_size_layout.addWidget(QLabel("x"))
            render_size_layout.addWidget(self._render_movie_height_spin)
            form.addRow("Render size", render_size_widget)
            self._register_parameter_hint(
                self._render_movie_width_spin,
                self._PARAMETER_HINTS["render_movie_render_size_xy"],
            )
            self._register_parameter_hint(
                self._render_movie_height_spin,
                self._PARAMETER_HINTS["render_movie_render_size_xy"],
            )

            self._render_movie_fps_spin = QSpinBox()
            self._render_movie_fps_spin.setRange(1, 240)
            form.addRow("FPS", self._render_movie_fps_spin)
            self._register_parameter_hint(
                self._render_movie_fps_spin,
                self._PARAMETER_HINTS["render_movie_default_transition_frames"],
            )

            self._render_movie_default_transition_frames_spin = QSpinBox()
            self._render_movie_default_transition_frames_spin.setRange(0, 10000)
            form.addRow(
                "Default gap frames",
                self._render_movie_default_transition_frames_spin,
            )
            self._register_parameter_hint(
                self._render_movie_default_transition_frames_spin,
                self._PARAMETER_HINTS["render_movie_default_transition_frames"],
            )

            self._render_movie_transition_frames_input = QLineEdit()
            self._render_movie_transition_frames_input.setPlaceholderText(
                "Per-gap overrides, for example 48,72,36"
            )
            form.addRow(
                "Per-gap frames",
                self._render_movie_transition_frames_input,
            )
            self._register_parameter_hint(
                self._render_movie_transition_frames_input,
                self._PARAMETER_HINTS["render_movie_transition_frames_by_gap"],
            )

            self._render_movie_hold_frames_spin = QSpinBox()
            self._render_movie_hold_frames_spin.setRange(0, 10000)
            form.addRow("Hold frames", self._render_movie_hold_frames_spin)
            self._register_parameter_hint(
                self._render_movie_hold_frames_spin,
                self._PARAMETER_HINTS["render_movie_hold_frames"],
            )

            self._render_movie_interpolation_combo = QComboBox()
            self._render_movie_interpolation_combo.addItem("Ease in/out", "ease_in_out")
            self._render_movie_interpolation_combo.addItem("Linear", "linear")
            form.addRow("Interpolation", self._render_movie_interpolation_combo)
            self._register_parameter_hint(
                self._render_movie_interpolation_combo,
                self._PARAMETER_HINTS["render_movie_interpolation_mode"],
            )

            self._render_movie_camera_effect_combo = QComboBox()
            self._render_movie_camera_effect_combo.addItem("None", "none")
            self._render_movie_camera_effect_combo.addItem("Orbit", "orbit")
            self._render_movie_camera_effect_combo.addItem("Flythrough", "flythrough")
            self._render_movie_camera_effect_combo.addItem("Zoom FX", "zoom_fx")
            self._render_movie_camera_effect_combo.currentIndexChanged.connect(
                self._set_render_movie_parameter_enabled_state
            )
            form.addRow("Camera effect", self._render_movie_camera_effect_combo)
            self._register_parameter_hint(
                self._render_movie_camera_effect_combo,
                self._PARAMETER_HINTS["render_movie_camera_effect"],
            )

            self._render_movie_orbit_degrees_spin = QDoubleSpinBox()
            self._render_movie_orbit_degrees_spin.setRange(0.0, 1080.0)
            self._render_movie_orbit_degrees_spin.setDecimals(2)
            form.addRow("Orbit degrees", self._render_movie_orbit_degrees_spin)
            self._register_parameter_hint(
                self._render_movie_orbit_degrees_spin,
                self._PARAMETER_HINTS["render_movie_camera_effect"],
            )

            self._render_movie_flythrough_distance_spin = QDoubleSpinBox()
            self._render_movie_flythrough_distance_spin.setRange(0.0, 10.0)
            self._render_movie_flythrough_distance_spin.setDecimals(3)
            self._render_movie_flythrough_distance_spin.setSingleStep(0.01)
            form.addRow(
                "Flythrough factor",
                self._render_movie_flythrough_distance_spin,
            )
            self._register_parameter_hint(
                self._render_movie_flythrough_distance_spin,
                self._PARAMETER_HINTS["render_movie_camera_effect"],
            )

            self._render_movie_zoom_effect_spin = QDoubleSpinBox()
            self._render_movie_zoom_effect_spin.setRange(0.0, 10.0)
            self._render_movie_zoom_effect_spin.setDecimals(3)
            self._render_movie_zoom_effect_spin.setSingleStep(0.01)
            form.addRow("Zoom factor", self._render_movie_zoom_effect_spin)
            self._register_parameter_hint(
                self._render_movie_zoom_effect_spin,
                self._PARAMETER_HINTS["render_movie_camera_effect"],
            )

            self._render_movie_overlay_title_input = QLineEdit()
            form.addRow("Title", self._render_movie_overlay_title_input)
            self._register_parameter_hint(
                self._render_movie_overlay_title_input,
                self._PARAMETER_HINTS["render_movie_overlay_text"],
            )

            self._render_movie_overlay_subtitle_input = QLineEdit()
            form.addRow("Subtitle", self._render_movie_overlay_subtitle_input)
            self._register_parameter_hint(
                self._render_movie_overlay_subtitle_input,
                self._PARAMETER_HINTS["render_movie_overlay_text"],
            )

            self._render_movie_overlay_frame_text_combo = QComboBox()
            self._render_movie_overlay_frame_text_combo.addItem("None", "none")
            self._render_movie_overlay_frame_text_combo.addItem(
                "Frame number",
                "frame_number",
            )
            self._render_movie_overlay_frame_text_combo.addItem(
                "Keyframe annotations",
                "keyframe_annotations",
            )
            form.addRow(
                "Frame text",
                self._render_movie_overlay_frame_text_combo,
            )
            self._register_parameter_hint(
                self._render_movie_overlay_frame_text_combo,
                self._PARAMETER_HINTS["render_movie_overlay_text"],
            )

            self._render_movie_overlay_scalebar_checkbox = QCheckBox("Draw scalebar")
            self._render_movie_overlay_scalebar_checkbox.stateChanged.connect(
                self._set_render_movie_parameter_enabled_state
            )
            form.addRow("Scalebar", self._render_movie_overlay_scalebar_checkbox)
            self._register_parameter_hint(
                self._render_movie_overlay_scalebar_checkbox,
                self._PARAMETER_HINTS["render_movie_overlay_scalebar"],
            )

            self._render_movie_overlay_scalebar_length_spin = QDoubleSpinBox()
            self._render_movie_overlay_scalebar_length_spin.setRange(0.001, 100000.0)
            self._render_movie_overlay_scalebar_length_spin.setDecimals(3)
            form.addRow(
                "Scalebar length (um)",
                self._render_movie_overlay_scalebar_length_spin,
            )
            self._register_parameter_hint(
                self._render_movie_overlay_scalebar_length_spin,
                self._PARAMETER_HINTS["render_movie_overlay_scalebar"],
            )

            self._render_movie_overlay_scalebar_position_combo = QComboBox()
            self._render_movie_overlay_scalebar_position_combo.addItem(
                "Bottom left",
                "bottom_left",
            )
            self._render_movie_overlay_scalebar_position_combo.addItem(
                "Bottom right",
                "bottom_right",
            )
            self._render_movie_overlay_scalebar_position_combo.addItem(
                "Top left",
                "top_left",
            )
            self._render_movie_overlay_scalebar_position_combo.addItem(
                "Top right",
                "top_right",
            )
            form.addRow(
                "Scalebar position",
                self._render_movie_overlay_scalebar_position_combo,
            )
            self._register_parameter_hint(
                self._render_movie_overlay_scalebar_position_combo,
                self._PARAMETER_HINTS["render_movie_overlay_scalebar"],
            )

            self._render_movie_output_directory_input = QLineEdit()
            self._render_movie_output_directory_input.setPlaceholderText("Auto")
            form.addRow("Output directory", self._render_movie_output_directory_input)
            self._register_parameter_hint(
                self._render_movie_output_directory_input,
                self._PARAMETER_HINTS["render_movie_output_directory"],
            )

        def _build_compile_movie_parameter_rows(self, form: QFormLayout) -> None:
            """Add compile-movie parameter controls to a form."""
            self._compile_movie_render_manifest_input = QLineEdit()
            self._compile_movie_render_manifest_input.setPlaceholderText(
                "Latest render manifest"
            )
            form.addRow(
                "Render manifest",
                self._compile_movie_render_manifest_input,
            )
            self._register_parameter_hint(
                self._compile_movie_render_manifest_input,
                self._PARAMETER_HINTS["compile_movie_render_manifest_path"],
            )

            self._compile_movie_rendered_level_spin = QSpinBox()
            self._compile_movie_rendered_level_spin.setRange(0, 64)
            form.addRow("Rendered level", self._compile_movie_rendered_level_spin)
            self._register_parameter_hint(
                self._compile_movie_rendered_level_spin,
                self._PARAMETER_HINTS["compile_movie_rendered_level"],
            )

            self._compile_movie_output_format_combo = QComboBox()
            self._compile_movie_output_format_combo.addItem("MP4", "mp4")
            self._compile_movie_output_format_combo.addItem("ProRes MOV", "prores")
            self._compile_movie_output_format_combo.addItem("Both", "both")
            self._compile_movie_output_format_combo.currentIndexChanged.connect(
                self._set_compile_movie_parameter_enabled_state
            )
            form.addRow("Output format", self._compile_movie_output_format_combo)
            self._register_parameter_hint(
                self._compile_movie_output_format_combo,
                self._PARAMETER_HINTS["compile_movie_output_format"],
            )

            self._compile_movie_fps_spin = QSpinBox()
            self._compile_movie_fps_spin.setRange(1, 240)
            form.addRow("FPS", self._compile_movie_fps_spin)
            self._register_parameter_hint(
                self._compile_movie_fps_spin,
                self._PARAMETER_HINTS["compile_movie_output_format"],
            )

            self._compile_movie_mp4_crf_spin = QSpinBox()
            self._compile_movie_mp4_crf_spin.setRange(0, 51)
            form.addRow("MP4 CRF", self._compile_movie_mp4_crf_spin)
            self._register_parameter_hint(
                self._compile_movie_mp4_crf_spin,
                self._PARAMETER_HINTS["compile_movie_output_format"],
            )

            self._compile_movie_mp4_preset_combo = QComboBox()
            for preset in (
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
            ):
                self._compile_movie_mp4_preset_combo.addItem(preset, preset)
            form.addRow("MP4 preset", self._compile_movie_mp4_preset_combo)
            self._register_parameter_hint(
                self._compile_movie_mp4_preset_combo,
                self._PARAMETER_HINTS["compile_movie_output_format"],
            )

            self._compile_movie_prores_profile_spin = QSpinBox()
            self._compile_movie_prores_profile_spin.setRange(0, 5)
            form.addRow(
                "ProRes profile",
                self._compile_movie_prores_profile_spin,
            )
            self._register_parameter_hint(
                self._compile_movie_prores_profile_spin,
                self._PARAMETER_HINTS["compile_movie_output_format"],
            )

            self._compile_movie_pixel_format_input = QLineEdit()
            self._compile_movie_pixel_format_input.setPlaceholderText("Auto")
            form.addRow("Pixel format", self._compile_movie_pixel_format_input)
            self._register_parameter_hint(
                self._compile_movie_pixel_format_input,
                self._PARAMETER_HINTS["compile_movie_output_format"],
            )

            resize_widget = QWidget()
            resize_layout = QHBoxLayout(resize_widget)
            resize_layout.setContentsMargins(0, 0, 0, 0)
            resize_layout.setSpacing(8)
            self._compile_movie_resize_width_spin = QSpinBox()
            self._compile_movie_resize_width_spin.setRange(0, 16384)
            self._compile_movie_resize_height_spin = QSpinBox()
            self._compile_movie_resize_height_spin.setRange(0, 16384)
            resize_layout.addWidget(self._compile_movie_resize_width_spin)
            resize_layout.addWidget(QLabel("x"))
            resize_layout.addWidget(self._compile_movie_resize_height_spin)
            form.addRow("Resize", resize_widget)
            self._register_parameter_hint(
                self._compile_movie_resize_width_spin,
                self._PARAMETER_HINTS["compile_movie_resize_xy"],
            )
            self._register_parameter_hint(
                self._compile_movie_resize_height_spin,
                self._PARAMETER_HINTS["compile_movie_resize_xy"],
            )

            self._compile_movie_output_directory_input = QLineEdit()
            self._compile_movie_output_directory_input.setPlaceholderText("Auto")
            form.addRow(
                "Output directory",
                self._compile_movie_output_directory_input,
            )
            self._register_parameter_hint(
                self._compile_movie_output_directory_input,
                self._PARAMETER_HINTS["compile_movie_output_directory"],
            )

            self._compile_movie_output_stem_input = QLineEdit()
            self._compile_movie_output_stem_input.setPlaceholderText("Auto")
            form.addRow("Output stem", self._compile_movie_output_stem_input)
            self._register_parameter_hint(
                self._compile_movie_output_stem_input,
                self._PARAMETER_HINTS["compile_movie_output_stem"],
            )

        def _build_volume_export_parameter_rows(self, form: QFormLayout) -> None:
            """Add volume-export parameter controls to a form."""
            self._volume_export_scope_combo = QComboBox()
            self._volume_export_scope_combo.addItem(
                "Current selection", "current_selection"
            )
            self._volume_export_scope_combo.addItem("All indices", "all_indices")
            self._volume_export_scope_combo.currentIndexChanged.connect(
                self._set_volume_export_parameter_enabled_state
            )
            form.addRow("Export scope", self._volume_export_scope_combo)
            self._register_parameter_hint(
                self._volume_export_scope_combo,
                self._PARAMETER_HINTS["volume_export_scope"],
            )

            volume_export_index_widget = QWidget()
            volume_export_index_layout = QHBoxLayout(volume_export_index_widget)
            volume_export_index_layout.setContentsMargins(0, 0, 0, 0)
            volume_export_index_layout.setSpacing(8)
            self._volume_export_t_spin = QSpinBox()
            self._volume_export_t_spin.setRange(0, 1_000_000)
            self._volume_export_p_spin = QSpinBox()
            self._volume_export_p_spin.setRange(0, 1_000_000)
            self._volume_export_c_spin = QSpinBox()
            self._volume_export_c_spin.setRange(0, 1_000_000)
            volume_export_index_layout.addWidget(QLabel("T"))
            volume_export_index_layout.addWidget(self._volume_export_t_spin)
            volume_export_index_layout.addWidget(QLabel("P"))
            volume_export_index_layout.addWidget(self._volume_export_p_spin)
            volume_export_index_layout.addWidget(QLabel("C"))
            volume_export_index_layout.addWidget(self._volume_export_c_spin)
            form.addRow("T/P/C index", volume_export_index_widget)
            for widget in (
                self._volume_export_t_spin,
                self._volume_export_p_spin,
                self._volume_export_c_spin,
            ):
                hint_key = {
                    self._volume_export_t_spin: "volume_export_t_index",
                    self._volume_export_p_spin: "volume_export_p_index",
                    self._volume_export_c_spin: "volume_export_c_index",
                }[widget]
                self._register_parameter_hint(
                    widget,
                    self._PARAMETER_HINTS[hint_key],
                )

            self._volume_export_resolution_level_spin = QSpinBox()
            self._volume_export_resolution_level_spin.setRange(0, 64)
            form.addRow(
                "Resolution level",
                self._volume_export_resolution_level_spin,
            )
            self._register_parameter_hint(
                self._volume_export_resolution_level_spin,
                self._PARAMETER_HINTS["volume_export_resolution_level"],
            )

            self._volume_export_format_combo = QComboBox()
            self._volume_export_format_combo.addItem("OME-Zarr", "ome-zarr")
            self._volume_export_format_combo.addItem("OME-TIFF", "ome-tiff")
            self._volume_export_format_combo.currentIndexChanged.connect(
                self._set_volume_export_parameter_enabled_state
            )
            form.addRow("Export format", self._volume_export_format_combo)
            self._register_parameter_hint(
                self._volume_export_format_combo,
                self._PARAMETER_HINTS["volume_export_export_format"],
            )

            self._volume_export_tiff_layout_combo = QComboBox()
            self._volume_export_tiff_layout_combo.addItem("Single file", "single_file")
            self._volume_export_tiff_layout_combo.addItem(
                "Per-volume files",
                "per_volume_files",
            )
            form.addRow("TIFF file layout", self._volume_export_tiff_layout_combo)
            self._register_parameter_hint(
                self._volume_export_tiff_layout_combo,
                self._PARAMETER_HINTS["volume_export_tiff_file_layout"],
            )

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
                component = resolve_analysis_input_component(
                    str(data).strip() if data is not None else "data"
                )
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
                if multiscale_policy == "auto_build":
                    multiscale_policy = "inherit"
                if multiscale_policy not in {"inherit", "require", "off"}:
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
                        "channels": self._normalize_visualization_channels(
                            raw_row.get("channels")
                        ),
                        "visible": self._coerce_optional_bool(raw_row.get("visible")),
                        "opacity": self._coerce_optional_unit_interval_float(
                            raw_row.get("opacity")
                        ),
                        "blending": blending,
                        "colormap": colormap,
                        "rendering": rendering,
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
            selected_value = (
                str(
                    analysis_operation_for_output_component(component) or component
                ).strip()
                or "data"
            )
            combo.blockSignals(True)
            combo_index = combo.findData(selected_value)
            if combo_index < 0:
                combo.addItem(f"Custom component ({component})", component)
                combo_index = combo.count() - 1
            combo.setCurrentIndex(combo_index)
            combo.blockSignals(False)

        def _sync_visualization_volume_layers_from_input_source(
            self,
            *,
            refresh_summary: bool = True,
        ) -> None:
            """Align primary visualization volume layer with input-source combo.

            Parameters
            ----------
            refresh_summary : bool, default=True
                Whether to refresh the layer summary label after synchronization.

            Returns
            -------
            None
                Visualization volume-layer rows are updated in-place.
            """
            combo = self._operation_input_combos.get("visualization")
            if combo is None:
                return
            combo_value = combo.currentData()
            selected_component = resolve_analysis_input_component(
                str(combo_value).strip() if combo_value is not None else "data"
            )

            rows = self._normalize_visualization_volume_layers(
                self._visualization_volume_layers
            )
            if not rows:
                rows = self._default_visualization_volume_layers()
            if not rows:
                return

            current_component = str(rows[0].get("component", "data")).strip() or "data"
            if current_component == selected_component:
                self._visualization_volume_layers = list(rows)
                return

            updated_first_row = dict(rows[0])
            updated_first_row["component"] = selected_component
            rows[0] = updated_first_row
            self._visualization_volume_layers = list(rows)

            if refresh_summary:
                self._refresh_visualization_volume_layers_summary()

        def _on_visualization_input_source_changed(self, _index: int) -> None:
            """Handle visualization input-source combo selection changes.

            Parameters
            ----------
            _index : int
                Selected combo index (unused).

            Returns
            -------
            None
                Primary visualization layer component is synchronized to match the
                selected input source.
            """
            self._sync_visualization_volume_layers_from_input_source(
                refresh_summary=True
            )

        def _registration_selected_input_source(self) -> str:
            """Return the registration input-source selector value.

            Parameters
            ----------
            None

            Returns
            -------
            str
                Selected registration input-source alias or component path.
            """
            combo = self._operation_input_combos.get("registration")
            if combo is None:
                return "data"
            selected = combo.currentData()
            return (str(selected).strip() if selected is not None else "data") or "data"

        def _registration_available_resolution_levels(self) -> tuple[int, ...]:
            """Return available registration resolution levels for current source.

            Parameters
            ----------
            None

            Returns
            -------
            tuple[int, ...]
                Sorted level values available for the selected registration input.
            """
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return (0,)

            requested_source = self._registration_selected_input_source()
            resolved_source = resolve_analysis_input_component(requested_source)
            try:
                root = zarr.open_group(store_path, mode="r")
            except Exception:
                return (0,)
            return _discover_component_resolution_levels(
                root=root,
                source_component=resolved_source,
            )

        def _registration_available_channels(self) -> tuple[int, ...]:
            """Return available registration channel indices for current source.

            Parameters
            ----------
            None

            Returns
            -------
            tuple[int, ...]
                Sorted channel values available for the selected registration input.
            """
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return (0,)

            requested_source = self._registration_selected_input_source()
            resolved_source = resolve_analysis_input_component(requested_source)
            try:
                root = zarr.open_group(store_path, mode="r")
            except Exception:
                return (0,)
            return _discover_component_channels(
                root=root,
                source_component=resolved_source,
            )

        def _volume_export_selected_input_source(self) -> str:
            """Return the selected volume-export input-source value."""
            combo = self._operation_input_combos.get("volume_export")
            if combo is None:
                return "data"
            selected = combo.currentData()
            return (str(selected).strip() if selected is not None else "data") or "data"

        def _volume_export_available_bounds(
            self,
        ) -> tuple[int, int, int, tuple[int, ...]]:
            """Return bounds for the selected volume-export input source."""
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return (1, 1, 1, (0,))

            requested_source = self._volume_export_selected_input_source()
            resolved_source = resolve_analysis_input_component(requested_source)
            try:
                root = zarr.open_group(store_path, mode="r")
            except Exception:
                return (1, 1, 1, (0,))

            time_count = 1
            position_count = 1
            channel_count = 1
            try:
                source = root[resolved_source]
                shape = tuple(getattr(source, "shape", ()))
            except Exception:
                shape = ()
            if len(shape) >= 3:
                try:
                    time_count = max(1, int(shape[0]))
                    position_count = max(1, int(shape[1]))
                    channel_count = max(1, int(shape[2]))
                except Exception:
                    time_count = 1
                    position_count = 1
                    channel_count = 1
            available_levels = _discover_component_resolution_levels(
                root=root,
                source_component=resolved_source,
            )
            return (time_count, position_count, channel_count, available_levels)

        def _refresh_volume_export_parameter_bounds(self) -> None:
            """Refresh source-aware volume-export bounds and clamp selection."""
            (
                time_count,
                position_count,
                channel_count,
                available_levels,
            ) = self._volume_export_available_bounds()
            discovered_max_level = max(
                0,
                max(available_levels) if available_levels else 0,
            )
            max_level = max(
                int(self._volume_export_resolution_level_spin.maximum()),
                discovered_max_level,
            )
            self._volume_export_t_spin.setMaximum(time_count - 1)
            self._volume_export_p_spin.setMaximum(position_count - 1)
            self._volume_export_c_spin.setMaximum(channel_count - 1)
            self._volume_export_resolution_level_spin.setMaximum(max_level)
            self._volume_export_t_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_t_spin.maximum()),
                        int(self._volume_export_t_spin.value()),
                    ),
                )
            )
            self._volume_export_p_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_p_spin.maximum()),
                        int(self._volume_export_p_spin.value()),
                    ),
                )
            )
            self._volume_export_c_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_c_spin.maximum()),
                        int(self._volume_export_c_spin.value()),
                    ),
                )
            )
            self._volume_export_resolution_level_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_resolution_level_spin.maximum()),
                        int(self._volume_export_resolution_level_spin.value()),
                    ),
                )
            )
            self._set_volume_export_parameter_enabled_state()

        def _refresh_registration_channel_options(
            self,
            *,
            preferred_channel: Optional[int] = None,
        ) -> None:
            """Refresh registration channel combo options.

            Parameters
            ----------
            preferred_channel : int, optional
                Preferred selected channel after options are rebuilt.

            Returns
            -------
            None
                Combo items and selected index are updated in-place.
            """
            combo = getattr(self, "_registration_channel_combo", None)
            if combo is None:
                return

            current_data = combo.currentData()
            try:
                current_channel = max(0, int(current_data))
            except (TypeError, ValueError):
                current_channel = 0

            if preferred_channel is None:
                target_channel = current_channel
            else:
                target_channel = max(0, int(preferred_channel))

            available_channels = self._registration_available_channels()
            selected_channel = int(available_channels[0])
            for channel in available_channels:
                if int(channel) <= target_channel:
                    selected_channel = int(channel)

            combo.blockSignals(True)
            combo.clear()
            for channel in available_channels:
                channel_value = int(channel)
                combo.addItem(f"Channel {channel_value}", channel_value)
            selected_index = combo.findData(int(selected_channel))
            if selected_index < 0:
                selected_index = 0
            combo.setCurrentIndex(selected_index)
            combo.blockSignals(False)

        def _refresh_registration_resolution_level_options(
            self,
            *,
            preferred_level: Optional[int] = None,
        ) -> None:
            """Refresh registration resolution-level combo options.

            Parameters
            ----------
            preferred_level : int, optional
                Preferred selected level after options are rebuilt.

            Returns
            -------
            None
                Combo items and selected index are updated in-place.
            """
            combo = getattr(self, "_registration_resolution_level_combo", None)
            if combo is None:
                return

            current_data = combo.currentData()
            try:
                current_level = max(
                    0,
                    int(current_data),
                )
            except (TypeError, ValueError):
                current_level = 0

            if preferred_level is None:
                target_level = current_level
            else:
                target_level = max(0, int(preferred_level))

            available_levels = self._registration_available_resolution_levels()
            selected_level = int(available_levels[0])
            for level in available_levels:
                if int(level) <= target_level:
                    selected_level = int(level)

            combo.blockSignals(True)
            combo.clear()
            for level in available_levels:
                level_value = int(level)
                combo.addItem(f"Level {level_value}", level_value)
            selected_index = combo.findData(int(selected_level))
            if selected_index < 0:
                selected_index = 0
            combo.setCurrentIndex(selected_index)
            combo.blockSignals(False)

        def _on_registration_input_source_changed(self, _index: int) -> None:
            """Refresh registration source-dependent choices after input changes.

            Parameters
            ----------
            _index : int
                Selected combo index (unused).

            Returns
            -------
            None
                Registration channel and resolution options are rebuilt in-place.
            """
            self._refresh_registration_channel_options()
            self._refresh_registration_channel_options()
            self._refresh_registration_resolution_level_options()

        def _on_volume_export_input_source_changed(self, _index: int) -> None:
            """Refresh volume-export bounds after input-source changes."""
            self._refresh_volume_export_parameter_bounds()

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
            _apply_initial_dialog_geometry(
                dialog,
                minimum_size=_VOLUME_LAYERS_DIALOG_MINIMUM_SIZE,
                preferred_size=_VOLUME_LAYERS_DIALOG_PREFERRED_SIZE,
            )
            dialog.setModal(True)
            dialog.setStyleSheet(_popup_dialog_stylesheet())

            root = QVBoxLayout(dialog)
            apply_popup_root_spacing(root)

            helper = QLabel(
                "Configure image/labels overlays for napari. "
                "Use the Channels selector to choose CH indices (leave empty for all)."
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
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(8, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(9, QHeaderView.ResizeMode.Interactive)
            table.setColumnWidth(0, 140)
            table.setColumnWidth(1, 280)
            table.setColumnWidth(2, 120)
            table.setColumnWidth(3, 130)
            table.setColumnWidth(5, 110)
            table.setColumnWidth(6, 150)
            table.setColumnWidth(7, 140)
            table.setColumnWidth(8, 150)
            table.setColumnWidth(9, 140)
            vertical_header = table.verticalHeader()
            vertical_header.setVisible(False)
            vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            row_height = max(40, int(table.fontMetrics().height()) + 18)
            vertical_header.setDefaultSectionSize(row_height)
            root.addWidget(table, stretch=1)

            component_options = self._visualization_volume_layer_component_options()
            blending_options = _available_visualization_blending_options()
            colormap_options = _available_visualization_colormap_options()
            rendering_options = _available_visualization_rendering_options()
            combo_height = max(30, row_height - 8)
            store_path = str(self._base_config.file or "").strip()
            store_root: Optional[Any] = None
            if store_path and is_zarr_store_path(store_path):
                try:
                    store_root = zarr.open_group(store_path, mode="r")
                except Exception:
                    store_root = None

            def _configure_combo_size(combo: QComboBox) -> QComboBox:
                combo.setMinimumHeight(combo_height)
                combo.setMaximumHeight(combo_height)
                combo.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                return cast(
                    QComboBox,
                    _install_table_row_selection_widget_hook(table, combo),
                )

            def _configure_line_edit_size(
                line_edit: QLineEdit,
                *,
                placeholder_text: str = "",
            ) -> QLineEdit:
                line_edit.setMinimumHeight(combo_height)
                line_edit.setMaximumHeight(combo_height)
                line_edit.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                if placeholder_text:
                    line_edit.setPlaceholderText(str(placeholder_text))
                return cast(
                    QLineEdit,
                    _install_table_row_selection_widget_hook(table, line_edit),
                )

            def _make_choice_combo(
                *,
                current_value: str,
                option_values: Sequence[str],
                auto_label: str = "Auto",
            ) -> QComboBox:
                combo = _configure_combo_size(QComboBox(table))
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
                combo.addItem(str(auto_label), "")
                for option in option_values:
                    text = str(option).strip()
                    if not text:
                        continue
                    combo.addItem(text, text)
                value = str(current_value).strip()
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
                elif value:
                    combo.addItem(value, value)
                    combo.setCurrentIndex(combo.count() - 1)
                else:
                    combo.setCurrentIndex(0)
                line_edit = combo.lineEdit()
                if line_edit is not None:
                    line_edit.setCursorPosition(0)
                return combo

            def _read_choice_combo_value(combo: QComboBox) -> str:
                text = str(combo.currentText()).strip()
                current_index = int(combo.currentIndex())
                if current_index >= 0:
                    selected_label = str(combo.itemText(current_index)).strip()
                    if text == selected_label:
                        selected_data = combo.itemData(current_index)
                        if selected_data is None:
                            return ""
                        return str(selected_data).strip()
                return text

            def _make_text_input(
                value: str,
                *,
                placeholder_text: str = "",
            ) -> QLineEdit:
                line_edit = _configure_line_edit_size(
                    QLineEdit(table),
                    placeholder_text=placeholder_text,
                )
                line_edit.setText(str(value))
                return line_edit

            def _make_component_combo(current_value: str) -> QComboBox:
                combo = _configure_combo_size(QComboBox(table))
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
                for component, label in component_options:
                    combo.addItem(label, component)
                value = str(current_value).strip()
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
                elif value:
                    combo.addItem(f"Custom component ({value})", value)
                    combo.setCurrentIndex(combo.count() - 1)
                else:
                    combo.setCurrentIndex(0)
                line_edit = combo.lineEdit()
                if line_edit is not None:
                    line_edit.setCursorPosition(0)
                return combo

            def _read_component_combo_value(combo: QComboBox) -> str:
                text = str(combo.currentText()).strip()
                current_index = int(combo.currentIndex())
                if current_index >= 0:
                    selected_label = str(combo.itemText(current_index)).strip()
                    if text == selected_label:
                        selected_data = combo.itemData(current_index)
                        selected_component = (
                            str(selected_data).strip()
                            if selected_data is not None
                            else ""
                        )
                        if selected_component:
                            return selected_component
                if text:
                    return text
                selected_data = combo.currentData()
                return str(selected_data).strip() if selected_data is not None else ""

            def _available_channels_for_component(
                component_value: str,
            ) -> tuple[int, ...]:
                component = str(component_value).strip() or "data"
                resolved_component = resolve_analysis_input_component(component)
                if store_root is None:
                    return (0,)
                try:
                    return _discover_component_channels(
                        root=store_root,
                        source_component=resolved_component,
                    )
                except Exception:
                    return (0,)

            def _set_channel_selector_state(
                button: QPushButton,
                *,
                selected_channels: Sequence[int],
                available_channels: Sequence[int],
            ) -> None:
                normalized_available = sorted(
                    {int(value) for value in available_channels if int(value) >= 0}
                )
                available_set = set(normalized_available)
                normalized_selected = sorted(
                    set(
                        int(value)
                        for value in self._normalize_visualization_channels(
                            selected_channels
                        )
                    )
                )
                for value in normalized_selected:
                    if value in available_set:
                        continue
                    normalized_available.append(int(value))
                    available_set.add(int(value))
                normalized_available = sorted(set(normalized_available))
                normalized_selected = [
                    int(value)
                    for value in normalized_selected
                    if int(value) in available_set
                ]
                button.setProperty(
                    "_clearexSelectedChannels",
                    [int(value) for value in normalized_selected],
                )
                button.setProperty(
                    "_clearexAvailableChannels",
                    [int(value) for value in normalized_available],
                )
                if not normalized_selected:
                    button.setText("All channels")
                else:
                    button.setText(
                        ", ".join(f"CH{int(value)}" for value in normalized_selected)
                    )
                button.setToolTip(
                    "Selected channels for this layer. "
                    "Leave empty to render all available channels."
                )

            def _open_channel_selector_dialog(
                *,
                anchor: QPushButton,
                component_combo: QComboBox,
            ) -> None:
                component_value = _read_component_combo_value(component_combo)
                available_channels = _available_channels_for_component(component_value)
                selected_channels = self._normalize_visualization_channels(
                    anchor.property("_clearexSelectedChannels")
                )
                selected_set = set(selected_channels)
                picker_channels = sorted(
                    set(int(value) for value in available_channels if int(value) >= 0)
                    | set(int(value) for value in selected_set if int(value) >= 0)
                )

                picker = QDialog(dialog)
                picker.setWindowTitle("Select Channels")
                picker.setModal(True)
                picker.setStyleSheet(_popup_dialog_stylesheet())
                _apply_initial_dialog_geometry(
                    picker,
                    minimum_size=(320, 220),
                    preferred_size=(420, 360),
                )
                picker_root = QVBoxLayout(picker)
                apply_popup_root_spacing(picker_root)

                picker_hint = QLabel(
                    "Choose channels for this layer. "
                    "Leave all unchecked to render all channels."
                )
                picker_hint.setWordWrap(True)
                picker_root.addWidget(picker_hint)

                channel_boxes: list[tuple[int, QCheckBox]] = []
                for channel in picker_channels:
                    value = int(channel)
                    checkbox = QCheckBox(f"CH{value}", picker)
                    checkbox.setChecked(value in selected_set)
                    picker_root.addWidget(checkbox)
                    channel_boxes.append((value, checkbox))

                if not channel_boxes:
                    picker_root.addWidget(QLabel("No channels available."))

                quick_controls = QHBoxLayout()
                select_all_button = QPushButton("All", picker)
                clear_button = QPushButton("None", picker)
                quick_controls.addWidget(select_all_button)
                quick_controls.addWidget(clear_button)
                quick_controls.addStretch(1)
                picker_root.addLayout(quick_controls)

                def _set_all_channels_checked(checked: bool) -> None:
                    for _, checkbox in channel_boxes:
                        checkbox.setChecked(bool(checked))

                select_all_button.clicked.connect(
                    lambda: _set_all_channels_checked(True)
                )
                clear_button.clicked.connect(lambda: _set_all_channels_checked(False))

                button_box = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok
                    | QDialogButtonBox.StandardButton.Cancel,
                    Qt.Orientation.Horizontal,
                    picker,
                )
                picker_root.addWidget(button_box)
                button_box.accepted.connect(picker.accept)
                button_box.rejected.connect(picker.reject)

                if picker.exec() != QDialog.DialogCode.Accepted:
                    return

                picked_channels = [
                    int(channel)
                    for channel, checkbox in channel_boxes
                    if checkbox.isChecked()
                ]
                _set_channel_selector_state(
                    anchor,
                    selected_channels=picked_channels,
                    available_channels=picker_channels,
                )

            def _make_channel_selector(
                *,
                current_channels: Sequence[int],
                component_combo: QComboBox,
            ) -> QPushButton:
                button = QPushButton(table)
                button.setMinimumHeight(combo_height)
                button.setMaximumHeight(combo_height)
                button.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                button = cast(
                    QPushButton,
                    _install_table_row_selection_widget_hook(table, button),
                )
                _set_channel_selector_state(
                    button,
                    selected_channels=current_channels,
                    available_channels=_available_channels_for_component(
                        _read_component_combo_value(component_combo)
                    ),
                )
                button.clicked.connect(
                    lambda _checked=False, anchor=button, combo=component_combo: _open_channel_selector_dialog(
                        anchor=anchor,
                        component_combo=combo,
                    )
                )
                return button

            def _sync_channel_selector_for_component(
                *,
                component_combo: QComboBox,
                channel_selector: QPushButton,
            ) -> None:
                available_channels = _available_channels_for_component(
                    _read_component_combo_value(component_combo)
                )
                _set_channel_selector_state(
                    channel_selector,
                    selected_channels=self._normalize_visualization_channels(
                        channel_selector.property("_clearexSelectedChannels")
                    ),
                    available_channels=available_channels,
                )

            def _make_type_combo(current_value: str) -> QComboBox:
                combo = _configure_combo_size(QComboBox(table))
                combo.addItem("Image", "image")
                combo.addItem("Labels", "labels")
                index = combo.findData(str(current_value).strip().lower() or "image")
                combo.setCurrentIndex(index if index >= 0 else 0)
                return combo

            def _make_visible_combo(current_value: Optional[bool]) -> QComboBox:
                combo = _configure_combo_size(QComboBox(table))
                combo.addItem("Auto", None)
                combo.addItem("True", True)
                combo.addItem("False", False)
                index = combo.findData(current_value)
                combo.setCurrentIndex(index if index >= 0 else 0)
                return combo

            def _make_multiscale_combo(current_value: str) -> QComboBox:
                combo = _configure_combo_size(QComboBox(table))
                combo.addItem("Inherit", "inherit")
                combo.addItem("Require", "require")
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
                component_combo = _make_component_combo(
                    str(
                        values.get("component", values.get("source_component", ""))
                    ).strip()
                )
                channels_selector = _make_channel_selector(
                    current_channels=channels,
                    component_combo=component_combo,
                )
                component_combo.currentIndexChanged.connect(
                    lambda _index, combo=component_combo, selector=channels_selector: _sync_channel_selector_for_component(
                        component_combo=combo,
                        channel_selector=selector,
                    )
                )
                component_combo.editTextChanged.connect(
                    lambda _text, combo=component_combo, selector=channels_selector: _sync_channel_selector_for_component(
                        component_combo=combo,
                        channel_selector=selector,
                    )
                )
                table.setCellWidget(
                    row_index,
                    0,
                    _make_text_input(
                        str(values.get("name", "")).strip(),
                        placeholder_text="Optional name",
                    ),
                )
                table.setCellWidget(
                    row_index,
                    1,
                    component_combo,
                )
                table.setCellWidget(
                    row_index,
                    2,
                    _make_type_combo(str(values.get("layer_type", "image"))),
                )
                table.setCellWidget(
                    row_index,
                    3,
                    channels_selector,
                )
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
                table.setCellWidget(
                    row_index,
                    5,
                    _make_text_input(
                        "" if opacity_value is None else f"{float(opacity_value):.3f}",
                        placeholder_text="Auto",
                    ),
                )
                table.setCellWidget(
                    row_index,
                    6,
                    _make_choice_combo(
                        current_value=str(values.get("blending", "")).strip().lower(),
                        option_values=blending_options,
                    ),
                )
                table.setCellWidget(
                    row_index,
                    7,
                    _make_choice_combo(
                        current_value=str(values.get("colormap", "")).strip(),
                        option_values=colormap_options,
                    ),
                )
                table.setCellWidget(
                    row_index,
                    8,
                    _make_choice_combo(
                        current_value=str(values.get("rendering", "")).strip().lower(),
                        option_values=rendering_options,
                    ),
                )
                table.setCellWidget(
                    row_index,
                    9,
                    _make_multiscale_combo(
                        str(values.get("multiscale_policy", "inherit"))
                    ),
                )
                table.setRowHeight(row_index, row_height)

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
            _apply_initial_dialog_geometry(
                dialog,
                minimum_size=_VOLUME_LAYERS_DIALOG_MINIMUM_SIZE,
                preferred_size=_VOLUME_LAYERS_DIALOG_PREFERRED_SIZE,
                content_size_hint=(
                    dialog.sizeHint().width(),
                    dialog.sizeHint().height(),
                ),
            )

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            rows: list[Dict[str, Any]] = []
            for row_index in range(table.rowCount()):
                name_widget = table.cellWidget(row_index, 0)
                component_widget = table.cellWidget(row_index, 1)
                type_widget = table.cellWidget(row_index, 2)
                channels_widget = table.cellWidget(row_index, 3)
                visible_widget = table.cellWidget(row_index, 4)
                opacity_widget = table.cellWidget(row_index, 5)
                blending_widget = table.cellWidget(row_index, 6)
                colormap_widget = table.cellWidget(row_index, 7)
                rendering_widget = table.cellWidget(row_index, 8)
                multiscale_widget = table.cellWidget(row_index, 9)

                component = ""
                if isinstance(component_widget, QComboBox):
                    component = _read_component_combo_value(component_widget)
                if not component:
                    continue
                row_payload: Dict[str, Any] = {
                    "name": (
                        str(name_widget.text()).strip()
                        if isinstance(name_widget, QLineEdit)
                        else ""
                    ),
                    "component": component,
                    "layer_type": (
                        str(type_widget.currentData() or "image").strip().lower()
                        if isinstance(type_widget, QComboBox)
                        else "image"
                    ),
                    "channels": self._normalize_visualization_channels(
                        (
                            channels_widget.property("_clearexSelectedChannels")
                            if isinstance(channels_widget, QPushButton)
                            else (
                                channels_widget.text()
                                if isinstance(channels_widget, QLineEdit)
                                else ""
                            )
                        )
                    ),
                    "visible": (
                        self._coerce_optional_bool(visible_widget.currentData())
                        if isinstance(visible_widget, QComboBox)
                        else None
                    ),
                    "opacity": self._coerce_optional_unit_interval_float(
                        opacity_widget.text()
                        if isinstance(opacity_widget, QLineEdit)
                        else ""
                    ),
                    "blending": (
                        _read_choice_combo_value(blending_widget).lower()
                        if isinstance(blending_widget, QComboBox)
                        else ""
                    ),
                    "colormap": (
                        _read_choice_combo_value(colormap_widget)
                        if isinstance(colormap_widget, QComboBox)
                        else ""
                    ),
                    "rendering": (
                        _read_choice_combo_value(rendering_widget).lower()
                        if isinstance(rendering_widget, QComboBox)
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
            _apply_initial_dialog_geometry(
                dialog,
                minimum_size=_LAYER_OVERRIDES_DIALOG_MINIMUM_SIZE,
                preferred_size=_LAYER_OVERRIDES_DIALOG_PREFERRED_SIZE,
            )
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
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            table.setColumnWidth(2, 150)
            table.setColumnWidth(3, 150)
            vertical_header = table.verticalHeader()
            vertical_header.setVisible(False)
            vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            row_height = max(40, int(table.fontMetrics().height()) + 18)
            vertical_header.setDefaultSectionSize(row_height)
            root.addWidget(table, stretch=1)

            combo_height = max(30, row_height - 8)

            def _configure_combo_size(combo: QComboBox) -> QComboBox:
                combo.setMinimumHeight(combo_height)
                combo.setMaximumHeight(combo_height)
                combo.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                return cast(
                    QComboBox,
                    _install_table_row_selection_widget_hook(table, combo),
                )

            def _make_text_input(
                value: str,
                *,
                placeholder_text: str = "",
            ) -> QLineEdit:
                line_edit = QLineEdit(table)
                line_edit.setMinimumHeight(combo_height)
                line_edit.setMaximumHeight(combo_height)
                line_edit.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                if placeholder_text:
                    line_edit.setPlaceholderText(str(placeholder_text))
                line_edit.setText(str(value))
                return cast(
                    QLineEdit,
                    _install_table_row_selection_widget_hook(table, line_edit),
                )

            def _make_visible_combo(current_value: Optional[bool]) -> QComboBox:
                combo = _configure_combo_size(QComboBox(table))
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

                table.setCellWidget(
                    row_index,
                    0,
                    _make_text_input(layer_name, placeholder_text="Layer name"),
                )
                table.setCellWidget(row_index, 1, _make_visible_combo(visible))
                table.setCellWidget(
                    row_index,
                    2,
                    _make_text_input(colormap, placeholder_text="Optional LUT"),
                )
                table.setCellWidget(
                    row_index,
                    3,
                    _make_text_input(rendering, placeholder_text="Optional rendering"),
                )
                table.setCellWidget(
                    row_index,
                    4,
                    _make_text_input(annotation, placeholder_text="Optional label"),
                )
                table.setRowHeight(row_index, row_height)

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
            _apply_initial_dialog_geometry(
                dialog,
                minimum_size=_LAYER_OVERRIDES_DIALOG_MINIMUM_SIZE,
                preferred_size=_LAYER_OVERRIDES_DIALOG_PREFERRED_SIZE,
                content_size_hint=(
                    dialog.sizeHint().width(),
                    dialog.sizeHint().height(),
                ),
            )

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            rows: list[Dict[str, Any]] = []
            for row_index in range(table.rowCount()):
                layer_widget = table.cellWidget(row_index, 0)
                visible_widget = table.cellWidget(row_index, 1)
                colormap_widget = table.cellWidget(row_index, 2)
                rendering_widget = table.cellWidget(row_index, 3)
                annotation_widget = table.cellWidget(row_index, 4)
                visible_value: Optional[bool] = None
                if isinstance(visible_widget, QComboBox):
                    visible_value = self._coerce_optional_bool(
                        visible_widget.currentData()
                    )

                row_payload: Dict[str, Any] = {
                    "layer_name": (
                        str(layer_widget.text()).strip()
                        if isinstance(layer_widget, QLineEdit)
                        else ""
                    ),
                    "visible": visible_value,
                    "colormap": (
                        str(colormap_widget.text()).strip()
                        if isinstance(colormap_widget, QLineEdit)
                        else ""
                    ),
                    "rendering": (
                        str(rendering_widget.text()).strip()
                        if isinstance(rendering_widget, QLineEdit)
                        else ""
                    ),
                    "annotation": (
                        str(annotation_widget.text()).strip()
                        if isinstance(annotation_widget, QLineEdit)
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
            self._mip_export_format_combo.addItem("OME-Zarr", "zarr")
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

        @staticmethod
        def _detect_local_gpu_available() -> bool:
            """Return whether at least one compatible local GPU is detected.

            Parameters
            ----------
            None

            Returns
            -------
            bool
                ``True`` when local GPU probes report one or more devices.
            """
            detected_gpu_count = 0
            try:
                recommendation = recommend_local_cluster_config()
            except Exception:
                recommendation = None
            if recommendation is not None:
                detected_gpu_count = int(
                    getattr(recommendation, "detected_gpu_count", 0)
                )
            if detected_gpu_count > 0:
                return True

            # VirtualGL-backed Linux sessions can expose GPU rendering even when
            # local CUDA probes are inconclusive.
            vgl_display = str(os.environ.get("VGL_DISPLAY", "")).strip().lower()
            if vgl_display and vgl_display not in {"0", "false", "off", "none"}:
                return True
            return False

        def _sync_operation_panel_geometry(self) -> None:
            """Recompute panel geometry and reset scroll to the top.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Panel geometry and scroll position are updated in-place.
            """
            if self._operation_panel_stack is None:
                return
            self._operation_panel_stack.updateGeometry()
            self._operation_panel_stack.adjustSize()
            current = self._operation_panel_stack.currentWidget()
            if current is not None:
                current.updateGeometry()
                current.adjustSize()
            if self._operation_panel_scroll is not None:
                self._operation_panel_scroll.ensureVisible(0, 0, 0, 0)
                self._operation_panel_scroll.verticalScrollBar().setValue(0)

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
            self._sync_operation_panel_geometry()
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
                return analysis_auxiliary_root("visualization")
            return self._OPERATION_OUTPUT_COMPONENTS.get(
                operation_name,
                f"clearex/runtime_cache/results/{operation_name}/latest/data",
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

        def _discover_store_output_components(self) -> Dict[str, str]:
            """Discover operation outputs currently present in the selected store.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, str]
                Mapping from operation key to component path for discovered
                existing outputs.
            """
            return _discover_available_operation_output_components(
                store_path=str(self._base_config.file or "").strip(),
                operation_output_components={
                    **{
                        operation_name: component
                        for operation_name, component in self._OPERATION_OUTPUT_COMPONENTS.items()
                    },
                },
            )

        def _provenance_confirmed_operation_names(self) -> list[str]:
            """Return operation keys with successful provenance history.

            Parameters
            ----------
            None

            Returns
            -------
            list[str]
                Operation keys confirmed by provenance history.
            """
            return [
                str(operation_name)
                for operation_name in self._PROVENANCE_HISTORY_OPERATIONS
                if bool(
                    self._operation_history_cache.get(operation_name, {}).get(
                        "has_successful_run", False
                    )
                )
            ]

        def _visualization_volume_layer_component_options(
            self,
        ) -> list[tuple[str, str]]:
            """Build component choices for visualization volume-layer rows.

            Parameters
            ----------
            None

            Returns
            -------
            list[tuple[str, str]]
                Ordered ``(component_path, label)`` choices including raw data
                and provenance-confirmed outputs present in the selected store.
            """
            return _build_visualization_volume_layer_component_options(
                selected_order=self._selected_operations_in_sequence(),
                operation_key_order=self._OPERATION_KEYS,
                operation_labels=self._OPERATION_LABELS,
                available_store_output_components=self._discover_store_output_components(),
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
            available_store_outputs = self._discover_store_output_components()

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
                    unavailable_label = (
                        f"Unavailable source ({current_source})"
                        if current_source in self._OPERATION_KEYS
                        else f"Custom component ({current_source})"
                    )
                    combo.addItem(unavailable_label, current_source)
                    selected_index = combo.count() - 1

                combo.setCurrentIndex(selected_index)
                combo.setEnabled(self._operation_checkboxes[operation_name].isChecked())
                combo.blockSignals(False)

            self._refresh_registration_resolution_level_options()

        def _has_particle_detection_output(self) -> bool:
            """Return whether the active store already contains detections.

            Parameters
            ----------
            None

            Returns
            -------
            bool
                ``True`` when the latest detections component exists.
            """
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return False
            try:
                root = zarr.open_group(store_path, mode="r")
            except Exception:
                return False
            return _zarr_component_exists_in_root(
                root,
                self._PARTICLE_DETECTION_OVERLAY_COMPONENT,
            )

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
                has_particle_detection_output=self._has_particle_detection_output(),
            )

        def _collect_available_dependency_components(
            self,
            references: Sequence[AnalysisInputReference],
        ) -> set[str]:
            """Collect store components available for dependency validation.

            Parameters
            ----------
            references : sequence[AnalysisInputReference]
                Requested analysis input references.

            Returns
            -------
            set[str]
                Components known to exist in the active analysis store.
            """
            available_components = {"data"}
            store_path = str(self._base_config.file or "").strip()
            if not store_path or not is_zarr_store_path(store_path):
                return available_components
            try:
                root = zarr.open_group(store_path, mode="r")
            except Exception:
                return available_components

            for component in self._discover_store_output_components().values():
                available_components.add(str(component))

            for reference in references:
                requested_source = str(reference.requested_source).strip()
                if (
                    not requested_source
                    or requested_source == "data"
                    or requested_source in ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS
                    or requested_source in available_components
                ):
                    continue
                if _zarr_component_exists_in_root(root, requested_source):
                    available_components.add(requested_source)
            return available_components

        def _validate_selected_analysis_dependencies(
            self,
            analysis_parameters: Mapping[str, Mapping[str, Any]],
        ) -> tuple[AnalysisInputDependencyIssue, ...]:
            """Validate currently selected analysis dependencies.

            Parameters
            ----------
            analysis_parameters : mapping[str, mapping[str, Any]]
                Normalized per-operation parameters for the current dialog state.

            Returns
            -------
            tuple[AnalysisInputDependencyIssue, ...]
                Dependency issues returned by the shared workflow validator.
            """
            execution_sequence = self._selected_operations_in_sequence()
            references = collect_analysis_input_references(
                execution_sequence=execution_sequence,
                analysis_parameters=analysis_parameters,
            )
            available_components = self._collect_available_dependency_components(
                references
            )
            return validate_analysis_input_references(
                execution_sequence=execution_sequence,
                analysis_parameters=analysis_parameters,
                available_components=available_components,
            )

        def _set_provenance_history_label_state(
            self,
            *,
            label: QLabel,
            state: str,
            text: str,
            tooltip: str,
        ) -> None:
            """Update one provenance status badge text, tooltip, and state.

            Parameters
            ----------
            label : QLabel
                Provenance status badge label.
            state : str
                Visual state key (for example ``confirmed`` or ``missing``).
            text : str
                Badge text.
            tooltip : str
                Hover hint text.

            Returns
            -------
            None
                Label state is updated in-place.
            """
            label.setText(str(text))
            label.setToolTip(str(tooltip))
            if str(label.property("state") or "") != str(state):
                label.setProperty("state", str(state))
                style = label.style()
                if style is not None:
                    style.unpolish(label)
                    style.polish(label)
                label.update()

        def _on_operation_history_context_menu(
            self,
            operation_name: str,
            position: QPoint,
        ) -> None:
            """Show right-click menu for copying provenance summary text.

            Parameters
            ----------
            operation_name : str
                Operation key associated with the history label.
            position : QPoint
                Click position in label-local coordinates.

            Returns
            -------
            None
                Context menu is shown and clipboard may be updated.
            """
            history_label = self._operation_history_labels.get(operation_name)
            if history_label is None:
                return

            payload = str(self._operation_history_copy_payloads.get(operation_name, ""))
            menu = QMenu(history_label)
            copy_action = menu.addAction("Copy Provenance Info")
            copy_action.setEnabled(bool(payload.strip()))
            selected_action = menu.exec(history_label.mapToGlobal(position))
            if selected_action is not copy_action or not payload.strip():
                return

            app = QApplication.instance()
            if app is None:
                return
            clipboard = app.clipboard()
            if clipboard is None:
                return
            clipboard.setText(payload)
            self._set_status(
                f"Copied {self._OPERATION_LABELS.get(operation_name, operation_name)} provenance info."
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
            empty_history: Dict[str, Any] = {
                "has_successful_run": False,
                "latest_success_run_id": None,
                "latest_success_ended_utc": None,
                "matches_parameters": False,
                "matching_run_id": None,
                "matching_ended_utc": None,
            }
            if not store_path or not is_zarr_store_path(store_path):
                for operation_name in self._PROVENANCE_HISTORY_OPERATIONS:
                    history_label = self._operation_history_labels.get(operation_name)
                    force_checkbox = self._operation_force_rerun_checkboxes.get(
                        operation_name
                    )
                    if history_label is not None:
                        self._set_provenance_history_label_state(
                            label=history_label,
                            state="unavailable",
                            text="Provenance Unavailable",
                            tooltip=(
                                "Current input is not a Zarr/N5 analysis store, so "
                                "provenance cannot be queried."
                            ),
                        )
                    self._operation_history_cache[operation_name] = dict(empty_history)
                    self._operation_history_copy_payloads[operation_name] = ""
                    if force_checkbox is not None:
                        force_checkbox.setVisible(False)
                        force_checkbox.setChecked(False)
                        force_checkbox.setEnabled(False)
                self._refresh_input_source_options()
                return

            for operation_name in self._PROVENANCE_HISTORY_OPERATIONS:
                history_label = self._operation_history_labels.get(operation_name)
                force_checkbox = self._operation_force_rerun_checkboxes.get(
                    operation_name
                )

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
                    history = dict(empty_history)
                    if history_label is not None:
                        self._set_provenance_history_label_state(
                            label=history_label,
                            state="unavailable",
                            text="Provenance Unavailable",
                            tooltip=f"Could not read provenance for this operation: {exc}",
                        )
                    self._operation_history_cache[operation_name] = dict(history)
                    self._operation_history_copy_payloads[operation_name] = ""
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
                    if history_label is not None:
                        self._set_provenance_history_label_state(
                            label=history_label,
                            state="missing",
                            text="No Provenance Record",
                            tooltip="No successful provenance run recorded yet.",
                        )
                    self._operation_history_copy_payloads[operation_name] = ""
                else:
                    selected_run_id = (
                        matching_run_id
                        if matches_parameters and matching_run_id
                        else latest_run_id
                    )
                    selected_ended = (
                        matching_ended
                        if matches_parameters and matching_run_id
                        else latest_ended
                    )
                    tooltip_lines = []
                    if selected_run_id:
                        tooltip_lines.append(f"run_id={selected_run_id}")
                    if selected_ended:
                        tooltip_lines.append(f"ended={selected_ended}")
                    tooltip_lines.append(
                        "parameters match current settings."
                        if matches_parameters
                        else "parameters differ from current settings."
                    )
                    tooltip_text = (
                        "\n".join(tooltip_lines)
                        if tooltip_lines
                        else "Successful provenance run exists."
                    )
                    if history_label is not None:
                        self._set_provenance_history_label_state(
                            label=history_label,
                            state="confirmed",
                            text="Provenance Confirmed",
                            tooltip=tooltip_text,
                        )
                    payload_lines = [f"operation={operation_name}"]
                    if selected_run_id:
                        payload_lines.append(f"run_id={selected_run_id}")
                    if selected_ended:
                        payload_lines.append(f"ended_utc={selected_ended}")
                    payload_lines.append(
                        f"matches_parameters={str(matches_parameters).lower()}"
                    )
                    self._operation_history_copy_payloads[operation_name] = "\n".join(
                        payload_lines
                    )

                if force_checkbox is not None:
                    force_checkbox.setVisible(has_successful_run)
                    if not has_successful_run:
                        force_checkbox.setChecked(False)
                    force_checkbox.setEnabled(
                        has_successful_run
                        and self._operation_checkboxes[operation_name].isChecked()
                    )
            self._refresh_input_source_options()

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
                shape = _analysis_store_runtime_source_shape_tpczyx(root)
            except Exception:
                return None
            if shape is None or len(shape) != 6:
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
                return _analysis_store_runtime_source_dtype_itemsize(root)
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
            self._set_registration_parameter_enabled_state()
            self._set_fusion_parameter_enabled_state()
            self._set_visualization_parameter_enabled_state()
            self._set_render_movie_parameter_enabled_state()
            self._set_compile_movie_parameter_enabled_state()
            self._refresh_volume_export_parameter_bounds()
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
                self._sync_operation_panel_geometry()
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
            self._set_render_movie_parameter_enabled_state()
            self._set_compile_movie_parameter_enabled_state()
            self._refresh_volume_export_parameter_bounds()
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
            _apply_initial_dialog_geometry(
                dialog,
                minimum_size=_SYNTHETIC_PSF_PREVIEW_DIALOG_MINIMUM_SIZE,
                preferred_size=_SYNTHETIC_PSF_PREVIEW_DIALOG_PREFERRED_SIZE,
            )
            dialog.setStyleSheet("""
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
                """)
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
            _apply_initial_dialog_geometry(
                dialog,
                minimum_size=_SYNTHETIC_PSF_PREVIEW_DIALOG_MINIMUM_SIZE,
                preferred_size=_SYNTHETIC_PSF_PREVIEW_DIALOG_PREFERRED_SIZE,
                content_size_hint=(
                    dialog.sizeHint().width(),
                    dialog.sizeHint().height(),
                ),
            )
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

        def _set_registration_parameter_enabled_state(self) -> None:
            """Enable/disable registration widgets based on selection and anchor mode.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widget enabled states are updated in-place.
            """
            registration_enabled = self._operation_checkboxes[
                "registration"
            ].isChecked()
            widgets = (
                self._registration_channel_combo,
                self._registration_type_combo,
                self._registration_resolution_level_combo,
                self._registration_anchor_mode_combo,
                self._registration_pairwise_overlap_z_spin,
                self._registration_pairwise_overlap_y_spin,
                self._registration_pairwise_overlap_x_spin,
                self._registration_max_pairwise_voxels_spin,
                self._registration_use_phase_correlation_check,
                self._registration_use_fft_initial_alignment_check,
            )
            for widget in widgets:
                widget.setEnabled(registration_enabled)

            anchor_mode = (
                str(self._registration_anchor_mode_combo.currentData() or "central")
                .strip()
                .lower()
                or "central"
            )
            manual_anchor = registration_enabled and anchor_mode == "manual"
            self._registration_anchor_position_spin.setEnabled(manual_anchor)
            self._registration_anchor_position_spin.setVisible(manual_anchor)
            self._registration_anchor_position_label.setVisible(manual_anchor)

        def _set_fusion_parameter_enabled_state(self) -> None:
            """Enable/disable fusion widgets based on selection and blend mode."""
            fusion_enabled = self._operation_checkboxes["fusion"].isChecked()
            widgets = (
                self._fusion_overlap_z_spin,
                self._fusion_overlap_y_spin,
                self._fusion_overlap_x_spin,
                self._fusion_blend_mode_combo,
                self._fusion_blend_exponent_spin,
                self._fusion_gain_clip_min_spin,
                self._fusion_gain_clip_max_spin,
            )
            for widget in widgets:
                widget.setEnabled(fusion_enabled)

            blend_mode = (
                str(self._fusion_blend_mode_combo.currentData() or "feather")
                .strip()
                .lower()
                or "feather"
            )
            self._fusion_blend_exponent_spin.setEnabled(
                fusion_enabled and blend_mode != "average"
            )
            gain_clip_enabled = (
                fusion_enabled and blend_mode == "gain_compensated_feather"
            )
            self._fusion_gain_clip_min_spin.setEnabled(gain_clip_enabled)
            self._fusion_gain_clip_max_spin.setEnabled(gain_clip_enabled)

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
            self._visualization_3d_checkbox.setEnabled(visualization_enabled)
            gpu_hint = self._PARAMETER_HINTS["require_gpu_rendering"]
            if not bool(self._local_gpu_available):
                gpu_hint = (
                    "No compatible GPU was auto-detected. Detection can be limited "
                    "on remote or virtualized sessions (for example VirtualGL/EGL). "
                    "You can still enable this option; runtime checks will validate "
                    "the OpenGL renderer."
                )
            self._visualization_require_gpu_checkbox.setEnabled(visualization_enabled)
            self._visualization_require_gpu_checkbox.setToolTip(gpu_hint)
            self._parameter_help_map[self._visualization_require_gpu_checkbox] = (
                gpu_hint
            )
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

        def _set_render_movie_parameter_enabled_state(self) -> None:
            """Enable/disable render-movie widgets based on selection and mode."""
            render_enabled = self._operation_checkboxes["render_movie"].isChecked()
            widgets = (
                self._render_movie_keyframe_manifest_input,
                self._render_movie_resolution_levels_input,
                self._render_movie_width_spin,
                self._render_movie_height_spin,
                self._render_movie_fps_spin,
                self._render_movie_default_transition_frames_spin,
                self._render_movie_transition_frames_input,
                self._render_movie_hold_frames_spin,
                self._render_movie_interpolation_combo,
                self._render_movie_camera_effect_combo,
                self._render_movie_overlay_title_input,
                self._render_movie_overlay_subtitle_input,
                self._render_movie_overlay_frame_text_combo,
                self._render_movie_overlay_scalebar_checkbox,
                self._render_movie_output_directory_input,
            )
            for widget in widgets:
                widget.setEnabled(render_enabled)

            camera_effect = (
                str(self._render_movie_camera_effect_combo.currentData() or "none")
                .strip()
                .lower()
                or "none"
            )
            self._render_movie_orbit_degrees_spin.setEnabled(
                render_enabled and camera_effect == "orbit"
            )
            self._render_movie_flythrough_distance_spin.setEnabled(
                render_enabled and camera_effect == "flythrough"
            )
            self._render_movie_zoom_effect_spin.setEnabled(
                render_enabled and camera_effect == "zoom_fx"
            )

            scalebar_enabled = render_enabled and bool(
                self._render_movie_overlay_scalebar_checkbox.isChecked()
            )
            self._render_movie_overlay_scalebar_length_spin.setEnabled(scalebar_enabled)
            self._render_movie_overlay_scalebar_position_combo.setEnabled(
                scalebar_enabled
            )

        def _set_compile_movie_parameter_enabled_state(self) -> None:
            """Enable/disable compile-movie widgets based on selection and format."""
            compile_enabled = self._operation_checkboxes["compile_movie"].isChecked()
            widgets = (
                self._compile_movie_render_manifest_input,
                self._compile_movie_rendered_level_spin,
                self._compile_movie_output_format_combo,
                self._compile_movie_fps_spin,
                self._compile_movie_pixel_format_input,
                self._compile_movie_resize_width_spin,
                self._compile_movie_resize_height_spin,
                self._compile_movie_output_directory_input,
                self._compile_movie_output_stem_input,
            )
            for widget in widgets:
                widget.setEnabled(compile_enabled)

            output_format = (
                str(self._compile_movie_output_format_combo.currentData() or "mp4")
                .strip()
                .lower()
                or "mp4"
            )
            mp4_enabled = compile_enabled and output_format in {"mp4", "both"}
            prores_enabled = compile_enabled and output_format in {"prores", "both"}
            self._compile_movie_mp4_crf_spin.setEnabled(mp4_enabled)
            self._compile_movie_mp4_preset_combo.setEnabled(mp4_enabled)
            self._compile_movie_prores_profile_spin.setEnabled(prores_enabled)

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

        def _set_volume_export_parameter_enabled_state(self) -> None:
            """Enable/disable volume-export widgets based on operation state."""
            volume_enabled = self._operation_checkboxes["volume_export"].isChecked()
            self._volume_export_scope_combo.setEnabled(volume_enabled)
            self._volume_export_format_combo.setEnabled(volume_enabled)
            self._volume_export_resolution_level_spin.setEnabled(volume_enabled)

            scope = str(
                self._volume_export_scope_combo.currentData() or "current_selection"
            ).strip()
            tpc_enabled = volume_enabled and scope == "current_selection"
            self._volume_export_t_spin.setEnabled(tpc_enabled)
            self._volume_export_p_spin.setEnabled(tpc_enabled)
            self._volume_export_c_spin.setEnabled(tpc_enabled)

            export_format = (
                str(self._volume_export_format_combo.currentData() or "ome-zarr")
                .strip()
                .lower()
            )
            self._volume_export_tiff_layout_combo.setEnabled(
                volume_enabled and export_format == "ome-tiff"
            )

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
            registration_params = dict(
                normalized_parameters.get("registration", self._registration_defaults)
            )
            fusion_params = dict(
                normalized_parameters.get("fusion", self._fusion_defaults)
            )
            visualization_params = dict(
                normalized_parameters.get("visualization", self._visualization_defaults)
            )
            render_movie_params = dict(
                normalized_parameters.get("render_movie", self._render_movie_defaults)
            )
            compile_movie_params = dict(
                normalized_parameters.get(
                    "compile_movie",
                    self._compile_movie_defaults,
                )
            )
            volume_export_params = dict(
                normalized_parameters.get(
                    "volume_export",
                    self._volume_export_defaults,
                )
            )
            mip_export_params = dict(
                normalized_parameters.get("mip_export", self._mip_export_defaults)
            )

            self._populate_analysis_scope_targets(initial)
            self._refresh_dask_backend_summary()

            checkbox_defaults = {
                "flatfield": bool(initial.flatfield),
                "deconvolution": bool(initial.deconvolution),
                "shear_transform": bool(initial.shear_transform),
                "particle_detection": bool(initial.particle_detection),
                "usegment3d": bool(getattr(initial, "usegment3d", False)),
                "registration": bool(initial.registration),
                "fusion": bool(getattr(initial, "fusion", False)),
                "display_pyramid": bool(getattr(initial, "display_pyramid", False)),
                "visualization": bool(initial.visualization),
                "render_movie": bool(getattr(initial, "render_movie", False)),
                "compile_movie": bool(getattr(initial, "compile_movie", False)),
                "volume_export": bool(getattr(initial, "volume_export", False)),
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
            max_source_resolution_level = 0
            store_xy_um: Optional[float] = None
            store_z_um: Optional[float] = None
            if initial.file and is_zarr_store_path(initial.file):
                try:
                    root = zarr.open_group(str(initial.file), mode="r")
                    source_component = _analysis_store_runtime_source_component(root)
                    shape = _analysis_store_runtime_source_shape_tpczyx(root)
                    if source_component is not None and shape is not None:
                        position_count = max(1, int(shape[1]))
                        channel_count = max(1, int(shape[2]))
                        voxel_size_um_zyx, _ = resolve_voxel_size_um_zyx_with_source(
                            root,
                            source_component=source_component,
                        )
                        store_z_um = float(voxel_size_um_zyx[0])
                        store_xy_um = float(voxel_size_um_zyx[2])
                        available_levels = _discover_component_resolution_levels(
                            root=root,
                            source_component=source_component,
                        )
                        if available_levels:
                            max_source_resolution_level = max(
                                max(0, int(level)) for level in available_levels
                            )
                except Exception:
                    position_count = 1
                    channel_count = 1
                    max_source_resolution_level = 0
                    store_xy_um = None
                    store_z_um = None
            self._visualization_position_spin.setMaximum(position_count - 1)
            self._particle_channel_spin.setMaximum(channel_count - 1)
            self._usegment3d_resolution_level_spin.setMaximum(
                max(0, int(max_source_resolution_level))
            )
            self._refresh_registration_channel_options()
            self._refresh_registration_resolution_level_options()
            self._registration_anchor_position_spin.setMaximum(position_count - 1)
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

            self._refresh_registration_channel_options(
                preferred_channel=max(
                    0,
                    int(registration_params.get("registration_channel", 0)),
                )
            )
            registration_type = (
                str(registration_params.get("registration_type", "translation"))
                .strip()
                .lower()
                or "translation"
            )
            registration_type_index = self._registration_type_combo.findData(
                registration_type
            )
            if registration_type_index < 0:
                registration_type_index = self._registration_type_combo.findData(
                    "translation"
                )
            if registration_type_index < 0:
                registration_type_index = 0
            self._registration_type_combo.setCurrentIndex(registration_type_index)

            registration_resolution_level = max(
                0,
                int(registration_params.get("input_resolution_level", 0)),
            )
            self._refresh_registration_resolution_level_options(
                preferred_level=int(registration_resolution_level)
            )

            anchor_mode = (
                str(registration_params.get("anchor_mode", "central")).strip().lower()
                or "central"
            )
            if anchor_mode not in {"central", "manual"}:
                anchor_mode = "central"
            anchor_mode_index = self._registration_anchor_mode_combo.findData(
                anchor_mode
            )
            if anchor_mode_index < 0:
                anchor_mode_index = self._registration_anchor_mode_combo.findData(
                    "central"
                )
            if anchor_mode_index < 0:
                anchor_mode_index = 0
            self._registration_anchor_mode_combo.setCurrentIndex(anchor_mode_index)
            anchor_position = registration_params.get("anchor_position")
            if anchor_position in {None, ""}:
                parsed_anchor_position = 0
            else:
                parsed_anchor_position = int(anchor_position)
            self._registration_anchor_position_spin.setValue(
                max(
                    0,
                    min(
                        int(self._registration_anchor_position_spin.maximum()),
                        int(parsed_anchor_position),
                    ),
                )
            )
            registration_overlap_zyx = registration_params.get(
                "pairwise_overlap_zyx",
                registration_params.get("overlap_zyx", [8, 32, 32]),
            )
            if (
                not isinstance(registration_overlap_zyx, (tuple, list))
                or len(registration_overlap_zyx) != 3
            ):
                registration_overlap_zyx = [8, 32, 32]
            self._registration_pairwise_overlap_z_spin.setValue(
                max(0, int(registration_overlap_zyx[0]))
            )
            self._registration_pairwise_overlap_y_spin.setValue(
                max(0, int(registration_overlap_zyx[1]))
            )
            self._registration_pairwise_overlap_x_spin.setValue(
                max(0, int(registration_overlap_zyx[2]))
            )

            fusion_blend_overlap_zyx = fusion_params.get(
                "blend_overlap_zyx",
                fusion_params.get("overlap_zyx", [8, 32, 32]),
            )
            if (
                not isinstance(fusion_blend_overlap_zyx, (tuple, list))
                or len(fusion_blend_overlap_zyx) != 3
            ):
                fusion_blend_overlap_zyx = [8, 32, 32]
            self._fusion_overlap_z_spin.setValue(
                max(0, int(fusion_blend_overlap_zyx[0]))
            )
            self._fusion_overlap_y_spin.setValue(
                max(0, int(fusion_blend_overlap_zyx[1]))
            )
            self._fusion_overlap_x_spin.setValue(
                max(0, int(fusion_blend_overlap_zyx[2]))
            )
            fusion_blend_mode = (
                str(fusion_params.get("blend_mode", "feather")).strip().lower()
                or "feather"
            )
            fusion_blend_index = self._fusion_blend_mode_combo.findData(
                fusion_blend_mode
            )
            if fusion_blend_index < 0:
                fusion_blend_index = self._fusion_blend_mode_combo.findData("feather")
            if fusion_blend_index < 0:
                fusion_blend_index = 0
            self._fusion_blend_mode_combo.setCurrentIndex(fusion_blend_index)
            self._fusion_blend_exponent_spin.setValue(
                max(0.10, float(fusion_params.get("blend_exponent", 1.0)))
            )
            gain_clip_range = fusion_params.get("gain_clip_range", [0.25, 4.0])
            if (
                not isinstance(gain_clip_range, (tuple, list))
                or len(gain_clip_range) != 2
            ):
                gain_clip_range = [0.25, 4.0]
            self._fusion_gain_clip_min_spin.setValue(
                max(0.01, float(gain_clip_range[0]))
            )
            self._fusion_gain_clip_max_spin.setValue(
                max(
                    float(self._fusion_gain_clip_min_spin.value()),
                    float(gain_clip_range[1]),
                )
            )

            self._registration_max_pairwise_voxels_spin.setValue(
                max(0, int(registration_params.get("max_pairwise_voxels", 500_000)))
            )
            self._registration_use_phase_correlation_check.setChecked(
                bool(registration_params.get("use_phase_correlation", False))
            )
            self._registration_use_fft_initial_alignment_check.setChecked(
                bool(registration_params.get("use_fft_initial_alignment", True))
            )

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
            self._visualization_3d_checkbox.setChecked(
                bool(visualization_params.get("use_3d_view", True))
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

            self._render_movie_keyframe_manifest_input.setText(
                str(render_movie_params.get("keyframe_manifest_path", "") or "").strip()
            )
            self._render_movie_resolution_levels_input.setText(
                ",".join(
                    str(int(value))
                    for value in render_movie_params.get("resolution_levels", [0])
                )
            )
            render_size_xy = render_movie_params.get("render_size_xy", [1920, 1080])
            if (
                not isinstance(render_size_xy, (tuple, list))
                or len(render_size_xy) != 2
            ):
                render_size_xy = [1920, 1080]
            self._render_movie_width_spin.setValue(max(64, int(render_size_xy[0])))
            self._render_movie_height_spin.setValue(max(64, int(render_size_xy[1])))
            self._render_movie_fps_spin.setValue(
                max(1, int(render_movie_params.get("fps", 24)))
            )
            self._render_movie_default_transition_frames_spin.setValue(
                max(0, int(render_movie_params.get("default_transition_frames", 48)))
            )
            self._render_movie_transition_frames_input.setText(
                ",".join(
                    str(int(value))
                    for value in render_movie_params.get(
                        "transition_frames_by_gap",
                        [],
                    )
                )
            )
            self._render_movie_hold_frames_spin.setValue(
                max(0, int(render_movie_params.get("hold_frames", 12)))
            )
            interpolation_mode = (
                str(render_movie_params.get("interpolation_mode", "ease_in_out"))
                .strip()
                .lower()
                or "ease_in_out"
            )
            interpolation_index = self._render_movie_interpolation_combo.findData(
                interpolation_mode
            )
            if interpolation_index < 0:
                interpolation_index = 0
            self._render_movie_interpolation_combo.setCurrentIndex(interpolation_index)
            camera_effect = (
                str(render_movie_params.get("camera_effect", "none")).strip().lower()
                or "none"
            )
            camera_effect_index = self._render_movie_camera_effect_combo.findData(
                camera_effect
            )
            if camera_effect_index < 0:
                camera_effect_index = 0
            self._render_movie_camera_effect_combo.setCurrentIndex(camera_effect_index)
            self._render_movie_orbit_degrees_spin.setValue(
                max(0.0, float(render_movie_params.get("orbit_degrees", 45.0)))
            )
            self._render_movie_flythrough_distance_spin.setValue(
                max(
                    0.0,
                    float(render_movie_params.get("flythrough_distance_factor", 0.10)),
                )
            )
            self._render_movie_zoom_effect_spin.setValue(
                max(0.0, float(render_movie_params.get("zoom_effect_factor", 0.15)))
            )
            self._render_movie_overlay_title_input.setText(
                str(render_movie_params.get("overlay_title", "") or "").strip()
            )
            self._render_movie_overlay_subtitle_input.setText(
                str(render_movie_params.get("overlay_subtitle", "") or "").strip()
            )
            frame_text_mode = (
                str(render_movie_params.get("overlay_frame_text_mode", "none"))
                .strip()
                .lower()
                or "none"
            )
            frame_text_index = self._render_movie_overlay_frame_text_combo.findData(
                frame_text_mode
            )
            if frame_text_index < 0:
                frame_text_index = 0
            self._render_movie_overlay_frame_text_combo.setCurrentIndex(
                frame_text_index
            )
            self._render_movie_overlay_scalebar_checkbox.setChecked(
                bool(render_movie_params.get("overlay_scalebar", False))
            )
            self._render_movie_overlay_scalebar_length_spin.setValue(
                max(
                    0.001,
                    float(render_movie_params.get("overlay_scalebar_length_um", 50.0)),
                )
            )
            scalebar_position = (
                str(render_movie_params.get("overlay_scalebar_position", "bottom_left"))
                .strip()
                .lower()
                or "bottom_left"
            )
            scalebar_position_index = (
                self._render_movie_overlay_scalebar_position_combo.findData(
                    scalebar_position
                )
            )
            if scalebar_position_index < 0:
                scalebar_position_index = 0
            self._render_movie_overlay_scalebar_position_combo.setCurrentIndex(
                scalebar_position_index
            )
            self._render_movie_output_directory_input.setText(
                str(render_movie_params.get("output_directory", "") or "").strip()
            )

            self._compile_movie_render_manifest_input.setText(
                str(compile_movie_params.get("render_manifest_path", "") or "").strip()
            )
            self._compile_movie_rendered_level_spin.setValue(
                max(0, int(compile_movie_params.get("rendered_level", 0)))
            )
            compile_output_format = (
                str(compile_movie_params.get("output_format", "mp4")).strip().lower()
                or "mp4"
            )
            compile_output_format_index = (
                self._compile_movie_output_format_combo.findData(compile_output_format)
            )
            if compile_output_format_index < 0:
                compile_output_format_index = 0
            self._compile_movie_output_format_combo.setCurrentIndex(
                compile_output_format_index
            )
            self._compile_movie_fps_spin.setValue(
                max(1, int(compile_movie_params.get("fps", 24)))
            )
            self._compile_movie_mp4_crf_spin.setValue(
                max(0, min(51, int(compile_movie_params.get("mp4_crf", 18))))
            )
            compile_mp4_preset = (
                str(compile_movie_params.get("mp4_preset", "slow")).strip().lower()
                or "slow"
            )
            compile_mp4_preset_index = self._compile_movie_mp4_preset_combo.findData(
                compile_mp4_preset
            )
            if compile_mp4_preset_index < 0:
                compile_mp4_preset_index = (
                    self._compile_movie_mp4_preset_combo.findData("slow")
                )
            if compile_mp4_preset_index < 0:
                compile_mp4_preset_index = 0
            self._compile_movie_mp4_preset_combo.setCurrentIndex(
                compile_mp4_preset_index
            )
            self._compile_movie_prores_profile_spin.setValue(
                max(0, min(5, int(compile_movie_params.get("prores_profile", 3))))
            )
            self._compile_movie_pixel_format_input.setText(
                str(compile_movie_params.get("pixel_format", "") or "").strip()
            )
            resize_xy = compile_movie_params.get("resize_xy", [])
            if not isinstance(resize_xy, (tuple, list)) or len(resize_xy) != 2:
                resize_xy = [0, 0]
            self._compile_movie_resize_width_spin.setValue(max(0, int(resize_xy[0])))
            self._compile_movie_resize_height_spin.setValue(max(0, int(resize_xy[1])))
            self._compile_movie_output_directory_input.setText(
                str(compile_movie_params.get("output_directory", "") or "").strip()
            )
            self._compile_movie_output_stem_input.setText(
                str(compile_movie_params.get("output_stem", "") or "").strip()
            )

            volume_export_scope = str(
                volume_export_params.get("export_scope", "current_selection")
            ).strip()
            volume_export_scope_index = self._volume_export_scope_combo.findData(
                volume_export_scope
            )
            if volume_export_scope_index < 0:
                volume_export_scope_index = self._volume_export_scope_combo.findData(
                    "current_selection"
                )
            if volume_export_scope_index < 0:
                volume_export_scope_index = 0
            self._volume_export_scope_combo.setCurrentIndex(volume_export_scope_index)
            self._refresh_volume_export_parameter_bounds()
            self._volume_export_t_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_t_spin.maximum()),
                        int(volume_export_params.get("t_index", 0)),
                    ),
                )
            )
            self._volume_export_p_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_p_spin.maximum()),
                        int(volume_export_params.get("p_index", 0)),
                    ),
                )
            )
            self._volume_export_c_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_c_spin.maximum()),
                        int(volume_export_params.get("c_index", 0)),
                    ),
                )
            )
            self._volume_export_resolution_level_spin.setValue(
                max(
                    0,
                    min(
                        int(self._volume_export_resolution_level_spin.maximum()),
                        int(volume_export_params.get("resolution_level", 0)),
                    ),
                )
            )
            volume_export_format = (
                str(volume_export_params.get("export_format", "ome-zarr"))
                .strip()
                .lower()
            )
            if volume_export_format in {"zarr", "ome_zarr", "ome.zarr"}:
                volume_export_format = "ome-zarr"
            volume_export_format_index = self._volume_export_format_combo.findData(
                volume_export_format
            )
            if volume_export_format_index < 0:
                volume_export_format_index = self._volume_export_format_combo.findData(
                    "ome-zarr"
                )
            if volume_export_format_index < 0:
                volume_export_format_index = 0
            self._volume_export_format_combo.setCurrentIndex(volume_export_format_index)
            volume_export_tiff_layout = (
                str(volume_export_params.get("tiff_file_layout", "single_file"))
                .strip()
                .lower()
            )
            volume_export_tiff_layout_index = (
                self._volume_export_tiff_layout_combo.findData(
                    volume_export_tiff_layout
                )
            )
            if volume_export_tiff_layout_index < 0:
                volume_export_tiff_layout_index = (
                    self._volume_export_tiff_layout_combo.findData("single_file")
                )
            if volume_export_tiff_layout_index < 0:
                volume_export_tiff_layout_index = 0
            self._volume_export_tiff_layout_combo.setCurrentIndex(
                volume_export_tiff_layout_index
            )
            self._refresh_volume_export_parameter_bounds()
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

            self._set_volume_export_parameter_enabled_state()
            self._refresh_operation_provenance_statuses()
            self._on_operation_selection_changed()
            if self._operation_panel_stack is not None:
                self._operation_panel_stack.setCurrentIndex(0)
                self._sync_operation_panel_geometry()
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
            self.setStyleSheet("""
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Avenir Next", "Helvetica Neue", "Arial", sans-serif;
                    font-size: 14px;
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
                    font-size: 14px;
                    color: #b8c8dd;
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
                QScrollArea#analysisDialogScroll {
                    border: none;
                    background: transparent;
                }
                QWidget#analysisDialogContent {
                    background: transparent;
                }
                QScrollArea {
                    border: none;
                    background: transparent;
                }
                QScrollArea#operationPanelScroll {
                    background-color: #111925;
                    border-radius: 10px;
                }
                QWidget#operationPanelViewport {
                    background-color: #111925;
                    border-radius: 10px;
                }
                QStackedWidget#operationPanelStack {
                    background-color: #111925;
                    border-radius: 10px;
                }
                QStackedWidget#operationPanelStack > QWidget {
                    background-color: #111925;
                }
                QScrollArea#usegment3dChannelScroll {
                    border: 1px solid #2b3f58;
                    border-radius: 8px;
                    background-color: #101a29;
                }
                QWidget#usegment3dChannelViewport {
                    background-color: #101a29;
                    border-radius: 8px;
                }
                QScrollArea#usegment3dChannelScroll > QWidget {
                    background-color: #101a29;
                }
                QWidget#usegment3dChannelContainer {
                    background-color: #101a29;
                    border-radius: 8px;
                }
                QWidget#operationPanel {
                    background-color: #111925;
                    border: 1px solid #2a3442;
                    border-radius: 10px;
                    padding: 8px;
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
                    font-size: 13px;
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
                QLabel:disabled, QCheckBox:disabled {
                    color: #9cb1cc;
                }
                QLabel#provenanceStatusLabel {
                    font-weight: 600;
                    border-radius: 8px;
                    padding: 4px 8px;
                }
                QLabel#provenanceStatusLabel[state=\"confirmed\"] {
                    color: #b8f2c8;
                    background-color: #143323;
                    border: 1px solid #27543a;
                }
                QLabel#provenanceStatusLabel[state=\"missing\"] {
                    color: #ffe2a7;
                    background-color: #3a2d14;
                    border: 1px solid #6a4d1d;
                }
                QLabel#provenanceStatusLabel[state=\"unavailable\"] {
                    color: #c7d5e8;
                    background-color: #1d2532;
                    border: 1px solid #2d3b50;
                }
                QLabel#statusLabel {
                    color: #c3d2e6;
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
                QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                    color: #afc1d8;
                    background-color: #142235;
                    border-color: #2a3c54;
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
                QPushButton:disabled {
                    color: #9cb1cc;
                    background-color: #142235;
                    border-color: #2a3c54;
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
                """)

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

        def _collect_registration_parameters(self) -> Dict[str, Any]:
            """Collect registration parameter values from widgets.

            Parameters
            ----------
            None

            Returns
            -------
            dict[str, Any]
                Registration parameter mapping.
            """
            anchor_mode = (
                str(self._registration_anchor_mode_combo.currentData() or "central")
                .strip()
                .lower()
                or "central"
            )
            anchor_position: Optional[int]
            if anchor_mode == "manual":
                anchor_position = int(self._registration_anchor_position_spin.value())
            else:
                anchor_position = None

            resolution_level_data = (
                self._registration_resolution_level_combo.currentData()
            )
            try:
                input_resolution_level = max(0, int(resolution_level_data))
            except (TypeError, ValueError):
                input_resolution_level = 0
            channel_data = self._registration_channel_combo.currentData()
            try:
                registration_channel = max(0, int(channel_data))
            except (TypeError, ValueError):
                registration_channel = 0

            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": True,
                "pairwise_overlap_zyx": [
                    int(self._registration_pairwise_overlap_z_spin.value()),
                    int(self._registration_pairwise_overlap_y_spin.value()),
                    int(self._registration_pairwise_overlap_x_spin.value()),
                ],
                "memory_overhead_factor": float(
                    self._registration_defaults.get("memory_overhead_factor", 2.5)
                ),
                "registration_channel": int(registration_channel),
                "registration_type": str(
                    self._registration_type_combo.currentData() or "rigid"
                ).strip()
                or "rigid",
                "input_resolution_level": int(input_resolution_level),
                "anchor_mode": anchor_mode,
                "anchor_position": anchor_position,
                "max_pairwise_voxels": int(
                    self._registration_max_pairwise_voxels_spin.value()
                ),
                "ants_iterations": list(
                    self._registration_defaults.get(
                        "ants_iterations", [200, 100, 50, 25]
                    )
                ),
                "ants_sampling_rate": float(
                    self._registration_defaults.get("ants_sampling_rate", 0.20)
                ),
                "use_phase_correlation": bool(
                    self._registration_use_phase_correlation_check.isChecked()
                ),
                "use_fft_initial_alignment": bool(
                    self._registration_use_fft_initial_alignment_check.isChecked()
                ),
            }

        def _collect_fusion_parameters(self) -> Dict[str, Any]:
            """Collect fusion parameter values from widgets."""
            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": True,
                "blend_overlap_zyx": [
                    int(self._fusion_overlap_z_spin.value()),
                    int(self._fusion_overlap_y_spin.value()),
                    int(self._fusion_overlap_x_spin.value()),
                ],
                "memory_overhead_factor": float(
                    self._fusion_defaults.get("memory_overhead_factor", 2.5)
                ),
                "blend_mode": str(
                    self._fusion_blend_mode_combo.currentData() or "feather"
                ).strip()
                or "feather",
                "blend_exponent": float(self._fusion_blend_exponent_spin.value()),
                "gain_clip_range": [
                    float(self._fusion_gain_clip_min_spin.value()),
                    float(
                        max(
                            self._fusion_gain_clip_min_spin.value(),
                            self._fusion_gain_clip_max_spin.value(),
                        )
                    ),
                ],
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
            self._sync_visualization_volume_layers_from_input_source(
                refresh_summary=False
            )
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
                "use_3d_view": bool(self._visualization_3d_checkbox.isChecked()),
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

        def _collect_render_movie_parameters(self) -> Dict[str, Any]:
            """Collect render-movie parameter values from widgets."""
            resolution_levels = [
                int(token)
                for token in self._split_csv_values(
                    self._render_movie_resolution_levels_input.text()
                )
            ]
            transition_frames = [
                int(token)
                for token in self._split_csv_values(
                    self._render_movie_transition_frames_input.text()
                )
            ]
            return {
                "chunk_basis": "2d",
                "detect_2d_per_slice": True,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._render_movie_defaults.get("memory_overhead_factor", 1.0)
                ),
                "keyframe_manifest_path": str(
                    self._render_movie_keyframe_manifest_input.text()
                ).strip(),
                "resolution_levels": resolution_levels
                or list(self._render_movie_defaults.get("resolution_levels", [0])),
                "render_size_xy": [
                    int(self._render_movie_width_spin.value()),
                    int(self._render_movie_height_spin.value()),
                ],
                "fps": int(self._render_movie_fps_spin.value()),
                "default_transition_frames": int(
                    self._render_movie_default_transition_frames_spin.value()
                ),
                "transition_frames_by_gap": list(transition_frames),
                "hold_frames": int(self._render_movie_hold_frames_spin.value()),
                "interpolation_mode": str(
                    self._render_movie_interpolation_combo.currentData()
                    or "ease_in_out"
                ).strip(),
                "camera_effect": str(
                    self._render_movie_camera_effect_combo.currentData() or "none"
                ).strip(),
                "orbit_degrees": float(self._render_movie_orbit_degrees_spin.value()),
                "flythrough_distance_factor": float(
                    self._render_movie_flythrough_distance_spin.value()
                ),
                "zoom_effect_factor": float(
                    self._render_movie_zoom_effect_spin.value()
                ),
                "overlay_title": str(
                    self._render_movie_overlay_title_input.text()
                ).strip(),
                "overlay_subtitle": str(
                    self._render_movie_overlay_subtitle_input.text()
                ).strip(),
                "overlay_frame_text_mode": str(
                    self._render_movie_overlay_frame_text_combo.currentData() or "none"
                ).strip(),
                "overlay_scalebar": bool(
                    self._render_movie_overlay_scalebar_checkbox.isChecked()
                ),
                "overlay_scalebar_length_um": float(
                    self._render_movie_overlay_scalebar_length_spin.value()
                ),
                "overlay_scalebar_position": str(
                    self._render_movie_overlay_scalebar_position_combo.currentData()
                    or "bottom_left"
                ).strip(),
                "output_directory": str(
                    self._render_movie_output_directory_input.text()
                ).strip(),
            }

        def _collect_compile_movie_parameters(self) -> Dict[str, Any]:
            """Collect compile-movie parameter values from widgets."""
            resize_xy: list[int] = []
            resize_width = int(self._compile_movie_resize_width_spin.value())
            resize_height = int(self._compile_movie_resize_height_spin.value())
            if resize_width > 0 and resize_height > 0:
                resize_xy = [resize_width, resize_height]
            return {
                "chunk_basis": "2d",
                "detect_2d_per_slice": True,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._compile_movie_defaults.get("memory_overhead_factor", 1.0)
                ),
                "render_manifest_path": str(
                    self._compile_movie_render_manifest_input.text()
                ).strip(),
                "rendered_level": int(self._compile_movie_rendered_level_spin.value()),
                "output_format": str(
                    self._compile_movie_output_format_combo.currentData() or "mp4"
                ).strip(),
                "fps": int(self._compile_movie_fps_spin.value()),
                "mp4_crf": int(self._compile_movie_mp4_crf_spin.value()),
                "mp4_preset": str(
                    self._compile_movie_mp4_preset_combo.currentData() or "slow"
                ).strip(),
                "prores_profile": int(self._compile_movie_prores_profile_spin.value()),
                "pixel_format": str(
                    self._compile_movie_pixel_format_input.text()
                ).strip(),
                "resize_xy": resize_xy,
                "output_directory": str(
                    self._compile_movie_output_directory_input.text()
                ).strip(),
                "output_stem": str(
                    self._compile_movie_output_stem_input.text()
                ).strip(),
            }

        def _collect_volume_export_parameters(self) -> Dict[str, Any]:
            """Collect volume-export parameter values from widgets."""
            return {
                "chunk_basis": "3d",
                "detect_2d_per_slice": False,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "memory_overhead_factor": float(
                    self._volume_export_defaults.get("memory_overhead_factor", 1.0)
                ),
                "export_scope": str(
                    self._volume_export_scope_combo.currentData() or "current_selection"
                ).strip(),
                "t_index": int(self._volume_export_t_spin.value()),
                "p_index": int(self._volume_export_p_spin.value()),
                "c_index": int(self._volume_export_c_spin.value()),
                "resolution_level": int(
                    self._volume_export_resolution_level_spin.value()
                ),
                "export_format": str(
                    self._volume_export_format_combo.currentData() or "ome-zarr"
                )
                .strip()
                .lower(),
                "tiff_file_layout": str(
                    self._volume_export_tiff_layout_combo.currentData() or "single_file"
                )
                .strip()
                .lower(),
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
            elif operation_name == "registration":
                defaults.update(self._collect_registration_parameters())
            elif operation_name == "fusion":
                defaults.update(self._collect_fusion_parameters())
            elif operation_name == "visualization":
                defaults.update(self._collect_visualization_parameters())
            elif operation_name == "render_movie":
                defaults.update(self._collect_render_movie_parameters())
            elif operation_name == "compile_movie":
                defaults.update(self._collect_compile_movie_parameters())
            elif operation_name == "volume_export":
                defaults.update(self._collect_volume_export_parameters())
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
            dependency_issues = self._validate_selected_analysis_dependencies(
                analysis_parameters
            )
            if dependency_issues:
                first_issue = dependency_issues[0]
                QMessageBox.warning(
                    self,
                    "Invalid Analysis Dependencies",
                    str(first_issue.message),
                )
                self._set_status(str(first_issue.message))
                return

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

            selected_target = self._current_analysis_target()
            workflow_kwargs: Dict[str, Any] = {
                "file": (
                    str(selected_target.store_path)
                    if selected_target is not None
                    else self._base_config.file
                ),
                "analysis_targets": self._analysis_targets,
                "analysis_selected_experiment_path": (
                    str(selected_target.experiment_path)
                    if selected_target is not None
                    else self._base_config.analysis_selected_experiment_path
                ),
                "analysis_apply_to_all": bool(
                    self._analysis_apply_to_all_checkbox.isChecked()
                    if self._analysis_apply_to_all_checkbox is not None
                    else False
                ),
                "prefer_dask": self._base_config.prefer_dask,
                "dask_backend": self._dask_backend_config,
                "chunks": self._base_config.chunks,
                "flatfield": selected_flags["flatfield"],
                "deconvolution": selected_flags["deconvolution"],
                "shear_transform": selected_flags["shear_transform"],
                "particle_detection": selected_flags["particle_detection"],
                "registration": selected_flags["registration"],
                "fusion": selected_flags["fusion"],
                "display_pyramid": selected_flags["display_pyramid"],
                "visualization": selected_flags["visualization"],
                "render_movie": selected_flags["render_movie"],
                "compile_movie": selected_flags["compile_movie"],
                "volume_export": selected_flags["volume_export"],
                "mip_export": selected_flags["mip_export"],
                "zarr_save": self._base_config.zarr_save,
                "spatial_calibration": self._base_config.spatial_calibration,
                "spatial_calibration_explicit": bool(
                    self._base_config.spatial_calibration_explicit
                ),
                "analysis_parameters": analysis_parameters,
            }
            dataclass_fields = getattr(WorkflowConfig, "__dataclass_fields__", {})
            if "usegment3d" in dataclass_fields:
                workflow_kwargs["usegment3d"] = selected_flags["usegment3d"]
            self.result_config = WorkflowConfig(**workflow_kwargs)
            self._persist_analysis_gui_state_for_target(selected_target)
            _save_last_used_dask_backend_config(self._dask_backend_config)
            sequence = self._selected_operations_in_sequence()
            sequence_text = " -> ".join(
                self._OPERATION_LABELS[name] for name in sequence
            )
            if (
                self.result_config.analysis_apply_to_all
                and len(self.result_config.analysis_targets) > 1
            ):
                self._set_status(
                    "Launching selected analysis routines for all listed "
                    f"experiments: {sequence_text}"
                )
            else:
                self._set_status(
                    f"Launching selected analysis routines: {sequence_text}"
                )
            self.accept()

        def reject(self) -> None:
            """Persist current GUI state before dismissing the dialog.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Dialog state is persisted best-effort before rejection.
            """
            self._persist_analysis_gui_state_for_target(self._active_analysis_target)
            super().reject()

        def closeEvent(self, event: QCloseEvent) -> None:
            """Persist current GUI state before the dialog closes.

            Parameters
            ----------
            event : QCloseEvent
                Close event emitted by Qt.

            Returns
            -------
            None
                Close handling is delegated to ``QDialog`` after persistence.
            """
            self._persist_analysis_gui_state_for_target(self._active_analysis_target)
            super().closeEvent(event)


def launch_gui(
    initial: Optional[WorkflowConfig] = None,
    run_callback: Optional[GuiRunCallback] = None,
    dask_client_lifecycle_callback: Optional[DaskClientLifecycleCallback] = None,
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
        ``(workflow, progress_callback, dask_client_lifecycle_callback=None)``.
    dask_client_lifecycle_callback : callable, optional
        Optional lifecycle callback forwarded into
        :func:`run_workflow_with_progress` during iterative GUI execution.

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
    zarr_settings_path = _resolve_zarr_save_settings_path(settings_directory)
    effective_initial = initial or WorkflowConfig()
    persisted_backend = _load_last_used_dask_backend_config(settings_path=settings_path)
    persisted_zarr_save = _load_last_used_zarr_save_config(
        settings_path=zarr_settings_path
    )
    if persisted_backend is not None and _should_apply_persisted_dask_backend(initial):
        effective_initial = replace(effective_initial, dask_backend=persisted_backend)
    if persisted_zarr_save is not None and _should_apply_persisted_zarr_save(initial):
        effective_initial = replace(effective_initial, zarr_save=persisted_zarr_save)

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    _apply_application_icon(app)
    get_dashboard_relay_manager()

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
                    dask_client_lifecycle_callback=dask_client_lifecycle_callback,
                )
                if not completed:
                    continue
                current = _reset_analysis_selection_for_next_run(current)
                _persist_reset_analysis_gui_state_for_workflow(current)

    if owns_app:
        app.quit()
        _shutdown_dashboard_relay_manager()
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
    operation_names = tuple(default_analysis_operation_parameters().keys())
    analysis_parameters = normalize_analysis_operation_parameters(
        workflow.analysis_parameters
    )
    for operation_name in operation_names:
        params = dict(analysis_parameters.get(operation_name, {}))
        params["force_rerun"] = False
        analysis_parameters[operation_name] = params

    workflow_kwargs: Dict[str, Any] = {
        "file": workflow.file,
        "analysis_targets": workflow.analysis_targets,
        "analysis_selected_experiment_path": (
            workflow.analysis_selected_experiment_path
        ),
        "analysis_apply_to_all": workflow.analysis_apply_to_all,
        "prefer_dask": workflow.prefer_dask,
        "dask_backend": workflow.dask_backend,
        "chunks": workflow.chunks,
        "zarr_save": workflow.zarr_save,
        "spatial_calibration": workflow.spatial_calibration,
        "spatial_calibration_explicit": workflow.spatial_calibration_explicit,
        "analysis_parameters": analysis_parameters,
    }
    dataclass_fields = getattr(WorkflowConfig, "__dataclass_fields__", {})
    for operation_name in operation_names:
        if operation_name in dataclass_fields:
            workflow_kwargs[operation_name] = False
    return WorkflowConfig(**workflow_kwargs)


def _analysis_gui_state_payload_from_workflow(
    workflow: WorkflowConfig,
    *,
    analysis_selected_tab: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize analysis-dialog state directly from a workflow config.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow whose analysis GUI state should be serialized.
    analysis_selected_tab : str, optional
        Persisted Analysis Methods tab label to preserve across GUI resets.

    Returns
    -------
    dict[str, Any]
        Dataset-local GUI-state payload compatible with
        ``persist_latest_analysis_gui_state``.
    """
    operation_names = tuple(default_analysis_operation_parameters().keys())
    payload: Dict[str, Any] = {
        operation_name: bool(getattr(workflow, operation_name, False))
        for operation_name in operation_names
    }
    payload["analysis_parameters"] = normalize_analysis_operation_parameters(
        workflow.analysis_parameters
    )
    payload["file"] = str(workflow.file or "")
    payload["analysis_selected_experiment_path"] = str(
        workflow.analysis_selected_experiment_path or ""
    )
    payload["spatial_calibration"] = spatial_calibration_to_dict(
        workflow.spatial_calibration
    )
    payload["spatial_calibration_explicit"] = bool(
        workflow.spatial_calibration_explicit
    )
    selected_tab = str(analysis_selected_tab or "").strip()
    if selected_tab:
        payload["analysis_selected_tab"] = selected_tab
    return payload


def _persist_reset_analysis_gui_state_for_workflow(workflow: WorkflowConfig) -> None:
    """Persist a post-success reset GUI state for the workflow scope.

    Parameters
    ----------
    workflow : WorkflowConfig
        Reset workflow state that should become the next default GUI state.

    Returns
    -------
    None
        Best-effort persistence only.
    """
    logger = logging.getLogger(__name__)
    for scoped_workflow in _workflows_for_selected_analysis_scope(workflow):
        store_path = str(scoped_workflow.file or "").strip()
        if not store_path or not is_zarr_store_path(store_path):
            continue
        analysis_selected_tab = ""
        try:
            existing_state = load_latest_analysis_gui_state(store_path)
        except Exception:
            existing_state = None
            logger.exception(
                "Failed to load existing analysis GUI state for %s while preserving "
                "the selected tab.",
                store_path,
            )
        if existing_state is not None and isinstance(
            existing_state.get("workflow"), Mapping
        ):
            analysis_selected_tab = str(
                existing_state["workflow"].get("analysis_selected_tab", "") or ""
            ).strip()
        try:
            persist_latest_analysis_gui_state(
                store_path,
                _analysis_gui_state_payload_from_workflow(
                    scoped_workflow,
                    analysis_selected_tab=analysis_selected_tab,
                ),
                source="analysis_dialog_success_reset",
            )
        except Exception:
            logger.exception(
                "Failed to persist reset analysis GUI state for %s.",
                store_path,
            )


def _analysis_target_run_label(target: Optional[AnalysisTarget]) -> str:
    """Return a compact progress label for one analysis target.

    Parameters
    ----------
    target : AnalysisTarget, optional
        Target to format.

    Returns
    -------
    str
        Short human-readable label for progress updates.
    """
    if target is None:
        return "selected experiment"
    experiment_path = Path(target.experiment_path)
    parent_name = str(experiment_path.parent.name).strip()
    if parent_name:
        return f"{parent_name}/{experiment_path.name}"
    return experiment_path.name or str(target.experiment_path)


def _workflows_for_selected_analysis_scope(
    workflow: WorkflowConfig,
) -> tuple[WorkflowConfig, ...]:
    """Expand one workflow into the selected single-target or batch scope.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow carrying analysis-scope state from the GUI.

    Returns
    -------
    tuple[WorkflowConfig, ...]
        One workflow for single-experiment runs, or one per target for batch
        analysis.
    """
    targets = _analysis_targets_for_workflow(workflow)

    def _target_spatial_calibration(target: AnalysisTarget) -> SpatialCalibrationConfig:
        if is_zarr_store_path(target.store_path):
            try:
                return load_store_spatial_calibration(target.store_path)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to load store spatial calibration for %s.",
                    target.store_path,
                )
        return workflow.spatial_calibration

    if not targets:
        return (workflow,)
    if workflow.analysis_apply_to_all and len(targets) > 1:
        return tuple(
            replace(
                workflow,
                file=str(target.store_path),
                analysis_selected_experiment_path=str(target.experiment_path),
                analysis_apply_to_all=False,
                spatial_calibration=_target_spatial_calibration(target),
                spatial_calibration_explicit=False,
            )
            for target in targets
        )
    selected_target = _resolve_selected_analysis_target(workflow, targets=targets)
    if selected_target is None:
        return (workflow,)
    if (
        str(workflow.file or "").strip() == str(selected_target.store_path)
        and str(workflow.analysis_selected_experiment_path or "").strip()
        == str(selected_target.experiment_path)
        and not workflow.analysis_apply_to_all
    ):
        return (
            replace(
                workflow,
                spatial_calibration=_target_spatial_calibration(selected_target),
                spatial_calibration_explicit=False,
            ),
        )
    return (
        replace(
            workflow,
            file=str(selected_target.store_path),
            analysis_selected_experiment_path=str(selected_target.experiment_path),
            analysis_apply_to_all=False,
            spatial_calibration=_target_spatial_calibration(selected_target),
            spatial_calibration_explicit=False,
        ),
    )


def run_workflow_with_progress(
    *,
    workflow: WorkflowConfig,
    run_callback: GuiRunCallback,
    dask_client_lifecycle_callback: Optional[DaskClientLifecycleCallback] = None,
) -> bool:
    """Execute a workflow while showing a modal GUI progress dialog.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration to execute.
    run_callback : callable
        Callback that performs workflow execution. Signature must be
        ``(workflow, progress_callback, dask_client_lifecycle_callback=None)``.
    dask_client_lifecycle_callback : callable, optional
        Optional lifecycle callback forwarded to the workflow run callback.

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
    logger = logging.getLogger(__name__)

    scoped_workflows = _workflows_for_selected_analysis_scope(workflow)
    execution_workflow = scoped_workflows[0]
    relay_lifecycle_callback: Optional[DaskClientLifecycleCallback] = None

    def _invoke_run_callback(
        workflow: WorkflowConfig,
        progress_callback: Callable[[int, str], None],
        dask_client_lifecycle_callback: Optional[DaskClientLifecycleCallback] = None,
    ) -> None:
        lifecycle_callback = (
            relay_lifecycle_callback
            if relay_lifecycle_callback is not None
            else dask_client_lifecycle_callback
        )
        if lifecycle_callback is None:
            run_callback(workflow, progress_callback)
        else:
            run_callback(workflow, progress_callback, lifecycle_callback)

    effective_run_callback = _invoke_run_callback
    if len(scoped_workflows) > 1:

        def _batch_run_callback(
            workflow: WorkflowConfig,
            progress_callback: Callable[[int, str], None],
            dask_client_lifecycle_callback: Optional[
                DaskClientLifecycleCallback
            ] = None,
        ) -> None:
            """Execute one run callback across multiple prepared workflows.

            Parameters
            ----------
            workflow : WorkflowConfig
                Ignored placeholder matching ``run_callback`` signature.
            progress_callback : callable
                Batch-aware GUI progress callback.

            Returns
            -------
            None
                Side-effect execution only.
            """
            del workflow
            callback = _invoke_run_callback

            total = len(scoped_workflows)
            for index, target_workflow in enumerate(scoped_workflows):
                target = _resolve_selected_analysis_target(target_workflow)
                target_label = _analysis_target_run_label(target)

                def _batch_progress(
                    percent: int,
                    message: str,
                    *,
                    batch_index: int = index,
                    batch_total: int = total,
                    label: str = target_label,
                ) -> None:
                    bounded_percent = max(0, min(100, int(percent)))
                    overall_fraction = (
                        float(batch_index) + (float(bounded_percent) / 100.0)
                    ) / float(batch_total)
                    overall_percent = max(
                        0,
                        min(100, int(round(overall_fraction * 100.0))),
                    )
                    progress_callback(
                        overall_percent,
                        f"[{batch_index + 1}/{batch_total}] {label}: {message}",
                    )

                progress_callback(
                    max(0, min(99, int(round((float(index) / float(total)) * 100.0)))),
                    f"Starting [{index + 1}/{total}] {target_label}",
                )
                callback(target_workflow, _batch_progress)
            progress_callback(
                100,
                f"Completed analysis for {len(scoped_workflows)} experiments.",
            )

        effective_run_callback = _batch_run_callback

    progress_dialog = AnalysisExecutionProgressDialog(parent=None)
    failure_payload: dict[str, str] = {}
    cancellation_payload: dict[str, str] = {}
    completed = {"ok": False}
    cancel_requested = {"value": False}
    progress_dashboard_setter = getattr(
        progress_dialog, "set_dashboard_available", None
    )
    progress_dashboard_bridge: Optional[QObject] = None

    if callable(progress_dashboard_setter) and isinstance(progress_dialog, QObject):

        class _ProgressDashboardBridge(QObject):
            dashboard_available_changed = pyqtSignal(bool, str)

        progress_dashboard_bridge = _ProgressDashboardBridge(progress_dialog)
        progress_dashboard_bridge.dashboard_available_changed.connect(
            progress_dialog.set_dashboard_available,
            Qt.ConnectionType.QueuedConnection,
        )

    def _request_cancel() -> None:
        """Record a user-requested cancellation event.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Cancellation state is updated in-place.
        """
        cancel_requested["value"] = True

    cancel_signal = getattr(progress_dialog, "cancel_requested", None)
    if cancel_signal is not None:
        cancel_signal.connect(_request_cancel)

    relay_manager = get_dashboard_relay_manager()
    active_analysis_client_ids: dict[int, str] = {}

    def _set_progress_dashboard_available() -> None:
        """Sync the progress dialog dashboard button to relay availability."""
        if not callable(progress_dashboard_setter):
            return
        if active_analysis_client_ids:
            available = True
            tooltip = "Open the local Dask dashboard relay for analysis clients."
        else:
            available = False
            tooltip = "No active analysis Dask client is registered."
        if progress_dashboard_bridge is not None:
            progress_dashboard_bridge.dashboard_available_changed.emit(
                available,
                tooltip,
            )
            return
        progress_dashboard_setter(available, tooltip)

    def _open_dashboard_from_relay() -> None:
        """Open the local relay URL for the active analysis dashboard."""
        try:
            dashboard_url = relay_manager.open_dashboard(workload="analysis")
        except Exception as exc:
            QMessageBox.warning(
                progress_dialog,
                "Dashboard Launch Failed",
                f"Could not open the dashboard relay.\n\n{exc}",
            )
            return

        try:
            opened = bool(webbrowser.open_new_tab(dashboard_url))
        except Exception as exc:
            QMessageBox.warning(
                progress_dialog,
                "Dashboard Launch Failed",
                f"Could not open dashboard URL.\n\n{exc}",
            )
            return

        if not opened:
            QMessageBox.warning(
                progress_dialog,
                "Dashboard Launch Failed",
                f"Browser did not confirm opening:\n{dashboard_url}",
            )

    def _relay_client_lifecycle_callback(
        event: str,
        workload: str,
        backend_mode: str,
        client: object,
    ) -> None:
        """Track analysis client lifecycle and forward user callbacks."""
        normalized_event = str(event).strip().lower()
        normalized_workload = str(workload).strip().lower()
        client_key = id(client)
        if normalized_workload == "analysis":
            if normalized_event == "started":
                try:
                    registered_client_id = relay_manager.register_client(
                        workload=workload,
                        backend_mode=backend_mode,
                        client=client,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to register analysis dashboard relay client: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    active_analysis_client_ids.pop(client_key, None)
                else:
                    active_analysis_client_ids[client_key] = registered_client_id
            elif normalized_event == "stopped":
                registered_client_id = active_analysis_client_ids.pop(client_key, None)
                if registered_client_id is not None:
                    try:
                        relay_manager.unregister_client(registered_client_id)
                    except Exception as exc:
                        logger.warning(
                            "Failed to unregister analysis dashboard relay client: %s: %s",
                            type(exc).__name__,
                            exc,
                        )
        _set_progress_dashboard_available()
        if dask_client_lifecycle_callback is not None:
            try:
                dask_client_lifecycle_callback(event, workload, backend_mode, client)
            except Exception as exc:
                logger.warning(
                    "Dask client lifecycle callback failed during %s "
                    "(workload=%s, backend=%s): %s: %s",
                    event,
                    normalized_workload,
                    backend_mode,
                    type(exc).__name__,
                    exc,
                )

    relay_lifecycle_callback = _relay_client_lifecycle_callback
    dashboard_requested_signal = getattr(progress_dialog, "dashboard_requested", None)
    if dashboard_requested_signal is not None:
        dashboard_requested_signal.connect(_open_dashboard_from_relay)
    _set_progress_dashboard_available()

    def _cleanup_dashboard_relay_clients() -> None:
        """Best-effort unregister any active analysis relay clients."""
        for client_id in list(active_analysis_client_ids.values()):
            try:
                relay_manager.unregister_client(client_id)
            except Exception as exc:
                logger.warning(
                    "Failed to unregister active analysis relay client during cleanup: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        active_analysis_client_ids.clear()
        _set_progress_dashboard_available()

    use_main_thread_execution = execution_workflow.dask_backend.mode in {
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
            if cancel_requested["value"]:
                raise WorkflowExecutionCancelled("Analysis cancelled by user.")
            progress_dialog.update_progress(int(percent), str(message))
            app.processEvents()
            if cancel_requested["value"]:
                raise WorkflowExecutionCancelled("Analysis cancelled by user.")

        try:
            if len(scoped_workflows) > 1:
                _progress_with_events(
                    1,
                    f"Starting batch analysis workflow for {len(scoped_workflows)} experiments...",
                )
            else:
                _progress_with_events(1, "Starting analysis workflow...")
            effective_run_callback(
                execution_workflow,
                _progress_with_events,
                relay_lifecycle_callback,
            )
            if len(scoped_workflows) == 1:
                _progress_with_events(100, "Analysis workflow completed.")
            completed["ok"] = True
            progress_dialog.accept()
        except WorkflowExecutionCancelled as exc:
            cancellation_payload.update({"summary": str(exc) or "Analysis cancelled."})
            progress_dialog.reject()
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
            workflow=execution_workflow,
            run_callback=effective_run_callback,
            dask_client_lifecycle_callback=relay_lifecycle_callback,
        )
        worker.progress_changed.connect(progress_dialog.update_progress)
        worker.succeeded.connect(lambda: completed.__setitem__("ok", True))
        worker.succeeded.connect(progress_dialog.accept)
        if cancel_signal is not None:
            cancel_signal.connect(worker.cancel)
        worker.cancelled.connect(
            lambda summary: cancellation_payload.update({"summary": str(summary)})
        )
        worker.cancelled.connect(lambda *_: progress_dialog.reject())
        worker.failed.connect(
            lambda summary, details: failure_payload.update(
                {"summary": str(summary), "details": str(details)}
            )
        )
        worker.failed.connect(lambda *_: progress_dialog.reject())

        worker.start()
        progress_dialog.exec()
        worker.wait()

    _cleanup_dashboard_relay_clients()
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
            _shutdown_dashboard_relay_manager()
        return False
    if cancellation_payload:
        if owns_app:
            app.quit()
            _shutdown_dashboard_relay_manager()
        return False

    if owns_app:
        app.quit()
        _shutdown_dashboard_relay_manager()
    return bool(completed["ok"])
