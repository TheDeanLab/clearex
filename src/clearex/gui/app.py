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
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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
    NavigateExperiment,
    create_dask_client,
    is_navigate_experiment_file,
    load_navigate_experiment,
    materialize_experiment_data_store,
    resolve_data_store_path,
    resolve_experiment_data_path,
)
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
    LocalClusterConfig,
    PTCZYX_AXES,
    SlurmClusterConfig,
    SlurmRunnerConfig,
    WorkflowConfig,
    ZarrSaveConfig,
    default_analysis_operation_parameters,
    format_dask_backend_summary,
    format_pyramid_levels,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    normalize_analysis_operation_parameters,
    parse_pyramid_levels,
)

# Third Party Imports
import zarr

try:
    from PyQt6.QtCore import QEvent, QMimeData, QObject, QThread, Qt, pyqtSignal
    from PyQt6.QtGui import (
        QDragEnterEvent,
        QDragMoveEvent,
        QDropEvent,
        QImage,
        QPixmap,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
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
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class GuiUnavailableError(RuntimeError):
    """Raised when the PyQt GUI cannot be launched in this environment."""


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

    return {
        "path": str(info.path),
        "shape": _format_optional_value(info.shape),
        "dtype": _format_optional_value(info.dtype),
        "axes": _format_optional_value(info.axes),
        "channels": _format_optional_value(channels),
        "positions": _format_optional_value(positions),
        "image_size": image_size,
        "time_points": _format_optional_value(time_points),
        "pixel_size": _format_optional_value(getattr(info, "pixel_size", None)),
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
            margin-top: 14px;
            padding: 12px;
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
            padding: 6px 8px;
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
        QLineEdit::placeholder, QPlainTextEdit {
            color: #a6b7d0;
        }
        QPushButton {
            background-color: #1a2635;
            border: 1px solid #2f4460;
            border-radius: 8px;
            padding: 8px 12px;
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
            self.setMinimumWidth(760)
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
            self.setMinimumWidth(820)
            self.setMinimumHeight(760)
            self.result_config: Optional[DaskBackendConfig] = None
            self._mode_index: Dict[str, int] = {}

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
        failed = pyqtSignal(str)

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
                self.failed.emit(str(exc))

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
        failed = pyqtSignal(str)

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
                self.failed.emit(str(exc))

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
            self.setMinimumWidth(960)
            self.setMinimumHeight(700)
            self.setAcceptDrops(True)

            self._opener = ImageOpener()
            self.result_config: Optional[WorkflowConfig] = None
            self._metadata_labels: Dict[str, QLabel] = {}
            self._dask_backend_config: DaskBackendConfig = initial.dask_backend
            self._zarr_save_config: ZarrSaveConfig = initial.zarr_save
            self._chunks = initial.chunks
            self._loaded_experiment: Optional[NavigateExperiment] = None
            self._loaded_experiment_path: Optional[Path] = None
            self._loaded_source_data_path: Optional[Path] = None
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
            title = QLabel("ClearEx")
            title.setObjectName("title")
            subtitle = QLabel("Experiment Setup")
            subtitle.setObjectName("subtitle")
            header_layout.addWidget(title)
            header_layout.addWidget(subtitle)
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
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #121a28, stop:1 #162538
                    );
                    border: 1px solid #2a3442;
                    border-radius: 12px;
                    padding: 10px;
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
                    margin-top: 14px;
                    padding: 12px;
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
                    padding: 8px;
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
                    padding: 8px 12px;
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
            self._loaded_experiment = None
            self._loaded_experiment_path = None
            self._loaded_source_data_path = None
            self._set_status(
                "Experiment path set from drag-and-drop. Click Load Metadata or Next."
            )
            return True

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
            source_data_path = resolve_experiment_data_path(experiment)
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
                QMessageBox.critical(
                    self,
                    "Metadata Load Failed",
                    f"Failed to load experiment metadata.\n\n{exc}",
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
            self._loaded_source_data_path = source_data_path

            target_store = resolve_data_store_path(experiment, source_data_path)
            self._set_status(f"Metadata loaded. Target store: {target_store}")

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
                deconvolution=False,
                particle_detection=False,
                registration=False,
                visualization=False,
                zarr_save=self._zarr_save_config,
            )
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

            if target_store.exists():
                self._set_status(
                    "Found existing data store. Opening analysis selection."
                )
                self._accept_with_store_path(target_store)
                return

            progress_dialog = MaterializationProgressDialog(parent=self)
            failure_messages: list[str] = []
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
            worker.failed.connect(lambda text: failure_messages.append(text))
            worker.failed.connect(lambda _: progress_dialog.reject())

            worker.start()
            progress_dialog.exec()
            worker.wait()
            self._materialization_worker = None

            if failure_messages:
                QMessageBox.critical(
                    self,
                    "Store Creation Failed",
                    f"Failed to create canonical data store.\n\n{failure_messages[0]}",
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
            "deconvolution",
            "particle_detection",
            "registration",
            "visualization",
        )
        _OPERATION_LABELS: Dict[str, str] = {
            "deconvolution": "Deconvolution",
            "particle_detection": "Particle Detection",
            "registration": "Registration",
            "visualization": "Visualization",
        }
        _OPERATION_OUTPUT_COMPONENTS: Dict[str, str] = {
            "deconvolution": "results/deconvolution/latest/data",
            "registration": "results/registration/latest/data",
        }
        _PARAMETER_HINTS: Dict[str, str] = {
            "input_source": (
                "Input source controls which dataset this operation reads from. "
                "Use raw data, or choose an upstream operation output."
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
                "results/particle_detection/latest/detections is available."
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
            self.setMinimumWidth(1120)
            self.setMinimumHeight(640)

            self._base_config = initial
            self.result_config: Optional[WorkflowConfig] = None
            self._status_label: Optional[QLabel] = None
            self._parameter_help_default = (
                "Hover over a parameter to see a detailed explanation."
            )

            self._operation_defaults = default_analysis_operation_parameters()
            self._decon_defaults = dict(
                self._operation_defaults.get("deconvolution", {})
            )
            self._particle_defaults = dict(
                self._operation_defaults.get("particle_detection", {})
            )
            self._visualization_defaults = dict(
                self._operation_defaults.get("visualization", {})
            )

            self._operation_checkboxes: Dict[str, QCheckBox] = {}
            self._operation_order_spins: Dict[str, QSpinBox] = {}
            self._operation_config_buttons: Dict[str, QPushButton] = {}
            self._operation_input_combos: Dict[str, QComboBox] = {}
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
            title = QLabel("Select Analysis Methods")
            title.setObjectName("title")
            subtitle = QLabel(
                "Enable routines, configure each operation, and define execution sequence."
            )
            subtitle.setObjectName("subtitle")
            header_layout.addWidget(title)
            header_layout.addWidget(subtitle)
            root.addWidget(header)

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
            root.addLayout(store_row)

            content_row = QHBoxLayout()
            apply_stack_spacing(content_row)
            root.addLayout(content_row, 1)

            analysis_group = QGroupBox("Analysis Selection")
            analysis_group.setMinimumWidth(430)
            analysis_layout = QVBoxLayout(analysis_group)
            apply_stack_spacing(analysis_layout)

            for operation_name in self._OPERATION_KEYS:
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
                order_spin.setMinimumWidth(64)
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

                analysis_layout.addLayout(row)

                checkbox.toggled.connect(self._on_operation_selection_changed)
                order_spin.valueChanged.connect(self._on_operation_order_changed)
                configure_button.clicked.connect(
                    lambda _checked=False,
                    op=operation_name: self._show_operation_configuration(op)
                )

            analysis_hint = QLabel(
                "Use Configure to edit one operation at a time. "
                "Only selected operations can be configured."
            )
            analysis_hint.setObjectName("statusLabel")
            analysis_hint.setWordWrap(True)
            analysis_layout.addWidget(analysis_hint)
            analysis_layout.addStretch(1)
            content_row.addWidget(analysis_group, 1)

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
            self._status_label = QLabel(
                "Configure selected analysis routines and click Run."
            )
            self._status_label.setObjectName("statusLabel")
            self._status_label.setWordWrap(True)
            footer.addWidget(self._status_label, 1)
            self._cancel_button = QPushButton("Cancel")
            self._run_button = QPushButton("Run")
            self._run_button.setObjectName("runButton")
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._run_button)
            root.addLayout(footer)

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
            self._particle_use_overlap_checkbox.toggled.connect(
                self._set_particle_parameter_enabled_state
            )
            self._particle_remove_close_checkbox.toggled.connect(
                self._set_particle_parameter_enabled_state
            )
            self._visualization_show_all_positions_checkbox.toggled.connect(
                self._set_visualization_position_selector_state
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
            elif operation_name == "particle_detection":
                self._build_particle_parameter_rows(form)
            elif operation_name == "visualization":
                self._build_visualization_parameter_rows(form)
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

            self._visualization_overlay_points_checkbox = QCheckBox(
                "Overlay particle detections"
            )
            form.addRow("Particle overlay", self._visualization_overlay_points_checkbox)
            self._register_parameter_hint(
                self._visualization_overlay_points_checkbox,
                self._PARAMETER_HINTS["overlay_particle_detections"],
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
                return "results/particle_detection/latest/detections"
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
        ) -> list[tuple[str, str]]:
            """Build input-source options for a specific operation.

            Parameters
            ----------
            operation_name : str
                Operation key for which options are generated.

            Returns
            -------
            list[tuple[str, str]]
                List of ``(value, label)`` input-source options.
            """
            options: list[tuple[str, str]] = [("data", "Raw data (data)")]
            selected_order = self._selected_operations_in_sequence()
            for upstream_name in selected_order:
                if upstream_name == operation_name:
                    break
                if upstream_name not in self._OPERATION_OUTPUT_COMPONENTS:
                    continue
                options.append(
                    (
                        upstream_name,
                        f"{self._OPERATION_LABELS[upstream_name]} output "
                        f"({self._operation_output_component(upstream_name)})",
                    )
                )
            return options

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
            for operation_name, combo in self._operation_input_combos.items():
                current_data = combo.currentData()
                current_source = (
                    str(current_data).strip() if current_data is not None else "data"
                ) or "data"
                options = self._input_options_for_operation(operation_name)
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

            self._refresh_input_source_options()
            self._set_deconvolution_parameter_enabled_state()
            self._set_particle_parameter_enabled_state()
            self._set_visualization_parameter_enabled_state()

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
            self._visualization_overlay_points_checkbox.setEnabled(
                visualization_enabled
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
            decon_params = dict(
                normalized_parameters.get("deconvolution", self._decon_defaults)
            )
            particle_params = dict(
                normalized_parameters.get("particle_detection", self._particle_defaults)
            )
            visualization_params = dict(
                normalized_parameters.get("visualization", self._visualization_defaults)
            )

            if self._store_label is not None:
                self._store_label.setText(initial.file or "n/a")

            checkbox_defaults = {
                "deconvolution": bool(initial.deconvolution),
                "particle_detection": bool(initial.particle_detection),
                "registration": bool(initial.registration),
                "visualization": bool(initial.visualization),
            }
            if not any(checkbox_defaults.values()):
                checkbox_defaults["particle_detection"] = True

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
            store_xy_um: Optional[float] = None
            store_z_um: Optional[float] = None
            if initial.file and str(initial.file).lower().endswith((".zarr", ".n5")):
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
                except Exception:
                    position_count = 1
                    channel_count = 1
                    store_xy_um = None
                    store_z_um = None
            self._visualization_position_spin.setMaximum(position_count - 1)
            self._particle_channel_spin.setMaximum(channel_count - 1)

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
            self._visualization_overlay_points_checkbox.setChecked(
                bool(visualization_params.get("overlay_particle_detections", True))
            )
            self._set_visualization_position_selector_state()

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
                #headerCard {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #121a28, stop:1 #162538
                    );
                    border: 1px solid #2a3442;
                    border-radius: 12px;
                    padding: 10px;
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
                QGroupBox {
                    border: 1px solid #2a3442;
                    border-radius: 10px;
                    margin-top: 14px;
                    padding: 12px;
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
                    min-height: 28px;
                    border: 1px solid #2b3f58;
                    border-radius: 8px;
                    background-color: #101a29;
                    color: #e6edf3;
                    padding: 4px 8px;
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
                QToolTip {
                    color: #e6edf3;
                    background-color: #0f1b2a;
                    border: 1px solid #2a3442;
                }
                QPushButton {
                    background-color: #1a2635;
                    border: 1px solid #2f4460;
                    border-radius: 8px;
                    padding: 8px 12px;
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
            return {
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
                "overlay_particle_detections": bool(
                    self._visualization_overlay_points_checkbox.isChecked()
                ),
                "particle_detection_component": str(
                    self._visualization_defaults.get(
                        "particle_detection_component",
                        "results/particle_detection/latest/detections",
                    )
                ),
                "launch_mode": str(
                    self._visualization_defaults.get("launch_mode", "auto")
                ),
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
            if operation_name == "deconvolution":
                defaults.update(self._collect_deconvolution_parameters())
            elif operation_name == "particle_detection":
                defaults.update(self._collect_particle_parameters())
            elif operation_name == "visualization":
                defaults.update(self._collect_visualization_parameters())
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

            self.result_config = WorkflowConfig(
                file=self._base_config.file,
                prefer_dask=self._base_config.prefer_dask,
                dask_backend=self._base_config.dask_backend,
                chunks=self._base_config.chunks,
                deconvolution=selected_flags["deconvolution"],
                particle_detection=selected_flags["particle_detection"],
                registration=selected_flags["registration"],
                visualization=selected_flags["visualization"],
                zarr_save=self._base_config.zarr_save,
                analysis_parameters=analysis_parameters,
            )
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

    setup_dialog = ClearExSetupDialog(initial=initial or WorkflowConfig())
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

    if owns_app:
        app.quit()
    return selected


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

    progress_dialog = AnalysisExecutionProgressDialog(parent=None)
    failure_messages: list[str] = []
    completed = {"ok": False}

    worker = AnalysisExecutionWorker(
        workflow=workflow,
        run_callback=run_callback,
    )
    worker.progress_changed.connect(progress_dialog.update_progress)
    worker.succeeded.connect(lambda: completed.__setitem__("ok", True))
    worker.succeeded.connect(progress_dialog.accept)
    worker.failed.connect(lambda text: failure_messages.append(str(text)))
    worker.failed.connect(progress_dialog.reject)

    worker.start()
    progress_dialog.exec()
    worker.wait()

    if failure_messages:
        QMessageBox.critical(
            progress_dialog,
            "Analysis Failed",
            f"Analysis execution failed.\n\n{failure_messages[0]}",
        )
        if owns_app:
            app.quit()
        return False

    if owns_app:
        app.quit()
    return bool(completed["ok"])
