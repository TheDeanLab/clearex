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
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Local Imports
from clearex.io.experiment import (
    NavigateExperiment,
    default_analysis_store_path,
    is_navigate_experiment_file,
    load_navigate_experiment,
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
    format_dask_backend_summary,
    format_chunks,
    format_pyramid_levels,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    parse_chunks,
    parse_pyramid_levels,
)

# Third Party Imports
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
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
            root.setSpacing(12)

            description = QLabel(
                "Configure Zarr save chunk sizes and downsampling pyramid in "
                "(p, t, c, z, y, x) axis order."
            )
            description.setWordWrap(True)
            root.addWidget(description)

            chunk_group = QGroupBox("Chunk Size")
            chunk_layout = QGridLayout(chunk_group)
            chunk_layout.setHorizontalSpacing(16)
            chunk_layout.setVerticalSpacing(8)

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
            pyramid_layout.setHorizontalSpacing(16)
            pyramid_layout.setVerticalSpacing(8)

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
            root.setSpacing(12)

            overview = QLabel(
                "Choose how ClearEx orchestrates Dask workers for loading and analysis. "
                "Mode guidance is shown below."
            )
            overview.setWordWrap(True)
            root.addWidget(overview)

            mode_row = QHBoxLayout()
            mode_label = QLabel("Backend mode:")
            self._mode_combo = QComboBox()
            mode_specs = (
                (DASK_BACKEND_LOCAL_CLUSTER, DASK_BACKEND_MODE_LABELS[DASK_BACKEND_LOCAL_CLUSTER]),
                (DASK_BACKEND_SLURM_RUNNER, DASK_BACKEND_MODE_LABELS[DASK_BACKEND_SLURM_RUNNER]),
                (DASK_BACKEND_SLURM_CLUSTER, DASK_BACKEND_MODE_LABELS[DASK_BACKEND_SLURM_CLUSTER]),
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
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

            self._runner_scheduler_file_input = QLineEdit()
            self._runner_scheduler_file_input.setPlaceholderText("Path to scheduler file")
            self._runner_scheduler_file_browse = QPushButton("Browse")
            self._runner_scheduler_file_browse.clicked.connect(
                self._on_browse_scheduler_file
            )
            scheduler_row = QHBoxLayout()
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
            root.setSpacing(10)

            worker_group = QGroupBox("Worker Job Settings")
            worker_form = QFormLayout(worker_group)
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
            self._cluster_local_directory_input.setText(cluster_cfg.local_directory or "")
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

    class ClearExDialog(QDialog):
        """GUI dialog for workflow selection and metadata preview.

        Parameters
        ----------
        initial : WorkflowConfig
            Initial configuration used to pre-populate the form controls.

        Attributes
        ----------
        result_config : WorkflowConfig, optional
            Final workflow configuration selected by the user.
        """

        def __init__(self, initial: WorkflowConfig) -> None:
            """Initialize the dialog and load initial workflow values.

            Parameters
            ----------
            initial : WorkflowConfig
                Initial workflow settings used to hydrate the UI controls.

            Returns
            -------
            None
                The dialog is initialized in-place.
            """
            super().__init__()
            self.setWindowTitle("ClearEx")
            self.setMinimumWidth(960)
            self.setMinimumHeight(720)

            self._opener = ImageOpener()
            self.result_config: Optional[WorkflowConfig] = None
            self._metadata_labels: Dict[str, QLabel] = {}
            self._dask_backend_config: DaskBackendConfig = DaskBackendConfig()
            self._zarr_save_config: ZarrSaveConfig = ZarrSaveConfig()

            self._build_ui()
            self._apply_theme()
            self._hydrate(initial)

        def _build_ui(self) -> None:
            """Build and wire all dialog widgets.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Widgets are created and connected in-place.
            """
            root = QVBoxLayout(self)
            root.setSpacing(14)

            header = QFrame()
            header.setObjectName("headerCard")
            header_layout = QVBoxLayout(header)
            title = QLabel("ClearEx")
            title.setObjectName("title")
            subtitle = QLabel("Scalable Image Analysis")
            subtitle.setObjectName("subtitle")
            header_layout.addWidget(title)
            header_layout.addWidget(subtitle)
            root.addWidget(header)

            data_group = QGroupBox("Data Source")
            data_layout = QVBoxLayout(data_group)

            path_row = QHBoxLayout()
            self._path_input = QLineEdit()
            self._path_input.setPlaceholderText(
                "Select a data file, Navigate experiment.yml, or directory "
                "(.yml/.yaml/.tif/.tiff/.zarr/.n5/.npy/.npz/.h5/.nd2)"
            )
            self._browse_file_button = QPushButton("Browse File")
            self._browse_directory_button = QPushButton("Browse Folder")
            self._load_button = QPushButton("Load Metadata")
            path_row.addWidget(self._path_input, 1)
            path_row.addWidget(self._browse_file_button)
            path_row.addWidget(self._browse_directory_button)
            path_row.addWidget(self._load_button)
            data_layout.addLayout(path_row)

            options_row = QHBoxLayout()
            self._dask_checkbox = QCheckBox("Use Dask / lazy loading")
            self._chunks_input = QLineEdit()
            self._chunks_input.setPlaceholderText(
                "Chunks (optional): e.g. 256 or 1,256,256"
            )
            options_row.addWidget(self._dask_checkbox)
            options_row.addWidget(self._chunks_input, 1)
            data_layout.addLayout(options_row)

            dask_backend_row = QHBoxLayout()
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
            metadata_layout.setHorizontalSpacing(18)
            metadata_layout.setVerticalSpacing(8)

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

            analysis_group = QGroupBox("Analysis Selection")
            analysis_layout = QGridLayout(analysis_group)
            analysis_layout.setSpacing(10)

            self._deconvolution_checkbox = QCheckBox("Deconvolution")
            self._particle_checkbox = QCheckBox("Particle Detection")
            self._registration_checkbox = QCheckBox("Registration")
            self._visualization_checkbox = QCheckBox("Visualization")

            analysis_layout.addWidget(self._deconvolution_checkbox, 0, 0)
            analysis_layout.addWidget(self._particle_checkbox, 0, 1)
            analysis_layout.addWidget(self._registration_checkbox, 1, 0)
            analysis_layout.addWidget(self._visualization_checkbox, 1, 1)
            root.addWidget(analysis_group)

            footer = QHBoxLayout()
            self._status_label = QLabel("Ready")
            self._status_label.setObjectName("statusLabel")
            self._cancel_button = QPushButton("Cancel")
            self._run_button = QPushButton("Run")
            self._run_button.setObjectName("runButton")
            footer.addWidget(self._status_label, 1)
            footer.addWidget(self._cancel_button)
            footer.addWidget(self._run_button)
            root.addLayout(footer)

            self._browse_file_button.clicked.connect(self._on_browse_file)
            self._browse_directory_button.clicked.connect(self._on_browse_directory)
            self._load_button.clicked.connect(self._on_load_metadata)
            self._dask_backend_button.clicked.connect(self._on_edit_dask_backend)
            self._zarr_config_button.clicked.connect(self._on_edit_zarr_settings)
            self._cancel_button.clicked.connect(self.reject)
            self._run_button.clicked.connect(self._on_run)

        def _hydrate(self, initial: WorkflowConfig) -> None:
            """Populate UI widgets from a workflow configuration.

            Parameters
            ----------
            initial : WorkflowConfig
                Initial values for path, chunking, and selected analyses.

            Returns
            -------
            None
                Widget state is updated in-place.
            """
            self._path_input.setText(initial.file or "")
            self._dask_checkbox.setChecked(initial.prefer_dask)
            self._dask_backend_config = initial.dask_backend
            self._chunks_input.setText(format_chunks(initial.chunks))
            self._deconvolution_checkbox.setChecked(initial.deconvolution)
            self._particle_checkbox.setChecked(initial.particle_detection)
            self._registration_checkbox.setChecked(initial.registration)
            self._visualization_checkbox.setChecked(initial.visualization)
            self._refresh_dask_backend_summary()
            self._zarr_save_config = initial.zarr_save
            self._refresh_zarr_save_summary()

        def _refresh_dask_backend_summary(self) -> None:
            """Refresh the main dialog summary for Dask backend settings.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary label is updated in-place.
            """
            summary = format_dask_backend_summary(self._dask_backend_config)
            self._dask_backend_summary.setText(summary)
            self._dask_backend_summary.setToolTip(summary)

        def _on_edit_dask_backend(self) -> None:
            """Launch the Dask backend settings popup and apply its result.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Updates stored backend config when dialog is accepted.
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

        def _refresh_zarr_save_summary(self) -> None:
            """Refresh the main dialog summary for Zarr save settings.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Summary label is updated in-place.
            """
            summary = _format_zarr_save_summary(self._zarr_save_config)
            self._zarr_config_summary.setText(summary)
            self._zarr_config_summary.setToolTip(summary)

        def _on_edit_zarr_settings(self) -> None:
            """Launch the Zarr save settings popup and apply its result.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Updates stored Zarr save config when dialog is accepted.
            """
            dialog = ZarrSaveConfigDialog(initial=self._zarr_save_config, parent=self)
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.result_config is None:
                return
            self._zarr_save_config = dialog.result_config
            self._refresh_zarr_save_summary()
            self._set_status("Updated Zarr save settings.")

        def _apply_theme(self) -> None:
            """Apply stylesheet-based dark theme for the dialog.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Styles are applied in-place to the dialog.
            """
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #0c1118;
                    color: #e6edf3;
                    font-family: "Segoe UI", "Avenir Next", sans-serif;
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
            """Update the footer status label.

            Parameters
            ----------
            text : str
                Status message displayed in the dialog footer.

            Returns
            -------
            None
                Label text is updated in-place.
            """
            self._status_label.setText(text)

        def _on_browse_file(self) -> None:
            """Open file picker and update the selected data path.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Updates the path field when a file is selected.
            """
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image Data",
                str(Path.cwd()),
                "All Files (*);;Image Data (*.yml *.yaml *.tif *.tiff *.zarr *.n5 *.npy *.npz *.h5 *.hdf5 *.hdf *.nd2)",
            )
            if file_path:
                self._path_input.setText(file_path)

        def _on_browse_directory(self) -> None:
            """Open directory picker and update the selected data path.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Updates the path field when a directory is selected.
            """
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Data Directory",
                str(Path.cwd()),
            )
            if directory:
                self._path_input.setText(directory)

        def _on_load_metadata(self) -> None:
            """Load metadata for the currently selected dataset path.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Metadata labels are refreshed in-place.

            Raises
            ------
            None
                Exceptions during loading are caught and presented as GUI
                error messages.
            """
            path = self._path_input.text().strip()
            if not path:
                QMessageBox.warning(self, "Missing Path", "Select a data path first.")
                return
            if not Path(path).exists():
                QMessageBox.warning(
                    self, "Missing Path", f"Path does not exist:\n{path}"
                )
                return

            try:
                chunks = parse_chunks(self._chunks_input.text())
                selected_path = Path(path)
                source_data_path = selected_path
                experiment = None
                if is_navigate_experiment_file(selected_path):
                    experiment = load_navigate_experiment(selected_path)
                    source_data_path = resolve_experiment_data_path(experiment)
                _, info = self._opener.open(
                    path=str(source_data_path),
                    prefer_dask=self._dask_checkbox.isChecked(),
                    chunks=chunks,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Metadata Load Failed",
                    f"Failed to load data metadata.\n\n{exc}",
                )
                self._set_status("Failed to load metadata.")
                return

            summary = summarize_image_info(info)
            if experiment is not None:
                summary = _apply_experiment_overrides(
                    summary=summary,
                    experiment_path=selected_path,
                    resolved_data_path=source_data_path,
                    experiment=experiment,
                )

            for key, value in summary.items():
                self._metadata_labels[key].setText(value)
            if experiment is not None:
                target_store = default_analysis_store_path(experiment)
                self._set_status(
                    "Metadata loaded from experiment.yml. "
                    f"Analysis store target: {target_store}"
                )
            else:
                self._set_status("Metadata loaded.")

        def _on_run(self) -> None:
            """Validate form values and finalize workflow selection.

            Parameters
            ----------
            None

            Returns
            -------
            None
                Stores a :class:`WorkflowConfig` and accepts the dialog when
                validation succeeds.

            Raises
            ------
            None
                Validation errors are reported via message boxes and handled
                without raising exceptions.
            """
            path = self._path_input.text().strip()
            if not path:
                QMessageBox.warning(
                    self, "Missing Path", "Select a data path before running."
                )
                return
            if not Path(path).exists():
                QMessageBox.warning(
                    self, "Missing Path", f"Path does not exist:\n{path}"
                )
                return

            try:
                chunks = parse_chunks(self._chunks_input.text())
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Chunks", str(exc))
                return

            self.result_config = WorkflowConfig(
                file=path,
                prefer_dask=self._dask_checkbox.isChecked(),
                dask_backend=self._dask_backend_config,
                chunks=chunks,
                deconvolution=self._deconvolution_checkbox.isChecked(),
                particle_detection=self._particle_checkbox.isChecked(),
                registration=self._registration_checkbox.isChecked(),
                visualization=self._visualization_checkbox.isChecked(),
                zarr_save=self._zarr_save_config,
            )
            self.accept()


def launch_gui(initial: Optional[WorkflowConfig] = None) -> Optional[WorkflowConfig]:
    """Launch the PyQt dialog and return selected workflow configuration.

    Parameters
    ----------
    initial : WorkflowConfig, optional
        Initial workflow values to pre-populate the GUI controls.

    Returns
    -------
    WorkflowConfig, optional
        User-selected workflow configuration. Returns ``None`` when the dialog
        is cancelled.

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

    dialog = ClearExDialog(initial=initial or WorkflowConfig())
    result = dialog.exec()
    selected = dialog.result_config if result == QDialog.DialogCode.Accepted else None

    if owns_app:
        app.quit()
    return selected
