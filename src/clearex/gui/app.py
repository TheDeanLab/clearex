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
from typing import Any, Dict, Optional

# Local Imports
from clearex.io.experiment import (
    NavigateExperiment,
    default_analysis_store_path,
    is_navigate_experiment_file,
    load_navigate_experiment,
    resolve_experiment_data_path,
)
from clearex.io.read import ImageInfo, ImageOpener
from clearex.workflow import WorkflowConfig, format_chunks, parse_chunks

# Third Party Imports
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
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


if HAS_PYQT6:

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
            self._chunks_input.setText(format_chunks(initial.chunks))
            self._deconvolution_checkbox.setChecked(initial.deconvolution)
            self._particle_checkbox.setChecked(initial.particle_detection)
            self._registration_checkbox.setChecked(initial.registration)
            self._visualization_checkbox.setChecked(initial.visualization)

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
                chunks=chunks,
                deconvolution=self._deconvolution_checkbox.isChecked(),
                particle_detection=self._particle_checkbox.isChecked(),
                registration=self._registration_checkbox.isChecked(),
                visualization=self._visualization_checkbox.isChecked(),
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
