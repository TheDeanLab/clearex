#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import clearex.gui.app as app_module
from clearex.io.experiment import NavigateChannel, NavigateExperiment
import pytest
import zarr


def _make_navigate_experiment(path: Path) -> NavigateExperiment:
    """Create a minimal NavigateExperiment instance for GUI tests."""

    return NavigateExperiment(
        path=Path(path).expanduser().resolve(),
        raw={},
        save_directory=Path(path).expanduser().resolve().parent,
        file_type="OME-TIFF",
        microscope_name="scope",
        image_mode="z-stack",
        timepoints=2,
        number_z_steps=10,
        y_pixels=32,
        x_pixels=64,
        multiposition_count=1,
        selected_channels=[
            NavigateChannel(
                name="ch0",
                laser=None,
                laser_index=None,
                exposure_ms=None,
                is_selected=True,
            )
        ],
        xy_pixel_size_um=0.12,
        z_step_um=0.45,
    )


def _create_gui_analysis_store(tmp_path: Path) -> Path:
    """Create a minimal analysis store for GUI parameter-selection tests."""
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data_shape = (1, 1, 2, 1, 4, 4)
    data_chunks = (1, 1, 2, 1, 4, 4)
    root.create_dataset(
        name="clearex/runtime_cache/source/data",
        shape=data_shape,
        chunks=data_chunks,
        dtype="uint16",
        overwrite=True,
    )
    root.require_group("clearex").require_group("metadata").attrs[
        "voxel_size_um_zyx"
    ] = [0.45, 0.12, 0.12]
    data_pyramid = root.require_group("clearex/runtime_cache/source/data_pyramid")
    data_pyramid.create_dataset(
        name="level_1",
        shape=(1, 1, 2, 1, 2, 2),
        chunks=(1, 1, 2, 1, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root["clearex/runtime_cache/source/data"].attrs["pyramid_levels"] = [
        "clearex/runtime_cache/source/data",
        "clearex/runtime_cache/source/data_pyramid/level_1",
    ]
    root["clearex/runtime_cache/source/data"].attrs["pyramid_factors_tpczyx"] = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2],
    ]
    root["clearex/runtime_cache/source/data_pyramid/level_1"].attrs[
        "source_component"
    ] = "clearex/runtime_cache/source/data"
    shear_latest = (
        root.require_group("clearex")
        .require_group("runtime_cache")
        .require_group("results")
        .require_group("shear_transform")
        .require_group("latest")
    )
    shear_latest.create_dataset(
        name="data",
        shape=(1, 1, 1, 1, 4, 4),
        chunks=(1, 1, 1, 1, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    return store_path


def _install_fake_gui_runtime(monkeypatch):
    """Install fake Qt primitives for deterministic run-loop tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    tuple[type, type]
        Fake ``QApplication`` and progress-dialog classes.
    """

    class _FakeSignal:
        def __init__(self) -> None:
            self._callbacks: list[object] = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self, *args, **kwargs) -> None:
            for callback in list(self._callbacks):
                callback(*args, **kwargs)

    class _FakeApplication:
        _instance = None

        def __init__(self, _argv) -> None:
            type(self)._instance = self
            self.process_events_calls = 0
            self.quit_calls = 0

        @classmethod
        def instance(cls):
            return cls._instance

        def processEvents(self) -> None:
            self.process_events_calls += 1

        def quit(self) -> None:
            self.quit_calls += 1

    class _FakeProgressDialog:
        last_instance = None

        def __init__(self, parent=None) -> None:
            del parent
            type(self).last_instance = self
            self.updates: list[tuple[int, str]] = []
            self.dashboard_enabled_states: list[tuple[bool, str]] = []
            self.shown = False
            self.accepted = False
            self.rejected = False
            self.closed = False
            self.cancel_requested = _FakeSignal()
            self.dashboard_requested = _FakeSignal()

        def update_progress(self, percent: int, message: str) -> None:
            self.updates.append((int(percent), str(message)))

        def set_dashboard_available(
            self, available: bool, tooltip: str | None = None
        ) -> None:
            self.dashboard_enabled_states.append((bool(available), str(tooltip or "")))

        def exec(self) -> int:
            return 0

        def show(self) -> None:
            self.shown = True

        def accept(self) -> None:
            self.accepted = True

        def reject(self) -> None:
            self.rejected = True

        def close(self) -> None:
            self.closed = True

    class _UnexpectedWorker:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise AssertionError(
                "AnalysisExecutionWorker must not be used in SLURM main-thread path."
            )

    monkeypatch.setattr(app_module, "HAS_PYQT6", True)
    monkeypatch.setattr(app_module, "_display_is_available", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_apply_application_icon",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(app_module, "QApplication", _FakeApplication, raising=False)
    monkeypatch.setattr(
        app_module,
        "AnalysisExecutionProgressDialog",
        _FakeProgressDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AnalysisExecutionWorker",
        _UnexpectedWorker,
        raising=False,
    )
    return _FakeApplication, _FakeProgressDialog


def test_progress_dialog_creates_disabled_dashboard_button() -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    dialog = app_module.AnalysisExecutionProgressDialog(parent=None)

    assert dialog._dashboard_button is not None
    assert dialog._dashboard_button.text() == "Open Dask Dashboard"
    assert dialog._dashboard_button.isEnabled() is False

    dialog.close()


def test_analysis_dialog_dashboard_button_uses_dashboard_relay_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    class _FakeRelayManager:
        def __init__(self) -> None:
            self.has_available_client_calls: list[str | None] = []
            self.open_dashboard_calls: list[str | None] = []

        def has_available_client(self, *, workload=None):
            self.has_available_client_calls.append(workload)
            return True

        def open_dashboard(self, *, workload=None):
            self.open_dashboard_calls.append(workload)
            return "http://127.0.0.1:8787/status?token=relay"

    relay_manager = _FakeRelayManager()
    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: relay_manager,
    )
    browser_urls: list[str] = []
    monkeypatch.setattr(
        app_module.webbrowser,
        "open_new_tab",
        lambda url: browser_urls.append(str(url)) or True,
    )
    warning_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warning_calls.append((title, message)),
    )

    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file="/tmp/test/data_store.fake")
    )

    assert relay_manager.has_available_client_calls == ["analysis"]
    assert dialog._dask_dashboard_button is not None
    assert dialog._dask_dashboard_button.isEnabled() is True
    assert dialog._dask_dashboard_button.toolTip() == (
        "Open the local Dask dashboard relay for analysis clients."
    )

    dialog._on_open_dask_dashboard()

    assert relay_manager.open_dashboard_calls == ["analysis"]
    assert warning_calls == []
    assert browser_urls == ["http://127.0.0.1:8787/status?token=relay"]
    dialog.close()


def test_analysis_dialog_dashboard_button_warns_when_relay_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    class _FakeRelayManager:
        def has_available_client(self, *, workload=None):
            del workload
            return False

        def open_dashboard(self, *, workload=None):
            del workload
            raise RuntimeError("relay startup failed")

    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: _FakeRelayManager(),
    )
    warning_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warning_calls.append((title, message)),
    )

    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file="/tmp/test/data_store.fake")
    )

    dialog._on_open_dask_dashboard()

    assert warning_calls
    assert warning_calls[0][0] == "Dashboard Launch Failed"
    assert "relay startup failed" in warning_calls[0][1]
    dialog.close()


def test_summarize_image_info_extracts_pixel_size_from_voxel_size_metadata() -> None:
    info = app_module.ImageInfo(
        path=Path("/tmp/data_store.zarr"),
        shape=(1, 1, 1, 10, 20, 30),
        dtype="uint16",
        axes="TPCZYX",
        metadata={"voxel_size_um_zyx": [0.2, 0.1669921875, 0.1669921875]},
    )

    summary = app_module.summarize_image_info(info)

    assert summary["pixel_size"] == "z=0.2, y=0.166992, x=0.166992"


def test_analysis_store_runtime_source_helpers_prefer_canonical_component(
    tmp_path: Path,
) -> None:
    store_path = _create_gui_analysis_store(tmp_path)
    root = zarr.open_group(str(store_path), mode="r")

    assert (
        app_module._analysis_store_runtime_source_component(root)
        == "clearex/runtime_cache/source/data"
    )
    assert app_module._analysis_store_runtime_source_shape_tpczyx(root) == (
        1,
        1,
        2,
        1,
        4,
        4,
    )
    assert app_module._analysis_store_runtime_source_dtype_itemsize(root) == 2


def test_discover_navigate_experiment_files_recurses_and_sorts(tmp_path) -> None:
    alpha = tmp_path / "alpha" / "experiment.yml"
    beta = tmp_path / "beta" / "nested" / "experiment.yaml"
    noise = tmp_path / "beta" / "nested" / "notes.yml"
    alpha.parent.mkdir(parents=True)
    beta.parent.mkdir(parents=True)
    alpha.write_text("alpha\n", encoding="utf-8")
    beta.write_text("beta\n", encoding="utf-8")
    noise.write_text("noise\n", encoding="utf-8")

    discovered = app_module._discover_navigate_experiment_files(tmp_path)

    assert discovered == [alpha.resolve(), beta.resolve()]


def test_resolve_initial_dialog_dimensions_keeps_larger_default_size() -> None:
    minimum_size, startup_size = app_module._resolve_initial_dialog_dimensions(
        minimum_size=app_module._SETUP_DIALOG_MINIMUM_SIZE,
        preferred_size=app_module._SETUP_DIALOG_PREFERRED_SIZE,
        available_size=(1920, 1200),
    )

    assert minimum_size == app_module._SETUP_DIALOG_MINIMUM_SIZE
    assert startup_size == app_module._SETUP_DIALOG_PREFERRED_SIZE


def test_gui_module_exports_fallback_helpers_when_pyqt6_is_unavailable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = f"""
import builtins
import importlib
import json
import sys

sys.path.insert(0, {str(repo_root / "src")!r})
original_import = builtins.__import__


def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "PyQt6" or name.startswith("PyQt6."):
        raise ImportError("blocked for test")
    return original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _blocked_import
module = importlib.import_module("clearex.gui.app")
print(
    json.dumps(
        {{
            "HAS_PYQT6": module.HAS_PYQT6,
            "has_apply_application_icon": hasattr(module, "_apply_application_icon"),
            "has_show_startup_splash": hasattr(module, "_show_startup_splash"),
        }}
    )
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert json.loads(result.stdout) == {
        "HAS_PYQT6": False,
        "has_apply_application_icon": True,
        "has_show_startup_splash": True,
    }


def test_resolve_initial_dialog_dimensions_clamps_to_available_screen() -> None:
    available_size = (1366, 900)
    minimum_size, startup_size = app_module._resolve_initial_dialog_dimensions(
        minimum_size=app_module._SETUP_DIALOG_MINIMUM_SIZE,
        preferred_size=app_module._SETUP_DIALOG_PREFERRED_SIZE,
        available_size=available_size,
    )

    expected_width = available_size[0] - app_module._DIALOG_SCREEN_MARGIN_PX
    expected_height = available_size[1] - app_module._DIALOG_SCREEN_MARGIN_PX

    assert startup_size == (expected_width, expected_height)
    assert minimum_size == (
        min(app_module._SETUP_DIALOG_MINIMUM_SIZE[0], expected_width),
        min(app_module._SETUP_DIALOG_MINIMUM_SIZE[1], expected_height),
    )


def test_resolve_initial_dialog_dimensions_expands_for_content_size_hint() -> None:
    minimum_size, startup_size = app_module._resolve_initial_dialog_dimensions(
        minimum_size=(1000, 900),
        preferred_size=(1200, 1000),
        available_size=(1800, 1600),
        content_size_hint=(1100, 1225),
    )

    assert minimum_size == (1100, 1225)
    assert startup_size == (1200, 1225)


def test_setup_dialog_keeps_experiment_controls_visible_on_short_screens(
    monkeypatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    dialog.show()
    app.processEvents()

    load_geometry = dialog._load_experiment_button.geometry()
    experiment_list_geometry = dialog._experiment_list.geometry()
    status_geometry = dialog._experiment_list_status_label.geometry()

    assert load_geometry.height() >= 36
    assert experiment_list_geometry.height() >= 220
    assert status_geometry.height() >= 28
    assert status_geometry.y() >= (
        experiment_list_geometry.y() + experiment_list_geometry.height()
    )

    dialog.close()


def test_setup_dialog_keeps_header_and_footer_pinned_while_body_scrolls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    dialog.show()
    app.processEvents()

    scroll = dialog.findChild(app_module.QScrollArea, "setupDialogScroll")
    header = dialog.findChild(app_module.QFrame, "headerCard")
    footer = dialog.findChild(app_module.QFrame, "setupFooterCard")

    assert scroll is not None
    assert header is not None
    assert footer is not None
    assert scroll.verticalScrollBar().maximum() > 0
    assert header.geometry().bottom() <= scroll.geometry().top()
    assert footer.geometry().top() >= scroll.geometry().bottom()
    assert dialog._next_button.geometry().height() > 0
    next_top = dialog._next_button.mapTo(dialog, app_module.QPoint(0, 0)).y()
    assert next_top >= footer.geometry().top()

    dialog.close()


def test_setup_dialog_places_compact_edit_buttons_at_top_right_of_summary_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (1200, 900)
    )

    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    dialog.show()
    app.processEvents()

    assert (
        dialog._zarr_config_button.geometry().x()
        > dialog._zarr_config_summary.geometry().x()
    )
    assert dialog._zarr_config_button.geometry().top() <= (
        dialog._zarr_config_summary.geometry().top() + 8
    )
    assert dialog._spatial_calibration_button.geometry().x() > (
        dialog._spatial_calibration_summary.geometry().x()
    )
    assert dialog._spatial_calibration_button.geometry().top() <= (
        dialog._spatial_calibration_summary.geometry().top() + 8
    )
    assert (
        dialog._dask_backend_button.geometry().x()
        > dialog._dask_backend_summary.geometry().x()
    )
    assert dialog._dask_backend_button.geometry().top() <= (
        dialog._dask_backend_summary.geometry().top() + 8
    )

    dialog.close()


def test_analysis_dialog_scrolls_body_on_short_screens(monkeypatch) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file="/tmp/test/data_store.fake")
    )
    dialog.show()
    app.processEvents()

    scroll = dialog.findChild(app_module.QScrollArea, "analysisDialogScroll")
    assert scroll is not None
    assert scroll.verticalScrollBar().maximum() > 0
    assert dialog._analysis_scope_summary_label is not None
    assert dialog._analysis_state_source_label is not None
    assert dialog._restore_latest_run_button is not None
    assert dialog._status_label is not None
    assert dialog._dask_backend_summary_label is not None
    assert dialog._cancel_button is not None
    assert dialog._run_button is not None
    assert dialog._analysis_scope_summary_label.geometry().height() >= 28
    assert dialog._analysis_state_source_label.geometry().height() >= 28
    assert dialog._restore_latest_run_button.geometry().height() >= 36
    assert dialog._status_label.geometry().height() >= 28
    assert dialog._dask_backend_summary_label.geometry().height() >= 28
    assert dialog._cancel_button.geometry().height() >= 36
    assert dialog._run_button.geometry().height() >= 36

    dialog.close()


def test_analysis_dialog_keeps_header_and_footer_pinned_while_body_scrolls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file="/tmp/test/data_store.fake")
    )
    dialog.show()
    app.processEvents()

    scroll = dialog.findChild(app_module.QScrollArea, "analysisDialogScroll")
    header = dialog.findChild(app_module.QFrame, "headerCard")
    footer = dialog.findChild(app_module.QFrame, "analysisFooterCard")

    assert scroll is not None
    assert header is not None
    assert footer is not None
    assert scroll.verticalScrollBar().maximum() > 0
    assert header.geometry().bottom() <= scroll.geometry().top()
    assert footer.geometry().top() >= scroll.geometry().bottom()
    assert dialog._run_button is not None
    assert dialog._dask_backend_button is not None
    run_top = dialog._run_button.mapTo(dialog, app_module.QPoint(0, 0)).y()
    dask_button_top = dialog._dask_backend_button.mapTo(
        dialog, app_module.QPoint(0, 0)
    ).y()
    assert run_top >= footer.geometry().top()
    assert dask_button_top >= footer.geometry().top()

    dialog.close()


def test_registration_operation_moves_to_preprocessing_tab() -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    tab_map = dict(app_module.AnalysisSelectionDialog._OPERATION_TABS)

    assert "display_pyramid" in tab_map["Preprocessing"]
    assert "registration" in tab_map["Preprocessing"]
    assert "registration" not in tab_map.get("Postprocessing", ())
    assert "render_movie" in tab_map["Visualization"]
    assert "compile_movie" in tab_map["Visualization"]


def test_analysis_dialog_payload_records_selected_analysis_tab(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    store_path = _create_gui_analysis_store(tmp_path)
    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )
    assert dialog._analysis_tabs is not None
    visualization_index = next(
        index
        for index in range(dialog._analysis_tabs.count())
        if dialog._analysis_tabs.tabText(index) == "Visualization"
    )
    dialog._analysis_tabs.setCurrentIndex(visualization_index)

    payload = dialog._analysis_gui_state_payload_from_current_widgets()

    assert payload["analysis_selected_tab"] == "Visualization"
    dialog.close()


def test_analysis_dialog_restores_saved_analysis_tab(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    store_path = _create_gui_analysis_store(tmp_path)
    zarr.open_group(str(store_path), mode="a").require_group("clearex").require_group(
        "provenance"
    )
    initial_workflow = app_module.WorkflowConfig(file=str(store_path))
    app_module.persist_latest_analysis_gui_state(
        str(store_path),
        app_module._analysis_gui_state_payload_from_workflow(
            initial_workflow,
            analysis_selected_tab="Visualization",
        ),
    )

    dialog = app_module.AnalysisSelectionDialog(initial=initial_workflow)

    assert dialog._analysis_tabs is not None
    current_index = dialog._analysis_tabs.currentIndex()
    assert dialog._analysis_tabs.tabText(current_index) == "Visualization"
    dialog.close()


def test_analysis_dialog_clamps_volume_export_indices_and_preserves_requested_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module,
        "_save_last_used_dask_backend_config",
        lambda _config: None,
    )

    store_path = _create_gui_analysis_store(tmp_path)
    zarr.open_group(str(store_path), mode="a").require_group("clearex").require_group(
        "provenance"
    )
    workflow = app_module.WorkflowConfig(
        file=str(store_path),
        volume_export=True,
        analysis_parameters={
            "volume_export": {
                "execution_order": 11,
                "input_source": "shear_transform",
                "force_rerun": False,
                "export_scope": "current_selection",
                "t_index": 9,
                "p_index": 8,
                "c_index": 7,
                "resolution_level": 12,
                "export_format": "ome-tiff",
                "tiff_file_layout": "per_volume_files",
            }
        },
    )
    payload = app_module._analysis_gui_state_payload_from_workflow(
        workflow,
        analysis_selected_tab="Visualization",
    )
    payload["analysis_parameters"]["volume_export"].update(
        {
            "t_index": 9,
            "p_index": 8,
            "c_index": 7,
            "resolution_level": 12,
            "input_source": "shear_transform",
            "export_scope": "current_selection",
            "export_format": "ome-tiff",
            "tiff_file_layout": "per_volume_files",
        }
    )
    app_module.persist_latest_analysis_gui_state(str(store_path), payload)

    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )

    assert dialog._operation_checkboxes["volume_export"].isChecked() is True
    assert (
        dialog._operation_input_combos["volume_export"].currentData()
        == "shear_transform"
    )
    assert dialog._volume_export_scope_combo.currentData() == "current_selection"
    assert dialog._volume_export_t_spin.maximum() == 0
    assert dialog._volume_export_t_spin.value() == 0
    assert dialog._volume_export_p_spin.maximum() == 0
    assert dialog._volume_export_p_spin.value() == 0
    assert dialog._volume_export_c_spin.maximum() == 0
    assert dialog._volume_export_c_spin.value() == 0
    assert dialog._volume_export_resolution_level_spin.maximum() == 64
    assert dialog._volume_export_resolution_level_spin.value() == 12
    assert dialog._volume_export_format_combo.currentData() == "ome-tiff"
    assert dialog._volume_export_t_spin.isEnabled() is True
    assert dialog._volume_export_p_spin.isEnabled() is True
    assert dialog._volume_export_c_spin.isEnabled() is True
    assert dialog._volume_export_tiff_layout_combo.isEnabled() is True
    assert dialog._volume_export_tiff_layout_combo.currentData() == "per_volume_files"

    input_combo = dialog._operation_input_combos["volume_export"]
    data_index = input_combo.findData("data")
    assert data_index >= 0
    input_combo.setCurrentIndex(data_index)
    app.processEvents()
    assert dialog._volume_export_c_spin.maximum() == 1
    assert dialog._volume_export_resolution_level_spin.maximum() == 64
    assert dialog._volume_export_resolution_level_spin.value() == 12

    dialog.close()


def test_analysis_dialog_persists_registration_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module,
        "_save_last_used_dask_backend_config",
        lambda _config: None,
    )

    store_path = _create_gui_analysis_store(tmp_path)
    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )
    dialog._persist_analysis_gui_state_for_target = lambda _target: None
    dialog._operation_checkboxes["registration"].setChecked(True)
    dialog._operation_checkboxes["fusion"].setChecked(True)
    dialog._registration_type_combo.setCurrentIndex(
        dialog._registration_type_combo.findData("similarity")
    )
    dialog._registration_anchor_mode_combo.setCurrentIndex(
        dialog._registration_anchor_mode_combo.findData("manual")
    )
    dialog._registration_anchor_position_spin.setMaximum(3)
    dialog._registration_anchor_position_spin.setValue(2)
    dialog._registration_pairwise_overlap_z_spin.setValue(5)
    dialog._registration_pairwise_overlap_y_spin.setValue(12)
    dialog._registration_pairwise_overlap_x_spin.setValue(20)
    dialog._fusion_overlap_z_spin.setValue(6)
    dialog._fusion_overlap_y_spin.setValue(14)
    dialog._fusion_overlap_x_spin.setValue(22)
    dialog._fusion_blend_mode_combo.setCurrentIndex(
        dialog._fusion_blend_mode_combo.findData("gain_compensated_feather")
    )
    dialog._fusion_blend_exponent_spin.setValue(1.8)
    dialog._fusion_gain_clip_min_spin.setValue(0.6)
    dialog._fusion_gain_clip_max_spin.setValue(2.4)
    input_combo = dialog._operation_input_combos["registration"]
    data_index = input_combo.findData("data")
    assert data_index >= 0
    input_combo.setCurrentIndex(data_index)
    app.processEvents()
    channel_1_index = dialog._registration_channel_combo.findData(1)
    assert channel_1_index >= 0
    dialog._registration_channel_combo.setCurrentIndex(channel_1_index)
    level_1_index = dialog._registration_resolution_level_combo.findData(1)
    expected_resolution_level = 0
    if level_1_index >= 0:
        dialog._registration_resolution_level_combo.setCurrentIndex(level_1_index)
        expected_resolution_level = 1
    app_module.AnalysisSelectionDialog._set_registration_parameter_enabled_state(dialog)
    app_module.AnalysisSelectionDialog._set_fusion_parameter_enabled_state(dialog)

    dialog._on_run()

    params = dialog.result_config.analysis_parameters["registration"]
    fusion = dialog.result_config.analysis_parameters["fusion"]
    assert dialog.result_config.registration is True
    assert dialog.result_config.fusion is True
    assert params["registration_type"] == "similarity"
    assert params["anchor_mode"] == "manual"
    assert params["anchor_position"] == 2
    assert params["pairwise_overlap_zyx"] == [5, 12, 20]
    assert params["registration_channel"] == 1
    assert params["input_resolution_level"] == expected_resolution_level
    assert fusion["blend_overlap_zyx"] == [6, 14, 22]
    assert fusion["blend_mode"] == "gain_compensated_feather"
    assert fusion["blend_exponent"] == pytest.approx(1.8)
    assert fusion["gain_clip_range"] == pytest.approx([0.6, 2.4])


def test_registration_resolution_levels_follow_selected_input_source(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    store_path = _create_gui_analysis_store(tmp_path)
    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )
    dialog._persist_analysis_gui_state_for_target = lambda _target: None
    dialog._operation_checkboxes["registration"].setChecked(True)
    input_combo = dialog._operation_input_combos["registration"]
    level_combo = dialog._registration_resolution_level_combo

    data_index = input_combo.findData("data")
    assert data_index >= 0
    input_combo.setCurrentIndex(data_index)
    app.processEvents()
    dialog._refresh_registration_resolution_level_options(preferred_level=1)
    data_levels = [int(level_combo.itemData(idx)) for idx in range(level_combo.count())]
    assert data_levels
    assert data_levels[0] == 0

    level_1_index = level_combo.findData(1)
    if level_1_index >= 0:
        level_combo.setCurrentIndex(level_1_index)
    else:
        level_combo.setCurrentIndex(level_combo.findData(0))

    shear_index = input_combo.findData("shear_transform")
    assert shear_index >= 0
    input_combo.setCurrentIndex(shear_index)
    app.processEvents()
    assert [int(level_combo.itemData(idx)) for idx in range(level_combo.count())] == [0]
    assert int(level_combo.currentData()) == 0
    params = dialog._collect_registration_parameters()
    assert params["input_resolution_level"] == 0


def test_registration_channels_follow_selected_input_source(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    store_path = _create_gui_analysis_store(tmp_path)
    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )
    dialog._persist_analysis_gui_state_for_target = lambda _target: None
    dialog._operation_checkboxes["registration"].setChecked(True)
    input_combo = dialog._operation_input_combos["registration"]
    channel_combo = dialog._registration_channel_combo

    data_index = input_combo.findData("data")
    assert data_index >= 0
    input_combo.setCurrentIndex(data_index)
    app.processEvents()
    assert [
        int(channel_combo.itemData(idx)) for idx in range(channel_combo.count())
    ] == [0, 1]

    channel_1_index = channel_combo.findData(1)
    assert channel_1_index >= 0
    channel_combo.setCurrentIndex(channel_1_index)

    shear_index = input_combo.findData("shear_transform")
    assert shear_index >= 0
    input_combo.setCurrentIndex(shear_index)
    app.processEvents()
    assert [
        int(channel_combo.itemData(idx)) for idx in range(channel_combo.count())
    ] == [0]
    assert int(channel_combo.currentData()) == 0
    params = dialog._collect_registration_parameters()
    assert params["registration_channel"] == 0


def test_registration_anchor_tile_visibility_follows_anchor_mode(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    store_path = _create_gui_analysis_store(tmp_path)
    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )
    dialog._persist_analysis_gui_state_for_target = lambda _target: None
    dialog._operation_checkboxes["registration"].setChecked(True)

    dialog._registration_anchor_mode_combo.setCurrentIndex(
        dialog._registration_anchor_mode_combo.findData("central")
    )
    app_module.AnalysisSelectionDialog._set_registration_parameter_enabled_state(dialog)
    assert dialog._registration_anchor_position_label.isHidden() is True
    assert dialog._registration_anchor_position_spin.isHidden() is True

    dialog._registration_anchor_mode_combo.setCurrentIndex(
        dialog._registration_anchor_mode_combo.findData("manual")
    )
    app_module.AnalysisSelectionDialog._set_registration_parameter_enabled_state(dialog)
    assert dialog._registration_anchor_position_label.isHidden() is False
    assert dialog._registration_anchor_position_spin.isHidden() is False


def test_zarr_dialog_scrolls_body_on_short_screens(monkeypatch) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.ZarrSaveConfigDialog(initial=app_module.ZarrSaveConfig())
    dialog.show()
    app.processEvents()

    scroll = dialog.findChild(app_module.QScrollArea, "popupDialogScroll")
    assert scroll is not None
    assert scroll.verticalScrollBar().maximum() > 0
    assert dialog._defaults_button.geometry().height() >= 36
    assert dialog._cancel_button.geometry().height() >= 36
    assert dialog._apply_button.geometry().height() >= 36

    dialog.close()


def test_dask_dialog_scrolls_body_on_short_screens(monkeypatch) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.DaskBackendConfigDialog(
        initial=app_module.DaskBackendConfig(),
        recommendation_shape_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_chunks_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_dtype_itemsize=2,
    )
    dialog.show()
    app.processEvents()

    scroll = dialog.findChild(app_module.QScrollArea, "popupDialogScroll")
    assert scroll is not None
    assert scroll.verticalScrollBar().maximum() > 0
    assert dialog._defaults_button.geometry().height() >= 36
    assert dialog._cancel_button.geometry().height() >= 36
    assert dialog._apply_button.geometry().height() >= 36
    assert dialog._mode_help_label.geometry().height() >= 28
    assert dialog._local_recommendation_label.geometry().height() >= 28

    dialog.close()


def test_save_and_load_experiment_list_round_trip(tmp_path) -> None:
    first = tmp_path / "first" / "experiment.yml"
    second = tmp_path / "second" / "experiment.yaml"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    list_path = tmp_path / f"batch{app_module._CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX}"

    saved_path = app_module._save_experiment_list_file(list_path, [first, second])
    loaded_paths = app_module._load_experiment_list_file(saved_path)

    assert saved_path == list_path.resolve()
    assert loaded_paths == [first.resolve(), second.resolve()]


def test_deduplicate_resolved_paths_preserves_first_seen_order(tmp_path) -> None:
    first = tmp_path / "cell_001" / "experiment.yml"
    second = tmp_path / "cell_002" / "experiment.yml"
    paths = [first, second, first.resolve(), second.resolve(), first]

    deduplicated = app_module._deduplicate_resolved_paths(paths)

    assert deduplicated == [first.resolve(), second.resolve()]


def test_plan_experiment_store_materialization_partitions_pending_requests(
    monkeypatch,
    tmp_path,
) -> None:
    first = tmp_path / "cell_001" / "experiment.yml"
    second = tmp_path / "cell_002" / "experiment.yml"
    third = tmp_path / "cell_003" / "experiment.yml"
    requests = [
        app_module.ExperimentStorePreparationRequest(
            experiment_path=first.resolve(),
            experiment=_make_navigate_experiment(first),
            source_data_path=first.parent / "CH00_000000.ome.tiff",
            target_store=first.parent / "data_store.zarr",
        ),
        app_module.ExperimentStorePreparationRequest(
            experiment_path=second.resolve(),
            experiment=_make_navigate_experiment(second),
            source_data_path=second.parent / "CH00_000000.ome.tiff",
            target_store=second.parent / "data_store.zarr",
        ),
        app_module.ExperimentStorePreparationRequest(
            experiment_path=third.resolve(),
            experiment=_make_navigate_experiment(third),
            source_data_path=third.parent / "CH00_000000.ome.tiff",
            target_store=third.parent / "data_store.zarr",
        ),
    ]

    monkeypatch.setattr(
        app_module,
        "_has_reusable_canonical_store",
        lambda store_path: (
            Path(store_path).expanduser().resolve()
            == requests[2].target_store.resolve()
        ),
    )

    selected_request, pending_requests, ready_requests = (
        app_module._plan_experiment_store_materialization(
            requests,
            selected_experiment_path=second,
        )
    )

    assert selected_request.experiment_path == second.resolve()
    assert [request.experiment_path for request in pending_requests] == [
        first.resolve(),
        second.resolve(),
    ]
    assert [request.experiment_path for request in ready_requests] == [third.resolve()]


def test_plan_experiment_store_materialization_rebuilds_all_when_requested(
    monkeypatch,
    tmp_path,
) -> None:
    first = tmp_path / "cell_001" / "experiment.yml"
    second = tmp_path / "cell_002" / "experiment.yml"
    requests = [
        app_module.ExperimentStorePreparationRequest(
            experiment_path=first.resolve(),
            experiment=_make_navigate_experiment(first),
            source_data_path=first.parent / "CH00_000000.ome.tiff",
            target_store=first.parent / "data_store.zarr",
        ),
        app_module.ExperimentStorePreparationRequest(
            experiment_path=second.resolve(),
            experiment=_make_navigate_experiment(second),
            source_data_path=second.parent / "CH00_000000.ome.tiff",
            target_store=second.parent / "data_store.zarr",
        ),
    ]

    monkeypatch.setattr(
        app_module,
        "_has_reusable_canonical_store",
        lambda store_path: True,
    )

    selected_request, pending_requests, ready_requests = (
        app_module._plan_experiment_store_materialization(
            requests,
            selected_experiment_path=second,
            rebuild_requested=True,
        )
    )

    assert selected_request.experiment_path == second.resolve()
    assert [request.experiment_path for request in pending_requests] == [
        first.resolve(),
        second.resolve(),
    ]
    assert ready_requests == []


def test_analysis_targets_for_workflow_falls_back_to_current_file() -> None:
    workflow = app_module.WorkflowConfig(file="/tmp/current/data_store.zarr")

    targets = app_module._analysis_targets_for_workflow(workflow)

    assert targets == (
        app_module.AnalysisTarget(
            experiment_path="/tmp/current/data_store.zarr",
            store_path="/tmp/current/data_store.zarr",
        ),
    )


def test_workflows_for_selected_analysis_scope_expands_batch_targets() -> None:
    workflow = app_module.WorkflowConfig(
        file="/tmp/cell_002/data_store.zarr",
        analysis_targets=(
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_001/experiment.yml",
                store_path="/tmp/cell_001/data_store.zarr",
            ),
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_002/experiment.yml",
                store_path="/tmp/cell_002/data_store.zarr",
            ),
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_003/experiment.yml",
                store_path="/tmp/cell_003/data_store.zarr",
            ),
        ),
        analysis_selected_experiment_path="/tmp/cell_002/experiment.yml",
        analysis_apply_to_all=True,
        flatfield=True,
    )

    scoped = app_module._workflows_for_selected_analysis_scope(workflow)

    assert [entry.file for entry in scoped] == [
        "/tmp/cell_001/data_store.zarr",
        "/tmp/cell_002/data_store.zarr",
        "/tmp/cell_003/data_store.zarr",
    ]
    assert [entry.analysis_selected_experiment_path for entry in scoped] == [
        "/tmp/cell_001/experiment.yml",
        "/tmp/cell_002/experiment.yml",
        "/tmp/cell_003/experiment.yml",
    ]
    assert all(entry.analysis_apply_to_all is False for entry in scoped)
    assert all(entry.flatfield is True for entry in scoped)


def test_reset_analysis_selection_for_next_run_preserves_scope() -> None:
    workflow = app_module.WorkflowConfig(
        file="/tmp/cell_002/data_store.zarr",
        analysis_targets=(
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_001/experiment.yml",
                store_path="/tmp/cell_001/data_store.zarr",
            ),
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_002/experiment.yml",
                store_path="/tmp/cell_002/data_store.zarr",
            ),
        ),
        analysis_selected_experiment_path="/tmp/cell_002/experiment.yml",
        analysis_apply_to_all=True,
        flatfield=True,
        registration=True,
        fusion=True,
        visualization=True,
        analysis_parameters={
            "flatfield": {"force_rerun": True},
            "registration": {"force_rerun": True},
            "fusion": {"force_rerun": True},
        },
    )

    reset = app_module._reset_analysis_selection_for_next_run(workflow)

    assert reset.file == "/tmp/cell_002/data_store.zarr"
    assert reset.analysis_targets == workflow.analysis_targets
    assert reset.analysis_selected_experiment_path == "/tmp/cell_002/experiment.yml"
    assert reset.analysis_apply_to_all is True
    assert reset.flatfield is False
    assert reset.registration is False
    assert reset.fusion is False
    assert reset.display_pyramid is False
    assert reset.visualization is False
    assert reset.analysis_parameters["flatfield"]["force_rerun"] is False
    assert reset.analysis_parameters["registration"]["force_rerun"] is False
    assert reset.analysis_parameters["fusion"]["force_rerun"] is False


def test_reset_analysis_selection_for_next_run_preserves_spatial_calibration() -> None:
    workflow = app_module.WorkflowConfig(
        file="/tmp/cell_001/data_store.zarr",
        spatial_calibration=app_module.SpatialCalibrationConfig(
            stage_axis_map_zyx=("+x", "none", "+y")
        ),
        spatial_calibration_explicit=True,
        visualization=True,
    )

    reset = app_module._reset_analysis_selection_for_next_run(workflow)

    assert reset.spatial_calibration == workflow.spatial_calibration
    assert reset.spatial_calibration_explicit is True


def test_persist_reset_analysis_gui_state_for_workflow_clears_saved_flags(
    tmp_path: Path,
) -> None:
    store_path = _create_gui_analysis_store(tmp_path)
    root = zarr.open_group(str(store_path), mode="a")
    root.require_group("clearex").require_group("provenance")
    app_module.persist_latest_analysis_gui_state(
        str(store_path),
        {
            "analysis_selected_tab": "Visualization",
            "analysis_parameters": {},
        },
    )
    workflow = app_module.WorkflowConfig(
        file=str(store_path),
        analysis_selected_experiment_path="/tmp/cell_001/experiment.yml",
        flatfield=True,
        registration=True,
        fusion=True,
        analysis_parameters={
            "flatfield": {"force_rerun": True},
            "registration": {"force_rerun": True},
            "fusion": {"force_rerun": True},
        },
    )

    reset = app_module._reset_analysis_selection_for_next_run(workflow)
    app_module._persist_reset_analysis_gui_state_for_workflow(reset)

    saved_state = app_module.load_latest_analysis_gui_state(str(store_path))

    assert saved_state is not None
    workflow_payload = saved_state["workflow"]
    assert workflow_payload["flatfield"] is False
    assert workflow_payload["registration"] is False
    assert workflow_payload["fusion"] is False
    assert workflow_payload["analysis_selected_tab"] == "Visualization"
    assert workflow_payload["analysis_parameters"]["flatfield"]["force_rerun"] is False
    assert (
        workflow_payload["analysis_parameters"]["registration"]["force_rerun"] is False
    )
    assert workflow_payload["analysis_parameters"]["fusion"]["force_rerun"] is False


def test_load_experiment_list_file_rejects_invalid_format(tmp_path) -> None:
    list_path = tmp_path / f"broken{app_module._CLEAREX_EXPERIMENT_LIST_FILE_SUFFIX}"
    list_path.write_text('{"format":"wrong","experiments":["a"]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported experiment list format"):
        app_module._load_experiment_list_file(list_path)


def test_summarize_image_info_extracts_pixel_size_from_navigate_metadata() -> None:
    info = app_module.ImageInfo(
        path=Path("/tmp/data_store.zarr"),
        shape=(1, 1, 1, 10, 20, 30),
        dtype="uint16",
        axes="TPCZYX",
        metadata={
            "navigate_experiment": {
                "xy_pixel_size_um": 0.15,
                "z_step_um": 0.4,
            }
        },
    )

    summary = app_module.summarize_image_info(info)

    assert summary["pixel_size"] == "z=0.4, y=0.15, x=0.15"


def test_apply_experiment_overrides_populates_pixel_size_field() -> None:
    summary = {
        "path": "/tmp/source",
        "shape": "1 x 1 x 1",
        "dtype": "uint16",
        "axes": "TPCZYX",
        "channels": "1",
        "positions": "1",
        "image_size": "30 x 20 x 10",
        "time_points": "1",
        "pixel_size": "n/a",
        "metadata_keys": "a,b",
    }
    experiment = NavigateExperiment(
        path=Path("/tmp/experiment.yml"),
        raw={},
        save_directory=Path("/tmp"),
        file_type="TIFF",
        microscope_name="scope",
        image_mode="z-stack",
        timepoints=1,
        number_z_steps=10,
        y_pixels=20,
        x_pixels=30,
        multiposition_count=1,
        selected_channels=[
            NavigateChannel(
                name="ch0",
                laser=None,
                laser_index=None,
                exposure_ms=None,
                is_selected=True,
            )
        ],
        xy_pixel_size_um=0.12,
        z_step_um=0.45,
    )

    updated = app_module._apply_experiment_overrides(
        summary=summary,
        experiment_path=Path("/tmp/experiment.yml"),
        resolved_data_path=Path("/tmp/source"),
        experiment=experiment,
    )

    assert updated["pixel_size"] == "z=0.45, y=0.12, x=0.12"


def test_workflows_for_selected_analysis_scope_uses_store_spatial_calibration(
    tmp_path: Path,
) -> None:
    first_store = tmp_path / "cell_001" / "data_store.zarr"
    second_store = tmp_path / "cell_002" / "data_store.zarr"
    for store in (first_store, second_store):
        store.parent.mkdir(parents=True, exist_ok=True)
        root = app_module.zarr.open_group(str(store), mode="w")
        root.create_dataset(
            name="data",
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="uint16",
            overwrite=True,
        )
    app_module.save_store_spatial_calibration(
        first_store,
        app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y")),
    )
    app_module.save_store_spatial_calibration(
        second_store,
        app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+f", "-y", "+x")),
    )

    workflow = app_module.WorkflowConfig(
        file=str(second_store),
        analysis_targets=(
            app_module.AnalysisTarget(
                experiment_path=str(tmp_path / "cell_001" / "experiment.yml"),
                store_path=str(first_store),
            ),
            app_module.AnalysisTarget(
                experiment_path=str(tmp_path / "cell_002" / "experiment.yml"),
                store_path=str(second_store),
            ),
        ),
        analysis_selected_experiment_path=str(tmp_path / "cell_002" / "experiment.yml"),
        analysis_apply_to_all=True,
        visualization=True,
    )

    scoped = app_module._workflows_for_selected_analysis_scope(workflow)

    assert [entry.spatial_calibration.stage_axis_map_zyx for entry in scoped] == [
        ("+x", "none", "+y"),
        ("+f", "-y", "+x"),
    ]


def test_setup_dialog_resolves_spatial_calibration_drafts_per_experiment() -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    original_branding_factory = app_module._create_scaled_branding_label
    app_module._create_scaled_branding_label = lambda **_kwargs: None
    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    try:
        first = Path("/tmp/cell_001/experiment.yml")
        second = Path("/tmp/cell_002/experiment.yml")
        dialog._spatial_calibration_drafts[first.resolve()] = (
            app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y"))
        )
        dialog._spatial_calibration_drafts[second.resolve()] = (
            app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+f", "-y", "+x"))
        )

        dialog._set_current_spatial_calibration(experiment_path=first)
        assert dialog._current_spatial_calibration.stage_axis_map_zyx == (
            "+x",
            "none",
            "+y",
        )

        dialog._set_current_spatial_calibration(experiment_path=second)
        assert dialog._current_spatial_calibration.stage_axis_map_zyx == (
            "+f",
            "-y",
            "+x",
        )
    finally:
        dialog.close()
        app_module._create_scaled_branding_label = original_branding_factory


def test_setup_dialog_delete_key_removes_selected_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    monkeypatch.setattr(
        dialog,
        "_load_metadata_for_experiment_path",
        lambda _experiment_path, *, force_reload=False: None,
    )
    first = Path("/tmp/cell_001/experiment.yml").resolve()
    second = Path("/tmp/cell_002/experiment.yml").resolve()
    dialog._set_experiment_list_paths((first, second), current_path=first)
    dialog._experiment_list.setCurrentRow(0)

    event = app_module.QKeyEvent(
        app_module.QEvent.Type.KeyPress,
        app_module.Qt.Key.Key_Delete,
        app_module.Qt.KeyboardModifier.NoModifier,
    )
    handled = dialog.eventFilter(dialog._experiment_list, event)

    assert handled is True
    assert dialog._experiment_list.count() == 1
    remaining = dialog._experiment_path_from_item(dialog._experiment_list.item(0))
    assert remaining == second

    dialog.close()


def test_setup_dialog_prefills_spatial_calibration_from_existing_store(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    store_path = tmp_path / "existing_store.zarr"
    root = app_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    app_module.save_store_spatial_calibration(
        store_path,
        app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y")),
    )

    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    dialog._set_current_spatial_calibration(
        experiment_path=tmp_path / "experiment.yml",
        target_store=store_path,
    )

    assert dialog._current_spatial_calibration.stage_axis_map_zyx == (
        "+x",
        "none",
        "+y",
    )
    assert "z=+x,y=none,x=+y" in dialog._spatial_calibration_summary.text()

    dialog.close()


def test_setup_dialog_persists_spatial_calibration_for_all_requests(
    tmp_path: Path,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    dialog = app_module.ClearExSetupDialog(initial=app_module.WorkflowConfig())
    experiment_a = tmp_path / "cell_001" / "experiment.yml"
    experiment_b = tmp_path / "cell_002" / "experiment.yml"
    store_a = tmp_path / "cell_001" / "data_store.zarr"
    store_b = tmp_path / "cell_002" / "data_store.zarr"
    for store in (store_a, store_b):
        store.parent.mkdir(parents=True, exist_ok=True)
        root = app_module.zarr.open_group(str(store), mode="w")
        root.create_dataset(
            name="data",
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="uint16",
            overwrite=True,
        )

    dialog._spatial_calibration_drafts[experiment_a.resolve()] = (
        app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y"))
    )
    dialog._spatial_calibration_drafts[experiment_b.resolve()] = (
        app_module.SpatialCalibrationConfig(stage_axis_map_zyx=("+f", "-y", "+x"))
    )
    requests = (
        app_module.ExperimentStorePreparationRequest(
            experiment_path=experiment_a.resolve(),
            experiment=_make_navigate_experiment(experiment_a),
            source_data_path=tmp_path / "cell_001" / "raw.tif",
            target_store=store_a.resolve(),
        ),
        app_module.ExperimentStorePreparationRequest(
            experiment_path=experiment_b.resolve(),
            experiment=_make_navigate_experiment(experiment_b),
            source_data_path=tmp_path / "cell_002" / "raw.tif",
            target_store=store_b.resolve(),
        ),
    )

    persisted = dialog._persist_spatial_calibration_for_requests(requests)

    assert persisted[store_a.resolve()].stage_axis_map_zyx == ("+x", "none", "+y")
    assert persisted[store_b.resolve()].stage_axis_map_zyx == ("+f", "-y", "+x")
    assert app_module.load_store_spatial_calibration(store_a).stage_axis_map_zyx == (
        "+x",
        "none",
        "+y",
    )
    assert app_module.load_store_spatial_calibration(store_b).stage_axis_map_zyx == (
        "+f",
        "-y",
        "+x",
    )

    dialog.close()


def test_run_workflow_with_progress_slurm_executes_callback_on_main_thread(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )
    seen_threads: list[threading.Thread] = []

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        assert callable(dask_client_lifecycle_callback)
        seen_threads.append(threading.current_thread())
        progress_callback(35, "SLURM setup")
        progress_callback(80, "running")

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert seen_threads == [threading.main_thread()]

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.process_events_calls >= 3
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is True
    assert dialog.accepted is True
    assert dialog.rejected is False
    assert dialog.closed is True
    assert dialog.updates[0] == (1, "Starting analysis workflow...")
    assert dialog.updates[-1] == (100, "Analysis workflow completed.")


def test_run_workflow_with_progress_slurm_forwards_lifecycle_callback(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )

    class _FakeRelayManager:
        def __init__(self) -> None:
            self.registered_clients: list[tuple[str, str, object]] = []
            self.unregistered_client_ids: list[str] = []

        def register_client(self, *, workload, backend_mode, client):
            self.registered_clients.append((workload, backend_mode, client))
            return f"{workload}-client"

        def unregister_client(self, client_id: str) -> None:
            self.unregistered_client_ids.append(client_id)

        def has_available_client(self, *, workload=None):
            del workload
            return bool(self.registered_clients)

        def open_dashboard(self, *, workload=None):
            del workload
            return "http://127.0.0.1:8787/status?token=relay"

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: _FakeRelayManager(),
    )
    lifecycle_events: list[tuple[str, str, str, object]] = []
    seen_callbacks: list[object] = []

    def _lifecycle_callback(event: str, workload: str, backend_mode: str, client):
        lifecycle_events.append((event, workload, backend_mode, client))

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        seen_callbacks.append(dask_client_lifecycle_callback)
        assert callable(dask_client_lifecycle_callback)
        dask_client_lifecycle_callback(
            "started",
            "analysis",
            app_module.DASK_BACKEND_SLURM_CLUSTER,
            object(),
        )
        progress_callback(40, "running")

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
        dask_client_lifecycle_callback=_lifecycle_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert len(seen_callbacks) == 1
    assert len(lifecycle_events) == 1

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is True
    assert dialog.accepted is True
    assert dialog.rejected is False
    assert dialog.closed is True


def test_run_workflow_with_progress_slurm_batches_all_selected_experiments(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )
    executed_files: list[str] = []

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        executed_files.append(str(workflow.file))
        assert callable(dask_client_lifecycle_callback)
        progress_callback(50, "running")

    workflow = app_module.WorkflowConfig(
        file="/tmp/cell_002/data_store.zarr",
        analysis_targets=(
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_001/experiment.yml",
                store_path="/tmp/cell_001/data_store.zarr",
            ),
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_002/experiment.yml",
                store_path="/tmp/cell_002/data_store.zarr",
            ),
        ),
        analysis_selected_experiment_path="/tmp/cell_002/experiment.yml",
        analysis_apply_to_all=True,
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        ),
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert executed_files == [
        "/tmp/cell_001/data_store.zarr",
        "/tmp/cell_002/data_store.zarr",
    ]

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is True
    assert dialog.accepted is True
    assert dialog.rejected is False
    assert dialog.closed is True
    assert dialog.updates[0] == (
        1,
        "Starting batch analysis workflow for 2 experiments...",
    )
    assert dialog.updates[-1] == (100, "Completed analysis for 2 experiments.")


def test_run_workflow_with_progress_slurm_batches_forward_lifecycle_callback(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )

    class _FakeRelayManager:
        def __init__(self) -> None:
            self.registered_clients: list[tuple[str, str, object]] = []
            self.unregistered_client_ids: list[str] = []

        def register_client(self, *, workload, backend_mode, client):
            self.registered_clients.append((workload, backend_mode, client))
            return f"{workload}-client"

        def unregister_client(self, client_id: str) -> None:
            self.unregistered_client_ids.append(client_id)

        def has_available_client(self, *, workload=None):
            del workload
            return bool(self.registered_clients)

        def open_dashboard(self, *, workload=None):
            del workload
            return "http://127.0.0.1:8787/status?token=relay"

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: _FakeRelayManager(),
    )
    executed_files: list[str] = []
    seen_callbacks: list[object] = []
    lifecycle_events: list[tuple[str, str, str, object]] = []

    def _lifecycle_callback(event: str, workload: str, backend_mode: str, client):
        lifecycle_events.append((event, workload, backend_mode, client))

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        executed_files.append(str(workflow.file))
        seen_callbacks.append(dask_client_lifecycle_callback)
        assert callable(dask_client_lifecycle_callback)
        dask_client_lifecycle_callback(
            "started",
            "analysis",
            app_module.DASK_BACKEND_SLURM_CLUSTER,
            object(),
        )
        progress_callback(50, "running")

    workflow = app_module.WorkflowConfig(
        file="/tmp/cell_002/data_store.zarr",
        analysis_targets=(
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_001/experiment.yml",
                store_path="/tmp/cell_001/data_store.zarr",
            ),
            app_module.AnalysisTarget(
                experiment_path="/tmp/cell_002/experiment.yml",
                store_path="/tmp/cell_002/data_store.zarr",
            ),
        ),
        analysis_selected_experiment_path="/tmp/cell_002/experiment.yml",
        analysis_apply_to_all=True,
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        ),
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
        dask_client_lifecycle_callback=_lifecycle_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert executed_files == [
        "/tmp/cell_001/data_store.zarr",
        "/tmp/cell_002/data_store.zarr",
    ]
    assert len(seen_callbacks) == 2
    assert len(lifecycle_events) == 2

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is True
    assert dialog.accepted is True
    assert dialog.rejected is False
    assert dialog.closed is True


def test_run_workflow_with_progress_slurm_shows_error_dialog_on_failure(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )
    seen_threads: list[threading.Thread] = []

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        assert callable(dask_client_lifecycle_callback)
        seen_threads.append(threading.current_thread())
        progress_callback(5, "starting")
        raise RuntimeError("boom")

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
    )

    assert ok is False
    assert seen_threads == [threading.main_thread()]
    assert len(themed_error_calls) == 1
    assert themed_error_calls[0]["summary"] == "RuntimeError: boom"
    assert "RuntimeError: boom" in themed_error_calls[0]["details"]

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is True
    assert dialog.accepted is False
    assert dialog.rejected is True
    assert dialog.closed is True


def test_run_workflow_with_progress_slurm_cancels_without_error_dialog(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        assert callable(dask_client_lifecycle_callback)
        dialog = fake_dialog_cls.last_instance
        assert dialog is not None
        dialog.cancel_requested.emit()
        progress_callback(25, "should cancel")

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
    )

    assert ok is False
    assert themed_error_calls == []

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is True
    assert dialog.accepted is False
    assert dialog.rejected is True
    assert dialog.closed is True
    assert dialog.updates == [(1, "Starting analysis workflow...")]


def test_run_workflow_with_progress_worker_forwards_lifecycle_callback(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )

    class _FakeRelayManager:
        def __init__(self) -> None:
            self.registered_clients: list[tuple[str, str, object]] = []
            self.unregistered_client_ids: list[str] = []

        def register_client(self, *, workload, backend_mode, client):
            self.registered_clients.append((workload, backend_mode, client))
            return f"{workload}-client"

        def unregister_client(self, client_id: str) -> None:
            self.unregistered_client_ids.append(client_id)

        def has_available_client(self, *, workload=None):
            del workload
            return bool(self.registered_clients)

        def open_dashboard(self, *, workload=None):
            del workload
            return "http://127.0.0.1:8787/status?token=relay"

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: _FakeRelayManager(),
    )
    seen_callbacks: list[object] = []
    lifecycle_events: list[tuple[str, str, str, object]] = []

    class _FakeSignal:
        def __init__(self) -> None:
            self._callbacks: list[object] = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self, *args, **kwargs) -> None:
            for callback in list(self._callbacks):
                callback(*args, **kwargs)

    class _FakeWorker:
        def __init__(
            self,
            *,
            workflow,
            run_callback,
            dask_client_lifecycle_callback=None,
        ) -> None:
            self.workflow = workflow
            self.run_callback = run_callback
            self.dask_client_lifecycle_callback = dask_client_lifecycle_callback
            self.progress_changed = _FakeSignal()
            self.succeeded = _FakeSignal()
            self.cancelled = _FakeSignal()
            self.failed = _FakeSignal()

        def cancel(self) -> None:
            return None

        def start(self) -> None:
            seen_callbacks.append(self.dask_client_lifecycle_callback)
            assert callable(self.dask_client_lifecycle_callback)

            def _progress_callback(percent: int, message: str) -> None:
                self.progress_changed.emit(percent, message)

            self.run_callback(
                self.workflow,
                _progress_callback,
                self.dask_client_lifecycle_callback,
            )
            self.succeeded.emit()

        def wait(self) -> None:
            return None

    def _lifecycle_callback(event: str, workload: str, backend_mode: str, client):
        lifecycle_events.append((event, workload, backend_mode, client))

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        seen_callbacks.append(dask_client_lifecycle_callback)
        assert callable(dask_client_lifecycle_callback)
        dask_client_lifecycle_callback(
            "started",
            "analysis",
            app_module.DASK_BACKEND_LOCAL_CLUSTER,
            object(),
        )
        progress_callback(55, "running")

    monkeypatch.setattr(
        app_module, "AnalysisExecutionWorker", _FakeWorker, raising=False
    )

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_LOCAL_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
        dask_client_lifecycle_callback=_lifecycle_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert len(seen_callbacks) == 2
    assert len(lifecycle_events) == 1

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1

    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.shown is False
    assert dialog.accepted is True
    assert dialog.rejected is False
    assert dialog.closed is False
    assert dialog.updates[-1] == (55, "running")


def test_run_workflow_with_progress_updates_dashboard_button_from_lifecycle_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )
    browser_urls: list[str] = []
    monkeypatch.setattr(
        app_module.webbrowser,
        "open_new_tab",
        lambda url: browser_urls.append(str(url)) or True,
    )

    class _FakeRelayManager:
        def __init__(self) -> None:
            self.registered_clients: list[tuple[str, str, object]] = []
            self.unregistered_client_ids: list[str] = []
            self.open_dashboard_calls: list[str | None] = []
            self._next_id = 0
            self._active = False

        def register_client(self, *, workload, backend_mode, client):
            self._next_id += 1
            self.registered_clients.append((workload, backend_mode, client))
            self._active = True
            return f"client-{self._next_id}"

        def unregister_client(self, client_id: str) -> None:
            self.unregistered_client_ids.append(client_id)
            self._active = False

        def has_available_client(self, *, workload=None):
            del workload
            return self._active

        def open_dashboard(self, *, workload=None):
            self.open_dashboard_calls.append(workload)
            if not self._active:
                raise ValueError("No active client")
            return "http://127.0.0.1:8787/status?token=relay"

        def shutdown(self) -> None:
            return None

    relay_manager = _FakeRelayManager()
    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: relay_manager,
    )

    lifecycle_events: list[tuple[str, str, str, object]] = []

    def _lifecycle_callback(event: str, workload: str, backend_mode: str, client):
        lifecycle_events.append((event, workload, backend_mode, client))

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        assert callable(dask_client_lifecycle_callback)
        client = object()
        dask_client_lifecycle_callback(
            "started",
            "analysis",
            app_module.DASK_BACKEND_LOCAL_CLUSTER,
            client,
        )
        dialog = fake_dialog_cls.last_instance
        assert dialog is not None
        dialog.dashboard_requested.emit()
        progress_callback(55, "running")
        dask_client_lifecycle_callback(
            "stopped",
            "analysis",
            app_module.DASK_BACKEND_LOCAL_CLUSTER,
            client,
        )

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_SLURM_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
        dask_client_lifecycle_callback=_lifecycle_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert relay_manager.registered_clients
    assert relay_manager.unregistered_client_ids == ["client-1"]
    assert relay_manager.open_dashboard_calls == ["analysis"]
    assert browser_urls == ["http://127.0.0.1:8787/status?token=relay"]
    dialog = fake_dialog_cls.last_instance
    assert dialog is not None
    assert dialog.dashboard_enabled_states[0] == (
        False,
        "No active analysis Dask client is registered.",
    )
    assert (
        True,
        "Open the local Dask dashboard relay for analysis clients.",
    ) in dialog.dashboard_enabled_states
    assert dialog.dashboard_enabled_states[-1] == (
        False,
        "No active analysis Dask client is registered.",
    )
    assert lifecycle_events[0][0] == "started"
    assert lifecycle_events[-1][0] == "stopped"

    app_instance = fake_app_cls.instance()
    assert app_instance is not None
    assert app_instance.quit_calls == 1


def test_run_workflow_with_progress_worker_routes_dashboard_updates_to_gui_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: (
            themed_error_calls.append(
                {"summary": str(summary), "details": str(details)}
            )
        ),
    )
    monkeypatch.setattr(
        app_module,
        "_apply_application_icon",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(app_module, "_display_is_available", lambda: True)

    class _RecordingProgressDialog(app_module.QDialog):
        cancel_requested = app_module.pyqtSignal()
        dashboard_requested = app_module.pyqtSignal()
        last_instance = None

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            type(self).last_instance = self
            self.dashboard_enabled_states: list[tuple[bool, str]] = []
            self.dashboard_thread_matches: list[bool] = []
            self.updates: list[tuple[int, str]] = []

        def update_progress(self, percent: int, message: str) -> None:
            self.updates.append((int(percent), str(message)))

        def set_dashboard_available(
            self, available: bool, tooltip: str | None = None
        ) -> None:
            self.dashboard_thread_matches.append(
                app_module.QThread.currentThread() is self.thread()
            )
            self.dashboard_enabled_states.append((bool(available), str(tooltip or "")))

    class _FakeRelayManager:
        def __init__(self) -> None:
            self.registered_clients: list[tuple[str, str, object]] = []
            self.unregistered_client_ids: list[str] = []

        def register_client(self, *, workload, backend_mode, client):
            self.registered_clients.append((str(workload), str(backend_mode), client))
            return "client-1"

        def unregister_client(self, client_id: str) -> None:
            self.unregistered_client_ids.append(str(client_id))

        def has_available_client(self, *, workload=None):
            del workload
            return bool(self.registered_clients) and not self.unregistered_client_ids

        def open_dashboard(self, *, workload=None):
            del workload
            return "http://127.0.0.1:8787/status?token=relay"

        def shutdown(self) -> None:
            return None

    relay_manager = _FakeRelayManager()
    monkeypatch.setattr(
        app_module,
        "AnalysisExecutionProgressDialog",
        _RecordingProgressDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "get_dashboard_relay_manager",
        lambda: relay_manager,
    )

    def _run_callback(
        workflow,
        progress_callback,
        dask_client_lifecycle_callback=None,
    ):
        del workflow
        assert callable(dask_client_lifecycle_callback)
        client = object()
        dask_client_lifecycle_callback(
            "started",
            "analysis",
            app_module.DASK_BACKEND_LOCAL_CLUSTER,
            client,
        )
        progress_callback(55, "running")
        dask_client_lifecycle_callback(
            "stopped",
            "analysis",
            app_module.DASK_BACKEND_LOCAL_CLUSTER,
            client,
        )

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(
            mode=app_module.DASK_BACKEND_LOCAL_CLUSTER
        )
    )

    ok = app_module.run_workflow_with_progress(
        workflow=workflow,
        run_callback=_run_callback,
    )

    assert ok is True
    assert themed_error_calls == []
    assert relay_manager.registered_clients
    assert relay_manager.unregistered_client_ids == ["client-1"]
    dialog = _RecordingProgressDialog.last_instance
    assert dialog is not None
    assert dialog.dashboard_thread_matches
    assert all(dialog.dashboard_thread_matches)
    assert dialog.dashboard_enabled_states[0] == (
        False,
        "No active analysis Dask client is registered.",
    )
    assert (
        True,
        "Open the local Dask dashboard relay for analysis clients.",
    ) in dialog.dashboard_enabled_states
    assert dialog.dashboard_enabled_states[-1] == (
        False,
        "No active analysis Dask client is registered.",
    )
    dialog.close()


def test_launch_gui_shuts_down_dashboard_relay_on_standalone_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"ensure": 0}
    shutdown_calls: list[str] = []

    class _FakeApplication:
        _instance = None

        def __init__(self, _argv) -> None:
            type(self)._instance = self
            self.quit_calls = 0

        @classmethod
        def instance(cls):
            return cls._instance

        def quit(self) -> None:
            self.quit_calls += 1

    class _FakeQDialog:
        class DialogCode:
            Accepted = 1
            Rejected = 0

    class _FakeSetupDialog:
        def __init__(self, initial) -> None:
            self.result_config = initial

        def exec(self) -> int:
            return _FakeQDialog.DialogCode.Accepted

    class _FakeAnalysisDialog:
        def __init__(self, initial) -> None:
            self.result_config = initial

        def exec(self) -> int:
            return _FakeQDialog.DialogCode.Accepted

    class _FakeRelayManager:
        def shutdown(self) -> None:
            shutdown_calls.append("shutdown")

    monkeypatch.setattr(app_module, "HAS_PYQT6", True)
    monkeypatch.setattr(app_module, "_display_is_available", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_ensure_clearex_settings_directory",
        lambda path=None: (
            calls.__setitem__("ensure", calls["ensure"] + 1) or Path("/tmp/.clearex")
        ),
    )
    monkeypatch.setattr(
        app_module,
        "_load_last_used_dask_backend_config",
        lambda settings_path=None: None,
    )
    monkeypatch.setattr(
        app_module,
        "_load_last_used_zarr_save_config",
        lambda settings_path=None: None,
    )
    monkeypatch.setattr(
        app_module,
        "_apply_application_icon",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "_show_startup_splash",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(app_module, "QApplication", _FakeApplication, raising=False)
    monkeypatch.setattr(app_module, "QDialog", _FakeQDialog, raising=False)
    monkeypatch.setattr(
        app_module,
        "ClearExSetupDialog",
        _FakeSetupDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AnalysisSelectionDialog",
        _FakeAnalysisDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "_dashboard_relay_manager_singleton",
        _FakeRelayManager(),
        raising=False,
    )

    selected = app_module.launch_gui(initial=app_module.WorkflowConfig())

    assert selected is not None
    assert calls["ensure"] == 1
    assert shutdown_calls == ["shutdown"]


def test_ensure_clearex_settings_directory_creates_target(tmp_path) -> None:
    target = tmp_path / "nested" / ".clearex"
    assert not target.exists()

    created = app_module._ensure_clearex_settings_directory(target)

    assert created == target
    assert target.is_dir()


def test_persisted_dask_backend_round_trip(tmp_path) -> None:
    settings_path = tmp_path / ".clearex" / "dask_backend_settings.json"
    config = app_module.DaskBackendConfig(
        mode=app_module.DASK_BACKEND_SLURM_RUNNER,
        local_cluster=app_module.LocalClusterConfig(
            n_workers=6,
            threads_per_worker=2,
            memory_limit="16GB",
            local_directory="/tmp/local",
        ),
        slurm_runner=app_module.SlurmRunnerConfig(
            scheduler_file="/tmp/scheduler.json",
            wait_for_workers=3,
        ),
        slurm_cluster=app_module.SlurmClusterConfig(
            workers=2,
            mail_user="user@example.com",
            queue="gpu",
        ),
    )

    saved = app_module._save_last_used_dask_backend_config(
        config,
        settings_path=settings_path,
    )
    loaded = app_module._load_last_used_dask_backend_config(
        settings_path=settings_path,
    )

    assert saved is True
    assert loaded == config


def test_persisted_zarr_save_round_trip(tmp_path) -> None:
    settings_path = tmp_path / ".clearex" / "zarr_save_settings.json"
    config = app_module.ZarrSaveConfig(
        chunks_ptczyx=(2, 1, 1, 128, 192, 256),
        pyramid_ptczyx=((1, 2), (1,), (1,), (1, 2), (1, 2, 4), (1, 2, 4)),
    )

    saved = app_module._save_last_used_zarr_save_config(
        config,
        settings_path=settings_path,
    )
    loaded = app_module._load_last_used_zarr_save_config(
        settings_path=settings_path,
    )

    assert saved is True
    assert loaded == config


def test_load_persisted_backend_returns_none_for_empty_or_invalid_json(
    tmp_path,
) -> None:
    settings_path = tmp_path / ".clearex" / "dask_backend_settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("  \n", encoding="utf-8")

    assert (
        app_module._load_last_used_dask_backend_config(settings_path=settings_path)
        is None
    )

    settings_path.write_text("{not-json}", encoding="utf-8")
    assert (
        app_module._load_last_used_dask_backend_config(settings_path=settings_path)
        is None
    )


def test_load_persisted_zarr_save_returns_none_for_empty_or_invalid_json(
    tmp_path,
) -> None:
    settings_path = tmp_path / ".clearex" / "zarr_save_settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("  \n", encoding="utf-8")

    assert (
        app_module._load_last_used_zarr_save_config(settings_path=settings_path) is None
    )

    settings_path.write_text("{not-json}", encoding="utf-8")
    assert (
        app_module._load_last_used_zarr_save_config(settings_path=settings_path) is None
    )


def test_launch_gui_applies_persisted_backend_defaults(monkeypatch, tmp_path) -> None:
    calls = {"ensure": 0}
    captured: dict[str, app_module.WorkflowConfig] = {}
    persisted_backend = app_module.DaskBackendConfig(
        mode=app_module.DASK_BACKEND_SLURM_RUNNER,
        slurm_runner=app_module.SlurmRunnerConfig(
            scheduler_file="/tmp/persisted-scheduler.json"
        ),
    )
    persisted_zarr = app_module.ZarrSaveConfig(
        chunks_ptczyx=(1, 1, 1, 64, 128, 128),
        pyramid_ptczyx=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    class _FakeApplication:
        _instance = None

        def __init__(self, _argv) -> None:
            type(self)._instance = self
            self.quit_calls = 0

        @classmethod
        def instance(cls):
            return cls._instance

        def quit(self) -> None:
            self.quit_calls += 1

    class _FakeQDialog:
        class DialogCode:
            Accepted = 1
            Rejected = 0

    class _FakeSetupDialog:
        def __init__(self, initial) -> None:
            captured["setup_initial"] = initial
            self.result_config = initial

        def exec(self) -> int:
            return _FakeQDialog.DialogCode.Accepted

    class _FakeAnalysisDialog:
        def __init__(self, initial) -> None:
            self.result_config = initial

        def exec(self) -> int:
            return _FakeQDialog.DialogCode.Accepted

    def _ensure_override(path=None):
        del path
        calls["ensure"] += 1
        settings_dir = tmp_path / ".clearex"
        settings_dir.mkdir(parents=True, exist_ok=True)
        return settings_dir

    monkeypatch.setattr(app_module, "HAS_PYQT6", True)
    monkeypatch.setattr(app_module, "_display_is_available", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_ensure_clearex_settings_directory",
        _ensure_override,
    )
    monkeypatch.setattr(
        app_module,
        "_load_last_used_dask_backend_config",
        lambda settings_path=None: persisted_backend,
    )
    monkeypatch.setattr(
        app_module,
        "_load_last_used_zarr_save_config",
        lambda settings_path=None: persisted_zarr,
    )
    monkeypatch.setattr(
        app_module,
        "_apply_application_icon",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "_show_startup_splash",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(app_module, "QApplication", _FakeApplication, raising=False)
    monkeypatch.setattr(app_module, "QDialog", _FakeQDialog, raising=False)
    monkeypatch.setattr(
        app_module,
        "ClearExSetupDialog",
        _FakeSetupDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AnalysisSelectionDialog",
        _FakeAnalysisDialog,
        raising=False,
    )

    initial = app_module.WorkflowConfig()
    selected = app_module.launch_gui(initial=initial)

    assert calls["ensure"] == 1
    assert captured["setup_initial"].dask_backend == persisted_backend
    assert captured["setup_initial"].zarr_save == persisted_zarr
    assert selected is not None


def test_launch_gui_persists_reset_state_after_successful_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _create_gui_analysis_store(tmp_path)
    selected_workflow = app_module.WorkflowConfig(
        file=str(store_path),
        analysis_selected_experiment_path="/tmp/cell_001/experiment.yml",
        flatfield=True,
        fusion=True,
        analysis_parameters={
            "flatfield": {"force_rerun": True},
            "fusion": {"force_rerun": True},
        },
    )
    captured: dict[str, app_module.WorkflowConfig] = {}
    lifecycle_callback = object()

    class _FakeApplication:
        _instance = None

        def __init__(self, _argv) -> None:
            type(self)._instance = self
            self.quit_calls = 0

        @classmethod
        def instance(cls):
            return cls._instance

        def quit(self) -> None:
            self.quit_calls += 1

    class _FakeQDialog:
        class DialogCode:
            Accepted = 1
            Rejected = 0

    class _FakeSetupDialog:
        def __init__(self, initial) -> None:
            self.result_config = initial

        def exec(self) -> int:
            return _FakeQDialog.DialogCode.Accepted

    class _FakeAnalysisDialog:
        call_count = 0

        def __init__(self, initial) -> None:
            type(self).call_count += 1
            self.result_config = selected_workflow
            if type(self).call_count == 1:
                captured["first_initial"] = initial
            else:
                captured["second_initial"] = initial

        def exec(self) -> int:
            if type(self).call_count == 1:
                return _FakeQDialog.DialogCode.Accepted
            self.result_config = None
            return _FakeQDialog.DialogCode.Rejected

    monkeypatch.setattr(app_module, "HAS_PYQT6", True)
    monkeypatch.setattr(app_module, "_display_is_available", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_ensure_clearex_settings_directory",
        lambda path=None: tmp_path / ".clearex",
    )
    monkeypatch.setattr(
        app_module,
        "_load_last_used_dask_backend_config",
        lambda settings_path=None: None,
    )
    monkeypatch.setattr(
        app_module,
        "_load_last_used_zarr_save_config",
        lambda settings_path=None: None,
    )
    monkeypatch.setattr(
        app_module,
        "_apply_application_icon",
        lambda _app: None,
        raising=False,
    )
    monkeypatch.setattr(app_module, "_show_startup_splash", lambda _app: None)
    monkeypatch.setattr(app_module, "QApplication", _FakeApplication, raising=False)
    monkeypatch.setattr(app_module, "QDialog", _FakeQDialog, raising=False)
    monkeypatch.setattr(
        app_module,
        "ClearExSetupDialog",
        _FakeSetupDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AnalysisSelectionDialog",
        _FakeAnalysisDialog,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "run_workflow_with_progress",
        lambda workflow, run_callback, dask_client_lifecycle_callback=None: (
            captured.__setitem__(
                "lifecycle_callback",
                dask_client_lifecycle_callback,
            )
            or True
        ),
    )
    monkeypatch.setattr(
        app_module,
        "_persist_reset_analysis_gui_state_for_workflow",
        lambda workflow: captured.__setitem__("reset_workflow", workflow),
    )

    result = app_module.launch_gui(
        initial=selected_workflow,
        run_callback=lambda workflow, progress_callback, dask_client_lifecycle_callback=None: None,
        dask_client_lifecycle_callback=lifecycle_callback,
    )

    assert result is None
    assert captured["lifecycle_callback"] is lifecycle_callback
    assert captured["reset_workflow"].flatfield is False
    assert captured["reset_workflow"].fusion is False
    assert (
        captured["reset_workflow"].analysis_parameters["flatfield"]["force_rerun"]
        is False
    )
    assert (
        captured["reset_workflow"].analysis_parameters["fusion"]["force_rerun"] is False
    )


def test_build_input_source_options_includes_existing_store_outputs() -> None:
    options = app_module._build_input_source_options(
        operation_name="visualization",
        selected_order=["visualization"],
        operation_key_order=(
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "fusion",
            "visualization",
        ),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shear Transform",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "fusion": "Fusion",
            "visualization": "Visualization",
        },
        operation_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
            "shear_transform": "clearex/runtime_cache/results/shear_transform/latest/data",
            "usegment3d": "clearex/runtime_cache/results/usegment3d/latest/data",
            "registration": "clearex/results/registration/latest",
            "fusion": "clearex/runtime_cache/results/fusion/latest/data",
        },
        available_store_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
        },
    )

    assert options[0] == ("data", "Raw data (data)")
    assert (
        "flatfield",
        "Flatfield Correction output (clearex/runtime_cache/results/flatfield/latest/data) [existing]",
    ) in options


def test_build_input_source_options_deduplicates_existing_and_selected() -> None:
    options = app_module._build_input_source_options(
        operation_name="visualization",
        selected_order=["flatfield", "visualization"],
        operation_key_order=(
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "fusion",
            "visualization",
        ),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shear Transform",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "fusion": "Fusion",
            "visualization": "Visualization",
        },
        operation_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
            "shear_transform": "clearex/runtime_cache/results/shear_transform/latest/data",
            "usegment3d": "clearex/runtime_cache/results/usegment3d/latest/data",
            "registration": "clearex/results/registration/latest",
            "fusion": "clearex/runtime_cache/results/fusion/latest/data",
        },
        available_store_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
        },
    )

    values = [value for value, _label in options]
    assert values.count("flatfield") == 1
    assert (
        "flatfield",
        "Flatfield Correction output (clearex/runtime_cache/results/flatfield/latest/data) [scheduled]",
    ) in options


def test_build_input_source_options_includes_scheduled_upstream_outputs() -> None:
    options = app_module._build_input_source_options(
        operation_name="visualization",
        selected_order=["deconvolution", "visualization"],
        operation_key_order=(
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "fusion",
            "visualization",
        ),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shear Transform",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "fusion": "Fusion",
            "visualization": "Visualization",
        },
        operation_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
            "shear_transform": "clearex/runtime_cache/results/shear_transform/latest/data",
            "usegment3d": "clearex/runtime_cache/results/usegment3d/latest/data",
            "registration": "clearex/results/registration/latest",
            "fusion": "clearex/runtime_cache/results/fusion/latest/data",
        },
        available_store_output_components={
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
        },
    )

    assert (
        "deconvolution",
        "Deconvolution output (clearex/runtime_cache/results/deconvolution/latest/data) [scheduled]",
    ) in options


def test_build_input_source_options_allows_visualization_for_render_movie() -> None:
    options = app_module._build_input_source_options(
        operation_name="render_movie",
        selected_order=["visualization", "render_movie"],
        operation_key_order=(
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
            "mip_export",
        ),
        operation_labels={
            "visualization": "Napari",
            "render_movie": "Render Movie",
        },
        operation_output_components={
            "visualization": "clearex/results/visualization/latest",
            "render_movie": "clearex/results/render_movie/latest",
        },
        available_store_output_components={},
    )

    assert ("data", "Raw data (data)") not in options
    assert (
        "visualization",
        "Napari output (clearex/results/visualization/latest) [scheduled]",
    ) in options


def test_build_input_source_options_allows_render_movie_for_compile_movie() -> None:
    options = app_module._build_input_source_options(
        operation_name="compile_movie",
        selected_order=["render_movie", "compile_movie"],
        operation_key_order=(
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
            "mip_export",
        ),
        operation_labels={
            "render_movie": "Render Movie",
            "compile_movie": "Compile Movie",
        },
        operation_output_components={
            "render_movie": "clearex/results/render_movie/latest",
            "compile_movie": "clearex/results/compile_movie/latest",
        },
        available_store_output_components={},
    )

    assert ("data", "Raw data (data)") not in options
    assert (
        "render_movie",
        "Render Movie output (clearex/results/render_movie/latest) [scheduled]",
    ) in options


def test_build_input_source_options_allows_registration_for_fusion() -> None:
    options = app_module._build_input_source_options(
        operation_name="fusion",
        selected_order=["registration", "fusion"],
        operation_key_order=(
            "flatfield",
            "deconvolution",
            "shear_transform",
            "registration",
            "fusion",
            "visualization",
        ),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shear Transform",
            "registration": "Registration",
            "fusion": "Fusion",
            "visualization": "Visualization",
        },
        operation_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
            "shear_transform": "clearex/runtime_cache/results/shear_transform/latest/data",
            "registration": "clearex/results/registration/latest",
            "fusion": "clearex/runtime_cache/results/fusion/latest/data",
        },
        available_store_output_components={
            "registration": "clearex/results/registration/latest",
        },
    )

    assert (
        "registration",
        "Registration output (clearex/results/registration/latest) [scheduled]",
    ) in options


def test_build_visualization_volume_layer_component_options_include_existing_outputs() -> (
    None
):
    options = app_module._build_visualization_volume_layer_component_options(
        selected_order=("visualization",),
        operation_key_order=(
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "visualization",
        ),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shear Transform",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "visualization": "Visualization",
        },
        available_store_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
        },
    )

    assert options[0] == ("data", "Raw data (data)")
    assert (
        "clearex/runtime_cache/results/flatfield/latest/data",
        "Flatfield Correction output (clearex/runtime_cache/results/flatfield/latest/data) [existing]",
    ) in options
    assert (
        "clearex/runtime_cache/results/deconvolution/latest/data",
        "Deconvolution output (clearex/runtime_cache/results/deconvolution/latest/data) [existing]",
    ) in options


def test_build_visualization_volume_layer_component_options_include_scheduled_outputs() -> (
    None
):
    options = app_module._build_visualization_volume_layer_component_options(
        selected_order=("flatfield", "visualization"),
        operation_key_order=(
            "flatfield",
            "deconvolution",
            "shear_transform",
            "particle_detection",
            "usegment3d",
            "registration",
            "visualization",
        ),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "shear_transform": "Shear Transform",
            "particle_detection": "Particle Detection",
            "usegment3d": "uSegment3D",
            "registration": "Registration",
            "visualization": "Visualization",
        },
        available_store_output_components={},
    )

    assert (
        "clearex/runtime_cache/results/flatfield/latest/data",
        "Flatfield Correction output (clearex/runtime_cache/results/flatfield/latest/data) [scheduled]",
    ) in options


def test_build_visualization_volume_layer_component_options_deduplicates_components() -> (
    None
):
    options = app_module._build_visualization_volume_layer_component_options(
        selected_order=("flatfield", "visualization"),
        operation_key_order=("flatfield", "deconvolution", "visualization"),
        operation_labels={
            "flatfield": "Flatfield Correction",
            "deconvolution": "Deconvolution",
            "visualization": "Visualization",
        },
        available_store_output_components={
            "flatfield": "results/shared/latest/data",
            "deconvolution": "results/shared/latest/data",
        },
    )

    components = [component for component, _ in options]
    assert components.count("results/shared/latest/data") == 1


def test_available_visualization_blending_options_include_expected_values() -> None:
    options = app_module._available_visualization_blending_options()

    assert "translucent" in options
    assert "additive" in options
    assert "opaque" in options


def test_available_visualization_rendering_options_include_expected_values() -> None:
    options = app_module._available_visualization_rendering_options()

    assert "mip" in options
    assert "attenuated_mip" in options
    assert "iso" in options


def test_available_visualization_colormap_options_include_expected_values() -> None:
    options = app_module._available_visualization_colormap_options()

    assert "gray" in options
    assert "green" in options
    assert "magenta" in options


def test_discover_available_operation_output_components(monkeypatch) -> None:
    class _FakeRoot:
        def __getitem__(self, key):
            if key == "clearex/runtime_cache/results/flatfield/latest/data":
                return object()
            raise KeyError(key)

    monkeypatch.setattr(app_module, "is_zarr_store_path", lambda _path: True)
    monkeypatch.setattr(
        app_module.zarr,
        "open_group",
        lambda _path, mode="r": _FakeRoot(),
    )

    discovered = app_module._discover_available_operation_output_components(
        store_path="/tmp/fake_store.zarr",
        operation_output_components={
            "flatfield": "clearex/runtime_cache/results/flatfield/latest/data",
            "deconvolution": "clearex/runtime_cache/results/deconvolution/latest/data",
        },
    )

    assert discovered == {
        "flatfield": "clearex/runtime_cache/results/flatfield/latest/data"
    }


def test_zarr_component_exists_in_root_uses_membership_semantics() -> None:
    class _ImplicitGroupRoot:
        def __contains__(self, key: object) -> bool:
            return str(key) == "clearex/runtime_cache/results/flatfield/latest/data"

        def __getitem__(self, key: object) -> object:
            del key
            return object()

    root = _ImplicitGroupRoot()
    assert (
        app_module._zarr_component_exists_in_root(
            root,
            "clearex/runtime_cache/results/flatfield/latest/data",
        )
        is True
    )
    assert (
        app_module._zarr_component_exists_in_root(
            root,
            "clearex/runtime_cache/results/deconvolution/latest/data",
        )
        is False
    )


def test_particle_overlay_available_when_particle_detection_runs_first() -> None:
    available = app_module._particle_overlay_available_for_visualization(
        selected_order=["particle_detection", "visualization"],
        has_particle_detection_output=False,
    )

    assert available is True


def test_particle_overlay_unavailable_when_no_outputs_or_history() -> None:
    available = app_module._particle_overlay_available_for_visualization(
        selected_order=["visualization", "particle_detection"],
        has_particle_detection_output=False,
    )

    assert available is False


def test_particle_overlay_available_with_existing_detections() -> None:
    from_store = app_module._particle_overlay_available_for_visualization(
        selected_order=["visualization"],
        has_particle_detection_output=True,
    )

    assert from_store is True


def test_sync_visualization_volume_layers_from_input_source_updates_primary_layer() -> (
    None
):
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    class _FakeCombo:
        def __init__(self, data: object) -> None:
            self._data = data

        def currentData(self) -> object:
            return self._data

    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    dialog._operation_input_combos = {"visualization": _FakeCombo("shear_transform")}
    dialog._visualization_volume_layers = [
        {"component": "data", "layer_type": "image"},
        {
            "component": "clearex/runtime_cache/results/deconvolution/latest/data",
            "layer_type": "image",
        },
    ]
    dialog._normalize_visualization_volume_layers = lambda rows: list(rows)
    dialog._default_visualization_volume_layers = lambda: [
        {"component": "data", "layer_type": "image"}
    ]
    refresh_calls = {"count": 0}
    dialog._refresh_visualization_volume_layers_summary = lambda: (
        refresh_calls.__setitem__("count", refresh_calls["count"] + 1)
    )

    app_module.AnalysisSelectionDialog._sync_visualization_volume_layers_from_input_source(
        dialog, refresh_summary=True
    )

    assert (
        dialog._visualization_volume_layers[0]["component"]
        == "clearex/runtime_cache/results/shear_transform/latest/data"
    )
    assert dialog._visualization_volume_layers[1]["component"] == (
        "clearex/runtime_cache/results/deconvolution/latest/data"
    )
    assert refresh_calls["count"] == 1


def test_collect_visualization_parameters_syncs_combo_with_primary_layer() -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    class _FakeCombo:
        def __init__(self, data: object) -> None:
            self._data = data

        def currentData(self) -> object:
            return self._data

    class _FakeCheckbox:
        def __init__(self, checked: bool) -> None:
            self._checked = bool(checked)

        def isChecked(self) -> bool:
            return bool(self._checked)

    class _FakeSpin:
        def __init__(self, value: int) -> None:
            self._value = int(value)

        def value(self) -> int:
            return int(self._value)

    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    dialog._operation_input_combos = {"visualization": _FakeCombo("shear_transform")}
    dialog._visualization_volume_layers = [{"component": "data", "layer_type": "image"}]
    dialog._visualization_defaults = {
        "memory_overhead_factor": 1.0,
        "launch_mode": "auto",
        "capture_keyframes": True,
        "keyframe_manifest_path": "",
        "particle_detection_component": "clearex/results/particle_detection/latest/detections",
    }
    dialog._visualization_keyframe_layer_overrides = []
    dialog._normalize_visualization_volume_layers = lambda rows: list(rows)
    dialog._default_visualization_volume_layers = lambda: [
        {"component": "data", "layer_type": "image"}
    ]
    dialog._refresh_visualization_volume_layers_summary = lambda: None
    dialog._is_particle_overlay_available = lambda: False
    dialog._visualization_show_all_positions_checkbox = _FakeCheckbox(False)
    dialog._visualization_position_spin = _FakeSpin(0)
    dialog._visualization_multiscale_checkbox = _FakeCheckbox(False)
    dialog._visualization_3d_checkbox = _FakeCheckbox(True)
    dialog._visualization_require_gpu_checkbox = _FakeCheckbox(False)
    dialog._visualization_overlay_points_checkbox = _FakeCheckbox(False)

    params = app_module.AnalysisSelectionDialog._collect_visualization_parameters(
        dialog
    )

    assert (
        params["input_source"]
        == "clearex/runtime_cache/results/shear_transform/latest/data"
    )
    assert (
        params["volume_layers"][0]["component"]
        == "clearex/runtime_cache/results/shear_transform/latest/data"
    )


def test_collect_volume_export_parameters_uses_widget_values() -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    class _FakeCombo:
        def __init__(self, data: object) -> None:
            self._data = data

        def currentData(self) -> object:
            return self._data

    class _FakeSpin:
        def __init__(self, value: int) -> None:
            self._value = int(value)

        def value(self) -> int:
            return int(self._value)

    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    dialog._volume_export_defaults = {"memory_overhead_factor": 3.25}
    dialog._volume_export_scope_combo = _FakeCombo("all_indices")
    dialog._volume_export_t_spin = _FakeSpin(4)
    dialog._volume_export_p_spin = _FakeSpin(2)
    dialog._volume_export_c_spin = _FakeSpin(1)
    dialog._volume_export_resolution_level_spin = _FakeSpin(6)
    dialog._volume_export_format_combo = _FakeCombo("ome-tiff")
    dialog._volume_export_tiff_layout_combo = _FakeCombo("per_volume_files")

    params = app_module.AnalysisSelectionDialog._collect_volume_export_parameters(
        dialog
    )

    assert params == {
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 3.25,
        "export_scope": "all_indices",
        "t_index": 4,
        "p_index": 2,
        "c_index": 1,
        "resolution_level": 6,
        "export_format": "ome-tiff",
        "tiff_file_layout": "per_volume_files",
    }


def test_volume_export_parameter_enablement_tracks_scope_and_format() -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    class _FakeCombo:
        def __init__(self, data: object) -> None:
            self._data = data
            self.enabled = True

        def currentData(self) -> object:
            return self._data

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = bool(enabled)

    class _FakeSpin:
        def __init__(self) -> None:
            self.enabled = True

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = bool(enabled)

    class _FakeCheckbox:
        def __init__(self, checked: bool) -> None:
            self._checked = bool(checked)

        def isChecked(self) -> bool:
            return bool(self._checked)

    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    dialog._operation_checkboxes = {"volume_export": _FakeCheckbox(True)}
    dialog._volume_export_scope_combo = _FakeCombo("current_selection")
    dialog._volume_export_t_spin = _FakeSpin()
    dialog._volume_export_p_spin = _FakeSpin()
    dialog._volume_export_c_spin = _FakeSpin()
    dialog._volume_export_resolution_level_spin = _FakeSpin()
    dialog._volume_export_format_combo = _FakeCombo("ome-zarr")
    dialog._volume_export_tiff_layout_combo = _FakeCombo("single_file")

    app_module.AnalysisSelectionDialog._set_volume_export_parameter_enabled_state(
        dialog
    )
    assert dialog._volume_export_scope_combo.enabled is True
    assert dialog._volume_export_t_spin.enabled is True
    assert dialog._volume_export_p_spin.enabled is True
    assert dialog._volume_export_c_spin.enabled is True
    assert dialog._volume_export_resolution_level_spin.enabled is True
    assert dialog._volume_export_format_combo.enabled is True
    assert dialog._volume_export_tiff_layout_combo.enabled is False

    dialog._volume_export_scope_combo = _FakeCombo("all_indices")
    dialog._volume_export_format_combo = _FakeCombo("ome-tiff")
    app_module.AnalysisSelectionDialog._set_volume_export_parameter_enabled_state(
        dialog
    )
    assert dialog._volume_export_t_spin.enabled is False
    assert dialog._volume_export_p_spin.enabled is False
    assert dialog._volume_export_c_spin.enabled is False
    assert dialog._volume_export_tiff_layout_combo.enabled is True


def test_on_run_propagates_display_pyramid_flag_into_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    class _FakeCheckbox:
        def __init__(self, checked: bool) -> None:
            self._checked = bool(checked)

        def isChecked(self) -> bool:
            return bool(self._checked)

    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    base_config = app_module.WorkflowConfig(file="/tmp/data_store.zarr")
    dialog._base_config = base_config
    dialog._analysis_targets = ()
    dialog._analysis_apply_to_all_checkbox = None
    dialog._dask_backend_config = base_config.dask_backend
    dialog._refresh_operation_provenance_statuses = lambda: None
    dialog._current_analysis_target = lambda: None
    dialog._persist_analysis_gui_state_for_target = lambda _target: None
    dialog._validate_selected_analysis_dependencies = lambda _params: ()
    dialog._selected_operations_in_sequence = lambda: ["display_pyramid"]
    dialog._set_status = lambda _message: None

    selected = {
        operation_name: _FakeCheckbox(False)
        for operation_name in dialog._OPERATION_KEYS
    }
    selected["display_pyramid"] = _FakeCheckbox(True)
    dialog._operation_checkboxes = selected

    normalized_defaults = app_module.normalize_analysis_operation_parameters(
        base_config.analysis_parameters
    )
    dialog._collect_operation_parameters = lambda operation_name: dict(
        normalized_defaults.get(str(operation_name), {})
    )

    accept_state = {"called": False}
    dialog.accept = lambda: accept_state.__setitem__("called", True)

    monkeypatch.setattr(
        app_module,
        "_save_last_used_dask_backend_config",
        lambda _config: None,
    )

    app_module.AnalysisSelectionDialog._on_run(dialog)

    assert accept_state["called"] is True
    assert dialog.result_config.display_pyramid is True
    assert dialog.result_config.visualization is False


def test_validate_selected_analysis_dependencies_rejects_later_scheduled_producer() -> (
    None
):
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    dialog._base_config = SimpleNamespace(file="")
    dialog._selected_operations_in_sequence = lambda: ["visualization", "flatfield"]
    dialog._discover_store_output_components = lambda: {}

    issues = (
        app_module.AnalysisSelectionDialog._validate_selected_analysis_dependencies(
            dialog,
            {
                "visualization": {"input_source": "flatfield"},
                "flatfield": {"input_source": "data"},
            },
        )
    )

    assert issues
    assert issues[0].reason == "producer_scheduled_after_consumer"


def test_shear_degree_to_coefficient_and_back_round_trip() -> None:
    for angle in (-70.0, -35.0, -5.5, 0.0, 8.25, 35.0, 70.0):
        coefficient = app_module._shear_degrees_to_coefficient(angle)
        restored = app_module._shear_coefficient_to_degrees(coefficient)
        assert math.isclose(restored, angle, rel_tol=0.0, abs_tol=1e-9)


def test_shear_degree_conversion_matches_expected_tangent() -> None:
    coefficient = app_module._shear_degrees_to_coefficient(35.0)
    assert math.isclose(
        coefficient,
        math.tan(math.radians(35.0)),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_analysis_selection_dialog_uses_napari_and_visualization_labels() -> None:
    if not app_module.HAS_PYQT6:
        return

    dialog_cls = app_module.AnalysisSelectionDialog
    assert dialog_cls._OPERATION_LABELS["visualization"] == "Napari"
    assert dialog_cls._OPERATION_LABELS["render_movie"] == "Render Movie"
    assert dialog_cls._OPERATION_LABELS["compile_movie"] == "Compile Movie"
    assert dialog_cls._OPERATION_LABELS["volume_export"] == "Volume Export"
    assert dialog_cls._OPERATION_LABELS["display_pyramid"] == "Pyramidal Downsampling"
    assert (
        "Preprocessing",
        (
            "flatfield",
            "deconvolution",
            "shear_transform",
            "registration",
            "fusion",
            "display_pyramid",
        ),
    ) in dialog_cls._OPERATION_TABS
    assert (
        "Visualization",
        (
            "visualization",
            "render_movie",
            "compile_movie",
            "volume_export",
            "mip_export",
        ),
    ) in dialog_cls._OPERATION_TABS
    assert (
        dialog_cls._OPERATION_OUTPUT_COMPONENTS["volume_export"]
        == "clearex/results/volume_export/latest"
    )


def test_analysis_selection_dialog_themes_rounded_scroll_surfaces() -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    apply_theme_source = inspect.getsource(
        app_module.AnalysisSelectionDialog._apply_theme
    )
    build_ui_source = inspect.getsource(app_module.AnalysisSelectionDialog._build_ui)
    usegment_source = inspect.getsource(
        app_module.AnalysisSelectionDialog._build_usegment3d_parameter_rows
    )

    assert "QScrollArea#operationPanelScroll" in apply_theme_source
    assert "QWidget#operationPanelViewport" in apply_theme_source
    assert "QStackedWidget#operationPanelStack" in apply_theme_source
    assert "QWidget#usegment3dChannelViewport" in apply_theme_source
    assert 'self._operation_panel_stack.setObjectName("operationPanelStack")' in (
        build_ui_source
    )
    assert 'scroll_object_name="operationPanelScroll"' in build_ui_source
    assert 'viewport_object_name="usegment3dChannelViewport"' in usegment_source


def test_visualization_popup_tables_use_uniform_widget_rows() -> None:
    if not hasattr(app_module, "AnalysisSelectionDialog"):
        return

    volume_layers_source = inspect.getsource(
        app_module.AnalysisSelectionDialog._open_visualization_volume_layers_dialog
    )
    layer_table_source = inspect.getsource(
        app_module.AnalysisSelectionDialog._open_visualization_layer_table_dialog
    )
    popup_stylesheet_source = inspect.getsource(app_module._popup_dialog_stylesheet)

    assert (
        "table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)"
        in volume_layers_source
    )
    assert (
        "header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)"
        in volume_layers_source
    )
    assert "table.setColumnWidth(0, 140)" in volume_layers_source
    assert 'placeholder_text="Optional name"' in volume_layers_source
    assert 'placeholder_text="Auto"' in volume_layers_source
    assert "line_edit.setPlaceholderText(str(placeholder_text))" in layer_table_source
    assert "table.setColumnWidth(2, 150)" in layer_table_source
    assert "QComboBox QLineEdit {" in popup_stylesheet_source


def test_analysis_selection_dialog_detect_local_gpu_available(monkeypatch) -> None:
    if not app_module.HAS_PYQT6:
        return

    class _Recommendation:
        def __init__(self, detected_gpu_count: int) -> None:
            self.detected_gpu_count = int(detected_gpu_count)

    monkeypatch.setattr(
        app_module,
        "recommend_local_cluster_config",
        lambda: _Recommendation(detected_gpu_count=0),
    )
    monkeypatch.delenv("VGL_DISPLAY", raising=False)
    assert app_module.AnalysisSelectionDialog._detect_local_gpu_available() is False

    monkeypatch.setattr(
        app_module,
        "recommend_local_cluster_config",
        lambda: _Recommendation(detected_gpu_count=2),
    )
    assert app_module.AnalysisSelectionDialog._detect_local_gpu_available() is True


def test_analysis_selection_dialog_detect_local_gpu_available_with_virtualgl_hint(
    monkeypatch,
) -> None:
    if not app_module.HAS_PYQT6:
        return

    class _Recommendation:
        def __init__(self, detected_gpu_count: int) -> None:
            self.detected_gpu_count = int(detected_gpu_count)

    monkeypatch.setattr(
        app_module,
        "recommend_local_cluster_config",
        lambda: _Recommendation(detected_gpu_count=0),
    )
    monkeypatch.setenv("VGL_DISPLAY", "EGL")
    assert app_module.AnalysisSelectionDialog._detect_local_gpu_available() is True
