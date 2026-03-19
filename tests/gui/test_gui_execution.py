#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

import math
import threading
from pathlib import Path

import clearex.gui.app as app_module
from clearex.io.experiment import NavigateChannel, NavigateExperiment
import pytest


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
            self.shown = False
            self.accepted = False
            self.rejected = False
            self.closed = False
            self.cancel_requested = _FakeSignal()

        def update_progress(self, percent: int, message: str) -> None:
            self.updates.append((int(percent), str(message)))

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
    monkeypatch.setattr(app_module, "_apply_application_icon", lambda _app: None)
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
        "_has_complete_configured_canonical_store",
        lambda store_path, zarr_save: Path(store_path).expanduser().resolve()
        == requests[2].target_store.resolve(),
    )

    selected_request, pending_requests, ready_requests = (
        app_module._plan_experiment_store_materialization(
            requests,
            selected_experiment_path=second,
            zarr_save=app_module.ZarrSaveConfig(),
        )
    )

    assert selected_request.experiment_path == second.resolve()
    assert [request.experiment_path for request in pending_requests] == [
        first.resolve(),
        second.resolve(),
    ]
    assert [request.experiment_path for request in ready_requests] == [third.resolve()]


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
    )

    reset = app_module._reset_analysis_selection_for_next_run(workflow)

    assert reset.file == "/tmp/cell_002/data_store.zarr"
    assert reset.analysis_targets == workflow.analysis_targets
    assert (
        reset.analysis_selected_experiment_path
        == "/tmp/cell_002/experiment.yml"
    )
    assert reset.analysis_apply_to_all is True
    assert reset.flatfield is False
    assert reset.registration is False


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


def test_run_workflow_with_progress_slurm_executes_callback_on_main_thread(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: themed_error_calls.append(
            {"summary": str(summary), "details": str(details)}
        ),
    )
    seen_threads: list[threading.Thread] = []

    def _run_callback(workflow, progress_callback):
        del workflow
        seen_threads.append(threading.current_thread())
        progress_callback(35, "SLURM setup")
        progress_callback(80, "running")

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(mode=app_module.DASK_BACKEND_SLURM_CLUSTER)
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


def test_run_workflow_with_progress_slurm_batches_all_selected_experiments(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: themed_error_calls.append(
            {"summary": str(summary), "details": str(details)}
        ),
    )
    executed_files: list[str] = []

    def _run_callback(workflow, progress_callback):
        executed_files.append(str(workflow.file))
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


def test_run_workflow_with_progress_slurm_shows_error_dialog_on_failure(
    monkeypatch,
) -> None:
    fake_app_cls, fake_dialog_cls = _install_fake_gui_runtime(monkeypatch)
    themed_error_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        app_module,
        "_show_themed_error_dialog",
        lambda _parent, _title, _message, *, summary=None, details=None: themed_error_calls.append(
            {"summary": str(summary), "details": str(details)}
        ),
    )
    seen_threads: list[threading.Thread] = []

    def _run_callback(workflow, progress_callback):
        del workflow
        seen_threads.append(threading.current_thread())
        progress_callback(5, "starting")
        raise RuntimeError("boom")

    workflow = app_module.WorkflowConfig(
        dask_backend=app_module.DaskBackendConfig(mode=app_module.DASK_BACKEND_SLURM_CLUSTER)
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
        lambda _parent, _title, _message, *, summary=None, details=None: themed_error_calls.append(
            {"summary": str(summary), "details": str(details)}
        ),
    )

    def _run_callback(workflow, progress_callback):
        del workflow
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


def test_load_persisted_backend_returns_none_for_empty_or_invalid_json(tmp_path) -> None:
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


def test_launch_gui_applies_persisted_backend_defaults(monkeypatch, tmp_path) -> None:
    calls = {"ensure": 0}
    captured: dict[str, app_module.WorkflowConfig] = {}
    persisted_backend = app_module.DaskBackendConfig(
        mode=app_module.DASK_BACKEND_SLURM_RUNNER,
        slurm_runner=app_module.SlurmRunnerConfig(
            scheduler_file="/tmp/persisted-scheduler.json"
        ),
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
    monkeypatch.setattr(app_module, "_apply_application_icon", lambda _app: None)
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

    initial = app_module.WorkflowConfig()
    selected = app_module.launch_gui(initial=initial)

    assert calls["ensure"] == 1
    assert captured["setup_initial"].dask_backend == persisted_backend
    assert selected is not None


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
        operation_output_components={
            "flatfield": "results/flatfield/latest/data",
            "deconvolution": "results/deconvolution/latest/data",
            "shear_transform": "results/shear_transform/latest/data",
            "usegment3d": "results/usegment3d/latest/data",
            "registration": "results/registration/latest/data",
        },
        available_store_output_components={
            "flatfield": "results/flatfield/latest/data",
        },
        provenance_confirmed_operations=["flatfield"],
    )

    assert options[0] == ("data", "Raw data (data)")
    assert (
        "flatfield",
        "Flatfield Correction output (results/flatfield/latest/data) [existing]",
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
        operation_output_components={
            "flatfield": "results/flatfield/latest/data",
            "deconvolution": "results/deconvolution/latest/data",
            "shear_transform": "results/shear_transform/latest/data",
            "usegment3d": "results/usegment3d/latest/data",
            "registration": "results/registration/latest/data",
        },
        available_store_output_components={
            "flatfield": "results/flatfield/latest/data",
        },
        provenance_confirmed_operations=["flatfield"],
    )

    values = [value for value, _label in options]
    assert values.count("flatfield") == 1
    assert (
        "flatfield",
        "Flatfield Correction output (results/flatfield/latest/data)",
    ) in options


def test_build_input_source_options_omits_unconfirmed_upstream_outputs() -> None:
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
        operation_output_components={
            "flatfield": "results/flatfield/latest/data",
            "deconvolution": "results/deconvolution/latest/data",
            "shear_transform": "results/shear_transform/latest/data",
            "usegment3d": "results/usegment3d/latest/data",
            "registration": "results/registration/latest/data",
        },
        available_store_output_components={
            "deconvolution": "results/deconvolution/latest/data",
        },
        provenance_confirmed_operations=[],
    )

    values = [value for value, _ in options]
    assert values == ["data"]


def test_build_visualization_volume_layer_component_options_only_uses_confirmed() -> None:
    options = app_module._build_visualization_volume_layer_component_options(
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
            "flatfield": "results/flatfield/latest/data",
            "deconvolution": "results/deconvolution/latest/data",
        },
        provenance_confirmed_operations=["flatfield"],
    )

    assert options[0] == ("data", "Raw data (data)")
    assert (
        "results/flatfield/latest/data",
        "Flatfield Correction output (results/flatfield/latest/data)",
    ) in options
    assert all(
        component != "results/deconvolution/latest/data"
        for component, _label in options
    )


def test_build_visualization_volume_layer_component_options_deduplicates_components() -> None:
    options = app_module._build_visualization_volume_layer_component_options(
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
        provenance_confirmed_operations=["flatfield", "deconvolution"],
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
            if key == "results/flatfield/latest/data":
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
            "flatfield": "results/flatfield/latest/data",
            "deconvolution": "results/deconvolution/latest/data",
        },
    )

    assert discovered == {"flatfield": "results/flatfield/latest/data"}


def test_particle_overlay_available_when_particle_detection_runs_first() -> None:
    available = app_module._particle_overlay_available_for_visualization(
        selected_order=["particle_detection", "visualization"],
        has_particle_detection_history=False,
    )

    assert available is True


def test_particle_overlay_unavailable_when_no_outputs_or_history() -> None:
    available = app_module._particle_overlay_available_for_visualization(
        selected_order=["visualization", "particle_detection"],
        has_particle_detection_history=False,
    )

    assert available is False


def test_particle_overlay_available_with_historical_detections() -> None:
    from_history = app_module._particle_overlay_available_for_visualization(
        selected_order=["visualization"],
        has_particle_detection_history=True,
    )

    assert from_history is True


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
    assert ("Visualization", ("visualization", "mip_export")) in dialog_cls._OPERATION_TABS


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
