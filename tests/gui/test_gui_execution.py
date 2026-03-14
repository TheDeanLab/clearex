#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

import threading

import clearex.gui.app as app_module


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
