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
