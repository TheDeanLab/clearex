#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
import logging
import sys

# Third Party Imports
import numpy as np
import pytest

# Local Imports
import clearex.main as main_module
from clearex.io.provenance import persist_run_provenance
from clearex.io.read import ImageInfo
from clearex.workflow import (
    SpatialCalibrationConfig,
    WorkflowConfig,
    WorkflowExecutionCancelled,
)
from clearex.workflow import DaskBackendConfig, LocalClusterConfig

_PROVENANCE_ROOT = "clearex/provenance"
_DECONV_CACHE_DATA = main_module.analysis_cache_data_component("deconvolution")
_USEGMENT3D_CACHE_DATA = main_module.analysis_cache_data_component("usegment3d")
_FLATFIELD_CACHE_DATA = main_module.analysis_cache_data_component("flatfield")
_PARTICLE_DETECTION_COMPONENT = (
    f"{main_module.analysis_auxiliary_root('particle_detection')}/detections"
)


def _test_logger(name: str) -> logging.Logger:
    """Create an isolated null logger for tests.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Logger configured with ``NullHandler`` and no propagation.
    """
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture(autouse=True)
def _disable_legacy_store_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat synthetic test stores as canonical unless a test overrides it."""
    monkeypatch.setattr(main_module, "is_legacy_clearex_store", lambda path: False)
    original_require = main_module._require_analysis_input_component

    def _require_with_data_fallback(
        *,
        zarr_path: str,
        operation_name: str,
        field_name: str,
        requested_source: str,
        resolved_component: str,
    ) -> None:
        if str(
            resolved_component
        ) == main_module.SOURCE_CACHE_COMPONENT and main_module._zarr_component_exists(
            zarr_path, "data"
        ):
            return
        original_require(
            zarr_path=zarr_path,
            operation_name=operation_name,
            field_name=field_name,
            requested_source=requested_source,
            resolved_component=resolved_component,
        )

    monkeypatch.setattr(
        main_module,
        "_require_analysis_input_component",
        _require_with_data_fallback,
    )


def test_zarr_component_exists_prefers_membership_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ImplicitGroupRoot:
        def __contains__(self, key: object) -> bool:
            return str(key) == "data"

        def __getitem__(self, key: object) -> object:
            del key
            return object()

    monkeypatch.setattr(
        main_module.zarr,
        "open_group",
        lambda _path, mode="r": _ImplicitGroupRoot(),
    )

    assert main_module._zarr_component_exists("/tmp/fake_store.n5", "data") is True
    assert (
        main_module._zarr_component_exists(
            "/tmp/fake_store.n5",
            "results/display_pyramid/latest",
        )
        is False
    )


def test_resolve_log_directory_for_workflow_uses_parent_for_missing_navigate_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "acq" / "experiment.yml"
    workflow = WorkflowConfig(file=str(experiment_path))
    fake_experiment = SimpleNamespace(path=experiment_path)
    missing_store_path = tmp_path / "acq" / "data_store.ome.zarr"

    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda _: True)
    monkeypatch.setattr(
        main_module,
        "load_navigate_experiment",
        lambda _: fake_experiment,
    )
    monkeypatch.setattr(
        main_module,
        "resolve_experiment_data_path",
        lambda _: tmp_path / "acq" / "raw_input.tif",
    )
    monkeypatch.setattr(
        main_module,
        "resolve_data_store_path",
        lambda *_args, **_kwargs: missing_store_path,
    )

    resolved = main_module._resolve_log_directory_for_workflow(workflow)

    assert resolved == missing_store_path.parent.resolve()


def test_resolve_log_directory_for_workflow_uses_parent_for_missing_zarr_path(
    tmp_path: Path,
) -> None:
    missing_store_path = tmp_path / "new_store.zarr"
    workflow = WorkflowConfig(file=str(missing_store_path))

    resolved = main_module._resolve_log_directory_for_workflow(workflow)

    assert resolved == tmp_path.resolve()


def test_resolve_log_directory_for_workflow_keeps_existing_zarr_path(
    tmp_path: Path,
) -> None:
    existing_store_path = tmp_path / "existing_store.zarr"
    existing_store_path.mkdir(parents=True, exist_ok=True)
    workflow = WorkflowConfig(file=str(existing_store_path))

    resolved = main_module._resolve_log_directory_for_workflow(workflow)

    assert resolved == existing_store_path.resolve()


def test_configure_dask_backend_uses_processes_for_multiworker_io(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyClient:
        def close(self) -> None:
            return None

    def _fake_create_dask_client(**kwargs):
        captured.update(kwargs)
        return _DummyClient()

    workflow = WorkflowConfig(
        prefer_dask=True,
        dask_backend=DaskBackendConfig(
            local_cluster=LocalClusterConfig(
                n_workers=4,
                threads_per_worker=1,
                memory_limit="7GiB",
            )
        ),
    )

    monkeypatch.setattr(main_module, "create_dask_client", _fake_create_dask_client)

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.configure_io_multi"),
            exit_stack=stack,
            workload="io",
        )

    assert client is not None
    assert captured["processes"] is True


def test_configure_dask_backend_uses_threads_for_single_worker_io(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyClient:
        def close(self) -> None:
            return None

    def _fake_create_dask_client(**kwargs):
        captured.update(kwargs)
        return _DummyClient()

    workflow = WorkflowConfig(
        prefer_dask=True,
        dask_backend=DaskBackendConfig(
            local_cluster=LocalClusterConfig(
                n_workers=1,
                threads_per_worker=4,
                memory_limit="14GiB",
            )
        ),
    )

    monkeypatch.setattr(main_module, "create_dask_client", _fake_create_dask_client)

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.configure_io_single"),
            exit_stack=stack,
            workload="io",
        )

    assert client is not None
    assert captured["processes"] is False


def test_configure_dask_backend_emits_client_lifecycle_callbacks(
    monkeypatch,
) -> None:
    captured: list[object] = []

    def _lifecycle_callback(
        event: str,
        workload: str,
        backend_mode: str,
        client: object,
    ) -> None:
        captured.append((event, workload, backend_mode, client))

    class _DummyClient:
        dashboard_link = "http://127.0.0.1:8787/status"

        def retire_workers(self, *args, **kwargs) -> None:
            del args, kwargs
            captured.append("retire_workers")

        def shutdown(self) -> None:
            captured.append("shutdown")

        def close(self) -> None:
            captured.append("close")

    def _fake_create_dask_client(**kwargs):
        del kwargs
        return _DummyClient()

    workflow = WorkflowConfig(
        prefer_dask=True,
        dask_backend=DaskBackendConfig(
            local_cluster=LocalClusterConfig(
                n_workers=2,
                threads_per_worker=1,
                memory_limit="8GB",
            )
        ),
    )

    monkeypatch.setattr(main_module, "create_dask_client", _fake_create_dask_client)

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.configure_lifecycle"),
            exit_stack=stack,
            workload="analysis",
            dask_client_lifecycle_callback=_lifecycle_callback,
        )

        assert client is not None
        assert captured == [
            (
                "started",
                "analysis",
                main_module.DASK_BACKEND_LOCAL_CLUSTER,
                client,
            )
        ]

    assert captured == [
        (
            "started",
            "analysis",
            main_module.DASK_BACKEND_LOCAL_CLUSTER,
            client,
        ),
        "retire_workers",
        "shutdown",
        (
            "stopped",
            "analysis",
            main_module.DASK_BACKEND_LOCAL_CLUSTER,
            client,
        ),
    ]


def test_configure_dask_backend_ignores_lifecycle_callback_failures(
    monkeypatch,
) -> None:
    class _DummyClient:
        dashboard_link = "http://127.0.0.1:8787/status"

        def shutdown(self) -> None:
            return None

        def close(self) -> None:
            return None

    def _fake_create_dask_client(**kwargs):
        del kwargs
        return _DummyClient()

    def _raising_callback(
        event: str,
        workload: str,
        backend_mode: str,
        client: object,
    ) -> None:
        del event, workload, backend_mode, client
        raise RuntimeError("callback boom")

    workflow = WorkflowConfig(
        prefer_dask=True,
        dask_backend=DaskBackendConfig(
            local_cluster=LocalClusterConfig(
                n_workers=2,
                threads_per_worker=1,
                memory_limit="8GB",
            )
        ),
    )

    monkeypatch.setattr(main_module, "create_dask_client", _fake_create_dask_client)

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.configure_lifecycle_best_effort"),
            exit_stack=stack,
            workload="analysis",
            dask_client_lifecycle_callback=_raising_callback,
        )

        assert client is not None


def test_configure_dask_backend_slurm_runner_closes_client_on_wait_failure(
    monkeypatch,
) -> None:
    events: list[object] = []

    class _FakeRunner:
        def __init__(self, *, scheduler_file: str) -> None:
            events.append(("runner_init", scheduler_file))
            self.n_workers = 1

        def __enter__(self):
            events.append("runner_enter")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            events.append("runner_exit")

    class _FakeClient:
        def __init__(self, runner) -> None:
            events.append(("client_init", runner))

        def wait_for_workers(self, count: int) -> None:
            events.append(("wait_for_workers", count))
            raise RuntimeError("worker wait failed")

        def close(self) -> None:
            events.append("client_close")

    fake_dask = ModuleType("dask")
    fake_distributed = ModuleType("dask.distributed")
    fake_distributed.Client = _FakeClient
    fake_dask.distributed = fake_distributed
    fake_jobqueue = ModuleType("dask_jobqueue")
    fake_slurm = ModuleType("dask_jobqueue.slurm")
    fake_slurm.SLURMRunner = _FakeRunner
    fake_jobqueue.slurm = fake_slurm

    monkeypatch.setitem(sys.modules, "dask", fake_dask)
    monkeypatch.setitem(sys.modules, "dask.distributed", fake_distributed)
    monkeypatch.setitem(sys.modules, "dask_jobqueue", fake_jobqueue)
    monkeypatch.setitem(sys.modules, "dask_jobqueue.slurm", fake_slurm)

    default_backend = DaskBackendConfig()
    workflow = WorkflowConfig(
        prefer_dask=True,
        dask_backend=replace(
            default_backend,
            mode=main_module.DASK_BACKEND_SLURM_RUNNER,
            slurm_runner=replace(
                default_backend.slurm_runner,
                scheduler_file="/tmp/scheduler.json",
                wait_for_workers=1,
            ),
        ),
    )

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.slurm_runner_wait_failure"),
            exit_stack=stack,
            workload="analysis",
        )

        assert client is None

    assert events[0] == ("runner_init", "/tmp/scheduler.json")
    assert events[1] == "runner_enter"
    assert isinstance(events[2], tuple)
    assert events[2][0] == "client_init"
    assert events[3] == ("wait_for_workers", 1)
    assert events[4] == "client_close"
    assert events[5] == "runner_exit"


def test_close_dask_client_prefers_shutdown_for_local_clusters() -> None:
    calls: list[object] = []

    class _DummyCluster:
        def close(self) -> None:
            calls.append("cluster.close")

    class _DummyClient:
        cluster = _DummyCluster()

        def retire_workers(self, *args, **kwargs) -> None:
            calls.append(("retire_workers", args, kwargs))

        def shutdown(self) -> None:
            calls.append("shutdown")

        def close(self) -> None:
            calls.append("close")

    main_module._close_dask_client(
        client=_DummyClient(),
        logger=_test_logger("clearex.test.main.close_client_shutdown"),
        allow_shutdown=True,
        retire_workers=True,
        close_attached_cluster=True,
    )

    assert any(
        isinstance(entry, tuple) and str(entry[0]) == "retire_workers"
        for entry in calls
    )
    assert "shutdown" in calls
    assert "close" not in calls
    assert "cluster.close" not in calls


def test_close_dask_client_falls_back_to_close_and_cluster_close() -> None:
    calls: list[str] = []

    class _DummyCluster:
        def close(self) -> None:
            calls.append("cluster.close")

    class _DummyClient:
        cluster = _DummyCluster()

        def close(self) -> None:
            calls.append("close")

    main_module._close_dask_client(
        client=_DummyClient(),
        logger=_test_logger("clearex.test.main.close_client_fallback"),
        allow_shutdown=False,
        retire_workers=False,
        close_attached_cluster=True,
    )

    assert calls == ["close", "cluster.close"]


def test_configure_dask_backend_caps_workers_for_gpu_usegment3d_analysis(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyClient:
        def close(self) -> None:
            return None

    def _fake_create_dask_client(**kwargs):
        captured.update(kwargs)
        return _DummyClient()

    def _fake_recommend_local_cluster_config(**kwargs):
        del kwargs
        return SimpleNamespace(detected_gpu_count=2)

    workflow = WorkflowConfig(
        prefer_dask=True,
        usegment3d=True,
        analysis_parameters={
            "usegment3d": {
                "gpu": True,
                "require_gpu": True,
            }
        },
        dask_backend=DaskBackendConfig(
            local_cluster=LocalClusterConfig(
                n_workers=8,
                threads_per_worker=1,
                memory_limit="24GiB",
            )
        ),
    )

    monkeypatch.setattr(main_module, "create_dask_client", _fake_create_dask_client)
    monkeypatch.setattr(
        main_module,
        "recommend_local_cluster_config",
        _fake_recommend_local_cluster_config,
    )

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.configure_gpu_cap"),
            exit_stack=stack,
            workload="analysis",
        )

    assert client is not None
    assert captured["n_workers"] == 2
    assert captured["processes"] is True


def test_configure_dask_backend_does_not_cap_workers_when_gpu_not_requested(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyClient:
        def close(self) -> None:
            return None

    def _fake_create_dask_client(**kwargs):
        captured.update(kwargs)
        return _DummyClient()

    def _unexpected_recommendation(**kwargs):
        del kwargs
        raise AssertionError("GPU recommendation should not be queried.")

    workflow = WorkflowConfig(
        prefer_dask=True,
        usegment3d=True,
        analysis_parameters={
            "usegment3d": {
                "gpu": False,
                "require_gpu": False,
            }
        },
        dask_backend=DaskBackendConfig(
            local_cluster=LocalClusterConfig(
                n_workers=8,
                threads_per_worker=1,
                memory_limit="24GiB",
            )
        ),
    )

    monkeypatch.setattr(main_module, "create_dask_client", _fake_create_dask_client)
    monkeypatch.setattr(
        main_module,
        "recommend_local_cluster_config",
        _unexpected_recommendation,
    )

    with ExitStack() as stack:
        client = main_module._configure_dask_backend(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.configure_gpu_no_cap"),
            exit_stack=stack,
            workload="analysis",
        )

    assert client is not None
    assert captured["n_workers"] == 8
    assert captured["processes"] is True


def test_run_workflow_visualization_only_skips_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        visualization=True,
    )
    main_module._run_workflow(
        workflow=workflow, logger=_test_logger("clearex.test.main.visualization")
    )

    assert workloads == []


def test_run_workflow_render_movie_only_skips_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        render_movie=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.render_movie"),
    )

    assert workloads == []


def test_run_workflow_compile_movie_only_skips_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        compile_movie=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.compile_movie"),
    )

    assert workloads == []


def test_run_workflow_propagates_lifecycle_callback_to_analysis_backend(
    monkeypatch,
) -> None:
    forwarded: list[tuple[str, str, str, object]] = []

    def _lifecycle_callback(
        event: str,
        workload: str,
        backend_mode: str,
        client: object,
    ) -> None:
        forwarded.append((event, workload, backend_mode, client))

    class _DummyClient:
        dashboard_link = "http://127.0.0.1:8787/status"

        def close(self) -> None:
            return None

    def _fake_configure_dask_backend(
        *,
        workflow,
        logger,
        exit_stack,
        workload="io",
        dask_client_lifecycle_callback=None,
    ):
        del workflow, logger, exit_stack
        if workload == "analysis" and dask_client_lifecycle_callback is not None:
            client = _DummyClient()
            dask_client_lifecycle_callback(
                "started",
                workload,
                main_module.DASK_BACKEND_LOCAL_CLUSTER,
                client,
            )
            dask_client_lifecycle_callback(
                "stopped",
                workload,
                main_module.DASK_BACKEND_LOCAL_CLUSTER,
                client,
            )
            return client
        return None

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(
        main_module,
        "_analysis_execution_requires_dask_client",
        lambda sequence: True,
    )
    monkeypatch.setattr(
        main_module,
        "resolve_analysis_execution_sequence",
        lambda **kwargs: (),
    )

    workflow = WorkflowConfig(file=None, prefer_dask=True)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.workflow_lifecycle"),
        dask_client_lifecycle_callback=_lifecycle_callback,
    )

    assert forwarded[0][:2] == ("started", "analysis")
    assert forwarded[1][:2] == ("stopped", "analysis")


def test_run_workflow_threads_volume_export_through_sequence_and_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_volume_export.zarr"
    captured: dict[str, object] = {}

    class _DummyOpener:
        def open(self, path, *, prefer_dask, chunks):
            del prefer_dask, chunks
            return None, ImageInfo(
                path=Path(path),
                shape=(1, 1, 1, 2, 2, 2),
                dtype=np.uint16,
                axes="TPCZYX",
                metadata={},
            )

    def _fake_resolve_analysis_execution_sequence(**kwargs):
        captured["sequence_kwargs"] = kwargs
        return ()

    def _fake_collect_available_analysis_components(*args, **kwargs):
        del args, kwargs
        return {"data"}

    def _fake_validate_analysis_input_references(*args, **kwargs):
        del args, kwargs
        return ()

    def _fake_resolve_effective_store_spatial_calibration(
        *, store_path, desired_calibration, persist
    ):
        del store_path, persist
        return desired_calibration

    def _fake_persist_run_provenance(
        *, zarr_path, workflow, image_info, steps, outputs, status, **kwargs
    ):
        del zarr_path, image_info, steps, outputs, status, kwargs
        captured["provenance_workflow_volume_export"] = workflow.volume_export
        captured["provenance_workflow_mip_export"] = workflow.mip_export
        return "run-1"

    monkeypatch.setattr(main_module, "ImageOpener", _DummyOpener)
    monkeypatch.setattr(
        main_module,
        "resolve_analysis_execution_sequence",
        _fake_resolve_analysis_execution_sequence,
    )
    monkeypatch.setattr(
        main_module,
        "_collect_available_analysis_components",
        _fake_collect_available_analysis_components,
    )
    monkeypatch.setattr(
        main_module,
        "validate_analysis_input_references",
        _fake_validate_analysis_input_references,
    )
    monkeypatch.setattr(
        main_module,
        "_resolve_effective_store_spatial_calibration",
        _fake_resolve_effective_store_spatial_calibration,
    )
    monkeypatch.setattr(
        main_module, "persist_run_provenance", _fake_persist_run_provenance
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=False,
        volume_export=True,
        mip_export=True,
    )

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.volume_export"),
    )

    assert captured["sequence_kwargs"]["volume_export"] is True
    assert captured["sequence_kwargs"]["mip_export"] is True
    assert captured["provenance_workflow_volume_export"] is True
    assert captured["provenance_workflow_mip_export"] is True


def test_run_workflow_particle_detection_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        particle_detection=True,
    )
    main_module._run_workflow(
        workflow=workflow, logger=_test_logger("clearex.test.main.particle")
    )

    assert workloads == ["analysis"]


def test_run_workflow_usegment3d_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        usegment3d=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.usegment3d"),
    )

    assert workloads == ["analysis"]


def test_run_workflow_shear_transform_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        shear_transform=True,
    )
    main_module._run_workflow(
        workflow=workflow, logger=_test_logger("clearex.test.main.shear")
    )

    assert workloads == ["analysis"]


def test_run_workflow_flatfield_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        flatfield=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.flatfield"),
    )

    assert workloads == ["analysis"]


def test_run_workflow_mip_export_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        mip_export=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.mip_export"),
    )

    assert workloads == ["analysis"]


def test_run_workflow_display_pyramid_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        display_pyramid=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.display_pyramid"),
    )

    assert workloads == ["analysis"]


def test_run_workflow_logs_operation_context_on_flatfield_failure(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_failure.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return object()

    def _boom(*, zarr_path, parameters, client, progress_callback):
        del zarr_path, parameters, client, progress_callback
        raise RuntimeError("linux boom")

    captured_messages: list[str] = []
    logger = _test_logger("clearex.test.main.flatfield_failure")

    def _capture_exception(message: str, *args, **kwargs) -> None:
        del kwargs
        captured_messages.append(message % args if args else message)

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_flatfield_analysis", _boom)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(logger, "exception", _capture_exception)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        flatfield=True,
    )

    with pytest.raises(RuntimeError, match="linux boom"):
        main_module._run_workflow(workflow=workflow, logger=logger)

    assert len(captured_messages) == 1
    assert "Analysis operation 'flatfield' failed" in captured_messages[0]
    assert str(store_path) in captured_messages[0]
    assert "requested_input=data" in captured_messages[0]


def test_run_workflow_persists_cancelled_provenance_status(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_cancelled.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_flatfield(*, zarr_path, parameters, client, progress_callback):
        del zarr_path, parameters, client
        progress_callback(50, "cancelling")
        raise AssertionError("progress callback should have cancelled execution")

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_flatfield_analysis", _fake_flatfield)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        flatfield=True,
    )

    def _cancel_on_progress(percent: int, message: str) -> None:
        if int(percent) >= 50 and "flatfield:" in str(message):
            raise WorkflowExecutionCancelled("Analysis cancelled by user.")

    with pytest.raises(WorkflowExecutionCancelled, match="Analysis cancelled by user"):
        main_module._run_workflow(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.cancelled"),
            analysis_progress_callback=_cancel_on_progress,
        )

    runs_group = main_module.zarr.open_group(str(store_path), mode="r")[
        f"{_PROVENANCE_ROOT}/runs"
    ]
    run_records = [
        dict(runs_group[run_id].attrs["record"]) for run_id in runs_group.group_keys()
    ]

    assert len(run_records) == 1
    assert run_records[0]["status"] == "cancelled"
    assert any(
        str(step.get("name")) == "flatfield"
        and dict(step.get("parameters", {})).get("status") == "cancelled"
        for step in run_records[0]["steps"]
    )


def test_run_workflow_registration_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module,
        "_configure_dask_backend",
        _fake_configure_dask_backend,
    )

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        registration=True,
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.registration"),
    )

    assert workloads == ["analysis"]


def test_run_workflow_chains_registration_output_to_visualization(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_registration_chain.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 2, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_registration(*, zarr_path, parameters, client, progress_callback):
        del parameters, client, progress_callback
        fake_root = main_module.zarr.open_group(str(zarr_path), mode="a")
        latest = (
            fake_root.require_group("clearex")
            .require_group("results")
            .require_group("registration")
            .require_group("latest")
        )
        latest.create_dataset(
            name="affines_tpx44",
            shape=(1, 2, 4, 4),
            chunks=(1, 1, 4, 4),
            dtype="float64",
            overwrite=True,
        )
        latest.create_dataset(
            name="transformed_bboxes_tpx6",
            shape=(1, 2, 6),
            chunks=(1, 1, 6),
            dtype="float64",
            overwrite=True,
        )
        return SimpleNamespace(
            component="clearex/results/registration/latest",
            affines_component="clearex/results/registration/latest/affines_tpx44",
            transformed_bboxes_component=(
                "clearex/results/registration/latest/transformed_bboxes_tpx6"
            ),
            source_component="data",
            pairwise_source_component="data",
            requested_source_component="data",
            requested_input_resolution_level=0,
            input_resolution_level=0,
            registration_channel=0,
            registration_type="rigid",
            anchor_positions=(0,),
            edge_count=1,
            active_edge_count=1,
            dropped_edge_count=0,
            output_origin_xyz_um=(0.0, 0.0, 0.0),
            output_shape_tpczyx=(1, 1, 1, 2, 2, 3),
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 3),
        )

    def _fake_fusion(*, zarr_path, parameters, client, progress_callback):
        del parameters, client, progress_callback
        fake_root = main_module.zarr.open_group(str(zarr_path), mode="a")
        latest = (
            fake_root.require_group("clearex")
            .require_group("runtime_cache")
            .require_group("results")
            .require_group("fusion")
            .require_group("latest")
        )
        latest.create_dataset(
            name="data",
            shape=(1, 1, 1, 2, 2, 3),
            chunks=(1, 1, 1, 2, 2, 3),
            dtype="uint16",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/fusion/latest",
            data_component="clearex/runtime_cache/results/fusion/latest/data",
            registration_component="clearex/results/registration/latest",
            source_component="data",
            blend_mode="feather",
            output_shape_tpczyx=(1, 1, 1, 2, 2, 3),
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 3),
        )

    captured: dict[str, object] = {}

    def _fake_visualization(*, zarr_path, parameters, progress_callback):
        del zarr_path, progress_callback
        captured["input_source"] = str(parameters["input_source"])
        fake_root = main_module.zarr.open_group(str(store_path), mode="a")
        latest = (
            fake_root.require_group("results")
            .require_group("visualization")
            .require_group("latest")
        )
        latest.attrs["source_component"] = str(parameters["input_source"])
        return SimpleNamespace(
            component="results/visualization/latest",
            source_component=str(parameters["input_source"]),
            source_components=(str(parameters["input_source"]),),
            position_index=0,
            overlay_points_count=0,
            launch_mode="in_process",
            viewer_pid=None,
            keyframe_manifest_path="",
            keyframe_count=0,
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_registration_analysis", _fake_registration)
    monkeypatch.setattr(main_module, "run_fusion_analysis", _fake_fusion)
    monkeypatch.setattr(main_module, "run_visualization_analysis", _fake_visualization)
    monkeypatch.setattr(
        main_module,
        "publish_analysis_collection_from_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        registration=True,
        visualization=True,
        analysis_parameters={
            "registration": {
                "execution_order": 1,
                "input_source": "data",
            },
            "visualization": {
                "execution_order": 2,
                "input_source": "registration",
            },
        },
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.registration_chain"),
    )

    assert (
        captured["input_source"] == "clearex/runtime_cache/results/fusion/latest/data"
    )


def test_run_workflow_chains_visualization_output_to_render_and_compile_movie(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_movie_chain.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    (
        root.require_group("clearex")
        .require_group("results")
        .require_group("visualization")
        .require_group("latest")
    ).attrs.update({"source_component": "data"})

    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    captured: dict[str, object] = {}

    def _fake_render_movie(*, zarr_path, parameters, progress_callback):
        del progress_callback
        captured["render_input_source"] = str(parameters["input_source"])
        fake_root = main_module.zarr.open_group(str(zarr_path), mode="a")
        (
            fake_root.require_group("clearex")
            .require_group("results")
            .require_group("render_movie")
            .require_group("latest")
        ).attrs.update(
            {
                "render_manifest_path": "/tmp/render_manifest.json",
                "output_directory": "/tmp/render_movie/latest",
            }
        )
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("render_movie"),
            visualization_component=main_module.analysis_auxiliary_root(
                "visualization"
            ),
            keyframe_manifest_path="/tmp/keyframes.json",
            render_manifest_path="/tmp/render_manifest.json",
            output_directory="/tmp/render_movie/latest",
            rendered_levels=(0,),
            frame_count=5,
            fps=24,
        )

    def _fake_compile_movie(*, zarr_path, parameters, progress_callback):
        del zarr_path, progress_callback
        captured["compile_input_source"] = str(parameters["input_source"])
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("compile_movie"),
            render_component=main_module.analysis_auxiliary_root("render_movie"),
            render_manifest_path="/tmp/render_manifest.json",
            output_directory="/tmp/compile_movie/latest",
            rendered_level=0,
            output_format="mp4",
            compiled_files=("/tmp/compile_movie/latest/movie.mp4",),
            fps=24,
        )

    monkeypatch.setattr(
        main_module,
        "_configure_dask_backend",
        _fake_configure_dask_backend,
    )
    monkeypatch.setattr(main_module, "run_render_movie_analysis", _fake_render_movie)
    monkeypatch.setattr(main_module, "run_compile_movie_analysis", _fake_compile_movie)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        render_movie=True,
        compile_movie=True,
        analysis_parameters={
            "render_movie": {
                "execution_order": 1,
                "input_source": "visualization",
            },
            "compile_movie": {
                "execution_order": 2,
                "input_source": "render_movie",
            },
        },
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.movie_chain"),
    )

    assert workloads == []
    assert captured["render_input_source"] == main_module.analysis_auxiliary_root(
        "visualization"
    )
    assert captured["compile_input_source"] == main_module.analysis_auxiliary_root(
        "render_movie"
    )


def test_run_workflow_non_experiment_file_skips_io_dask_startup(
    tmp_path: Path, monkeypatch
) -> None:
    workloads: list[str] = []

    class _DummyOpener:
        def open(self, file_path, *, prefer_dask, chunks):
            del prefer_dask, chunks
            return (
                None,
                ImageInfo(
                    path=Path(file_path),
                    shape=(1, 1, 1, 1, 1, 1),
                    dtype=np.uint16,
                    axes="TPCZYX",
                    metadata={},
                ),
            )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "ImageOpener", _DummyOpener)

    workflow = WorkflowConfig(
        file=str(tmp_path / "input.tif"),
        prefer_dask=True,
    )
    main_module._run_workflow(
        workflow=workflow, logger=_test_logger("clearex.test.main.io_skip")
    )

    assert workloads == []


def test_build_workflow_config_maps_usegment3d_flag() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices=None,
        registration=False,
        visualization=False,
        mip_export=False,
    )
    workflow = main_module._build_workflow_config(args)
    assert workflow.usegment3d is True


def test_build_workflow_config_propagates_volume_export_flag() -> None:
    parser = main_module.create_parser()
    args = parser.parse_args(["--file", "/tmp/data_store.zarr", "--volume-export"])

    workflow = main_module._build_workflow_config(args)

    assert workflow.volume_export is True


def test_build_workflow_config_applies_persisted_dask_backend(monkeypatch) -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices=None,
        registration=False,
        visualization=False,
        mip_export=False,
    )

    persisted = DaskBackendConfig(
        local_cluster=LocalClusterConfig(
            n_workers=4,
            threads_per_worker=1,
            memory_limit="150GiB",
        )
    )
    monkeypatch.setattr(
        main_module,
        "_load_persisted_dask_backend_config",
        lambda: persisted,
    )

    workflow = main_module._build_workflow_config(args)
    assert workflow.dask_backend.local_cluster.n_workers == 4
    assert workflow.dask_backend.local_cluster.memory_limit == "150GiB"


def test_build_workflow_config_maps_usegment3d_channel_indices() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices="2,0,2",
        registration=False,
        visualization=False,
        mip_export=False,
    )
    workflow = main_module._build_workflow_config(args)
    params = workflow.analysis_parameters["usegment3d"]
    assert params["channel_indices"] == [2, 0]
    assert params["channel_index"] == 2
    assert params["all_channels"] is False


def test_build_workflow_config_maps_usegment3d_channel_indices_all() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices="all",
        registration=False,
        visualization=False,
        mip_export=False,
    )
    workflow = main_module._build_workflow_config(args)
    params = workflow.analysis_parameters["usegment3d"]
    assert params["all_channels"] is True


def test_build_workflow_config_maps_usegment3d_input_resolution_level() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices=None,
        input_resolution_level=2,
        registration=False,
        visualization=False,
        mip_export=False,
    )
    workflow = main_module._build_workflow_config(args)
    params = workflow.analysis_parameters["usegment3d"]
    assert params["input_resolution_level"] == 2


def test_build_workflow_config_parses_stage_axis_map() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=False,
        channel_indices=None,
        input_resolution_level=None,
        registration=False,
        visualization=False,
        mip_export=False,
        stage_axis_map="z=+x,y=none,x=+y",
    )

    workflow = main_module._build_workflow_config(args)

    assert workflow.spatial_calibration == SpatialCalibrationConfig(
        stage_axis_map_zyx=("+x", "none", "+y")
    )
    assert workflow.spatial_calibration_explicit is True


def test_build_workflow_config_rejects_invalid_input_resolution_level() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices=None,
        input_resolution_level=-1,
        registration=False,
        visualization=False,
        mip_export=False,
    )
    with pytest.raises(ValueError, match="--input-resolution-level"):
        _ = main_module._build_workflow_config(args)


def test_build_workflow_config_rejects_invalid_channel_indices() -> None:
    args = SimpleNamespace(
        file=None,
        dask=True,
        chunks=None,
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=True,
        channel_indices="1,bad",
        registration=False,
        visualization=False,
        mip_export=False,
    )
    with pytest.raises(ValueError, match="Invalid --channel-indices value"):
        _ = main_module._build_workflow_config(args)


def test_run_workflow_experiment_file_starts_io_dask_startup(
    tmp_path: Path, monkeypatch
) -> None:
    workloads: list[str] = []
    source_path = tmp_path / "source.tif"
    experiment = SimpleNamespace(
        file_type="TIFF",
        timepoints=1,
        multiposition_count=1,
        channel_count=1,
        number_z_steps=1,
    )
    image_info = ImageInfo(
        path=source_path,
        shape=(1, 1, 1, 1, 1, 1),
        dtype=np.uint16,
        axes="TPCZYX",
        metadata={},
    )
    store_path = tmp_path / "store.zarr"
    main_module.zarr.open_group(str(store_path), mode="w")
    materialized = SimpleNamespace(
        source_image_info=image_info,
        data_image_info=image_info,
        source_path=source_path,
        source_component="data",
        store_path=store_path,
        chunks_tpczyx=(1, 1, 1, 1, 1, 1),
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: True)
    monkeypatch.setattr(
        main_module, "load_navigate_experiment", lambda path: experiment
    )
    monkeypatch.setattr(
        main_module, "resolve_experiment_data_path", lambda experiment: source_path
    )
    monkeypatch.setattr(
        main_module,
        "materialize_experiment_data_store",
        lambda *, experiment, source_path, chunks, pyramid_factors, client: (
            materialized
        ),
    )

    workflow = WorkflowConfig(
        file=str(tmp_path / "experiment.yml"),
        prefer_dask=True,
    )
    main_module._run_workflow(
        workflow=workflow, logger=_test_logger("clearex.test.main.io_start")
    )

    assert workloads == ["io"]


def test_run_workflow_dispatches_volume_export_and_publishes_collection(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_volume_export.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name=main_module.SOURCE_CACHE_COMPONENT,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    workflow = WorkflowConfig(
        file=str(store_path),
        volume_export=True,
        analysis_parameters={
            "volume_export": {
                "input_source": "data",
                "export_scope": "current_selection",
                "t_index": 0,
                "p_index": 0,
                "c_index": 0,
                "resolution_level": 0,
                "export_format": "ome-zarr",
                "tiff_file_layout": "single_file",
            }
        },
    )

    published = {"analysis_name": None}
    called = {"value": False}

    def _fake_volume_export(
        *, zarr_path, parameters, progress_callback, run_id=None, client=None
    ):
        del zarr_path, parameters, progress_callback, run_id, client
        called["value"] = True
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("volume_export"),
            data_component="clearex/runtime_cache/results/volume_export/latest/data",
            source_component=main_module.SOURCE_CACHE_COMPONENT,
            resolved_resolution_component=main_module.SOURCE_CACHE_COMPONENT,
            export_scope="current_selection",
            resolution_level=0,
            generated_resolution_level=False,
            export_format="ome-zarr",
            tiff_file_layout="single_file",
            artifact_paths=(),
        )

    def _fake_publish(zarr_path, analysis_name):
        del zarr_path
        published["analysis_name"] = analysis_name

    monkeypatch.setattr(main_module, "run_volume_export_analysis", _fake_volume_export)
    monkeypatch.setattr(
        main_module, "publish_analysis_collection_from_cache", _fake_publish
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "is_legacy_clearex_store", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.volume_export_dispatch"),
    )

    assert called["value"] is True
    assert published["analysis_name"] == "volume_export"


def test_run_workflow_starts_analysis_backend_for_volume_export(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_volume_export_dask.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name=main_module.SOURCE_CACHE_COMPONENT,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    workflow = WorkflowConfig(
        file=str(store_path),
        volume_export=True,
        analysis_parameters={
            "volume_export": {
                "input_source": "data",
                "export_scope": "current_selection",
                "t_index": 0,
                "p_index": 0,
                "c_index": 0,
                "resolution_level": 0,
                "export_format": "ome-zarr",
                "tiff_file_layout": "single_file",
            }
        },
    )

    workloads: list[str] = []
    analysis_client = object()
    seen: dict[str, object | None] = {"client": None}

    def _fake_configure_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        if workload == "analysis":
            return analysis_client
        return None

    def _fake_volume_export(
        *, zarr_path, parameters, progress_callback, run_id=None, client=None
    ):
        del zarr_path, parameters, progress_callback, run_id
        seen["client"] = client
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("volume_export"),
            data_component="clearex/runtime_cache/results/volume_export/latest/data",
            source_component=main_module.SOURCE_CACHE_COMPONENT,
            resolved_resolution_component=main_module.SOURCE_CACHE_COMPONENT,
            export_scope="current_selection",
            resolution_level=0,
            generated_resolution_level=False,
            export_format="ome-zarr",
            tiff_file_layout="single_file",
            artifact_paths=(),
        )

    monkeypatch.setattr(main_module, "_configure_dask_backend", _fake_configure_backend)
    monkeypatch.setattr(main_module, "run_volume_export_analysis", _fake_volume_export)
    monkeypatch.setattr(
        main_module,
        "publish_analysis_collection_from_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "is_legacy_clearex_store", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.volume_export_analysis_client"),
    )

    assert workloads == ["analysis"]
    assert seen["client"] is analysis_client


def test_run_workflow_dispatches_volume_export_tiff_without_public_publish(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_volume_export_tiff.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name=main_module.SOURCE_CACHE_COMPONENT,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.require_group(main_module.public_analysis_root("volume_export"))

    workflow = WorkflowConfig(
        file=str(store_path),
        volume_export=True,
        analysis_parameters={
            "volume_export": {
                "input_source": "data",
                "export_scope": "current_selection",
                "t_index": 0,
                "p_index": 0,
                "c_index": 0,
                "resolution_level": 0,
                "export_format": "ome-tiff",
                "tiff_file_layout": "single_file",
            }
        },
    )

    published = {"called": False}

    def _fake_volume_export(
        *, zarr_path, parameters, progress_callback, run_id=None, client=None
    ):
        del zarr_path, parameters, progress_callback, run_id, client
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("volume_export"),
            data_component="clearex/runtime_cache/results/volume_export/latest/data",
            source_component=main_module.SOURCE_CACHE_COMPONENT,
            resolved_resolution_component=main_module.SOURCE_CACHE_COMPONENT,
            export_scope="current_selection",
            resolution_level=0,
            generated_resolution_level=False,
            export_format="ome-tiff",
            tiff_file_layout="single_file",
            artifact_paths=("clearex/results/volume_export/latest/files/test.ome.tif",),
        )

    def _fake_publish(*args, **kwargs):
        del args, kwargs
        published["called"] = True

    monkeypatch.setattr(main_module, "run_volume_export_analysis", _fake_volume_export)
    monkeypatch.setattr(
        main_module, "publish_analysis_collection_from_cache", _fake_publish
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "is_legacy_clearex_store", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.volume_export_tiff_dispatch"),
    )

    assert published["called"] is False
    runtime_root = main_module.zarr.open_group(str(store_path), mode="r")
    with pytest.raises(Exception):
        _ = runtime_root[main_module.public_analysis_root("volume_export")]


def test_run_workflow_existing_store_persists_explicit_spatial_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    persisted: dict[str, object] = {}

    class _DummyOpener:
        def open(self, path, *, prefer_dask, chunks):
            del prefer_dask, chunks
            return None, ImageInfo(
                path=Path(path),
                shape=(1, 1, 1, 2, 2, 2),
                dtype=np.uint16,
                axes="TPCZYX",
                metadata={},
            )

    monkeypatch.setattr(main_module, "ImageOpener", _DummyOpener)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(
        main_module,
        "save_store_spatial_calibration",
        lambda path, calibration: persisted.update(
            {"path": str(path), "calibration": calibration}
        )
        or calibration,
    )
    monkeypatch.setattr(
        main_module,
        "persist_run_provenance",
        lambda *, zarr_path, workflow, image_info, **kwargs: persisted.update(
            {
                "provenance_store": str(zarr_path),
                "provenance_calibration": workflow.spatial_calibration,
                "image_shape": tuple(image_info.shape),
            }
        )
        or "run-1",
    )

    workflow = WorkflowConfig(
        file=str(store_path),
        spatial_calibration=SpatialCalibrationConfig(
            stage_axis_map_zyx=("+x", "none", "+y")
        ),
        spatial_calibration_explicit=True,
    )

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.spatial_existing_store"),
    )

    assert persisted["path"] == str(store_path)
    assert persisted["calibration"] == SpatialCalibrationConfig(
        stage_axis_map_zyx=("+x", "none", "+y")
    )
    assert persisted["provenance_store"] == str(store_path)
    assert persisted["provenance_calibration"] == SpatialCalibrationConfig(
        stage_axis_map_zyx=("+x", "none", "+y")
    )


def test_run_workflow_experiment_input_persists_explicit_identity_spatial_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.tif"
    store_path = tmp_path / "prepared_store.zarr"
    experiment = SimpleNamespace(
        file_type="TIFF",
        timepoints=1,
        multiposition_count=1,
        channel_count=1,
        number_z_steps=1,
    )
    image_info = ImageInfo(
        path=source_path,
        shape=(1, 1, 1, 1, 1, 1),
        dtype=np.uint16,
        axes="TPCZYX",
        metadata={},
    )
    materialized = SimpleNamespace(
        source_image_info=image_info,
        data_image_info=image_info,
        source_path=source_path,
        source_component="data",
        store_path=store_path,
        chunks_tpczyx=(1, 1, 1, 1, 1, 1),
    )
    persisted: dict[str, object] = {}

    monkeypatch.setattr(main_module, "_configure_dask_backend", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: True)
    monkeypatch.setattr(
        main_module, "load_navigate_experiment", lambda path: experiment
    )
    monkeypatch.setattr(
        main_module, "resolve_experiment_data_path", lambda experiment: source_path
    )
    monkeypatch.setattr(
        main_module,
        "materialize_experiment_data_store",
        lambda *, experiment, source_path, chunks, pyramid_factors, client: materialized,
    )
    monkeypatch.setattr(
        main_module,
        "save_store_spatial_calibration",
        lambda path, calibration: persisted.update(
            {"path": str(path), "calibration": calibration}
        )
        or calibration,
    )
    monkeypatch.setattr(
        main_module,
        "persist_run_provenance",
        lambda *, zarr_path, workflow, image_info, **kwargs: persisted.update(
            {
                "provenance_store": str(zarr_path),
                "provenance_calibration": workflow.spatial_calibration,
            }
        )
        or "run-1",
    )

    workflow = WorkflowConfig(
        file=str(tmp_path / "experiment.yml"),
        spatial_calibration=SpatialCalibrationConfig(),
        spatial_calibration_explicit=True,
    )

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.spatial_experiment_input"),
    )

    assert persisted["path"] == str(store_path)
    assert persisted["calibration"] == SpatialCalibrationConfig()
    assert persisted["provenance_store"] == str(store_path)
    assert persisted["provenance_calibration"] == SpatialCalibrationConfig()


def test_run_workflow_experiment_input_without_override_preserves_store_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.tif"
    store_path = tmp_path / "prepared_store.zarr"
    experiment = SimpleNamespace(
        file_type="TIFF",
        timepoints=1,
        multiposition_count=1,
        channel_count=1,
        number_z_steps=1,
    )
    image_info = ImageInfo(
        path=source_path,
        shape=(1, 1, 1, 1, 1, 1),
        dtype=np.uint16,
        axes="TPCZYX",
        metadata={},
    )
    materialized = SimpleNamespace(
        source_image_info=image_info,
        data_image_info=image_info,
        source_path=source_path,
        source_component="data",
        store_path=store_path,
        chunks_tpczyx=(1, 1, 1, 1, 1, 1),
    )
    calls: dict[str, object] = {"saved": False}

    monkeypatch.setattr(main_module, "_configure_dask_backend", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: True)
    monkeypatch.setattr(
        main_module, "load_navigate_experiment", lambda path: experiment
    )
    monkeypatch.setattr(
        main_module, "resolve_experiment_data_path", lambda experiment: source_path
    )
    monkeypatch.setattr(
        main_module,
        "materialize_experiment_data_store",
        lambda *, experiment, source_path, chunks, pyramid_factors, client: materialized,
    )
    monkeypatch.setattr(
        main_module,
        "load_store_spatial_calibration",
        lambda path: SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y")),
    )
    monkeypatch.setattr(
        main_module,
        "save_store_spatial_calibration",
        lambda path, calibration: calls.update({"saved": True}) or calibration,
    )
    monkeypatch.setattr(
        main_module,
        "persist_run_provenance",
        lambda *, zarr_path, workflow, image_info, **kwargs: calls.update(
            {"provenance_calibration": workflow.spatial_calibration}
        )
        or "run-1",
    )

    workflow = WorkflowConfig(file=str(tmp_path / "experiment.yml"))

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.spatial_experiment_preserve"),
    )

    assert calls["saved"] is False
    assert calls["provenance_calibration"] == SpatialCalibrationConfig(
        stage_axis_map_zyx=("+x", "none", "+y")
    )


def test_run_workflow_skips_matching_provenance_analysis(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name=_DECONV_CACHE_DATA,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.require_group(main_module.analysis_auxiliary_root("deconvolution"))

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        deconvolution=True,
        analysis_parameters={
            "deconvolution": {
                "input_source": "data",
                "psf_mode": "measured",
                "measured_psf_paths": ["/tmp/psf.tif"],
                "measured_psf_xy_um": [0.1],
                "measured_psf_z_um": [0.5],
                "force_rerun": False,
            }
        },
    )

    persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=ImageInfo(
            path=store_path,
            shape=(1, 1, 1, 2, 2, 2),
            dtype=np.uint16,
            axes=["t", "p", "c", "z", "y", "x"],
        ),
        steps=[{"name": "deconvolution", "parameters": {}}],
        repo_root=tmp_path,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "deconvolution should have been skipped by provenance match"
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(
        main_module,
        "summarize_analysis_history",
        lambda *args, **kwargs: {
            "matches_parameters": True,
            "matching_run_id": "run-1",
            "matching_ended_utc": None,
        },
    )
    monkeypatch.setattr(main_module, "run_deconvolution_analysis", _should_not_run)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.skip_match"),
    )


def test_run_workflow_force_rerun_ignores_matching_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_force.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name=_DECONV_CACHE_DATA,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    baseline_workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        deconvolution=True,
        analysis_parameters={
            "deconvolution": {
                "input_source": "data",
                "psf_mode": "measured",
                "measured_psf_paths": ["/tmp/psf.tif"],
                "measured_psf_xy_um": [0.1],
                "measured_psf_z_um": [0.5],
                "force_rerun": False,
            }
        },
    )
    persist_run_provenance(
        zarr_path=store_path,
        workflow=baseline_workflow,
        image_info=ImageInfo(
            path=store_path,
            shape=(1, 1, 1, 2, 2, 2),
            dtype=np.uint16,
            axes=["t", "p", "c", "z", "y", "x"],
        ),
        steps=[{"name": "deconvolution", "parameters": {}}],
        repo_root=tmp_path,
    )

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        deconvolution=True,
        analysis_parameters={
            "deconvolution": {
                "input_source": "data",
                "psf_mode": "measured",
                "measured_psf_paths": ["/tmp/psf.tif"],
                "measured_psf_xy_um": [0.1],
                "measured_psf_z_um": [0.5],
                "force_rerun": True,
            }
        },
    )

    called = {"value": False}

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_deconvolution(*, zarr_path, parameters, client, progress_callback):
        del zarr_path, parameters, client, progress_callback
        called["value"] = True
        return SimpleNamespace(
            component="results/deconvolution/latest",
            data_component=_DECONV_CACHE_DATA,
            volumes_processed=1,
            channel_count=1,
            psf_mode="measured",
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 2),
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_deconvolution_analysis", _fake_deconvolution)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.force_rerun"),
    )

    assert called["value"] is True


def test_run_workflow_does_not_skip_mip_export_on_matching_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_mip_match.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name=main_module.SOURCE_CACHE_COMPONENT,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.require_group(main_module.analysis_auxiliary_root("mip_export"))

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        mip_export=True,
        analysis_parameters={
            "mip_export": {
                "input_source": "data",
                "position_mode": "per_position",
                "export_format": "ome-tiff",
                "output_directory": "",
                "force_rerun": False,
            }
        },
    )

    persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=ImageInfo(
            path=store_path,
            shape=(1, 1, 1, 2, 2, 2),
            dtype=np.uint16,
            axes=["t", "p", "c", "z", "y", "x"],
        ),
        steps=[{"name": "mip_export", "parameters": {}}],
        repo_root=tmp_path,
    )

    called = {"value": False}

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_mip_export(*, zarr_path, parameters, client, progress_callback):
        del zarr_path, parameters, client, progress_callback
        called["value"] = True
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("mip_export"),
            source_component="data",
            output_directory=str(tmp_path / "mip_output"),
            export_format="ome-tiff",
            position_mode="per_position",
            task_count=3,
            exported_files=3,
            projections=("xy", "xz", "yz"),
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_mip_export_analysis", _fake_mip_export)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "is_legacy_clearex_store", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.mip_force_execution"),
    )

    assert called["value"] is True


def test_run_workflow_skips_matching_provenance_usegment3d(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name=_USEGMENT3D_CACHE_DATA,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.require_group(main_module.analysis_auxiliary_root("usegment3d"))

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        usegment3d=True,
        analysis_parameters={
            "usegment3d": {
                "input_source": "data",
                "force_rerun": False,
            }
        },
    )

    persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=ImageInfo(
            path=store_path,
            shape=(1, 1, 1, 2, 2, 2),
            dtype=np.uint16,
            axes=["t", "p", "c", "z", "y", "x"],
        ),
        steps=[{"name": "usegment3d", "parameters": {}}],
        repo_root=tmp_path,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("usegment3d should have been skipped by provenance match")

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(
        main_module,
        "summarize_analysis_history",
        lambda *args, **kwargs: {
            "matches_parameters": True,
            "matching_run_id": "run-1",
            "matching_ended_utc": None,
        },
    )
    monkeypatch.setattr(main_module, "run_usegment3d_analysis", _should_not_run)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.skip_match_usegment3d"),
    )


def test_run_workflow_chains_usegment3d_output_to_visualization(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_chain.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_usegment3d(*, zarr_path, parameters, client, progress_callback):
        del parameters, client, progress_callback
        fake_root = main_module.zarr.open_group(str(zarr_path), mode="a")
        fake_root.create_dataset(
            name=_USEGMENT3D_CACHE_DATA,
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="uint16",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/usegment3d/latest",
            data_component=_USEGMENT3D_CACHE_DATA,
            source_component="data",
        )

    captured: dict[str, object] = {}

    def _fake_visualization(*, zarr_path, parameters, progress_callback):
        del zarr_path, progress_callback
        captured["input_source"] = str(parameters["input_source"])
        fake_root = main_module.zarr.open_group(str(store_path), mode="a")
        latest = (
            fake_root.require_group("results")
            .require_group("visualization")
            .require_group("latest")
        )
        latest.attrs["source_component"] = str(parameters["input_source"])
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("visualization"),
            source_component=str(parameters["input_source"]),
            source_components=(str(parameters["input_source"]),),
            position_index=0,
            overlay_points_count=0,
            launch_mode="in_process",
            viewer_pid=None,
            keyframe_manifest_path="",
            keyframe_count=0,
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_usegment3d_analysis", _fake_usegment3d)
    monkeypatch.setattr(main_module, "run_visualization_analysis", _fake_visualization)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        usegment3d=True,
        visualization=True,
        analysis_parameters={
            "usegment3d": {
                "execution_order": 1,
                "input_source": "data",
            },
            "visualization": {
                "execution_order": 2,
                "input_source": "usegment3d",
            },
        },
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.usegment3d_chain"),
    )

    assert captured["input_source"] == _USEGMENT3D_CACHE_DATA
    latest_ref = dict(
        main_module.zarr.open_group(str(store_path), mode="r")[
            f"{_PROVENANCE_ROOT}/latest_outputs/usegment3d"
        ].attrs
    )
    assert latest_ref["component"] == "results/usegment3d/latest"


def test_run_workflow_chains_flatfield_output_to_visualization(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_flatfield.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_flatfield(*, zarr_path, parameters, client, progress_callback):
        del parameters, client, progress_callback
        fake_root = main_module.zarr.open_group(str(zarr_path), mode="a")
        fake_root.create_dataset(
            name=_FLATFIELD_CACHE_DATA,
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="float32",
            overwrite=True,
        )
        fake_root.create_dataset(
            name="clearex/results/flatfield/latest/flatfield_pcyx",
            shape=(1, 1, 2, 2),
            chunks=(1, 1, 2, 2),
            dtype="float32",
            overwrite=True,
        )
        fake_root.create_dataset(
            name="clearex/results/flatfield/latest/darkfield_pcyx",
            shape=(1, 1, 2, 2),
            chunks=(1, 1, 2, 2),
            dtype="float32",
            overwrite=True,
        )
        fake_root.create_dataset(
            name="clearex/results/flatfield/latest/baseline_pctz",
            shape=(1, 1, 1, 2),
            chunks=(1, 1, 1, 2),
            dtype="float32",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/flatfield/latest",
            data_component=_FLATFIELD_CACHE_DATA,
            flatfield_component="clearex/results/flatfield/latest/flatfield_pcyx",
            darkfield_component="clearex/results/flatfield/latest/darkfield_pcyx",
            baseline_component="clearex/results/flatfield/latest/baseline_pctz",
            profile_count=1,
            transformed_volumes=1,
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 2),
            output_dtype="float32",
        )

    captured: dict[str, object] = {}

    def _fake_visualization(*, zarr_path, parameters, progress_callback):
        del zarr_path, progress_callback
        captured["input_source"] = str(parameters["input_source"])
        fake_root = main_module.zarr.open_group(str(store_path), mode="a")
        latest = (
            fake_root.require_group("results")
            .require_group("visualization")
            .require_group("latest")
        )
        latest.attrs["source_component"] = str(parameters["input_source"])
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("visualization"),
            source_component=str(parameters["input_source"]),
            source_components=(str(parameters["input_source"]),),
            position_index=0,
            overlay_points_count=0,
            launch_mode="in_process",
            viewer_pid=None,
            keyframe_manifest_path="",
            keyframe_count=0,
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_flatfield_analysis", _fake_flatfield)
    monkeypatch.setattr(main_module, "run_visualization_analysis", _fake_visualization)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        flatfield=True,
        visualization=True,
        analysis_parameters={
            "flatfield": {
                "execution_order": 1,
                "input_source": "data",
            },
            "visualization": {
                "execution_order": 2,
                "input_source": "flatfield",
            },
        },
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.flatfield_chain"),
    )

    assert captured["input_source"] == _FLATFIELD_CACHE_DATA
    latest_ref = dict(
        main_module.zarr.open_group(str(store_path), mode="r")[
            f"{_PROVENANCE_ROOT}/latest_outputs/flatfield"
        ].attrs
    )
    assert latest_ref["component"] == "results/flatfield/latest"


def test_run_workflow_chains_deconvolution_output_to_particle_detection(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_decon_chain.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_deconvolution(*, zarr_path, parameters, client, progress_callback):
        del parameters, client, progress_callback
        fake_root = main_module.zarr.open_group(str(zarr_path), mode="a")
        fake_root.create_dataset(
            name=_DECONV_CACHE_DATA,
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="uint16",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/deconvolution/latest",
            data_component=_DECONV_CACHE_DATA,
            volumes_processed=1,
            channel_count=1,
            psf_mode="measured",
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 2),
        )

    captured: dict[str, object] = {}

    def _fake_particle_detection(*, zarr_path, parameters, client, progress_callback):
        del zarr_path, client, progress_callback
        captured["input_source"] = str(parameters["input_source"])
        return SimpleNamespace(
            component=_PARTICLE_DETECTION_COMPONENT,
            detections=3,
            chunks_processed=1,
            channel_index=0,
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_deconvolution_analysis", _fake_deconvolution)
    monkeypatch.setattr(
        main_module,
        "run_particle_detection_analysis",
        _fake_particle_detection,
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        deconvolution=True,
        particle_detection=True,
        analysis_parameters={
            "deconvolution": {
                "execution_order": 1,
                "input_source": "data",
                "psf_mode": "measured",
                "measured_psf_paths": ["/tmp/psf.tif"],
                "measured_psf_xy_um": [0.1],
                "measured_psf_z_um": [0.5],
            },
            "particle_detection": {
                "execution_order": 2,
                "input_source": "deconvolution",
            },
        },
    )
    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.decon_particle_chain"),
    )

    assert captured["input_source"] == _DECONV_CACHE_DATA


def test_run_workflow_fails_when_scheduled_output_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_missing_upstream.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _fake_flatfield(*, zarr_path, parameters, client, progress_callback):
        del zarr_path, parameters, client, progress_callback
        return SimpleNamespace(
            component="results/flatfield/latest",
            data_component=_FLATFIELD_CACHE_DATA,
            flatfield_component="clearex/results/flatfield/latest/flatfield_pcyx",
            darkfield_component="clearex/results/flatfield/latest/darkfield_pcyx",
            baseline_component="clearex/results/flatfield/latest/baseline_pctz",
            profile_count=1,
            transformed_volumes=1,
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 2),
            output_dtype="float32",
        )

    def _should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "deconvolution should not run when upstream output is missing"
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_flatfield_analysis", _fake_flatfield)
    monkeypatch.setattr(
        main_module,
        "publish_analysis_collection_from_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(main_module, "run_deconvolution_analysis", _should_not_run)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        flatfield=True,
        deconvolution=True,
        analysis_parameters={
            "flatfield": {"execution_order": 1, "input_source": "data"},
            "deconvolution": {
                "execution_order": 2,
                "input_source": "flatfield",
                "psf_mode": "measured",
                "measured_psf_paths": ["/tmp/psf.tif"],
                "measured_psf_xy_um": [0.1],
                "measured_psf_z_um": [0.5],
            },
        },
    )

    with pytest.raises(main_module.AnalysisDependencyError, match="ClearEx stops"):
        main_module._run_workflow(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.missing_scheduled_output"),
        )

    runs_group = main_module.zarr.open_group(str(store_path), mode="r")[
        f"{_PROVENANCE_ROOT}/runs"
    ]
    run_records = [
        dict(runs_group[run_id].attrs["record"]) for run_id in runs_group.group_keys()
    ]

    assert len(run_records) == 1
    assert run_records[0]["status"] == "failed"
    assert any(
        str(step.get("name")) == "deconvolution"
        and dict(step.get("parameters", {})).get("status") == "failed"
        and dict(step.get("parameters", {})).get("reason") == "missing_input_dependency"
        and dict(step.get("parameters", {})).get("requested_input") == "flatfield"
        and dict(step.get("parameters", {})).get("resolved_input")
        == _FLATFIELD_CACHE_DATA
        for step in run_records[0]["steps"]
    )
    assert (
        run_records[0]["workflow"]["analysis_parameters"]["deconvolution"][
            "input_source"
        ]
        == _FLATFIELD_CACHE_DATA
    )


def test_run_workflow_fails_for_missing_custom_component(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_missing_custom.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        deconvolution=True,
        analysis_parameters={
            "deconvolution": {
                "execution_order": 1,
                "input_source": "results/custom/latest/data",
                "psf_mode": "measured",
                "measured_psf_paths": ["/tmp/psf.tif"],
                "measured_psf_xy_um": [0.1],
                "measured_psf_z_um": [0.5],
            }
        },
    )

    with pytest.raises(
        main_module.AnalysisDependencyError,
        match="not available in the current analysis store",
    ):
        main_module._run_workflow(
            workflow=workflow,
            logger=_test_logger("clearex.test.main.missing_custom_component"),
        )

    runs_group = main_module.zarr.open_group(str(store_path), mode="r")[
        f"{_PROVENANCE_ROOT}/runs"
    ]
    run_records = [
        dict(runs_group[run_id].attrs["record"]) for run_id in runs_group.group_keys()
    ]

    assert len(run_records) == 1
    assert run_records[0]["status"] == "failed"
    assert any(
        str(step.get("name")) == "deconvolution"
        and dict(step.get("parameters", {})).get("status") == "failed"
        and dict(step.get("parameters", {})).get("reason") == "missing_component"
        for step in run_records[0]["steps"]
    )
    assert (
        run_records[0]["workflow"]["analysis_parameters"]["deconvolution"][
            "input_source"
        ]
        == "results/custom/latest/data"
    )


def test_run_workflow_chains_from_provenance_skipped_upstream_output(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_skip_chain.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name=_USEGMENT3D_CACHE_DATA,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.require_group(main_module.analysis_auxiliary_root("usegment3d"))

    workflow = WorkflowConfig(
        file=str(store_path),
        prefer_dask=True,
        usegment3d=True,
        visualization=True,
        analysis_parameters={
            "usegment3d": {"execution_order": 1, "input_source": "data"},
            "visualization": {"execution_order": 2, "input_source": "usegment3d"},
        },
    )

    persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=ImageInfo(
            path=store_path,
            shape=(1, 1, 1, 2, 2, 2),
            dtype=np.uint16,
            axes=["t", "p", "c", "z", "y", "x"],
        ),
        steps=[{"name": "usegment3d", "parameters": {}}],
        repo_root=tmp_path,
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack, workload
        return None

    def _should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("usegment3d should have been skipped by provenance match")

    captured: dict[str, object] = {}

    def _fake_visualization(*, zarr_path, parameters, progress_callback):
        del zarr_path, progress_callback
        captured["input_source"] = str(parameters["input_source"])
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("visualization"),
            source_component=str(parameters["input_source"]),
            source_components=(str(parameters["input_source"]),),
            position_index=0,
            overlay_points_count=0,
            launch_mode="in_process",
            viewer_pid=None,
            keyframe_manifest_path="",
            keyframe_count=0,
        )

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(
        main_module,
        "summarize_analysis_history",
        lambda *args, **kwargs: {
            "matches_parameters": True,
            "matching_run_id": "run-1",
            "matching_ended_utc": None,
        },
    )
    monkeypatch.setattr(main_module, "run_usegment3d_analysis", _should_not_run)
    monkeypatch.setattr(main_module, "run_visualization_analysis", _fake_visualization)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.skip_chain"),
    )

    assert captured["input_source"] == _USEGMENT3D_CACHE_DATA
