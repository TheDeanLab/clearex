#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import logging

# Third Party Imports
import numpy as np
import pytest

# Local Imports
import clearex.main as main_module
from clearex.io.provenance import persist_run_provenance
from clearex.io.read import ImageInfo
from clearex.workflow import (
    ExecutionPolicy,
    SpatialCalibrationConfig,
    WorkflowConfig,
    WorkflowExecutionCancelled,
)
from clearex.workflow import DaskBackendConfig, LocalClusterConfig


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
    missing_store_path = tmp_path / "acq" / "data_store.zarr"

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
        execution_policy=ExecutionPolicy(mode="advanced"),
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
        execution_policy=ExecutionPolicy(mode="advanced"),
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
        execution_policy=ExecutionPolicy(mode="advanced"),
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
        execution_policy=ExecutionPolicy(mode="advanced"),
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

    runs_group = main_module.zarr.open_group(str(store_path), mode="r")["provenance"][
        "runs"
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


def test_run_workflow_registration_skips_without_crashing(monkeypatch) -> None:
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

    assert workloads == []


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
    monkeypatch.setattr(main_module, "load_navigate_experiment", lambda path: experiment)
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
    monkeypatch.setattr(main_module, "load_navigate_experiment", lambda path: experiment)
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
        name="results/deconvolution/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
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
        name="results/deconvolution/latest/data",
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
            data_component="results/deconvolution/latest/data",
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
        name="results/usegment3d/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

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
        latest = (
            fake_root.require_group("results")
            .require_group("usegment3d")
            .require_group("latest")
        )
        latest.create_dataset(
            name="data",
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="uint16",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/usegment3d/latest",
            data_component="results/usegment3d/latest/data",
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

    assert captured["input_source"] == "results/usegment3d/latest/data"
    latest_ref = dict(
        main_module.zarr.open_group(str(store_path), mode="r")["provenance"][
            "latest_outputs"
        ]["usegment3d"].attrs
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
        latest = (
            fake_root.require_group("results")
            .require_group("flatfield")
            .require_group("latest")
        )
        latest.create_dataset(
            name="data",
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="float32",
            overwrite=True,
        )
        latest.create_dataset(
            name="flatfield_pcyx",
            shape=(1, 1, 2, 2),
            chunks=(1, 1, 2, 2),
            dtype="float32",
            overwrite=True,
        )
        latest.create_dataset(
            name="darkfield_pcyx",
            shape=(1, 1, 2, 2),
            chunks=(1, 1, 2, 2),
            dtype="float32",
            overwrite=True,
        )
        latest.create_dataset(
            name="baseline_pctz",
            shape=(1, 1, 1, 2),
            chunks=(1, 1, 1, 2),
            dtype="float32",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/flatfield/latest",
            data_component="results/flatfield/latest/data",
            flatfield_component="results/flatfield/latest/flatfield_pcyx",
            darkfield_component="results/flatfield/latest/darkfield_pcyx",
            baseline_component="results/flatfield/latest/baseline_pctz",
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

    assert captured["input_source"] == "results/flatfield/latest/data"
    latest_ref = dict(
        main_module.zarr.open_group(str(store_path), mode="r")["provenance"][
            "latest_outputs"
        ]["flatfield"].attrs
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
        latest = (
            fake_root.require_group("results")
            .require_group("deconvolution")
            .require_group("latest")
        )
        latest.create_dataset(
            name="data",
            shape=(1, 1, 1, 2, 2, 2),
            chunks=(1, 1, 1, 2, 2, 2),
            dtype="uint16",
            overwrite=True,
        )
        return SimpleNamespace(
            component="results/deconvolution/latest",
            data_component="results/deconvolution/latest/data",
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
            component="results/particle_detection/latest/detections",
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

    assert captured["input_source"] == "results/deconvolution/latest/data"


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
            data_component="results/flatfield/latest/data",
            flatfield_component="results/flatfield/latest/flatfield_pcyx",
            darkfield_component="results/flatfield/latest/darkfield_pcyx",
            baseline_component="results/flatfield/latest/baseline_pctz",
            profile_count=1,
            transformed_volumes=1,
            output_chunks_tpczyx=(1, 1, 1, 2, 2, 2),
            output_dtype="float32",
        )

    def _should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("deconvolution should not run when upstream output is missing")

    monkeypatch.setattr(
        main_module, "_configure_dask_backend", _fake_configure_dask_backend
    )
    monkeypatch.setattr(main_module, "run_flatfield_analysis", _fake_flatfield)
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

    runs_group = main_module.zarr.open_group(str(store_path), mode="r")["provenance"][
        "runs"
    ]
    run_records = [
        dict(runs_group[run_id].attrs["record"]) for run_id in runs_group.group_keys()
    ]

    assert len(run_records) == 1
    assert run_records[0]["status"] == "failed"
    assert any(
        str(step.get("name")) == "deconvolution"
        and dict(step.get("parameters", {})).get("status") == "failed"
        and dict(step.get("parameters", {})).get("reason")
        == "missing_input_dependency"
        and dict(step.get("parameters", {})).get("requested_input") == "flatfield"
        and dict(step.get("parameters", {})).get("resolved_input")
        == "results/flatfield/latest/data"
        for step in run_records[0]["steps"]
    )
    assert (
        run_records[0]["workflow"]["analysis_parameters"]["deconvolution"][
            "input_source"
        ]
        == "results/flatfield/latest/data"
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

    runs_group = main_module.zarr.open_group(str(store_path), mode="r")["provenance"][
        "runs"
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
        name="results/usegment3d/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

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
    monkeypatch.setattr(main_module, "run_usegment3d_analysis", _should_not_run)
    monkeypatch.setattr(main_module, "run_visualization_analysis", _fake_visualization)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.skip_chain"),
    )

    assert captured["input_source"] == "results/usegment3d/latest/data"
