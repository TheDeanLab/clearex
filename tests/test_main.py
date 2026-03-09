#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path
from types import SimpleNamespace
import logging

# Third Party Imports
import numpy as np

# Local Imports
import clearex.main as main_module
from clearex.io.read import ImageInfo
from clearex.workflow import WorkflowConfig


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


def test_run_workflow_visualization_only_skips_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(main_module, "_configure_dask_backend", _fake_configure_dask_backend)

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        visualization=True,
    )
    main_module._run_workflow(workflow=workflow, logger=_test_logger("clearex.test.main.visualization"))

    assert workloads == []


def test_run_workflow_particle_detection_starts_analysis_dask_startup(
    monkeypatch,
) -> None:
    workloads: list[str] = []

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(main_module, "_configure_dask_backend", _fake_configure_dask_backend)

    workflow = WorkflowConfig(
        file=None,
        prefer_dask=True,
        particle_detection=True,
    )
    main_module._run_workflow(workflow=workflow, logger=_test_logger("clearex.test.main.particle"))

    assert workloads == ["analysis"]


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

    monkeypatch.setattr(main_module, "_configure_dask_backend", _fake_configure_dask_backend)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "ImageOpener", _DummyOpener)

    workflow = WorkflowConfig(
        file=str(tmp_path / "input.tif"),
        prefer_dask=True,
    )
    main_module._run_workflow(workflow=workflow, logger=_test_logger("clearex.test.main.io_skip"))

    assert workloads == []


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
    materialized = SimpleNamespace(
        source_image_info=image_info,
        data_image_info=image_info,
        source_path=source_path,
        source_component="data",
        store_path=tmp_path / "store.tmp",
        chunks_tpczyx=(1, 1, 1, 1, 1, 1),
    )

    def _fake_configure_dask_backend(*, workflow, logger, exit_stack, workload="io"):
        del workflow, logger, exit_stack
        workloads.append(str(workload))
        return object()

    monkeypatch.setattr(main_module, "_configure_dask_backend", _fake_configure_dask_backend)
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: True)
    monkeypatch.setattr(main_module, "load_navigate_experiment", lambda path: experiment)
    monkeypatch.setattr(main_module, "resolve_experiment_data_path", lambda experiment: source_path)
    monkeypatch.setattr(
        main_module,
        "materialize_experiment_data_store",
        lambda *, experiment, source_path, chunks, pyramid_factors, client: materialized,
    )

    workflow = WorkflowConfig(
        file=str(tmp_path / "experiment.yml"),
        prefer_dask=True,
    )
    main_module._run_workflow(workflow=workflow, logger=_test_logger("clearex.test.main.io_start"))

    assert workloads == ["io"]
