"""BioHPC-only archive-backed workflow integration tests.

These tests exercise real acquisition fixtures staged from the BioHPC archive
into a scratch directory. They are intentionally marked ``biohpc`` so they can
run on lab infrastructure without becoming part of the default GitHub Actions
suite.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Mapping

import pytest
import zarr

from clearex.main import _run_workflow
from clearex.workflow import WorkflowConfig

_BIOHPC_DATA_ROOT = Path(
    "/archive/bioinformatics/Danuser_lab/Dean/dean/2026-03-24-test-data/"
    "kevin/20260324_lung_mv3_488nm"
)
_CELL_NAMES = (
    "cell_001",
    "cell_002",
    "cell_003",
    "cell_004",
    "cell_005",
)
_SOURCE_COMPONENT = "clearex/runtime_cache/source/data"
_SHEAR_COMPONENT = "clearex/runtime_cache/results/shear_transform/latest/data"
_DISPLAY_COMPONENT = "clearex/results/display_pyramid/latest"
_DETECTIONS_COMPONENT = "clearex/results/particle_detection/latest/detections"
_MIP_COMPONENT = "clearex/results/mip_export/latest"

pytestmark = pytest.mark.biohpc


@dataclass(frozen=True)
class WorkflowSequenceCase:
    """One staged workflow sequence exercised against archive fixtures.

    Parameters
    ----------
    name : str
        Stable identifier used for pytest parameter IDs and logger names.
    enabled_operations : mapping[str, bool]
        WorkflowConfig boolean flags for enabled analyses.
    analysis_parameters : mapping[str, mapping[str, object]]
        Per-analysis runtime parameters applied to the workflow.
    required_components : tuple[str, ...]
        Canonical store components that must exist after the workflow finishes.
    mip_source_component : str
        Canonical image component consumed by the MIP export step.
    output_suffix : str
        Expected filename suffix for exported MIP artifacts.
    required_output_names : tuple[str, ...]
        Representative exported artifact names that must be present.
    """

    name: str
    enabled_operations: Mapping[str, bool]
    analysis_parameters: Mapping[str, Mapping[str, object]]
    required_components: tuple[str, ...]
    mip_source_component: str
    output_suffix: str
    required_output_names: tuple[str, ...]


_WORKFLOW_SEQUENCE_CASES = (
    WorkflowSequenceCase(
        name="shear_display_export",
        enabled_operations={
            "shear_transform": True,
            "display_pyramid": True,
            "mip_export": True,
        },
        analysis_parameters={
            "shear_transform": {
                "execution_order": 1,
                "input_source": "data",
            },
            "display_pyramid": {
                "execution_order": 2,
                "input_source": "shear_transform",
            },
            "mip_export": {
                "execution_order": 3,
                "input_source": "shear_transform",
                "position_mode": "per_position",
                "export_format": "ome-tiff",
            },
        },
        required_components=(
            _SHEAR_COMPONENT,
            _DISPLAY_COMPONENT,
            _MIP_COMPONENT,
        ),
        mip_source_component=_SHEAR_COMPONENT,
        output_suffix=".tif",
        required_output_names=(
            "mip_xy_p0000_t0000_c0000.tif",
            "mip_yz_p0000_t0000_c0000.tif",
        ),
    ),
    WorkflowSequenceCase(
        name="detection_shear_export",
        enabled_operations={
            "shear_transform": True,
            "particle_detection": True,
            "mip_export": True,
        },
        analysis_parameters={
            "particle_detection": {
                "execution_order": 1,
                "input_source": "data",
                "channel_index": 0,
                "bg_sigma": 12.0,
                "threshold": 0.2,
            },
            "shear_transform": {
                "execution_order": 2,
                "input_source": "data",
            },
            "mip_export": {
                "execution_order": 3,
                "input_source": "shear_transform",
                "position_mode": "multi_position",
                "export_format": "zarr",
            },
        },
        required_components=(
            _DETECTIONS_COMPONENT,
            _SHEAR_COMPONENT,
            _MIP_COMPONENT,
        ),
        mip_source_component=_SHEAR_COMPONENT,
        output_suffix=".ome.zarr",
        required_output_names=(
            "mip_xy_t0000_c0000.ome.zarr",
            "mip_xz_t0000_c0000.ome.zarr",
            "mip_yz_t0000_c0000.ome.zarr",
        ),
    ),
)
_WORKFLOW_CASE_BY_CELL = {
    "cell_001": _WORKFLOW_SEQUENCE_CASES[0],
    "cell_002": _WORKFLOW_SEQUENCE_CASES[1],
    "cell_003": _WORKFLOW_SEQUENCE_CASES[0],
    "cell_004": _WORKFLOW_SEQUENCE_CASES[1],
    "cell_005": _WORKFLOW_SEQUENCE_CASES[0],
}


def _test_logger(name: str) -> logging.Logger:
    """Return a deterministic integration-test logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Logger configured with one stream handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


def _stage_archive_experiment_directory(
    *,
    scratch_root: Path,
    source_directory: Path,
) -> Path:
    """Stage one archive experiment directory into scratch storage.

    Parameters
    ----------
    scratch_root : pathlib.Path
        Scratch root for the staged experiment directory.
    source_directory : pathlib.Path
        Archive directory containing one ``experiment.yml`` and acquisition
        payload.

    Returns
    -------
    pathlib.Path
        Path to the staged ``experiment.yml``.
    """
    staged_directory = scratch_root / source_directory.name
    staged_directory.mkdir(parents=True, exist_ok=True)
    for child in source_directory.iterdir():
        target = staged_directory / child.name
        if child.name == "experiment.yml":
            continue
        target.symlink_to(child.resolve(), target_is_directory=child.is_dir())

    payload = json.loads((source_directory / "experiment.yml").read_text())
    saving = payload.setdefault("Saving", {})
    saving["root_directory"] = str(staged_directory.parent)
    saving["save_directory"] = str(staged_directory)
    experiment_path = staged_directory / "experiment.yml"
    experiment_path.write_text(json.dumps(payload, indent=4))
    return experiment_path


def _assert_component_exists(root: zarr.Group, component: str) -> None:
    """Assert that one component path exists in a Zarr group.

    Parameters
    ----------
    root : zarr.Group
        Open store root.
    component : str
        Slash-delimited component path.

    Returns
    -------
    None
        Assertion side effects only.
    """
    try:
        _ = root[component]
    except Exception as exc:  # pragma: no cover - assertion helper
        pytest.fail(f"Expected component '{component}' to exist: {exc}")


def _expected_export_count(
    *,
    root: zarr.Group,
    source_component: str,
    position_mode: str,
) -> int:
    """Return the expected number of MIP artifacts for one source component.

    Parameters
    ----------
    root : zarr.Group
        Open canonical analysis-store root.
    source_component : str
        Canonical image component consumed by MIP export.
    position_mode : str
        Normalized MIP export position mode.

    Returns
    -------
    int
        Expected number of exported projection artifacts.
    """
    source = root[source_component]
    shape_tpczyx = tuple(int(value) for value in source.shape)
    base = int(shape_tpczyx[0]) * int(shape_tpczyx[2]) * 3
    if str(position_mode).strip() == "per_position":
        return base * int(shape_tpczyx[1])
    return base


@pytest.mark.parametrize(
    "cell_name",
    _CELL_NAMES,
    ids=[
        f"{cell_name}-{_WORKFLOW_CASE_BY_CELL[cell_name].name}"
        for cell_name in _CELL_NAMES
    ],
)
def test_biohpc_archive_workflows_cover_ingestion_and_permuted_sequences(
    tmp_path: Path,
    cell_name: str,
) -> None:
    """Run permuted archive-backed workflows against each fixture cell.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test scratch directory supplied by pytest.
    cell_name : str
        Archive fixture cell directory name.
    Returns
    -------
    None
        Assertions validate successful runtime side effects.
    """
    if not _BIOHPC_DATA_ROOT.is_dir():
        pytest.skip(f"BioHPC archive fixture root is unavailable: {_BIOHPC_DATA_ROOT}")

    source_directory = _BIOHPC_DATA_ROOT / cell_name
    if not source_directory.is_dir():
        pytest.skip(
            f"BioHPC archive fixture directory is unavailable: {source_directory}"
        )

    sequence_case = _WORKFLOW_CASE_BY_CELL[cell_name]
    experiment_path = _stage_archive_experiment_directory(
        scratch_root=tmp_path / sequence_case.name,
        source_directory=source_directory,
    )
    store_path = experiment_path.parent / "data_store.ome.zarr"
    workflow = WorkflowConfig(
        file=str(experiment_path),
        prefer_dask=False,
        analysis_parameters={
            name: dict(parameters)
            for name, parameters in sequence_case.analysis_parameters.items()
        },
        **dict(sequence_case.enabled_operations),
    )

    _run_workflow(
        workflow=workflow,
        logger=_test_logger(
            f"clearex.test.integration.biohpc.{sequence_case.name}.{cell_name}"
        ),
    )

    assert store_path.is_dir()
    root = zarr.open_group(str(store_path), mode="r")
    _assert_component_exists(root, _SOURCE_COMPONENT)
    for component in sequence_case.required_components:
        _assert_component_exists(root, component)

    mip_attrs = dict(root[_MIP_COMPONENT].attrs)
    output_directory = Path(str(mip_attrs["output_directory"])).expanduser().resolve()
    expected_export_count = _expected_export_count(
        root=root,
        source_component=sequence_case.mip_source_component,
        position_mode=str(
            sequence_case.analysis_parameters["mip_export"]["position_mode"]
        ).strip(),
    )
    assert output_directory.is_dir()
    assert int(mip_attrs["exported_files"]) == expected_export_count

    output_paths = sorted(
        path
        for path in output_directory.iterdir()
        if path.name.endswith(sequence_case.output_suffix)
    )
    assert len(output_paths) == expected_export_count
    output_names = {path.name for path in output_paths}
    for required_name in sequence_case.required_output_names:
        assert required_name in output_names

    if _DETECTIONS_COMPONENT in sequence_case.required_components:
        detections = root[_DETECTIONS_COMPONENT]
        assert len(detections.shape) == 2
        assert int(detections.shape[1]) > 0
