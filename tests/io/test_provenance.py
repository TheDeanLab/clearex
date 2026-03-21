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

# Standard Library Imports
from pathlib import Path

# Third Party Imports
import numpy as np
import zarr

# Local Imports
from clearex.io.provenance import (
    is_zarr_store_path,
    load_latest_analysis_gui_state,
    load_latest_completed_workflow_state,
    persist_latest_analysis_gui_state,
    persist_run_provenance,
    store_latest_analysis_output,
    summarize_analysis_history,
    verify_provenance_chain,
)
from clearex.io.read import ImageInfo
from clearex.io.zarr_storage import (
    create_or_overwrite_array,
    open_group as open_zarr_group,
)
from clearex.workflow import SpatialCalibrationConfig, WorkflowConfig


def _wrap_test_zarr_group(group):
    class _CompatZarrGroup:
        def __init__(self, inner_group):
            self._inner_group = inner_group

        def __getattr__(self, name):
            return getattr(self._inner_group, name)

        def __contains__(self, key):
            return key in self._inner_group

        def __delitem__(self, key):
            del self._inner_group[key]

        def __getitem__(self, key):
            item = self._inner_group[key]
            if hasattr(item, "array_keys") and hasattr(item, "group_keys"):
                return _wrap_test_zarr_group(item)
            return item

        def create_dataset(self, name, **kwargs):
            return create_or_overwrite_array(root=self._inner_group, name=name, **kwargs)

        def create_group(self, name, **kwargs):
            return _wrap_test_zarr_group(self._inner_group.create_group(name, **kwargs))

        def require_group(self, name, **kwargs):
            return _wrap_test_zarr_group(
                self._inner_group.require_group(name, **kwargs)
            )

    return _CompatZarrGroup(group)


def _open_test_zarr_group(
    path: Path | str,
    *,
    mode: str = "a",
    zarr_format: int | None = None,
):
    if zarr_format is None and mode in {"w", "w-"}:
        zarr_format = 2
    return _wrap_test_zarr_group(
        open_zarr_group(path, mode=mode, zarr_format=zarr_format)
    )


def test_is_zarr_store_path():
    assert is_zarr_store_path("sample.zarr") is True
    assert is_zarr_store_path("sample.n5") is True
    assert is_zarr_store_path("sample.tif") is False


def test_persist_run_provenance_hash_chain(tmp_path: Path):
    store_path = tmp_path / "provenance_test.zarr"
    _open_test_zarr_group(store_path, mode="w")

    workflow = WorkflowConfig(
        file=str(store_path),
        flatfield=True,
        deconvolution=True,
        usegment3d=True,
        registration=True,
        display_pyramid=True,
    )
    image_info = ImageInfo(
        path=store_path,
        shape=(4, 8, 8),
        dtype=np.uint16,
        axes=["z", "y", "x"],
    )

    run_id_1 = persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=image_info,
        repo_root=tmp_path,
    )
    run_id_2 = persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=image_info,
        repo_root=tmp_path,
    )

    root = zarr.open_group(str(store_path), mode="r")
    runs_group = root["provenance"]["runs"]
    record_1 = dict(runs_group[run_id_1].attrs["record"])
    record_2 = dict(runs_group[run_id_2].attrs["record"])

    assert root["provenance"].attrs["run_count"] == 2
    assert record_2["hash_chain"]["prev_hash"] == record_1["hash_chain"]["self_hash"]
    assert record_1["workflow"]["dask_backend_summary"].startswith("LocalCluster")
    assert record_1["workflow"]["dask_backend"]["mode"] == "local_cluster"
    assert record_1["workflow"]["flatfield"] is True
    assert record_1["workflow"]["usegment3d"] is True
    assert record_1["workflow"]["display_pyramid"] is True
    assert "flatfield" in record_1["workflow"]["selected_analyses"]
    assert "display_pyramid" in record_1["workflow"]["selected_analyses"]
    assert "usegment3d" in record_1["workflow"]["selected_analyses"]
    assert record_1["workflow"]["spatial_calibration_text"] == "z=+z,y=+y,x=+x"
    assert record_1["workflow"]["zarr_chunks_ptczyx"] == "p=1, t=1, c=1, z=256, y=256, x=256"
    assert "z=1,2,4,8" in record_1["workflow"]["zarr_pyramid_ptczyx"]

    valid, issues = verify_provenance_chain(store_path)
    assert valid is True
    assert issues == []


def test_persist_run_provenance_records_spatial_calibration(tmp_path: Path) -> None:
    store_path = tmp_path / "spatial_provenance.zarr"
    _open_test_zarr_group(store_path, mode="w")
    workflow = WorkflowConfig(
        file=str(store_path),
        visualization=True,
        spatial_calibration=SpatialCalibrationConfig(
            stage_axis_map_zyx=("+x", "none", "+y")
        ),
    )

    run_id = persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=ImageInfo(path=store_path, shape=(2, 2), dtype=np.uint8),
        repo_root=tmp_path,
    )

    root = zarr.open_group(str(store_path), mode="r")
    record = dict(root["provenance"]["runs"][run_id].attrs["record"])

    assert record["workflow"]["spatial_calibration"] == {
        "schema": "clearex.spatial_calibration.v1",
        "stage_axis_map_zyx": {"z": "+x", "y": "none", "x": "+y"},
        "theta_mode": "rotate_zy_about_x",
    }
    assert record["workflow"]["spatial_calibration_text"] == "z=+x,y=none,x=+y"


def test_verify_provenance_chain_detects_tampering(tmp_path: Path):
    store_path = tmp_path / "tamper_test.zarr"
    _open_test_zarr_group(store_path, mode="w")

    workflow = WorkflowConfig(file=str(store_path), visualization=True)
    image_info = ImageInfo(path=store_path, shape=(2, 2), dtype=np.uint8)
    run_id = persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=image_info,
        repo_root=tmp_path,
    )

    root = zarr.open_group(str(store_path), mode="a")
    run_group = root["provenance"]["runs"][run_id]
    record = dict(run_group.attrs["record"])
    record["status"] = "tampered"
    run_group.attrs["record"] = record

    valid, issues = verify_provenance_chain(store_path)
    assert valid is False
    assert len(issues) >= 1


def test_store_latest_analysis_output_overwrites_previous_version(tmp_path: Path):
    store_path = tmp_path / "output_policy_test.zarr"
    _open_test_zarr_group(store_path, mode="w")

    first = np.zeros((4, 4), dtype=np.uint8)
    second = np.ones((4, 4), dtype=np.uint8)

    component_1 = store_latest_analysis_output(
        zarr_path=store_path,
        analysis_name="deconvolution",
        output_array=first,
        run_id="run-a",
    )
    component_2 = store_latest_analysis_output(
        zarr_path=store_path,
        analysis_name="deconvolution",
        output_array=second,
        run_id="run-b",
    )

    assert component_1 == "results/deconvolution/latest"
    assert component_2 == "results/deconvolution/latest"

    root = zarr.open_group(str(store_path), mode="r")
    analysis_group = root["results"]["deconvolution"]
    latest_data = np.array(analysis_group["latest"])

    assert set(analysis_group.keys()) == {"latest"}
    assert np.array_equal(latest_data, second)

    latest_ref = dict(root["provenance"]["latest_outputs"]["deconvolution"].attrs)
    assert latest_ref["component"] == "results/deconvolution/latest"
    assert latest_ref["run_id"] == "run-b"
    assert latest_ref["storage_policy"] == "latest_only"


def test_summarize_analysis_history_reports_matching_parameters(tmp_path: Path):
    store_path = tmp_path / "history_test.zarr"
    root = _open_test_zarr_group(store_path, mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    parameters = {
        "input_source": "data",
        "execution_order": 1,
        "force_rerun": False,
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 2.0,
        "psf_mode": "measured",
        "measured_psf_paths": ["/tmp/psf.tif"],
        "measured_psf_xy_um": [0.1],
        "measured_psf_z_um": [0.5],
    }
    workflow = WorkflowConfig(
        file=str(store_path),
        deconvolution=True,
        analysis_parameters={"deconvolution": parameters},
    )
    normalized_parameters = dict(workflow.analysis_parameters["deconvolution"])
    image_info = ImageInfo(path=store_path, shape=(1, 1, 1, 2, 2, 2), dtype=np.uint16)
    run_id = persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=image_info,
        steps=[
            {
                "name": "deconvolution",
                "parameters": {
                    "component": "results/deconvolution/latest",
                },
            }
        ],
        repo_root=tmp_path,
    )

    summary = summarize_analysis_history(
        store_path,
        "deconvolution",
        parameters=normalized_parameters,
    )
    assert summary["has_successful_run"] is True
    assert summary["matches_parameters"] is True
    assert summary["matching_run_id"] == run_id
    assert summary["latest_success_run_id"] == run_id


def test_summarize_analysis_history_ignores_skipped_steps(tmp_path: Path):
    store_path = tmp_path / "history_skip_test.zarr"
    root = _open_test_zarr_group(store_path, mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    parameters = {
        "input_source": "data",
        "execution_order": 2,
        "force_rerun": False,
    }
    workflow = WorkflowConfig(
        file=str(store_path),
        particle_detection=True,
        analysis_parameters={"particle_detection": parameters},
    )
    normalized_parameters = dict(workflow.analysis_parameters["particle_detection"])
    image_info = ImageInfo(path=store_path, shape=(1, 1, 1, 2, 2, 2), dtype=np.uint16)
    _ = persist_run_provenance(
        zarr_path=store_path,
        workflow=workflow,
        image_info=image_info,
        steps=[
            {
                "name": "particle_detection",
                "parameters": {
                    "status": "skipped",
                    "reason": "provenance_parameter_match",
                },
            }
        ],
        repo_root=tmp_path,
    )

    summary = summarize_analysis_history(
        store_path,
        "particle_detection",
        parameters=normalized_parameters,
    )
    assert summary["has_successful_run"] is False
    assert summary["matches_parameters"] is False


def test_load_latest_completed_workflow_state_skips_cancelled_runs(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "latest_completed_workflow.zarr"
    _open_test_zarr_group(store_path, mode="w")

    completed_workflow = WorkflowConfig(file=str(store_path), visualization=True)
    image_info = ImageInfo(path=store_path, shape=(2, 2), dtype=np.uint8)
    completed_run_id = persist_run_provenance(
        zarr_path=store_path,
        workflow=completed_workflow,
        image_info=image_info,
        status="completed",
        repo_root=tmp_path,
    )
    _ = persist_run_provenance(
        zarr_path=store_path,
        workflow=WorkflowConfig(file=str(store_path), shear_transform=True),
        image_info=image_info,
        status="cancelled",
        repo_root=tmp_path,
    )

    loaded = load_latest_completed_workflow_state(store_path)

    assert loaded is not None
    assert loaded["run_id"] == completed_run_id
    assert loaded["workflow"]["visualization"] is True
    assert loaded["workflow"]["shear_transform"] is False


def test_latest_analysis_gui_state_round_trip(tmp_path: Path) -> None:
    store_path = tmp_path / "analysis_gui_state.zarr"
    _open_test_zarr_group(store_path, mode="w")
    payload = {
        "flatfield": True,
        "deconvolution": False,
        "analysis_parameters": {
            "flatfield": {
                "execution_order": 1,
                "input_source": "data",
                "force_rerun": False,
                "get_darkfield": False,
            }
        },
    }

    persist_latest_analysis_gui_state(
        store_path,
        payload,
        source="unit_test",
    )
    loaded = load_latest_analysis_gui_state(store_path)

    assert loaded is not None
    assert loaded["source"] == "unit_test"
    assert loaded["updated_utc"] is not None
    assert loaded["workflow"]["flatfield"] is True
    assert loaded["workflow"]["analysis_parameters"]["flatfield"]["execution_order"] == 1
