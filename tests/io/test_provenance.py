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
    persist_run_provenance,
    store_latest_analysis_output,
    verify_provenance_chain,
)
from clearex.io.read import ImageInfo
from clearex.workflow import WorkflowConfig


def test_is_zarr_store_path():
    assert is_zarr_store_path("sample.zarr") is True
    assert is_zarr_store_path("sample.n5") is True
    assert is_zarr_store_path("sample.tif") is False


def test_persist_run_provenance_hash_chain(tmp_path: Path):
    store_path = tmp_path / "provenance_test.zarr"
    zarr.open_group(str(store_path), mode="w")

    workflow = WorkflowConfig(
        file=str(store_path),
        deconvolution=True,
        registration=True,
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

    valid, issues = verify_provenance_chain(store_path)
    assert valid is True
    assert issues == []


def test_verify_provenance_chain_detects_tampering(tmp_path: Path):
    store_path = tmp_path / "tamper_test.zarr"
    zarr.open_group(str(store_path), mode="w")

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
    zarr.open_group(str(store_path), mode="w")

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
