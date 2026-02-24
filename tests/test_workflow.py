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

import pytest

from clearex.workflow import (
    DASK_BACKEND_LOCAL_CLUSTER,
    DASK_BACKEND_SLURM_CLUSTER,
    DASK_BACKEND_SLURM_RUNNER,
    DEFAULT_ZARR_CHUNKS_PTCZYX,
    DEFAULT_ZARR_PYRAMID_PTCZYX,
    DaskBackendConfig,
    LocalClusterConfig,
    SlurmClusterConfig,
    SlurmRunnerConfig,
    WorkflowConfig,
    ZarrSaveConfig,
    dask_backend_to_dict,
    format_dask_backend_summary,
    format_chunks,
    format_pyramid_levels,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    parse_chunks,
    parse_pyramid_levels,
    to_tpczyx_chunks,
    to_tpczyx_pyramid,
)


class TestParseChunks:
    def test_parse_none(self):
        assert parse_chunks(None) is None

    def test_parse_empty(self):
        assert parse_chunks("") is None
        assert parse_chunks("   ") is None

    def test_parse_integer(self):
        assert parse_chunks("256") == 256

    def test_parse_tuple(self):
        assert parse_chunks("1,256,256") == (1, 256, 256)

    def test_reject_non_positive_values(self):
        with pytest.raises(ValueError):
            parse_chunks("0")
        with pytest.raises(ValueError):
            parse_chunks("1,0,64")

    def test_reject_non_numeric_values(self):
        with pytest.raises(ValueError):
            parse_chunks("abc")
        with pytest.raises(ValueError):
            parse_chunks("1,abc,64")


class TestFormatChunks:
    def test_format_none(self):
        assert format_chunks(None) == ""

    def test_format_integer(self):
        assert format_chunks(64) == "64"

    def test_format_tuple(self):
        assert format_chunks((1, 128, 128)) == "1,128,128"


class TestWorkflowConfig:
    def test_has_analysis_selection(self):
        cfg = WorkflowConfig()
        assert cfg.has_analysis_selection() is False

        cfg.registration = True
        assert cfg.has_analysis_selection() is True

    def test_default_zarr_save_config(self):
        cfg = WorkflowConfig()
        assert cfg.zarr_save.chunks_ptczyx == DEFAULT_ZARR_CHUNKS_PTCZYX
        assert cfg.zarr_save.pyramid_ptczyx == DEFAULT_ZARR_PYRAMID_PTCZYX
        assert cfg.dask_backend.mode == DASK_BACKEND_LOCAL_CLUSTER


class TestZarrSaveConfig:
    def test_default_tpczyx_conversion(self):
        cfg = ZarrSaveConfig()
        assert cfg.chunks_tpczyx() == (1, 1, 1, 256, 256, 256)
        assert cfg.pyramid_tpczyx() == (
            (1,),
            (1,),
            (1,),
            (1, 2, 4, 8),
            (1, 2, 4, 8),
            (1, 2, 4, 8),
        )

    def test_conversions_reorder_position_and_time(self):
        assert to_tpczyx_chunks((9, 7, 5, 3, 2, 1)) == (7, 9, 5, 3, 2, 1)
        assert to_tpczyx_pyramid(
            ((1, 4), (1, 2), (1,), (1, 2), (1, 2), (1, 2))
        ) == (
            (1, 2),
            (1, 4),
            (1,),
            (1, 2),
            (1, 2),
            (1, 2),
        )

    def test_reject_invalid_pyramid_start(self):
        with pytest.raises(ValueError):
            ZarrSaveConfig(
                pyramid_ptczyx=((2,), (1,), (1,), (1, 2), (1, 2), (1, 2))
            )


class TestPyramidLevelFormatting:
    def test_parse_and_format_pyramid_levels(self):
        parsed = parse_pyramid_levels("1, 2, 4, 8", axis_name="z")
        assert parsed == (1, 2, 4, 8)
        assert format_pyramid_levels(parsed) == "1,2,4,8"

    def test_parse_pyramid_levels_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            parse_pyramid_levels("", axis_name="z")
        with pytest.raises(ValueError):
            parse_pyramid_levels("0,2", axis_name="z")
        with pytest.raises(ValueError):
            parse_pyramid_levels("2,4", axis_name="z")

    def test_summary_formatters(self):
        cfg = ZarrSaveConfig()
        assert "p=1" in format_zarr_chunks_ptczyx(cfg.chunks_ptczyx)
        assert "z=1,2,4,8" in format_zarr_pyramid_ptczyx(cfg.pyramid_ptczyx)


class TestDaskBackendConfig:
    def test_local_backend_summary(self):
        cfg = DaskBackendConfig()
        assert format_dask_backend_summary(cfg).startswith("LocalCluster")

    def test_slurm_runner_summary(self):
        cfg = DaskBackendConfig(
            mode=DASK_BACKEND_SLURM_RUNNER,
            slurm_runner=SlurmRunnerConfig(scheduler_file="/tmp/scheduler.json"),
        )
        summary = format_dask_backend_summary(cfg)
        assert summary.startswith("SLURMRunner")
        assert "/tmp/scheduler.json" in summary

    def test_slurm_cluster_serialization(self):
        cfg = DaskBackendConfig(
            mode=DASK_BACKEND_SLURM_CLUSTER,
            local_cluster=LocalClusterConfig(),
            slurm_runner=SlurmRunnerConfig(),
            slurm_cluster=SlurmClusterConfig(mail_user="user@example.com"),
        )
        payload = dask_backend_to_dict(cfg)
        assert payload["mode"] == DASK_BACKEND_SLURM_CLUSTER
        assert payload["slurm_cluster"]["mail_user"] == "user@example.com"

    def test_reject_invalid_mode(self):
        with pytest.raises(ValueError):
            DaskBackendConfig(mode="invalid")  # type: ignore[arg-type]
