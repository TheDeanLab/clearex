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
    ANALYSIS_OPERATION_ORDER,
    DASK_BACKEND_LOCAL_CLUSTER,
    DASK_BACKEND_SLURM_CLUSTER,
    DASK_BACKEND_SLURM_RUNNER,
    DEFAULT_ANALYSIS_OPERATION_PARAMETERS,
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
    normalize_analysis_operation_parameters,
    resolve_analysis_execution_sequence,
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
        assert "particle_detection" in cfg.analysis_parameters
        assert cfg.analysis_parameters["particle_detection"]["bg_sigma"] == 20.0
        assert cfg.analysis_parameters["particle_detection"]["execution_order"] == 2
        assert cfg.analysis_parameters["particle_detection"]["input_source"] == "data"
        assert cfg.analysis_parameters["visualization"]["position_index"] == 0
        assert cfg.analysis_parameters["visualization"]["use_multiscale"] is True

    def test_normalizes_particle_analysis_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "particle_detection": {
                    "channel_index": "2",
                    "bg_sigma": "12.5",
                    "overlap_zyx": [1, 2, 3],
                    "execution_order": "4",
                    "input_source": "deconvolution",
                }
            }
        )
        params = cfg.analysis_parameters["particle_detection"]
        assert params["channel_index"] == 2
        assert params["bg_sigma"] == 12.5
        assert params["overlap_zyx"] == [1, 2, 3]
        assert params["execution_order"] == 4
        assert params["input_source"] == "deconvolution"

    def test_rejects_invalid_particle_overlap_shape(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={"particle_detection": {"overlap_zyx": [1, 2]}}
            )

    def test_rejects_invalid_common_execution_order(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={"deconvolution": {"execution_order": 0}}
            )

    def test_normalizes_deconvolution_analysis_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "deconvolution": {
                    "psf_mode": "synthetic",
                    "synthetic_microscopy_mode": "confocal",
                    "synthetic_excitation_nm": "488,561",
                    "synthetic_illumination_numerical_aperture": "0.2,0.25",
                    "synthetic_emission_nm": "520,610",
                    "synthetic_detection_numerical_aperture": "0.8,1.0",
                    "hann_window_bounds": "0.75,1.0",
                    "wiener_alpha": "0.01",
                    "background": "120",
                    "decon_iterations": "4",
                    "block_size_zyx": "64,128,128",
                    "batch_size_zyx": "256,256,256",
                    "cpus_per_task": "3",
                }
            }
        )
        params = cfg.analysis_parameters["deconvolution"]
        assert params["psf_mode"] == "synthetic"
        assert params["synthetic_microscopy_mode"] == "confocal"
        assert params["synthetic_illumination_wavelength_nm"] == [488.0, 561.0]
        assert params["synthetic_excitation_nm"] == [488.0, 561.0]
        assert params["synthetic_illumination_numerical_aperture"] == [0.2, 0.25]
        assert params["synthetic_emission_nm"] == [520.0, 610.0]
        assert params["synthetic_detection_numerical_aperture"] == [0.8, 1.0]
        assert params["synthetic_numerical_aperture"] == [0.8, 1.0]
        assert params["hann_window_bounds"] == [0.75, 1.0]
        assert params["wiener_alpha"] == 0.01
        assert params["background"] == 120.0
        assert params["decon_iterations"] == 4
        assert params["block_size_zyx"] == [64, 128, 128]
        assert params["batch_size_zyx"] == [256, 256, 256]
        assert params["cpus_per_task"] == 3

    def test_normalizes_lightsheet_mode_alias(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "deconvolution": {
                    "psf_mode": "synthetic",
                    "synthetic_microscopy_mode": "lightsheet",
                    "synthetic_illumination_wavelength_nm": "488",
                    "synthetic_illumination_numerical_aperture": "0.2",
                    "synthetic_emission_nm": "520",
                    "synthetic_detection_numerical_aperture": "0.8",
                }
            }
        )
        params = cfg.analysis_parameters["deconvolution"]
        assert params["synthetic_microscopy_mode"] == "light_sheet"
        assert params["synthetic_illumination_wavelength_nm"] == [488.0]
        assert params["synthetic_illumination_numerical_aperture"] == [0.2]
        assert params["synthetic_detection_numerical_aperture"] == [0.8]

    def test_rejects_invalid_deconvolution_synthetic_mode(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={
                    "deconvolution": {
                        "psf_mode": "synthetic",
                        "synthetic_microscopy_mode": "invalid_mode",
                    }
                }
            )

    def test_rejects_invalid_deconvolution_hann_bounds(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={
                    "deconvolution": {"hann_window_bounds": [1.0, 0.5]}
                }
            )

    def test_normalizes_visualization_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "visualization": {
                    "position_index": "3",
                    "use_multiscale": 0,
                    "overlay_particle_detections": 1,
                    "launch_mode": "subprocess",
                }
            }
        )
        params = cfg.analysis_parameters["visualization"]
        assert params["position_index"] == 3
        assert params["use_multiscale"] is False
        assert params["overlay_particle_detections"] is True
        assert params["launch_mode"] == "subprocess"


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
        assert to_tpczyx_pyramid(((1, 4), (1, 2), (1,), (1, 2), (1, 2), (1, 2))) == (
            (1, 2),
            (1, 4),
            (1,),
            (1, 2),
            (1, 2),
            (1, 2),
        )

    def test_reject_invalid_pyramid_start(self):
        with pytest.raises(ValueError):
            ZarrSaveConfig(pyramid_ptczyx=((2,), (1,), (1,), (1, 2), (1, 2), (1, 2)))


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


def test_normalize_analysis_operation_parameters_returns_defaults():
    normalized = normalize_analysis_operation_parameters(None)
    assert normalized["particle_detection"]["bg_sigma"] == 20.0
    assert "registration" in normalized
    assert (
        normalized["particle_detection"]["memory_overhead_factor"]
        == DEFAULT_ANALYSIS_OPERATION_PARAMETERS["particle_detection"][
            "memory_overhead_factor"
        ]
    )
    assert normalized["deconvolution"]["execution_order"] == 1
    assert normalized["visualization"]["input_source"] == "data"
    assert normalized["visualization"]["use_multiscale"] is True


def test_resolve_analysis_execution_sequence_uses_execution_order():
    sequence = resolve_analysis_execution_sequence(
        deconvolution=True,
        particle_detection=True,
        registration=True,
        visualization=False,
        analysis_parameters={
            "deconvolution": {"execution_order": 2},
            "particle_detection": {"execution_order": 3},
            "registration": {"execution_order": 1},
        },
    )
    assert sequence == ("registration", "deconvolution", "particle_detection")


def test_analysis_operation_order_contains_expected_keys():
    assert ANALYSIS_OPERATION_ORDER == (
        "deconvolution",
        "particle_detection",
        "registration",
        "visualization",
    )
