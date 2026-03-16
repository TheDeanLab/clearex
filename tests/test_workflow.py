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

import numpy as np
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
    dask_backend_from_dict,
    dask_backend_to_dict,
    format_dask_backend_summary,
    format_local_cluster_recommendation_summary,
    format_chunks,
    format_pyramid_levels,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    parse_chunks,
    parse_pyramid_levels,
    normalize_analysis_operation_parameters,
    recommend_local_cluster_config,
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
        assert "flatfield" in cfg.analysis_parameters
        assert cfg.analysis_parameters["flatfield"]["execution_order"] == 1
        assert cfg.analysis_parameters["flatfield"]["get_darkfield"] is False
        assert cfg.analysis_parameters["flatfield"]["fit_mode"] == "tiled"
        assert cfg.analysis_parameters["flatfield"]["fit_tile_shape_yx"] == [256, 256]
        assert cfg.analysis_parameters["flatfield"]["blend_tiles"] is False
        assert "shear_transform" in cfg.analysis_parameters
        assert cfg.analysis_parameters["shear_transform"]["execution_order"] == 3
        assert cfg.analysis_parameters["shear_transform"]["interpolation"] == "linear"
        assert "particle_detection" in cfg.analysis_parameters
        assert cfg.analysis_parameters["particle_detection"]["bg_sigma"] == 20.0
        assert cfg.analysis_parameters["particle_detection"]["execution_order"] == 4
        assert cfg.analysis_parameters["particle_detection"]["input_source"] == "data"
        assert "usegment3d" in cfg.analysis_parameters
        assert cfg.analysis_parameters["usegment3d"]["execution_order"] == 5
        assert cfg.analysis_parameters["usegment3d"]["input_source"] == "data"
        assert cfg.analysis_parameters["visualization"]["show_all_positions"] is False
        assert cfg.analysis_parameters["visualization"]["position_index"] == 0
        assert cfg.analysis_parameters["visualization"]["use_multiscale"] is True
        assert cfg.analysis_parameters["visualization"]["capture_keyframes"] is True
        assert cfg.analysis_parameters["visualization"]["keyframe_manifest_path"] == ""
        assert cfg.analysis_parameters["visualization"]["keyframe_layer_overrides"] == []
        assert "mip_export" in cfg.analysis_parameters
        assert cfg.analysis_parameters["mip_export"]["execution_order"] == 8
        assert cfg.analysis_parameters["mip_export"]["position_mode"] == "multi_position"
        assert cfg.analysis_parameters["mip_export"]["export_format"] == "tiff"

    def test_normalizes_flatfield_analysis_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "flatfield": {
                    "get_darkfield": 0,
                    "smoothness_flatfield": "2.5",
                    "working_size": "192",
                    "is_timelapse": 1,
                    "overlap_zyx": [4, 8, 16],
                    "fit_mode": "FULL-VOLUME",
                    "fit_tile_shape_yx": "320,448",
                    "blend_tiles": 1,
                    "input_source": "deconvolution",
                }
            }
        )
        params = cfg.analysis_parameters["flatfield"]
        assert params["get_darkfield"] is False
        assert params["smoothness_flatfield"] == 2.5
        assert params["working_size"] == 192
        assert params["is_timelapse"] is True
        assert params["overlap_zyx"] == [4, 8, 16]
        assert params["fit_mode"] == "full_volume"
        assert params["fit_tile_shape_yx"] == [320, 448]
        assert params["blend_tiles"] is True
        assert params["input_source"] == "deconvolution"

    def test_rejects_invalid_flatfield_working_size(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={"flatfield": {"working_size": 0}}
            )

    def test_rejects_invalid_flatfield_fit_mode(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={"flatfield": {"fit_mode": "chunked"}}
            )

    def test_rejects_invalid_flatfield_fit_tile_shape(self):
        with pytest.raises(ValueError):
            WorkflowConfig(
                analysis_parameters={"flatfield": {"fit_tile_shape_yx": [128]}}
            )

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
                    "show_all_positions": 1,
                    "position_index": "3",
                    "use_multiscale": 0,
                    "overlay_particle_detections": 1,
                    "launch_mode": "subprocess",
                    "capture_keyframes": 0,
                    "keyframe_manifest_path": " keyframes.json ",
                    "keyframe_layer_overrides": [
                        {
                            "layer_name": "data (p=0, c=0)",
                            "visible": "true",
                            "colormap": "green",
                            "rendering": "attenuated_mip",
                            "annotation": "Nuclei",
                        }
                    ],
                }
            }
        )
        params = cfg.analysis_parameters["visualization"]
        assert params["show_all_positions"] is True
        assert params["position_index"] == 3
        assert params["use_multiscale"] is False
        assert params["overlay_particle_detections"] is True
        assert params["launch_mode"] == "subprocess"
        assert params["capture_keyframes"] is False
        assert params["keyframe_manifest_path"] == "keyframes.json"
        assert params["keyframe_layer_overrides"] == [
            {
                "layer_name": "data (p=0, c=0)",
                "visible": True,
                "colormap": "green",
                "rendering": "attenuated_mip",
                "annotation": "Nuclei",
            }
        ]

    def test_normalizes_mip_export_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "mip_export": {
                    "position_mode": "PER_POSITION",
                    "export_format": "ZARR",
                    "output_directory": " exports/mip ",
                }
            }
        )
        params = cfg.analysis_parameters["mip_export"]
        assert params["position_mode"] == "per_position"
        assert params["export_format"] == "zarr"
        assert params["output_directory"] == "exports/mip"

    def test_normalizes_shear_transform_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "shear_transform": {
                    "shear_xy": "0.12",
                    "shear_xz": "-0.3",
                    "shear_yz": "0.45",
                    "rotation_deg_x": "10",
                    "rotation_deg_y": "-5",
                    "rotation_deg_z": "2.5",
                    "auto_rotate_from_shear": 1,
                    "interpolation": "NearestNeighbor",
                    "fill_value": "15.5",
                    "output_dtype": "uint16",
                    "roi_padding_zyx": [4, 8, 12],
                    "execution_order": "6",
                }
            }
        )
        params = cfg.analysis_parameters["shear_transform"]
        assert params["shear_xy"] == 0.12
        assert params["shear_xz"] == -0.3
        assert params["shear_yz"] == 0.45
        assert params["shear_xy_deg"] == pytest.approx(np.degrees(np.arctan(0.12)))
        assert params["shear_xz_deg"] == pytest.approx(np.degrees(np.arctan(-0.3)))
        assert params["shear_yz_deg"] == pytest.approx(np.degrees(np.arctan(0.45)))
        assert params["rotation_deg_x"] == 10.0
        assert params["rotation_deg_y"] == -5.0
        assert params["rotation_deg_z"] == 2.5
        assert params["auto_rotate_from_shear"] is True
        assert params["interpolation"] == "nearestneighbor"
        assert params["fill_value"] == 15.5
        assert params["output_dtype"] == "uint16"
        assert params["roi_padding_zyx"] == [4, 8, 12]
        assert params["execution_order"] == 6

    def test_normalizes_shear_transform_degree_inputs(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "shear_transform": {
                    "shear_xy": 99.0,
                    "shear_xy_deg": 35.0,
                    "shear_xz_deg": -10.0,
                    "shear_yz_deg": 0.0,
                }
            }
        )
        params = cfg.analysis_parameters["shear_transform"]
        assert params["shear_xy_deg"] == 35.0
        assert params["shear_xz_deg"] == -10.0
        assert params["shear_yz_deg"] == 0.0
        assert params["shear_xy"] == pytest.approx(np.tan(np.deg2rad(35.0)))
        assert params["shear_xz"] == pytest.approx(np.tan(np.deg2rad(-10.0)))
        assert params["shear_yz"] == pytest.approx(0.0)

    def test_normalizes_usegment3d_parameters(self):
        cfg = WorkflowConfig(
            analysis_parameters={
                "usegment3d": {
                    "execution_order": "11",
                    "input_source": "deconvolution",
                    "memory_overhead_factor": "3.5",
                    "overlap_zyx": [1, 2, 3],
                    "force_rerun": 1,
                }
            }
        )
        params = cfg.analysis_parameters["usegment3d"]
        assert params["execution_order"] == 11
        assert params["input_source"] == "deconvolution"
        assert params["memory_overhead_factor"] == 3.5
        assert params["overlap_zyx"] == [1, 2, 3]
        assert params["force_rerun"] is True


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

    def test_slurm_cluster_round_trip_serialization(self):
        cfg = DaskBackendConfig(
            mode=DASK_BACKEND_SLURM_CLUSTER,
            local_cluster=LocalClusterConfig(
                n_workers=8,
                threads_per_worker=2,
                memory_limit="10GB",
                local_directory="/tmp/clearex-local",
            ),
            slurm_runner=SlurmRunnerConfig(
                scheduler_file="/tmp/scheduler.json",
                wait_for_workers=4,
            ),
            slurm_cluster=SlurmClusterConfig(
                workers=3,
                cores=12,
                processes=2,
                memory="48GB",
                local_directory="/tmp/clearex-slurm",
                interface="ib1",
                walltime="08:00:00",
                job_name="clearex-test",
                queue="gpu",
                death_timeout="700s",
                mail_user="user@example.com",
                job_extra_directives=("--mail-type=FAIL", "--qos=long"),
                dashboard_address=":9100",
                scheduler_interface="ib1",
                idle_timeout="4000s",
                allowed_failures=5,
            ),
        )
        payload = dask_backend_to_dict(cfg)
        parsed = dask_backend_from_dict(payload)
        assert parsed == cfg

    def test_deserialization_invalid_payload_falls_back_to_defaults(self):
        payload = {
            "mode": "unknown_mode",
            "local_cluster": {"n_workers": 0, "threads_per_worker": "bad"},
            "slurm_runner": {"scheduler_file": 42, "wait_for_workers": 0},
            "slurm_cluster": {"workers": "NaN", "queue": ""},
        }
        parsed = dask_backend_from_dict(payload)
        assert parsed == DaskBackendConfig()

    def test_reject_invalid_mode(self):
        with pytest.raises(ValueError):
            DaskBackendConfig(mode="invalid")  # type: ignore[arg-type]


class TestLocalClusterRecommendation:
    def test_recommendation_uses_dataset_and_chunk_sizes(self):
        recommendation = recommend_local_cluster_config(
            shape_tpczyx=(1, 1, 1, 512, 2048, 2048),
            chunks_tpczyx=(1, 1, 1, 256, 256, 256),
            dtype_itemsize=2,
            cpu_count=16,
            memory_bytes=64 << 30,
            gpu_count=0,
        )

        assert recommendation.config.n_workers == 16
        assert recommendation.config.threads_per_worker == 1
        assert recommendation.config.memory_limit == "3584MiB"
        assert recommendation.estimated_chunk_count == 128
        summary = format_local_cluster_recommendation_summary(recommendation)
        assert "Detected 16 CPUs" in summary
        assert "recommend 16 worker(s)" in summary

    def test_recommendation_reduces_workers_for_large_chunks(self):
        recommendation = recommend_local_cluster_config(
            shape_tpczyx=(1, 1, 1, 1024, 2048, 2048),
            chunks_tpczyx=(1, 1, 1, 512, 1024, 1024),
            dtype_itemsize=2,
            cpu_count=32,
            memory_bytes=12 << 30,
            gpu_count=0,
        )

        assert recommendation.config.n_workers == 3
        assert recommendation.config.threads_per_worker == 1
        assert recommendation.config.memory_limit == "3328MiB"

    def test_recommendation_includes_gpu_diagnostics(self):
        recommendation = recommend_local_cluster_config(
            chunks_tpczyx=(1, 1, 1, 256, 256, 256),
            dtype_itemsize=2,
            cpu_count=24,
            memory_bytes=96 << 30,
            gpu_count=2,
            gpu_memory_bytes=160 << 30,
        )

        assert recommendation.detected_gpu_count == 2
        assert recommendation.detected_gpu_memory_bytes == (160 << 30)
        summary = format_local_cluster_recommendation_summary(recommendation)
        assert "2 GPU(s)" in summary


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
    assert normalized["deconvolution"]["execution_order"] == 2
    assert normalized["flatfield"]["execution_order"] == 1
    assert normalized["shear_transform"]["execution_order"] == 3
    assert normalized["usegment3d"]["execution_order"] == 5
    assert normalized["visualization"]["input_source"] == "data"
    assert normalized["visualization"]["show_all_positions"] is False
    assert normalized["visualization"]["use_multiscale"] is True
    assert normalized["visualization"]["capture_keyframes"] is True
    assert normalized["visualization"]["keyframe_layer_overrides"] == []
    assert normalized["mip_export"]["execution_order"] == 8
    assert normalized["mip_export"]["position_mode"] == "multi_position"
    assert normalized["mip_export"]["export_format"] == "tiff"


def test_resolve_analysis_execution_sequence_uses_execution_order():
    sequence = resolve_analysis_execution_sequence(
        flatfield=True,
        deconvolution=True,
        shear_transform=True,
        particle_detection=True,
        usegment3d=False,
        registration=True,
        visualization=False,
        mip_export=False,
        analysis_parameters={
            "flatfield": {"execution_order": 4},
            "deconvolution": {"execution_order": 2},
            "shear_transform": {"execution_order": 5},
            "particle_detection": {"execution_order": 3},
            "registration": {"execution_order": 1},
        },
    )
    assert sequence == (
        "registration",
        "deconvolution",
        "particle_detection",
        "flatfield",
        "shear_transform",
    )


def test_resolve_analysis_execution_sequence_includes_mip_export_last_by_default():
    sequence = resolve_analysis_execution_sequence(
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        particle_detection=False,
        usegment3d=False,
        registration=False,
        visualization=True,
        mip_export=True,
        analysis_parameters=None,
    )
    assert sequence == ("visualization", "mip_export")


def test_analysis_operation_order_contains_expected_keys():
    assert ANALYSIS_OPERATION_ORDER == (
        "flatfield",
        "deconvolution",
        "shear_transform",
        "particle_detection",
        "usegment3d",
        "registration",
        "visualization",
        "mip_export",
    )
