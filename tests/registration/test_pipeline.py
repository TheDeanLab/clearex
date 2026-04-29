#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import dask as dask_module
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import zarr

import clearex.registration.pipeline as registration_pipeline
from clearex.workflow import SPATIAL_CALIBRATION_SCHEMA, SpatialCalibrationConfig


def _translation_matrix(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Build a homogeneous XYZ translation matrix."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray([x, y, z], dtype=np.float64)
    return matrix


def _edge_result(
    edge: registration_pipeline._EdgeSpec,
    matrix_xyz: np.ndarray,
    *,
    success: bool = True,
    intensity_success: bool = False,
    intensity_scale: float = 1.0,
    intensity_offset: float = 0.0,
) -> dict[str, object]:
    """Build one synthetic pairwise edge result."""
    return {
        "fixed_position": int(edge.fixed_position),
        "moving_position": int(edge.moving_position),
        "success": bool(success),
        "reason": "",
        "correction_matrix_xyz": np.asarray(matrix_xyz, dtype=np.float64).tolist(),
        "overlap_voxels": int(edge.overlap_voxels),
        "nominal_overlap_pixels": int(edge.overlap_voxels),
        "intensity_success": bool(intensity_success),
        "intensity_scale": float(intensity_scale),
        "intensity_offset": float(intensity_offset),
        "intensity_samples": int(edge.overlap_voxels),
    }


def _write_multiposition_sidecar(directory: Path, rows: list[list[float]]) -> Path:
    """Write minimal Navigate experiment and multiposition sidecar files."""
    experiment_path = directory / "experiment.yml"
    experiment_path.write_text("Saving:\n  file_type: OME-ZARR\n", encoding="utf-8")
    sidecar_path = directory / "multi_positions.yml"
    payload = [["X", "Y", "Z", "Theta", "F"], *rows]
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    return experiment_path


def _create_registration_store(
    tmp_path: Path,
    *,
    timepoints: int = 2,
    positions: int = 3,
    channels: int = 1,
    shape_zyx: tuple[int, int, int] = (4, 4, 6),
    include_pyramid: bool = True,
) -> Path:
    """Create a synthetic canonical 6D store with overlapping multiposition tiles."""
    store_path = tmp_path / "registration_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data_shape = (timepoints, positions, channels, *shape_zyx)
    data = root.create_dataset(
        name="data",
        shape=data_shape,
        chunks=(1, 1, 1, *shape_zyx),
        dtype="uint16",
        overwrite=True,
    )
    data.attrs["voxel_size_um_zyx"] = [1.0, 1.0, 1.0]
    root.attrs["voxel_size_um_zyx"] = [1.0, 1.0, 1.0]
    root.attrs["spatial_calibration"] = {
        "schema": SPATIAL_CALIBRATION_SCHEMA,
        "stage_axis_map_zyx": {"z": "+z", "y": "+y", "x": "+x"},
        "theta_mode": "rotate_zy_about_x",
    }
    experiment_path = _write_multiposition_sidecar(
        tmp_path,
        rows=[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0, 0.0],
        ][:positions],
    )
    root.attrs["source_experiment"] = str(experiment_path)
    root.attrs["data_pyramid_factors_tpczyx"] = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2],
    ]

    for t_index in range(timepoints):
        for position_index in range(positions):
            volume = np.full(
                shape_zyx,
                fill_value=np.uint16((position_index + 1) * 10 + t_index),
                dtype=np.uint16,
            )
            data[t_index, position_index, 0] = volume

    if include_pyramid:
        pyramid = root.require_group("data_pyramid")
        level_1 = pyramid.create_dataset(
            name="level_1",
            shape=(timepoints, positions, channels, 2, 2, 3),
            chunks=(1, 1, 1, 2, 2, 3),
            dtype="uint16",
            overwrite=True,
        )
        level_1[:] = data[:, :, :, ::2, ::2, ::2]

    return store_path


def test_extract_voxel_size_uses_source_component_chain(tmp_path: Path) -> None:
    """Registration voxel-size lookup should follow ``source_component`` ancestry."""
    store_path = tmp_path / "registration_voxel_chain.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    source = root.create_dataset(
        name="clearex/runtime_cache/source/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    source.attrs["voxel_size_um_zyx"] = [6.0, 1.1, 1.1]
    flatfield = root.create_dataset(
        name="clearex/runtime_cache/results/flatfield/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="float32",
        overwrite=True,
    )
    flatfield.attrs["source_component"] = "clearex/runtime_cache/source/data"
    shear = root.create_dataset(
        name="clearex/runtime_cache/results/shear_transform/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="float32",
        overwrite=True,
    )
    shear.attrs["source_component"] = (
        "clearex/runtime_cache/results/flatfield/latest/data"
    )

    voxel = registration_pipeline._extract_voxel_size_um_zyx(
        root,
        "clearex/runtime_cache/results/shear_transform/latest/data",
    )

    assert voxel == (6.0, 1.1, 1.1)


def test_build_edge_specs_only_keeps_overlapping_neighbors() -> None:
    nominal = {
        0: _translation_matrix(0.0),
        1: _translation_matrix(4.0),
        2: _translation_matrix(8.0),
    }

    edges = registration_pipeline._build_edge_specs(
        nominal,
        positions=(0, 1, 2),
        tile_extent_xyz=np.asarray([6.0, 4.0, 4.0], dtype=np.float64),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
    )

    assert [(edge.fixed_position, edge.moving_position) for edge in edges] == [
        (0, 1),
        (1, 2),
    ]


def test_parse_multiposition_stage_rows_accepts_mapping_rows() -> None:
    parsed = registration_pipeline._parse_multiposition_stage_rows(
        [
            {"x": 1.5, "y": -2.0, "z": 3.0, "theta": 4.0, "f": 5.0},
            {"X": 2.5, "Y": -1.0, "Z": 1.0, "THETA": 8.0, "F": 9.0},
        ]
    )

    assert len(parsed) == 2
    assert parsed[0] == {
        "x": pytest.approx(1.5),
        "y": pytest.approx(-2.0),
        "z": pytest.approx(3.0),
        "theta": pytest.approx(4.0),
        "f": pytest.approx(5.0),
    }
    assert parsed[1] == {
        "x": pytest.approx(2.5),
        "y": pytest.approx(-1.0),
        "z": pytest.approx(1.0),
        "theta": pytest.approx(8.0),
        "f": pytest.approx(9.0),
    }


def test_resolve_source_components_for_level_uses_requested_pyramid(
    tmp_path: Path,
) -> None:
    store_path = _create_registration_store(tmp_path)
    root = zarr.open_group(str(store_path), mode="r")

    source_component, pairwise_component, level = (
        registration_pipeline._resolve_source_components_for_level(
            root=root,
            requested_source_component="data",
            input_resolution_level=1,
        )
    )

    assert source_component == "data"
    assert pairwise_component == "data_pyramid/level_1"
    assert level == 1


def test_resolve_source_components_for_level_rejects_missing_level(
    tmp_path: Path,
) -> None:
    store_path = _create_registration_store(tmp_path, include_pyramid=False)
    root = zarr.open_group(str(store_path), mode="r")

    with pytest.raises(ValueError, match="input_resolution_level=1"):
        registration_pipeline._resolve_source_components_for_level(
            root=root,
            requested_source_component="data",
            input_resolution_level=1,
        )


def test_resolve_source_components_for_level_uses_source_adjacent_pyramid(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "registration_source_adjacent_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="results/shear_transform/latest/data",
        shape=(1, 2, 1, 4, 4, 4),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name="results/shear_transform/latest/data_pyramid/level_1",
        shape=(1, 2, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    source_component, pairwise_component, level = (
        registration_pipeline._resolve_source_components_for_level(
            root=root,
            requested_source_component="results/shear_transform/latest/data",
            input_resolution_level=1,
        )
    )

    assert source_component == "results/shear_transform/latest/data"
    assert pairwise_component == "results/shear_transform/latest/data_pyramid/level_1"
    assert level == 1


def test_position_centroid_anchor_prefers_central_tile() -> None:
    stage_rows = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0, "f": 0.0},
        {"x": 4.0, "y": 0.0, "z": 0.0, "theta": 0.0, "f": 0.0},
        {"x": 8.0, "y": 0.0, "z": 0.0, "theta": 0.0, "f": 0.0},
    ]

    anchor = registration_pipeline._position_centroid_anchor(
        stage_rows,
        SpatialCalibrationConfig(),
        positions=(0, 1, 2),
    )

    assert anchor == 1


def test_solve_with_pruning_recovers_translation_and_prunes_outlier() -> None:
    edges = [
        registration_pipeline._EdgeSpec(
            0, 1, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 10_000
        ),
        registration_pipeline._EdgeSpec(
            1, 2, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 10_000
        ),
        registration_pipeline._EdgeSpec(0, 2, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 1),
    ]
    results = [
        _edge_result(edges[0], _translation_matrix(1.0)),
        _edge_result(edges[1], _translation_matrix(1.0)),
        _edge_result(edges[2], _translation_matrix(20.0)),
    ]

    solved, active_mask, residuals = registration_pipeline._solve_with_pruning(
        positions=(0, 1, 2),
        edges=edges,
        edge_results=results,
        anchor_position=0,
        registration_type="translation",
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
    )

    assert active_mask.tolist() == [True, True, False]
    assert solved[1][:3, 3] == pytest.approx([1.0, 0.0, 0.0])
    assert solved[2][:3, 3] == pytest.approx([2.0, 0.0, 0.0])
    assert np.isnan(residuals[2])


def test_solve_with_pruning_preserves_disconnected_component_identity() -> None:
    edges = [
        registration_pipeline._EdgeSpec(0, 1, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 32),
        registration_pipeline._EdgeSpec(1, 2, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 32),
    ]
    results = [
        _edge_result(edges[0], _translation_matrix(1.0)),
        _edge_result(edges[1], np.eye(4, dtype=np.float64), success=False),
    ]

    solved, active_mask, _ = registration_pipeline._solve_with_pruning(
        positions=(0, 1, 2),
        edges=edges,
        edge_results=results,
        anchor_position=0,
        registration_type="translation",
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
    )

    assert active_mask.tolist() == [True, False]
    assert solved[1][:3, 3] == pytest.approx([1.0, 0.0, 0.0])
    assert solved[2] == pytest.approx(np.eye(4, dtype=np.float64))


def test_solve_with_pruning_recovers_rigid_transform() -> None:
    edge = registration_pipeline._EdgeSpec(
        0, 1, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 48
    )
    measured = np.eye(4, dtype=np.float64)
    measured[:3, :3] = Rotation.from_rotvec([0.0, 0.0, 0.15]).as_matrix()
    measured[:3, 3] = np.asarray([1.25, -0.5, 0.0], dtype=np.float64)

    solved, active_mask, residuals = registration_pipeline._solve_with_pruning(
        positions=(0, 1),
        edges=[edge],
        edge_results=[_edge_result(edge, measured)],
        anchor_position=0,
        registration_type="rigid",
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
    )

    assert active_mask.tolist() == [True]
    assert residuals[0] == pytest.approx(0.0, abs=1e-5)
    assert solved[1] == pytest.approx(measured, abs=1e-5)


def test_solve_with_pruning_recovers_similarity_transform() -> None:
    edge = registration_pipeline._EdgeSpec(
        0, 1, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 48
    )
    measured = registration_pipeline._matrix_from_pose(
        np.asarray([0.0, 0.0, 0.1, 0.75, 0.5, 0.0, np.log(1.08)], dtype=np.float64),
        "similarity",
    )

    solved, active_mask, residuals = registration_pipeline._solve_with_pruning(
        positions=(0, 1),
        edges=[edge],
        edge_results=[_edge_result(edge, measured)],
        anchor_position=0,
        registration_type="similarity",
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
    )

    assert active_mask.tolist() == [True]
    assert residuals[0] == pytest.approx(0.0, abs=1e-5)
    assert solved[1] == pytest.approx(measured, abs=1e-5)


def test_run_registration_analysis_fuses_output_and_writes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = _create_registration_store(tmp_path)
    original_compute = dask_module.compute

    def _sync_compute(*args, **kwargs):
        del kwargs
        return original_compute(*args, scheduler="synchronous")

    def _fake_pairwise(**kwargs):
        edge = kwargs["edge"]
        t_index = int(kwargs["t_index"])
        correction = np.eye(4, dtype=np.float64)
        if (
            t_index == 1
            and int(edge.fixed_position) == 1
            and int(edge.moving_position) == 2
        ):
            correction[0, 3] = -1.0
        return {
            "fixed_position": int(edge.fixed_position),
            "moving_position": int(edge.moving_position),
            "success": True,
            "reason": "",
            "correction_matrix_xyz": correction.tolist(),
            "overlap_voxels": int(edge.overlap_voxels),
            "nominal_overlap_pixels": int(edge.overlap_voxels),
        }

    monkeypatch.setattr(registration_pipeline.dask, "compute", _sync_compute)
    monkeypatch.setattr(
        registration_pipeline,
        "_register_pairwise_overlap",
        _fake_pairwise,
    )

    progress_updates: list[tuple[int, str]] = []
    summary = registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "rigid",
            "input_resolution_level": 1,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
        progress_callback=lambda percent, message: progress_updates.append(
            (int(percent), str(message))
        ),
    )

    root = zarr.open_group(str(store_path), mode="r")
    latest = root["clearex/results/registration/latest"]
    affines = latest["affines_tpx44"]

    assert summary.source_component == "data"
    assert summary.pairwise_source_component == "data_pyramid/level_1"
    assert summary.input_resolution_level == 1
    assert summary.anchor_positions == (1, 1)
    assert summary.edge_count == 2
    assert summary.output_shape_tpczyx == (2, 1, 1, 4, 4, 14)
    assert latest["edges_pe2"].shape == (2, 2)
    assert latest["pairwise_affines_tex44"].shape == (2, 2, 4, 4)
    assert latest["edge_status_te"].shape == (2, 2)
    assert latest["edge_residual_te"].shape == (2, 2)
    assert latest["transformed_bboxes_tpx6"].shape == (2, 3, 6)
    assert affines[1, 2, 0, 3] == pytest.approx(3.0, abs=1e-6)
    assert latest.attrs["pairwise_source_component"] == "data_pyramid/level_1"
    assert latest.attrs["input_resolution_level"] == 1
    assert latest.attrs["pairwise_overlap_zyx"] == [0, 0, 2]
    assert latest.attrs["max_pairwise_voxels"] == int(
        registration_pipeline._DEFAULT_MAX_PAIRWISE_VOXELS
    )
    assert latest.attrs["ants_iterations"] == list(
        registration_pipeline._ANTS_AFF_ITERATIONS
    )
    assert latest.attrs["ants_sampling_rate"] == pytest.approx(
        registration_pipeline._ANTS_RANDOM_SAMPLING_RATE
    )
    assert latest.attrs["use_phase_correlation"] is False
    assert latest.attrs["use_fft_initial_alignment"] is True
    assert progress_updates
    assert progress_updates[-1][0] == 100
    # Verify we get granular progress (not just start and end).
    progress_percents = [p for p, _ in progress_updates]
    assert len(progress_percents) >= 5

    fusion_summary = registration_pipeline.run_fusion_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "registration",
            "blend_mode": "feather",
            "blend_overlap_zyx": [0, 0, 2],
            "blend_exponent": 1.0,
        },
        client=None,
    )
    root = zarr.open_group(str(store_path), mode="r")
    fusion_latest = root["clearex/results/fusion/latest"]
    data = root["clearex/runtime_cache/results/fusion/latest/data"]
    assert fusion_summary.blend_mode == "feather"
    assert fusion_summary.output_shape_tpczyx == (2, 1, 1, 4, 4, 14)
    assert data.shape == (2, 1, 1, 4, 4, 14)
    assert int(data[0, 0, 0, 0, 0, 4]) == 10
    assert int(data[0, 0, 0, 0, 0, 5]) == 20
    assert int(data[0, 0, 0, 0, 0, 8]) == 20
    assert int(data[0, 0, 0, 0, 0, 9]) == 30
    assert fusion_latest.attrs["blend_mode"] == "feather"
    assert fusion_latest.attrs["blend_exponent"] == pytest.approx(1.0)
    assert fusion_latest.attrs["blend_overlap_zyx"] == [0, 0, 2]
    assert "blend_weights" in fusion_latest
    assert fusion_latest["blend_weights"]["profile_z"].shape == (4,)
    assert fusion_latest["blend_weights"]["profile_y"].shape == (4,)
    assert fusion_latest["blend_weights"]["profile_x"].shape == (6,)
    np.testing.assert_array_equal(
        fusion_latest["intensity_gains_tp"][:], np.ones((2, 3))
    )
    np.testing.assert_array_equal(
        fusion_latest["intensity_offsets_tp"][:],
        np.zeros((2, 3)),
    )


def test_run_registration_analysis_uses_namespaced_stage_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "registration_namespaced_metadata_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = root.create_dataset(
        name="data",
        shape=(1, 2, 1, 3, 3, 4),
        chunks=(1, 1, 1, 3, 3, 4),
        dtype="uint16",
        overwrite=True,
    )
    data[:] = np.uint16(1)

    metadata_group = root.require_group("clearex/metadata")
    metadata_group.attrs.update(
        {
            "schema": "clearex.store_metadata.v1",
            "spatial_calibration": {
                "schema": SPATIAL_CALIBRATION_SCHEMA,
                "stage_axis_map_zyx": {"z": "+z", "y": "+y", "x": "+x"},
                "theta_mode": "rotate_zy_about_x",
            },
            "stage_rows": [
                {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0, "f": 0.0},
                {"x": 4.0, "y": 0.0, "z": 0.0, "theta": 0.0, "f": 0.0},
            ],
        }
    )

    def _raise_after_stage_metadata(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("sentinel-after-stage-metadata")

    monkeypatch.setattr(
        registration_pipeline,
        "_build_nominal_transforms_xyz",
        _raise_after_stage_metadata,
    )

    with pytest.raises(RuntimeError, match="sentinel-after-stage-metadata"):
        registration_pipeline.run_registration_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "registration_channel": 0,
                "registration_type": "rigid",
                "input_resolution_level": 0,
                "anchor_mode": "central",
                "blend_mode": "feather",
            },
            client=None,
        )


def test_run_registration_analysis_gain_compensated_feather_corrects_overlap_intensity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )
    root = zarr.open_group(str(store_path), mode="a")
    data = root["data"]
    data[0, 0, 0] = np.full((4, 4, 6), 10, dtype=np.uint16)
    data[0, 1, 0] = np.full((4, 4, 6), 20, dtype=np.uint16)

    def _sync_compute(*tasks, scheduler=None):
        del scheduler
        results = []
        for task in tasks:
            if hasattr(task, "compute"):
                results.append(task.compute())
            else:
                results.append(task)
        return tuple(results)

    def _fake_pairwise(**kwargs):
        edge = kwargs["edge"]
        correction = np.eye(4, dtype=np.float64)
        return _edge_result(edge, correction)

    def _fake_intensity(**kwargs):
        edge = kwargs["edge"]
        return {
            "fixed_position": int(edge.fixed_position),
            "moving_position": int(edge.moving_position),
            "success": True,
            "overlap_voxels": int(edge.overlap_voxels),
            "intensity_success": True,
            "intensity_scale": 0.5,
            "intensity_offset": 0.0,
            "intensity_samples": int(edge.overlap_voxels),
        }

    monkeypatch.setattr(registration_pipeline.dask, "compute", _sync_compute)
    monkeypatch.setattr(
        registration_pipeline,
        "_register_pairwise_overlap",
        _fake_pairwise,
    )
    monkeypatch.setattr(
        registration_pipeline,
        "_estimate_pairwise_overlap_intensity",
        _fake_intensity,
    )

    registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
    )

    fusion_summary = registration_pipeline.run_fusion_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "registration",
            "blend_mode": "gain_compensated_feather",
            "blend_overlap_zyx": [0, 0, 2],
            "blend_exponent": 1.0,
            "gain_clip_range": [0.25, 4.0],
        },
        client=None,
    )

    root = zarr.open_group(str(store_path), mode="r")
    latest = root["clearex/results/fusion/latest"]
    fused = root["clearex/runtime_cache/results/fusion/latest/data"]

    assert fusion_summary.blend_mode == "gain_compensated_feather"
    assert latest.attrs["blend_mode"] == "gain_compensated_feather"
    np.testing.assert_allclose(
        latest["intensity_gains_tp"][:],
        np.asarray([[1.0, 0.5]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        latest["intensity_offsets_tp"][:],
        np.zeros((1, 2), dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused[0, 0, 0, 0, 0, 4:6], dtype=np.float32),
        np.asarray([10.0, 10.0], dtype=np.float32),
        atol=0.5,
    )


def test_run_fusion_analysis_parallelizes_intensity_estimation_with_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )
    root = zarr.open_group(str(store_path), mode="a")
    data = root["data"]
    data[0, 0, 0] = np.full((4, 4, 6), 10, dtype=np.uint16)
    data[0, 1, 0] = np.full((4, 4, 6), 20, dtype=np.uint16)

    def _sync_compute(*tasks, scheduler=None):
        del scheduler
        results = []
        for task in tasks:
            if hasattr(task, "compute"):
                results.append(task.compute())
            else:
                results.append(task)
        return tuple(results)

    def _fake_pairwise(**kwargs):
        edge = kwargs["edge"]
        correction = np.eye(4, dtype=np.float64)
        return _edge_result(edge, correction)

    def _fake_intensity(**kwargs):
        edge = kwargs["edge"]
        return {
            "fixed_position": int(edge.fixed_position),
            "moving_position": int(edge.moving_position),
            "success": True,
            "overlap_voxels": int(edge.overlap_voxels),
            "intensity_success": True,
            "intensity_scale": 0.5,
            "intensity_offset": 0.0,
            "intensity_samples": int(edge.overlap_voxels),
        }

    monkeypatch.setattr(registration_pipeline.dask, "compute", _sync_compute)
    monkeypatch.setattr(
        registration_pipeline,
        "_register_pairwise_overlap",
        _fake_pairwise,
    )
    monkeypatch.setattr(
        registration_pipeline,
        "_estimate_pairwise_overlap_intensity",
        _fake_intensity,
    )

    registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
    )

    from dask.distributed import Client, LocalCluster

    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    ) as cluster:
        with Client(cluster) as client:
            compute_calls: list[int] = []
            original_compute = client.compute

            def _counting_compute(*args, **kwargs):
                compute_calls.append(len(args[0]) if args else 0)
                return original_compute(*args, **kwargs)

            client.compute = _counting_compute  # type: ignore[method-assign]

            monkeypatch.setattr(
                registration_pipeline,
                "_estimate_worker_thread_capacity",
                lambda _client: 1,
            )
            monkeypatch.setattr(
                registration_pipeline,
                "_process_and_write_registration_chunk",
                lambda **kwargs: 1,
            )

            registration_pipeline.run_fusion_analysis(
                zarr_path=store_path,
                parameters={
                    "input_source": "registration",
                    "blend_mode": "gain_compensated_feather",
                    "blend_overlap_zyx": [0, 0, 2],
                    "blend_exponent": 1.0,
                    "gain_clip_range": [0.25, 4.0],
                },
                client=client,
            )

    root = zarr.open_group(str(store_path), mode="r")
    latest = root["clearex/results/fusion/latest"]
    np.testing.assert_allclose(
        latest["intensity_gains_tp"][:],
        np.asarray([[1.0, 0.5]], dtype=np.float32),
        atol=1e-6,
    )
    assert compute_calls == [1]


def test_source_subvolume_for_overlap_returns_tighter_slices() -> None:
    """_source_subvolume_for_overlap should return slices narrower than the full tile."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    source_shape_zyx = (20, 100, 40)
    reference_origin_xyz = np.asarray([5.0, 10.0, 2.0], dtype=np.float64)
    reference_shape_zyx = (4, 8, 6)
    voxel_size = (1.0, 1.0, 1.0)

    slices, adjusted = registration_pipeline._source_subvolume_for_overlap(
        transform,
        reference_origin_xyz=reference_origin_xyz,
        reference_shape_zyx=reference_shape_zyx,
        source_shape_zyx=source_shape_zyx,
        voxel_size_um_zyx=voxel_size,
        padding=2,
    )

    for axis_slice, full_size in zip(slices, source_shape_zyx):
        assert axis_slice.start >= 0
        assert axis_slice.stop <= full_size
        assert (axis_slice.stop - axis_slice.start) <= full_size

    # The sub-volume must still be usable for resampling: verify that
    # resampling the sub-volume with the adjusted transform produces the same
    # result as resampling the full tile with the original transform.
    rng = np.random.default_rng(42)
    full_volume = rng.random(source_shape_zyx).astype(np.float32)
    sub_volume = full_volume[slices[0], slices[1], slices[2]]

    full_result = registration_pipeline._resample_source_to_world_grid(
        full_volume,
        transform,
        reference_origin_xyz=reference_origin_xyz,
        reference_shape_zyx=reference_shape_zyx,
        voxel_size_um_zyx=voxel_size,
        order=1,
        cval=0.0,
    )
    sub_result = registration_pipeline._resample_source_to_world_grid(
        sub_volume,
        adjusted,
        reference_origin_xyz=reference_origin_xyz,
        reference_shape_zyx=reference_shape_zyx,
        voxel_size_um_zyx=voxel_size,
        order=1,
        cval=0.0,
    )
    np.testing.assert_allclose(sub_result, full_result, atol=1e-5)


def test_source_subvolume_for_overlap_returns_empty_slices_outside_source() -> None:
    """Out-of-bounds overlap crops should not fall back to the full tile."""
    transform = np.eye(4, dtype=np.float64)
    slices, adjusted = registration_pipeline._source_subvolume_for_overlap(
        transform,
        reference_origin_xyz=np.asarray([1000.0, 1000.0, 1000.0], dtype=np.float64),
        reference_shape_zyx=(8, 8, 8),
        source_shape_zyx=(20, 100, 40),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        padding=2,
    )

    assert slices == (slice(0, 0), slice(0, 0), slice(0, 0))
    assert registration_pipeline._slices_zyx_are_empty(slices) is True
    np.testing.assert_array_equal(adjusted, transform)


def test_register_pairwise_overlap_returns_failure_for_empty_subvolume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairwise registration should fail cleanly when no source crop remains."""
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )
    edge = registration_pipeline._EdgeSpec(
        fixed_position=0,
        moving_position=1,
        overlap_bbox_xyz=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
        overlap_voxels=8,
    )

    monkeypatch.setattr(
        registration_pipeline,
        "_source_subvolume_for_overlap",
        lambda *args, **kwargs: (
            (slice(0, 0), slice(0, 0), slice(0, 0)),
            np.eye(4, dtype=np.float64),
        ),
    )

    result = registration_pipeline._register_pairwise_overlap(
        zarr_path=str(store_path),
        source_component="data",
        t_index=0,
        registration_channel=0,
        edge=edge,
        nominal_fixed_transform_xyz=np.eye(4, dtype=np.float64),
        nominal_moving_transform_xyz=np.eye(4, dtype=np.float64),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        overlap_zyx=(0, 0, 0),
        registration_type="translation",
    )

    assert result["success"] is False
    assert result["reason"] == "overlap_outside_source_bounds"
    assert result["nominal_overlap_pixels"] == 0


def test_process_and_write_registration_chunk_skips_empty_source_subvolumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fusion should skip tiles whose chunk crop maps outside the source tile."""
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=1,
        channels=1,
        shape_zyx=(2, 3, 4),
        include_pyramid=False,
    )
    _, output_component, affines_component, blend_weights_component = (
        registration_pipeline._prepare_output_group(
            analysis_name="fusion",
            zarr_path=store_path,
            source_component="data",
            parameters={"input_source": "data", "blend_mode": "feather"},
            output_shape_tpczyx=(1, 1, 1, 2, 3, 4),
            output_chunks_tpczyx=(1, 1, 1, 2, 3, 4),
            voxel_size_um_zyx=(1.0, 1.0, 1.0),
            voxel_size_resolution_source="data",
            output_origin_xyz=(0.0, 0.0, 0.0),
            source_tile_shape_zyx=(2, 3, 4),
            blend_mode="feather",
            overlap_zyx=(1, 1, 1),
            blend_exponent=1.0,
        )
    )
    root = zarr.open_group(str(store_path), mode="a")
    auxiliary_root = registration_pipeline.analysis_auxiliary_root("fusion")
    root[affines_component][0, 0] = np.eye(4, dtype=np.float64)
    auxiliary_group = root[auxiliary_root]
    auxiliary_group.create_array(
        name="transformed_bboxes_tpx6",
        data=np.asarray([[[0.0, 0.0, 0.0, 4.0, 3.0, 2.0]]], dtype=np.float64),
        overwrite=True,
    )
    auxiliary_group.create_array(
        name="intensity_gains_tp",
        data=np.ones((1, 1), dtype=np.float32),
        overwrite=True,
    )
    auxiliary_group.create_array(
        name="intensity_offsets_tp",
        data=np.zeros((1, 1), dtype=np.float32),
        overwrite=True,
    )

    monkeypatch.setattr(
        registration_pipeline,
        "_source_subvolume_for_overlap",
        lambda *args, **kwargs: (
            (slice(0, 0), slice(0, 0), slice(0, 0)),
            np.eye(4, dtype=np.float64),
        ),
    )

    written = registration_pipeline._process_and_write_registration_chunk(
        zarr_path=str(store_path),
        source_component="data",
        output_component=output_component,
        affines_component=affines_component,
        transformed_bboxes_component=(f"{auxiliary_root}/transformed_bboxes_tpx6"),
        blend_weights_component=blend_weights_component,
        intensity_gains_component=f"{auxiliary_root}/intensity_gains_tp",
        intensity_offsets_component=f"{auxiliary_root}/intensity_offsets_tp",
        t_index=0,
        c_index=0,
        z_bounds=(0, 2),
        y_bounds=(0, 3),
        x_bounds=(0, 4),
        output_origin_xyz=(0.0, 0.0, 0.0),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        output_dtype="float32",
    )

    assert written == 1
    np.testing.assert_array_equal(
        np.asarray(root[output_component][:], dtype=np.float32),
        np.zeros((1, 1, 1, 2, 3, 4), dtype=np.float32),
    )


def test_process_and_write_registration_chunk_limits_nested_zarr_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fusion workers should bound nested Zarr I/O fan-out inside each task."""

    class _ConfigAssertingArray:
        def __init__(self, data: np.ndarray) -> None:
            self._data = np.asarray(data)
            self.shape = self._data.shape

        def __getitem__(self, key: object) -> np.ndarray:
            assert zarr.config.get("async.concurrency") == 1
            assert zarr.config.get("threading.max_workers") == 1
            return np.asarray(self._data[key])

    class _ConfigAssertingTarget:
        def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
            self._data = np.zeros(shape, dtype=dtype)
            self.shape = self._data.shape

        def __setitem__(self, key: object, value: object) -> None:
            assert zarr.config.get("async.concurrency") == 1
            assert zarr.config.get("threading.max_workers") == 1
            self._data[key] = value

    class _FakeGroup:
        def __init__(
            self, mapping: dict[str, object], attrs: dict[str, object]
        ) -> None:
            self._mapping = dict(mapping)
            self.attrs = dict(attrs)

        def __getitem__(self, key: str) -> object:
            return self._mapping[key]

    class _FakeRoot:
        def __init__(self, mapping: dict[str, object]) -> None:
            self._mapping = dict(mapping)

        def __getitem__(self, key: str) -> object:
            return self._mapping[key]

    source = _ConfigAssertingArray(np.ones((1, 1, 1, 2, 3, 4), dtype=np.uint16))
    affines = _ConfigAssertingArray(np.asarray([[np.eye(4, dtype=np.float64)]]))
    transformed_bboxes = _ConfigAssertingArray(
        np.asarray([[[0.0, 0.0, 0.0, 4.0, 3.0, 2.0]]], dtype=np.float64)
    )
    intensity_gains = _ConfigAssertingArray(np.ones((1, 1), dtype=np.float32))
    intensity_offsets = _ConfigAssertingArray(np.zeros((1, 1), dtype=np.float32))
    output = _ConfigAssertingTarget((1, 1, 1, 2, 3, 4), np.dtype(np.float32))
    blend_group = _FakeGroup(
        mapping={
            "profile_z": _ConfigAssertingArray(np.ones((2,), dtype=np.float32)),
            "profile_y": _ConfigAssertingArray(np.ones((3,), dtype=np.float32)),
            "profile_x": _ConfigAssertingArray(np.ones((4,), dtype=np.float32)),
        },
        attrs={"blend_mode": "feather"},
    )
    fake_root = _FakeRoot(
        {
            "data": source,
            "fusion/blend_weights": blend_group,
            "fusion/intensity_gains_tp": intensity_gains,
            "fusion/intensity_offsets_tp": intensity_offsets,
            "registration/affines_tpx44": affines,
            "registration/transformed_bboxes_tpx6": transformed_bboxes,
            "fusion/output": output,
        }
    )

    monkeypatch.setattr(
        registration_pipeline.zarr, "open_group", lambda *args, **kwargs: fake_root
    )
    monkeypatch.setattr(
        registration_pipeline,
        "_source_subvolume_for_overlap",
        lambda *args, **kwargs: (
            (slice(0, 2), slice(0, 3), slice(0, 4)),
            np.eye(4, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        registration_pipeline,
        "_resample_source_to_world_grid",
        lambda source_volume, *args, **kwargs: np.asarray(
            source_volume, dtype=np.float32
        ),
    )

    written = registration_pipeline._process_and_write_registration_chunk(
        zarr_path="ignored.ome.zarr",
        source_component="data",
        output_component="fusion/output",
        affines_component="registration/affines_tpx44",
        transformed_bboxes_component="registration/transformed_bboxes_tpx6",
        blend_weights_component="fusion/blend_weights",
        intensity_gains_component="fusion/intensity_gains_tp",
        intensity_offsets_component="fusion/intensity_offsets_tp",
        t_index=0,
        c_index=0,
        z_bounds=(0, 2),
        y_bounds=(0, 3),
        x_bounds=(0, 4),
        output_origin_xyz=(0.0, 0.0, 0.0),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        output_dtype="float32",
    )

    assert written == 1
    np.testing.assert_allclose(
        output._data,
        np.ones((1, 1, 1, 2, 3, 4), dtype=np.float32),
    )


def test_estimate_fusion_chunk_bytes_uses_overlap_bounded_source_subvolume() -> None:
    """Fusion memory estimates should follow the clipped overlap crop, not full tile."""
    chunk_bounds_zyx = ((10, 14), (20, 24), (30, 34))
    chunk_shape_zyx = tuple(int(bounds[1] - bounds[0]) for bounds in chunk_bounds_zyx)
    source_shape_zyx = (100, 100, 100)
    affines_p44 = np.asarray([np.eye(4, dtype=np.float64)])
    transformed_bboxes_px6 = np.asarray(
        [[0.0, 0.0, 0.0, 100.0, 100.0, 100.0]],
        dtype=np.float64,
    )

    estimated = registration_pipeline._estimate_fusion_chunk_bytes(
        chunk_bounds_zyx=chunk_bounds_zyx,
        output_origin_xyz=(0.0, 0.0, 0.0),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        source_shape_zyx=source_shape_zyx,
        affines_p44=affines_p44,
        transformed_bboxes_px6=transformed_bboxes_px6,
    )

    source_slices, _ = registration_pipeline._source_subvolume_for_overlap(
        np.eye(4, dtype=np.float64),
        reference_origin_xyz=np.asarray([30.0, 20.0, 10.0], dtype=np.float64),
        reference_shape_zyx=chunk_shape_zyx,
        source_shape_zyx=source_shape_zyx,
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
    )
    bounded_source_voxels = int(
        np.prod(
            [
                int(axis_slice.stop) - int(axis_slice.start)
                for axis_slice in source_slices
            ]
        )
    )
    chunk_voxels = int(np.prod(chunk_shape_zyx))
    expected = (2 * chunk_voxels + 4 * bounded_source_voxels) * 4
    full_tile_estimate = (2 * chunk_voxels + 4 * int(np.prod(source_shape_zyx))) * 4

    assert estimated == expected
    assert estimated < full_tile_estimate


def test_downsample_crop_for_registration_reduces_volume() -> None:
    """_downsample_crop_for_registration should reduce volume beyond the budget."""
    rng = np.random.default_rng(7)
    volume = rng.random((20, 100, 40)).astype(np.float32)
    voxel_size = (1.0, 1.0, 1.0)

    # Volume is 80 000 voxels; set budget to 10 000.
    down, eff_voxel = registration_pipeline._downsample_crop_for_registration(
        volume,
        max_voxels=10_000,
        voxel_size_um_zyx=voxel_size,
    )

    assert down.size < volume.size
    assert all(float(e) > float(o) for e, o in zip(eff_voxel, voxel_size))


def test_downsample_crop_for_registration_noop_when_small() -> None:
    """_downsample_crop_for_registration should be a noop for small volumes."""
    rng = np.random.default_rng(8)
    volume = rng.random((4, 4, 4)).astype(np.float32)
    voxel_size = (1.0, 1.0, 1.0)

    down, eff_voxel = registration_pipeline._downsample_crop_for_registration(
        volume,
        max_voxels=500_000,
        voxel_size_um_zyx=voxel_size,
    )

    assert down is volume
    assert eff_voxel == pytest.approx(voxel_size)


def test_pairwise_registration_grid_downsample_bounds_voxel_budget() -> None:
    """Pairwise registration should derive a coarse grid before allocation."""
    shape, voxel_size, factor = (
        registration_pipeline._registration_grid_for_pairwise_budget(
            crop_shape_zyx=(1070, 22780, 768),
            voxel_size_um_zyx=(3.5355, 1.0599, 1.0599),
            max_pairwise_voxels=10_000_000,
        )
    )

    assert factor > 1
    assert int(np.prod(shape)) <= 10_000_000
    assert shape == (83, 1753, 60)
    assert voxel_size == pytest.approx(
        (
            3.5355 * factor,
            1.0599 * factor,
            1.0599 * factor,
        )
    )


def test_register_pairwise_overlap_reads_strided_sources_for_large_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large pairwise crops should be downsampled at source read/resample time."""

    class _FakeSource:
        shape = (1, 2, 1, 100, 100, 100)

    class _FakeRoot:
        def __getitem__(self, key: str) -> _FakeSource:
            assert key == "data"
            return _FakeSource()

    read_selections: list[object] = []
    resample_calls: list[tuple[tuple[int, int, int], tuple[float, float, float]]] = []

    def _fake_worker_read(array: object, selection: object, *, dtype: object = None):
        _ = array
        _ = dtype
        read_selections.append(selection)
        slices = selection[3:]
        shape = tuple(len(range(s.start, s.stop, s.step or 1)) for s in slices)
        return np.ones(shape, dtype=np.float32)

    def _fake_resample(
        source_zyx: np.ndarray,
        local_to_world_xyz: np.ndarray,
        *,
        reference_origin_xyz: np.ndarray,
        reference_shape_zyx: tuple[int, int, int],
        voxel_size_um_zyx: tuple[float, float, float],
        order: int,
        cval: float,
    ) -> np.ndarray:
        _ = source_zyx
        _ = local_to_world_xyz
        _ = reference_origin_xyz
        _ = order
        _ = cval
        resample_calls.append((reference_shape_zyx, voxel_size_um_zyx))
        grid = np.indices(reference_shape_zyx, dtype=np.float32)
        return np.asarray(grid[0] + grid[1] + grid[2] + 1.0, dtype=np.float32)

    monkeypatch.setattr(registration_pipeline.zarr, "open_group", lambda *a, **k: _FakeRoot())
    monkeypatch.setattr(registration_pipeline, "_worker_zarr_read", _fake_worker_read)
    monkeypatch.setattr(
        registration_pipeline,
        "_resample_source_to_world_grid",
        _fake_resample,
    )
    monkeypatch.setattr(
        registration_pipeline,
        "_phase_cross_correlation",
        lambda *a, **k: (np.zeros(3, dtype=np.float64), 0.0, 0.0),
    )

    result = registration_pipeline._register_pairwise_overlap(
        zarr_path="unused.zarr",
        source_component="data",
        t_index=0,
        registration_channel=0,
        edge=registration_pipeline._EdgeSpec(
            fixed_position=0,
            moving_position=1,
            overlap_bbox_xyz=((0.0, 40.0), (0.0, 40.0), (0.0, 40.0)),
            overlap_voxels=40 * 40 * 40,
        ),
        nominal_fixed_transform_xyz=np.eye(4, dtype=np.float64),
        nominal_moving_transform_xyz=np.eye(4, dtype=np.float64),
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        overlap_zyx=(0, 0, 0),
        registration_type="translation",
        max_pairwise_voxels=1_000,
        use_phase_correlation=True,
    )

    assert result["success"] is True
    assert result["reason"] == "phase_correlation"
    assert len(read_selections) == 2
    assert all(selection[3].step == 4 for selection in read_selections)
    assert all(selection[4].step == 4 for selection in read_selections)
    assert all(selection[5].step == 4 for selection in read_selections)
    assert resample_calls == [
        ((10, 10, 10), (4.0, 4.0, 4.0)),
        ((10, 10, 10), (4.0, 4.0, 4.0)),
    ]


def test_phase_correlation_translation_recovers_known_shift() -> None:
    """_phase_correlation_translation should recover a known translation."""
    pytest.importorskip("skimage")
    rng = np.random.default_rng(99)
    fixed = rng.random((16, 32, 32)).astype(np.float32) * 100.0
    # Shift by 3 voxels in Z, 5 in Y, -2 in X.
    from scipy.ndimage import shift as ndimage_shift

    moving = ndimage_shift(fixed, (3.0, 5.0, -2.0), order=1, mode="constant")
    voxel_size = (2.0, 1.0, 0.5)

    correction = registration_pipeline._phase_correlation_translation(
        fixed,
        moving,
        voxel_size_um_zyx=voxel_size,
    )

    # phase_cross_correlation returns the shift to align moving → fixed.
    # moving was shifted by (3, 5, -2) ZYX from fixed, so the correction
    # shift is (-3, -5, 2) ZYX voxels → (2*0.5, -5*1.0, -3*2.0) XYZ µm.
    expected_xyz = np.asarray([1.0, -5.0, -6.0], dtype=np.float64)
    np.testing.assert_allclose(correction[:3, 3], expected_xyz, atol=1.0)
    np.testing.assert_allclose(correction[:3, :3], np.eye(3), atol=1e-10)


def test_estimate_linear_intensity_map_recovers_scale_and_offset() -> None:
    moving = np.linspace(1.0, 20.0, 128, dtype=np.float32).reshape(4, 4, 8)
    fixed = (1.5 * moving) + 7.0

    success, scale, offset, samples = (
        registration_pipeline._estimate_linear_intensity_map(
            fixed,
            moving,
            gain_clip_range=(0.25, 4.0),
        )
    )

    assert success is True
    assert samples == moving.size
    assert scale == pytest.approx(1.5, rel=1e-3)
    assert offset == pytest.approx(7.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Separable blend-weight profile tests
# ---------------------------------------------------------------------------


class TestBlendWeightProfiles:
    """Verify that the separable 1D profiles reproduce the full 3D volume."""

    def test_profiles_reconstruct_full_volume(self):
        shape = (8, 12, 10)
        overlap = (4, 6, 5)
        full_vol = registration_pipeline._blend_weight_volume(
            shape,
            blend_mode="feather",
            overlap_zyx=overlap,
        )
        pz, py, px = registration_pipeline._blend_weight_profiles(
            shape,
            blend_mode="feather",
            overlap_zyx=overlap,
        )
        reconstructed = (
            pz[:, np.newaxis, np.newaxis]
            * py[np.newaxis, :, np.newaxis]
            * px[np.newaxis, np.newaxis, :]
        )
        np.testing.assert_allclose(reconstructed, full_vol, atol=1e-7)

    def test_subvolume_matches_full_slice(self):
        shape = (8, 12, 10)
        overlap = (4, 6, 5)
        full_vol = registration_pipeline._blend_weight_volume(
            shape,
            blend_mode="feather",
            overlap_zyx=overlap,
        )
        pz, py, px = registration_pipeline._blend_weight_profiles(
            shape,
            blend_mode="feather",
            overlap_zyx=overlap,
        )
        slices = (slice(2, 6), slice(3, 9), slice(1, 7))
        sub = registration_pipeline._blend_weight_subvolume_from_profiles(
            pz,
            py,
            px,
            slices,
        )
        np.testing.assert_allclose(sub, full_vol[slices], atol=1e-7)

    def test_average_mode_profiles(self):
        shape = (4, 6, 8)
        pz, py, px = registration_pipeline._blend_weight_profiles(
            shape,
            blend_mode="average",
            overlap_zyx=(2, 3, 4),
        )
        np.testing.assert_array_equal(pz, np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(py, np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(px, np.ones(8, dtype=np.float32))

    def test_center_weighted_profiles_fall_off_faster_than_feather(self):
        shape = (1, 1, 10)
        feather = registration_pipeline._blend_weight_profiles(
            shape,
            blend_mode="feather",
            overlap_zyx=(0, 0, 5),
            blend_exponent=1.0,
        )[2]
        center_weighted = registration_pipeline._blend_weight_profiles(
            shape,
            blend_mode="center_weighted",
            overlap_zyx=(0, 0, 5),
            blend_exponent=1.0,
        )[2]
        assert center_weighted[1] < feather[1]
        assert center_weighted[2] < feather[2]
        assert center_weighted.max() == pytest.approx(1.0)

    def test_memory_estimate_positive(self):
        est = registration_pipeline._estimate_fusion_chunk_bytes(
            chunk_bounds_zyx=((0, 64), (0, 64), (0, 64)),
            output_origin_xyz=(0.0, 0.0, 0.0),
            voxel_size_um_zyx=(1.0, 1.0, 1.0),
            source_shape_zyx=(576, 30730, 5112),
            affines_p44=np.repeat(
                np.eye(4, dtype=np.float64)[None, :, :],
                4,
                axis=0,
            ),
            transformed_bboxes_px6=np.repeat(
                np.asarray([[0.0, 0.0, 0.0, 64.0, 64.0, 64.0]], dtype=np.float64),
                4,
                axis=0,
            ),
        )
        assert est > 0
        # Overlap-bounded estimates should remain well below the full-tile bound.
        assert est < 2_000_000_000_000  # < 2 TiB sanity check
