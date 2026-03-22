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
            "blend_mode": "feather",
            "overlap_zyx": [0, 0, 2],
        },
        client=None,
        progress_callback=lambda percent, message: progress_updates.append(
            (int(percent), str(message))
        ),
    )

    root = zarr.open_group(str(store_path), mode="r")
    latest = root["results/registration/latest"]
    data = latest["data"]
    affines = latest["affines_tpx44"]

    assert summary.source_component == "data"
    assert summary.pairwise_source_component == "data_pyramid/level_1"
    assert summary.input_resolution_level == 1
    assert summary.anchor_positions == (1, 1)
    assert summary.edge_count == 2
    assert summary.output_shape_tpczyx == (2, 1, 1, 4, 4, 14)
    assert data.shape == (2, 1, 1, 4, 4, 14)
    assert latest["edges_pe2"].shape == (2, 2)
    assert latest["pairwise_affines_tex44"].shape == (2, 2, 4, 4)
    assert latest["edge_status_te"].shape == (2, 2)
    assert latest["edge_residual_te"].shape == (2, 2)
    assert latest["transformed_bboxes_tpx6"].shape == (2, 3, 6)
    assert affines[1, 2, 0, 3] == pytest.approx(3.0, abs=1e-6)
    assert int(data[0, 0, 0, 0, 0, 4]) == 10
    assert int(data[0, 0, 0, 0, 0, 5]) == 20
    assert int(data[0, 0, 0, 0, 0, 8]) == 20
    assert int(data[0, 0, 0, 0, 0, 9]) == 30
    assert latest.attrs["pairwise_source_component"] == "data_pyramid/level_1"
    assert latest.attrs["input_resolution_level"] == 1
    assert latest.attrs["blend_mode"] == "feather"
    assert "blend_weights_zyx" in latest
    assert latest["blend_weights_zyx"].shape == (4, 4, 6)
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

