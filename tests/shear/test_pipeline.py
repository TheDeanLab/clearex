#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from clearex.shear import pipeline as shear_pipeline
from clearex.shear.pipeline import run_shear_transform_analysis


def _build_extreme_slanted_volume(
    *,
    angle_deg: float,
    shape_zyx: tuple[int, int, int],
    edge_width_x: int = 4,
    slab_thickness_z: int = 6,
    z_offset: int = 8,
) -> np.ndarray:
    """Create a synthetic volume with matching slanted structures at X extremes.

    Parameters
    ----------
    angle_deg : float
        Physical tilt angle in degrees.
    shape_zyx : tuple[int, int, int]
        Target ``(z, y, x)`` shape.
    edge_width_x : int, default=4
        Width of each X-extreme slab in voxels.
    slab_thickness_z : int, default=6
        Slab thickness in Z.
    z_offset : int, default=8
        Base Z offset for the slab centerline.

    Returns
    -------
    numpy.ndarray
        Synthetic ``(z, y, x)`` volume with values in ``[0, 1]``.
    """
    z_count, y_count, x_count = (int(v) for v in shape_zyx)
    volume = np.zeros((z_count, y_count, x_count), dtype=np.float32)
    slope = float(np.tan(np.deg2rad(float(angle_deg))))
    half_thickness = int(max(1, slab_thickness_z // 2))
    left_stop = max(1, min(x_count, int(edge_width_x)))
    right_start = max(0, min(x_count - 1, x_count - int(edge_width_x)))
    for y_index in range(int(y_count)):
        z_center = int(round(float(z_offset) + slope * float(y_index)))
        z_start = max(0, z_center - half_thickness)
        z_stop = min(z_count, z_center + half_thickness + 1)
        if z_start >= z_stop:
            continue
        volume[z_start:z_stop, y_index, 0:left_stop] = 1.0
        volume[z_start:z_stop, y_index, right_start:x_count] = 1.0
    return volume


def test_estimate_shear_yz_deg_from_source_extremes_recovers_tilt() -> None:
    angle_deg = 18.0
    source = np.zeros((1, 1, 1, 96, 128, 40), dtype=np.float32)
    source[0, 0, 0, :, :, :] = _build_extreme_slanted_volume(
        angle_deg=angle_deg,
        shape_zyx=(96, 128, 40),
    )
    estimated = shear_pipeline._estimate_shear_yz_deg_from_source_extremes(
        source_array=source,
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        parameters={
            "auto_estimate_t_index": 0,
            "auto_estimate_p_index": 0,
            "auto_estimate_c_index": 0,
            "auto_estimate_extreme_fraction_x": 0.10,
            "auto_estimate_zy_stride": 1,
            "auto_estimate_signal_fraction": 0.05,
        },
    )
    assert estimated is not None
    assert np.isclose(float(estimated), float(angle_deg), atol=1.5)


def test_run_shear_transform_auto_estimate_updates_applied_shear(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "shear_auto_estimate.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    source = np.zeros((1, 1, 1, 48, 72, 28), dtype=np.float32)
    expected_angle_deg = 14.0
    source[0, 0, 0, :, :, :] = _build_extreme_slanted_volume(
        angle_deg=expected_angle_deg,
        shape_zyx=(48, 72, 28),
        edge_width_x=3,
        slab_thickness_z=5,
        z_offset=6,
    )
    root.create_dataset(
        name="data",
        data=source,
        chunks=(1, 1, 1, 16, 24, 14),
        dtype="float32",
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [1.0, 1.0, 1.0]

    summary = run_shear_transform_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "auto_estimate_shear_yz": True,
            "auto_estimate_extreme_fraction_x": 0.10,
            "auto_estimate_zy_stride": 1,
            "auto_estimate_signal_fraction": 0.05,
            "auto_rotate_from_shear": True,
            "interpolation": "linear",
            "output_dtype": "float32",
            "roi_padding_zyx": [1, 1, 1],
        },
        client=None,
    )

    observed_angle_deg = float(np.degrees(np.arctan(float(summary.applied_shear_yz))))
    assert np.isclose(observed_angle_deg, expected_angle_deg, atol=2.0)


def test_run_shear_transform_identity_preserves_data(tmp_path: Path) -> None:
    store_path = tmp_path / "shear_identity.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 4 * 4 * 4, dtype=np.uint16).reshape(
        (1, 1, 1, 4, 4, 4)
    )
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [1.0, 1.0, 1.0]

    summary = run_shear_transform_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "interpolation": "nearestneighbor",
            "output_dtype": "uint16",
            "shear_xy": 0.0,
            "shear_xz": 0.0,
            "shear_yz": 0.0,
            "rotation_deg_x": 0.0,
            "rotation_deg_y": 0.0,
            "rotation_deg_z": 0.0,
            "auto_rotate_from_shear": False,
            "roi_padding_zyx": [1, 1, 1],
        },
        client=None,
    )

    output = np.asarray(
        zarr.open_group(str(store_path), mode="r")["results/shear_transform/latest/data"]
    )
    assert summary.data_component == "results/shear_transform/latest/data"
    assert output.shape == data.shape
    np.testing.assert_array_equal(output, data)


def test_run_shear_transform_emits_larger_bounds_for_nonzero_shear(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "shear_skew.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.zeros((1, 1, 1, 6, 6, 6), dtype=np.float32)
    data[0, 0, 0, 2:4, 2:4, 2:4] = 1.0
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 3, 3, 3),
        dtype="float32",
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [2.0, 1.0, 1.0]

    summary = run_shear_transform_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "shear_yz": 0.5,
            "auto_rotate_from_shear": True,
            "interpolation": "linear",
            "output_dtype": "float32",
            "roi_padding_zyx": [2, 2, 2],
        },
        client=None,
    )

    output = np.asarray(
        zarr.open_group(str(store_path), mode="r")["results/shear_transform/latest/data"]
    )
    assert output.shape == summary.output_shape_tpczyx
    assert np.max(output) > 0.0
    assert summary.output_shape_tpczyx[3:] != data.shape[3:]
    assert summary.applied_rotation_deg_xyz[0] != 0.0


def test_run_shear_transform_identity_with_distributed_client(
    tmp_path: Path,
) -> None:
    from dask.distributed import Client, LocalCluster

    store_path = tmp_path / "shear_identity_distributed.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 4 * 4 * 4, dtype=np.uint16).reshape(
        (1, 1, 1, 4, 4, 4)
    )
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [1.0, 1.0, 1.0]

    with LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    ) as cluster:
        with Client(cluster) as client:
            summary = run_shear_transform_analysis(
                zarr_path=store_path,
                parameters={
                    "input_source": "data",
                    "interpolation": "nearestneighbor",
                    "output_dtype": "uint16",
                    "shear_xy": 0.0,
                    "shear_xz": 0.0,
                    "shear_yz": 0.0,
                    "rotation_deg_x": 0.0,
                    "rotation_deg_y": 0.0,
                    "rotation_deg_z": 0.0,
                    "auto_rotate_from_shear": False,
                    "roi_padding_zyx": [1, 1, 1],
                },
                client=client,
            )

    output = np.asarray(
        zarr.open_group(str(store_path), mode="r")["results/shear_transform/latest/data"]
    )
    assert output.shape == data.shape
    assert summary.output_shape_tpczyx == data.shape
    np.testing.assert_array_equal(output, data)
