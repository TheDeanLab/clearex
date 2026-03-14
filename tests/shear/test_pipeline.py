#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from clearex.shear.pipeline import run_shear_transform_analysis


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
