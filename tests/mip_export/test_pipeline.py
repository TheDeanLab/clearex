#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import tifffile
import zarr

from clearex.mip_export import pipeline
from clearex.mip_export.pipeline import run_mip_export_analysis


def _ome_pixels_metadata(path: Path) -> tuple[dict[str, str], str]:
    """Return OME Pixels attributes and series axes for one TIFF file."""
    with tifffile.TiffFile(str(path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        root = ET.fromstring(ome_xml)
        pixels = root.find(".//ome:Pixels", ns)
        assert pixels is not None
        return dict(pixels.attrib), str(tif.series[0].axes)


class _GuardedSourceArray:
    """Array wrapper that fails when callers request whole-volume slices."""

    def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...]) -> None:
        self._data = np.asarray(data)
        self.shape = tuple(int(value) for value in self._data.shape)
        self.dtype = self._data.dtype
        self.chunks = tuple(int(value) for value in chunks)
        self.read_count = 0

    @staticmethod
    def _is_full_slice(selector: object, length: int) -> bool:
        """Return whether a selector spans a complete axis range."""
        if not isinstance(selector, slice):
            return False
        start = 0 if selector.start is None else int(selector.start)
        stop = int(length) if selector.stop is None else int(selector.stop)
        step = 1 if selector.step is None else int(selector.step)
        return start == 0 and stop == int(length) and step == 1

    def __getitem__(self, key: object) -> np.ndarray:
        """Return selected data and reject whole-volume requests."""
        if not isinstance(key, tuple) or len(key) != 6:
            raise AssertionError("Expected 6D canonical indexing for source reads.")

        self.read_count += 1
        per_position_full = (
            isinstance(key[0], (int, np.integer))
            and isinstance(key[1], (int, np.integer))
            and isinstance(key[2], (int, np.integer))
            and self._is_full_slice(key[3], self.shape[3])
            and self._is_full_slice(key[4], self.shape[4])
            and self._is_full_slice(key[5], self.shape[5])
        )
        multi_position_full = (
            isinstance(key[0], (int, np.integer))
            and self._is_full_slice(key[1], self.shape[1])
            and isinstance(key[2], (int, np.integer))
            and self._is_full_slice(key[3], self.shape[3])
            and self._is_full_slice(key[4], self.shape[4])
            and self._is_full_slice(key[5], self.shape[5])
        )
        if per_position_full or multi_position_full:
            raise RuntimeError("Detected forbidden full-volume source read.")
        return np.asarray(self._data[key])


def test_run_mip_export_analysis_writes_uint16_ome_tiff_outputs_with_calibration(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "mip_tiff_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 2 * 1 * 3 * 4 * 5, dtype=np.float32).reshape(
        (1, 2, 1, 3, 4, 5)
    )
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 3, 4, 5),
        dtype="float32",
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [5.0, 2.0, 3.0]

    summary = run_mip_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "position_mode": "per_position",
            "export_format": "tiff",
            "output_directory": str(tmp_path / "mip_tiff_outputs"),
        },
        client=None,
    )

    assert summary.component == "results/mip_export/latest"
    assert summary.export_format == "ome-tiff"
    assert summary.position_mode == "per_position"
    assert summary.exported_files == 6
    output_directory = Path(summary.output_directory)
    assert output_directory.is_dir()

    xy_path = output_directory / "mip_xy_p0001_t0000_c0000.tif"
    xz_path = output_directory / "mip_xz_p0001_t0000_c0000.tif"
    yz_path = output_directory / "mip_yz_p0001_t0000_c0000.tif"
    assert xy_path.exists()
    assert xz_path.exists()
    assert yz_path.exists()

    xy = np.asarray(tifffile.imread(str(xy_path)))
    xz = np.asarray(tifffile.imread(str(xz_path)))
    yz = np.asarray(tifffile.imread(str(yz_path)))
    expected_source = data[0, 1, 0, :, :, :]
    xy_expected = np.max(expected_source, axis=0).astype(np.uint16)
    xz_expected = np.max(expected_source, axis=1).astype(np.uint16)
    yz_expected = np.max(expected_source, axis=2).astype(np.uint16)
    np.testing.assert_array_equal(xy, xy_expected)
    assert int(xz.shape[1]) == int(xz_expected.shape[1])
    assert int(yz.shape[1]) == int(yz_expected.shape[1])
    assert int(xz.shape[0]) == pipeline._resampled_axis_length(
        axis_length=int(xz_expected.shape[0]),
        source_spacing_um=5.0,
        target_spacing_um=3.0,
    )
    assert int(yz.shape[0]) == pipeline._resampled_axis_length(
        axis_length=int(yz_expected.shape[0]),
        source_spacing_um=5.0,
        target_spacing_um=2.0,
    )
    np.testing.assert_array_equal(xz[0, :], xz_expected[0, :])
    np.testing.assert_array_equal(xz[-1, :], xz_expected[-1, :])
    np.testing.assert_array_equal(yz[0, :], yz_expected[0, :])
    np.testing.assert_array_equal(yz[-1, :], yz_expected[-1, :])
    assert xy.dtype == np.uint16

    xy_pixels, xy_axes = _ome_pixels_metadata(xy_path)
    xz_pixels, xz_axes = _ome_pixels_metadata(xz_path)
    yz_pixels, yz_axes = _ome_pixels_metadata(yz_path)
    assert xy_axes == "YX"
    assert xz_axes == "YX"
    assert yz_axes == "YX"
    assert xy_pixels["PhysicalSizeX"] == "3.0"
    assert xy_pixels["PhysicalSizeY"] == "2.0"
    assert xz_pixels["PhysicalSizeX"] == "3.0"
    assert xz_pixels["PhysicalSizeY"] == "3.0"
    assert yz_pixels["PhysicalSizeX"] == "2.0"
    assert yz_pixels["PhysicalSizeY"] == "2.0"

    latest_attrs = dict(
        zarr.open_group(str(store_path), mode="r")["results"]["mip_export"]["latest"].attrs
    )
    assert latest_attrs["exported_files"] == 6
    assert latest_attrs["export_format"] == "ome-tiff"
    assert latest_attrs["voxel_size_um_zyx"] == [5.0, 2.0, 3.0]
    latest_ref_attrs = dict(
        zarr.open_group(str(store_path), mode="r")["provenance"]["latest_outputs"][
            "mip_export"
        ].attrs
    )
    assert latest_ref_attrs["component"] == "results/mip_export/latest"


def test_run_mip_export_analysis_writes_multi_position_zarr_outputs(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "mip_zarr_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 3 * 2 * 4 * 3 * 5, dtype=np.uint16).reshape(
        (2, 3, 2, 4, 3, 5)
    )
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 3, 5),
        dtype="uint16",
        overwrite=True,
    )

    summary = run_mip_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "position_mode": "multi_position",
            "export_format": "zarr",
            "output_directory": str(tmp_path / "mip_zarr_outputs"),
        },
        client=None,
    )

    assert summary.export_format == "zarr"
    assert summary.position_mode == "multi_position"
    assert summary.task_count == 12
    assert summary.exported_files == 12
    output_directory = Path(summary.output_directory)
    assert output_directory.is_dir()

    xy_path = output_directory / "mip_xy_t0001_c0001.zarr"
    yz_path = output_directory / "mip_yz_t0000_c0000.zarr"
    assert xy_path.exists()
    assert yz_path.exists()

    xy_root = zarr.open_group(str(xy_path), mode="r")
    xy = np.asarray(xy_root["data"])
    assert tuple(xy.shape) == (3, 3, 5)
    assert list(xy_root["data"].attrs["axes"]) == ["p", "y", "x"]

    yz_root = zarr.open_group(str(yz_path), mode="r")
    yz = np.asarray(yz_root["data"])
    assert tuple(yz.shape) == (3, 4, 3)
    assert list(yz_root["data"].attrs["axes"]) == ["p", "z", "y"]


@pytest.mark.parametrize(("projection", "expected_axis"), [("xy", 0), ("xz", 1), ("yz", 2)])
def test_run_export_task_reads_single_position_source_in_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
    expected_axis: int,
) -> None:
    data = np.arange(1 * 1 * 1 * 4 * 6 * 8, dtype=np.float32).reshape((1, 1, 1, 4, 6, 8))
    guarded = _GuardedSourceArray(data, chunks=(1, 1, 1, 2, 3, 4))
    fake_group = {"data": guarded}
    monkeypatch.setattr(pipeline.zarr, "open_group", lambda *_args, **_kwargs: fake_group)

    task = pipeline._MipExportTask(
        projection=projection,
        t_index=0,
        c_index=0,
        p_index=0,
    )
    result = pipeline._run_export_task(
        zarr_path="ignored.zarr",
        source_component="data",
        output_directory=str(tmp_path),
        export_format="tiff",
        position_mode="per_position",
        task=task,
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        resample_z_to_lateral=False,
    )

    expected = np.max(data[0, 0, 0, :, :, :], axis=expected_axis).astype(np.uint16)
    observed = np.asarray(tifffile.imread(str(result["path"])))
    np.testing.assert_array_equal(observed, expected)
    assert guarded.read_count > 1


def test_run_export_task_reads_multi_position_source_in_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.arange(1 * 3 * 1 * 4 * 6 * 8, dtype=np.float32).reshape((1, 3, 1, 4, 6, 8))
    guarded = _GuardedSourceArray(data, chunks=(1, 3, 1, 2, 3, 4))
    fake_group = {"data": guarded}
    monkeypatch.setattr(pipeline.zarr, "open_group", lambda *_args, **_kwargs: fake_group)

    task = pipeline._MipExportTask(
        projection="xy",
        t_index=0,
        c_index=0,
        p_index=None,
    )
    result = pipeline._run_export_task(
        zarr_path="ignored.zarr",
        source_component="data",
        output_directory=str(tmp_path),
        export_format="tiff",
        position_mode="multi_position",
        task=task,
        voxel_size_um_zyx=(1.0, 1.0, 1.0),
        resample_z_to_lateral=False,
    )

    expected = np.max(data[0, :, 0, :, :, :], axis=1).astype(np.uint16)
    observed = np.asarray(tifffile.imread(str(result["path"])))
    np.testing.assert_array_equal(observed, expected)
    assert guarded.read_count > 1


def test_run_mip_export_analysis_distributed_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    distributed = pytest.importorskip("dask.distributed")

    store_path = tmp_path / "mip_distributed_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 2 * 1 * 4 * 6 * 8, dtype=np.uint16).reshape(
        (1, 2, 1, 4, 6, 8)
    )
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 3, 4),
        dtype="uint16",
        overwrite=True,
    )

    with distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    ) as cluster:
        with distributed.Client(cluster) as client:
            summary = run_mip_export_analysis(
                zarr_path=store_path,
                parameters={
                    "input_source": "data",
                    "position_mode": "per_position",
                    "export_format": "zarr",
                    "output_directory": str(tmp_path / "mip_distributed_outputs"),
                },
                client=client,
            )

    assert summary.export_format == "zarr"
    assert summary.position_mode == "per_position"
    assert summary.task_count == 6
    assert summary.exported_files == 6

    output_directory = Path(summary.output_directory)
    xy_path = output_directory / "mip_xy_p0001_t0000_c0000.zarr"
    xz_path = output_directory / "mip_xz_p0001_t0000_c0000.zarr"
    yz_path = output_directory / "mip_yz_p0001_t0000_c0000.zarr"
    assert xy_path.exists()
    assert xz_path.exists()
    assert yz_path.exists()

    expected_source = data[0, 1, 0, :, :, :]
    xy = np.asarray(zarr.open_group(str(xy_path), mode="r")["data"])
    xz = np.asarray(zarr.open_group(str(xz_path), mode="r")["data"])
    yz = np.asarray(zarr.open_group(str(yz_path), mode="r")["data"])
    np.testing.assert_array_equal(xy, np.max(expected_source, axis=0))
    np.testing.assert_array_equal(xz, np.max(expected_source, axis=1))
    np.testing.assert_array_equal(yz, np.max(expected_source, axis=2))
