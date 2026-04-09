from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import zarr
import tifffile

from clearex.io.ome_store import publish_analysis_collection_from_cache
from clearex.io.ome_store import source_cache_component
from clearex.visualization import (
    VolumeExportSummary,
    run_volume_export_analysis as lazy_run_volume_export_analysis,
)
from clearex.visualization.volume_export import run_volume_export_analysis


def _write_source_cache_volume(root: zarr.Group, data: np.ndarray[Any, Any]) -> str:
    source_component = source_cache_component()
    source_parent = root.require_group(source_component.rsplit("/", 1)[0])
    source_parent.create_dataset(
        name=source_component.rsplit("/", 1)[1],
        data=data,
        chunks=(1, 1, 1, 3, 4, 5),
        overwrite=True,
    )
    source_node = root[source_component]
    assert isinstance(source_node, zarr.Array)
    source_node.attrs["voxel_size_um_zyx"] = [4.0, 2.0, 2.0]
    return source_component


def _write_source_adjacent_level(
    root: zarr.Group,
    *,
    source_component: str,
    level_component: str,
    data: np.ndarray[Any, Any],
) -> None:
    parent = root.require_group(level_component.rsplit("/", 1)[0])
    parent.create_dataset(
        name=level_component.rsplit("/", 1)[1],
        data=data,
        chunks=data.shape,
        overwrite=True,
    )
    source_node = root[source_component]
    assert isinstance(source_node, zarr.Array)
    source_node.attrs["display_pyramid_levels"] = [source_component, level_component]
    source_node.attrs["display_pyramid_factors_tpczyx"] = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2],
    ]
    root.attrs["display_pyramid_levels_by_component"] = {
        source_component: [source_component, level_component]
    }


def _ome_tiff_metadata(
    path: Path,
) -> tuple[list[dict[str, str]], list[str], list[tuple[int, ...]], bool]:
    """Return OME Pixels metadata and series layout for one TIFF file."""
    with tifffile.TiffFile(str(path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        root = ET.fromstring(ome_xml)
        pixels = [
            dict(image.find("ome:Pixels", ns).attrib)
            for image in root.findall("ome:Image", ns)
        ]
        series_axes = [str(series.axes) for series in tif.series]
        series_shapes = [
            tuple(int(value) for value in series.shape) for series in tif.series
        ]
        return pixels, series_axes, series_shapes, bool(tif.is_bigtiff)


def test_run_volume_export_analysis_writes_current_selection_cache_and_publishable_ome_zarr(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)

    assert lazy_run_volume_export_analysis is run_volume_export_analysis

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 1,
            "p_index": 0,
            "c_index": 1,
            "resolution_level": 0,
            "export_format": "ome-zarr",
        },
    )
    assert isinstance(summary, VolumeExportSummary)
    assert summary.source_component == source_component
    assert (
        summary.data_component
        == "clearex/runtime_cache/results/volume_export/latest/data"
    )

    runtime_cache_component = "clearex/runtime_cache/results/volume_export/latest/data"
    runtime_cache = zarr.open_group(str(store_path), mode="r")
    assert runtime_cache_component in runtime_cache
    exported = runtime_cache[runtime_cache_component]
    assert isinstance(exported, zarr.Array)
    assert exported.shape == (1, 1, 1, 3, 4, 5)
    assert np.array_equal(exported[0, 0, 0], data[1, 0, 1])

    assert summary.component == "clearex/results/volume_export/latest"

    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["component"] == "clearex/results/volume_export/latest"
    assert auxiliary.attrs["data_component"] == runtime_cache_component
    assert auxiliary.attrs["source_component"] == source_component

    provenance = runtime_cache["clearex/provenance/latest_outputs/volume_export"]
    assert provenance.attrs["analysis_name"] == "volume_export"
    assert provenance.attrs["component"] == "clearex/results/volume_export/latest"
    assert provenance.attrs["storage_policy"] == "latest_only"

    publish_analysis_collection_from_cache(
        store_path,
        analysis_name="volume_export",
    )

    published = zarr.open_group(str(store_path), mode="r")[
        "results/volume_export/latest/A/1/0/0"
    ]
    assert isinstance(published, zarr.Array)
    assert published.shape == (1, 1, 3, 4, 5)


def test_run_volume_export_analysis_writes_current_selection_ome_tiff_with_calibration(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 1,
            "p_index": 0,
            "c_index": 1,
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "single_file",
        },
    )

    assert isinstance(summary, VolumeExportSummary)
    assert summary.source_component == source_component
    assert summary.export_format == "ome-tiff"
    assert summary.tiff_file_layout == "single_file"
    assert len(summary.artifact_paths) == 1
    assert summary.artifact_paths[0].startswith(
        "clearex/results/volume_export/latest/files/"
    )

    artifact_path = store_path / summary.artifact_paths[0]
    assert artifact_path.exists()

    pixels, series_axes, series_shapes, is_bigtiff = _ome_tiff_metadata(artifact_path)
    assert is_bigtiff is True
    assert series_axes == ["ZYX"]
    assert series_shapes == [(3, 4, 5)]
    assert len(pixels) == 1
    assert pixels[0]["PhysicalSizeZ"] == "4.0"
    assert pixels[0]["PhysicalSizeY"] == "2.0"
    assert pixels[0]["PhysicalSizeX"] == "2.0"

    volume = np.asarray(tifffile.imread(str(artifact_path)))
    np.testing.assert_array_equal(volume, data[1, 0, 1])

    runtime_cache = zarr.open_group(str(store_path), mode="r")
    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["export_format"] == "ome-tiff"
    assert auxiliary.attrs["tiff_file_layout"] == "single_file"
    assert auxiliary.attrs["artifact_paths"] == list(summary.artifact_paths)
    assert auxiliary.attrs["voxel_size_um_zyx"] == [4.0, 2.0, 2.0]


def test_run_volume_export_analysis_writes_downsampled_ome_tiff_with_scaled_calibration(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)
    level_component = source_cache_component(level_index=1)
    _write_source_adjacent_level(
        root,
        source_component=source_component,
        level_component=level_component,
        data=data[:, :, :, ::2, ::2, ::2],
    )

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 1,
            "p_index": 0,
            "c_index": 1,
            "resolution_level": 1,
            "export_format": "ome-tiff",
            "tiff_file_layout": "single_file",
        },
    )

    [artifact_relative_path] = summary.artifact_paths
    pixels, series_axes, series_shapes, is_bigtiff = _ome_tiff_metadata(
        store_path / artifact_relative_path
    )
    assert is_bigtiff is True
    assert series_axes == ["ZYX"]
    assert series_shapes == [(2, 2, 3)]
    assert len(pixels) == 1
    assert pixels[0]["PhysicalSizeZ"] == "8.0"
    assert pixels[0]["PhysicalSizeY"] == "4.0"
    assert pixels[0]["PhysicalSizeX"] == "4.0"

    runtime_cache = zarr.open_group(str(store_path), mode="r")
    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["voxel_size_um_zyx"] == [8.0, 4.0, 4.0]


def test_run_volume_export_analysis_writes_all_indices_single_file_ome_tiff_series_per_position(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "single_file",
        },
    )

    assert isinstance(summary, VolumeExportSummary)
    assert summary.source_component == source_component
    assert summary.export_format == "ome-tiff"
    assert summary.tiff_file_layout == "single_file"
    assert len(summary.artifact_paths) == 1

    artifact_path = store_path / summary.artifact_paths[0]
    assert artifact_path.exists()

    pixels, series_axes, series_shapes, is_bigtiff = _ome_tiff_metadata(artifact_path)
    assert is_bigtiff is True
    assert series_axes == ["TCZYX", "TCZYX"]
    assert series_shapes == [(2, 2, 3, 4, 5), (2, 2, 3, 4, 5)]
    assert len(pixels) == 2
    for pixel in pixels:
        assert pixel["PhysicalSizeZ"] == "4.0"
        assert pixel["PhysicalSizeY"] == "2.0"
        assert pixel["PhysicalSizeX"] == "2.0"

    runtime_cache = zarr.open_group(str(store_path), mode="r")
    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["export_format"] == "ome-tiff"
    assert auxiliary.attrs["tiff_file_layout"] == "single_file"
    assert auxiliary.attrs["artifact_paths"] == list(summary.artifact_paths)
    assert auxiliary.attrs["voxel_size_um_zyx"] == [4.0, 2.0, 2.0]


def test_run_volume_export_analysis_writes_all_indices_per_volume_files_ome_tiff(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "per_volume_files",
        },
    )

    assert isinstance(summary, VolumeExportSummary)
    assert summary.source_component == source_component
    assert summary.export_format == "ome-tiff"
    assert summary.tiff_file_layout == "per_volume_files"
    assert len(summary.artifact_paths) == 8
    assert all(
        path.startswith("clearex/results/volume_export/latest/files/")
        for path in summary.artifact_paths
    )

    runtime_cache = zarr.open_group(str(store_path), mode="r")
    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["export_format"] == "ome-tiff"
    assert auxiliary.attrs["tiff_file_layout"] == "per_volume_files"
    assert auxiliary.attrs["artifact_paths"] == list(summary.artifact_paths)
    assert auxiliary.attrs["voxel_size_um_zyx"] == [4.0, 2.0, 2.0]

    for artifact_path in summary.artifact_paths:
        file_path = store_path / artifact_path
        assert file_path.exists()
        pixels, series_axes, series_shapes, is_bigtiff = _ome_tiff_metadata(file_path)
        assert is_bigtiff is True
        assert series_axes == ["ZYX"]
        assert series_shapes == [(3, 4, 5)]
        assert len(pixels) == 1
        assert pixels[0]["PhysicalSizeZ"] == "4.0"
        assert pixels[0]["PhysicalSizeY"] == "2.0"
        assert pixels[0]["PhysicalSizeX"] == "2.0"


def test_run_volume_export_analysis_ome_zarr_clears_prior_tiff_artifacts(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    _ = _write_source_cache_volume(root, data)

    tiff_summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 0,
            "p_index": 0,
            "c_index": 0,
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "single_file",
        },
    )
    [artifact_path] = tiff_summary.artifact_paths
    stale_artifact = store_path / artifact_path
    assert stale_artifact.exists()

    zarr_summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 0,
            "p_index": 0,
            "c_index": 0,
            "resolution_level": 0,
            "export_format": "ome-zarr",
        },
    )

    assert zarr_summary.export_format == "ome-zarr"
    assert not stale_artifact.exists()
    assert not (store_path / "clearex/results/volume_export/latest/files").exists()


def test_run_volume_export_analysis_writes_all_indices_cache_and_publishable_ome_zarr(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "t_index": 1,
            "p_index": 0,
            "c_index": 1,
            "resolution_level": 0,
            "export_format": "ome-zarr",
        },
    )
    assert isinstance(summary, VolumeExportSummary)
    assert summary.source_component == source_component
    assert summary.resolved_resolution_component == source_component
    assert summary.generated_resolution_level is False
    assert (
        summary.data_component
        == "clearex/runtime_cache/results/volume_export/latest/data"
    )

    runtime_cache_component = "clearex/runtime_cache/results/volume_export/latest/data"
    runtime_cache = zarr.open_group(str(store_path), mode="r")
    assert runtime_cache_component in runtime_cache
    exported = runtime_cache[runtime_cache_component]
    assert isinstance(exported, zarr.Array)
    assert exported.shape == data.shape
    assert np.array_equal(exported[...], data)

    assert summary.component == "clearex/results/volume_export/latest"

    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["component"] == "clearex/results/volume_export/latest"
    assert auxiliary.attrs["data_component"] == runtime_cache_component
    assert auxiliary.attrs["source_component"] == source_component


def test_run_volume_export_analysis_writes_all_indices_and_generates_missing_resolution_level(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)
    assert source_cache_component(level_index=1) not in root

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 1,
            "export_format": "ome-zarr",
        },
    )

    assert isinstance(summary, VolumeExportSummary)
    assert summary.source_component == source_component
    assert summary.export_scope == "all_indices"
    assert summary.resolution_level == 1
    assert summary.generated_resolution_level is True
    assert summary.resolved_resolution_component == source_cache_component(
        level_index=1
    )
    assert (
        summary.data_component
        == "clearex/runtime_cache/results/volume_export/latest/data"
    )

    runtime_cache = zarr.open_group(str(store_path), mode="r")
    runtime_cache_component = "clearex/runtime_cache/results/volume_export/latest/data"
    exported = runtime_cache[runtime_cache_component]
    assert isinstance(exported, zarr.Array)
    generated_level = runtime_cache[source_cache_component(level_index=1)]
    assert isinstance(generated_level, zarr.Array)
    assert exported.shape == generated_level.shape
    assert np.array_equal(exported[...], generated_level[...])
    assert runtime_cache_component in runtime_cache


def test_run_volume_export_analysis_extends_partial_existing_pyramid(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 4 * 4 * 4, dtype=np.uint16).reshape((2, 2, 2, 4, 4, 4))
    source_component = _write_source_cache_volume(root, data)
    level_1_component = source_cache_component(level_index=1)
    level_1_data = data[:, :, :, ::2, ::2, ::2]
    _write_source_adjacent_level(
        root,
        source_component=source_component,
        level_component=level_1_component,
        data=level_1_data,
    )

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 2,
            "export_format": "ome-zarr",
        },
    )

    assert summary.resolution_level == 2
    assert summary.generated_resolution_level is True
    assert summary.resolved_resolution_component == source_cache_component(
        level_index=2
    )
    runtime_cache = zarr.open_group(str(store_path), mode="r")
    generated_level = runtime_cache[source_cache_component(level_index=2)]
    assert isinstance(generated_level, zarr.Array)
    assert generated_level.shape == (2, 2, 2, 1, 1, 1)
    assert generated_level.attrs["source_component"] == source_cache_component(
        level_index=1
    )
    assert generated_level.attrs["downsample_factors_tpczyx"] == [1, 1, 1, 4, 4, 4]
    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["voxel_size_um_zyx"] == [16.0, 8.0, 8.0]
    assert (
        auxiliary.attrs["voxel_size_resolution_source"]
        == f"component:{source_component}"
    )


def test_run_volume_export_analysis_uses_discovered_noncanonical_level(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)
    discovered_level = "clearex/runtime_cache/source/custom_levels/level_one"
    _write_source_adjacent_level(
        root,
        source_component=source_component,
        level_component=discovered_level,
        data=data[:, :, :, ::2, ::2, ::2],
    )

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 1,
            "export_format": "ome-zarr",
        },
    )

    assert summary.resolved_resolution_component == discovered_level
    assert summary.generated_resolution_level is False
    runtime_cache = zarr.open_group(str(store_path), mode="r")
    auxiliary = runtime_cache["clearex/results/volume_export/latest"]
    assert auxiliary.attrs["voxel_size_um_zyx"] == [8.0, 4.0, 4.0]
    assert (
        auxiliary.attrs["voxel_size_resolution_source"]
        == f"component:{source_component}"
    )

    publish_analysis_collection_from_cache(
        store_path,
        analysis_name="volume_export",
    )

    image_group = zarr.open_group(str(store_path), mode="r")[
        "results/volume_export/latest/A/1/0"
    ]
    transforms = image_group.attrs["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    assert transforms[0]["type"] == "scale"
    assert transforms[0]["scale"] == [1.0, 1.0, 8.0, 4.0, 4.0]


def test_run_volume_export_analysis_all_indices_ignores_stale_selection_indices(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    source_component = _write_source_cache_volume(root, data)

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "t_index": 99,
            "p_index": 99,
            "c_index": 99,
            "resolution_level": 0,
            "export_format": "ome-zarr",
        },
    )

    assert summary.source_component == source_component
    runtime_cache = zarr.open_group(str(store_path), mode="r")
    exported = runtime_cache["clearex/runtime_cache/results/volume_export/latest/data"]
    assert isinstance(exported, zarr.Array)
    assert exported.shape == data.shape
