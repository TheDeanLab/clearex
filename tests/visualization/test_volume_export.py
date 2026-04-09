from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

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
