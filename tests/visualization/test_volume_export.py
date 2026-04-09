from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from clearex.io.ome_store import publish_analysis_collection_from_cache
from clearex.visualization.volume_export import run_volume_export_analysis


def test_run_volume_export_analysis_writes_current_selection_cache_and_publishable_ome_zarr(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 2, 3, 4, 5))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 3, 4, 5),
        overwrite=True,
    )
    root.attrs["voxel_size_um_zyx"] = [4.0, 2.0, 2.0]

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

    runtime_cache_component = "clearex/runtime_cache/results/volume_export/latest/data"
    runtime_cache = zarr.open_group(str(store_path), mode="r")
    assert runtime_cache_component in runtime_cache
    exported = runtime_cache[runtime_cache_component]
    assert exported.shape == (1, 1, 1, 3, 4, 5)
    assert np.array_equal(exported[0, 0, 0], data[1, 0, 1])

    assert summary.component == "clearex/results/volume_export/latest"

    publish_analysis_collection_from_cache(
        store_path,
        analysis_name="volume_export",
    )

    published = zarr.open_group(str(store_path), mode="r")[
        "results/volume_export/latest/A/1/0/0"
    ]
    assert published.shape == (1, 1, 3, 4, 5)
