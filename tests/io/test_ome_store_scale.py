#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from clearex.io.ome_store import (
    load_store_metadata,
    publish_analysis_collection_from_cache,
    resolve_navigate_oblique_geometry_with_source,
    resolve_voxel_size_um_zyx_with_source,
    update_store_metadata,
)


def test_resolve_voxel_size_uses_source_component_chain(tmp_path: Path) -> None:
    """Resolver should traverse ``source_component`` ancestry before metadata."""
    store_path = tmp_path / "scale_chain.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    source = root.create_array(
        "clearex/runtime_cache/source/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype=np.uint16,
        overwrite=True,
    )
    source.attrs["voxel_size_um_zyx"] = [5.0, 1.5, 1.25]
    flatfield = root.create_array(
        "clearex/runtime_cache/results/flatfield/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype=np.float32,
        overwrite=True,
    )
    flatfield.attrs["source_component"] = "clearex/runtime_cache/source/data"
    shear = root.create_array(
        "clearex/runtime_cache/results/shear_transform/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype=np.float32,
        overwrite=True,
    )
    shear.attrs["source_component"] = (
        "clearex/runtime_cache/results/flatfield/latest/data"
    )

    update_store_metadata(root, voxel_size_um_zyx=[9.0, 9.0, 9.0])

    voxel_size_um_zyx, source = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component="clearex/runtime_cache/results/shear_transform/latest/data",
    )

    assert voxel_size_um_zyx == (5.0, 1.5, 1.25)
    assert source == "component:clearex/runtime_cache/source/data"


def test_resolve_voxel_size_uses_store_metadata_navigate_fallback(
    tmp_path: Path,
) -> None:
    """Resolver should use namespaced Navigate metadata before default fallback."""
    store_path = tmp_path / "scale_metadata_fallback.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    update_store_metadata(
        root,
        navigate_experiment={
            "xy_pixel_size_um": 0.8,
            "z_step_um": 3.5,
        },
    )

    voxel_size_um_zyx, source = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component="missing/component",
    )

    assert voxel_size_um_zyx == (3.5, 0.8, 0.8)
    assert source == "store_metadata_navigate"


def test_resolve_navigate_oblique_geometry_uses_store_metadata_fallback(
    tmp_path: Path,
) -> None:
    """Resolver should recover Navigate stage-geometry metadata from the store."""
    store_path = tmp_path / "navigate_geometry_metadata_fallback.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    payload = {
        "schema": "clearex.navigate_oblique_geometry.v1",
        "mode": "stage_scan",
        "scan_axis": "z",
        "stage_axis": "y",
        "scan_step_um": 5.0,
        "shear_dimension": "yz",
        "microscope_name": "Macroscale",
    }
    update_store_metadata(root, navigate_oblique_geometry=payload)

    geometry, source = resolve_navigate_oblique_geometry_with_source(
        root,
        source_component="missing/component",
    )

    assert geometry == payload
    assert source == "store_metadata"


def test_load_store_metadata_read_only_missing_group_returns_schema_default(
    tmp_path: Path,
) -> None:
    """Metadata loading must remain read-only safe when group is absent."""
    store_path = tmp_path / "metadata_read_only_default.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_array(
        "data",
        shape=(1, 1, 1, 1, 1, 1),
        chunks=(1, 1, 1, 1, 1, 1),
        dtype=np.uint16,
        overwrite=True,
    )

    read_only_root = zarr.open_group(str(store_path), mode="r")
    payload = load_store_metadata(read_only_root)
    assert payload == {"schema": "clearex.ome_store.v1"}
    assert "clearex" not in read_only_root


def test_publish_analysis_collection_uses_resolved_voxel_size(tmp_path: Path) -> None:
    """Public OME scale should be derived from resolved runtime-cache voxel size."""
    store_path = tmp_path / "publish_scale.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    source = root.create_array(
        "clearex/runtime_cache/source/data",
        shape=(1, 1, 1, 2, 3, 4),
        chunks=(1, 1, 1, 2, 3, 4),
        dtype=np.uint16,
        overwrite=True,
    )
    source.attrs["voxel_size_um_zyx"] = [4.0, 1.2, 1.2]
    shear = root.create_array(
        "clearex/runtime_cache/results/shear_transform/latest/data",
        data=np.arange(1 * 1 * 1 * 2 * 3 * 4, dtype=np.float32).reshape(
            (1, 1, 1, 2, 3, 4)
        ),
        chunks=(1, 1, 1, 2, 3, 4),
        overwrite=True,
    )
    shear.attrs["source_component"] = "clearex/runtime_cache/source/data"

    publish_analysis_collection_from_cache(
        store_path,
        analysis_name="shear_transform",
    )

    image_group = zarr.open_group(str(store_path), mode="r")[
        "results/shear_transform/latest/A/1/0"
    ]
    ome = image_group.attrs["ome"]
    transforms = ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert transforms[0]["type"] == "scale"
    assert transforms[0]["scale"] == [1.0, 1.0, 4.0, 1.2, 1.2]
    assert image_group.attrs["voxel_size_resolution_source"] == (
        "component:clearex/runtime_cache/source/data"
    )
