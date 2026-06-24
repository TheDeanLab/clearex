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
from clearex.io.stage_metadata_repair import repair_multiposition_stage_metadata
from clearex.workflow import SPATIAL_CALIBRATION_SCHEMA


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


def test_repair_multiposition_stage_metadata_from_relocated_experiment(
    tmp_path: Path,
) -> None:
    """Repair should patch only store metadata from a relocated sidecar."""
    store_path = tmp_path / "relocated_store.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_array(
        "clearex/runtime_cache/source/data",
        shape=(1, 2, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype=np.uint16,
        overwrite=True,
    )
    update_store_metadata(
        root,
        source_experiment="/missing/original/experiment.yml",
        navigate_experiment={"experiment_path": "/missing/original/experiment.yml"},
        spatial_calibration={
            "schema": SPATIAL_CALIBRATION_SCHEMA,
            "stage_axis_map_zyx": {"z": "+z", "y": "+x", "x": "-y"},
            "theta_mode": "rotate_zy_about_x",
        },
        stage_rows=None,
        position_translations_zyx_um=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    experiment_path = tmp_path / "experiment.yml"
    experiment_path.write_text(
        """
Saving:
  save_directory: .
  file_type: N5
MicroscopeState:
  is_multiposition: true
  timepoints: 1
  number_z_steps: 2
  step_size: 5.0
  channels:
    channel_1:
      is_selected: true
CameraParameters:
  img_x_pixels: 2
  img_y_pixels: 2
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "multi_positions.yml").write_text(
        """
- [X, Y, Z, THETA, F]
- [3901.21, -9594.1, -19588.2, 0.0, -9411.6]
- [3901.21, -4717.9, -19588.2, 0.0, -9411.6]
""".strip(),
        encoding="utf-8",
    )

    summary = repair_multiposition_stage_metadata(store_path, experiment_path)
    payload = load_store_metadata(store_path)

    assert summary.position_count == 2
    assert summary.stage_row_count == 2
    assert payload["source_experiment"] == str(experiment_path.resolve())
    assert payload["navigate_experiment"]["experiment_path"] == str(
        experiment_path.resolve()
    )
    assert payload["stage_rows"] == [
        {
            "x": 3901.21,
            "y": -9594.1,
            "z": -19588.2,
            "theta": 0.0,
            "f": -9411.6,
        },
        {
            "x": 3901.21,
            "y": -4717.9,
            "z": -19588.2,
            "theta": 0.0,
            "f": -9411.6,
        },
    ]
    assert payload["position_translations_zyx_um"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -4876.200000000001],
    ]


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
