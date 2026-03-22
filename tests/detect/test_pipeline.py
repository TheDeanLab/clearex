#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path

# Third Party Imports
import numpy as np
import zarr

# Local Imports
from clearex.detect.pipeline import run_particle_detection_analysis
from clearex.io.experiment import create_dask_client
import clearex.detect.pipeline as particle_pipeline


def test_run_particle_detection_analysis_writes_latest_results(
    tmp_path: Path, monkeypatch
):
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    def _fake_detect_particles_for_region(
        *,
        zarr_path: str,
        source_component: str,
        read_region,
        core_region,
        parameters,
    ) -> np.ndarray:
        del zarr_path, source_component, read_region, parameters
        return np.asarray(
            [
                [
                    float(core_region[0][0]),
                    float(core_region[1][0]),
                    float(core_region[2][0]),
                    float(core_region[3][0]),
                    float(core_region[4][0]),
                    float(core_region[5][0]),
                    1.25,
                    99.0,
                ]
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        particle_pipeline,
        "_detect_particles_for_region",
        _fake_detect_particles_for_region,
    )

    client = create_dask_client(n_workers=1, threads_per_worker=2, processes=False)
    try:
        summary = run_particle_detection_analysis(
            zarr_path=store_path,
            parameters={
                "channel_index": 0,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.component == "results/particle_detection/latest"
    assert summary.channel_index == 0
    assert summary.chunks_processed == 8
    assert summary.detections == 8

    output_root = zarr.open_group(str(store_path), mode="r")
    detections = np.asarray(
        output_root["results"]["particle_detection"]["latest"]["detections"]
    )
    points = np.asarray(
        output_root["results"]["particle_detection"]["latest"]["points_tzyx"]
    )
    assert detections.shape == (8, 8)
    assert points.shape == (8, 4)
    assert list(
        output_root["results"]["particle_detection"]["latest"]["detections"].attrs[
            "columns"
        ]
    ) == ["t", "p", "c", "z", "y", "x", "sigma", "intensity"]
    assert (
        output_root["results"]["particle_detection"]["latest"].attrs[
            "napari_points_component"
        ]
        == "results/particle_detection/latest/points_tzyx"
    )


def test_run_particle_detection_analysis_rejects_invalid_channel(tmp_path: Path):
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    try:
        run_particle_detection_analysis(
            zarr_path=store_path,
            parameters={"channel_index": 2},
            client=None,
        )
    except ValueError as exc:
        assert "out of bounds" in str(exc)
        return
    raise AssertionError("Expected ValueError for out-of-bounds channel index")
