#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path

# Third Party Imports
import zarr

# Local Imports
from clearex.deconvolution.pipeline import run_deconvolution_analysis
import clearex.deconvolution.pipeline as decon_pipeline


def _create_store(
    *,
    store_path: Path,
    shape_tpczyx: tuple[int, int, int, int, int, int],
) -> None:
    """Create a minimal canonical analysis store for deconvolution tests.

    Parameters
    ----------
    store_path : pathlib.Path
        Zarr store path.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Canonical source shape.

    Returns
    -------
    None
        Store is created in-place.
    """
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=shape_tpczyx,
        chunks=(1, 1, 1, shape_tpczyx[3], shape_tpczyx[4], shape_tpczyx[5]),
        dtype="uint16",
        overwrite=True,
    )
    root.attrs["voxel_size_um_zyx"] = [1.5, 0.8, 0.8]


def test_run_deconvolution_analysis_schedules_all_tpc_volumes(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    _create_store(store_path=store_path, shape_tpczyx=(2, 3, 2, 4, 4, 4))

    captured: dict[str, int | str] = {}

    def _fake_compute(*tasks, scheduler: str):
        captured["task_count"] = int(len(tasks))
        captured["scheduler"] = str(scheduler)
        return tuple(1 for _ in tasks)

    monkeypatch.setattr(decon_pipeline.dask, "compute", _fake_compute)

    summary = run_deconvolution_analysis(
        zarr_path=store_path,
        parameters={"psf_mode": "measured"},
        client=None,
    )

    assert captured["scheduler"] == "processes"
    assert captured["task_count"] == 12
    assert summary.volumes_processed == 12
    assert summary.channel_count == 2
    assert summary.component == "results/deconvolution/latest"
    assert summary.data_component == "results/deconvolution/latest/data"

    output_root = zarr.open_group(str(store_path), mode="r")
    output = output_root["results"]["deconvolution"]["latest"]["data"]
    assert tuple(output.shape) == (2, 3, 2, 4, 4, 4)


def test_run_deconvolution_analysis_respects_channel_selection(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    _create_store(store_path=store_path, shape_tpczyx=(2, 3, 2, 3, 3, 3))

    captured: dict[str, int] = {}

    def _fake_compute(*tasks, scheduler: str):
        del scheduler
        captured["task_count"] = int(len(tasks))
        return tuple(1 for _ in tasks)

    monkeypatch.setattr(decon_pipeline.dask, "compute", _fake_compute)

    summary = run_deconvolution_analysis(
        zarr_path=store_path,
        parameters={
            "psf_mode": "measured",
            "channel_indices": [1],
        },
        client=None,
    )

    assert captured["task_count"] == 6
    assert summary.volumes_processed == 6
    assert summary.channel_count == 1
    assert summary.output_chunks_tpczyx == (1, 1, 1, 3, 3, 3)
    assert summary.psf_mode == "measured"
