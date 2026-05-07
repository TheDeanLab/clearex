#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path

# Third Party Imports
import zarr

# Local Imports
from clearex.deconvolution.pipeline import run_deconvolution_analysis
import clearex.deconvolution.pipeline as decon_pipeline
from clearex.deconvolution.petakit import (
    MATLAB_RUNTIME_ROOT_ENV,
    PETAKIT5D_ROOT_ENV,
)

_DECONV_PUBLIC_COMPONENT = "results/deconvolution/latest"
_DECONV_CACHE_DATA = "clearex/runtime_cache/results/deconvolution/latest/data"


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
        parameters={"psf_mode": "measured", "mcc_mode": False},
        client=None,
    )

    assert captured["scheduler"] == "processes"
    assert captured["task_count"] == 12
    assert summary.volumes_processed == 12
    assert summary.channel_count == 2
    assert summary.component == _DECONV_PUBLIC_COMPONENT
    assert summary.data_component == _DECONV_CACHE_DATA

    output_root = zarr.open_group(str(store_path), mode="r")
    output = output_root[_DECONV_CACHE_DATA]
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
            "mcc_mode": False,
        },
        client=None,
    )

    assert captured["task_count"] == 6
    assert summary.volumes_processed == 6
    assert summary.channel_count == 1
    assert summary.output_chunks_tpczyx == (1, 1, 1, 3, 3, 3)
    assert summary.psf_mode == "measured"


def test_run_deconvolution_analysis_preflights_mcc_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    _create_store(store_path=store_path, shape_tpczyx=(1, 1, 1, 3, 3, 3))
    monkeypatch.delenv(PETAKIT5D_ROOT_ENV, raising=False)
    monkeypatch.delenv(MATLAB_RUNTIME_ROOT_ENV, raising=False)

    try:
        run_deconvolution_analysis(
            zarr_path=store_path,
            parameters={"psf_mode": "measured"},
            client=None,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("deconvolution should fail before submitting work.")

    assert PETAKIT5D_ROOT_ENV in message
    assert MATLAB_RUNTIME_ROOT_ENV in message

    root = zarr.open_group(str(store_path), mode="r")
    assert _DECONV_CACHE_DATA not in root
