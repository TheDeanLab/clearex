#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from clearex.flatfield.pipeline import run_flatfield_analysis
from clearex.io.experiment import create_dask_client
import clearex.flatfield.pipeline as flatfield_pipeline


def test_run_flatfield_analysis_writes_latest_results(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 1 * 2 * 3 * 4 * 4, dtype=np.uint16).reshape(
        (2, 1, 2, 3, 4, 4)
    )
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 3, 2, 2),
        overwrite=True,
    )

    class _FakeBaSiC:
        def __init__(
            self,
            *,
            fitting_mode,
            get_darkfield,
            smoothness_flatfield,
            working_size,
            device,
        ) -> None:
            del fitting_mode, get_darkfield, smoothness_flatfield, working_size, device
            self.flatfield = np.empty((0, 0), dtype=np.float32)
            self.darkfield = np.empty((0, 0), dtype=np.float32)
            self.baseline = np.empty((0,), dtype=np.float32)

        def fit(self, images, skip_shape_warning=False) -> None:
            del skip_shape_warning
            self.flatfield = np.full(images.shape[1:], 2.0, dtype=np.float32)
            self.darkfield = np.full(images.shape[1:], 0.5, dtype=np.float32)
            self.baseline = np.arange(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)

    client = create_dask_client(n_workers=1, threads_per_worker=2, processes=False)
    try:
        summary = run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "get_darkfield": True,
                "smoothness_flatfield": 1.0,
                "working_size": 128,
                "use_map_overlap": True,
                "overlap_zyx": [0, 1, 1],
                "is_timelapse": False,
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.component == "results/flatfield/latest"
    assert summary.data_component == "results/flatfield/latest/data"
    assert summary.flatfield_component == "results/flatfield/latest/flatfield_pcyx"
    assert summary.darkfield_component == "results/flatfield/latest/darkfield_pcyx"
    assert summary.baseline_component == "results/flatfield/latest/baseline_pctz"
    assert summary.source_component == "data"
    assert summary.profile_count == 2
    assert summary.transformed_volumes == 4
    assert summary.output_chunks_tpczyx == (1, 1, 1, 3, 2, 2)
    assert summary.output_dtype == "float32"

    output_root = zarr.open_group(str(store_path), mode="r")
    latest = output_root["results"]["flatfield"]["latest"]
    corrected = np.asarray(latest["data"], dtype=np.float32)
    flatfield = np.asarray(latest["flatfield_pcyx"], dtype=np.float32)
    darkfield = np.asarray(latest["darkfield_pcyx"], dtype=np.float32)
    baseline = np.asarray(latest["baseline_pctz"], dtype=np.float32)

    assert corrected.shape == data.shape
    assert corrected.dtype == np.float32
    assert flatfield.shape == (1, 2, 4, 4)
    assert darkfield.shape == (1, 2, 4, 4)
    assert baseline.shape == (1, 2, 2, 3)
    assert np.allclose(flatfield, 2.0)
    assert np.allclose(darkfield, 0.5)
    assert np.array_equal(baseline[0, 0], np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32))
    assert np.isclose(corrected[0, 0, 0, 0, 0, 0], (data[0, 0, 0, 0, 0, 0] - 0.5) / 2.0)
    assert latest.attrs["source_component"] == "data"
    assert latest.attrs["data_component"] == "results/flatfield/latest/data"
    assert latest.attrs["transformed_volumes"] == 4
    assert latest.attrs["output_dtype"] == "float32"
    assert (
        output_root["provenance"]["latest_outputs"]["flatfield"].attrs["component"]
        == "results/flatfield/latest"
    )
