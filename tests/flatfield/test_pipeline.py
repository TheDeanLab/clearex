#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest
import zarr

from clearex.flatfield.pipeline import run_flatfield_analysis
from clearex.io.experiment import create_dask_client
import clearex.flatfield.pipeline as flatfield_pipeline


def test_run_flatfield_analysis_writes_latest_results(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 1 * 2 * 3 * 4 * 4, dtype=np.uint16).reshape((2, 1, 2, 3, 4, 4))
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
    assert np.array_equal(
        baseline[0, 0], np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    )
    assert np.isclose(corrected[0, 0, 0, 0, 0, 0], (data[0, 0, 0, 0, 0, 0] - 0.5) / 2.0)
    assert latest.attrs["source_component"] == "data"
    assert latest.attrs["data_component"] == "results/flatfield/latest/data"
    assert latest.attrs["transformed_volumes"] == 4
    assert latest.attrs["output_dtype"] == "float32"
    assert (
        output_root["provenance"]["latest_outputs"]["flatfield"].attrs["component"]
        == "results/flatfield/latest"
    )


def test_run_flatfield_analysis_materializes_output_pyramid(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_pyramid.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 2 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 2, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 2, 2),
        overwrite=True,
    )
    root["data"].attrs.update(
        {
            "pyramid_levels": ["data", "data_pyramid/level_1"],
            "pyramid_factors_tpczyx": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 2, 2],
            ],
        }
    )
    root.attrs.update(
        {
            "data_pyramid_levels": ["data", "data_pyramid/level_1"],
            "data_pyramid_factors_tpczyx": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 2, 2],
            ],
        }
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
            self.flatfield = np.ones(images.shape[1:], dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "fit_mode": "full_volume",
                "smoothness_flatfield": 1.0,
                "working_size": 64,
                "use_map_overlap": False,
                "is_timelapse": False,
            },
            client=client,
        )
    finally:
        client.close()

    output_root = zarr.open_group(str(store_path), mode="r")
    latest = output_root["results"]["flatfield"]["latest"]
    corrected = np.asarray(latest["data"], dtype=np.float32)
    level_1 = np.asarray(latest["data_pyramid"]["level_1"], dtype=np.float32)
    expected = corrected[:, :, :, :, ::2, ::2]

    assert level_1.shape == expected.shape
    assert np.array_equal(level_1, expected)
    assert latest["data"].attrs["pyramid_levels"] == [
        "results/flatfield/latest/data",
        "results/flatfield/latest/data_pyramid/level_1",
    ]
    assert latest["data"].attrs["pyramid_factors_tpczyx"] == [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 2, 2],
    ]


def test_run_flatfield_analysis_defaults_darkfield_off(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        data=np.arange(1 * 1 * 1 * 2 * 2 * 2, dtype=np.uint16).reshape(
            (1, 1, 1, 2, 2, 2)
        ),
        chunks=(1, 1, 1, 2, 2, 2),
        overwrite=True,
    )

    seen_get_darkfield: list[bool] = []

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
            del fitting_mode, smoothness_flatfield, working_size, device
            seen_get_darkfield.append(bool(get_darkfield))
            self.flatfield = np.empty((0, 0), dtype=np.float32)
            self.darkfield = np.empty((0, 0), dtype=np.float32)
            self.baseline = np.empty((0,), dtype=np.float32)

        def fit(self, images, skip_shape_warning=False) -> None:
            del skip_shape_warning
            self.flatfield = np.ones(images.shape[1:], dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "smoothness_flatfield": 1.0,
                "working_size": 128,
                "use_map_overlap": False,
                "is_timelapse": False,
            },
            client=client,
        )
    finally:
        client.close()

    assert seen_get_darkfield == [False]


def test_run_flatfield_analysis_tiled_fit_without_blending(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_tiled_no_blend.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 1, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 1, 2, 2),
        overwrite=True,
    )

    seen_shapes: list[tuple[int, int, int]] = []

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
            seen_shapes.append(tuple(int(v) for v in images.shape))
            marker = float(images[0, 0, 0]) + 10.0
            self.flatfield = np.full(images.shape[1:], marker, dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "fit_mode": "tiled",
                "fit_tile_shape_yx": [2, 2],
                "blend_tiles": False,
                "smoothness_flatfield": 1.0,
                "working_size": 64,
                "use_map_overlap": True,
                "overlap_zyx": [0, 1, 1],
            },
            client=client,
        )
    finally:
        client.close()

    assert len(seen_shapes) == 4
    assert all(shape == (1, 3, 3) for shape in seen_shapes)

    output_root = zarr.open_group(str(store_path), mode="r")
    flatfield = np.asarray(
        output_root["results"]["flatfield"]["latest"]["flatfield_pcyx"],
        dtype=np.float32,
    )
    assert np.array_equal(
        flatfield[0, 0],
        np.asarray(
            [
                [10.0, 10.0, 11.0, 11.0],
                [10.0, 10.0, 11.0, 11.0],
                [14.0, 14.0, 15.0, 15.0],
                [14.0, 14.0, 15.0, 15.0],
            ],
            dtype=np.float32,
        ),
    )


def test_run_flatfield_analysis_tiled_fit_blends_tiles(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_tiled_blend.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 1, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 1, 2, 2),
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
            marker = float(images[0, 0, 0]) + 10.0
            self.flatfield = np.full(images.shape[1:], marker, dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "fit_mode": "tiled",
                "fit_tile_shape_yx": [2, 2],
                "blend_tiles": True,
                "smoothness_flatfield": 1.0,
                "working_size": 64,
                "use_map_overlap": True,
                "overlap_zyx": [0, 1, 1],
            },
            client=client,
        )
    finally:
        client.close()

    output_root = zarr.open_group(str(store_path), mode="r")
    flatfield = np.asarray(
        output_root["results"]["flatfield"]["latest"]["flatfield_pcyx"],
        dtype=np.float32,
    )[0, 0]
    assert flatfield.shape == (4, 4)
    assert np.unique(flatfield).size > 4
    assert flatfield[1, 1] > 10.0
    assert flatfield[1, 1] < 14.0


def test_run_flatfield_analysis_resumes_pending_transform_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_resume_transform.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape((2, 1, 1, 1, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 1, 2, 2),
        overwrite=True,
    )

    fit_calls = {"count": 0}

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
            fit_calls["count"] += 1
            self.flatfield = np.full(images.shape[1:], 2.0, dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)
    original_transform = flatfield_pipeline._transform_region
    failing_transform_calls = {"count": 0}

    def _failing_transform(**kwargs):
        failing_transform_calls["count"] += 1
        if failing_transform_calls["count"] == 3:
            raise RuntimeError("simulated transform interruption")
        return original_transform(**kwargs)

    monkeypatch.setattr(flatfield_pipeline, "_transform_region", _failing_transform)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        with pytest.raises(RuntimeError, match="simulated transform interruption"):
            run_flatfield_analysis(
                zarr_path=store_path,
                parameters={
                    "input_source": "data",
                    "fit_mode": "tiled",
                    "fit_tile_shape_yx": [2, 2],
                    "blend_tiles": False,
                    "smoothness_flatfield": 1.0,
                    "working_size": 64,
                    "use_map_overlap": True,
                    "overlap_zyx": [0, 1, 1],
                    "is_timelapse": False,
                },
                client=client,
            )
    finally:
        client.close()

    output_root = zarr.open_group(str(store_path), mode="r")
    checkpoint = output_root["results"]["flatfield"]["latest"]["checkpoint"]
    assert (
        int(np.count_nonzero(np.asarray(checkpoint["fit_tile_done_pcyx"], dtype=bool)))
        == 4
    )
    assert (
        int(
            np.count_nonzero(np.asarray(checkpoint["transform_done_tpcyx"], dtype=bool))
        )
        == 2
    )
    assert fit_calls["count"] == 4

    resumed_transform_trace = tmp_path / "resumed_transform_calls.txt"

    def _counting_transform(**kwargs):
        with resumed_transform_trace.open("a", encoding="utf-8") as handle:
            handle.write("1\n")
        return original_transform(**kwargs)

    monkeypatch.setattr(flatfield_pipeline, "_transform_region", _counting_transform)
    fit_calls_before_resume = int(fit_calls["count"])

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "fit_mode": "tiled",
                "fit_tile_shape_yx": [2, 2],
                "blend_tiles": False,
                "smoothness_flatfield": 1.0,
                "working_size": 64,
                "use_map_overlap": True,
                "overlap_zyx": [0, 1, 1],
                "is_timelapse": False,
            },
            client=client,
        )
    finally:
        client.close()

    resumed_root = zarr.open_group(str(store_path), mode="r")
    resumed_latest = resumed_root["results"]["flatfield"]["latest"]
    resumed_checkpoint = resumed_latest["checkpoint"]
    assert fit_calls["count"] == fit_calls_before_resume
    assert resumed_transform_trace.read_text(encoding="utf-8").count("\n") == 6
    assert np.all(np.asarray(resumed_checkpoint["transform_done_tpcyx"], dtype=bool))
    assert bool(resumed_latest.attrs["resumed_from_checkpoint"]) is True


def test_run_flatfield_analysis_restarts_when_parameters_change(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_resume_parameters.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 1, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 1, 2, 2),
        overwrite=True,
    )

    fit_calls = {"count": 0}

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
            fit_calls["count"] += 1
            self.flatfield = np.full(images.shape[1:], 2.0, dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)

    parameters = {
        "input_source": "data",
        "fit_mode": "tiled",
        "fit_tile_shape_yx": [2, 2],
        "blend_tiles": False,
        "smoothness_flatfield": 1.0,
        "working_size": 64,
        "use_map_overlap": True,
        "overlap_zyx": [0, 1, 1],
        "is_timelapse": False,
    }
    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters=parameters,
            client=client,
        )
    finally:
        client.close()

    first_run_fit_calls = int(fit_calls["count"])
    assert first_run_fit_calls == 4
    first_latest = zarr.open_group(str(store_path), mode="r")["results"]["flatfield"][
        "latest"
    ]
    first_fingerprint = str(first_latest.attrs["resume_parameter_fingerprint"])

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters=parameters,
            client=client,
        )
    finally:
        client.close()

    second_latest = zarr.open_group(str(store_path), mode="r")["results"]["flatfield"][
        "latest"
    ]
    second_fingerprint = str(second_latest.attrs["resume_parameter_fingerprint"])
    assert fit_calls["count"] == first_run_fit_calls
    assert second_fingerprint == first_fingerprint
    assert bool(second_latest.attrs["resumed_from_checkpoint"]) is True

    changed_parameters = dict(parameters)
    changed_parameters["smoothness_flatfield"] = 1.5
    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters=changed_parameters,
            client=client,
        )
    finally:
        client.close()

    third_latest = zarr.open_group(str(store_path), mode="r")["results"]["flatfield"][
        "latest"
    ]
    third_fingerprint = str(third_latest.attrs["resume_parameter_fingerprint"])
    assert fit_calls["count"] == first_run_fit_calls + 4
    assert third_fingerprint != first_fingerprint
    assert bool(third_latest.attrs["resumed_from_checkpoint"]) is False


def test_run_flatfield_analysis_tiled_n5_checkpoint_chunks_keep_full_rank(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_tiled_n5_chunk_headers.n5"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 4 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 4, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 2, 2),
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
            self.flatfield = np.ones(images.shape[1:], dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)
    parameters = {
        "input_source": "data",
        "fit_mode": "tiled",
        "fit_tile_shape_yx": [2, 2],
        "blend_tiles": False,
        "smoothness_flatfield": 1.0,
        "working_size": 64,
        "use_map_overlap": True,
        "overlap_zyx": [0, 1, 1],
        "is_timelapse": False,
    }

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters=parameters,
            client=client,
        )
    finally:
        client.close()

    output_root = zarr.open_group(str(store_path), mode="r")
    checkpoint_array = output_root["results"]["flatfield"]["latest"]["checkpoint"][
        "fit_baseline_sum_pctz"
    ]
    expected_chunk_shape = tuple(int(v) for v in checkpoint_array.chunks)

    chunk_root = (
        store_path
        / "results"
        / "flatfield"
        / "latest"
        / "checkpoint"
        / "fit_baseline_sum_pctz"
    )
    chunk_files = [
        p for p in chunk_root.rglob("*") if p.is_file() and p.name != "attributes.json"
    ]
    assert chunk_files, "Expected at least one written checkpoint chunk in N5 store."
    for chunk_file in chunk_files:
        with chunk_file.open("rb") as handle:
            header = handle.read(64)
        num_dims = struct.unpack(">H", header[2:4])[0]
        chunk_shape = tuple(
            struct.unpack(">I", header[index : index + 4])[0]
            for index in range(4, 4 + 4 * num_dims, 4)
        )[::-1]
        assert chunk_shape == expected_chunk_shape


def test_run_flatfield_analysis_restarts_on_malformed_n5_checkpoint_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_resume_malformed_checkpoint.n5"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 4 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 4, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 2, 2),
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
            self.flatfield = np.ones(images.shape[1:], dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = np.zeros(images.shape[0], dtype=np.float32)

    monkeypatch.setattr(flatfield_pipeline, "_load_basic_class", lambda: _FakeBaSiC)
    parameters = {
        "input_source": "data",
        "fit_mode": "tiled",
        "fit_tile_shape_yx": [2, 2],
        "blend_tiles": False,
        "smoothness_flatfield": 1.0,
        "working_size": 64,
        "use_map_overlap": True,
        "overlap_zyx": [0, 1, 1],
        "is_timelapse": False,
    }

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters=parameters,
            client=client,
        )
    finally:
        client.close()

    writable_root = zarr.open_group(str(store_path), mode="a")
    malformed = writable_root["results"]["flatfield"]["latest"]["checkpoint"][
        "fit_baseline_sum_pctz"
    ]
    malformed[0, 0, :, :] = np.asarray(
        malformed[0, 0, :, :], dtype=np.float32
    ) + np.float32(1.0)
    with pytest.raises(AssertionError, match="Expected chunk of shape"):
        np.asarray(malformed[0:1, 0:1, :, :], dtype=np.float32)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters=parameters,
            client=client,
        )
    finally:
        client.close()

    resumed_root = zarr.open_group(str(store_path), mode="r")
    resumed_latest = resumed_root["results"]["flatfield"]["latest"]
    assert bool(resumed_latest.attrs["resumed_from_checkpoint"]) is False


def test_run_flatfield_analysis_tiled_fallback_uses_full_volume_profile(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_tiled_fallback_full_volume.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 1, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 1, 2, 2),
        overwrite=True,
    )

    def _failing_tile(
        *,
        zarr_path,
        source_component,
        tile,
        parameters,
    ):
        del zarr_path, source_component, parameters
        if int(tile.tile_y_index) == 0 and int(tile.tile_x_index) == 0:
            raise RuntimeError(
                "torch._C._LinAlgError: linalg.svd: The algorithm failed to converge "
                "because the input matrix is ill-conditioned or has too many repeated "
                "singular values (error code: 1)."
            )
        y0, y1 = int(tile.y_core_bounds[0]), int(tile.y_core_bounds[1])
        x0, x1 = int(tile.x_core_bounds[0]), int(tile.x_core_bounds[1])
        return flatfield_pipeline.FlatfieldTileFitResult(
            position_index=int(tile.position_index),
            channel_index=int(tile.channel_index),
            tile_y_index=int(tile.tile_y_index),
            tile_x_index=int(tile.tile_x_index),
            y_bounds=(y0, y1),
            x_bounds=(x0, x1),
            flatfield_payload_yx=np.full((y1 - y0, x1 - x0), 99.0, dtype=np.float32),
            darkfield_payload_yx=np.zeros((y1 - y0, x1 - x0), dtype=np.float32),
            baseline_tz=np.zeros((1, 1), dtype=np.float32),
            weight_payload_yx=None,
        )

    def _full_volume_profile(
        *,
        zarr_path,
        source_component,
        position_index,
        channel_index,
        parameters,
    ):
        del zarr_path, source_component, parameters
        return flatfield_pipeline.FlatfieldProfileResult(
            position_index=int(position_index),
            channel_index=int(channel_index),
            flatfield_yx=np.full((4, 4), 5.0, dtype=np.float32),
            darkfield_yx=np.full((4, 4), 2.0, dtype=np.float32),
            baseline_tz=np.zeros((1, 1), dtype=np.float32),
        )

    monkeypatch.setattr(flatfield_pipeline, "_fit_profile_tile", _failing_tile)
    monkeypatch.setattr(
        flatfield_pipeline, "_fit_profile_full_volume", _full_volume_profile
    )

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "fit_mode": "tiled",
                "fit_tile_shape_yx": [2, 2],
                "blend_tiles": False,
                "smoothness_flatfield": 1.0,
                "working_size": 64,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "is_timelapse": False,
            },
            client=client,
        )
    finally:
        client.close()

    latest = zarr.open_group(str(store_path), mode="r")["results"]["flatfield"][
        "latest"
    ]
    checkpoint = latest["checkpoint"]
    fallback_records = list(checkpoint.attrs["fit_fallback_records"])

    assert str(checkpoint.attrs["run_status"]) == "complete_with_warnings"
    assert int(checkpoint.attrs["fit_warning_count"]) == 1
    assert len(fallback_records) == 1
    assert str(fallback_records[0]["fallback_mode"]) == "full_volume"
    assert str(latest.attrs["run_status"]) == "complete_with_warnings"
    assert int(latest.attrs["fit_warning_count"]) == 1

    flatfield = np.asarray(latest["flatfield_pcyx"], dtype=np.float32)[0, 0]
    darkfield = np.asarray(latest["darkfield_pcyx"], dtype=np.float32)[0, 0]
    corrected = np.asarray(latest["data"], dtype=np.float32)
    assert np.allclose(flatfield, 5.0)
    assert np.allclose(darkfield, 2.0)
    assert np.allclose(corrected, (data.astype(np.float32) - 2.0) / 5.0)


def test_run_flatfield_analysis_tiled_fallback_uses_identity_when_retry_fails(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_tiled_fallback_identity.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape((1, 1, 1, 1, 4, 4))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 1, 2, 2),
        overwrite=True,
    )

    def _always_failing_tile(
        *,
        zarr_path,
        source_component,
        tile,
        parameters,
    ):
        del zarr_path, source_component, tile, parameters
        raise RuntimeError(
            "torch._C._LinAlgError: linalg.svd: The algorithm failed to converge "
            "because the input matrix is ill-conditioned or has too many repeated "
            "singular values (error code: 1)."
        )

    def _failing_full_volume(
        *,
        zarr_path,
        source_component,
        position_index,
        channel_index,
        parameters,
    ):
        del zarr_path, source_component, position_index, channel_index, parameters
        raise RuntimeError("full-volume fallback failed")

    monkeypatch.setattr(flatfield_pipeline, "_fit_profile_tile", _always_failing_tile)
    monkeypatch.setattr(
        flatfield_pipeline, "_fit_profile_full_volume", _failing_full_volume
    )

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        run_flatfield_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "fit_mode": "tiled",
                "fit_tile_shape_yx": [2, 2],
                "blend_tiles": False,
                "smoothness_flatfield": 1.0,
                "working_size": 64,
                "use_map_overlap": False,
                "overlap_zyx": [0, 0, 0],
                "is_timelapse": False,
            },
            client=client,
        )
    finally:
        client.close()

    latest = zarr.open_group(str(store_path), mode="r")["results"]["flatfield"][
        "latest"
    ]
    checkpoint = latest["checkpoint"]
    fallback_records = list(checkpoint.attrs["fit_fallback_records"])

    assert str(checkpoint.attrs["run_status"]) == "complete_with_warnings"
    assert int(checkpoint.attrs["fit_warning_count"]) == 1
    assert len(fallback_records) == 1
    assert str(fallback_records[0]["fallback_mode"]) == "identity"
    assert "full-volume fallback failed" in str(fallback_records[0]["retry_error"])
    assert str(latest.attrs["run_status"]) == "complete_with_warnings"
    assert int(latest.attrs["fit_warning_count"]) == 1

    flatfield = np.asarray(latest["flatfield_pcyx"], dtype=np.float32)[0, 0]
    darkfield = np.asarray(latest["darkfield_pcyx"], dtype=np.float32)[0, 0]
    baseline = np.asarray(latest["baseline_pctz"], dtype=np.float32)[0, 0]
    corrected = np.asarray(latest["data"], dtype=np.float32)
    assert np.allclose(flatfield, 1.0)
    assert np.allclose(darkfield, 0.0)
    assert np.allclose(baseline, 0.0)
    assert np.allclose(corrected, data.astype(np.float32))
