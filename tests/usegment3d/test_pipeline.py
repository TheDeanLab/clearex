#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from clearex.io.experiment import create_dask_client
from clearex.usegment3d.pipeline import run_usegment3d_analysis
import clearex.usegment3d.pipeline as usegment_pipeline

_USEGMENT3D_PUBLIC_COMPONENT = "results/usegment3d/latest"
_USEGMENT3D_CACHE_DATA = "clearex/runtime_cache/results/usegment3d/latest/data"
_USEGMENT3D_PROVENANCE_ROOT = "clearex/provenance"


class _FakeParameterModule:
    """Minimal parameter-factory module used to mock segment3D.parameters."""

    @staticmethod
    def get_preprocess_params() -> dict[str, object]:
        return {
            "factor": 1.0,
            "voxel_res": [1.0, 1.0, 1.0],
            "do_bg_correction": True,
            "bg_ds": 16,
            "bg_sigma": 5.0,
            "normalize_min": 2.0,
            "normalize_max": 99.8,
            "do_avg_imgs": False,
        }

    @staticmethod
    def get_Cellpose_autotune_params() -> dict[str, object]:
        return {
            "cellpose_modelname": "cyto",
            "cellpose_channels": "grayscale",
            "hist_norm": False,
            "ksize": 15,
            "use_Cellpose_auto_diameter": True,
            "gpu": False,
            "best_diam": None,
            "model_invert": False,
            "test_slice": None,
            "diam_range": np.arange(10, 121, 2.5),
            "use_edge": True,
            "debug_viz": False,
            "saveplotsfolder": None,
        }

    @staticmethod
    def get_2D_to_3D_aggregation_params() -> dict[str, object]:
        return {
            "combine_cell_probs": {
                "prob_thresh": None,
                "threshold_n_levels": 3,
                "threshold_level": 1,
                "min_prob_thresh": 0.0,
            },
            "connected_component": {
                "min_area": 5,
                "smooth_sigma": 1.0,
                "thresh_factor": 0.0,
            },
            "postprocess_binary": {
                "binary_fill_holes": False,
            },
            "gradient_descent": {
                "gradient_decay": 0.0,
                "n_iter": 200,
                "momenta": 0.98,
                "do_mp": False,
                "tile_shape": (128, 256, 256),
                "tile_overlap_ratio": 0.25,
            },
            "indirect_method": {
                "n_cpu": None,
                "dtform_method": "cellpose_improve",
                "edt_fixed_point_percentile": 0.01,
            },
        }

    @staticmethod
    def get_postprocess_segmentation_params() -> dict[str, object]:
        return {
            "size_filters": {
                "min_size": 200,
            },
            "flow_consistency": {
                "do_flow_remove": True,
                "flow_threshold": 0.85,
                "dtform_method": "cellpose_improve",
                "edt_fixed_point_percentile": 0.01,
                "n_cpu": None,
            },
        }


class _FakeRuntimeModule:
    """Minimal runtime module used to mock segment3D.usegment3d."""

    @staticmethod
    def preprocess_imgs(img: np.ndarray, params: dict[str, object]) -> np.ndarray:
        del params
        return np.asarray(img, dtype=np.float32)

    @staticmethod
    def Cellpose2D_model_auto(
        img: np.ndarray,
        *,
        view: str,
        params: dict[str, object],
        basename: object,
        savefolder: object,
    ) -> tuple[
        tuple[object, object, object, object], np.ndarray, np.ndarray, np.ndarray
    ]:
        del params, basename, savefolder
        view_weight = {"xy": 1.0, "xz": 2.0, "yz": 3.0}.get(str(view), 1.0)
        shape = tuple(int(v) for v in img.shape[:3])
        probability = np.full(shape, view_weight, dtype=np.float32)
        flows = np.zeros((2, shape[0], shape[1], shape[2]), dtype=np.float32)
        return (
            (None, None, None, None),
            probability,
            flows,
            np.empty((0,), dtype=np.float32),
        )

    @staticmethod
    def aggregate_2D_to_3D_segmentation_direct_method(
        *,
        probs: list[object],
        gradients: list[object],
        params: dict[str, object],
        savefolder: object,
        basename: object,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        del gradients, params, savefolder, basename
        reference = None
        for item in probs:
            if isinstance(item, np.ndarray) and item.size > 0:
                reference = item
                break
        assert reference is not None
        labels = np.asarray(reference > 0, dtype=np.uint32)
        combined_prob = np.asarray(reference, dtype=np.float32)
        combined_gradients = np.zeros(labels.shape + (3,), dtype=np.float32)
        return labels, (combined_prob, combined_gradients)

    @staticmethod
    def postprocess_3D_cell_segmentation(
        segmentation: np.ndarray,
        *,
        aggregation_params: dict[str, object],
        postprocess_params: dict[str, object],
        cell_gradients: np.ndarray,
        savefolder: object,
        basename: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del aggregation_params, postprocess_params, cell_gradients, savefolder, basename
        return np.asarray(segmentation, dtype=np.uint32), {}


class _CompatRuntimeModule:
    """Runtime stub that emulates Cellpose auto-diameter unpack mismatch."""

    auto_flags_seen: list[bool] = []

    @staticmethod
    def preprocess_imgs(img: np.ndarray, params: dict[str, object]) -> np.ndarray:
        del params
        return np.asarray(img, dtype=np.float32)

    @classmethod
    def Cellpose2D_model_auto(
        cls,
        img: np.ndarray,
        *,
        view: str,
        params: dict[str, object],
        basename: object,
        savefolder: object,
    ) -> tuple[
        tuple[object, object, object, object], np.ndarray, np.ndarray, np.ndarray
    ]:
        del basename, savefolder
        auto_flag = bool(params.get("use_Cellpose_auto_diameter", False))
        cls.auto_flags_seen.append(auto_flag)
        if auto_flag:
            raise ValueError("not enough values to unpack (expected 4, got 3)")
        view_weight = {"xy": 1.0, "xz": 2.0, "yz": 3.0}.get(str(view), 1.0)
        shape = tuple(int(v) for v in img.shape[:3])
        probability = np.full(shape, view_weight, dtype=np.float32)
        flows = np.zeros((2, shape[0], shape[1], shape[2]), dtype=np.float32)
        return (
            (None, None, None, None),
            probability,
            flows,
            np.empty((0,), dtype=np.float32),
        )

    @staticmethod
    def aggregate_2D_to_3D_segmentation_direct_method(
        *,
        probs: list[object],
        gradients: list[object],
        params: dict[str, object],
        savefolder: object,
        basename: object,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        del gradients, params, savefolder, basename
        reference = None
        for item in probs:
            if isinstance(item, np.ndarray) and item.size > 0:
                reference = item
                break
        assert reference is not None
        labels = np.asarray(reference > 0, dtype=np.uint32)
        combined_prob = np.asarray(reference, dtype=np.float32)
        combined_gradients = np.zeros(labels.shape + (3,), dtype=np.float32)
        return labels, (combined_prob, combined_gradients)

    @staticmethod
    def postprocess_3D_cell_segmentation(
        segmentation: np.ndarray,
        *,
        aggregation_params: dict[str, object],
        postprocess_params: dict[str, object],
        cell_gradients: np.ndarray,
        savefolder: object,
        basename: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del aggregation_params, postprocess_params, cell_gradients, savefolder, basename
        return np.asarray(segmentation, dtype=np.uint32), {}


@pytest.mark.parametrize(
    ("raw_line", "expected"),
    [
        (
            "no CUDA. trying torch for normalizing",
            "uSegment3D: CuPy CUDA normalization path unavailable; trying Torch backend.",
        ),
        (
            "no gpu. falling back to CPU for resizing",
            "uSegment3D: GPU resizing backends unavailable; falling back to CPU.",
        ),
        ("single channel", "single channel"),
        ("   ", ""),
    ],
)
def test_remap_usegment3d_runtime_stdout_line(
    raw_line: str,
    expected: str,
) -> None:
    assert usegment_pipeline._remap_usegment3d_runtime_stdout_line(raw_line) == expected


class _PreprocessPrintRuntimeModule(_FakeRuntimeModule):
    """Runtime stub that emits legacy fallback prints during preprocessing."""

    @staticmethod
    def preprocess_imgs(img: np.ndarray, params: dict[str, object]) -> np.ndarray:
        del params
        print("no CUDA. trying torch for normalizing")
        print("single channel")
        return np.asarray(img, dtype=np.float32)


def test_segment_volume_rewrites_legacy_no_cuda_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _PreprocessPrintRuntimeModule()),
    )
    params = dict(usegment_pipeline._normalize_parameters({}))
    _ = usegment_pipeline._segment_volume(
        image_zyx=np.ones((4, 5, 6), dtype=np.float32),
        parameters=params,
    )

    captured = capsys.readouterr()
    assert (
        "uSegment3D: CuPy CUDA normalization path unavailable; trying Torch backend."
        in captured.out
    )
    assert "no CUDA. trying torch for normalizing" not in captured.out
    assert "single channel" in captured.out


def test_run_usegment3d_analysis_writes_latest_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 2 * 2 * 4 * 5 * 6, dtype=np.uint16).reshape((1, 2, 2, 4, 5, 6))
    root.create_dataset(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 3, 3),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=1, threads_per_worker=2, processes=False)
    try:
        summary = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "channel_index": 1,
                "use_views": ["xy", "yz"],
                "gpu": True,
                "require_gpu": False,
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.component == _USEGMENT3D_PUBLIC_COMPONENT
    assert summary.data_component == _USEGMENT3D_CACHE_DATA
    assert summary.source_component == "data"
    assert summary.volumes_processed == 2
    assert summary.channel_indices == (1,)
    assert summary.channel_index == 1
    assert summary.views == ("xy", "yz")
    assert summary.gpu_requested is True
    assert summary.gpu_enabled is False

    output_root = zarr.open_group(str(store_path), mode="r")
    labels = np.asarray(output_root[_USEGMENT3D_CACHE_DATA])
    assert tuple(labels.shape) == (1, 2, 1, 4, 5, 6)
    assert labels.dtype == np.uint32
    assert int(labels.max()) == 1

    latest_ref = dict(
        output_root[f"{_USEGMENT3D_PROVENANCE_ROOT}/latest_outputs/usegment3d"].attrs
    )
    assert latest_ref["component"] == _USEGMENT3D_PUBLIC_COMPONENT
    assert latest_ref["metadata"].get("channel_indices", []) == [1]
    assert latest_ref["metadata"].get("aggregation_tile_mode_effective") is latest_ref[
        "metadata"
    ].get("aggregation_tile_mode")
    assert latest_ref["metadata"].get("dropped_parameter_keys", []) == []


def test_run_usegment3d_analysis_supports_multiple_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_multi_channel.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 3, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=2, threads_per_worker=1, processes=False)
    try:
        summary = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "channel_indices": [0, 2],
                "use_views": ["xy"],
                "gpu": False,
                "require_gpu": False,
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.volumes_processed == 2
    assert summary.channel_indices == (0, 2)
    assert summary.channel_index == 0

    output_root = zarr.open_group(str(store_path), mode="r")
    labels = np.asarray(output_root[_USEGMENT3D_CACHE_DATA])
    assert tuple(labels.shape) == (1, 1, 2, 2, 2, 2)
    assert int(labels.max()) == 1

    latest_ref = dict(
        output_root[f"{_USEGMENT3D_PROVENANCE_ROOT}/latest_outputs/usegment3d"].attrs
    )
    metadata = dict(latest_ref.get("metadata", {}))
    assert metadata.get("channel_indices") == [0, 2]
    assert metadata.get("channels_processed") == 2


def test_run_usegment3d_analysis_supports_all_channels_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_all_channels.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 3, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=3, threads_per_worker=1, processes=False)
    try:
        summary = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "all_channels": True,
                "use_views": ["xy"],
                "gpu": False,
                "require_gpu": False,
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.volumes_processed == 3
    assert summary.channel_indices == (0, 1, 2)
    assert summary.channel_index == 0

    output_root = zarr.open_group(str(store_path), mode="r")
    labels = np.asarray(output_root[_USEGMENT3D_CACHE_DATA])
    assert tuple(labels.shape) == (1, 1, 3, 2, 2, 2)
    assert int(labels.max()) == 1

    latest_ref = dict(
        output_root[f"{_USEGMENT3D_PROVENANCE_ROOT}/latest_outputs/usegment3d"].attrs
    )
    metadata = dict(latest_ref.get("metadata", {}))
    assert metadata.get("channel_indices") == [0, 1, 2]
    assert metadata.get("channels_processed") == 3
    assert metadata.get("parameters", {}).get("all_channels") is True


def test_run_usegment3d_analysis_drops_unknown_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_drop_keys.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        _ = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "channel_index": 0,
                "use_views": ["xy"],
                "gpu": False,
                "require_gpu": False,
                "nonstandard_blob": bytes(1024),
            },
            client=client,
        )
    finally:
        client.close()

    output_root = zarr.open_group(str(store_path), mode="r")
    latest_ref = dict(
        output_root[f"{_USEGMENT3D_PROVENANCE_ROOT}/latest_outputs/usegment3d"].attrs
    )
    metadata = dict(latest_ref.get("metadata", {}))
    assert "nonstandard_blob" not in metadata.get("parameters", {})
    assert "nonstandard_blob" in metadata.get("dropped_parameter_keys", [])


def test_run_usegment3d_disables_tile_multiprocessing_with_distributed_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_tile_mode.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        _ = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "channel_index": 0,
                "use_views": ["xy"],
                "gpu": False,
                "require_gpu": False,
                "aggregation_tile_mode": True,
            },
            client=client,
        )
    finally:
        client.close()

    output_root = zarr.open_group(str(store_path), mode="r")
    latest_ref = dict(
        output_root[f"{_USEGMENT3D_PROVENANCE_ROOT}/latest_outputs/usegment3d"].attrs
    )
    metadata = dict(latest_ref.get("metadata", {}))
    assert metadata.get("aggregation_tile_mode_requested") is True
    assert metadata.get("aggregation_tile_mode_effective") is False
    assert metadata.get("aggregation_tile_mode") is False


def test_run_usegment3d_analysis_segments_pyramid_level_and_upscales_to_level0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_level_select.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    pyramid_group = root.require_group("data_pyramid")
    pyramid_group.create_dataset(
        name="level_1",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.attrs["voxel_size_um_zyx"] = [0.2, 0.1, 0.1]
    root.attrs["data_pyramid_factors_tpczyx"] = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2],
    ]

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        summary = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "input_resolution_level": 1,
                "output_reference_space": "level0",
                "save_native_labels": True,
                "channel_index": 0,
                "use_views": ["xy"],
                "gpu": False,
                "require_gpu": False,
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.volumes_processed == 1
    assert summary.source_component == "data_pyramid/level_1"

    output_root = zarr.open_group(str(store_path), mode="r")
    labels_level0 = np.asarray(output_root[_USEGMENT3D_CACHE_DATA])
    labels_native = np.asarray(
        output_root["clearex/runtime_cache/results/usegment3d/latest/data_native"]
    )
    assert tuple(labels_level0.shape) == (1, 1, 1, 4, 4, 4)
    assert tuple(labels_native.shape) == (1, 1, 1, 2, 2, 2)
    assert int(labels_level0.max()) == 1
    assert int(labels_native.max()) == 1

    latest_ref = dict(
        output_root[f"{_USEGMENT3D_PROVENANCE_ROOT}/latest_outputs/usegment3d"].attrs
    )
    metadata = dict(latest_ref.get("metadata", {}))
    assert metadata.get("source_component") == "data_pyramid/level_1"
    assert metadata.get("requested_source_component") == "data"
    assert metadata.get("output_reference_component") == "data"
    assert metadata.get("input_resolution_level") == 1
    assert metadata.get("output_resolution_level") == 0
    assert metadata.get("output_reference_space") == "level0"
    assert metadata.get("save_native_labels_effective") is True
    assert metadata.get("source_voxel_size_um_zyx") == [0.4, 0.2, 0.2]
    assert metadata.get("output_voxel_size_um_zyx") == [0.2, 0.1, 0.1]


def test_run_usegment3d_analysis_rejects_missing_requested_resolution_level(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_missing_level.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    with pytest.raises(ValueError, match="input_resolution_level"):
        run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "data",
                "input_resolution_level": 2,
                "channel_index": 0,
                "gpu": False,
                "require_gpu": False,
            },
            client=None,
        )


def test_run_usegment3d_analysis_rejects_invalid_channel(tmp_path: Path) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_invalid.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    with pytest.raises(ValueError, match="out of bounds"):
        run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={"channel_index": 5},
            client=None,
        )


def test_run_usegment3d_analysis_require_gpu_fails_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_gpu.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    with pytest.raises(RuntimeError, match="require_gpu=True"):
        run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "channel_index": 0,
                "gpu": True,
                "require_gpu": True,
            },
            client=None,
        )


def test_run_usegment3d_analysis_raises_diagnostic_on_cancelled_distributed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from distributed.client import FutureCancelledError

    store_path = tmp_path / "analysis_store_usegment3d_cancelled.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _FakeRuntimeModule()),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: True)
    monkeypatch.setattr(
        "dask.distributed.as_completed",
        lambda futures: iter(futures),
    )

    class _CancelledFuture:
        def result(self):
            raise FutureCancelledError(
                key="run",
                reason="unknown",
                msg=None,
            )

    class _DummyClient:
        def scheduler_info(self) -> dict[str, object]:
            return {
                "workers": {
                    "worker-0": {"resources": {"GPU": 1.0}, "nthreads": 1},
                }
            }

        def submit(self, *args, **kwargs):
            del args, kwargs
            return _CancelledFuture()

    with pytest.raises(RuntimeError, match="distributed task was cancelled"):
        run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "channel_index": 0,
                "gpu": True,
                "require_gpu": False,
            },
            client=_DummyClient(),
        )


def test_run_usegment3d_analysis_retries_when_auto_diameter_unpack_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store_usegment3d_auto_diam_compat.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 4, 5, 6),
        chunks=(1, 1, 1, 4, 5, 6),
        dtype="uint16",
        overwrite=True,
    )

    _CompatRuntimeModule.auto_flags_seen = []
    monkeypatch.setattr(
        usegment_pipeline,
        "_load_usegment3d_runtime",
        lambda: (_FakeParameterModule(), _CompatRuntimeModule),
    )
    monkeypatch.setattr(usegment_pipeline, "_is_gpu_available", lambda: False)

    client = create_dask_client(n_workers=1, threads_per_worker=1, processes=False)
    try:
        summary = run_usegment3d_analysis(
            zarr_path=store_path,
            parameters={
                "channel_index": 0,
                "use_views": ["xy"],
                "gpu": False,
                "require_gpu": False,
                "cellpose_use_auto_diameter": True,
                "cellpose_best_diameter": None,
            },
            client=client,
        )
    finally:
        client.close()

    assert summary.volumes_processed == 1
    assert True in _CompatRuntimeModule.auto_flags_seen
    assert False in _CompatRuntimeModule.auto_flags_seen
