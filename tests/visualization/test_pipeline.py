#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
import json
from pathlib import Path
import sys

# Third Party Imports
import dask.array as da
import numpy as np
import pytest
import zarr

# Local Imports
import clearex.visualization.export as visualization_export
import clearex.visualization.pipeline as visualization_pipeline
from clearex.visualization.pipeline import (
    run_display_pyramid_analysis,
    run_visualization_analysis,
)
from clearex.workflow import spatial_calibration_to_dict

_DISPLAY_PYRAMID_COMPONENT = "clearex/results/display_pyramid/latest"
_VISUALIZATION_COMPONENT = "clearex/results/visualization/latest"
_VISUALIZATION_LATEST_OUTPUT_COMPONENT = (
    "clearex/provenance/latest_outputs/visualization"
)


def _visualization_keyframe_manifest_path(store_path: Path) -> str:
    """Return the canonical in-store keyframe manifest path for tests."""
    return str((store_path.resolve() / _VISUALIZATION_COMPONENT / "keyframes.json"))


def _render_movie_output_directory(store_path: Path) -> str:
    """Return the canonical in-store render output directory for tests."""
    return str((store_path.resolve() / "clearex/results/render_movie/latest"))


def _single_image_volume_layers(
    component: str = "data",
    source_components: tuple[str, ...] = ("data",),
) -> tuple[visualization_pipeline.ResolvedVolumeLayer, ...]:
    """Return one default image-layer configuration for viewer tests."""
    return (
        visualization_pipeline.ResolvedVolumeLayer(
            component=str(component),
            source_components=tuple(str(item) for item in source_components),
            layer_type="image",
            channels=tuple(),
            visible=None,
            opacity=None,
            blending="",
            colormap="",
            rendering="",
            name="",
            multiscale_policy="inherit",
            multiscale_status="existing",
        ),
    )


def test_run_visualization_analysis_in_process_writes_latest_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(2, 2, 1, 4, 4, 4),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name="data_pyramid/level_1",
        shape=(2, 2, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.attrs.update(
        {
            "data_pyramid_levels": ["data", "data_pyramid/level_1"],
            "source_data_path": "/tmp/input.ome.tif",
            "source_data_component": "data",
        }
    )
    root["data"].attrs.update(
        {
            "pyramid_levels": ["data", "data_pyramid/level_1"],
            "scale_tpczyx": [2.0, 1.0, 1.0, 3.0, 4.0, 5.0],
        }
    )

    detections = np.asarray(
        [
            [0, 0, 0, 1, 2, 3, 1.2, 100.0],
            [1, 1, 0, 2, 1, 0, 0.8, 80.0],
        ],
        dtype=np.float32,
    )
    latest_group = root.require_group("clearex/results/particle_detection/latest")
    latest_group.create_dataset(
        name="detections",
        data=detections,
        overwrite=True,
    )

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        volume_layers,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
        require_gpu_rendering,
        capture_keyframes,
        keyframe_manifest_path,
        keyframe_layer_overrides,
    ) -> None:
        del zarr_path
        del point_properties_by_position
        del require_gpu_rendering
        del capture_keyframes
        del keyframe_layer_overrides
        if keyframe_manifest_path is not None:
            manifest_path = Path(str(keyframe_manifest_path))
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps({"schema_version": 2, "keyframes": []}),
                encoding="utf-8",
            )
        captured["volume_layers"] = list(volume_layers)
        captured["source_components"] = tuple(volume_layers[0].source_components)
        captured["selected_positions"] = tuple(
            int(value) for value in selected_positions
        )
        captured["points_by_position"] = {
            int(key): np.asarray(value, dtype=np.float32)
            for key, value in dict(points_by_position).items()
        }
        captured["position_affines_tczyx"] = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in dict(position_affines_tczyx).items()
        }
        captured["axis_labels"] = tuple(axis_labels)
        captured["scale_tczyx"] = tuple(float(value) for value in scale_tczyx)
        captured["image_metadata"] = dict(image_metadata)
        captured["points_metadata"] = dict(points_metadata)

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_viewer",
        _fake_launch_napari_viewer,
    )

    summary = run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "position_index": 1,
            "use_multiscale": True,
            "use_3d_view": False,
            "launch_mode": "in_process",
        },
    )

    assert summary.component == _VISUALIZATION_COMPONENT
    assert summary.source_component == "data"
    assert summary.source_components == ("data", "data_pyramid/level_1")
    assert summary.position_index == 1
    assert summary.selected_positions == (1,)
    assert summary.show_all_positions is False
    assert summary.overlay_points_count == 1
    assert summary.launch_mode == "in_process"
    assert summary.viewer_pid is None
    assert summary.viewer_ndisplay_requested == 2
    assert summary.viewer_ndisplay_effective == 2
    assert summary.display_mode_fallback_reason is None
    assert summary.keyframe_manifest_path == _visualization_keyframe_manifest_path(
        store_path
    )
    assert summary.keyframe_count == 0
    assert Path(summary.keyframe_manifest_path).exists()

    assert captured["source_components"] == ("data", "data_pyramid/level_1")
    assert len(captured["volume_layers"]) == 1
    assert captured["volume_layers"][0].component == "data"
    assert captured["selected_positions"] == (1,)
    points = np.asarray(captured["points_by_position"][1], dtype=np.float32)
    assert points.shape == (1, 5)
    assert np.allclose(points[0], np.asarray([1, 0, 2, 1, 0], dtype=np.float32))
    affine = np.asarray(captured["position_affines_tczyx"][1], dtype=np.float64)
    assert np.allclose(affine, np.eye(6, dtype=np.float64))
    assert captured["axis_labels"] == ("t", "c", "z", "y", "x")
    assert captured["scale_tczyx"] == (2.0, 1.0, 3.0, 4.0, 5.0)
    image_metadata = dict(captured["image_metadata"])
    assert image_metadata["source_component"] == "data"
    assert image_metadata["volume_layers"][0]["component"] == "data"
    assert image_metadata["position_index"] == 1
    assert image_metadata["selected_positions"] == [1]
    assert image_metadata["show_all_positions"] is False
    assert image_metadata["source_data_path"] == "/tmp/input.ome.tif"
    assert image_metadata["multiscale_levels"][0]["component"] == "data"
    points_metadata = dict(captured["points_metadata"])
    assert points_metadata["overlay_points_count"] == 1
    assert points_metadata["selected_positions"] == [1]
    assert points_metadata["coordinate_axes_tczyx"] == ["t", "c", "z", "y", "x"]

    output_root = zarr.open_group(str(store_path), mode="r")
    latest_attrs = dict(output_root[_VISUALIZATION_COMPONENT].attrs)
    assert latest_attrs["position_index"] == 1
    assert latest_attrs["selected_positions"] == [1]
    assert latest_attrs["show_all_positions"] is False
    assert latest_attrs["launch_mode"] == "in_process"
    assert latest_attrs["viewer_ndisplay_requested"] == 2
    assert latest_attrs["viewer_ndisplay_effective"] == 2
    assert latest_attrs["overlay_points_count"] == 1
    assert latest_attrs["source_components"] == ["data", "data_pyramid/level_1"]
    assert latest_attrs["volume_layers"][0]["component"] == "data"
    assert latest_attrs["capture_keyframes"] is True
    assert latest_attrs["keyframe_count"] == 0
    assert latest_attrs["keyframe_manifest_path"] == summary.keyframe_manifest_path
    assert latest_attrs["keyframe_layer_overrides"] == []
    assert (
        output_root[_VISUALIZATION_LATEST_OUTPUT_COMPONENT].attrs["component"]
        == _VISUALIZATION_COMPONENT
    )


def test_run_visualization_analysis_subprocess_launch(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    captured: dict[str, object] = {}

    def _fake_launch_napari_subprocess(
        *,
        zarr_path,
        normalized_parameters,
    ) -> int:
        captured["zarr_path"] = str(zarr_path)
        captured["parameters"] = dict(normalized_parameters)
        return 43210

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_subprocess",
        _fake_launch_napari_subprocess,
    )

    summary = run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "launch_mode": "subprocess",
            "overlay_particle_detections": False,
        },
    )

    assert summary.launch_mode == "subprocess"
    assert summary.viewer_pid == 43210
    assert summary.position_index == 0
    assert summary.selected_positions == (0,)
    assert summary.show_all_positions is False
    assert summary.viewer_ndisplay_requested == 3
    assert summary.viewer_ndisplay_effective == 3
    assert summary.display_mode_fallback_reason is None
    assert summary.keyframe_manifest_path == _visualization_keyframe_manifest_path(
        store_path
    )
    assert summary.keyframe_count == 0
    assert captured["zarr_path"] == str(store_path)
    assert dict(captured["parameters"])["launch_mode"] == "in_process"
    assert dict(captured["parameters"])[
        "keyframe_manifest_path"
    ] == _visualization_keyframe_manifest_path(store_path)
    assert dict(captured["parameters"])["keyframe_layer_overrides"] == []

    output_root = zarr.open_group(str(store_path), mode="r")
    latest_attrs = dict(output_root[_VISUALIZATION_COMPONENT].attrs)
    assert latest_attrs["viewer_pid"] == 43210
    assert latest_attrs["launch_mode"] == "subprocess"
    assert latest_attrs["viewer_ndisplay_requested"] == 3
    assert latest_attrs["viewer_ndisplay_effective"] == 3
    assert latest_attrs["capture_keyframes"] is True
    assert latest_attrs["keyframe_count"] == 0
    assert latest_attrs["keyframe_layer_overrides"] == []


def test_resolve_effective_launch_mode_auto_prefers_subprocess_with_qt_app(
    monkeypatch,
) -> None:
    class _FakeQApplication:
        @staticmethod
        def instance() -> object:
            return object()

    class _FakeQtWidgets:
        QApplication = _FakeQApplication

    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", _FakeQtWidgets)
    assert visualization_pipeline._resolve_effective_launch_mode("auto") == "subprocess"


def test_resolve_effective_launch_mode_auto_prefers_in_process_without_qt_app(
    monkeypatch,
) -> None:
    class _FakeQApplication:
        @staticmethod
        def instance() -> None:
            return None

    class _FakeQtWidgets:
        QApplication = _FakeQApplication

    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", _FakeQtWidgets)
    assert visualization_pipeline._resolve_effective_launch_mode("auto") == "in_process"


def test_run_visualization_analysis_auto_uses_subprocess_with_active_qt_app(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_array(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    class _FakeQApplication:
        @staticmethod
        def instance() -> object:
            return object()

    class _FakeQtWidgets:
        QApplication = _FakeQApplication

    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", _FakeQtWidgets)

    captured: dict[str, object] = {}

    def _fake_launch_napari_subprocess(
        *,
        zarr_path,
        normalized_parameters,
    ) -> int:
        captured["zarr_path"] = str(zarr_path)
        captured["parameters"] = dict(normalized_parameters)
        return 98765

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_subprocess",
        _fake_launch_napari_subprocess,
    )

    summary = run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "launch_mode": "auto",
            "overlay_particle_detections": False,
        },
    )

    assert summary.launch_mode == "subprocess"
    assert summary.viewer_pid == 98765
    assert captured["zarr_path"] == str(store_path)
    assert dict(captured["parameters"])["launch_mode"] == "in_process"


def test_run_visualization_analysis_rejects_invalid_position(tmp_path: Path) -> None:
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
        run_visualization_analysis(
            zarr_path=store_path,
            parameters={"position_index": 3, "launch_mode": "subprocess"},
        )
    except ValueError as exc:
        assert "out of bounds" in str(exc)
        return
    raise AssertionError("Expected ValueError for invalid position index")


def test_run_visualization_analysis_require_policy_rejects_without_pyramid(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    with pytest.raises(ValueError, match="requires multiscale levels"):
        run_visualization_analysis(
            zarr_path=store_path,
            parameters={
                "launch_mode": "in_process",
                "overlay_particle_detections": False,
                "volume_layers": [
                    {
                        "component": "data",
                        "multiscale_policy": "require",
                    }
                ],
            },
        )


def test_run_display_pyramid_analysis_materializes_levels_and_contrast_metadata(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="results/shear_transform/latest/data",
        shape=(1, 1, 2, 8, 8, 8),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )

    summary = run_display_pyramid_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "results/shear_transform/latest/data",
        },
    )

    assert summary.component == _DISPLAY_PYRAMID_COMPONENT
    assert summary.source_component == "results/shear_transform/latest/data"
    assert len(summary.source_components) > 1
    assert summary.channel_count == 2
    assert len(summary.contrast_limits_by_channel) == 2
    assert summary.reused_existing_levels is False

    output_root = zarr.open_group(str(store_path), mode="r")
    source_attrs = dict(output_root["results/shear_transform/latest/data"].attrs)
    latest_attrs = dict(output_root[_DISPLAY_PYRAMID_COMPONENT].attrs)
    assert len(source_attrs["display_pyramid_levels"]) > 1
    assert all(
        str(component).startswith("results/shear_transform/latest/data_pyramid/")
        or str(component) == "results/shear_transform/latest/data"
        for component in source_attrs["display_pyramid_levels"]
    )
    assert source_attrs["display_contrast_percentiles"] == [1.0, 95.0]
    assert len(source_attrs["display_contrast_limits_by_channel"]) == 2
    assert source_attrs["display_contrast_source_component"] == str(
        summary.source_components[-1]
    )
    assert latest_attrs["source_component"] == "results/shear_transform/latest/data"
    assert latest_attrs["display_contrast_source_component"] == str(
        summary.source_components[-1]
    )
    assert latest_attrs["reused_existing_levels"] is False


def test_run_display_pyramid_analysis_reuses_existing_levels(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 8, 8, 8),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name="data_pyramid/level_1",
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    root["data_pyramid/level_1"].attrs.update(
        {
            "axes": ["t", "p", "c", "z", "y", "x"],
            "pyramid_level": 1,
            "downsample_factors_tpczyx": [1, 1, 1, 2, 2, 2],
            "chunk_shape_tpczyx": [1, 1, 1, 4, 4, 4],
            "source_component": "data",
        }
    )
    root.attrs["data_pyramid_levels"] = ["data", "data_pyramid/level_1"]
    root["data"].attrs["pyramid_levels"] = ["data", "data_pyramid/level_1"]

    summary = run_display_pyramid_analysis(
        zarr_path=store_path,
        parameters={"input_source": "data"},
    )

    assert summary.source_components == ("data", "data_pyramid/level_1")
    assert summary.reused_existing_levels is True


def test_run_display_pyramid_analysis_rebuilds_incomplete_existing_levels(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    source_component = "results/shear_transform/latest/data"
    level_1_component = "results/shear_transform/latest/data_pyramid/level_1"
    stale_component = "results/shear_transform/latest/data_pyramid/stale_extra"

    source_data = np.arange(8 * 8 * 8, dtype=np.uint16).reshape((1, 1, 1, 8, 8, 8))
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name=source_component,
        shape=tuple(int(size) for size in source_data.shape),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    root[source_component][:] = source_data
    root.create_dataset(
        name=level_1_component,
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root[level_1_component][:] = np.full((1, 1, 1, 4, 4, 4), 65535, dtype=np.uint16)
    root.create_dataset(
        name=stale_component,
        shape=(1, 1, 1, 1, 1, 1),
        chunks=(1, 1, 1, 1, 1, 1),
        dtype="uint16",
        overwrite=True,
    )

    summary = run_display_pyramid_analysis(
        zarr_path=store_path,
        parameters={"input_source": source_component},
    )

    assert summary.reused_existing_levels is False

    output_root = zarr.open_group(str(store_path), mode="r")
    level_1 = np.asarray(output_root[level_1_component][:])
    expected_level_1 = source_data[:, :, :, ::2, ::2, ::2]
    np.testing.assert_array_equal(level_1, expected_level_1)
    assert stale_component not in output_root


def test_run_display_pyramid_analysis_avoids_forced_rechunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="results/shear_transform/latest/data",
        shape=(1, 1, 1, 8, 8, 8),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )

    def _unexpected_rechunk(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("Display pyramid should not force rechunk writes.")

    monkeypatch.setattr(da.Array, "rechunk", _unexpected_rechunk)

    summary = run_display_pyramid_analysis(
        zarr_path=store_path,
        parameters={"input_source": "results/shear_transform/latest/data"},
    )
    assert len(summary.source_components) > 1


def test_run_display_pyramid_analysis_emits_level_write_progress(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="results/shear_transform/latest/data",
        shape=(1, 1, 1, 16, 16, 16),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )

    events: list[tuple[int, str]] = []
    run_display_pyramid_analysis(
        zarr_path=store_path,
        parameters={"input_source": "results/shear_transform/latest/data"},
        progress_callback=lambda percent, message: events.append(
            (int(percent), str(message))
        ),
    )

    level_messages = [
        str(message) for _, message in events if "pyramid level" in message
    ]
    assert any("Writing pyramid level 1/" in message for message in level_messages)
    assert any("Wrote pyramid level 1/" in message for message in level_messages)
    assert any(
        30 < int(percent) < 70
        for percent, message in events
        if "pyramid level" in str(message)
    )


def test_run_display_pyramid_analysis_logs_level_chunk_estimates(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="results/shear_transform/latest/data",
        shape=(1, 1, 1, 16, 16, 16),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )

    with caplog.at_level("INFO", logger="clearex.visualization.pipeline"):
        run_display_pyramid_analysis(
            zarr_path=store_path,
            parameters={"input_source": "results/shear_transform/latest/data"},
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("writing level 1/" in message for message in messages)
    assert any("estimated_chunks=" in message for message in messages)


def test_run_display_pyramid_analysis_rebuilds_legacy_component_layout(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="results/shear_transform/latest/data",
        shape=(1, 1, 1, 8, 8, 8),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    legacy_level_component = (
        "results/display_pyramid/by_component/legacy_shear_cache/level_1"
    )
    root.create_dataset(
        name=legacy_level_component,
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    root["results/shear_transform/latest/data"].attrs["display_pyramid_levels"] = [
        "results/shear_transform/latest/data",
        legacy_level_component,
    ]
    root.attrs["display_pyramid_levels_by_component"] = {
        "results/shear_transform/latest/data": [
            "results/shear_transform/latest/data",
            legacy_level_component,
        ]
    }

    summary = run_display_pyramid_analysis(
        zarr_path=store_path,
        parameters={"input_source": "results/shear_transform/latest/data"},
    )

    assert summary.reused_existing_levels is False
    assert len(summary.source_components) > 1
    assert summary.source_components[0] == "results/shear_transform/latest/data"
    assert summary.source_components[1].startswith(
        "results/shear_transform/latest/data_pyramid/level_"
    )

    output_root = zarr.open_group(str(store_path), mode="r")
    assert (
        "results/shear_transform/latest/data_pyramid/level_1" in output_root
    ), "Expected source-adjacent level_1 pyramid after migration."
    source_attrs = dict(output_root["results/shear_transform/latest/data"].attrs)
    assert source_attrs["display_pyramid_levels"][1].startswith(
        "results/shear_transform/latest/data_pyramid/level_"
    )


def test_run_visualization_analysis_uses_experiment_spacing_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    experiment_path = tmp_path / "experiment.yml"
    experiment_payload = {
        "Saving": {
            "save_directory": str(tmp_path),
            "file_type": "TIFF",
        },
        "CameraParameters": {
            "img_x_pixels": 8,
            "img_y_pixels": 8,
            "Mesoscale": {
                "pixel_size": 0.75,
            },
        },
        "MicroscopeState": {
            "microscope_name": "Mesoscale",
            "timepoints": 2,
            "number_z_steps": 2,
            "timepoint_interval": 2.0,
            "step_size": 1.5,
            "channels": {
                "channel_1": {
                    "is_selected": True,
                }
            },
        },
    }
    experiment_path.write_text(json.dumps(experiment_payload))

    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(2, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.attrs.update({"source_experiment": str(experiment_path)})

    captured: dict[str, str | None] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        volume_layers,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
        require_gpu_rendering,
        capture_keyframes,
        keyframe_manifest_path,
        keyframe_layer_overrides,
    ) -> None:
        del zarr_path
        del volume_layers
        del selected_positions
        del points_by_position
        del point_properties_by_position
        del position_affines_tczyx
        del axis_labels
        del points_metadata
        del require_gpu_rendering
        del capture_keyframes
        del keyframe_manifest_path
        del keyframe_layer_overrides
        captured["scale_tczyx"] = tuple(float(value) for value in scale_tczyx)
        captured["image_metadata"] = dict(image_metadata)

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_viewer",
        _fake_launch_napari_viewer,
    )

    summary = run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "launch_mode": "in_process",
            "overlay_particle_detections": False,
        },
    )

    assert summary.launch_mode == "in_process"
    assert captured["scale_tczyx"] == (2.0, 1.0, 1.5, 0.75, 0.75)
    image_metadata = dict(captured["image_metadata"])
    source_experiment_metadata = image_metadata.get("source_experiment_metadata")
    assert isinstance(source_experiment_metadata, dict)
    assert source_experiment_metadata["MicroscopeState"]["step_size"] == 1.5


def test_run_visualization_analysis_prefers_voxel_size_um_attrs(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.attrs.update({"voxel_size_um_zyx": [2.5, 3.5, 4.5]})

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        volume_layers,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
        require_gpu_rendering,
        capture_keyframes,
        keyframe_manifest_path,
        keyframe_layer_overrides,
    ) -> None:
        del zarr_path
        del volume_layers
        del selected_positions
        del points_by_position
        del point_properties_by_position
        del position_affines_tczyx
        del axis_labels
        del image_metadata
        del points_metadata
        del require_gpu_rendering
        del capture_keyframes
        del keyframe_manifest_path
        del keyframe_layer_overrides
        captured["scale_tczyx"] = tuple(float(value) for value in scale_tczyx)

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_viewer",
        _fake_launch_napari_viewer,
    )

    run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "launch_mode": "in_process",
            "overlay_particle_detections": False,
        },
    )

    assert captured["scale_tczyx"] == (1.0, 1.0, 2.5, 3.5, 4.5)


def test_run_visualization_analysis_resolves_scale_from_source_chain(
    tmp_path: Path, monkeypatch
) -> None:
    """Visualization should recover scale from source-component ancestry."""
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    source = root.create_dataset(
        name="clearex/runtime_cache/source/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    source.attrs["voxel_size_um_zyx"] = [4.0, 1.5, 1.5]
    flatfield = root.create_dataset(
        name="clearex/runtime_cache/results/flatfield/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="float32",
        overwrite=True,
    )
    flatfield.attrs["source_component"] = "clearex/runtime_cache/source/data"
    shear = root.create_dataset(
        name="clearex/runtime_cache/results/shear_transform/latest/data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="float32",
        overwrite=True,
    )
    shear.attrs["source_component"] = (
        "clearex/runtime_cache/results/flatfield/latest/data"
    )

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        volume_layers,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
        require_gpu_rendering,
        capture_keyframes,
        keyframe_manifest_path,
        keyframe_layer_overrides,
    ) -> None:
        del zarr_path
        del volume_layers
        del selected_positions
        del points_by_position
        del point_properties_by_position
        del position_affines_tczyx
        del axis_labels
        del points_metadata
        del require_gpu_rendering
        del capture_keyframes
        del keyframe_manifest_path
        del keyframe_layer_overrides
        captured["scale_tczyx"] = tuple(float(value) for value in scale_tczyx)
        captured["image_metadata"] = dict(image_metadata)

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_viewer",
        _fake_launch_napari_viewer,
    )

    run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "clearex/runtime_cache/results/shear_transform/latest/data",
            "launch_mode": "in_process",
            "overlay_particle_detections": False,
        },
    )

    assert captured["scale_tczyx"] == (1.0, 1.0, 4.0, 1.5, 1.5)
    image_metadata = dict(captured["image_metadata"])
    assert image_metadata["voxel_size_um_zyx"] == [4.0, 1.5, 1.5]
    assert image_metadata["voxel_size_resolution_source"] == (
        "component:clearex/runtime_cache/source/data"
    )


def test_run_visualization_analysis_show_all_positions_uses_stage_affines(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    experiment_path = tmp_path / "experiment.yml"
    experiment_payload = {
        "Saving": {
            "save_directory": str(tmp_path),
            "file_type": "TIFF",
        },
        "CameraParameters": {
            "img_x_pixels": 8,
            "img_y_pixels": 8,
        },
        "MicroscopeState": {
            "timepoints": 1,
            "number_z_steps": 2,
            "channels": {
                "channel_1": {
                    "is_selected": True,
                }
            },
        },
    }
    experiment_path.write_text(json.dumps(experiment_payload))
    (tmp_path / "multi_positions.yml").write_text(
        json.dumps(
            [
                ["X", "Y", "Z", "THETA", "F"],
                [0, 0, 0, 0, 0],
                [10, 20, 30, 15, 0],
            ]
        )
    )

    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 2, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root.attrs.update({"source_experiment": str(experiment_path)})
    root["data"].attrs.update({"scale_tpczyx": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0]})

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        volume_layers,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
        require_gpu_rendering,
        capture_keyframes,
        keyframe_manifest_path,
        keyframe_layer_overrides,
    ) -> None:
        del zarr_path
        del volume_layers
        del points_by_position
        del point_properties_by_position
        del axis_labels
        del scale_tczyx
        del points_metadata
        del require_gpu_rendering
        del capture_keyframes
        del keyframe_manifest_path
        del keyframe_layer_overrides
        captured["selected_positions"] = tuple(
            int(value) for value in selected_positions
        )
        captured["position_affines_tczyx"] = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in dict(position_affines_tczyx).items()
        }
        captured["image_metadata"] = dict(image_metadata)

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_viewer",
        _fake_launch_napari_viewer,
    )

    summary = run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "show_all_positions": True,
            "overlay_particle_detections": False,
            "launch_mode": "in_process",
        },
    )

    assert summary.position_index == 0
    assert summary.selected_positions == (0, 1)
    assert summary.show_all_positions is True
    assert summary.overlay_points_count == 0
    assert captured["selected_positions"] == (0, 1)

    affines = dict(captured["position_affines_tczyx"])
    assert np.allclose(np.asarray(affines[0]), np.eye(6, dtype=np.float64))
    affine_1 = np.asarray(affines[1], dtype=np.float64)
    theta = np.deg2rad(15.0)
    assert np.isclose(affine_1[2, 2], np.cos(theta))
    assert np.isclose(affine_1[2, 3], np.sin(theta))
    assert np.isclose(affine_1[3, 2], -np.sin(theta))
    assert np.isclose(affine_1[3, 3], np.cos(theta))
    assert np.isclose(affine_1[2, 5], 30.0)
    assert np.isclose(affine_1[3, 5], 20.0)
    assert np.isclose(affine_1[4, 5], 10.0)

    image_metadata = dict(captured["image_metadata"])
    assert image_metadata["selected_positions"] == [0, 1]
    assert image_metadata["show_all_positions"] is True
    assert image_metadata["spatial_calibration_text"] == "z=+z,y=+y,x=+x"
    assert image_metadata["stage_positions_xyzthetaf"][1]["f"] == 0.0

    latest_attrs = dict(
        zarr.open_group(str(store_path), mode="r")[_VISUALIZATION_COMPONENT].attrs
    )
    assert latest_attrs["position_index"] == 0
    assert latest_attrs["selected_positions"] == [0, 1]
    assert latest_attrs["show_all_positions"] is True
    assert latest_attrs["spatial_calibration_text"] == "z=+z,y=+y,x=+x"


def test_resolve_position_affines_tczyx_supports_focus_axis_and_sign_inversion(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    experiment_path.write_text(
        json.dumps(
            {
                "Saving": {"save_directory": str(tmp_path), "file_type": "TIFF"},
                "MicroscopeState": {"timepoints": 1, "number_z_steps": 1},
            }
        )
    )
    (tmp_path / "multi_positions.yml").write_text(
        json.dumps(
            [
                ["X", "Y", "Z", "THETA", "F"],
                [0, 0, 0, 0, 0],
                [10, 20, 30, 15, 5],
            ]
        )
    )

    affines, stage_rows, spatial_calibration = (
        visualization_pipeline._resolve_position_affines_tczyx(
            root_attrs={
                "source_experiment": str(experiment_path),
                "spatial_calibration": {
                    "schema": "clearex.spatial_calibration.v1",
                    "stage_axis_map_zyx": {"z": "+f", "y": "-y", "x": "+x"},
                    "theta_mode": "rotate_zy_about_x",
                },
            },
            selected_positions=(0, 1),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        )
    )

    affine = np.asarray(affines[1], dtype=np.float64)

    assert stage_rows[1]["f"] == 5.0
    assert affine[2, 5] == 5.0
    assert affine[3, 5] == -20.0
    assert affine[4, 5] == 10.0
    assert spatial_calibration.stage_axis_map_zyx == ("+f", "-y", "+x")


def test_resolve_position_affines_tczyx_supports_none_and_nontrivial_mapping(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    experiment_path.write_text(
        json.dumps(
            {
                "Saving": {"save_directory": str(tmp_path), "file_type": "TIFF"},
                "MicroscopeState": {"timepoints": 1, "number_z_steps": 1},
            }
        )
    )
    (tmp_path / "multi_positions.yml").write_text(
        json.dumps(
            [
                ["X", "Y", "Z", "THETA", "F"],
                [0, 0, 0, 0, 0],
                [10, 20, 30, 15, 5],
            ]
        )
    )

    affines, _stage_rows, spatial_calibration = (
        visualization_pipeline._resolve_position_affines_tczyx(
            root_attrs={
                "source_experiment": str(experiment_path),
                "spatial_calibration": spatial_calibration_to_dict(
                    visualization_pipeline.SpatialCalibrationConfig(
                        stage_axis_map_zyx=("+x", "none", "+y")
                    )
                ),
            },
            selected_positions=(0, 1),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        )
    )

    affine = np.asarray(affines[1], dtype=np.float64)

    assert affine[2, 5] == 10.0
    assert affine[3, 5] == 0.0
    assert affine[4, 5] == 20.0
    assert spatial_calibration.stage_axis_map_zyx == ("+x", "none", "+y")


def test_launch_napari_viewer_applies_axis_labels_after_layer_load(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 2, 3, 4, 5), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.dims = _FakeDims()
            self.image_calls: list[tuple[object, dict[str, object]]] = []
            self.points_calls: list[tuple[object, dict[str, object]]] = []

        def add_image(self, data, **kwargs):
            self.image_calls.append((data, dict(kwargs)))
            if isinstance(data, list) and data:
                ndim = int(np.asarray(data[0]).ndim)
            else:
                ndim = int(np.asarray(data).ndim)
            self.dims.ndim = ndim
            return None

        def add_points(self, data, **kwargs):
            self.points_calls.append((data, dict(kwargs)))
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None
            self.run_called = False

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            self.run_called = True

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        points_by_position={
            0: np.asarray([[0, 0, 1, 2, 3]], dtype=np.float32),
        },
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert fake_napari.run_called is True
    assert viewer.dims.axis_labels == ("t", "c", "z", "y", "x")
    assert len(viewer.image_calls) == 2

    first_image_data, first_image_kwargs = viewer.image_calls[0]
    second_image_data, second_image_kwargs = viewer.image_calls[1]
    assert np.asarray(first_image_data).shape == (1, 1, 3, 4, 5)
    assert np.asarray(second_image_data).shape == (1, 1, 3, 4, 5)

    assert first_image_kwargs["colormap"] == "green"
    assert second_image_kwargs["colormap"] == "magenta"
    assert first_image_kwargs["blending"] == "additive"
    assert first_image_kwargs["projection_mode"] == "max"
    assert second_image_kwargs["projection_mode"] == "max"
    assert first_image_kwargs["rendering"] == "attenuated_mip"
    assert first_image_kwargs["opacity"] == 0.9
    assert second_image_kwargs["opacity"] == 0.9
    assert tuple(first_image_kwargs["contrast_limits"]) == (0.0, 65535.0)
    first_affine = np.asarray(first_image_kwargs["affine"], dtype=np.float64)
    second_affine = np.asarray(second_image_kwargs["affine"], dtype=np.float64)
    assert np.allclose(first_affine, np.eye(6, dtype=np.float64))
    assert np.allclose(second_affine, np.eye(6, dtype=np.float64))

    assert len(viewer.points_calls) == 1
    _, points_kwargs = viewer.points_calls[0]
    assert points_kwargs["border_color"] == "white"
    assert points_kwargs["opacity"] == 0.5
    assert points_kwargs["blending"] == "translucent"


def test_launch_napari_viewer_three_channels_use_requested_default_colormaps(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 3, 2, 2, 2), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.dims = _FakeDims()
            self.image_calls: list[dict[str, object]] = []

        def add_image(self, data, **kwargs):
            del data
            self.image_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert [kwargs["colormap"] for kwargs in viewer.image_calls] == [
        "green",
        "magenta",
        "bop orange",
    ]
    assert all(float(kwargs["opacity"]) == 0.8 for kwargs in viewer.image_calls)


def test_launch_napari_viewer_defaults_multiposition_images_to_translucent(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 2, 1, 3, 4, 5), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.dims = _FakeDims()
            self.image_calls: list[dict[str, object]] = []

        def add_image(self, data, **kwargs):
            del data
            self.image_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0, 1),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={
            0: np.eye(6, dtype=np.float64),
            1: np.eye(6, dtype=np.float64),
        },
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={"viewer_ndisplay_requested": 2, "viewer_ndisplay_effective": 2},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert len(viewer.image_calls) == 2
    assert all(
        str(kwargs["blending"]) == "translucent" for kwargs in viewer.image_calls
    )


def test_launch_napari_viewer_requests_3d_display_mode(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 1, 3, 4, 5), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del show
            self.initial_ndisplay = int(ndisplay)
            self.dims = _FakeDims()
            self.image_calls: list[dict[str, object]] = []

        def add_image(self, data, **kwargs):
            del data
            self.image_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert viewer.initial_ndisplay == 3
    assert len(viewer.image_calls) == 1
    assert viewer.image_calls[0]["rendering"] == "attenuated_mip"


def test_launch_napari_viewer_requests_2d_display_mode_when_metadata_disables_3d(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 1, 3, 4, 5), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del show
            self.initial_ndisplay = int(ndisplay)
            self.dims = _FakeDims()
            self.image_calls: list[dict[str, object]] = []

        def add_image(self, data, **kwargs):
            del data
            self.image_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={"viewer_ndisplay": 2},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert viewer.initial_ndisplay == 2
    assert len(viewer.image_calls) == 1


def test_launch_napari_viewer_resolves_per_layer_scale_for_downsampled_components(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 8, 8, 8),
        chunks=(1, 1, 1, 4, 4, 4),
        dtype="uint16",
        overwrite=True,
    )
    root.create_dataset(
        name="results/usegment3d/latest/data",
        shape=(1, 1, 1, 4, 4, 4),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint32",
        overwrite=True,
    )
    root.attrs["voxel_size_um_zyx"] = [1.5, 0.5, 0.5]

    primary_layer = visualization_pipeline.ResolvedVolumeLayer(
        component="data",
        source_components=("data",),
        layer_type="image",
        channels=tuple(),
        visible=None,
        opacity=None,
        blending="",
        colormap="",
        rendering="",
        name="Raw",
        multiscale_policy="off",
        multiscale_status="off",
    )
    labels_layer = visualization_pipeline.ResolvedVolumeLayer(
        component="results/usegment3d/latest/data",
        source_components=("results/usegment3d/latest/data",),
        layer_type="labels",
        channels=tuple(),
        visible=None,
        opacity=None,
        blending="",
        colormap="",
        rendering="",
        name="uSegment3D",
        multiscale_policy="off",
        multiscale_status="off",
    )

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.dims = _FakeDims()
            self.image_calls: list[dict[str, object]] = []
            self.labels_calls: list[dict[str, object]] = []

        def add_image(self, data, **kwargs):
            del data
            self.image_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_labels(self, data, **kwargs):
            del data
            self.labels_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=store_path,
        volume_layers=(primary_layer, labels_layer),
        selected_positions=(0,),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert len(viewer.image_calls) == 1
    assert len(viewer.labels_calls) == 1
    assert tuple(float(v) for v in viewer.image_calls[0]["scale"]) == (
        1.0,
        1.0,
        1.5,
        0.5,
        0.5,
    )
    assert tuple(float(v) for v in viewer.labels_calls[0]["scale"]) == (
        1.0,
        1.0,
        3.0,
        1.0,
        1.0,
    )


def test_run_visualization_analysis_large_3d_request_forces_2d_and_persists_reason(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 400, 1000, 1000),
        chunks=(1, 1, 1, 40, 100, 100),
        dtype="uint16",
        overwrite=True,
    )

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(**kwargs):
        captured["volume_layers"] = list(kwargs["volume_layers"])
        captured["image_metadata"] = dict(kwargs["image_metadata"])
        return {"keyframe_count": 0, "renderer": {}}

    monkeypatch.setattr(
        visualization_pipeline,
        "_launch_napari_viewer",
        _fake_launch_napari_viewer,
    )

    summary = run_visualization_analysis(
        zarr_path=store_path,
        parameters={
            "launch_mode": "in_process",
            "overlay_particle_detections": False,
            "use_3d_view": True,
        },
    )

    assert summary.viewer_ndisplay_requested == 3
    assert summary.viewer_ndisplay_effective == 2
    assert summary.display_mode_fallback_reason is not None
    assert "Falling back to 2D" in str(summary.display_mode_fallback_reason)
    image_metadata = dict(captured["image_metadata"])
    assert image_metadata["viewer_ndisplay_requested"] == 3
    assert image_metadata["viewer_ndisplay_effective"] == 2
    assert "display_mode_fallback_reason" in image_metadata

    latest_attrs = dict(
        zarr.open_group(str(store_path), mode="r")[_VISUALIZATION_COMPONENT].attrs
    )
    assert latest_attrs["viewer_ndisplay_requested"] == 3
    assert latest_attrs["viewer_ndisplay_effective"] == 2
    assert "display_mode_fallback_reason" in latest_attrs


def test_launch_napari_viewer_3d_uses_base_array_only_and_skips_percentiles(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 1, 4, 4, 4), dtype=np.uint16)
    level1 = np.zeros((1, 1, 1, 2, 2, 2), dtype=np.uint16)
    requested_components: list[str] = []

    def _fake_from_zarr(_path: str, *, component: str):
        requested_components.append(str(component))
        if component == "data":
            return source
        if component == "data_pyramid/level_1":
            return level1
        raise AssertionError(component)

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)
    monkeypatch.setattr(
        visualization_pipeline.da,
        "percentile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("viewer launch should not compute percentiles")
        ),
    )

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del show
            self.initial_ndisplay = int(ndisplay)
            self.dims = _FakeDims()
            self.image_calls: list[dict[str, object]] = []

        def add_image(self, data, **kwargs):
            del data
            self.image_calls.append(dict(kwargs))
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(
            source_components=("data", "data_pyramid/level_1")
        ),
        selected_positions=(0,),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={"viewer_ndisplay_requested": 3, "viewer_ndisplay_effective": 3},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=False,
        keyframe_manifest_path=None,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert viewer.initial_ndisplay == 3
    assert requested_components == ["data"]
    assert len(viewer.image_calls) == 1
    assert viewer.image_calls[0]["multiscale"] is False


def test_launch_napari_viewer_rejects_software_renderer_when_required(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 1, 2, 2, 2), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)
    monkeypatch.setattr(
        visualization_pipeline,
        "_probe_napari_opengl_renderer",
        lambda _viewer: {
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe",
            "version": "4.5",
            "software_renderer": True,
            "gpu_renderer": False,
        },
    )

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.dims = _FakeDims()

        def add_image(self, data, **kwargs):
            del data, kwargs
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def Viewer(self, *, ndisplay: int, show: bool):
            return _FakeViewer(ndisplay=ndisplay, show=show)

        def run(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "napari", _FakeNapari())

    with pytest.raises(RuntimeError, match="software-based"):
        visualization_pipeline._launch_napari_viewer(
            zarr_path=tmp_path / "analysis_store.zarr",
            volume_layers=_single_image_volume_layers(),
            selected_positions=(0,),
            points_by_position={},
            point_properties_by_position={},
            position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
            axis_labels=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
            image_metadata={},
            points_metadata={},
            require_gpu_rendering=True,
            capture_keyframes=False,
            keyframe_manifest_path=None,
            keyframe_layer_overrides=[],
        )


def test_normalize_visualization_volume_layers_treats_auto_as_unspecified() -> None:
    rows = visualization_pipeline._normalize_visualization_volume_layers(
        [
            {
                "component": "data",
                "layer_type": "image",
                "blending": "Auto",
                "colormap": "AUTO",
                "rendering": "auto",
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["blending"] == ""
    assert row["colormap"] == ""
    assert row["rendering"] == ""


def test_launch_napari_viewer_rejects_unconfirmed_renderer_when_required(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 1, 2, 2, 2), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)
    monkeypatch.setattr(
        visualization_pipeline,
        "_probe_napari_opengl_renderer",
        lambda _viewer: {
            "vendor": "",
            "renderer": "",
            "version": "",
            "software_renderer": False,
            "gpu_renderer": False,
            "gpu_vendor_hint": False,
            "error": "OpenGL context unavailable",
        },
    )

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.dims = _FakeDims()

        def add_image(self, data, **kwargs):
            del data, kwargs
            self.dims.ndim = 5
            return None

        def add_points(self, *_args, **_kwargs):
            return None

    class _FakeNapari:
        def Viewer(self, *, ndisplay: int, show: bool):
            return _FakeViewer(ndisplay=ndisplay, show=show)

        def run(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "napari", _FakeNapari())

    with pytest.raises(RuntimeError, match="did not confirm GPU rendering"):
        visualization_pipeline._launch_napari_viewer(
            zarr_path=tmp_path / "analysis_store.zarr",
            volume_layers=_single_image_volume_layers(),
            selected_positions=(0,),
            points_by_position={},
            point_properties_by_position={},
            position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
            axis_labels=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
            image_metadata={},
            points_metadata={},
            require_gpu_rendering=True,
            capture_keyframes=False,
            keyframe_manifest_path=None,
            keyframe_layer_overrides=[],
        )


def test_launch_napari_viewer_keyframe_hotkeys_write_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    source = np.zeros((1, 1, 1, 3, 4, 5), dtype=np.uint16)

    def _fake_from_zarr(_path: str, *, component: str):
        assert component == "data"
        return source

    monkeypatch.setattr(visualization_pipeline.da, "from_zarr", _fake_from_zarr)

    class _FakeCloseEvent:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

    class _FakeDims:
        def __init__(self) -> None:
            self.ndim = 2
            self.axis_labels = ("0", "1")
            self.current_step = (0, 0, 0, 0, 0)
            self.ndisplay = 3

    class _FakeLayer:
        def __init__(self, *, name: str, colormap: str | None = None) -> None:
            self.name = name
            self.visible = True
            self.opacity = 1.0
            self.blending = "opaque"
            self.contrast_limits = (0.0, 1.0)
            self.gamma = 1.0
            self.rendering = "attenuated_mip"
            if colormap is not None:
                self.colormap = type("Colormap", (), {"name": colormap})()

    class _FakeViewer:
        def __init__(self, *, ndisplay: int, show: bool) -> None:
            del ndisplay, show
            self.camera = type(
                "Camera",
                (),
                {
                    "angles": (11.0, 22.0, 33.0),
                    "zoom": 2.5,
                    "center": (4.0, 5.0, 6.0),
                },
            )()
            self.dims = _FakeDims()
            self.layers = []
            self.bound_keys: dict[str, object] = {}
            self.events = type("Events", (), {"close": _FakeCloseEvent()})()
            self.status = ""

        def add_image(self, data, **kwargs):
            del data
            layer = _FakeLayer(
                name=str(kwargs.get("name", "")),
                colormap=str(kwargs.get("colormap", "")) or None,
            )
            layer.opacity = float(kwargs.get("opacity", 1.0))
            layer.blending = str(kwargs.get("blending", "opaque"))
            layer.contrast_limits = tuple(kwargs.get("contrast_limits", (0.0, 1.0)))
            layer.rendering = str(kwargs.get("rendering", "attenuated_mip"))
            self.layers.append(layer)
            self.dims.ndim = 5
            return layer

        def add_points(self, data, **kwargs):
            del data, kwargs
            layer = _FakeLayer(name="Particle Detections")
            layer.blending = "translucent"
            layer.opacity = 0.5
            self.layers.append(layer)
            return layer

        def bind_key(self, key: str, callback=None, overwrite: bool = False):
            del overwrite
            if callback is None:

                def _decorator(func):
                    self.bound_keys[str(key)] = func
                    return func

                return _decorator
            self.bound_keys[str(key)] = callback
            return callback

    class _FakeNapari:
        def __init__(self) -> None:
            self.viewer: _FakeViewer | None = None

        def Viewer(self, *, ndisplay: int, show: bool):
            self.viewer = _FakeViewer(ndisplay=ndisplay, show=show)
            return self.viewer

        def run(self) -> None:
            return None

    fake_napari = _FakeNapari()
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    manifest_path = tmp_path / "viewer_keyframes.json"
    summary = visualization_pipeline._launch_napari_viewer(
        zarr_path=tmp_path / "analysis_store.zarr",
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        points_by_position={0: np.asarray([[0, 0, 1, 2, 3]], dtype=np.float32)},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={},
        points_metadata={},
        require_gpu_rendering=False,
        capture_keyframes=True,
        keyframe_manifest_path=manifest_path,
        keyframe_layer_overrides=[],
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert summary["keyframe_manifest_path"] == str(manifest_path)
    assert summary["keyframe_count"] == 0
    assert "k" in viewer.bound_keys
    assert "Shift-K" in viewer.bound_keys

    viewer.bound_keys["k"](viewer)
    viewer.bound_keys["k"](viewer)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["viewer_type"] == "3d"
    assert payload["selected_positions"] == [0]
    assert payload["keyframe_layer_overrides"] == []
    assert len(payload["keyframes"]) == 2
    assert payload["keyframes"][0]["id"] == "keyframe_0000"
    assert payload["keyframes"][0]["order_index"] == 0
    assert payload["keyframes"][0]["camera"]["angles"] == [11.0, 22.0, 33.0]
    assert payload["keyframes"][0]["dims"]["axis_labels"] == ["t", "c", "z", "y", "x"]
    assert payload["keyframes"][0]["viewer"]["selected_layers"] == []
    assert payload["keyframes"][0]["viewer"]["active_layer"] is None
    assert payload["keyframes"][0]["layers"][0]["name"] == "data (p=0)"
    assert payload["keyframes"][0]["layers"][0]["selected"] is False

    viewer.bound_keys["Shift-K"](viewer)
    payload_after_pop = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload_after_pop["keyframes"]) == 1


def test_run_render_movie_analysis_writes_frames_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_movie.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    keyframe_manifest_path = store_path / _VISUALIZATION_COMPONENT / "keyframes.json"
    keyframe_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    keyframe_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "captured_overlay_layers": [
                    {
                        "name": "Manual Points",
                        "type": "Points",
                        "data": [[0, 0, 1, 2, 3]],
                        "properties": {"track_id": [1]},
                        "visible": True,
                        "opacity": 0.7,
                        "blending": "translucent",
                        "size": [6.0],
                    },
                    {
                        "name": "Manual Tracks",
                        "type": "Tracks",
                        "data": [[0, 0, 1, 2], [1, 0, 1.5, 2.5]],
                        "visible": True,
                        "opacity": 0.8,
                        "blending": "translucent",
                        "tail_length": 10.0,
                    },
                ],
                "keyframes": [
                    {
                        "id": "keyframe_0000",
                        "index": 0,
                        "order_index": 0,
                        "camera": {
                            "angles": [0.0, 0.0, 0.0],
                            "zoom": 1.0,
                            "center": [0.0, 0.0, 0.0],
                            "perspective": 0.0,
                            "field_of_view": 0.0,
                        },
                        "dims": {
                            "current_step": [0, 0, 0, 0, 0],
                            "ndisplay": 3,
                            "order": [0, 1, 2, 3, 4],
                        },
                        "layers": [
                            {
                                "name": "Volume",
                                "visible": True,
                                "opacity": 1.0,
                                "blending": "opaque",
                                "rendering": "attenuated_mip",
                                "contrast_limits": [0.0, 1.0],
                                "scale": [1, 1, 1, 1, 1],
                            },
                            {
                                "name": "Manual Points",
                                "visible": True,
                                "opacity": 0.7,
                                "blending": "translucent",
                                "size": [6.0],
                            },
                            {
                                "name": "Manual Tracks",
                                "visible": True,
                                "opacity": 0.8,
                                "blending": "translucent",
                                "tail_length": 10.0,
                            },
                        ],
                    },
                    {
                        "id": "keyframe_0001",
                        "index": 1,
                        "order_index": 1,
                        "camera": {
                            "angles": [30.0, 40.0, 50.0],
                            "zoom": 2.0,
                            "center": [1.0, 2.0, 3.0],
                            "perspective": 5.0,
                            "field_of_view": 45.0,
                        },
                        "dims": {
                            "current_step": [0, 0, 1, 1, 1],
                            "ndisplay": 3,
                            "order": [0, 1, 2, 3, 4],
                        },
                        "layers": [
                            {
                                "name": "Volume",
                                "visible": True,
                                "opacity": 0.5,
                                "blending": "opaque",
                                "rendering": "attenuated_mip",
                                "contrast_limits": [0.0, 1.0],
                                "scale": [1, 1, 1, 1, 1],
                                "annotation": "End frame",
                            },
                            {
                                "name": "Manual Points",
                                "visible": True,
                                "opacity": 0.4,
                                "blending": "translucent",
                                "size": [8.0],
                            },
                            {
                                "name": "Manual Tracks",
                                "visible": True,
                                "opacity": 0.6,
                                "blending": "translucent",
                                "tail_length": 16.0,
                            },
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    root.require_group(_VISUALIZATION_COMPONENT).attrs.update(
        {
            "source_component": "data",
            "source_components": ["data", "data_pyramid/level_1"],
            "position_index": 0,
            "selected_positions": [0],
            "show_all_positions": False,
            "overlay_points_count": 0,
            "viewer_ndisplay_effective": 3,
            "keyframe_manifest_path": str(keyframe_manifest_path),
            "parameters": {
                "input_source": "data",
                "position_index": 0,
                "show_all_positions": False,
                "use_multiscale": True,
                "use_3d_view": True,
                "overlay_particle_detections": False,
                "volume_layers": [
                    {
                        "component": "data",
                        "name": "",
                        "layer_type": "image",
                        "channels": [],
                        "visible": None,
                        "opacity": None,
                        "blending": "",
                        "colormap": "",
                        "rendering": "",
                        "multiscale_policy": "inherit",
                    }
                ],
            },
        }
    )

    fake_scene = visualization_pipeline.PreparedVisualizationScene(
        normalized_parameters={},
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        reference_position_index=0,
        points_by_position={},
        point_properties_by_position={},
        total_overlay_points=0,
        source_component="data",
        source_components=("data", "data_pyramid/level_1"),
        viewer_ndisplay_requested=3,
        viewer_ndisplay_effective=3,
        display_mode_fallback_reason=None,
        napari_payload=visualization_pipeline.NapariLayerPayload(
            axis_labels_tczyx=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
            image_metadata={},
            points_metadata={},
        ),
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        spatial_calibration=visualization_pipeline.SpatialCalibrationConfig(),
    )

    class _FakeDims:
        def __init__(self) -> None:
            self.current_step = (0, 0, 0, 0, 0)
            self.order = (0, 1, 2, 3, 4)
            self.ndisplay = 3

    class _FakeCamera:
        def __init__(self) -> None:
            self.angles = (0.0, 0.0, 0.0)
            self.zoom = 1.0
            self.center = (0.0, 0.0, 0.0)
            self.perspective = 0.0
            self.fov = 0.0

    class _FakeLayer:
        def __init__(self, name: str = "Volume") -> None:
            self.name = name
            self.visible = True
            self.opacity = 1.0
            self.blending = "opaque"
            self.rendering = "attenuated_mip"
            self.contrast_limits = (0.0, 1.0)
            self.scale = (1.0, 1.0, 1.0, 1.0, 1.0)
            self.size = (6.0,)
            self.tail_length = 0.0
            self.refresh_calls = 0

        def refresh(self, *args, **kwargs) -> None:
            del args, kwargs
            self.refresh_calls += 1

    class _FakeCanvasNative:
        def __init__(self) -> None:
            self.resize_calls: list[tuple[int, int]] = []

        def resize(self, width: int, height: int) -> None:
            self.resize_calls.append((int(width), int(height)))

    class _FakeCanvas:
        def __init__(self) -> None:
            self.native = _FakeCanvasNative()

    class _FakeQtViewer:
        def __init__(self) -> None:
            self.canvas = _FakeCanvas()

    class _FakeWindow:
        def __init__(self) -> None:
            self.qt_viewer = _FakeQtViewer()
            self.resize_calls: list[tuple[int, int]] = []

        def resize(self, width: int, height: int) -> None:
            self.resize_calls.append((int(width), int(height)))

    class _FakeViewer:
        def __init__(self) -> None:
            self.dims = _FakeDims()
            self.camera = _FakeCamera()
            self.layers = [_FakeLayer()]
            self.window = _FakeWindow()
            self.screenshot_sizes: list[tuple[int, int]] = []

        def screenshot(self, *, canvas_only: bool, flash: bool, size):
            del canvas_only, flash
            self.screenshot_sizes.append(tuple(int(value) for value in size))
            height, width = size
            image = np.zeros((int(height), int(width), 4), dtype=np.uint8)
            image[..., 3] = 255
            return image

        def close(self) -> None:
            return None

        def add_points(self, data, **kwargs):
            del data
            layer = _FakeLayer(name=str(kwargs.get("name", "Points")))
            layer.opacity = float(kwargs.get("opacity", 1.0))
            layer.blending = str(kwargs.get("blending", "translucent"))
            if "size" in kwargs:
                layer.size = tuple(np.asarray(kwargs["size"]).tolist())
            self.layers.append(layer)
            return layer

        def add_tracks(self, data, **kwargs):
            del data
            layer = _FakeLayer(name=str(kwargs.get("name", "Tracks")))
            layer.opacity = float(kwargs.get("opacity", 1.0))
            layer.blending = str(kwargs.get("blending", "translucent"))
            layer.tail_length = float(kwargs.get("tail_length", 0.0))
            self.layers.append(layer)
            return layer

    built_viewers: list[_FakeViewer] = []
    build_scene_calls: list[dict[str, object]] = []

    def _fake_build_napari_viewer_scene(**kwargs):
        build_scene_calls.append(dict(kwargs))
        built_viewers.append(_FakeViewer())
        return visualization_pipeline.BuiltNapariScene(
            viewer=built_viewers[-1],
            renderer_info={"gpu_renderer": True},
            manifest_path=None,
            primary_source_component="data",
            primary_source_components=("data", "data_pyramid/level_1"),
            serialized_volume_layers=[],
            axis_labels_tczyx=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        )

    monkeypatch.setattr(
        visualization_pipeline,
        "_prepare_visualization_scene",
        lambda **kwargs: fake_scene,
    )
    monkeypatch.setattr(
        visualization_pipeline,
        "_process_pending_qt_events",
        lambda: None,
    )
    monkeypatch.setattr(visualization_pipeline.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        visualization_pipeline,
        "_build_napari_viewer_scene",
        _fake_build_napari_viewer_scene,
    )

    summary = visualization_pipeline.run_render_movie_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "visualization",
            "launch_mode": "in_process",
            "resolution_levels": [0, 1],
            "render_size_xy": [96, 64],
            "fps": 12,
            "default_transition_frames": 2,
            "hold_frames": 0,
            "overlay_frame_text_mode": "frame_number",
        },
    )

    assert summary.component == "clearex/results/render_movie/latest"
    assert summary.rendered_levels == (0, 1)
    assert summary.frame_count == 4
    assert summary.output_directory == _render_movie_output_directory(store_path)
    manifest = json.loads(
        Path(summary.render_manifest_path).read_text(encoding="utf-8")
    )
    assert manifest["rendered_levels"] == [0, 1]
    assert manifest["frame_count"] == 4
    assert len(built_viewers) == 2
    assert len(build_scene_calls) == 2
    assert all(bool(call.get("show", False)) for call in build_scene_calls)
    for viewer in built_viewers:
        layer_names = [str(layer.name) for layer in viewer.layers]
        assert "Manual Points" in layer_names
        assert "Manual Tracks" in layer_names
        assert viewer.window.qt_viewer.canvas.native.resize_calls == [(96, 64)]
        assert viewer.window.resize_calls == [(296, 114)]
        assert viewer.screenshot_sizes == [(64, 96)] * 4
        assert all(int(layer.refresh_calls) >= 4 for layer in viewer.layers)
    for level_index in (0, 1):
        frame_dir = Path(summary.output_directory) / f"level_{level_index:02d}_frames"
        assert frame_dir.exists()
        assert len(sorted(frame_dir.glob("frame_*.png"))) == 4


def test_run_render_movie_analysis_raises_when_visible_probe_is_blank(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_movie_blank_probe.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    keyframe_manifest_path = store_path / _VISUALIZATION_COMPONENT / "keyframes.json"
    keyframe_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    keyframe_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "keyframes": [
                    {
                        "id": "keyframe_0000",
                        "index": 0,
                        "order_index": 0,
                        "camera": {
                            "angles": [0.0, 0.0, 0.0],
                            "zoom": 1.0,
                            "center": [0.0, 0.0, 0.0],
                            "perspective": 0.0,
                            "field_of_view": 0.0,
                        },
                        "dims": {
                            "current_step": [0, 0, 0, 0, 0],
                            "ndisplay": 2,
                            "order": [0, 1, 2, 3, 4],
                        },
                        "layers": [{"name": "Volume", "visible": True}],
                    },
                    {
                        "id": "keyframe_0001",
                        "index": 1,
                        "order_index": 1,
                        "camera": {
                            "angles": [0.0, 0.0, 0.0],
                            "zoom": 1.0,
                            "center": [0.0, 0.0, 0.0],
                            "perspective": 0.0,
                            "field_of_view": 0.0,
                        },
                        "dims": {
                            "current_step": [0, 0, 0, 0, 0],
                            "ndisplay": 2,
                            "order": [0, 1, 2, 3, 4],
                        },
                        "layers": [{"name": "Volume", "visible": True}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    root.require_group(_VISUALIZATION_COMPONENT).attrs.update(
        {
            "source_component": "data",
            "source_components": ["data"],
            "position_index": 0,
            "selected_positions": [0],
            "show_all_positions": False,
            "overlay_points_count": 0,
            "viewer_ndisplay_effective": 2,
            "keyframe_manifest_path": str(keyframe_manifest_path),
            "parameters": {
                "input_source": "data",
                "position_index": 0,
                "show_all_positions": False,
                "use_multiscale": True,
                "use_3d_view": False,
                "overlay_particle_detections": False,
                "volume_layers": [
                    {
                        "component": "data",
                        "name": "",
                        "layer_type": "image",
                        "channels": [],
                        "visible": None,
                        "opacity": None,
                        "blending": "",
                        "colormap": "",
                        "rendering": "",
                        "multiscale_policy": "inherit",
                    }
                ],
            },
        }
    )

    fake_scene = visualization_pipeline.PreparedVisualizationScene(
        normalized_parameters={},
        volume_layers=_single_image_volume_layers(),
        selected_positions=(0,),
        reference_position_index=0,
        points_by_position={},
        point_properties_by_position={},
        total_overlay_points=0,
        source_component="data",
        source_components=("data",),
        viewer_ndisplay_requested=2,
        viewer_ndisplay_effective=2,
        display_mode_fallback_reason=None,
        napari_payload=visualization_pipeline.NapariLayerPayload(
            axis_labels_tczyx=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
            image_metadata={},
            points_metadata={},
        ),
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        spatial_calibration=visualization_pipeline.SpatialCalibrationConfig(),
    )

    class _FakeDims:
        def __init__(self) -> None:
            self.current_step = (0, 0, 0, 0, 0)
            self.order = (0, 1, 2, 3, 4)
            self.ndisplay = 2

    class _FakeCamera:
        def __init__(self) -> None:
            self.angles = (0.0, 0.0, 0.0)
            self.zoom = 1.0
            self.center = (0.0, 0.0, 0.0)
            self.perspective = 0.0
            self.fov = 0.0

    class _FakeLayer:
        def __init__(self) -> None:
            self.name = "Volume"
            self.visible = True
            self.opacity = 1.0
            self.blending = "opaque"
            self.refresh_calls = 0

        def refresh(self, *args, **kwargs) -> None:
            del args, kwargs
            self.refresh_calls += 1

    class _FakeCanvasNative:
        def resize(self, width: int, height: int) -> None:
            del width, height

    class _FakeCanvas:
        def __init__(self) -> None:
            self.native = _FakeCanvasNative()

    class _FakeQtViewer:
        def __init__(self) -> None:
            self.canvas = _FakeCanvas()

    class _FakeWindow:
        def __init__(self) -> None:
            self.qt_viewer = _FakeQtViewer()

        def resize(self, width: int, height: int) -> None:
            del width, height

    class _FakeViewer:
        def __init__(self) -> None:
            self.dims = _FakeDims()
            self.camera = _FakeCamera()
            self.layers = [_FakeLayer()]
            self.window = _FakeWindow()
            self.closed = False

        def screenshot(self, *, canvas_only: bool, flash: bool, size):
            del canvas_only, flash
            height, width = size
            image = np.zeros((int(height), int(width), 4), dtype=np.uint8)
            image[..., 3] = 255
            return image

        def close(self) -> None:
            self.closed = True

    build_scene_calls: list[dict[str, object]] = []

    def _fake_build_napari_viewer_scene(**kwargs):
        build_scene_calls.append(dict(kwargs))
        return visualization_pipeline.BuiltNapariScene(
            viewer=_FakeViewer(),
            renderer_info={"gpu_renderer": True},
            manifest_path=None,
            primary_source_component="data",
            primary_source_components=("data",),
            serialized_volume_layers=[],
            axis_labels_tczyx=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        )

    monkeypatch.setattr(
        visualization_pipeline,
        "_prepare_visualization_scene",
        lambda **kwargs: fake_scene,
    )
    monkeypatch.setattr(
        visualization_pipeline,
        "_process_pending_qt_events",
        lambda: None,
    )
    monkeypatch.setattr(visualization_pipeline.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        visualization_pipeline,
        "_build_napari_viewer_scene",
        _fake_build_napari_viewer_scene,
    )

    with pytest.raises(RuntimeError, match="empty napari probe frame"):
        visualization_pipeline.run_render_movie_analysis(
            zarr_path=store_path,
            parameters={
                "input_source": "visualization",
                "launch_mode": "in_process",
                "resolution_levels": [0],
                "render_size_xy": [96, 64],
                "fps": 12,
                "default_transition_frames": 1,
                "hold_frames": 0,
            },
        )

    assert [bool(call.get("show", False)) for call in build_scene_calls] == [True]


def test_run_render_movie_analysis_auto_uses_subprocess_with_active_qt_app(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_render_movie_subprocess.zarr"
    zarr.open_group(str(store_path), mode="w")

    class _FakeQApplication:
        @staticmethod
        def instance() -> object:
            return object()

    class _FakeQtWidgets:
        QApplication = _FakeQApplication

    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", _FakeQtWidgets)

    captured: dict[str, object] = {}
    captured_parameters: dict[str, object] = {}

    def _fake_run_render_movie_subprocess(
        *,
        zarr_path: str | Path,
        normalized_parameters: dict[str, object],
        run_id: str | None,
    ) -> None:
        captured["zarr_path"] = str(zarr_path)
        captured_parameters.clear()
        captured_parameters.update(dict(normalized_parameters))
        captured["run_id"] = run_id
        output_directory = Path(_render_movie_output_directory(store_path))
        output_directory.mkdir(parents=True, exist_ok=True)
        render_manifest_path = output_directory / "render_manifest.json"
        render_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "rendered_levels": [0],
                    "levels": [
                        {
                            "requested_level": 0,
                            "frame_directory": str(
                                (output_directory / "level_00_frames").resolve()
                            ),
                            "frame_pattern": "frame_%06d.png",
                            "frame_count": 3,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        visualization_pipeline._save_render_movie_metadata(
            zarr_path=zarr_path,
            visualization_component=_VISUALIZATION_COMPONENT,
            keyframe_manifest_path=_visualization_keyframe_manifest_path(store_path),
            render_manifest_path=str(render_manifest_path.resolve()),
            output_directory=str(output_directory.resolve()),
            rendered_levels=(0,),
            frame_count=3,
            fps=24,
            frame_directories_by_level={
                "0": str((output_directory / "level_00_frames").resolve())
            },
            parameters=normalized_parameters,
            source_component="data",
            source_components=("data",),
            run_id=run_id,
        )

    monkeypatch.setattr(
        visualization_pipeline,
        "_run_render_movie_subprocess",
        _fake_run_render_movie_subprocess,
    )

    summary = visualization_pipeline.run_render_movie_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "visualization",
            "resolution_levels": [0],
            "render_size_xy": [96, 64],
        },
        run_id="run-123",
    )

    assert captured["zarr_path"] == str(store_path)
    assert captured_parameters["launch_mode"] == "in_process"
    assert captured["run_id"] == "run-123"
    assert summary.component == "clearex/results/render_movie/latest"
    assert summary.visualization_component == _VISUALIZATION_COMPONENT
    assert summary.rendered_levels == (0,)
    assert summary.frame_count == 3
    assert summary.output_directory == _render_movie_output_directory(store_path)


def test_build_movie_render_viewer_scene_uses_visible_viewer(
    monkeypatch,
) -> None:
    call_log: list[dict[str, object]] = []

    class _DummyViewer:
        def close(self) -> None:
            return None

    def _fake_build_napari_viewer_scene(**kwargs):
        call_log.append(dict(kwargs))
        return visualization_pipeline.BuiltNapariScene(
            viewer=_DummyViewer(),
            renderer_info={"gpu_renderer": True},
            manifest_path=None,
            primary_source_component="data",
            primary_source_components=("data",),
            serialized_volume_layers=[],
            axis_labels_tczyx=("t", "c", "z", "y", "x"),
            scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        )

    monkeypatch.setattr(
        visualization_pipeline,
        "_build_napari_viewer_scene",
        _fake_build_napari_viewer_scene,
    )

    built_scene, used_visible_fallback = (
        visualization_pipeline._build_movie_render_viewer_scene(
            zarr_path=Path("/tmp/example.zarr"),
            scene=visualization_pipeline.PreparedVisualizationScene(
                normalized_parameters={},
                volume_layers=_single_image_volume_layers(),
                selected_positions=(0,),
                reference_position_index=0,
                points_by_position={},
                point_properties_by_position={},
                total_overlay_points=0,
                source_component="data",
                source_components=("data",),
                viewer_ndisplay_requested=2,
                viewer_ndisplay_effective=2,
                display_mode_fallback_reason=None,
                napari_payload=visualization_pipeline.NapariLayerPayload(
                    axis_labels_tczyx=("t", "c", "z", "y", "x"),
                    scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
                    image_metadata={},
                    points_metadata={},
                ),
                position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
                spatial_calibration=visualization_pipeline.SpatialCalibrationConfig(),
            ),
            movie_level_index=0,
        )
    )

    assert used_visible_fallback is True
    assert [bool(call.get("show", False)) for call in call_log] == [True]
    assert built_scene.renderer_info["gpu_renderer"] is True


def test_resolve_render_movie_keyframe_manifest_path_uses_store_default_when_needed(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store_default_manifest.zarr"
    zarr.open_group(str(store_path), mode="w")
    manifest_path = store_path / _VISUALIZATION_COMPONENT / "keyframes.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 2, "keyframes": []}),
        encoding="utf-8",
    )

    resolved = visualization_pipeline._resolve_render_movie_keyframe_manifest_path(
        zarr_path=store_path,
        parameters={"keyframe_manifest_path": ""},
        visualization_metadata={},
    )

    assert resolved == str(manifest_path.resolve())


def test_resolve_render_movie_keyframe_manifest_path_falls_back_to_legacy_sidecar(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "analysis_store_legacy_manifest.zarr"
    zarr.open_group(str(store_path), mode="w")
    manifest_path = Path(f"{store_path}.visualization_keyframes.json")
    manifest_path.write_text(
        json.dumps({"schema_version": 2, "keyframes": []}),
        encoding="utf-8",
    )

    resolved = visualization_pipeline._resolve_render_movie_keyframe_manifest_path(
        zarr_path=store_path,
        parameters={"keyframe_manifest_path": ""},
        visualization_metadata={},
    )

    assert resolved == str(manifest_path.resolve())


def test_run_subprocess_entrypoint_dispatches_render_movie(
    monkeypatch,
) -> None:
    captured: dict[str, str | None] = {}
    captured_parameters: dict[str, object] = {}

    def _fake_run_render_movie_analysis(
        *,
        zarr_path: str | Path,
        parameters: dict[str, object],
        progress_callback: object,
        run_id: str | None = None,
    ) -> visualization_pipeline.RenderMovieSummary:
        del progress_callback
        captured["zarr_path"] = str(zarr_path)
        captured_parameters.clear()
        captured_parameters.update(dict(parameters))
        captured["run_id"] = run_id
        return visualization_pipeline.RenderMovieSummary(
            component="clearex/results/render_movie/latest",
            visualization_component=_VISUALIZATION_COMPONENT,
            keyframe_manifest_path="/tmp/keyframes.json",
            render_manifest_path="/tmp/render_manifest.json",
            output_directory="/tmp/render_movie/latest",
            rendered_levels=(0,),
            frame_count=1,
            fps=24,
        )

    monkeypatch.setattr(
        visualization_pipeline,
        "run_render_movie_analysis",
        _fake_run_render_movie_analysis,
    )

    exit_code = visualization_pipeline._run_subprocess_entrypoint(
        [
            "--mode",
            "render_movie",
            "--zarr-path",
            "/tmp/example.ome.zarr",
            "--parameters-json",
            '{"input_source":"visualization","launch_mode":"subprocess"}',
            "--run-id",
            "movie-run-1",
        ]
    )

    assert exit_code == 0
    assert captured["zarr_path"] == "/tmp/example.ome.zarr"
    assert captured_parameters["launch_mode"] == "in_process"
    assert captured["run_id"] == "movie-run-1"


def test_run_compile_movie_analysis_writes_selected_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_compile.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(2):
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        image[..., 3] = 255
        visualization_pipeline.Image.fromarray(image, mode="RGBA").save(
            frames_dir / f"frame_{frame_index:06d}.png"
        )
    render_manifest_path = tmp_path / "render_manifest.json"
    render_manifest_path.write_text(
        json.dumps(
            {
                "fps": 18,
                "levels": [
                    {
                        "requested_level": 1,
                        "frame_directory": str(frames_dir),
                        "frame_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (
        root.require_group("clearex")
        .require_group("results")
        .require_group("render_movie")
        .require_group("latest")
    ).attrs.update({"render_manifest_path": str(render_manifest_path)})

    created_outputs: list[str] = []

    def _fake_mp4(**kwargs):
        output = Path(kwargs["output_path"])
        output.write_text("mp4", encoding="utf-8")
        created_outputs.append(str(output))
        return str(output)

    def _fake_prores(**kwargs):
        output = Path(kwargs["output_path"])
        output.write_text("mov", encoding="utf-8")
        created_outputs.append(str(output))
        return str(output)

    monkeypatch.setattr(visualization_pipeline, "compile_png_frames_to_mp4", _fake_mp4)
    monkeypatch.setattr(
        visualization_pipeline,
        "compile_png_frames_to_prores",
        _fake_prores,
    )

    summary = visualization_pipeline.run_compile_movie_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "render_movie",
            "rendered_level": 1,
            "output_format": "both",
            "fps": 24,
            "output_stem": "publication_movie",
        },
    )

    assert summary.component == "clearex/results/compile_movie/latest"
    assert summary.rendered_level == 1
    assert summary.output_format == "both"
    assert len(summary.compiled_files) == 2
    assert all(Path(path).exists() for path in summary.compiled_files)
    assert len(created_outputs) == 2


def test_compile_png_frames_to_mp4_uses_default_pixel_format_for_none(
    tmp_path: Path, monkeypatch
) -> None:
    frames_dir = tmp_path / "frames_mp4"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((16, 16, 4), dtype=np.uint8)
    frame[..., 3] = 255
    visualization_pipeline.Image.fromarray(frame, mode="RGBA").save(
        frames_dir / "frame_000000.png"
    )
    captured: dict[str, list[str]] = {}

    def _fake_run_ffmpeg(command: list[str]) -> None:
        captured["command"] = list(command)

    monkeypatch.setattr(visualization_export, "_run_ffmpeg", _fake_run_ffmpeg)

    visualization_export.compile_png_frames_to_mp4(
        frames_directory=frames_dir,
        output_path=tmp_path / "movie.mp4",
        fps=24,
        pixel_format=None,
    )

    command = captured["command"]
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"


def test_compile_png_frames_to_prores_uses_default_pixel_format_for_blank_values(
    tmp_path: Path, monkeypatch
) -> None:
    frames_dir = tmp_path / "frames_prores"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((16, 16, 4), dtype=np.uint8)
    frame[..., 3] = 255
    visualization_pipeline.Image.fromarray(frame, mode="RGBA").save(
        frames_dir / "frame_000000.png"
    )
    captured: dict[str, list[str]] = {}

    def _fake_run_ffmpeg(command: list[str]) -> None:
        captured["command"] = list(command)

    monkeypatch.setattr(visualization_export, "_run_ffmpeg", _fake_run_ffmpeg)

    visualization_export.compile_png_frames_to_prores(
        frames_directory=frames_dir,
        output_path=tmp_path / "movie.mov",
        fps=24,
        pixel_format="None",
    )

    command = captured["command"]
    assert command[command.index("-pix_fmt") + 1] == "yuv422p10le"
