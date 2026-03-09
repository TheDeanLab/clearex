#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
import json
from pathlib import Path
import sys

# Third Party Imports
import numpy as np
import zarr

# Local Imports
import clearex.visualization.pipeline as visualization_pipeline
from clearex.visualization.pipeline import run_visualization_analysis


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
    latest_group = (
        root.require_group("results")
        .require_group("particle_detection")
        .require_group("latest")
    )
    latest_group.create_dataset(
        name="detections",
        data=detections,
        overwrite=True,
    )

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        source_components,
        source_component,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
    ) -> None:
        del zarr_path, source_component, point_properties_by_position
        captured["source_components"] = tuple(source_components)
        captured["selected_positions"] = tuple(int(value) for value in selected_positions)
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
            "launch_mode": "in_process",
        },
    )

    assert summary.component == "results/visualization/latest"
    assert summary.source_component == "data"
    assert summary.source_components == ("data", "data_pyramid/level_1")
    assert summary.position_index == 1
    assert summary.selected_positions == (1,)
    assert summary.show_all_positions is False
    assert summary.overlay_points_count == 1
    assert summary.launch_mode == "in_process"
    assert summary.viewer_pid is None

    assert captured["source_components"] == ("data", "data_pyramid/level_1")
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
    latest_attrs = dict(output_root["results"]["visualization"]["latest"].attrs)
    assert latest_attrs["position_index"] == 1
    assert latest_attrs["selected_positions"] == [1]
    assert latest_attrs["show_all_positions"] is False
    assert latest_attrs["launch_mode"] == "in_process"
    assert latest_attrs["overlay_points_count"] == 1
    assert latest_attrs["source_components"] == ["data", "data_pyramid/level_1"]
    assert (
        output_root["provenance"]["latest_outputs"]["visualization"].attrs["component"]
        == "results/visualization/latest"
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
    assert captured["zarr_path"] == str(store_path)
    assert dict(captured["parameters"])["launch_mode"] == "in_process"

    output_root = zarr.open_group(str(store_path), mode="r")
    latest_attrs = dict(output_root["results"]["visualization"]["latest"].attrs)
    assert latest_attrs["viewer_pid"] == 43210
    assert latest_attrs["launch_mode"] == "subprocess"


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

    captured: dict[str, object] = {}

    def _fake_launch_napari_viewer(
        *,
        zarr_path,
        source_components,
        source_component,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
    ) -> None:
        del zarr_path
        del source_components
        del source_component
        del selected_positions
        del points_by_position
        del point_properties_by_position
        del position_affines_tczyx
        del axis_labels
        del points_metadata
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
        source_components,
        source_component,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
    ) -> None:
        del zarr_path
        del source_components
        del source_component
        del selected_positions
        del points_by_position
        del point_properties_by_position
        del position_affines_tczyx
        del axis_labels
        del image_metadata
        del points_metadata
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
        source_components,
        source_component,
        selected_positions,
        points_by_position,
        point_properties_by_position,
        position_affines_tczyx,
        axis_labels,
        scale_tczyx,
        image_metadata,
        points_metadata,
    ) -> None:
        del zarr_path
        del source_components
        del source_component
        del points_by_position
        del point_properties_by_position
        del axis_labels
        del scale_tczyx
        del points_metadata
        captured["selected_positions"] = tuple(int(value) for value in selected_positions)
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

    latest_attrs = dict(zarr.open_group(str(store_path), mode="r")["results"]["visualization"]["latest"].attrs)
    assert latest_attrs["position_index"] == 0
    assert latest_attrs["selected_positions"] == [0, 1]
    assert latest_attrs["show_all_positions"] is True


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
        source_components=("data",),
        source_component="data",
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
    assert first_image_kwargs["rendering"] == "attenuated_mip"
    assert first_image_kwargs["opacity"] == 0.9
    assert second_image_kwargs["opacity"] == 0.9
    assert tuple(first_image_kwargs["contrast_limits"]) == (0.0, 1.0)

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
        source_components=("data",),
        source_component="data",
        selected_positions=(0,),
        points_by_position={},
        point_properties_by_position={},
        position_affines_tczyx={0: np.eye(6, dtype=np.float64)},
        axis_labels=("t", "c", "z", "y", "x"),
        scale_tczyx=(1.0, 1.0, 1.0, 1.0, 1.0),
        image_metadata={},
        points_metadata={},
    )

    viewer = fake_napari.viewer
    assert viewer is not None
    assert [kwargs["colormap"] for kwargs in viewer.image_calls] == [
        "green",
        "magenta",
        "bop orange",
    ]
    assert all(float(kwargs["opacity"]) == 0.8 for kwargs in viewer.image_calls)
