#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

# Standard Library Imports
from contextlib import ExitStack
from pathlib import Path
import json

# Third Party Imports
import dask.array as da
import h5py
import numpy as np
import pytest
import tifffile
import zarr

# Local Imports
import clearex.io.experiment as experiment_module
import clearex.io.read as read_module
from clearex.io.experiment import (
    default_analysis_store_path,
    find_experiment_data_candidates,
    has_canonical_data_component,
    has_complete_canonical_data_store,
    initialize_analysis_store,
    is_navigate_experiment_file,
    load_navigate_experiment,
    load_navigate_experiment_source_image_info,
    load_store_spatial_calibration,
    materialize_experiment_data_store,
    resolve_data_store_path,
    resolve_experiment_data_path,
    save_store_spatial_calibration,
    write_zyx_block,
)
from clearex.io.ome_store import (
    SOURCE_CACHE_COMPONENT,
    SOURCE_CACHE_PYRAMID_ROOT,
    source_cache_component,
)
from clearex.io.read import ImageInfo
from clearex.workflow import SpatialCalibrationConfig


def _write_minimal_experiment(
    path: Path,
    save_directory: Path,
    file_type: str = "H5",
    *,
    is_multiposition: bool = False,
):
    payload = {
        "Saving": {
            "save_directory": str(save_directory),
            "file_type": file_type,
        },
        "MicroscopeState": {
            "timepoints": 2,
            "number_z_steps": 4,
            "is_multiposition": is_multiposition,
            "channels": {
                "channel_1": {"is_selected": True, "laser": "488nm"},
                "channel_2": {"is_selected": True, "laser": "562nm"},
                "channel_3": {"is_selected": False, "laser": "642nm"},
            },
        },
        "CameraParameters": {
            "img_x_pixels": 16,
            "img_y_pixels": 8,
        },
        "MultiPositions": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_multipositions_sidecar(path: Path, count: int) -> None:
    header = ["X", "Y", "Z", "THETA", "F", "X_PIXEL", "Y_PIXEL"]
    rows = [header]
    for idx in range(count):
        rows.append([float(idx), float(idx), float(idx), 0.0, 0.0, "NaN", "NaN"])
    path.write_text(json.dumps(rows, indent=2))


def _write_bdv_xml(
    path: Path,
    *,
    loader_format: str,
    data_file_name: str,
    setup_channel_tile: dict[int, tuple[int, int]],
) -> None:
    if loader_format == "bdv.hdf5":
        loader_key = "hdf5"
    elif loader_format == "bdv.n5":
        loader_key = "n5"
    else:
        loader_key = "zarr"
    setup_blocks = []
    for setup_index in sorted(setup_channel_tile):
        channel_index, tile_index = setup_channel_tile[setup_index]
        setup_blocks.append(f"""
      <ViewSetup>
        <id>{setup_index}</id>
        <name>{setup_index}</name>
        <size>4 3 2</size>
        <voxelSize>
          <unit>um</unit>
          <size>1.0 1.0 1.0</size>
        </voxelSize>
        <attributes>
          <illumination>0</illumination>
          <channel>{channel_index}</channel>
          <tile>{tile_index}</tile>
          <angle>0</angle>
        </attributes>
      </ViewSetup>
""".rstrip())

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="{loader_format}">
      <{loader_key} type="relative">{data_file_name}</{loader_key}>
    </ImageLoader>
    <ViewSetups>
{chr(10).join(setup_blocks)}
    </ViewSetups>
    <Timepoints type="range">
      <first>0</first>
      <last>0</last>
    </Timepoints>
  </SequenceDescription>
</SpimData>
"""
    path.write_text(xml)


def _tensorstore_module():
    """Import TensorStore for real N5 fixture creation."""
    return pytest.importorskip("tensorstore")


def _write_real_n5_dataset(
    root_path: Path,
    *,
    component: str,
    data_xyz: np.ndarray,
    block_size_xyz: tuple[int, int, int],
) -> None:
    """Create one real N5 dataset using TensorStore."""
    ts = _tensorstore_module()
    spec = {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": str(root_path)},
        "path": component,
        "metadata": {
            "blockSize": [int(v) for v in block_size_xyz],
            "compression": {
                "type": "blosc",
                "cname": "lz4",
                "clevel": 5,
                "shuffle": 1,
                "blocksize": 0,
            },
            "dataType": np.dtype(data_xyz.dtype).name,
            "dimensions": [int(v) for v in data_xyz.shape],
        },
        "create": True,
        "delete_existing": True,
    }
    dataset = ts.open(spec).result()
    dataset[...] = data_xyz


def _strip_n5_component_payload_files(root_path: Path, *, component: str) -> None:
    """Remove chunk payload files while preserving ``attributes.json``."""
    component_path = root_path / component
    if not component_path.exists():
        return
    for path in component_path.rglob("*"):
        if path.is_file() and path.name != "attributes.json":
            path.unlink()
    for path in sorted(
        (entry for entry in component_path.rglob("*") if entry.is_dir()),
        key=lambda entry: len(entry.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            continue


def _write_legacy_n5_group(
    root_path: Path,
    *,
    component: str,
    schema: str = "clearex.analysis_store.v1",
) -> None:
    """Write a stale legacy ClearEx group marker into an N5 tree."""
    group_path = root_path / component
    group_path.mkdir(parents=True, exist_ok=True)
    (group_path / "attributes.json").write_text(json.dumps({"schema": schema}))


def test_normalize_gpu_device_ids_deduplicates_and_strips() -> None:
    values = [0, "1", " 1 ", "", "2", "0", "  "]
    assert experiment_module._normalize_gpu_device_ids(values) == ["0", "1", "2"]


def test_detect_visible_gpu_device_ids_parses_nvidia_smi(monkeypatch) -> None:
    class _Result:
        returncode = 0
        stdout = "0\n1\n2\n"

    monkeypatch.setattr(
        experiment_module.subprocess,
        "run",
        lambda *args, **kwargs: _Result(),
    )

    assert experiment_module._detect_visible_gpu_device_ids() == ["0", "1", "2"]


def test_library_path_env_vars_for_platform_windows() -> None:
    assert experiment_module._library_path_env_vars_for_platform(
        os_name="nt",
        platform="win32",
    ) == ("PATH",)


def test_library_path_env_vars_for_platform_linux() -> None:
    assert experiment_module._library_path_env_vars_for_platform(
        os_name="posix",
        platform="linux",
    ) == ("LD_LIBRARY_PATH",)


def test_build_library_path_environment_updates_merges_discovered_and_inherited() -> (
    None
):
    updates = experiment_module._build_library_path_environment_updates(
        ["/cuda/runtime/lib", "/cuda/cudnn/lib"],
        env={"LD_LIBRARY_PATH": "/cluster/custom/lib"},
        env_var_names=("LD_LIBRARY_PATH",),
    )
    assert updates == {
        "LD_LIBRARY_PATH": "/cuda/runtime/lib:/cuda/cudnn/lib:/cluster/custom/lib"
    }


def test_create_dask_client_gpu_worker_env_merges_inherited_library_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyClient:
        def __init__(self, cluster):
            captured["cluster"] = cluster

    class _DummyCluster:
        pass

    def _fake_spec_cluster(*, scheduler, workers, asynchronous):
        captured["scheduler"] = scheduler
        captured["workers"] = workers
        captured["asynchronous"] = asynchronous
        return _DummyCluster()

    monkeypatch.setattr("distributed.deploy.spec.SpecCluster", _fake_spec_cluster)
    monkeypatch.setattr("dask.distributed.Client", _DummyClient)
    monkeypatch.setattr(
        experiment_module,
        "_normalize_gpu_device_ids",
        lambda values: ["0"],
    )
    monkeypatch.setattr(
        experiment_module,
        "_detect_visible_gpu_device_ids",
        lambda: ["0"],
    )
    monkeypatch.setattr(
        experiment_module,
        "_detect_cuda_library_paths",
        lambda: ["/cuda/runtime/lib", "/cuda/cudnn/lib"],
    )
    path_env_var = experiment_module._library_path_env_vars_for_platform()[0]
    monkeypatch.setenv(path_env_var, "/cluster/custom/lib")

    _ = experiment_module.create_dask_client(
        gpu_enabled=True,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
    )

    workers = dict(captured["workers"])
    worker_options = dict(workers["gpu-worker-00"]["options"])
    worker_env = dict(worker_options["env"])
    assert worker_env[path_env_var] == (
        "/cuda/runtime/lib:/cuda/cudnn/lib:/cluster/custom/lib"
    )


def test_is_navigate_experiment_file():
    assert is_navigate_experiment_file("experiment.yml") is True
    assert is_navigate_experiment_file("experiment.yaml") is True
    assert is_navigate_experiment_file("settings.yml") is False


def test_load_and_resolve_experiment_data_path(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="H5")

    # Two H5 candidates; CH00 should resolve first via deterministic sorting.
    (tmp_path / "CH01_000000.h5").write_bytes(b"")
    (tmp_path / "CH00_000000.h5").write_bytes(b"")

    experiment = load_navigate_experiment(experiment_path)
    resolved = resolve_experiment_data_path(experiment)

    assert experiment.channel_count == 2
    assert experiment.timepoints == 2
    assert experiment.multiposition_count == 3
    assert resolved.name == "CH00_000000.h5"


def test_load_navigate_experiment_preserves_windows_absolute_save_directory(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    payload = {
        "Saving": {
            "save_directory": r"Z:\bioinformatics\Danuser_lab\Dean\acquisition_001",
            "file_type": "N5",
        },
        "MicroscopeState": {
            "timepoints": 1,
            "number_z_steps": 2,
            "channels": {"channel_1": {"is_selected": True, "laser": "488nm"}},
        },
        "CameraParameters": {"img_x_pixels": 8, "img_y_pixels": 8},
    }
    experiment_path.write_text(json.dumps(payload, indent=2))

    experiment = load_navigate_experiment(experiment_path)

    assert (
        str(experiment.save_directory)
        == r"Z:\bioinformatics\Danuser_lab\Dean\acquisition_001"
    )


def test_resolve_experiment_data_path_falls_back_to_experiment_directory(
    tmp_path: Path,
):
    experiment_dir = tmp_path / "bundle"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = experiment_dir / "experiment.yml"
    payload = {
        "Saving": {
            "save_directory": r"Z:\bioinformatics\Danuser_lab\Dean\acquisition_001",
            "file_type": "N5",
        },
        "MicroscopeState": {
            "timepoints": 1,
            "number_z_steps": 2,
            "channels": {"channel_1": {"is_selected": True, "laser": "488nm"}},
        },
        "CameraParameters": {"img_x_pixels": 8, "img_y_pixels": 8},
    }
    experiment_path.write_text(json.dumps(payload, indent=2))
    local_store = experiment_dir / "CH00_000000.n5"
    local_store.mkdir()

    experiment = load_navigate_experiment(experiment_path)
    resolved = resolve_experiment_data_path(experiment)

    assert resolved == local_store.resolve()


def test_resolve_experiment_data_path_uses_override_directory(tmp_path: Path):
    experiment_dir = tmp_path / "bundle"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = experiment_dir / "experiment.yml"
    missing_remote_dir = tmp_path / "remote_missing"
    _write_minimal_experiment(
        experiment_path,
        save_directory=missing_remote_dir,
        file_type="N5",
    )

    override_dir = tmp_path / "override_data"
    override_dir.mkdir(parents=True, exist_ok=True)
    override_store = override_dir / "CH00_000000.n5"
    override_store.mkdir()

    experiment = load_navigate_experiment(experiment_path)
    resolved = resolve_experiment_data_path(
        experiment,
        search_directory=override_dir,
    )

    assert resolved == override_store.resolve()


def test_initialize_analysis_store_creates_6d_layout(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="H5")
    experiment = load_navigate_experiment(experiment_path)
    store_path = default_analysis_store_path(experiment)

    image_info = ImageInfo(
        path=tmp_path / "dummy.h5",
        shape=(4, 8, 16),
        dtype=np.uint16,
        axes="ZYX",
    )

    output = initialize_analysis_store(
        experiment=experiment,
        zarr_path=store_path,
        image_info=image_info,
        overwrite=True,
    )
    assert output == store_path.resolve()

    root = zarr.open_group(str(store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (2, 3, 2, 4, 8, 16)
    assert list(root[SOURCE_CACHE_COMPONENT].attrs["axes"]) == [
        "t",
        "p",
        "c",
        "z",
        "y",
        "x",
    ]
    assert metadata["storage_policy_analysis_outputs"] == "latest_only"
    assert list(root[SOURCE_CACHE_COMPONENT].attrs["chunk_shape_tpczyx"]) == [
        1,
        1,
        1,
        4,
        8,
        16,
    ]
    assert list(metadata["configured_chunks_tpczyx"]) == [1, 1, 1, 256, 256, 256]


def test_initialize_analysis_store_applies_custom_chunks_and_pyramid(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="H5")
    experiment = load_navigate_experiment(experiment_path)
    store_path = default_analysis_store_path(experiment)

    image_info = ImageInfo(
        path=tmp_path / "dummy.h5",
        shape=(16, 64, 64),
        dtype=np.uint16,
        axes="ZYX",
    )

    initialize_analysis_store(
        experiment=experiment,
        zarr_path=store_path,
        image_info=image_info,
        overwrite=True,
        chunks=(1, 1, 1, 8, 32, 32),
        pyramid_factors=((1,), (1,), (1,), (1, 2, 4), (1, 2), (1, 2, 4)),
    )

    root = zarr.open_group(str(store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].chunks) == (1, 1, 1, 8, 32, 32)
    assert list(root[SOURCE_CACHE_COMPONENT].attrs["chunk_shape_tpczyx"]) == [
        1,
        1,
        1,
        8,
        32,
        32,
    ]
    assert list(metadata["configured_chunks_tpczyx"]) == [1, 1, 1, 8, 32, 32]
    assert metadata["resolution_pyramid_factors_tpczyx"] == [
        [1],
        [1],
        [1],
        [1, 2, 4],
        [1, 2],
        [1, 2, 4],
    ]


def test_initialize_analysis_store_backfills_identity_spatial_calibration(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="H5")
    experiment = load_navigate_experiment(experiment_path)
    store_path = default_analysis_store_path(experiment)

    initialize_analysis_store(
        experiment=experiment, zarr_path=store_path, overwrite=True
    )

    calibration = load_store_spatial_calibration(store_path)

    assert calibration == SpatialCalibrationConfig()


def test_load_store_spatial_calibration_defaults_to_identity_for_legacy_store(
    tmp_path: Path,
):
    store_path = tmp_path / "legacy_store.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    calibration = load_store_spatial_calibration(store_path)

    assert calibration == SpatialCalibrationConfig()


def test_save_store_spatial_calibration_round_trip_and_preserves_existing_mapping(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="H5")
    experiment = load_navigate_experiment(experiment_path)
    store_path = tmp_path / "store_with_mapping.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    saved = save_store_spatial_calibration(
        store_path,
        SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y")),
    )
    initialize_analysis_store(
        experiment=experiment,
        zarr_path=store_path,
        overwrite=False,
    )

    reloaded = load_store_spatial_calibration(store_path)

    assert saved == SpatialCalibrationConfig(stage_axis_map_zyx=("+x", "none", "+y"))
    assert reloaded == saved


def test_write_zyx_block_numpy(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="H5")
    experiment = load_navigate_experiment(experiment_path)
    store_path = default_analysis_store_path(experiment)
    initialize_analysis_store(
        experiment=experiment, zarr_path=store_path, overwrite=True
    )

    block = np.ones((4, 8, 16), dtype=np.uint16)
    write_zyx_block(
        zarr_path=store_path,
        block=block,
        t_index=1,
        p_index=2,
        c_index=1,
    )

    root = zarr.open_group(str(store_path), mode="r")
    loaded = np.array(root[SOURCE_CACHE_COMPONENT][1, 2, 1, :, :, :])
    assert np.array_equal(loaded, block)


def test_resolve_ome_tiff_prefers_primary_over_mip(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="OME-TIFF"
    )

    mip_file = tmp_path / "MIP" / "P0000_CH00_000000.tiff"
    primary_file = tmp_path / "Position0" / "P0000_CH00_000000.tiff"
    mip_file.parent.mkdir(parents=True, exist_ok=True)
    primary_file.parent.mkdir(parents=True, exist_ok=True)
    mip_file.write_bytes(b"")
    primary_file.write_bytes(b"")

    experiment = load_navigate_experiment(experiment_path)
    candidates = find_experiment_data_candidates(experiment)
    resolved = resolve_experiment_data_path(experiment)

    assert len(candidates) == 1
    assert candidates[0] == primary_file
    assert resolved == primary_file


def test_resolve_ome_zarr_alias(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="OME-ZARR"
    )

    ome_zarr = tmp_path / "CH00_000000.ome.zarr"
    ome_zarr.mkdir(parents=True, exist_ok=True)

    experiment = load_navigate_experiment(experiment_path)
    resolved = resolve_experiment_data_path(experiment)

    assert resolved == ome_zarr


def test_load_uses_multi_positions_sidecar_when_multiposition_enabled(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="H5",
        is_multiposition=True,
    )
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=24)

    experiment = load_navigate_experiment(experiment_path)

    assert experiment.multiposition_count == 24


def test_load_uses_experiment_dir_multi_positions_sidecar_when_save_directory_is_windows_path(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    payload = {
        "Saving": {
            "save_directory": r"E:\acquisition\remote_only",
            "file_type": "N5",
        },
        "MicroscopeState": {
            "timepoints": 1,
            "number_z_steps": 2,
            "is_multiposition": True,
            "multiposition_count": 1,
            "channels": {"channel_1": {"is_selected": True, "laser": "488nm"}},
        },
        "CameraParameters": {"img_x_pixels": 8, "img_y_pixels": 8},
        "MultiPositions": [[0.0, 0.0, 0.0]],
    }
    experiment_path.write_text(json.dumps(payload, indent=2))
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=2)

    experiment = load_navigate_experiment(experiment_path)

    assert experiment.multiposition_count == 2


def test_load_infers_xy_pixel_size_from_zoom_and_binning(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    payload = {
        "Saving": {
            "save_directory": str(tmp_path),
            "file_type": "TIFF",
        },
        "MicroscopeState": {
            "microscope_name": "Mesoscale",
            "zoom": "0.5x",
            "timepoints": 1,
            "number_z_steps": 2,
            "channels": {
                "channel_1": {"is_selected": True},
            },
        },
        "CameraParameters": {
            "img_x_pixels": 16,
            "img_y_pixels": 8,
            "Mesoscale": {
                "img_x_pixels": 16,
                "img_y_pixels": 8,
                "binning": "2x2",
                "pixel_size": 6.5,
            },
        },
        "MultiPositions": [[0, 0, 0]],
    }
    experiment_path.write_text(json.dumps(payload, indent=2))

    experiment = load_navigate_experiment(experiment_path)

    assert experiment.xy_pixel_size_um is not None
    assert np.isclose(experiment.xy_pixel_size_um, 26.0)


def test_resolve_data_store_path_uses_experiment_directory_for_non_zarr(tmp_path: Path):
    experiment_dir = tmp_path / "metadata"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_directory = tmp_path / "acquisition"
    save_directory.mkdir(parents=True, exist_ok=True)

    experiment_path = experiment_dir / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=save_directory, file_type="H5"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_path = save_directory / "source.npy"
    np.save(source_path, np.zeros((2, 3, 4), dtype=np.uint16))

    resolved = resolve_data_store_path(experiment, source_path)

    assert resolved == (experiment_dir / "data_store.ome.zarr").resolve()


def test_materialize_experiment_data_store_creates_data_store_for_non_zarr(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    expected_store = (experiment_path.parent / "data_store.ome.zarr").resolve()
    assert materialized.store_path == expected_store
    root = zarr.open_group(str(expected_store), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 1, 1, 2, 3, 4)
    assert tuple(root[SOURCE_CACHE_COMPONENT].chunks) == (1, 1, 1, 2, 2, 2)
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert metadata["data_pyramid_levels"] == [
        SOURCE_CACHE_COMPONENT,
        source_cache_component(level_index=1),
    ]
    assert tuple(root[source_cache_component(level_index=1)].shape) == (
        1,
        1,
        1,
        1,
        2,
        2,
    )
    assert np.array_equal(
        np.array(root[source_cache_component(level_index=1)][0, 0, 0, :, :, :]),
        source_data[::2, ::2, ::2],
    )


def test_materialize_experiment_data_store_batches_chunk_writes(
    tmp_path: Path, monkeypatch
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    compute_calls: list[object] = []
    original_compute = experiment_module._compute_dask_graph

    def _counting_compute(graph, *, client=None):
        compute_calls.append(graph)
        return original_compute(graph, client=client)

    monkeypatch.setattr(
        experiment_module,
        "_estimate_write_batch_region_count",
        lambda **kwargs: 1,
    )
    monkeypatch.setattr(experiment_module, "_compute_dask_graph", _counting_compute)

    materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 1, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    expected_store = (experiment_path.parent / "data_store.ome.zarr").resolve()
    root = zarr.open_group(str(expected_store), mode="r")
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert len(compute_calls) == 8


def test_materialize_experiment_data_store_resumes_after_interrupted_base_write(
    tmp_path: Path,
    monkeypatch,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    original_compute = experiment_module._compute_dask_graph

    monkeypatch.setattr(
        experiment_module,
        "_estimate_write_batch_region_count",
        lambda **kwargs: 1,
    )

    failing_call_count = {"value": 0}

    def _failing_compute(graph, *, client=None):
        failing_call_count["value"] += 1
        if failing_call_count["value"] == 4:
            raise RuntimeError("simulated interruption")
        return original_compute(graph, client=client)

    monkeypatch.setattr(experiment_module, "_compute_dask_graph", _failing_compute)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_experiment_data_store(
            experiment=experiment,
            source_path=source_path,
            chunks=(1, 1, 1, 1, 2, 2),
            pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        )

    expected_store = (experiment_path.parent / "data_store.ome.zarr").resolve()
    root = zarr.open_group(str(expected_store), mode="r")
    progress = dict(root.attrs["ingestion_progress"])
    assert progress["status"] == "in_progress"
    assert progress["base_progress"]["completed_regions"] == 3
    assert progress["base_progress"]["total_regions"] == 8

    resume_call_count = {"value": 0}

    def _counting_compute(graph, *, client=None):
        resume_call_count["value"] += 1
        return original_compute(graph, client=client)

    monkeypatch.setattr(experiment_module, "_compute_dask_graph", _counting_compute)

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 1, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    assert resume_call_count["value"] == 5
    root = zarr.open_group(str(materialized.store_path), mode="r")
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    progress = dict(root.attrs["ingestion_progress"])
    assert progress["status"] == "completed"
    assert progress["base_progress"]["completed_regions"] == 8
    assert progress["base_progress"]["total_regions"] == 8


def test_materialize_experiment_data_store_handles_multibatch_base_and_pyramid(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(5 * 17 * 17, dtype=np.uint16).reshape(5, 17, 17)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 1, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert np.array_equal(
        np.array(root[source_cache_component(level_index=1)][0, 0, 0, :, :, :]),
        source_data[::2, ::2, ::2],
    )


def test_materialize_experiment_data_store_reuses_existing_zarr_store(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="OME-ZARR"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_store = tmp_path / "source.ome.zarr"
    source_root = zarr.open_group(str(source_store), mode="w")
    source_root.create_dataset(
        "raw", data=source_data, chunks=(1, 3, 4), overwrite=True
    )
    source_root["raw"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_store,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    assert materialized.store_path == source_store.resolve()
    root = zarr.open_group(str(source_store), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert SOURCE_CACHE_COMPONENT in root
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 1, 1, 2, 3, 4)
    assert tuple(root[SOURCE_CACHE_COMPONENT].chunks) == (1, 1, 1, 2, 2, 2)
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert metadata["data_pyramid_levels"] == [
        SOURCE_CACHE_COMPONENT,
        source_cache_component(level_index=1),
    ]
    assert tuple(root[source_cache_component(level_index=1)].shape) == (
        1,
        1,
        1,
        1,
        2,
        2,
    )
    assert not (experiment_path.parent / "data_store.ome.zarr").exists()


def test_has_canonical_data_component_detects_ready_store(tmp_path: Path):
    store_path = tmp_path / "ready_store.n5"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        SOURCE_CACHE_COMPONENT,
        shape=(1, 2, 3, 4, 5, 6),
        chunks=(1, 1, 1, 2, 3, 3),
        dtype="uint16",
        overwrite=True,
    )
    root[SOURCE_CACHE_COMPONENT].attrs["axes"] = ["t", "p", "c", "z", "y", "x"]

    assert has_canonical_data_component(store_path) is True


def test_has_canonical_data_component_rejects_noncanonical_store(tmp_path: Path):
    store_path = tmp_path / "raw_source.n5"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        "data",
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        dtype="uint16",
        overwrite=True,
    )
    root["data"].attrs["axes"] = ["z", "y", "x"]

    assert has_canonical_data_component(store_path) is False


def test_has_complete_canonical_data_store_rejects_missing_expected_pyramid(
    tmp_path: Path,
):
    store_path = tmp_path / "incomplete_store.n5"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        "data",
        shape=(1, 1, 1, 2, 4, 4),
        chunks=(1, 1, 1, 1, 2, 2),
        dtype="uint16",
        overwrite=True,
    )
    root["data"].attrs["axes"] = ["t", "p", "c", "z", "y", "x"]

    assert (
        has_complete_canonical_data_store(
            store_path,
            expected_chunks_tpczyx=(1, 1, 1, 1, 2, 2),
            expected_pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
        )
        is False
    )


def test_has_complete_canonical_data_store_accepts_stale_progress_when_public_ome_is_valid(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 1, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    assert (
        has_complete_canonical_data_store(
            materialized.store_path,
            expected_chunks_tpczyx=(1, 1, 1, 1, 2, 2),
            expected_pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        )
        is True
    )

    root = zarr.open_group(str(materialized.store_path), mode="a")
    progress = dict(root.attrs["ingestion_progress"])
    progress["status"] = "in_progress"
    progress["base_progress"] = {
        "total_regions": int(progress.get("base_progress", {}).get("total_regions", 1)),
        "completed_regions": 0,
    }
    root.attrs["ingestion_progress"] = progress

    assert (
        has_complete_canonical_data_store(
            materialized.store_path,
            expected_chunks_tpczyx=(1, 1, 1, 1, 2, 2),
            expected_pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        )
        is True
    )


def test_has_complete_canonical_data_store_rejects_stale_progress_when_public_ome_is_invalid(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 1, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="a")
    progress = dict(root.attrs["ingestion_progress"])
    progress["status"] = "in_progress"
    progress["base_progress"] = {
        "total_regions": int(progress.get("base_progress", {}).get("total_regions", 1)),
        "completed_regions": 0,
    }
    root.attrs["ingestion_progress"] = progress
    # Break OME-root validation to ensure stale/incomplete progress is rejected.
    root.attrs["ome"] = {"version": "0.5"}

    assert (
        has_complete_canonical_data_store(
            materialized.store_path,
            expected_chunks_tpczyx=(1, 1, 1, 1, 2, 2),
            expected_pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        )
        is False
    )


def test_materialize_experiment_data_store_reuses_complete_store_by_default_and_force_rebuilds(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="TIFF"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source_data)

    initial = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 1, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    reused = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 3, 4),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    root = zarr.open_group(str(initial.store_path), mode="r")
    assert reused.store_path == initial.store_path
    assert tuple(root[SOURCE_CACHE_COMPONENT].chunks) == (1, 1, 1, 1, 2, 2)
    assert root[SOURCE_CACHE_COMPONENT].attrs["pyramid_levels"] == [
        SOURCE_CACHE_COMPONENT
    ]
    if SOURCE_CACHE_PYRAMID_ROOT in root:
        assert list(root[SOURCE_CACHE_PYRAMID_ROOT].array_keys()) == []

    rebuilt = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 3, 4),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
        force_rebuild=True,
    )

    rebuilt_root = zarr.open_group(str(rebuilt.store_path), mode="r")
    assert tuple(rebuilt_root[SOURCE_CACHE_COMPONENT].chunks) == (1, 1, 1, 2, 3, 4)
    assert rebuilt_root[SOURCE_CACHE_COMPONENT].attrs["pyramid_levels"] == [
        SOURCE_CACHE_COMPONENT,
        source_cache_component(level_index=1),
    ]


def test_materialize_experiment_data_store_handles_same_component_rewrite(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="OME-ZARR"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_store = tmp_path / "source_data.zarr"
    source_root = zarr.open_group(str(source_store), mode="w")
    source_root.create_dataset(
        "data", data=source_data, chunks=(1, 3, 4), overwrite=True
    )
    source_root["data"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_store,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    expected_store = (experiment_path.parent / "data_store.ome.zarr").resolve()
    assert materialized.store_path == expected_store
    root = zarr.open_group(str(expected_store), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 1, 1, 2, 3, 4)
    assert tuple(root[SOURCE_CACHE_COMPONENT].chunks) == (1, 1, 1, 2, 2, 2)
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert metadata["data_pyramid_levels"] == [
        SOURCE_CACHE_COMPONENT,
        source_cache_component(level_index=1),
    ]
    assert tuple(root[source_cache_component(level_index=1)].shape) == (
        1,
        1,
        1,
        1,
        2,
        2,
    )


def test_materialize_experiment_data_store_stacks_tiff_positions_and_channels(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="TIFF",
        is_multiposition=True,
    )
    experiment = load_navigate_experiment(experiment_path)

    expected_blocks: dict[tuple[int, int], np.ndarray] = {}
    for position_index in range(3):
        position_dir = tmp_path / f"Position{position_index}"
        position_dir.mkdir(parents=True, exist_ok=True)
        for channel_index in range(2):
            block = np.full(
                (2, 3, 4),
                fill_value=(position_index * 10 + channel_index),
                dtype=np.uint16,
            )
            expected_blocks[(position_index, channel_index)] = block
            path = position_dir / f"CH{channel_index:02d}_000000.tiff"
            tifffile.imwrite(str(path), block, metadata={"axes": "ZYX"})

    resolved = resolve_experiment_data_path(experiment)
    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=resolved,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 3, 2, 2, 3, 4)
    assert metadata["source_data_path"] == str(tmp_path.resolve())

    for position_index in range(3):
        for channel_index in range(2):
            loaded = np.array(
                root[SOURCE_CACHE_COMPONENT][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(
                loaded,
                expected_blocks[(position_index, channel_index)],
            )


def test_materialize_experiment_data_store_stacks_bdv_h5_setups(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="H5",
        is_multiposition=True,
    )
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=2)
    experiment = load_navigate_experiment(experiment_path)

    source_path = tmp_path / "CH00_000000.h5"
    expected_blocks = {
        (0, 0): np.full((2, 3, 4), fill_value=10, dtype=np.uint16),
        (1, 0): np.full((2, 3, 4), fill_value=20, dtype=np.uint16),
        (0, 1): np.full((2, 3, 4), fill_value=30, dtype=np.uint16),
        (1, 1): np.full((2, 3, 4), fill_value=40, dtype=np.uint16),
    }
    with h5py.File(str(source_path), mode="w") as handle:
        handle.create_dataset(
            "t00000/s00/0/cells",
            data=expected_blocks[(0, 0)],
            chunks=(1, 3, 4),
        )
        handle.create_dataset(
            "t00000/s01/0/cells",
            data=expected_blocks[(1, 0)],
            chunks=(1, 3, 4),
        )
        handle.create_dataset(
            "t00000/s02/0/cells",
            data=expected_blocks[(0, 1)],
            chunks=(1, 3, 4),
        )
        handle.create_dataset(
            "t00000/s03/0/cells",
            data=expected_blocks[(1, 1)],
            chunks=(1, 3, 4),
        )
        # Ensure extra setup groups not listed in XML are ignored.
        handle.create_dataset(
            "t00000/s99/0/cells",
            data=np.zeros((2, 3, 4), dtype=np.uint16),
            chunks=(1, 3, 4),
        )

    _write_bdv_xml(
        tmp_path / "CH00_000000.xml",
        loader_format="bdv.hdf5",
        data_file_name=source_path.name,
        setup_channel_tile={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    )

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 2, 2, 2, 3, 4)
    assert metadata["source_data_path"] == str(source_path.resolve())

    for position_index in range(2):
        for channel_index in range(2):
            loaded = np.array(
                root[SOURCE_CACHE_COMPONENT][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(
                loaded, expected_blocks[(position_index, channel_index)]
            )


def test_materialize_experiment_data_store_stacks_bdv_n5_setups(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="N5",
        is_multiposition=True,
    )
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=2)
    experiment = load_navigate_experiment(experiment_path)

    source_path = tmp_path / "CH00_000000.n5"
    expected_blocks = {
        (0, 0): np.full((2, 3, 4), fill_value=11, dtype=np.uint16),
        (1, 0): np.full((2, 3, 4), fill_value=21, dtype=np.uint16),
        (0, 1): np.full((2, 3, 4), fill_value=31, dtype=np.uint16),
        (1, 1): np.full((2, 3, 4), fill_value=41, dtype=np.uint16),
    }
    for setup_index, block in {
        0: expected_blocks[(0, 0)],
        1: expected_blocks[(1, 0)],
        2: expected_blocks[(0, 1)],
        3: expected_blocks[(1, 1)],
        99: np.zeros((2, 3, 4), dtype=np.uint16),
    }.items():
        _write_real_n5_dataset(
            source_path,
            component=f"setup{setup_index}/timepoint0/s0",
            data_xyz=np.transpose(block, (2, 1, 0)),
            block_size_xyz=(4, 3, 1),
        )
    _write_legacy_n5_group(source_path, component="data")
    _write_legacy_n5_group(source_path, component="data_pyramid")
    _write_legacy_n5_group(source_path, component="results")
    _write_legacy_n5_group(source_path, component="provenance")

    _write_bdv_xml(
        tmp_path / "CH00_000000.xml",
        loader_format="bdv.n5",
        data_file_name=source_path.name,
        setup_channel_tile={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    )

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 2, 2, 2, 3, 4)
    assert root["clearex/metadata"].attrs["source_data_path"] == str(
        source_path.resolve()
    )
    assert materialized.source_image_info.shape == (1, 2, 2, 4, 3, 2)
    assert materialized.source_image_info.axes == "TPCXYZ"
    assert root["A/1/0/0"].shape == (1, 2, 2, 3, 4)

    for position_index in range(2):
        for channel_index in range(2):
            loaded = np.array(
                root[SOURCE_CACHE_COMPONENT][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(
                loaded, expected_blocks[(position_index, channel_index)]
            )


def test_materialize_experiment_data_store_skips_n5_setups_without_persisted_chunks(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="N5",
        is_multiposition=True,
    )
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=2)
    experiment = load_navigate_experiment(experiment_path)

    source_path = tmp_path / "CH00_000000.n5"
    expected_blocks = {
        (0, 0): np.full((2, 3, 4), fill_value=13, dtype=np.uint16),
        (1, 0): np.full((2, 3, 4), fill_value=23, dtype=np.uint16),
    }
    for setup_index, block in {
        0: expected_blocks[(0, 0)],
        1: expected_blocks[(1, 0)],
        2: np.zeros((2, 3, 4), dtype=np.uint16),
    }.items():
        _write_real_n5_dataset(
            source_path,
            component=f"setup{setup_index}/timepoint0/s0",
            data_xyz=np.transpose(block, (2, 1, 0)),
            block_size_xyz=(4, 3, 1),
        )
    _strip_n5_component_payload_files(source_path, component="setup2/timepoint0/s0")

    # Do not provide XML here: fallback setup indexing should still ignore
    # placeholder setups without persisted chunks.
    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 2, 1, 2, 3, 4)

    for position_index in range(2):
        loaded = np.array(root[SOURCE_CACHE_COMPONENT][0, position_index, 0, :, :, :])
        assert np.array_equal(loaded, expected_blocks[(position_index, 0)])


def test_load_navigate_experiment_source_image_info_summarizes_bdv_n5(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="N5",
        is_multiposition=True,
    )
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=2)
    experiment = load_navigate_experiment(experiment_path)

    source_path = tmp_path / "CH00_000000.n5"
    for setup_index, fill_value in enumerate((11, 21, 31, 41)):
        canonical_block = np.full((2, 3, 4), fill_value=fill_value, dtype=np.uint16)
        _write_real_n5_dataset(
            source_path,
            component=f"setup{setup_index}/timepoint0/s0",
            data_xyz=np.transpose(canonical_block, (2, 1, 0)),
            block_size_xyz=(4, 3, 1),
        )

    _write_bdv_xml(
        tmp_path / "CH00_000000.xml",
        loader_format="bdv.n5",
        data_file_name=source_path.name,
        setup_channel_tile={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    )

    class _FailingOpener:
        def open(self, *args, **kwargs):
            raise AssertionError("ImageOpener.open should not be used for BDV N5.")

    info = load_navigate_experiment_source_image_info(
        experiment=experiment,
        source_path=source_path,
        opener=_FailingOpener(),
    )

    assert info.path == source_path.resolve()
    assert info.shape == (1, 2, 2, 2, 3, 4)
    assert info.dtype == np.dtype(np.uint16)
    assert info.axes == "TPCZYX"
    assert info.metadata is not None
    assert info.metadata["source_layout"] == "navigate_bdv_n5"
    assert info.metadata["source_component"] == "setup0/timepoint0/s0"
    assert info.metadata["positions"] == 2
    assert info.metadata["channels"] == 2


def test_load_navigate_experiment_source_image_info_reads_tiff_header_only(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="TIFF",
        is_multiposition=False,
    )
    experiment = load_navigate_experiment(experiment_path)

    source_path = tmp_path / "CH00_000000.tiff"
    expected = np.arange(2 * 3 * 4, dtype=np.uint16).reshape((2, 3, 4))
    tifffile.imwrite(
        str(source_path),
        expected,
        metadata={"axes": "ZYX", "spacing": 0.2, "unit": "um"},
    )

    class _FailingOpener:
        def open(self, *args, **kwargs):
            raise AssertionError("ImageOpener.open should not be used for TIFF.")

    info = load_navigate_experiment_source_image_info(
        experiment=experiment,
        source_path=source_path,
        opener=_FailingOpener(),
    )

    assert info.path == source_path.resolve()
    assert info.shape == expected.shape
    assert info.dtype == np.dtype(np.uint16)
    assert info.axes == "ZYX"
    assert info.metadata is not None
    assert int(info.metadata["tiff_page_count"]) >= 1
    assert "ImageDescription" in info.metadata
    assert info.metadata["axes"] == "ZYX"
    assert info.metadata["spacing"] == 0.2
    assert info.metadata["unit"] == "um"


def test_open_source_as_dask_rejects_standalone_n5_source(tmp_path: Path) -> None:
    source_path = tmp_path / "standalone.n5"
    _write_real_n5_dataset(
        source_path,
        component="setup0/timepoint0/s0",
        data_xyz=np.zeros((4, 3, 2), dtype=np.uint16),
        block_size_xyz=(4, 3, 1),
    )

    with ExitStack() as exit_stack, pytest.raises(ValueError, match="Standalone N5"):
        experiment_module._open_source_as_dask(
            source_path,
            exit_stack=exit_stack,
        )


def test_open_source_as_dask_tiff_falls_back_when_store_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.tiff"
    expected = np.arange(2 * 3 * 4, dtype=np.uint16).reshape((2, 3, 4))
    tifffile.imwrite(str(source_path), expected, metadata={"axes": "ZYX"})

    def _raise_unsupported_store(*_args, **_kwargs):
        raise TypeError("Unsupported type for store_like: 'ZarrTiffStore'")

    monkeypatch.setattr(read_module.da, "from_zarr", _raise_unsupported_store)

    with ExitStack() as exit_stack:
        source_array, source_axes, source_meta = experiment_module._open_source_as_dask(
            source_path,
            exit_stack=exit_stack,
        )

    assert isinstance(source_array, da.Array)
    assert tuple(source_array.shape) == expected.shape
    assert np.array_equal(source_array.compute(), expected)
    assert tuple(source_axes or ()) == ("z", "y", "x")
    assert source_meta["source_path"] == str(source_path)


def test_materialize_experiment_data_store_stacks_bdv_ome_zarr_setups(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="OME-ZARR",
        is_multiposition=True,
    )
    _write_multipositions_sidecar(tmp_path / "multi_positions.yml", count=2)
    experiment = load_navigate_experiment(experiment_path)

    source_path = tmp_path / "CH00_000000.ome.zarr"
    source_root = zarr.open_group(str(source_path), mode="w")
    expected_blocks = {
        (0, 0): np.full((2, 3, 4), fill_value=12, dtype=np.uint16),
        (1, 0): np.full((2, 3, 4), fill_value=22, dtype=np.uint16),
        (0, 1): np.full((2, 3, 4), fill_value=32, dtype=np.uint16),
        (1, 1): np.full((2, 3, 4), fill_value=42, dtype=np.uint16),
    }
    source_root.create_dataset(
        "setup0/timepoint0/s0",
        data=expected_blocks[(0, 0)],
        chunks=(1, 3, 4),
        overwrite=True,
    )
    source_root.create_dataset(
        "setup1/timepoint0/s0",
        data=expected_blocks[(1, 0)],
        chunks=(1, 3, 4),
        overwrite=True,
    )
    source_root.create_dataset(
        "setup2/timepoint0/s0",
        data=expected_blocks[(0, 1)],
        chunks=(1, 3, 4),
        overwrite=True,
    )
    source_root.create_dataset(
        "setup3/timepoint0/s0",
        data=expected_blocks[(1, 1)],
        chunks=(1, 3, 4),
        overwrite=True,
    )
    # Ensure extra setup groups not listed in XML are ignored.
    source_root.create_dataset(
        "setup99/timepoint0/s0",
        data=np.zeros((2, 3, 4), dtype=np.uint16),
        chunks=(1, 3, 4),
        overwrite=True,
    )

    # Navigate can write XML as CH00_000000.xml for OME-Zarr stores.
    _write_bdv_xml(
        tmp_path / "CH00_000000.xml",
        loader_format="bdv.ome.zarr",
        data_file_name=source_path.name,
        setup_channel_tile={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    )

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert tuple(root[SOURCE_CACHE_COMPONENT].shape) == (1, 2, 2, 2, 3, 4)
    assert metadata["source_data_path"] == str(source_path.resolve())

    for position_index in range(2):
        for channel_index in range(2):
            loaded = np.array(
                root[SOURCE_CACHE_COMPONENT][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(
                loaded, expected_blocks[(position_index, channel_index)]
            )


def test_should_use_source_aligned_plane_writes_detects_plane_chunk_pattern():
    source = da.from_array(
        np.zeros((1, 1, 1, 6, 8, 10), dtype=np.uint16),
        chunks=(1, 1, 1, 1, 8, 10),
    )

    assert (
        experiment_module._should_use_source_aligned_plane_writes(
            array=source,
            shape_tpczyx=(1, 1, 1, 6, 8, 10),
            target_chunks_tpczyx=(1, 1, 1, 2, 2, 2),
        )
        is True
    )
    assert (
        experiment_module._should_use_source_aligned_plane_writes(
            array=source,
            shape_tpczyx=(1, 1, 1, 6, 8, 10),
            target_chunks_tpczyx=(1, 1, 1, 2, 8, 10),
        )
        is False
    )


def test_materialize_experiment_data_store_uses_source_aligned_plane_writes(
    tmp_path: Path, monkeypatch
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="OME-ZARR",
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(4 * 8 * 10, dtype=np.uint16).reshape(4, 8, 10)
    source_store = tmp_path / "source.ome.zarr"
    source_root = zarr.open_group(str(source_store), mode="w")
    source_root.create_dataset(
        "raw", data=source_data, chunks=(1, 8, 10), overwrite=True
    )
    source_root["raw"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    original_source_writer = (
        experiment_module._write_dask_array_source_aligned_plane_batches
    )
    original_chunk_writer = experiment_module._write_dask_array_in_batches
    writer_calls = {"source_aligned": 0, "chunk_batched": 0}

    def _count_source_writer(**kwargs):
        writer_calls["source_aligned"] += 1
        return original_source_writer(**kwargs)

    def _count_chunk_writer(**kwargs):
        writer_calls["chunk_batched"] += 1
        return original_chunk_writer(**kwargs)

    monkeypatch.setattr(
        experiment_module,
        "_write_dask_array_source_aligned_plane_batches",
        _count_source_writer,
    )
    monkeypatch.setattr(
        experiment_module,
        "_write_dask_array_in_batches",
        _count_chunk_writer,
    )

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_store,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert writer_calls["source_aligned"] == 1
    assert writer_calls["chunk_batched"] == 0
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert metadata["materialization_write_strategy"] == "source_aligned_plane_batches"
    assert metadata["source_aligned_z_batch_depth"] == 2
    assert metadata["source_aligned_worker_count"] is None
    assert metadata["source_aligned_worker_memory_limit_bytes"] is None


def test_materialize_experiment_data_store_falls_back_to_chunk_batched_writes(
    tmp_path: Path, monkeypatch
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="OME-ZARR",
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(4 * 8 * 10, dtype=np.uint16).reshape(4, 8, 10)
    source_store = tmp_path / "source.ome.zarr"
    source_root = zarr.open_group(str(source_store), mode="w")
    source_root.create_dataset(
        "raw", data=source_data, chunks=(2, 8, 10), overwrite=True
    )
    source_root["raw"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    original_source_writer = (
        experiment_module._write_dask_array_source_aligned_plane_batches
    )
    original_chunk_writer = experiment_module._write_dask_array_in_batches
    writer_calls = {"source_aligned": 0, "chunk_batched": 0}

    def _count_source_writer(**kwargs):
        writer_calls["source_aligned"] += 1
        return original_source_writer(**kwargs)

    def _count_chunk_writer(**kwargs):
        writer_calls["chunk_batched"] += 1
        return original_chunk_writer(**kwargs)

    monkeypatch.setattr(
        experiment_module,
        "_write_dask_array_source_aligned_plane_batches",
        _count_source_writer,
    )
    monkeypatch.setattr(
        experiment_module,
        "_write_dask_array_in_batches",
        _count_chunk_writer,
    )

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_store,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    metadata = root["clearex/metadata"].attrs
    assert writer_calls["source_aligned"] == 0
    assert writer_calls["chunk_batched"] == 1
    assert np.array_equal(
        np.array(root[SOURCE_CACHE_COMPONENT][0, 0, 0, :, :, :]), source_data
    )
    assert metadata["materialization_write_strategy"] == "chunk_region_batches"
    assert metadata["source_aligned_z_batch_depth"] is None
    assert metadata["source_aligned_worker_count"] is None
    assert metadata["source_aligned_worker_memory_limit_bytes"] is None


def test_detect_client_worker_resources_extracts_min_limit():
    class _FakeClient:
        def scheduler_info(self):
            return {
                "workers": {
                    "tcp://127.0.0.1:1111": {"memory_limit": 12 << 30},
                    "tcp://127.0.0.1:2222": {"memory_limit": 8 << 30},
                }
            }

    worker_count, worker_memory_limit = (
        experiment_module._detect_client_worker_resources(_FakeClient())
    )

    assert worker_count == 2
    assert worker_memory_limit == (8 << 30)


def test_estimate_source_plane_batch_depth_respects_worker_memory_limit():
    depth = experiment_module._estimate_source_plane_batch_depth(
        shape_tpczyx=(1, 1, 1, 4200, 3840, 5112),
        target_chunks_tpczyx=(1, 1, 1, 256, 256, 256),
        dtype_itemsize=2,
        worker_memory_limit_bytes=7 << 30,
    )

    assert depth < 256
    assert depth <= 64


def test_source_aligned_writes_require_serial_submission_when_chunk_misaligned() -> (
    None
):
    assert (
        experiment_module._source_aligned_writes_require_serial_submission(
            z_batch_depth=128,
            target_chunks_tpczyx=(1, 1, 1, 256, 256, 256),
        )
        is True
    )
    assert (
        experiment_module._source_aligned_writes_require_serial_submission(
            z_batch_depth=256,
            target_chunks_tpczyx=(1, 1, 1, 256, 256, 256),
        )
        is False
    )


def test_write_source_aligned_batches_serializes_when_chunk_misaligned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "source_aligned_serialized.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        SOURCE_CACHE_COMPONENT,
        shape=(1, 1, 1, 6, 4, 4),
        chunks=(1, 1, 1, 4, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    source_data = np.arange(1 * 1 * 1 * 6 * 4 * 4, dtype=np.uint16).reshape(
        (1, 1, 1, 6, 4, 4)
    )
    source = da.from_array(source_data, chunks=(1, 1, 1, 1, 4, 4))
    compute_batch_sizes: list[int] = []
    original_compute = experiment_module._compute_dask_graph

    monkeypatch.setattr(
        experiment_module,
        "_estimate_source_aligned_submission_batch_count",
        lambda **kwargs: 8,
    )

    def _capture_compute(graph, *, client=None):
        compute_batch_sizes.append(len(list(graph)))
        return original_compute(graph, client=client)

    monkeypatch.setattr(experiment_module, "_compute_dask_graph", _capture_compute)

    experiment_module._write_dask_array_source_aligned_plane_batches(
        array=source,
        store_path=store_path,
        component=SOURCE_CACHE_COMPONENT,
        shape_tpczyx=(1, 1, 1, 6, 4, 4),
        z_batch_depth=2,
        dtype_itemsize=int(np.dtype(source_data.dtype).itemsize),
        target_chunks_tpczyx=(1, 1, 1, 4, 2, 2),
    )

    reloaded = np.asarray(
        zarr.open_group(str(store_path), mode="r")[SOURCE_CACHE_COMPONENT]
    )
    assert np.array_equal(reloaded, source_data)
    assert compute_batch_sizes == [1, 1, 1]
