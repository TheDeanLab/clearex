#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
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
from pathlib import Path
import json
import subprocess

# Third Party Imports
import dask.array as da
import h5py
import numpy as np
import pytest
import tifffile
import zarr

# Local Imports
import clearex.io.experiment as experiment_module
from clearex.io.experiment import (
    default_analysis_store_path,
    find_experiment_data_candidates,
    has_canonical_data_component,
    has_complete_canonical_data_store,
    initialize_analysis_store,
    is_navigate_experiment_file,
    load_navigate_experiment,
    load_store_spatial_calibration,
    materialize_experiment_data_store,
    migrate_analysis_store,
    resolve_data_store_path,
    resolve_experiment_data_path,
    save_store_spatial_calibration,
    write_zyx_block,
)
from clearex.io.read import ImageInfo
from clearex.io.zarr_storage import (
    create_or_overwrite_array,
    open_group as open_zarr_group,
    resolve_external_analysis_store_path,
    resolve_staging_store_path,
)
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


def _wrap_test_zarr_group(group):
    class _CompatZarrGroup:
        def __init__(self, inner_group):
            self._inner_group = inner_group

        def __getattr__(self, name):
            return getattr(self._inner_group, name)

        def __contains__(self, key):
            return key in self._inner_group

        def __delitem__(self, key):
            del self._inner_group[key]

        def __getitem__(self, key):
            item = self._inner_group[key]
            if hasattr(item, "array_keys") and hasattr(item, "group_keys"):
                return _wrap_test_zarr_group(item)
            return item

        def create_dataset(self, name, **kwargs):
            return create_or_overwrite_array(root=self._inner_group, name=name, **kwargs)

        def create_group(self, name, **kwargs):
            return _wrap_test_zarr_group(self._inner_group.create_group(name, **kwargs))

        def require_group(self, name, **kwargs):
            return _wrap_test_zarr_group(
                self._inner_group.require_group(name, **kwargs)
            )

    return _CompatZarrGroup(group)


def _open_test_zarr_group(
    path: Path | str,
    *,
    mode: str = "a",
    zarr_format: int | None = None,
):
    if zarr_format is None and mode in {"w", "w-"}:
        zarr_format = 2
    return _wrap_test_zarr_group(
        open_zarr_group(path, mode=mode, zarr_format=zarr_format)
    )


def _write_real_n5_store(path: Path, entries: dict[str, np.ndarray]) -> None:
    command_prefix = experiment_module._legacy_n5_helper_command_prefix()
    if command_prefix is None:
        pytest.skip("No zarr2-compatible Python with N5Store is available.")

    payload = {name: np.asarray(array).tolist() for name, array in entries.items()}
    command = [
        *command_prefix,
        "-c",
        (
            "from pathlib import Path; "
            "import json, numpy as np, sys, zarr; "
            "target = Path(sys.argv[1]); "
            "entries = json.loads(sys.argv[2]); "
            "root = zarr.group(store=zarr.N5Store(str(target)), overwrite=True); "
            "[root.create_dataset(name, data=np.asarray(values, dtype=np.uint16), "
            "chunks=(1, 3, 4), overwrite=True) for name, values in entries.items()]"
        ),
        str(path),
        json.dumps(payload),
    ]
    subprocess.run(command, check=True)


def test_legacy_n5_helper_command_prefix_prefers_direct_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_module,
        "_legacy_n5_helper_python",
        lambda: "/tmp/legacy-python",
    )

    prefix = experiment_module._legacy_n5_helper_command_prefix()

    assert prefix == ("/tmp/legacy-python",)


def test_legacy_n5_helper_command_prefix_falls_back_to_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_module, "_legacy_n5_helper_python", lambda: None)
    monkeypatch.setattr(experiment_module.shutil, "which", lambda name: "/usr/bin/uv")

    class _ProbeResult:
        returncode = 0

    commands: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs: object) -> _ProbeResult:
        del kwargs
        commands.append(command)
        return _ProbeResult()

    monkeypatch.setattr(experiment_module.subprocess, "run", _fake_run)

    prefix = experiment_module._legacy_n5_helper_command_prefix()

    assert prefix == ("/usr/bin/uv", "run", "--with", "zarr<3", "python")
    assert commands == [
        [
            "/usr/bin/uv",
            "run",
            "--with",
            "zarr<3",
            "python",
            "-c",
            "import zarr,sys; sys.exit(0 if hasattr(zarr, 'N5Store') else 1)",
        ]
    ]


def test_extract_client_scheduler_address_prefers_scheduler_attr() -> None:
    class _FakeScheduler:
        address = "tcp://127.0.0.1:8786"

    class _FakeClient:
        scheduler = _FakeScheduler()

        def scheduler_info(self):  # pragma: no cover - should not be called
            raise AssertionError("scheduler_info should not be queried")

    assert (
        experiment_module._extract_client_scheduler_address(_FakeClient())
        == "tcp://127.0.0.1:8786"
    )


def test_extract_client_scheduler_address_falls_back_to_scheduler_info() -> None:
    class _FakeClient:
        scheduler = None

        def scheduler_info(self):
            return {"address": "tcp://10.0.0.2:8786"}

    assert (
        experiment_module._extract_client_scheduler_address(_FakeClient())
        == "tcp://10.0.0.2:8786"
    )


def test_materialize_n5_via_legacy_helper_forwards_scheduler_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="N5")
    experiment = load_navigate_experiment(experiment_path)
    source_path = tmp_path / "CH00_000000.n5"
    source_path.mkdir()
    output_store = tmp_path / "CH00_000000.n5.clearex.zarr"

    monkeypatch.setattr(
        experiment_module,
        "_legacy_n5_helper_command_prefix",
        lambda: ("/usr/bin/python3",),
    )

    class _Scheduler:
        address = "tcp://127.0.0.1:8786"

    class _Client:
        scheduler = _Scheduler()

    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return None

    monkeypatch.setattr(experiment_module.subprocess, "run", _fake_run)

    returned = experiment_module._materialize_n5_via_legacy_helper(
        experiment=experiment,
        source_path=source_path,
        output_store_path=output_store,
        chunks=(1, 1, 1, 64, 64, 64),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        client=_Client(),
    )

    command = captured["command"]
    assert "--scheduler-address" in command
    flag_index = command.index("--scheduler-address")
    assert command[flag_index + 1] == "tcp://127.0.0.1:8786"
    assert returned == output_store.with_name(f"{output_store.name}.legacy-v2.zarr")


def test_materialize_n5_via_legacy_helper_omits_scheduler_address_without_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="N5")
    experiment = load_navigate_experiment(experiment_path)
    source_path = tmp_path / "CH00_000000.n5"
    source_path.mkdir()
    output_store = tmp_path / "CH00_000000.n5.clearex.zarr"

    monkeypatch.setattr(
        experiment_module,
        "_legacy_n5_helper_command_prefix",
        lambda: ("/usr/bin/python3",),
    )

    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return None

    monkeypatch.setattr(experiment_module.subprocess, "run", _fake_run)

    _ = experiment_module._materialize_n5_via_legacy_helper(
        experiment=experiment,
        source_path=source_path,
        output_store_path=output_store,
        chunks=(1, 1, 1, 64, 64, 64),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        client=None,
    )

    command = captured["command"]
    assert "--scheduler-address" not in command


def test_materialize_experiment_data_store_passes_client_to_legacy_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="N5")
    experiment = load_navigate_experiment(experiment_path)
    source_path = tmp_path / "CH00_000000.n5"
    source_path.mkdir()
    final_store = source_path.with_name(f"{source_path.name}.clearex.zarr")
    root = _open_test_zarr_group(final_store, mode="w", zarr_format=3)
    root.create_dataset(
        name="data",
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    class _Client:
        pass

    client = _Client()
    captured: dict[str, object] = {}

    def _fake_legacy_helper(**kwargs):
        captured["legacy_client"] = kwargs.get("client")
        legacy_store = tmp_path / "legacy-output.zarr"
        legacy_store.mkdir(exist_ok=True)
        return legacy_store

    monkeypatch.setattr(experiment_module, "is_clearex_analysis_store", lambda _path: False)
    monkeypatch.setattr(
        experiment_module,
        "_materialize_n5_via_legacy_helper",
        _fake_legacy_helper,
    )
    monkeypatch.setattr(
        experiment_module,
        "migrate_analysis_store",
        lambda _path, keep_backup=False: tmp_path / "legacy-migrated.zarr",
    )
    monkeypatch.setattr(
        experiment_module,
        "replace_store_path",
        lambda **kwargs: None,
    )

    result = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
        client=client,
    )

    assert captured["legacy_client"] is client
    assert result.store_path == final_store.resolve()


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
        setup_blocks.append(
            f"""
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
""".rstrip()
        )

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


def test_build_library_path_environment_updates_merges_discovered_and_inherited() -> None:
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
    for extra_env_var in experiment_module._library_path_env_vars_for_platform()[1:]:
        monkeypatch.delenv(extra_env_var, raising=False)
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
    assert tuple(root["data"].shape) == (2, 3, 2, 4, 8, 16)
    assert list(root["data"].attrs["axes"]) == ["t", "p", "c", "z", "y", "x"]
    assert root.attrs["storage_policy_analysis_outputs"] == "latest_only"
    assert list(root["data"].attrs["chunk_shape_tpczyx"]) == [1, 1, 1, 4, 8, 16]
    assert list(root.attrs["configured_chunks_tpczyx"]) == [1, 1, 1, 256, 256, 256]


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
    assert tuple(root["data"].chunks) == (1, 1, 1, 8, 32, 32)
    assert list(root["data"].attrs["chunk_shape_tpczyx"]) == [1, 1, 1, 8, 32, 32]
    assert list(root.attrs["configured_chunks_tpczyx"]) == [1, 1, 1, 8, 32, 32]
    assert root.attrs["resolution_pyramid_factors_tpczyx"] == [
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

    initialize_analysis_store(experiment=experiment, zarr_path=store_path, overwrite=True)

    calibration = load_store_spatial_calibration(store_path)

    assert calibration == SpatialCalibrationConfig()


def test_load_store_spatial_calibration_defaults_to_identity_for_legacy_store(
    tmp_path: Path,
):
    store_path = tmp_path / "legacy_store.zarr"
    root = _open_test_zarr_group(store_path, mode="w")
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
    root = _open_test_zarr_group(store_path, mode="w")
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
    initialize_analysis_store(experiment=experiment, zarr_path=store_path, overwrite=True)

    block = np.ones((4, 8, 16), dtype=np.uint16)
    write_zyx_block(
        zarr_path=store_path,
        block=block,
        t_index=1,
        p_index=2,
        c_index=1,
    )

    root = zarr.open_group(str(store_path), mode="r")
    loaded = np.array(root["data"][1, 2, 1, :, :, :])
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
    _write_minimal_experiment(experiment_path, save_directory=save_directory, file_type="H5")
    experiment = load_navigate_experiment(experiment_path)

    source_path = save_directory / "source.npy"
    np.save(source_path, np.zeros((2, 3, 4), dtype=np.uint16))

    resolved = resolve_data_store_path(experiment, source_path)

    assert resolved == (experiment_dir / "data_store.zarr").resolve()


def test_materialize_experiment_data_store_creates_data_store_for_non_zarr(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="TIFF")
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

    expected_store = (experiment_path.parent / "data_store.zarr").resolve()
    assert materialized.store_path == expected_store
    root = zarr.open_group(str(expected_store), mode="r")
    assert tuple(root["data"].shape) == (1, 1, 1, 2, 3, 4)
    assert tuple(root["data"].chunks) == (1, 1, 1, 2, 2, 2)
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert root.attrs["data_pyramid_levels"] == ["data", "data_pyramid/level_1"]
    assert tuple(root["data_pyramid/level_1"].shape) == (1, 1, 1, 1, 2, 2)
    assert np.array_equal(
        np.array(root["data_pyramid/level_1"][0, 0, 0, :, :, :]),
        source_data[::2, ::2, ::2],
    )


def test_resolve_data_store_path_uses_sibling_store_for_external_zarr(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path,
        save_directory=tmp_path,
        file_type="OME-ZARR",
    )
    experiment = load_navigate_experiment(experiment_path)
    source_store = tmp_path / "input.ome.zarr"
    _open_test_zarr_group(source_store, mode="w")

    resolved = resolve_data_store_path(experiment, source_store)

    assert resolved == resolve_external_analysis_store_path(source_store)


def test_migrate_analysis_store_converts_v2_store_in_place(tmp_path: Path):
    store_path = tmp_path / "analysis_store.zarr"
    root = _open_test_zarr_group(store_path, mode="w")
    root.attrs["schema"] = "clearex.analysis_store.v1"
    root.attrs["axes"] = ["t", "p", "c", "z", "y", "x"]
    root.create_dataset(
        "data",
        data=np.arange(24, dtype=np.uint16).reshape(1, 1, 1, 2, 3, 4),
        chunks=(1, 1, 1, 1, 3, 4),
        overwrite=True,
    )

    migrated = migrate_analysis_store(store_path)

    assert migrated == store_path.resolve()
    assert (store_path / "zarr.json").exists()
    reopened = zarr.open_group(str(store_path), mode="r")
    assert np.array_equal(
        np.asarray(reopened["data"]),
        np.arange(24, dtype=np.uint16).reshape(1, 1, 1, 2, 3, 4),
    )


def test_materialize_experiment_data_store_batches_chunk_writes(
    tmp_path: Path, monkeypatch
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="TIFF")
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

    expected_store = (experiment_path.parent / "data_store.zarr").resolve()
    root = zarr.open_group(str(expected_store), mode="r")
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert not resolve_staging_store_path(expected_store).exists()
    assert len(compute_calls) == 8


def test_materialize_experiment_data_store_resumes_after_interrupted_base_write(
    tmp_path: Path,
    monkeypatch,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="TIFF")
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

    staging_store = resolve_staging_store_path(experiment_path.parent / "data_store.zarr")
    assert not (experiment_path.parent / "data_store.zarr").exists()
    root = zarr.open_group(str(staging_store), mode="r")
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
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert not staging_store.exists()
    progress = dict(root.attrs["ingestion_progress"])
    assert progress["status"] == "completed"
    assert progress["base_progress"]["completed_regions"] == 8
    assert progress["base_progress"]["total_regions"] == 8


def test_materialize_experiment_data_store_handles_multibatch_base_and_pyramid(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="TIFF")
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
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert np.array_equal(
        np.array(root["data_pyramid/level_1"][0, 0, 0, :, :, :]),
        source_data[::2, ::2, ::2],
    )


def test_materialize_experiment_data_store_materializes_external_zarr_store_to_sibling_store(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="OME-ZARR"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_store = tmp_path / "source.ome.zarr"
    source_root = _open_test_zarr_group(source_store, mode="w")
    source_root.create_dataset("raw", data=source_data, chunks=(1, 3, 4), overwrite=True)
    source_root["raw"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_store,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    expected_store = resolve_external_analysis_store_path(source_store)
    assert materialized.store_path == expected_store
    root = zarr.open_group(str(materialized.store_path), mode="r")
    assert "data" in root
    assert tuple(root["data"].shape) == (1, 1, 1, 2, 3, 4)
    assert tuple(root["data"].chunks) == (1, 1, 1, 2, 2, 2)
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert root.attrs["data_pyramid_levels"] == ["data", "data_pyramid/level_1"]
    assert tuple(root["data_pyramid/level_1"].shape) == (1, 1, 1, 1, 2, 2)
    assert not (experiment_path.parent / "data_store.zarr").exists()
    source_root = zarr.open_group(str(source_store), mode="r")
    assert "raw" in source_root
    assert "data" not in source_root


def test_has_canonical_data_component_detects_ready_store(tmp_path: Path):
    store_path = tmp_path / "ready_store.n5"
    root = _open_test_zarr_group(store_path, mode="w")
    root.create_dataset(
        "data",
        shape=(1, 2, 3, 4, 5, 6),
        chunks=(1, 1, 1, 2, 3, 3),
        dtype="uint16",
        overwrite=True,
    )
    root["data"].attrs["axes"] = ["t", "p", "c", "z", "y", "x"]

    assert has_canonical_data_component(store_path) is True


def test_has_canonical_data_component_rejects_noncanonical_store(tmp_path: Path):
    store_path = tmp_path / "raw_source.n5"
    root = _open_test_zarr_group(store_path, mode="w")
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
    root = _open_test_zarr_group(store_path, mode="w")
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


def test_has_complete_canonical_data_store_requires_completed_progress_record(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="TIFF")
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
    root.attrs["ingestion_progress"] = progress

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
    _write_minimal_experiment(experiment_path, save_directory=tmp_path, file_type="TIFF")
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
    assert tuple(root["data"].chunks) == (1, 1, 1, 1, 2, 2)
    assert root.attrs["data_pyramid_levels"] == ["data"]
    if "data_pyramid" in root:
        assert list(root["data_pyramid"].array_keys()) == []

    rebuilt = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 3, 4),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
        force_rebuild=True,
    )

    rebuilt_root = zarr.open_group(str(rebuilt.store_path), mode="r")
    assert tuple(rebuilt_root["data"].chunks) == (1, 1, 1, 2, 3, 4)
    assert rebuilt_root.attrs["data_pyramid_levels"] == ["data", "data_pyramid/level_1"]


def test_materialize_experiment_data_store_keeps_external_source_store_immutable(
    tmp_path: Path,
):
    experiment_path = tmp_path / "experiment.yml"
    _write_minimal_experiment(
        experiment_path, save_directory=tmp_path, file_type="OME-ZARR"
    )
    experiment = load_navigate_experiment(experiment_path)

    source_data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    source_store = tmp_path / "source_data.zarr"
    source_root = _open_test_zarr_group(source_store, mode="w")
    source_root.create_dataset("data", data=source_data, chunks=(1, 3, 4), overwrite=True)
    source_root["data"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_store,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1, 2), (1, 2), (1, 2)),
    )

    root = zarr.open_group(str(source_store), mode="r")
    assert tuple(root["data"].shape) == (2, 3, 4)
    assert tuple(root["data"].chunks) == (1, 3, 4)
    assert np.array_equal(np.array(root["data"][:]), source_data)
    assert "data_pyramid" not in root
    sibling_root = zarr.open_group(str(materialized.store_path), mode="r")
    assert tuple(sibling_root["data"].shape) == (1, 1, 1, 2, 3, 4)
    assert tuple(sibling_root["data_pyramid/level_1"].shape) == (1, 1, 1, 1, 2, 2)


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
    assert tuple(root["data"].shape) == (1, 3, 2, 2, 3, 4)
    assert root.attrs["source_data_path"] == str(tmp_path.resolve())

    for position_index in range(3):
        for channel_index in range(2):
            loaded = np.array(
                root["data"][0, position_index, channel_index, :, :, :]
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
    assert tuple(root["data"].shape) == (1, 2, 2, 2, 3, 4)
    assert root.attrs["source_data_path"] == str(source_path.resolve())

    for position_index in range(2):
        for channel_index in range(2):
            loaded = np.array(
                root["data"][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(loaded, expected_blocks[(position_index, channel_index)])


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
    _write_real_n5_store(
        source_path,
        {
            "setup0/timepoint0/s0": expected_blocks[(0, 0)],
            "setup1/timepoint0/s0": expected_blocks[(1, 0)],
            "setup2/timepoint0/s0": expected_blocks[(0, 1)],
            "setup3/timepoint0/s0": expected_blocks[(1, 1)],
            "setup99/timepoint0/s0": np.zeros((2, 3, 4), dtype=np.uint16),
        },
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

    materialized = materialize_experiment_data_store(
        experiment=experiment,
        source_path=source_path,
        chunks=(1, 1, 1, 2, 2, 2),
        pyramid_factors=((1,), (1,), (1,), (1,), (1,), (1,)),
    )

    root = zarr.open_group(str(materialized.store_path), mode="r")
    assert tuple(root["data"].shape) == (1, 2, 2, 2, 3, 4)
    assert root.attrs["source_data_path"] == str(source_path.resolve())

    for position_index in range(2):
        for channel_index in range(2):
            loaded = np.array(
                root["data"][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(loaded, expected_blocks[(position_index, channel_index)])


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
    source_root = _open_test_zarr_group(source_path, mode="w")
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
    assert tuple(root["data"].shape) == (1, 2, 2, 2, 3, 4)
    assert root.attrs["source_data_path"] == str(source_path.resolve())

    for position_index in range(2):
        for channel_index in range(2):
            loaded = np.array(
                root["data"][0, position_index, channel_index, :, :, :]
            )
            assert np.array_equal(loaded, expected_blocks[(position_index, channel_index)])


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
    source_root = _open_test_zarr_group(source_store, mode="w")
    source_root.create_dataset("raw", data=source_data, chunks=(1, 8, 10), overwrite=True)
    source_root["raw"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    original_source_writer = experiment_module._write_dask_array_source_aligned_plane_batches
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
    assert writer_calls["source_aligned"] == 1
    assert writer_calls["chunk_batched"] == 0
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert root.attrs["materialization_write_strategy"] == "source_aligned_plane_batches"
    assert root.attrs["source_aligned_z_batch_depth"] == 2
    assert root.attrs["source_aligned_worker_count"] is None
    assert root.attrs["source_aligned_worker_memory_limit_bytes"] is None


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
    source_root = _open_test_zarr_group(source_store, mode="w")
    source_root.create_dataset("raw", data=source_data, chunks=(2, 8, 10), overwrite=True)
    source_root["raw"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    original_source_writer = experiment_module._write_dask_array_source_aligned_plane_batches
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
    assert writer_calls["source_aligned"] == 0
    assert writer_calls["chunk_batched"] == 1
    assert np.array_equal(np.array(root["data"][0, 0, 0, :, :, :]), source_data)
    assert root.attrs["materialization_write_strategy"] == "chunk_region_batches"
    assert root.attrs["source_aligned_z_batch_depth"] is None
    assert root.attrs["source_aligned_worker_count"] is None
    assert root.attrs["source_aligned_worker_memory_limit_bytes"] is None


def test_detect_client_worker_resources_extracts_min_limit():
    class _FakeClient:
        def scheduler_info(self):
            return {
                "workers": {
                    "tcp://127.0.0.1:1111": {"memory_limit": 12 << 30},
                    "tcp://127.0.0.1:2222": {"memory_limit": 8 << 30},
                }
            }

    worker_count, worker_memory_limit = experiment_module._detect_client_worker_resources(
        _FakeClient()
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
