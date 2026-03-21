#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import clearex.io.n5_legacy_helper as helper_module
import pytest


def test_helper_uses_scheduler_address_when_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def close(self) -> None:
            captured["client_closed"] = True

    def _fake_create_dask_client(*, scheduler_address=None, **kwargs):
        del kwargs
        captured["scheduler_address"] = scheduler_address
        return _FakeClient()

    def _fake_load_navigate_experiment(path: Path):
        captured["experiment_path"] = Path(path)
        return object()

    def _fake_materialize_experiment_data_store(**kwargs):
        captured["materialize_kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(helper_module, "create_dask_client", _fake_create_dask_client)
    monkeypatch.setattr(
        helper_module, "load_navigate_experiment", _fake_load_navigate_experiment
    )
    monkeypatch.setattr(
        helper_module,
        "materialize_experiment_data_store",
        _fake_materialize_experiment_data_store,
    )

    experiment_path = tmp_path / "experiment.yml"
    source_path = tmp_path / "source.n5"
    output_store = tmp_path / "out.zarr"
    source_path.mkdir(parents=True)

    monkeypatch.setattr(
        helper_module,
        "main",
        helper_module.main,
    )
    monkeypatch.setattr(
        helper_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "_Args",
            (),
            {
                "experiment_path": str(experiment_path),
                "source_path": str(source_path),
                "output_store": str(output_store),
                "chunks": "1,1,1,8,8,8",
                "pyramid_factors": "[[1],[1],[1],[1],[1],[1]]",
                "scheduler_address": "tcp://scheduler:8786",
                "local_n_workers": None,
                "local_threads_per_worker": None,
                "local_memory_limit": None,
            },
        )(),
    )

    exit_code = helper_module.main()

    assert exit_code == 0
    assert captured["scheduler_address"] == "tcp://scheduler:8786"
    materialize_kwargs = captured["materialize_kwargs"]
    assert materialize_kwargs["client"] is not None
    assert materialize_kwargs["force_rebuild"] is True
    assert captured.get("client_closed", False) is True


def test_helper_does_not_create_client_without_scheduler_address(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_create_dask_client(*, scheduler_address=None, **kwargs):
        del scheduler_address, kwargs
        raise AssertionError("create_dask_client should not be called")

    def _fake_load_navigate_experiment(path: Path):
        captured["experiment_path"] = Path(path)
        return object()

    def _fake_materialize_experiment_data_store(**kwargs):
        captured["materialize_kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(helper_module, "create_dask_client", _fake_create_dask_client)
    monkeypatch.setattr(
        helper_module, "load_navigate_experiment", _fake_load_navigate_experiment
    )
    monkeypatch.setattr(
        helper_module,
        "materialize_experiment_data_store",
        _fake_materialize_experiment_data_store,
    )

    experiment_path = tmp_path / "experiment.yml"
    source_path = tmp_path / "source.n5"
    output_store = tmp_path / "out.zarr"
    source_path.mkdir(parents=True)

    monkeypatch.setattr(
        helper_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "_Args",
            (),
            {
                "experiment_path": str(experiment_path),
                "source_path": str(source_path),
                "output_store": str(output_store),
                "chunks": "1,1,1,8,8,8",
                "pyramid_factors": "[[1],[1],[1],[1],[1],[1]]",
                "scheduler_address": "",
                "local_n_workers": None,
                "local_threads_per_worker": None,
                "local_memory_limit": None,
            },
        )(),
    )

    exit_code = helper_module.main()

    assert exit_code == 0
    assert captured["materialize_kwargs"]["client"] is None


def test_helper_closes_client_on_materialize_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def close(self) -> None:
            captured["client_closed"] = True

    def _fake_create_dask_client(*, scheduler_address=None, **kwargs):
        del kwargs
        captured["scheduler_address"] = scheduler_address
        return _FakeClient()

    def _fake_load_navigate_experiment(path: Path):
        captured["experiment_path"] = Path(path)
        return object()

    def _fake_materialize_experiment_data_store(**kwargs):
        del kwargs
        raise RuntimeError("materialize failed")

    monkeypatch.setattr(helper_module, "create_dask_client", _fake_create_dask_client)
    monkeypatch.setattr(
        helper_module, "load_navigate_experiment", _fake_load_navigate_experiment
    )
    monkeypatch.setattr(
        helper_module,
        "materialize_experiment_data_store",
        _fake_materialize_experiment_data_store,
    )

    experiment_path = tmp_path / "experiment.yml"
    source_path = tmp_path / "source.n5"
    output_store = tmp_path / "out.zarr"
    source_path.mkdir(parents=True)

    monkeypatch.setattr(
        helper_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "_Args",
            (),
            {
                "experiment_path": str(experiment_path),
                "source_path": str(source_path),
                "output_store": str(output_store),
                "chunks": "1,1,1,8,8,8",
                "pyramid_factors": "[[1],[1],[1],[1],[1],[1]]",
                "scheduler_address": "tcp://scheduler:8786",
                "local_n_workers": None,
                "local_threads_per_worker": None,
                "local_memory_limit": None,
            },
        )(),
    )

    with pytest.raises(RuntimeError, match="materialize failed"):
        helper_module.main()

    assert captured["scheduler_address"] == "tcp://scheduler:8786"
    assert captured.get("client_closed", False) is True


def test_helper_uses_local_hints_when_scheduler_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def close(self) -> None:
            captured["client_closed"] = True

    def _fake_create_dask_client(**kwargs):
        captured["create_kwargs"] = dict(kwargs)
        return _FakeClient()

    def _fake_load_navigate_experiment(path: Path):
        captured["experiment_path"] = Path(path)
        return object()

    def _fake_materialize_experiment_data_store(**kwargs):
        captured["materialize_kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(helper_module, "create_dask_client", _fake_create_dask_client)
    monkeypatch.setattr(
        helper_module, "load_navigate_experiment", _fake_load_navigate_experiment
    )
    monkeypatch.setattr(
        helper_module,
        "materialize_experiment_data_store",
        _fake_materialize_experiment_data_store,
    )

    experiment_path = tmp_path / "experiment.yml"
    source_path = tmp_path / "source.n5"
    output_store = tmp_path / "out.zarr"
    source_path.mkdir(parents=True)

    monkeypatch.setattr(
        helper_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "_Args",
            (),
            {
                "experiment_path": str(experiment_path),
                "source_path": str(source_path),
                "output_store": str(output_store),
                "chunks": "1,1,1,8,8,8",
                "pyramid_factors": "[[1],[1],[1],[1],[1],[1]]",
                "scheduler_address": "",
                "local_n_workers": 4,
                "local_threads_per_worker": 2,
                "local_memory_limit": "123456789",
            },
        )(),
    )

    exit_code = helper_module.main()

    assert exit_code == 0
    assert captured["create_kwargs"]["n_workers"] == 4
    assert captured["create_kwargs"]["threads_per_worker"] == 2
    assert captured["create_kwargs"]["processes"] is False
    assert captured["create_kwargs"]["memory_limit"] == "123456789"
    assert captured["materialize_kwargs"]["client"] is not None
    assert captured.get("client_closed", False) is True
