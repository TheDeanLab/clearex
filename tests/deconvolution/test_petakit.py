#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path

# Local Imports
from clearex.deconvolution.petakit import (
    MATLAB_RUNTIME_ROOT_ENV,
    PETAKIT5D_ROOT_ENV,
    resolve_petakit_runtime_paths,
    run_petakit_deconvolution,
    validate_petakit_runtime,
)
import clearex.deconvolution.petakit as petakit


def _make_runtime_tree(tmp_path: Path) -> tuple[Path, Path]:
    petakit_root = tmp_path / "PetaKit5D"
    matlab_runtime_root = tmp_path / "MATLAB_Runtime" / "R2023a"
    mcc_dir = petakit_root / "mcc" / "linux"
    mcc_dir.mkdir(parents=True)
    matlab_runtime_root.mkdir(parents=True)
    launcher = mcc_dir / "run_mccMaster.sh"
    launcher.write_text("#!/bin/bash\n", encoding="utf-8")
    launcher.chmod(0o755)
    mcc_binary = mcc_dir / "mccMaster"
    mcc_binary.write_text("stub\n", encoding="utf-8")
    mcc_binary.chmod(0o755)
    return petakit_root, matlab_runtime_root


def test_validate_petakit_runtime_requires_environment_variables(monkeypatch) -> None:
    monkeypatch.delenv(PETAKIT5D_ROOT_ENV, raising=False)
    monkeypatch.delenv(MATLAB_RUNTIME_ROOT_ENV, raising=False)

    try:
        validate_petakit_runtime(mcc_mode=True)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("validate_petakit_runtime should fail without env vars.")

    assert PETAKIT5D_ROOT_ENV in message
    assert MATLAB_RUNTIME_ROOT_ENV in message
    assert "scripts/install_petakit_runtime.sh" in message


def test_resolve_petakit_runtime_paths_uses_environment_variables(
    tmp_path: Path, monkeypatch
) -> None:
    petakit_root, matlab_runtime_root = _make_runtime_tree(tmp_path)
    monkeypatch.setenv(PETAKIT5D_ROOT_ENV, str(petakit_root))
    monkeypatch.setenv(MATLAB_RUNTIME_ROOT_ENV, str(matlab_runtime_root))

    paths = resolve_petakit_runtime_paths()

    assert paths.petakit5d_root == petakit_root
    assert paths.matlab_runtime_root == matlab_runtime_root
    assert paths.mcc_master_launcher == (
        petakit_root / "mcc" / "linux" / "run_mccMaster.sh"
    )


def test_run_petakit_deconvolution_uses_configured_mcc_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    petakit_root, matlab_runtime_root = _make_runtime_tree(tmp_path)
    monkeypatch.setenv(PETAKIT5D_ROOT_ENV, str(petakit_root))
    monkeypatch.setenv(MATLAB_RUNTIME_ROOT_ENV, str(matlab_runtime_root))

    captured: dict[str, list[str]] = {}

    def _fake_run(command, *, check):
        captured["command"] = [str(item) for item in command]
        captured["check"] = [str(check)]

    monkeypatch.setattr(petakit.subprocess, "run", _fake_run)

    run_petakit_deconvolution(
        data_paths=[tmp_path],
        channel_patterns=["input.tif"],
        psf_fullpaths=[tmp_path / "psf.tif"],
        xy_pixel_size_um=0.2,
        dz_um=0.3,
        dz_psf_um=0.4,
        mcc_mode=True,
    )

    command = captured["command"]
    assert command[0] == str(petakit_root / "mcc" / "linux" / "run_mccMaster.sh")
    assert command[1] == str(matlab_runtime_root)
    assert command[2] == "XR_decon_data_wrapper"
    assert PETAKIT5D_ROOT_ENV not in " ".join(command)
    assert "channelPatterns" in command
    assert "{'input.tif'}" in command
    assert captured["check"] == ["True"]
