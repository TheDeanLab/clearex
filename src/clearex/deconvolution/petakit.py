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

"""Typed helpers for running PyPetaKit5D deconvolution wrappers."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Sequence, Tuple, Union, cast

PETAKIT5D_ROOT_ENV = "CLEAREX_PETAKIT5D_ROOT"
MATLAB_RUNTIME_ROOT_ENV = "CLEAREX_MATLAB_RUNTIME_ROOT"
PETAKIT_INSTALL_SCRIPT = "scripts/install_petakit_runtime.sh"
PETAKIT_MCC_FUNCTION_NAME = "XR_decon_data_wrapper"


@dataclass(frozen=True)
class PetakitRuntimePaths:
    """Resolved paths required for PyPetaKit5D MCC execution."""

    petakit5d_root: Path
    matlab_runtime_root: Path
    mcc_master_launcher: Path


def _runtime_install_hint() -> str:
    """Return a concise installation hint for missing runtime assets."""
    return (
        "Install the runtime assets with "
        f"`bash {PETAKIT_INSTALL_SCRIPT}` and source the generated "
        "`clearex_petakit_env.sh` file before running deconvolution."
    )


def _resolve_required_env_path(env_name: str) -> Path:
    """Resolve one required runtime path from the environment."""
    raw_value = str(os.environ.get(env_name, "")).strip()
    if not raw_value:
        raise RuntimeError(
            "PyPetaKit5D MCC runtime is not configured. "
            f"Set {PETAKIT5D_ROOT_ENV} and {MATLAB_RUNTIME_ROOT_ENV}. "
            f"{_runtime_install_hint()}"
        )
    return Path(raw_value).expanduser().resolve()


def resolve_petakit_runtime_paths() -> PetakitRuntimePaths:
    """Resolve and validate PyPetaKit5D MCC runtime paths from env vars.

    Returns
    -------
    PetakitRuntimePaths
        Existing runtime root paths and MCC launcher path.

    Raises
    ------
    RuntimeError
        If required environment variables or runtime files are missing.
    """
    petakit_root = _resolve_required_env_path(PETAKIT5D_ROOT_ENV)
    matlab_runtime_root = _resolve_required_env_path(MATLAB_RUNTIME_ROOT_ENV)
    launcher = petakit_root / "mcc" / "linux" / "run_mccMaster.sh"
    mcc_binary = petakit_root / "mcc" / "linux" / "mccMaster"

    missing: list[str] = []
    if not petakit_root.is_dir():
        missing.append(f"{PETAKIT5D_ROOT_ENV} directory: {petakit_root}")
    if not matlab_runtime_root.is_dir():
        missing.append(f"{MATLAB_RUNTIME_ROOT_ENV} directory: {matlab_runtime_root}")
    if not launcher.is_file():
        missing.append(f"MCC launcher: {launcher}")
    elif not os.access(launcher, os.X_OK):
        missing.append(f"executable MCC launcher: {launcher}")
    if not mcc_binary.is_file():
        missing.append(f"MCC executable: {mcc_binary}")
    elif not os.access(mcc_binary, os.X_OK):
        missing.append(f"executable MCC executable: {mcc_binary}")
    if missing:
        details = "\n- ".join(missing)
        raise RuntimeError(
            "PyPetaKit5D MCC runtime files are missing:\n"
            f"- {details}\n"
            f"{_runtime_install_hint()}"
        )

    return PetakitRuntimePaths(
        petakit5d_root=petakit_root,
        matlab_runtime_root=matlab_runtime_root,
        mcc_master_launcher=launcher,
    )


def validate_petakit_runtime(*, mcc_mode: bool = True) -> None:
    """Fail early when PyPetaKit5D MCC runtime assets are unavailable."""
    if not mcc_mode:
        return
    resolve_petakit_runtime_paths()


def _format_mcc_cell(values: Sequence[Union[str, Path]]) -> str:
    """Format a Python sequence as the MATLAB cell string wrapper expects."""
    return "{" + ",".join(f"'{str(value)}'" for value in values) + "}"


def _format_mcc_numeric_array(values: Sequence[Union[int, float]]) -> str:
    """Format a numeric sequence as the MATLAB vector string wrapper expects."""
    return "[" + ",".join(str(value) for value in values) + "]"


def _append_petakit_param(
    command: list[str],
    *,
    name: str,
    value: object,
    value_type: str,
) -> None:
    """Append one PyPetaKit5D wrapper-style parameter to an MCC command."""
    if value_type == "char":
        if not value:
            return
        command.extend([name, str(value)])
        return
    if value_type == "cell":
        if not value:
            return
        command.extend(
            [name, _format_mcc_cell(cast(Sequence[Union[str, Path]], value))]
        )
        return
    if value_type == "logical":
        if isinstance(value, (list, tuple)) and not value:
            command.extend([name, "[]"])
            return
        command.extend([name, str(bool(value)).lower()])
        return
    if value_type == "numericArr":
        if isinstance(value, (list, tuple)):
            values = tuple(value)
        else:
            values = (value,)
        if not values:
            return
        command.extend(
            [name, _format_mcc_numeric_array(cast(Sequence[Union[int, float]], values))]
        )
        return
    if value_type == "numericScalar":
        if isinstance(value, (list, tuple)):
            if not value:
                return
            value = value[0]
        command.extend([name, str(value)])


def _run_petakit_mcc_deconvolution(
    *,
    data_paths: Sequence[Union[str, Path]],
    params: dict[str, object],
) -> None:
    """Run PyPetaKit5D deconvolution through configured MCC runtime paths."""
    runtime_paths = resolve_petakit_runtime_paths()
    command = [
        str(runtime_paths.mcc_master_launcher),
        str(runtime_paths.matlab_runtime_root),
        PETAKIT_MCC_FUNCTION_NAME,
        _format_mcc_cell([str(Path(item).expanduser()) for item in data_paths]),
    ]
    param_types = {
        "resultDirName": "char",
        "overwrite": "logical",
        "channelPatterns": "cell",
        "skewAngle": "numericScalar",
        "dz": "numericScalar",
        "xyPixelSize": "numericArr",
        "save16bit": "logical",
        "parseSettingFile": "logical",
        "flipZstack": "logical",
        "background": "numericScalar",
        "dzPSF": "numericScalar",
        "edgeErosion": "numericScalar",
        "erodeByFTP": "logical",
        "psfFullpaths": "cell",
        "deconIter": "numericScalar",
        "RLMethod": "char",
        "wienerAlpha": "numericScalar",
        "OTFCumThresh": "numericScalar",
        "hannWinBounds": "numericArr",
        "skewed": "logical",
        "debug": "logical",
        "saveStep": "numericScalar",
        "psfGen": "logical",
        "GPUJob": "logical",
        "deconRotate": "logical",
        "batchSize": "numericArr",
        "blockSize": "numericArr",
        "largeFile": "logical",
        "largeMethod": "char",
        "zarrFile": "logical",
        "saveZarr": "logical",
        "dampFactor": "numericScalar",
        "scaleFactor": "numericScalar",
        "deconOffset": "numericScalar",
        "maskFullpaths": "cell",
        "parseCluster": "logical",
        "parseParfor": "logical",
        "masterCompute": "logical",
        "jobLogDir": "char",
        "cpusPerTask": "numericScalar",
        "uuid": "char",
        "unitWaitTime": "numericScalar",
        "maxTrialNum": "numericScalar",
        "mccMode": "logical",
        "configFile": "char",
        "GPUConfigFile": "char",
    }
    for name, value_type in param_types.items():
        _append_petakit_param(
            command,
            name=name,
            value=params.get(name),
            value_type=value_type,
        )
    subprocess.run(command, check=True)


def run_petakit_deconvolution(
    *,
    data_paths: Sequence[Union[str, Path]],
    channel_patterns: Sequence[str],
    psf_fullpaths: Sequence[Union[str, Path]],
    xy_pixel_size_um: float,
    dz_um: float,
    dz_psf_um: float,
    hann_win_bounds: Tuple[float, float] = (0.8, 1.0),
    wiener_alpha: float = 0.005,
    background: float = 110.0,
    decon_iter: int = 2,
    result_dir_name: str = "decon",
    rl_method: str = "omw",
    otf_cum_thresh: float = 0.6,
    save_16bit: bool = True,
    zarr_file: bool = False,
    save_zarr: bool = False,
    large_file: bool = False,
    block_size_zyx: Tuple[int, int, int] = (256, 256, 256),
    batch_size_zyx: Tuple[int, int, int] = (1024, 1024, 1024),
    parse_cluster: bool = False,
    gpu_job: bool = False,
    debug: bool = False,
    cpus_per_task: int = 2,
    mcc_mode: bool = True,
    config_file: str = "",
    gpu_config_file: str = "",
    psf_gen: bool = False,
    overwrite: bool = True,
) -> None:
    """Run a PyPetaKit5D deconvolution job.

    Parameters
    ----------
    data_paths : sequence of str or pathlib.Path
        Input image paths passed to ``XR_decon_data_wrapper``.
    channel_patterns : sequence of str
        Channel/file patterns used by the wrapper to select image content.
    psf_fullpaths : sequence of str or pathlib.Path
        PSF file paths aligned to channel patterns.
    xy_pixel_size_um : float
        Input image XY pixel size in microns.
    dz_um : float
        Input image Z step size in microns.
    dz_psf_um : float
        PSF Z step size in microns.
    hann_win_bounds : tuple[float, float], default=(0.8, 1.0)
        OTF Hann window lower and upper bounds.
    wiener_alpha : float, default=0.005
        Wiener regularization alpha.
    background : float, default=110.0
        Background offset used by the deconvolution routine.
    decon_iter : int, default=2
        Number of Richardson-Lucy iterations.
    result_dir_name : str, default="decon"
        Output directory name under each input location.
    rl_method : str, default="omw"
        Richardson-Lucy method variant.
    otf_cum_thresh : float, default=0.6
        OTF cumulative threshold.
    save_16bit : bool, default=True
        Whether to save 16-bit outputs.
    zarr_file : bool, default=False
        Whether input paths should be interpreted as Zarr inputs.
    save_zarr : bool, default=False
        Whether outputs should be saved as Zarr.
    large_file : bool, default=False
        Enables block/batch processing for large volumes.
    block_size_zyx : tuple[int, int, int], default=(256, 256, 256)
        Block size used when ``large_file`` is enabled.
    batch_size_zyx : tuple[int, int, int], default=(1024, 1024, 1024)
        Batch size used when ``large_file`` is enabled.
    parse_cluster : bool, default=False
        Enables PyPetaKit5D cluster parsing logic.
    gpu_job : bool, default=False
        Enables GPU execution mode in PyPetaKit5D.
    debug : bool, default=False
        Enables debug mode in PyPetaKit5D.
    cpus_per_task : int, default=2
        CPU count passed to PyPetaKit5D for the launched job.
    mcc_mode : bool, default=True
        Run via MATLAB-compiled runtime mode.
    config_file : str, default=""
        Optional external PyPetaKit5D config file path.
    gpu_config_file : str, default=""
        Optional GPU config file path.
    psf_gen : bool, default=False
        Whether PyPetaKit5D should generate PSFs internally.
    overwrite : bool, default=True
        Whether existing outputs should be overwritten.

    Returns
    -------
    None
        Wrapper side effects only.

    Raises
    ------
    RuntimeError
        If ``PyPetaKit5D`` is unavailable.
    ValueError
        If required parameters are missing or malformed.
    """
    if not data_paths:
        raise ValueError("data_paths cannot be empty.")
    if not channel_patterns:
        raise ValueError("channel_patterns cannot be empty.")
    if not psf_gen and not psf_fullpaths:
        raise ValueError("psf_fullpaths cannot be empty when psf_gen is False.")
    if len(hann_win_bounds) != 2:
        raise ValueError("hann_win_bounds must contain exactly two values.")
    if len(block_size_zyx) != 3:
        raise ValueError("block_size_zyx must contain exactly three values.")
    if len(batch_size_zyx) != 3:
        raise ValueError("batch_size_zyx must contain exactly three values.")

    params = {
        "resultDirName": str(result_dir_name),
        "overwrite": bool(overwrite),
        "channelPatterns": [str(item) for item in channel_patterns],
        "skewAngle": 32.45,
        "dz": float(dz_um),
        "xyPixelSize": float(xy_pixel_size_um),
        "save16bit": bool(save_16bit),
        "parseSettingFile": False,
        "flipZstack": False,
        "background": float(background),
        "dzPSF": float(dz_psf_um),
        "edgeErosion": 0,
        "erodeByFTP": True,
        "psfFullpaths": [str(Path(item).expanduser()) for item in psf_fullpaths],
        "deconIter": int(decon_iter),
        "RLMethod": str(rl_method),
        "wienerAlpha": float(wiener_alpha),
        "OTFCumThresh": float(otf_cum_thresh),
        "hannWinBounds": [float(hann_win_bounds[0]), float(hann_win_bounds[1])],
        "skewed": [],
        "debug": bool(debug),
        "saveStep": 5,
        "psfGen": bool(psf_gen),
        "GPUJob": bool(gpu_job),
        "deconRotate": False,
        "batchSize": [int(v) for v in batch_size_zyx],
        "blockSize": [int(v) for v in block_size_zyx],
        "largeFile": bool(large_file),
        "largeMethod": "inmemory",
        "zarrFile": bool(zarr_file),
        "saveZarr": bool(save_zarr),
        "dampFactor": 1,
        "scaleFactor": [],
        "deconOffset": 0,
        "maskFullpaths": [],
        "parseCluster": bool(parse_cluster),
        "parseParfor": False,
        "masterCompute": True,
        "jobLogDir": "../job_logs",
        "cpusPerTask": int(cpus_per_task),
        "uuid": "",
        "unitWaitTime": 1,
        "maxTrialNum": 3,
        "mccMode": bool(mcc_mode),
        "configFile": str(config_file),
        "GPUConfigFile": str(gpu_config_file),
    }
    if mcc_mode:
        _run_petakit_mcc_deconvolution(data_paths=data_paths, params=params)
        return

    try:
        from PyPetaKit5D import XR_decon_data_wrapper
    except Exception as exc:  # pragma: no cover - environment-dependent dependency
        raise RuntimeError(
            "PyPetaKit5D is unavailable. Install with `pip install clearex[decon]`."
        ) from exc

    XR_decon_data_wrapper(
        [str(Path(item).expanduser()) for item in data_paths],
        **params,
    )


def richardson_lucy(
    data_paths: list[str],
    PSFPath: str,
    xy_pixel: float,
    z_pixel: float,
    channel_pattern: list[str],
) -> None:
    """Compatibility shim for existing Richardson-Lucy wrapper usage.

    Parameters
    ----------
    data_paths : list[str]
        Input image paths.
    PSFPath : str
        PSF file path.
    xy_pixel : float
        Input image XY pixel size in microns.
    z_pixel : float
        Input image Z step size in microns.
    channel_pattern : list[str]
        Channel/file patterns.

    Returns
    -------
    None
        Wrapper side effects only.
    """
    run_petakit_deconvolution(
        data_paths=data_paths,
        channel_patterns=channel_pattern,
        psf_fullpaths=[PSFPath],
        xy_pixel_size_um=float(xy_pixel),
        dz_um=float(z_pixel),
        dz_psf_um=float(z_pixel),
        hann_win_bounds=(0.8, 1.0),
        wiener_alpha=0.005,
        background=110.0,
        decon_iter=2,
        result_dir_name="decon",
        rl_method="omw",
        otf_cum_thresh=0.6,
        save_16bit=True,
        zarr_file=False,
        save_zarr=False,
        large_file=False,
        parse_cluster=False,
        gpu_job=False,
        debug=False,
        cpus_per_task=60,
        mcc_mode=True,
        gpu_config_file="",
        config_file="",
        psf_gen=False,
        overwrite=True,
    )
