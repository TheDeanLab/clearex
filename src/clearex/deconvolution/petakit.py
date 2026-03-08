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

"""Typed helpers for running PyPetaKit5D deconvolution wrappers."""

from __future__ import annotations

# Standard Library Imports
from pathlib import Path
from typing import Sequence, Tuple, Union


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

    try:
        from PyPetaKit5D import XR_decon_data_wrapper
    except Exception as exc:  # pragma: no cover - environment-dependent dependency
        raise RuntimeError(
            "PyPetaKit5D is unavailable. Install with `pip install clearex[decon]`."
        ) from exc

    params = {
        "channelPatterns": [str(item) for item in channel_patterns],
        "resultDirName": str(result_dir_name),
        "xyPixelSize": float(xy_pixel_size_um),
        "dz": float(dz_um),
        "dzPSF": float(dz_psf_um),
        "hannWinBounds": [float(hann_win_bounds[0]), float(hann_win_bounds[1])],
        "psfFullpaths": [str(Path(item).expanduser()) for item in psf_fullpaths],
        "parseSettingFile": False,
        "RLMethod": str(rl_method),
        "wienerAlpha": float(wiener_alpha),
        "OTFCumThresh": float(otf_cum_thresh),
        "edgeErosion": 0,
        "background": float(background),
        "deconIter": int(decon_iter),
        "save16bit": bool(save_16bit),
        "zarrFile": bool(zarr_file),
        "saveZarr": bool(save_zarr),
        "parseCluster": bool(parse_cluster),
        "largeFile": bool(large_file),
        "GPUJob": bool(gpu_job),
        "debug": bool(debug),
        "cpusPerTask": int(cpus_per_task),
        "mccMode": bool(mcc_mode),
        "GPUConfigFile": str(gpu_config_file),
        "configFile": str(config_file),
        "psfGen": bool(psf_gen),
        "blockSize": [int(v) for v in block_size_zyx],
        "batchSize": [int(v) for v in batch_size_zyx],
        "overwrite": bool(overwrite),
    }
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
