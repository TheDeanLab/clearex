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

"""Chunk-parallel deconvolution workflow on canonical 6D Zarr stores."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union

# Third Party Imports
import dask
from dask import delayed
import numpy as np
import tifffile
import zarr

# Local Imports
from clearex.deconvolution.petakit import run_petakit_deconvolution
from clearex.io.provenance import register_latest_output_reference

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class DeconvolutionSummary:
    """Summary metadata for one deconvolution run.

    Attributes
    ----------
    component : str
        Output latest component group path.
    data_component : str
        Output latest data-array component path.
    volumes_processed : int
        Number of processed ``(t, p, c)`` volumes.
    channel_count : int
        Number of channels processed per position/timepoint.
    psf_mode : str
        Effective PSF mode (``"measured"`` or ``"synthetic"``).
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Chunk shape used for saved deconvolution output.
    """

    component: str
    data_component: str
    volumes_processed: int
    channel_count: int
    psf_mode: str
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]


def _as_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings.

    Parameters
    ----------
    value : Any
        Candidate value. Accepts string, sequence, or ``None``.

    Returns
    -------
    list[str]
        Normalized non-empty string list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.replace("\n", ",")
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _as_float_list(value: Any) -> list[float]:
    """Normalize a value into a list of floats.

    Parameters
    ----------
    value : Any
        Candidate value. Accepts numeric, string, sequence, or ``None``.

    Returns
    -------
    list[float]
        Normalized float list.
    """
    out: list[float] = []
    for item in _as_string_list(value):
        out.append(float(item))
    return out


def _as_int_triplet(
    value: Any,
    *,
    default: tuple[int, int, int],
    field_name: str,
) -> tuple[int, int, int]:
    """Normalize a value into a positive integer triplet.

    Parameters
    ----------
    value : Any
        Candidate value.
    default : tuple[int, int, int]
        Default value used when input is empty.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    tuple[int, int, int]
        Normalized positive integer triplet.

    Raises
    ------
    ValueError
        If the provided value is not a positive integer triplet.
    """
    parts = _as_string_list(value)
    if not parts:
        return default
    if len(parts) != 3:
        raise ValueError(f"{field_name} must define exactly three values.")
    triplet = (int(parts[0]), int(parts[1]), int(parts[2]))
    if any(v <= 0 for v in triplet):
        raise ValueError(f"{field_name} must contain positive integers.")
    return triplet


def _broadcast_channel_value(
    values: Sequence[Any],
    *,
    channel_index: int,
    field_name: str,
) -> Any:
    """Select channel-specific value with scalar-or-vector broadcasting.

    Parameters
    ----------
    values : sequence of Any
        Candidate values for channels. Length 1 broadcasts to all channels.
    channel_index : int
        Zero-based channel index.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    Any
        Value selected for ``channel_index``.

    Raises
    ------
    ValueError
        If no values exist or index is out of range for multi-value lists.
    """
    if not values:
        raise ValueError(f"{field_name} cannot be empty.")
    if len(values) == 1:
        return values[0]
    if channel_index < len(values):
        return values[channel_index]
    raise ValueError(
        f"{field_name} length={len(values)} does not cover channel index {channel_index}."
    )


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize deconvolution runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized deconvolution parameters.

    Raises
    ------
    ValueError
        If required values are malformed.
    """
    normalized = dict(parameters)
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    psf_mode = str(normalized.get("psf_mode", "measured")).strip().lower()
    if psf_mode not in {"measured", "synthetic"}:
        raise ValueError("deconvolution psf_mode must be 'measured' or 'synthetic'.")
    normalized["psf_mode"] = psf_mode

    normalized["channel_indices"] = [
        max(0, int(value))
        for value in _as_float_list(normalized.get("channel_indices"))
    ]
    normalized["measured_psf_paths"] = _as_string_list(
        normalized.get("measured_psf_paths")
    )
    normalized["measured_psf_xy_um"] = _as_float_list(
        normalized.get("measured_psf_xy_um")
    )
    normalized["measured_psf_z_um"] = _as_float_list(
        normalized.get("measured_psf_z_um")
    )
    normalized["synthetic_excitation_nm"] = _as_float_list(
        normalized.get("synthetic_excitation_nm")
    )
    normalized["synthetic_emission_nm"] = _as_float_list(
        normalized.get("synthetic_emission_nm")
    )
    normalized["synthetic_numerical_aperture"] = _as_float_list(
        normalized.get("synthetic_numerical_aperture")
    )
    normalized["synthetic_refractive_index"] = float(
        normalized.get("synthetic_refractive_index", 1.33)
    )
    normalized["synthetic_psf_size_zyx"] = _as_int_triplet(
        normalized.get("synthetic_psf_size_zyx"),
        default=(65, 129, 129),
        field_name="deconvolution synthetic_psf_size_zyx",
    )

    hann_values = _as_float_list(normalized.get("hann_window_bounds", [0.8, 1.0]))
    if len(hann_values) != 2:
        raise ValueError("deconvolution hann_window_bounds must define two values.")
    hann_low, hann_high = float(hann_values[0]), float(hann_values[1])
    if hann_low <= 0 or hann_high <= 0 or hann_low > hann_high:
        raise ValueError(
            "deconvolution hann_window_bounds must satisfy 0 < low <= high."
        )
    normalized["hann_window_bounds"] = [hann_low, hann_high]

    normalized["wiener_alpha"] = float(normalized.get("wiener_alpha", 0.005))
    if normalized["wiener_alpha"] < 0:
        raise ValueError("deconvolution wiener_alpha cannot be negative.")
    normalized["background"] = float(normalized.get("background", 110.0))
    if normalized["background"] < 0:
        raise ValueError("deconvolution background cannot be negative.")
    normalized["decon_iterations"] = max(1, int(normalized.get("decon_iterations", 2)))
    normalized["otf_cum_thresh"] = float(normalized.get("otf_cum_thresh", 0.6))
    normalized["data_xy_pixel_um"] = float(normalized.get("data_xy_pixel_um", 0.0))
    normalized["data_z_pixel_um"] = float(normalized.get("data_z_pixel_um", 0.0))
    normalized["large_file"] = bool(normalized.get("large_file", False))
    normalized["save_16bit"] = bool(normalized.get("save_16bit", True))
    normalized["save_zarr"] = bool(normalized.get("save_zarr", False))
    normalized["gpu_job"] = bool(normalized.get("gpu_job", False))
    normalized["debug"] = bool(normalized.get("debug", False))
    normalized["mcc_mode"] = bool(normalized.get("mcc_mode", True))
    normalized["cpus_per_task"] = max(1, int(normalized.get("cpus_per_task", 2)))
    normalized["config_file"] = str(normalized.get("config_file", "")).strip()
    normalized["gpu_config_file"] = str(normalized.get("gpu_config_file", "")).strip()
    normalized["block_size_zyx"] = list(
        _as_int_triplet(
            normalized.get("block_size_zyx"),
            default=(256, 256, 256),
            field_name="deconvolution block_size_zyx",
        )
    )
    normalized["batch_size_zyx"] = list(
        _as_int_triplet(
            normalized.get("batch_size_zyx"),
            default=(1024, 1024, 1024),
            field_name="deconvolution batch_size_zyx",
        )
    )
    return normalized


def _extract_store_voxel_sizes_um(
    root: zarr.Group,
) -> tuple[Optional[float], Optional[float]]:
    """Extract voxel sizes from analysis-store metadata.

    Parameters
    ----------
    root : zarr.Group
        Open analysis-store root group.

    Returns
    -------
    tuple[float | None, float | None]
        ``(xy_um, z_um)`` values when available.
    """
    root_attrs = dict(root.attrs)
    data_attrs = dict(root["data"].attrs) if "data" in root else {}

    for attrs in (data_attrs, root_attrs):
        voxel = attrs.get("voxel_size_um_zyx")
        if isinstance(voxel, (list, tuple)) and len(voxel) >= 3:
            return float(voxel[2]), float(voxel[0])
    for attrs in (root_attrs, data_attrs):
        navigate = attrs.get("navigate_experiment")
        if not isinstance(navigate, dict):
            continue
        xy_value = navigate.get("xy_pixel_size_um")
        z_value = navigate.get("z_step_um")
        if xy_value is None or z_value is None:
            continue
        return float(xy_value), float(z_value)
    return None, None


def _resolve_data_voxel_sizes_um(
    *,
    root: zarr.Group,
    parameters: Mapping[str, Any],
) -> tuple[float, float]:
    """Resolve deconvolution input voxel sizes.

    Parameters
    ----------
    root : zarr.Group
        Open analysis-store root group.
    parameters : mapping[str, Any]
        Normalized deconvolution parameter mapping.

    Returns
    -------
    tuple[float, float]
        ``(xy_um, z_um)`` voxel size in microns.

    Raises
    ------
    ValueError
        If voxel size is unavailable from both parameters and store metadata.
    """
    configured_xy = float(parameters.get("data_xy_pixel_um", 0.0))
    configured_z = float(parameters.get("data_z_pixel_um", 0.0))
    if configured_xy > 0 and configured_z > 0:
        return configured_xy, configured_z

    store_xy, store_z = _extract_store_voxel_sizes_um(root)
    xy_um = configured_xy if configured_xy > 0 else (store_xy or 0.0)
    z_um = configured_z if configured_z > 0 else (store_z or 0.0)
    if xy_um <= 0 or z_um <= 0:
        raise ValueError(
            "Deconvolution requires voxel size information. Configure "
            "data_xy_pixel_um and data_z_pixel_um in operation parameters."
        )
    return float(xy_um), float(z_um)


def _gaussian_psf_kernel(
    *,
    size_zyx: tuple[int, int, int],
    sigma_zyx_px: tuple[float, float, float],
) -> np.ndarray:
    """Generate a normalized Gaussian PSF kernel.

    Parameters
    ----------
    size_zyx : tuple[int, int, int]
        Output PSF shape in ``(z, y, x)`` order.
    sigma_zyx_px : tuple[float, float, float]
        Standard deviations in pixels for ``(z, y, x)``.

    Returns
    -------
    numpy.ndarray
        Float32 normalized PSF kernel with sum ``1``.
    """
    z_size, y_size, x_size = size_zyx
    z_sigma, y_sigma, x_sigma = sigma_zyx_px

    z_axis = np.arange(z_size, dtype=np.float32) - (float(z_size - 1) / 2.0)
    y_axis = np.arange(y_size, dtype=np.float32) - (float(y_size - 1) / 2.0)
    x_axis = np.arange(x_size, dtype=np.float32) - (float(x_size - 1) / 2.0)
    zz, yy, xx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")

    kernel = np.exp(
        -0.5
        * (
            (zz / max(1e-6, z_sigma)) ** 2
            + (yy / max(1e-6, y_sigma)) ** 2
            + (xx / max(1e-6, x_sigma)) ** 2
        )
    ).astype(np.float32)
    total = float(np.sum(kernel))
    if total <= 0:
        return np.zeros(size_zyx, dtype=np.float32)
    return kernel / total


def _synthetic_sigma_zyx_px(
    *,
    emission_nm: float,
    numerical_aperture: float,
    voxel_xy_um: float,
    voxel_z_um: float,
    refractive_index: float,
) -> tuple[float, float, float]:
    """Approximate synthetic PSF sigma values in pixel units.

    Parameters
    ----------
    emission_nm : float
        Emission wavelength in nanometers.
    numerical_aperture : float
        Objective numerical aperture.
    voxel_xy_um : float
        Data XY pixel size in microns.
    voxel_z_um : float
        Data Z step size in microns.
    refractive_index : float
        Refractive index used for axial resolution approximation.

    Returns
    -------
    tuple[float, float, float]
        ``(sigma_z_px, sigma_y_px, sigma_x_px)``.
    """
    emission_um = float(emission_nm) / 1000.0
    lateral_fwhm_um = 0.61 * emission_um / max(1e-6, float(numerical_aperture))
    axial_fwhm_um = (
        2.0
        * float(refractive_index)
        * emission_um
        / max(1e-6, float(numerical_aperture) ** 2)
    )
    lateral_sigma_um = lateral_fwhm_um / 2.355
    axial_sigma_um = axial_fwhm_um / 2.355
    sigma_z = max(0.5, axial_sigma_um / max(1e-6, voxel_z_um))
    sigma_xy = max(0.5, lateral_sigma_um / max(1e-6, voxel_xy_um))
    return (float(sigma_z), float(sigma_xy), float(sigma_xy))


def _select_psf_for_channel(
    *,
    params: Mapping[str, Any],
    channel_index: int,
    voxel_xy_um: float,
    voxel_z_um: float,
    temp_dir: Path,
) -> tuple[Path, float]:
    """Resolve or synthesize PSF path for one channel.

    Parameters
    ----------
    params : mapping[str, Any]
        Normalized deconvolution parameters.
    channel_index : int
        Channel index.
    voxel_xy_um : float
        Input image XY voxel size.
    voxel_z_um : float
        Input image Z voxel size.
    temp_dir : pathlib.Path
        Temporary directory used for synthetic PSF outputs.

    Returns
    -------
    tuple[pathlib.Path, float]
        PSF path and PSF Z-step in microns.

    Raises
    ------
    ValueError
        If required PSF settings are missing.
    FileNotFoundError
        If measured PSF file does not exist.
    """
    psf_mode = str(params["psf_mode"])
    if psf_mode == "measured":
        psf_path = Path(
            str(
                _broadcast_channel_value(
                    params["measured_psf_paths"],
                    channel_index=channel_index,
                    field_name="deconvolution measured_psf_paths",
                )
            )
        ).expanduser()
        if not psf_path.exists():
            raise FileNotFoundError(psf_path)
        psf_z_um = float(
            _broadcast_channel_value(
                params["measured_psf_z_um"],
                channel_index=channel_index,
                field_name="deconvolution measured_psf_z_um",
            )
        )
        _ = _broadcast_channel_value(
            params["measured_psf_xy_um"],
            channel_index=channel_index,
            field_name="deconvolution measured_psf_xy_um",
        )
        return psf_path.resolve(), psf_z_um

    emission_nm = float(
        _broadcast_channel_value(
            params["synthetic_emission_nm"],
            channel_index=channel_index,
            field_name="deconvolution synthetic_emission_nm",
        )
    )
    _ = float(
        _broadcast_channel_value(
            params["synthetic_excitation_nm"],
            channel_index=channel_index,
            field_name="deconvolution synthetic_excitation_nm",
        )
    )
    numerical_aperture = float(
        _broadcast_channel_value(
            params["synthetic_numerical_aperture"],
            channel_index=channel_index,
            field_name="deconvolution synthetic_numerical_aperture",
        )
    )
    sigma_zyx = _synthetic_sigma_zyx_px(
        emission_nm=emission_nm,
        numerical_aperture=numerical_aperture,
        voxel_xy_um=float(voxel_xy_um),
        voxel_z_um=float(voxel_z_um),
        refractive_index=float(params["synthetic_refractive_index"]),
    )
    psf_shape = tuple(int(v) for v in params["synthetic_psf_size_zyx"])
    psf_data = _gaussian_psf_kernel(
        size_zyx=(psf_shape[0], psf_shape[1], psf_shape[2]),
        sigma_zyx_px=sigma_zyx,
    )
    psf_path = temp_dir / f"synthetic_psf_ch{int(channel_index):02d}.tif"
    tifffile.imwrite(str(psf_path), psf_data, photometric="minisblack")
    return psf_path.resolve(), float(voxel_z_um)


def _load_decon_output(path: Path) -> np.ndarray:
    """Load deconvolution output volume from a supported file type.

    Parameters
    ----------
    path : pathlib.Path
        Output path from external deconvolution execution.

    Returns
    -------
    numpy.ndarray
        Loaded 3D volume in ``(z, y, x)`` order.

    Raises
    ------
    ValueError
        If output data cannot be interpreted as a 3D volume.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        data = np.asarray(tifffile.imread(str(path)))
    elif suffix == ".npy":
        data = np.asarray(np.load(str(path)))
    elif suffix == ".npz":
        with np.load(str(path)) as archive:
            first_key = next(iter(archive.keys()))
            data = np.asarray(archive[first_key])
    elif suffix in {".zarr", ".n5"}:
        root = zarr.open_group(str(path), mode="r")
        array_names = list(root.array_keys())
        if "data" in array_names:
            data = np.asarray(root["data"])
        elif array_names:
            data = np.asarray(root[array_names[0]])
        else:
            group_names = list(root.group_keys())
            if not group_names:
                raise ValueError(f"No arrays found in deconvolution output {path}.")
            child = root[group_names[0]]
            child_arrays = list(child.array_keys())
            if not child_arrays:
                raise ValueError(f"No arrays found in deconvolution output {path}.")
            data = np.asarray(child[child_arrays[0]])
    else:
        data = np.asarray(tifffile.imread(str(path)))

    if data.ndim == 4 and int(data.shape[0]) == 1:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D deconvolution output volume, got shape {tuple(data.shape)}."
        )
    return np.asarray(data)


def _find_decon_output_path(
    *,
    temp_dir: Path,
    input_path: Path,
    psf_path: Path,
) -> Path:
    """Locate the most likely deconvolution output path.

    Parameters
    ----------
    temp_dir : pathlib.Path
        Temporary working directory used for deconvolution.
    input_path : pathlib.Path
        Input volume path written for deconvolution.
    psf_path : pathlib.Path
        PSF file path used for deconvolution.

    Returns
    -------
    pathlib.Path
        Selected output path.

    Raises
    ------
    FileNotFoundError
        If no output candidate is found.
    """
    candidates: list[Path] = []
    supported_suffixes = {".tif", ".tiff", ".npy", ".npz", ".zarr", ".n5"}
    for path in temp_dir.rglob("*"):
        if not path.is_file() and path.suffix.lower() not in {".zarr", ".n5"}:
            continue
        if path == input_path or path == psf_path:
            continue
        if path.suffix.lower() not in supported_suffixes:
            continue
        name_lower = path.name.lower()
        if "psf" in name_lower:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No deconvolution output was produced under temporary path {temp_dir}."
        )
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _run_deconvolution_for_volume(
    *,
    zarr_path: str,
    source_component: str,
    output_data_component: str,
    t_index: int,
    p_index: int,
    c_index: int,
    parameters: Mapping[str, Any],
    voxel_xy_um: float,
    voxel_z_um: float,
) -> int:
    """Run deconvolution for one canonical ``(t, p, c)`` volume.

    Parameters
    ----------
    zarr_path : str
        Zarr store path.
    source_component : str
        Source data component path.
    output_data_component : str
        Output data component path.
    t_index : int
        Time index.
    p_index : int
        Position index.
    c_index : int
        Channel index.
    parameters : mapping[str, Any]
        Normalized deconvolution parameters.
    voxel_xy_um : float
        Source XY voxel size in microns.
    voxel_z_um : float
        Source Z voxel size in microns.

    Returns
    -------
    int
        Constant ``1`` used for task completion accounting.

    Raises
    ------
    ValueError
        If output shape does not match source shape.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    source_volume = np.asarray(source[t_index, p_index, c_index, :, :, :])

    with TemporaryDirectory(
        prefix=f"clearex_decon_t{t_index}_p{p_index}_c{c_index}_"
    ) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_path = (
            temp_dir / f"input_t{t_index:04d}_p{p_index:04d}_c{c_index:04d}.tif"
        )
        tifffile.imwrite(str(input_path), source_volume, photometric="minisblack")

        psf_path, dz_psf_um = _select_psf_for_channel(
            params=parameters,
            channel_index=int(c_index),
            voxel_xy_um=float(voxel_xy_um),
            voxel_z_um=float(voxel_z_um),
            temp_dir=temp_dir,
        )
        run_petakit_deconvolution(
            data_paths=[temp_dir],
            channel_patterns=[input_path.name],
            psf_fullpaths=[psf_path],
            xy_pixel_size_um=float(voxel_xy_um),
            dz_um=float(voxel_z_um),
            dz_psf_um=float(dz_psf_um),
            hann_win_bounds=(
                float(parameters["hann_window_bounds"][0]),
                float(parameters["hann_window_bounds"][1]),
            ),
            wiener_alpha=float(parameters["wiener_alpha"]),
            background=float(parameters["background"]),
            decon_iter=int(parameters["decon_iterations"]),
            result_dir_name="decon",
            rl_method="omw",
            otf_cum_thresh=float(parameters["otf_cum_thresh"]),
            save_16bit=bool(parameters["save_16bit"]),
            zarr_file=False,
            save_zarr=bool(parameters["save_zarr"]),
            large_file=bool(parameters["large_file"]),
            block_size_zyx=(
                int(parameters["block_size_zyx"][0]),
                int(parameters["block_size_zyx"][1]),
                int(parameters["block_size_zyx"][2]),
            ),
            batch_size_zyx=(
                int(parameters["batch_size_zyx"][0]),
                int(parameters["batch_size_zyx"][1]),
                int(parameters["batch_size_zyx"][2]),
            ),
            parse_cluster=False,
            gpu_job=bool(parameters["gpu_job"]),
            debug=bool(parameters["debug"]),
            cpus_per_task=int(parameters["cpus_per_task"]),
            mcc_mode=bool(parameters["mcc_mode"]),
            config_file=str(parameters["config_file"]),
            gpu_config_file=str(parameters["gpu_config_file"]),
            psf_gen=False,
            overwrite=True,
        )
        output_path = _find_decon_output_path(
            temp_dir=temp_dir,
            input_path=input_path,
            psf_path=psf_path,
        )
        output_volume = _load_decon_output(output_path)

    if tuple(output_volume.shape) != tuple(source_volume.shape):
        raise ValueError(
            "Deconvolution output shape mismatch: "
            f"expected {tuple(source_volume.shape)}, got {tuple(output_volume.shape)}."
        )

    write_root = zarr.open_group(str(zarr_path), mode="a")
    write_root[output_data_component][t_index, p_index, c_index, :, :, :] = (
        output_volume.astype(write_root[output_data_component].dtype, copy=False)
    )
    return 1


def _prepare_output_array(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    parameters: Mapping[str, Any],
) -> tuple[str, str, tuple[int, int, int, int, int, int], tuple[int, ...]]:
    """Prepare latest deconvolution output dataset in a store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis store path.
    source_component : str
        Source component path.
    parameters : mapping[str, Any]
        Normalized deconvolution parameters.

    Returns
    -------
    tuple[str, str, tuple[int, int, int, int, int, int], tuple[int, ...]]
        ``(component, data_component, shape_tpczyx, chunks_tpczyx)``.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    source = root[source_component]
    shape = tuple(int(v) for v in source.shape)
    chunks = (
        1,
        1,
        1,
        int(shape[3]),
        int(shape[4]),
        int(shape[5]),
    )

    results_group = root.require_group("results")
    decon_group = results_group.require_group("deconvolution")
    if "latest" in decon_group:
        del decon_group["latest"]
    latest = decon_group.create_group("latest")
    latest.create_dataset(
        name="data",
        shape=shape,
        chunks=chunks,
        dtype=source.dtype,
        overwrite=True,
    )
    latest.attrs.update(
        {
            "storage_policy": "latest_only",
            "source_component": str(source_component),
            "parameters": {str(key): value for key, value in dict(parameters).items()},
            "run_id": None,
        }
    )
    component = "results/deconvolution/latest"
    data_component = "results/deconvolution/latest/data"
    return (
        component,
        data_component,
        (
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            int(shape[3]),
            int(shape[4]),
            int(shape[5]),
        ),
        (
            int(chunks[0]),
            int(chunks[1]),
            int(chunks[2]),
            int(chunks[3]),
            int(chunks[4]),
            int(chunks[5]),
        ),
    )


def run_deconvolution_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> DeconvolutionSummary:
    """Run deconvolution for canonical 6D data and persist latest output.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Deconvolution parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    DeconvolutionSummary
        Summary metadata for the completed deconvolution run.

    Raises
    ------
    ValueError
        If source component is missing or incompatible.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_parameters(parameters)
    root = zarr.open_group(str(zarr_path), mode="r")
    source_component = str(normalized.get("input_source", "data")).strip() or "data"
    if source_component not in root:
        raise ValueError(
            f"Deconvolution input component '{source_component}' was not found in {zarr_path}."
        )
    source = root[source_component]
    source_shape = tuple(int(v) for v in source.shape)
    if len(source_shape) != 6:
        raise ValueError(
            "Deconvolution requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )
    voxel_xy_um, voxel_z_um = _resolve_data_voxel_sizes_um(
        root=root,
        parameters=normalized,
    )
    _emit(5, "Prepared deconvolution inputs and voxel metadata")

    component, data_component, shape_tpczyx, output_chunks = _prepare_output_array(
        zarr_path=zarr_path,
        source_component=source_component,
        parameters=normalized,
    )
    _emit(10, "Initialized latest deconvolution output dataset")

    t_count, p_count, c_count = (
        int(shape_tpczyx[0]),
        int(shape_tpczyx[1]),
        int(shape_tpczyx[2]),
    )
    selected_channels = sorted(
        {
            int(channel)
            for channel in normalized.get("channel_indices", [])
            if 0 <= int(channel) < c_count
        }
    )
    if not selected_channels:
        selected_channels = list(range(c_count))

    tasks = [
        delayed(_run_deconvolution_for_volume)(
            zarr_path=str(zarr_path),
            source_component=source_component,
            output_data_component=data_component,
            t_index=t_index,
            p_index=p_index,
            c_index=c_index,
            parameters=normalized,
            voxel_xy_um=float(voxel_xy_um),
            voxel_z_um=float(voxel_z_um),
        )
        for t_index in range(t_count)
        for p_index in range(p_count)
        for c_index in selected_channels
    ]

    total = int(len(tasks))
    if total == 0:
        register_latest_output_reference(
            zarr_path=zarr_path,
            analysis_name="deconvolution",
            component=component,
            metadata={
                "data_component": data_component,
                "volumes_processed": 0,
                "source_component": source_component,
                "output_chunks_tpczyx": list(output_chunks),
                "psf_mode": str(normalized["psf_mode"]),
            },
        )
        _emit(100, "No deconvolution tasks to run.")
        return DeconvolutionSummary(
            component=component,
            data_component=data_component,
            volumes_processed=0,
            channel_count=int(len(selected_channels)),
            psf_mode=str(normalized["psf_mode"]),
            output_chunks_tpczyx=output_chunks,
        )

    if client is None:
        _emit(15, "Running deconvolution tasks with local process scheduler")
        _ = dask.compute(*tasks, scheduler="processes")
        _emit(95, f"Completed {total} deconvolution tasks")
    else:
        from dask.distributed import as_completed

        _emit(15, f"Submitting {total} deconvolution tasks to Dask client")
        futures = client.compute(tasks)
        completed = 0
        for future in as_completed(futures):
            _ = int(future.result())
            completed += 1
            progress = 15 + int((completed / max(1, total)) * 80)
            _emit(progress, f"Processed deconvolution volume {completed}/{total}")

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="deconvolution",
        component=component,
        metadata={
            "data_component": data_component,
            "volumes_processed": total,
            "source_component": source_component,
            "output_chunks_tpczyx": list(output_chunks),
            "psf_mode": str(normalized["psf_mode"]),
            "data_xy_pixel_um": float(voxel_xy_um),
            "data_z_pixel_um": float(voxel_z_um),
            "parameters": {str(key): value for key, value in normalized.items()},
        },
    )
    _emit(100, "Deconvolution complete")
    return DeconvolutionSummary(
        component=component,
        data_component=data_component,
        volumes_processed=total,
        channel_count=int(len(selected_channels)),
        psf_mode=str(normalized["psf_mode"]),
        output_chunks_tpczyx=output_chunks,
    )
