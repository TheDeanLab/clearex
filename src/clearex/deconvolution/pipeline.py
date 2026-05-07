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

"""Chunk-parallel deconvolution workflow on canonical 6D Zarr stores."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
from io import BytesIO
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
from clearex.deconvolution.petakit import (
    run_petakit_deconvolution,
    validate_petakit_runtime,
)
from clearex.io.ome_store import (
    SOURCE_CACHE_COMPONENT,
    analysis_auxiliary_root,
    analysis_cache_data_component,
    analysis_cache_root,
    load_store_metadata,
    public_analysis_root,
)
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


@dataclass(frozen=True)
class SyntheticPsfArtifacts:
    """Synthetic PSF arrays and preview payload for one channel.

    Attributes
    ----------
    combined_psf_zyx : numpy.ndarray
        Final PSF used for deconvolution in ``(z, y, x)`` order.
    detection_psf_zyx : numpy.ndarray
        Detection PSF used to construct ``combined_psf_zyx``.
    illumination_psf_zyx : numpy.ndarray, optional
        Illumination PSF used only for light-sheet mode.
    preview_png_bytes : bytes
        PNG bytes showing PSF slices/profile preview.
    metadata : dict[str, Any]
        Lightweight metadata describing synthetic PSF generation settings.
    """

    combined_psf_zyx: np.ndarray
    detection_psf_zyx: np.ndarray
    illumination_psf_zyx: Optional[np.ndarray]
    preview_png_bytes: bytes
    metadata: dict[str, Any]


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
    mode_raw = str(normalized.get("synthetic_microscopy_mode", "widefield")).strip()
    microscopy_mode = mode_raw.lower().replace("-", "_").replace(" ", "_")
    if microscopy_mode == "lightsheet":
        microscopy_mode = "light_sheet"
    if microscopy_mode not in {"widefield", "confocal", "light_sheet"}:
        raise ValueError(
            "deconvolution synthetic_microscopy_mode must be one of "
            "'widefield', 'confocal', or 'light_sheet'."
        )
    normalized["synthetic_microscopy_mode"] = microscopy_mode

    default_illumination_wavelengths = [488.0]
    illumination_wavelengths = _as_float_list(
        normalized.get("synthetic_illumination_wavelength_nm", [])
    )
    excitation_wavelengths = _as_float_list(
        normalized.get("synthetic_excitation_nm", [])
    )
    if (
        illumination_wavelengths
        and excitation_wavelengths
        and illumination_wavelengths == default_illumination_wavelengths
        and excitation_wavelengths != default_illumination_wavelengths
    ):
        illumination_wavelengths = list(excitation_wavelengths)
    elif not illumination_wavelengths:
        illumination_wavelengths = (
            list(excitation_wavelengths)
            if excitation_wavelengths
            else list(default_illumination_wavelengths)
        )
    if not illumination_wavelengths:
        illumination_wavelengths = [488.0]
    normalized["synthetic_illumination_wavelength_nm"] = [
        float(value) for value in illumination_wavelengths
    ]
    # Backward-compatible alias used by existing configs.
    normalized["synthetic_excitation_nm"] = [
        float(value) for value in illumination_wavelengths
    ]

    illumination_na = _as_float_list(
        normalized.get("synthetic_illumination_numerical_aperture", [0.2])
    )
    if not illumination_na:
        illumination_na = [0.2]
    normalized["synthetic_illumination_numerical_aperture"] = [
        float(value) for value in illumination_na
    ]

    normalized["synthetic_emission_nm"] = _as_float_list(
        normalized.get("synthetic_emission_nm", [520.0])
    )
    default_detection_na = [0.7]
    detection_na = _as_float_list(
        normalized.get("synthetic_detection_numerical_aperture", [])
    )
    legacy_detection_na = _as_float_list(
        normalized.get("synthetic_numerical_aperture", [])
    )
    if (
        detection_na
        and legacy_detection_na
        and detection_na == default_detection_na
        and legacy_detection_na != default_detection_na
    ):
        detection_na = list(legacy_detection_na)
    elif not detection_na:
        detection_na = (
            list(legacy_detection_na)
            if legacy_detection_na
            else list(default_detection_na)
        )
    if not detection_na:
        detection_na = [0.7]
    normalized["synthetic_detection_numerical_aperture"] = [
        float(value) for value in detection_na
    ]
    # Backward-compatible alias used by existing configs.
    normalized["synthetic_numerical_aperture"] = [
        float(value) for value in detection_na
    ]

    for field_name in (
        "synthetic_illumination_wavelength_nm",
        "synthetic_illumination_numerical_aperture",
        "synthetic_emission_nm",
        "synthetic_detection_numerical_aperture",
    ):
        for value in normalized.get(field_name, []):
            if float(value) <= 0:
                raise ValueError(f"deconvolution {field_name} values must be positive.")
    if (
        normalized["synthetic_microscopy_mode"] == "light_sheet"
        and not normalized["synthetic_illumination_wavelength_nm"]
    ):
        raise ValueError(
            "deconvolution light_sheet mode requires "
            "synthetic_illumination_wavelength_nm."
        )
    if (
        normalized["synthetic_microscopy_mode"] == "light_sheet"
        and not normalized["synthetic_illumination_numerical_aperture"]
    ):
        raise ValueError(
            "deconvolution light_sheet mode requires "
            "synthetic_illumination_numerical_aperture."
        )
    if not normalized["synthetic_emission_nm"]:
        raise ValueError("deconvolution synthetic_emission_nm cannot be empty.")
    if not normalized["synthetic_detection_numerical_aperture"]:
        raise ValueError(
            "deconvolution synthetic_detection_numerical_aperture cannot be empty."
        )

    normalized["synthetic_numerical_aperture"] = _as_float_list(
        normalized.get(
            "synthetic_detection_numerical_aperture",
            normalized.get("synthetic_numerical_aperture", [0.7]),
        )
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
    store_metadata = load_store_metadata(root)
    data_attrs = (
        dict(root[SOURCE_CACHE_COMPONENT].attrs)
        if SOURCE_CACHE_COMPONENT in root
        else {}
    )

    for attrs in (data_attrs, store_metadata, root_attrs):
        voxel = attrs.get("voxel_size_um_zyx")
        if isinstance(voxel, (list, tuple)) and len(voxel) >= 3:
            return float(voxel[2]), float(voxel[0])
    for attrs in (store_metadata, root_attrs, data_attrs):
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


def _normalize_psf(psf_zyx: np.ndarray) -> np.ndarray:
    """Normalize a PSF volume to non-negative unit-sum float32.

    Parameters
    ----------
    psf_zyx : numpy.ndarray
        Candidate PSF volume in ``(z, y, x)`` order.

    Returns
    -------
    numpy.ndarray
        Normalized float32 PSF volume with sum ``1`` when possible.
    """
    out = np.asarray(psf_zyx, dtype=np.float32)
    out = np.maximum(out, 0.0).astype(np.float32, copy=False)
    total = float(np.sum(out))
    if total <= 0:
        return np.zeros_like(out, dtype=np.float32)
    return (out / total).astype(np.float32, copy=False)


def _center_crop_zyx(
    *,
    volume_zyx: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Center-crop a 3D volume to a requested shape.

    Parameters
    ----------
    volume_zyx : numpy.ndarray
        Source volume in ``(z, y, x)`` order.
    target_shape_zyx : tuple[int, int, int]
        Target shape in ``(z, y, x)`` order.

    Returns
    -------
    numpy.ndarray
        Cropped volume in ``(z, y, x)`` order.
    """
    source = np.asarray(volume_zyx)
    z_target, y_target, x_target = (
        int(target_shape_zyx[0]),
        int(target_shape_zyx[1]),
        int(target_shape_zyx[2]),
    )
    z_size, y_size, x_size = source.shape
    z_start = max(0, int((z_size - z_target) // 2))
    y_start = max(0, int((y_size - y_target) // 2))
    x_start = max(0, int((x_size - x_target) // 2))
    z_stop = min(z_size, z_start + z_target)
    y_stop = min(y_size, y_start + y_target)
    x_stop = min(x_size, x_start + x_target)
    cropped = source[z_start:z_stop, y_start:y_stop, x_start:x_stop]
    return np.asarray(cropped)


def _vectorial_detection_psf(
    *,
    size_zyx: tuple[int, int, int],
    voxel_xy_um: float,
    voxel_z_um: float,
    emission_nm: float,
    numerical_aperture: float,
    refractive_index: float,
    confocal_mode: bool,
) -> np.ndarray:
    """Generate a vectorial detection PSF with PSFmodels.

    Parameters
    ----------
    size_zyx : tuple[int, int, int]
        Target PSF size in ``(z, y, x)`` order.
    voxel_xy_um : float
        Data XY pixel size in microns.
    voxel_z_um : float
        Data Z step size in microns.
    emission_nm : float
        Emission wavelength in nanometers.
    numerical_aperture : float
        Detection numerical aperture.
    refractive_index : float
        Refractive index used for specimen/immersion values.
    confocal_mode : bool
        Whether to use confocal PSF simulation.

    Returns
    -------
    numpy.ndarray
        Detection PSF in ``(z, y, x)`` order.

    Raises
    ------
    RuntimeError
        If PSFmodels cannot be imported.
    """
    try:
        import psfmodels
    except Exception as exc:  # pragma: no cover - environment-dependent dependency
        raise RuntimeError(
            "PSFmodels is unavailable. Install with `pip install clearex[decon]`."
        ) from exc

    z_size, y_size, x_size = size_zyx
    nx = int(max(y_size, x_size))
    emission_um = float(emission_nm) / 1000.0
    if confocal_mode:
        psf = np.asarray(
            psfmodels.confocal_psf(
                z=int(z_size),
                nx=nx,
                dxy=float(voxel_xy_um),
                dz=float(voxel_z_um),
                NA=float(numerical_aperture),
                ex_wvl=float(emission_um),
                em_wvl=float(emission_um),
                ns=float(refractive_index),
                ni=float(refractive_index),
                ni0=float(refractive_index),
                normalize=True,
                model="vectorial",
            ),
            dtype=np.float32,
        )
    else:
        psf = np.asarray(
            psfmodels.make_psf(
                z=int(z_size),
                nx=nx,
                dxy=float(voxel_xy_um),
                dz=float(voxel_z_um),
                NA=float(numerical_aperture),
                wvl=float(emission_um),
                ns=float(refractive_index),
                ni=float(refractive_index),
                ni0=float(refractive_index),
                normalize=True,
                model="vectorial",
            ),
            dtype=np.float32,
        )
    cropped = _center_crop_zyx(volume_zyx=psf, target_shape_zyx=size_zyx)
    return _normalize_psf(np.asarray(cropped, dtype=np.float32))


def _render_psf_preview_png(
    *,
    combined_psf_zyx: np.ndarray,
    detection_psf_zyx: np.ndarray,
    illumination_psf_zyx: Optional[np.ndarray],
    microscopy_mode: str,
    channel_index: int,
) -> bytes:
    """Render a dark-themed PNG preview for synthetic PSF volumes.

    Parameters
    ----------
    combined_psf_zyx : numpy.ndarray
        Final PSF volume in ``(z, y, x)`` order.
    detection_psf_zyx : numpy.ndarray
        Detection PSF volume in ``(z, y, x)`` order.
    illumination_psf_zyx : numpy.ndarray, optional
        Illumination PSF volume for light-sheet mode.
    microscopy_mode : str
        Synthetic microscopy mode key.
    channel_index : int
        Channel index represented by this preview.

    Returns
    -------
    bytes
        PNG image bytes.
    """
    import matplotlib.pyplot as plt

    combined = np.asarray(combined_psf_zyx, dtype=np.float32)
    detection = np.asarray(detection_psf_zyx, dtype=np.float32)
    illumination = (
        np.asarray(illumination_psf_zyx, dtype=np.float32)
        if illumination_psf_zyx is not None
        else None
    )
    z_mid = int(combined.shape[0] // 2)
    y_mid = int(combined.shape[1] // 2)
    x_mid = int(combined.shape[2] // 2)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 6.0),
        facecolor="#0c1118",
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_facecolor("#111925")
        for spine in ax.spines.values():
            spine.set_color("#2a3442")
        ax.tick_params(colors="#9ab0ca", labelsize=8)

    axes[0, 0].imshow(combined[z_mid], cmap="magma")
    axes[0, 0].set_title("Combined XY", color="#e6edf3", fontsize=10)
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(combined[:, y_mid, :], cmap="magma", aspect="auto")
    axes[0, 1].set_title("Combined ZX", color="#e6edf3", fontsize=10)
    axes[0, 1].set_axis_off()

    axes[1, 0].imshow(detection[z_mid], cmap="viridis")
    axes[1, 0].set_title("Detection XY", color="#e6edf3", fontsize=10)
    axes[1, 0].set_axis_off()

    if illumination is not None:
        axes[1, 1].imshow(illumination[:, x_mid, :], cmap="cividis", aspect="auto")
        axes[1, 1].set_title("Illumination ZY", color="#e6edf3", fontsize=10)
        axes[1, 1].set_axis_off()
    else:
        z_axis = np.arange(combined.shape[0], dtype=np.float32)
        axes[1, 1].plot(z_axis, combined[:, y_mid, x_mid], color="#9cc6ff", linewidth=2)
        axes[1, 1].set_title("Axial Profile", color="#e6edf3", fontsize=10)
        axes[1, 1].set_xlabel("z (px)", color="#9ab0ca", fontsize=8)
        axes[1, 1].set_ylabel("intensity", color="#9ab0ca", fontsize=8)

    fig.suptitle(
        f"Synthetic PSF Preview | channel {channel_index} | {microscopy_mode}",
        color="#f0f5ff",
        fontsize=11,
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


def _generate_synthetic_psf_artifacts(
    *,
    params: Mapping[str, Any],
    channel_index: int,
    voxel_xy_um: float,
    voxel_z_um: float,
) -> SyntheticPsfArtifacts:
    """Generate synthetic PSF arrays and preview bytes for one channel.

    Parameters
    ----------
    params : mapping[str, Any]
        Normalized deconvolution parameters.
    channel_index : int
        Channel index.
    voxel_xy_um : float
        Source XY voxel size in microns.
    voxel_z_um : float
        Source Z voxel size in microns.

    Returns
    -------
    SyntheticPsfArtifacts
        Generated PSF arrays and preview payload.

    Raises
    ------
    ValueError
        If required synthetic parameters are missing.
    """
    emission_nm = float(
        _broadcast_channel_value(
            params["synthetic_emission_nm"],
            channel_index=channel_index,
            field_name="deconvolution synthetic_emission_nm",
        )
    )
    detection_na = float(
        _broadcast_channel_value(
            params["synthetic_detection_numerical_aperture"],
            channel_index=channel_index,
            field_name="deconvolution synthetic_detection_numerical_aperture",
        )
    )
    microscopy_mode = str(params.get("synthetic_microscopy_mode", "widefield"))
    psf_shape = tuple(int(v) for v in params["synthetic_psf_size_zyx"])
    refractive_index = float(params["synthetic_refractive_index"])
    confocal_mode = microscopy_mode == "confocal"
    detection_psf = _vectorial_detection_psf(
        size_zyx=(psf_shape[0], psf_shape[1], psf_shape[2]),
        voxel_xy_um=float(voxel_xy_um),
        voxel_z_um=float(voxel_z_um),
        emission_nm=float(emission_nm),
        numerical_aperture=float(detection_na),
        refractive_index=float(refractive_index),
        confocal_mode=bool(confocal_mode),
    )

    illumination_psf: Optional[np.ndarray] = None
    combined_psf = detection_psf
    metadata: dict[str, Any] = {
        "microscopy_mode": microscopy_mode,
        "emission_nm": float(emission_nm),
        "detection_numerical_aperture": float(detection_na),
        "voxel_xy_um": float(voxel_xy_um),
        "voxel_z_um": float(voxel_z_um),
    }

    if microscopy_mode == "light_sheet":
        illumination_nm = float(
            _broadcast_channel_value(
                params["synthetic_illumination_wavelength_nm"],
                channel_index=channel_index,
                field_name="deconvolution synthetic_illumination_wavelength_nm",
            )
        )
        illumination_na = float(
            _broadcast_channel_value(
                params["synthetic_illumination_numerical_aperture"],
                channel_index=channel_index,
                field_name="deconvolution synthetic_illumination_numerical_aperture",
            )
        )
        # Generate illumination PSF with swapped z/y extent so transposition lands on
        # canonical (z, y, x) with the long axis aligned to Y.
        illumination_native = _vectorial_detection_psf(
            size_zyx=(psf_shape[1], psf_shape[0], psf_shape[2]),
            voxel_xy_um=float(voxel_xy_um),
            voxel_z_um=float(voxel_z_um),
            emission_nm=float(illumination_nm),
            numerical_aperture=float(illumination_na),
            refractive_index=float(refractive_index),
            confocal_mode=False,
        )
        illumination_psf = np.transpose(illumination_native, (1, 0, 2))
        illumination_psf = _normalize_psf(illumination_psf)
        combined_psf = _normalize_psf(illumination_psf * detection_psf)
        metadata["illumination_wavelength_nm"] = float(illumination_nm)
        metadata["illumination_numerical_aperture"] = float(illumination_na)

    preview_png = _render_psf_preview_png(
        combined_psf_zyx=combined_psf,
        detection_psf_zyx=detection_psf,
        illumination_psf_zyx=illumination_psf,
        microscopy_mode=microscopy_mode,
        channel_index=int(channel_index),
    )
    return SyntheticPsfArtifacts(
        combined_psf_zyx=np.asarray(combined_psf, dtype=np.float32),
        detection_psf_zyx=np.asarray(detection_psf, dtype=np.float32),
        illumination_psf_zyx=(
            np.asarray(illumination_psf, dtype=np.float32)
            if illumination_psf is not None
            else None
        ),
        preview_png_bytes=preview_png,
        metadata=metadata,
    )


def generate_synthetic_psf_preview(
    *,
    parameters: Mapping[str, Any],
    channel_index: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    """Generate a synthetic-PSF preview PNG from deconvolution parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate deconvolution parameter mapping.
    channel_index : int, default=0
        Channel index to preview.

    Returns
    -------
    tuple[bytes, dict[str, Any]]
        ``(preview_png_bytes, metadata)``.

    Raises
    ------
    ValueError
        If parameters are incompatible with synthetic PSF mode.
    """
    normalized = _normalize_parameters(parameters)
    if str(normalized.get("psf_mode", "measured")) != "synthetic":
        raise ValueError(
            "Synthetic preview is only available when psf_mode='synthetic'."
        )
    voxel_xy_um = float(normalized.get("data_xy_pixel_um", 0.0))
    voxel_z_um = float(normalized.get("data_z_pixel_um", 0.0))
    if voxel_xy_um <= 0 or voxel_z_um <= 0:
        raise ValueError(
            "Synthetic preview requires positive data_xy_pixel_um and data_z_pixel_um."
        )
    artifacts = _generate_synthetic_psf_artifacts(
        params=normalized,
        channel_index=int(channel_index),
        voxel_xy_um=float(voxel_xy_um),
        voxel_z_um=float(voxel_z_um),
    )
    return artifacts.preview_png_bytes, dict(artifacts.metadata)


def _select_psf_for_channel(
    *,
    params: Mapping[str, Any],
    channel_index: int,
    voxel_xy_um: float,
    voxel_z_um: float,
    temp_dir: Path,
    zarr_path: Union[str, Path],
    synthetic_psf_components: Optional[Mapping[int, str]] = None,
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
    zarr_path : str or pathlib.Path
        Zarr store path used to load persisted synthetic PSF assets.
    synthetic_psf_components : mapping[int, str], optional
        Channel-to-component mapping for persisted synthetic PSFs.

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

    psf_data: np.ndarray
    if (
        synthetic_psf_components is not None
        and int(channel_index) in synthetic_psf_components
    ):
        component = str(synthetic_psf_components[int(channel_index)])
        root = zarr.open_group(str(zarr_path), mode="r")
        try:
            psf_data = np.asarray(root[component], dtype=np.float32)
        except KeyError:
            artifacts = _generate_synthetic_psf_artifacts(
                params=params,
                channel_index=int(channel_index),
                voxel_xy_um=float(voxel_xy_um),
                voxel_z_um=float(voxel_z_um),
            )
            psf_data = artifacts.combined_psf_zyx
    else:
        artifacts = _generate_synthetic_psf_artifacts(
            params=params,
            channel_index=int(channel_index),
            voxel_xy_um=float(voxel_xy_um),
            voxel_z_um=float(voxel_z_um),
        )
        psf_data = artifacts.combined_psf_zyx
    psf_path = temp_dir / f"synthetic_psf_ch{int(channel_index):02d}.tif"
    tifffile.imwrite(str(psf_path), psf_data, photometric="minisblack")
    return psf_path.resolve(), float(voxel_z_um)


def _persist_synthetic_psf_assets(
    *,
    zarr_path: Union[str, Path],
    latest_component: str,
    params: Mapping[str, Any],
    selected_channels: Sequence[int],
    voxel_xy_um: float,
    voxel_z_um: float,
) -> dict[int, str]:
    """Generate and persist synthetic PSF assets in the decon latest group.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis-store path.
    latest_component : str
        Decon latest group component path.
    params : mapping[str, Any]
        Normalized deconvolution parameters.
    selected_channels : sequence[int]
        Channels selected for deconvolution.
    voxel_xy_um : float
        Source XY voxel size in microns.
    voxel_z_um : float
        Source Z voxel size in microns.

    Returns
    -------
    dict[int, str]
        Mapping from channel index to persisted combined-PSF component path.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    latest_group = root[latest_component]
    if "synthetic_psf" in latest_group:
        del latest_group["synthetic_psf"]
    synthetic_group = latest_group.create_group("synthetic_psf")
    synthetic_group.attrs.update(
        {
            "microscopy_mode": str(
                params.get("synthetic_microscopy_mode", "widefield")
            ),
            "storage_policy": "latest_only",
        }
    )

    components: dict[int, str] = {}
    for channel_index in sorted({int(value) for value in selected_channels}):
        artifacts = _generate_synthetic_psf_artifacts(
            params=params,
            channel_index=int(channel_index),
            voxel_xy_um=float(voxel_xy_um),
            voxel_z_um=float(voxel_z_um),
        )
        channel_group = synthetic_group.create_group(f"ch{int(channel_index):02d}")
        channel_group.create_array(
            name="combined_psf_zyx",
            data=np.asarray(artifacts.combined_psf_zyx, dtype=np.float32),
            overwrite=True,
        )
        channel_group.create_array(
            name="detection_psf_zyx",
            data=np.asarray(artifacts.detection_psf_zyx, dtype=np.float32),
            overwrite=True,
        )
        if artifacts.illumination_psf_zyx is not None:
            channel_group.create_array(
                name="illumination_psf_zyx",
                data=np.asarray(artifacts.illumination_psf_zyx, dtype=np.float32),
                overwrite=True,
            )
        preview_bytes = np.frombuffer(artifacts.preview_png_bytes, dtype=np.uint8)
        channel_group.create_array(
            name="preview_png",
            data=preview_bytes,
            overwrite=True,
        )
        channel_group.attrs.update(
            {
                "channel_index": int(channel_index),
                "metadata": {
                    str(key): value for key, value in dict(artifacts.metadata).items()
                },
                "preview_format": "png",
            }
        )
        components[int(channel_index)] = (
            f"{latest_component}/synthetic_psf/ch{int(channel_index):02d}/combined_psf_zyx"
        )
    return components


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
    synthetic_psf_components: Optional[Mapping[int, str]] = None,
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
    synthetic_psf_components : mapping[int, str], optional
        Optional channel-to-component mapping for persisted synthetic PSFs.

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
            zarr_path=zarr_path,
            synthetic_psf_components=synthetic_psf_components,
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

    cache_component = analysis_cache_data_component("deconvolution")
    cache_root = analysis_cache_root("deconvolution")
    auxiliary_root = analysis_auxiliary_root("deconvolution")
    if cache_root in root:
        del root[cache_root]
    if auxiliary_root in root:
        del root[auxiliary_root]
    latest = root.require_group(cache_root)
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
    root.require_group(auxiliary_root).attrs.update(
        {
            "storage_policy": "latest_only",
            "source_component": str(source_component),
            "parameters": {str(key): value for key, value in dict(parameters).items()},
            "run_id": None,
            "data_component": cache_component,
        }
    )
    component = public_analysis_root("deconvolution")
    data_component = cache_component
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
    validate_petakit_runtime(mcc_mode=bool(normalized["mcc_mode"]))
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

    synthetic_psf_components: Optional[dict[int, str]] = None
    if str(normalized.get("psf_mode", "measured")) == "synthetic":
        _emit(12, "Generating vectorial synthetic PSF assets")
        synthetic_psf_components = _persist_synthetic_psf_assets(
            zarr_path=zarr_path,
            latest_component=analysis_auxiliary_root("deconvolution"),
            params=normalized,
            selected_channels=selected_channels,
            voxel_xy_um=float(voxel_xy_um),
            voxel_z_um=float(voxel_z_um),
        )
        _emit(14, "Stored synthetic PSFs and previews in data_store.ome.zarr")

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
            synthetic_psf_components=synthetic_psf_components,
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
                "synthetic_psf_components": (
                    {str(key): value for key, value in synthetic_psf_components.items()}
                    if synthetic_psf_components
                    else {}
                ),
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
            "synthetic_psf_components": (
                {str(key): value for key, value in synthetic_psf_components.items()}
                if synthetic_psf_components
                else {}
            ),
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
