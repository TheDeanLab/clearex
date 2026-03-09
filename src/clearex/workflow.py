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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union


ChunkSpec = Optional[Union[int, Tuple[int, ...]]]
ZarrAxisSpec = Tuple[int, int, int, int, int, int]
ZarrPyramidAxisSpec = Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
]

PTCZYX_AXES = ("p", "t", "c", "z", "y", "x")
TPCZYX_AXES = ("t", "p", "c", "z", "y", "x")
_PTCZYX_TO_TPCZYX_INDICES = (1, 0, 2, 3, 4, 5)
ANALYSIS_OPERATION_ORDER = (
    "deconvolution",
    "particle_detection",
    "registration",
    "visualization",
)

DEFAULT_ZARR_CHUNKS_PTCZYX: ZarrAxisSpec = (1, 1, 1, 256, 256, 256)
DEFAULT_ZARR_PYRAMID_PTCZYX: ZarrPyramidAxisSpec = (
    (1,),
    (1,),
    (1,),
    (1, 2, 4, 8),
    (1, 2, 4, 8),
    (1, 2, 4, 8),
)

DEFAULT_ANALYSIS_OPERATION_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "deconvolution": {
        "execution_order": 1,
        "input_source": "data",
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 2.0,
        "psf_mode": "measured",
        "channel_indices": [],
        "measured_psf_paths": [],
        "measured_psf_xy_um": [],
        "measured_psf_z_um": [],
        "synthetic_microscopy_mode": "widefield",
        "synthetic_illumination_wavelength_nm": [488.0],
        "synthetic_illumination_numerical_aperture": [0.2],
        "synthetic_excitation_nm": [488.0],
        "synthetic_emission_nm": [520.0],
        "synthetic_detection_numerical_aperture": [0.7],
        "synthetic_numerical_aperture": [0.7],
        "synthetic_refractive_index": 1.33,
        "synthetic_psf_size_zyx": [65, 129, 129],
        "data_xy_pixel_um": 0.0,
        "data_z_pixel_um": 0.0,
        "hann_window_bounds": [0.8, 1.0],
        "wiener_alpha": 0.005,
        "background": 110.0,
        "decon_iterations": 2,
        "otf_cum_thresh": 0.6,
        "large_file": False,
        "block_size_zyx": [256, 256, 256],
        "batch_size_zyx": [1024, 1024, 1024],
        "save_16bit": True,
        "save_zarr": False,
        "gpu_job": False,
        "debug": False,
        "cpus_per_task": 2,
        "mcc_mode": True,
        "config_file": "",
        "gpu_config_file": "",
    },
    "particle_detection": {
        "execution_order": 2,
        "input_source": "data",
        "channel_index": 0,
        "chunk_basis": "3d",
        "detect_2d_per_slice": True,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 1.5,
        "bg_sigma": 20.0,
        "fwhm_px": 3.0,
        "sigma_min_factor": 1.0,
        "sigma_max_factor": 3.0,
        "threshold": 0.1,
        "overlap": 0.5,
        "exclude_border": 5,
        "eliminate_insignificant_particles": False,
        "remove_close_particles": False,
        "min_distance_sigma": 10.0,
    },
    "registration": {
        "execution_order": 3,
        "input_source": "data",
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": True,
        "overlap_zyx": [8, 32, 32],
        "memory_overhead_factor": 2.5,
    },
    "visualization": {
        "execution_order": 4,
        "input_source": "data",
        "chunk_basis": "2d",
        "detect_2d_per_slice": True,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 1.0,
        "show_all_positions": False,
        "position_index": 0,
        "use_multiscale": True,
        "overlay_particle_detections": True,
        "particle_detection_component": "results/particle_detection/latest/detections",
        "launch_mode": "auto",
    },
}


def default_analysis_operation_parameters() -> Dict[str, Dict[str, Any]]:
    """Return independent default analysis operation parameters.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, dict[str, Any]]
        Deep-copied default analysis parameter mapping keyed by analysis name.
    """
    return deepcopy(DEFAULT_ANALYSIS_OPERATION_PARAMETERS)


def _normalize_common_operation_parameters(
    operation_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize common analysis-operation runtime parameters.

    Parameters
    ----------
    operation_name : str
        Analysis operation name used for validation error context.
    params : dict[str, Any]
        Candidate operation parameters.

    Returns
    -------
    dict[str, Any]
        Normalized operation parameter mapping.

    Raises
    ------
    ValueError
        If common operation fields are malformed.
    """
    normalized = dict(params)
    normalized["chunk_basis"] = str(normalized.get("chunk_basis", "3d")).strip() or "3d"
    normalized["detect_2d_per_slice"] = bool(
        normalized.get("detect_2d_per_slice", False)
    )
    normalized["use_map_overlap"] = bool(normalized.get("use_map_overlap", False))

    overlap_value = normalized.get("overlap_zyx", [0, 0, 0])
    if not isinstance(overlap_value, (tuple, list)) or len(overlap_value) != 3:
        raise ValueError(
            f"{operation_name} overlap_zyx must define three integers in (z, y, x) order."
        )
    normalized["overlap_zyx"] = [max(0, int(v)) for v in overlap_value]

    memory_overhead = float(normalized.get("memory_overhead_factor", 1.0))
    if memory_overhead <= 0:
        raise ValueError(
            f"{operation_name} memory_overhead_factor must be greater than zero."
        )
    normalized["memory_overhead_factor"] = memory_overhead

    execution_order = int(normalized.get("execution_order", 1))
    if execution_order < 1:
        raise ValueError(f"{operation_name} execution_order must be at least one.")
    normalized["execution_order"] = execution_order

    input_source = str(normalized.get("input_source", "data")).strip() or "data"
    normalized["input_source"] = input_source
    return normalized


def _normalize_parameter_string_list(value: Any) -> list[str]:
    """Normalize parameter value into a list of non-empty strings.

    Parameters
    ----------
    value : Any
        Candidate value. Accepts strings, sequences, and ``None``.

    Returns
    -------
    list[str]
        Normalized list of non-empty strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.replace("\n", ",")
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, (tuple, list)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _normalize_parameter_float_list(value: Any) -> list[float]:
    """Normalize parameter value into a float list.

    Parameters
    ----------
    value : Any
        Candidate value. Accepts strings, sequences, and scalars.

    Returns
    -------
    list[float]
        Parsed float list.
    """
    out: list[float] = []
    for item in _normalize_parameter_string_list(value):
        out.append(float(item))
    return out


def _normalize_parameter_int_triplet(
    value: Any,
    *,
    field_name: str,
    default: tuple[int, int, int],
) -> list[int]:
    """Normalize parameter value into a positive integer triplet.

    Parameters
    ----------
    value : Any
        Candidate value to parse.
    field_name : str
        Field name used in validation errors.
    default : tuple[int, int, int]
        Default triplet when value is empty.

    Returns
    -------
    list[int]
        Positive integer triplet.

    Raises
    ------
    ValueError
        If value does not define exactly three positive integers.
    """
    tokens = _normalize_parameter_string_list(value)
    if not tokens:
        return [int(default[0]), int(default[1]), int(default[2])]
    if len(tokens) != 3:
        raise ValueError(f"{field_name} must define exactly three values.")
    parsed = [int(tokens[0]), int(tokens[1]), int(tokens[2])]
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{field_name} values must be greater than zero.")
    return parsed


def _normalize_deconvolution_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize deconvolution runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate deconvolution parameters.

    Returns
    -------
    dict[str, Any]
        Normalized deconvolution parameter mapping.

    Raises
    ------
    ValueError
        If required values are invalid.
    """
    normalized = _normalize_common_operation_parameters("deconvolution", params)

    psf_mode = str(normalized.get("psf_mode", "measured")).strip().lower()
    if psf_mode not in {"measured", "synthetic"}:
        raise ValueError("deconvolution psf_mode must be 'measured' or 'synthetic'.")
    normalized["psf_mode"] = psf_mode

    normalized["channel_indices"] = [
        max(0, int(value))
        for value in _normalize_parameter_float_list(
            normalized.get("channel_indices", [])
        )
    ]
    normalized["measured_psf_paths"] = _normalize_parameter_string_list(
        normalized.get("measured_psf_paths", [])
    )
    normalized["measured_psf_xy_um"] = _normalize_parameter_float_list(
        normalized.get("measured_psf_xy_um", [])
    )
    normalized["measured_psf_z_um"] = _normalize_parameter_float_list(
        normalized.get("measured_psf_z_um", [])
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

    default_illumination_wavelengths = [
        float(value)
        for value in DEFAULT_ANALYSIS_OPERATION_PARAMETERS["deconvolution"].get(
            "synthetic_illumination_wavelength_nm",
            [488.0],
        )
    ]
    illumination_wavelengths = _normalize_parameter_float_list(
        normalized.get("synthetic_illumination_wavelength_nm", [])
    )
    excitation_wavelengths = _normalize_parameter_float_list(
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
    # Compatibility alias used by existing configs.
    normalized["synthetic_excitation_nm"] = [
        float(value) for value in illumination_wavelengths
    ]

    normalized["synthetic_illumination_numerical_aperture"] = [
        float(value)
        for value in _normalize_parameter_float_list(
            normalized.get("synthetic_illumination_numerical_aperture", [0.2])
        )
    ]
    normalized["synthetic_excitation_nm"] = _normalize_parameter_float_list(
        normalized.get("synthetic_excitation_nm", [488.0])
    )
    normalized["synthetic_emission_nm"] = _normalize_parameter_float_list(
        normalized.get("synthetic_emission_nm", [520.0])
    )
    default_detection_na = [
        float(value)
        for value in DEFAULT_ANALYSIS_OPERATION_PARAMETERS["deconvolution"].get(
            "synthetic_detection_numerical_aperture",
            [0.7],
        )
    ]
    detection_na = _normalize_parameter_float_list(
        normalized.get("synthetic_detection_numerical_aperture", [])
    )
    legacy_detection_na = _normalize_parameter_float_list(
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
    # Compatibility alias used by existing configs.
    normalized["synthetic_numerical_aperture"] = [
        float(value) for value in detection_na
    ]

    for field_name in (
        "synthetic_illumination_wavelength_nm",
        "synthetic_emission_nm",
        "synthetic_detection_numerical_aperture",
        "synthetic_illumination_numerical_aperture",
    ):
        for value in normalized.get(field_name, []):
            if float(value) <= 0:
                raise ValueError(f"deconvolution {field_name} values must be positive.")

    if (
        normalized["synthetic_microscopy_mode"] == "light_sheet"
        and not normalized["synthetic_illumination_wavelength_nm"]
    ):
        raise ValueError(
            "deconvolution light_sheet mode requires synthetic_illumination_wavelength_nm."
        )
    if (
        normalized["synthetic_microscopy_mode"] == "light_sheet"
        and not normalized["synthetic_illumination_numerical_aperture"]
    ):
        raise ValueError(
            "deconvolution light_sheet mode requires "
            "synthetic_illumination_numerical_aperture."
        )
    normalized["synthetic_refractive_index"] = float(
        normalized.get("synthetic_refractive_index", 1.33)
    )
    if normalized["synthetic_refractive_index"] <= 0:
        raise ValueError("deconvolution synthetic_refractive_index must be positive.")

    normalized["synthetic_psf_size_zyx"] = _normalize_parameter_int_triplet(
        normalized.get("synthetic_psf_size_zyx", [65, 129, 129]),
        field_name="deconvolution synthetic_psf_size_zyx",
        default=(65, 129, 129),
    )
    hann_bounds = _normalize_parameter_float_list(
        normalized.get("hann_window_bounds", [0.8, 1.0])
    )
    if len(hann_bounds) != 2:
        raise ValueError("deconvolution hann_window_bounds must define two values.")
    if hann_bounds[0] <= 0 or hann_bounds[1] <= 0 or hann_bounds[0] > hann_bounds[1]:
        raise ValueError(
            "deconvolution hann_window_bounds must satisfy 0 < low <= high."
        )
    normalized["hann_window_bounds"] = [float(hann_bounds[0]), float(hann_bounds[1])]

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
    if normalized["data_xy_pixel_um"] < 0 or normalized["data_z_pixel_um"] < 0:
        raise ValueError("deconvolution data pixel sizes cannot be negative.")

    normalized["large_file"] = bool(normalized.get("large_file", False))
    normalized["save_16bit"] = bool(normalized.get("save_16bit", True))
    normalized["save_zarr"] = bool(normalized.get("save_zarr", False))
    normalized["gpu_job"] = bool(normalized.get("gpu_job", False))
    normalized["debug"] = bool(normalized.get("debug", False))
    normalized["mcc_mode"] = bool(normalized.get("mcc_mode", True))
    normalized["cpus_per_task"] = max(1, int(normalized.get("cpus_per_task", 2)))
    normalized["config_file"] = str(normalized.get("config_file", "")).strip()
    normalized["gpu_config_file"] = str(normalized.get("gpu_config_file", "")).strip()
    normalized["block_size_zyx"] = _normalize_parameter_int_triplet(
        normalized.get("block_size_zyx", [256, 256, 256]),
        field_name="deconvolution block_size_zyx",
        default=(256, 256, 256),
    )
    normalized["batch_size_zyx"] = _normalize_parameter_int_triplet(
        normalized.get("batch_size_zyx", [1024, 1024, 1024]),
        field_name="deconvolution batch_size_zyx",
        default=(1024, 1024, 1024),
    )
    return normalized


def _normalize_particle_detection_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize particle-detection runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate particle-detection parameters.

    Returns
    -------
    dict[str, Any]
        Normalized particle-detection parameters.

    Raises
    ------
    ValueError
        If required values are invalid.
    """
    normalized = _normalize_common_operation_parameters("particle_detection", params)

    normalized["channel_index"] = max(0, int(normalized.get("channel_index", 0)))
    normalized["detect_2d_per_slice"] = bool(
        normalized.get("detect_2d_per_slice", True)
    )

    normalized["bg_sigma"] = float(normalized.get("bg_sigma", 20.0))
    normalized["fwhm_px"] = float(normalized.get("fwhm_px", 3.0))
    normalized["sigma_min_factor"] = float(normalized.get("sigma_min_factor", 1.0))
    normalized["sigma_max_factor"] = float(normalized.get("sigma_max_factor", 3.0))
    normalized["threshold"] = float(normalized.get("threshold", 0.1))
    normalized["overlap"] = float(normalized.get("overlap", 0.5))
    normalized["exclude_border"] = max(0, int(normalized.get("exclude_border", 5)))
    normalized["eliminate_insignificant_particles"] = bool(
        normalized.get("eliminate_insignificant_particles", False)
    )
    normalized["remove_close_particles"] = bool(
        normalized.get("remove_close_particles", False)
    )
    normalized["min_distance_sigma"] = float(normalized.get("min_distance_sigma", 10.0))
    if normalized["min_distance_sigma"] < 0:
        raise ValueError("particle_detection min_distance_sigma cannot be negative.")

    return normalized


def _normalize_visualization_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize visualization runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate visualization parameters.

    Returns
    -------
    dict[str, Any]
        Normalized visualization parameters.

    Raises
    ------
    ValueError
        If launch mode is unsupported.
    """
    normalized = _normalize_common_operation_parameters("visualization", params)
    normalized["show_all_positions"] = bool(
        normalized.get("show_all_positions", False)
    )
    normalized["position_index"] = max(0, int(normalized.get("position_index", 0)))
    normalized["use_multiscale"] = bool(normalized.get("use_multiscale", True))
    normalized["overlay_particle_detections"] = bool(
        normalized.get("overlay_particle_detections", True)
    )
    normalized["particle_detection_component"] = (
        str(
            normalized.get(
                "particle_detection_component",
                "results/particle_detection/latest/detections",
            )
        ).strip()
        or "results/particle_detection/latest/detections"
    )
    launch_mode = str(normalized.get("launch_mode", "auto")).strip().lower() or "auto"
    if launch_mode not in {"auto", "in_process", "subprocess"}:
        raise ValueError(
            "visualization launch_mode must be one of auto, in_process, subprocess."
        )
    normalized["launch_mode"] = launch_mode
    return normalized


def normalize_analysis_operation_parameters(
    parameters: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Normalize analysis operation parameter mappings.

    Parameters
    ----------
    parameters : dict[str, dict[str, Any]], optional
        Candidate parameter mapping keyed by analysis operation name.

    Returns
    -------
    dict[str, dict[str, Any]]
        Normalized mapping merged with defaults.

    Raises
    ------
    ValueError
        If known operation parameters are malformed.
    """
    merged = default_analysis_operation_parameters()
    if parameters is None:
        return merged

    for key, value in parameters.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"Analysis operation parameters for '{key}' must be a dictionary."
            )
        merged[str(key)] = {**merged.get(str(key), {}), **value}

    for operation_name in ANALYSIS_OPERATION_ORDER:
        if operation_name not in merged:
            continue
        if operation_name == "deconvolution":
            merged[operation_name] = _normalize_deconvolution_parameters(
                merged[operation_name]
            )
        elif operation_name == "particle_detection":
            merged[operation_name] = _normalize_particle_detection_parameters(
                merged[operation_name]
            )
        elif operation_name == "visualization":
            merged[operation_name] = _normalize_visualization_parameters(
                merged[operation_name]
            )
        else:
            merged[operation_name] = _normalize_common_operation_parameters(
                operation_name,
                merged[operation_name],
            )
    return merged


def selected_analysis_operations(
    *,
    deconvolution: bool,
    particle_detection: bool,
    registration: bool,
    visualization: bool,
) -> Tuple[str, ...]:
    """Collect enabled analysis operation names.

    Parameters
    ----------
    deconvolution : bool
        Whether deconvolution is enabled.
    particle_detection : bool
        Whether particle detection is enabled.
    registration : bool
        Whether registration is enabled.
    visualization : bool
        Whether visualization is enabled.

    Returns
    -------
    tuple[str, ...]
        Selected operations in canonical declaration order.
    """
    selected: list[str] = []
    if deconvolution:
        selected.append("deconvolution")
    if particle_detection:
        selected.append("particle_detection")
    if registration:
        selected.append("registration")
    if visualization:
        selected.append("visualization")
    return tuple(selected)


def resolve_analysis_execution_sequence(
    *,
    deconvolution: bool,
    particle_detection: bool,
    registration: bool,
    visualization: bool,
    analysis_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, ...]:
    """Resolve selected analyses into execution order.

    Parameters
    ----------
    deconvolution : bool
        Whether deconvolution is enabled.
    particle_detection : bool
        Whether particle detection is enabled.
    registration : bool
        Whether registration is enabled.
    visualization : bool
        Whether visualization is enabled.
    analysis_parameters : dict[str, dict[str, Any]], optional
        Candidate per-operation parameter mapping. The ``execution_order``
        field in each selected operation controls ordering.

    Returns
    -------
    tuple[str, ...]
        Selected operation names sorted by ``execution_order`` and then by
        canonical declaration order as a stable tie-breaker.
    """
    normalized = normalize_analysis_operation_parameters(analysis_parameters)
    selected = selected_analysis_operations(
        deconvolution=deconvolution,
        particle_detection=particle_detection,
        registration=registration,
        visualization=visualization,
    )
    index_map = {name: idx for idx, name in enumerate(ANALYSIS_OPERATION_ORDER)}
    return tuple(
        sorted(
            selected,
            key=lambda name: (
                int(
                    normalized.get(name, {}).get("execution_order", index_map[name] + 1)
                ),
                int(index_map[name]),
            ),
        )
    )


def _normalize_positive_sequence(
    values: Sequence[int], *, field_name: str
) -> Tuple[int, ...]:
    """Normalize a sequence into positive integers.

    Parameters
    ----------
    values : sequence of int
        Candidate values to normalize.
    field_name : str
        Human-readable field name for validation errors.

    Returns
    -------
    tuple[int, ...]
        Normalized integer tuple.

    Raises
    ------
    ValueError
        If the sequence is empty, contains non-integers, or contains
        non-positive values.
    """
    if not values:
        raise ValueError(f"{field_name} cannot be empty.")

    normalized: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integers.") from exc
        if parsed <= 0:
            raise ValueError(f"{field_name} values must be greater than zero.")
        normalized.append(parsed)

    return tuple(normalized)


def _normalize_ptczyx_chunks(chunks_ptczyx: Sequence[int]) -> ZarrAxisSpec:
    """Normalize chunk sizes in ``(p, t, c, z, y, x)`` axis order.

    Parameters
    ----------
    chunks_ptczyx : sequence of int
        Chunk sizes in ``(p, t, c, z, y, x)`` order.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Normalized chunk tuple.

    Raises
    ------
    ValueError
        If chunk count does not match six axes or values are invalid.
    """
    if len(chunks_ptczyx) != len(PTCZYX_AXES):
        axes_text = ", ".join(PTCZYX_AXES)
        raise ValueError(
            f"Zarr chunk sizes must define exactly six axes in ({axes_text}) order."
        )
    normalized = _normalize_positive_sequence(
        chunks_ptczyx,
        field_name="Zarr chunk sizes",
    )
    return (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
        normalized[4],
        normalized[5],
    )


def _normalize_ptczyx_pyramid(
    pyramid_ptczyx: Sequence[Sequence[int]],
) -> ZarrPyramidAxisSpec:
    """Normalize pyramid factors in ``(p, t, c, z, y, x)`` axis order.

    Parameters
    ----------
    pyramid_ptczyx : sequence of sequence of int
        Per-axis downsampling factors in ``(p, t, c, z, y, x)`` order.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Normalized pyramid factors per axis.

    Raises
    ------
    ValueError
        If axis count is not six, factors are invalid, or any axis factors do
        not start with ``1``.
    """
    if len(pyramid_ptczyx) != len(PTCZYX_AXES):
        axes_text = ", ".join(PTCZYX_AXES)
        raise ValueError(
            f"Zarr pyramid factors must define exactly six axes in ({axes_text}) order."
        )

    normalized_axes: list[Tuple[int, ...]] = []
    for axis_name, axis_levels in zip(PTCZYX_AXES, pyramid_ptczyx, strict=False):
        normalized_levels = _normalize_positive_sequence(
            axis_levels,
            field_name=f"Zarr pyramid factors for axis '{axis_name}'",
        )
        if normalized_levels[0] != 1:
            raise ValueError(
                f"Zarr pyramid factors for axis '{axis_name}' must start with 1."
            )
        normalized_axes.append(normalized_levels)

    return (
        normalized_axes[0],
        normalized_axes[1],
        normalized_axes[2],
        normalized_axes[3],
        normalized_axes[4],
        normalized_axes[5],
    )


def parse_pyramid_levels(levels: Optional[str], *, axis_name: str) -> Tuple[int, ...]:
    """Parse comma-separated pyramid factors for a single axis.

    Parameters
    ----------
    levels : str, optional
        Comma-separated factors such as ``"1,2,4,8"``.
    axis_name : str
        Axis label used in validation messages.

    Returns
    -------
    tuple[int, ...]
        Parsed positive integer factors.

    Raises
    ------
    ValueError
        If input is missing, contains invalid integers, includes non-positive
        values, or does not start with ``1``.
    """
    if levels is None:
        raise ValueError(f"Zarr pyramid factors for axis '{axis_name}' are required.")

    text = levels.strip()
    if not text:
        raise ValueError(f"Zarr pyramid factors for axis '{axis_name}' are required.")

    try:
        parsed = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(
            f"Zarr pyramid factors for axis '{axis_name}' must be integers."
        ) from exc

    normalized = _normalize_positive_sequence(
        parsed,
        field_name=f"Zarr pyramid factors for axis '{axis_name}'",
    )
    if normalized[0] != 1:
        raise ValueError(
            f"Zarr pyramid factors for axis '{axis_name}' must start with 1."
        )
    return normalized


def format_pyramid_levels(levels: Sequence[int]) -> str:
    """Format pyramid factors for one axis.

    Parameters
    ----------
    levels : sequence of int
        Pyramid factors.

    Returns
    -------
    str
        Comma-separated factors.

    Raises
    ------
    ValueError
        If factors are empty or invalid.
    """
    normalized = _normalize_positive_sequence(
        levels,
        field_name="Zarr pyramid factors",
    )
    return ",".join(str(value) for value in normalized)


def to_tpczyx_chunks(chunks_ptczyx: Sequence[int]) -> ZarrAxisSpec:
    """Convert chunk sizes from ``(p, t, c, z, y, x)`` to ``(t, p, c, z, y, x)``.

    Parameters
    ----------
    chunks_ptczyx : sequence of int
        Chunk sizes in ``(p, t, c, z, y, x)`` order.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Chunk sizes in ``(t, p, c, z, y, x)`` order.

    Raises
    ------
    ValueError
        If chunk values are invalid.
    """
    normalized = _normalize_ptczyx_chunks(chunks_ptczyx)
    reordered = tuple(normalized[index] for index in _PTCZYX_TO_TPCZYX_INDICES)
    return (
        reordered[0],
        reordered[1],
        reordered[2],
        reordered[3],
        reordered[4],
        reordered[5],
    )


def to_tpczyx_pyramid(pyramid_ptczyx: Sequence[Sequence[int]]) -> ZarrPyramidAxisSpec:
    """Convert pyramid factors from ``(p, t, c, z, y, x)`` to ``(t, p, c, z, y, x)``.

    Parameters
    ----------
    pyramid_ptczyx : sequence of sequence of int
        Pyramid factors per axis in ``(p, t, c, z, y, x)`` order.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Pyramid factors in ``(t, p, c, z, y, x)`` order.

    Raises
    ------
    ValueError
        If factors are invalid.
    """
    normalized = _normalize_ptczyx_pyramid(pyramid_ptczyx)
    reordered = tuple(normalized[index] for index in _PTCZYX_TO_TPCZYX_INDICES)
    return (
        reordered[0],
        reordered[1],
        reordered[2],
        reordered[3],
        reordered[4],
        reordered[5],
    )


def format_zarr_chunks_ptczyx(chunks_ptczyx: Sequence[int]) -> str:
    """Format chunk sizes for display in ``(p, t, c, z, y, x)`` order.

    Parameters
    ----------
    chunks_ptczyx : sequence of int
        Chunk sizes in ``(p, t, c, z, y, x)`` order.

    Returns
    -------
    str
        Formatted axis/value pairs such as ``"p=1, t=1, c=1, z=256, y=256, x=256"``.

    Raises
    ------
    ValueError
        If chunk values are invalid.
    """
    normalized = _normalize_ptczyx_chunks(chunks_ptczyx)
    return ", ".join(
        f"{axis}={size}" for axis, size in zip(PTCZYX_AXES, normalized, strict=False)
    )


def format_zarr_pyramid_ptczyx(pyramid_ptczyx: Sequence[Sequence[int]]) -> str:
    """Format pyramid factors for display in ``(p, t, c, z, y, x)`` order.

    Parameters
    ----------
    pyramid_ptczyx : sequence of sequence of int
        Pyramid factors per axis in ``(p, t, c, z, y, x)`` order.

    Returns
    -------
    str
        Formatted axis/value list such as
        ``"p=1; t=1; c=1; z=1,2,4,8; y=1,2,4,8; x=1,2,4,8"``.

    Raises
    ------
    ValueError
        If pyramid values are invalid.
    """
    normalized = _normalize_ptczyx_pyramid(pyramid_ptczyx)
    return "; ".join(
        f"{axis}={format_pyramid_levels(levels)}"
        for axis, levels in zip(PTCZYX_AXES, normalized, strict=False)
    )


@dataclass(frozen=True)
class ZarrSaveConfig:
    """Configuration for analysis-store chunking and pyramid downsampling.

    Attributes
    ----------
    chunks_ptczyx : tuple[int, int, int, int, int, int]
        Chunk sizes in ``(p, t, c, z, y, x)`` order.
    pyramid_ptczyx : tuple[tuple[int, ...], ...]
        Per-axis pyramid factors in ``(p, t, c, z, y, x)`` order.
    """

    chunks_ptczyx: ZarrAxisSpec = DEFAULT_ZARR_CHUNKS_PTCZYX
    pyramid_ptczyx: ZarrPyramidAxisSpec = DEFAULT_ZARR_PYRAMID_PTCZYX

    def __post_init__(self) -> None:
        """Validate and normalize the save configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Values are normalized in-place on the frozen dataclass.

        Raises
        ------
        ValueError
            If chunk sizes or pyramid factors are invalid.
        """
        object.__setattr__(
            self,
            "chunks_ptczyx",
            _normalize_ptczyx_chunks(self.chunks_ptczyx),
        )
        object.__setattr__(
            self,
            "pyramid_ptczyx",
            _normalize_ptczyx_pyramid(self.pyramid_ptczyx),
        )

    def chunks_tpczyx(self) -> ZarrAxisSpec:
        """Return chunk sizes in canonical ``(t, p, c, z, y, x)`` order.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[int, int, int, int, int, int]
            Chunk sizes in canonical axis order.
        """
        return to_tpczyx_chunks(self.chunks_ptczyx)

    def pyramid_tpczyx(self) -> ZarrPyramidAxisSpec:
        """Return pyramid factors in canonical ``(t, p, c, z, y, x)`` order.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Pyramid factors in canonical axis order.
        """
        return to_tpczyx_pyramid(self.pyramid_ptczyx)


DASK_BACKEND_LOCAL_CLUSTER = "local_cluster"
DASK_BACKEND_SLURM_RUNNER = "slurm_runner"
DASK_BACKEND_SLURM_CLUSTER = "slurm_cluster"
DaskBackendMode = Literal[
    "local_cluster",
    "slurm_runner",
    "slurm_cluster",
]

DASK_BACKEND_MODE_LABELS: Dict[str, str] = {
    DASK_BACKEND_LOCAL_CLUSTER: "LocalCluster",
    DASK_BACKEND_SLURM_RUNNER: "SLURMRunner",
    DASK_BACKEND_SLURM_CLUSTER: "SLURMCluster",
}

DEFAULT_SLURM_CLUSTER_JOB_EXTRA_DIRECTIVES: Tuple[str, ...] = (
    "--nodes=1",
    "--ntasks=1",
    "--mail-type=FAIL",
    "-o job_%j.out",
    "-e job_%j.err",
)


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    """Normalize optional text inputs.

    Parameters
    ----------
    value : str, optional
        Input text.

    Returns
    -------
    str, optional
        Stripped text, or ``None`` when empty.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_positive_int(
    value: Optional[int], *, field_name: str
) -> Optional[int]:
    """Normalize optional positive-integer values.

    Parameters
    ----------
    value : int, optional
        Input value.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    int, optional
        Parsed positive integer or ``None``.

    Raises
    ------
    ValueError
        If the provided value is not a positive integer.
    """
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return parsed


def _normalize_timeout_text(value: str, *, field_name: str) -> str:
    """Normalize timeout/walltime text values.

    Parameters
    ----------
    value : str
        Input text.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    str
        Stripped non-empty text.

    Raises
    ------
    ValueError
        If text is empty.
    """
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty.")
    return text


@dataclass(frozen=True)
class LocalClusterConfig:
    """Runtime options for local Dask distributed execution.

    Attributes
    ----------
    n_workers : int, optional
        Number of local workers. ``None`` defers to Dask defaults.
    threads_per_worker : int
        Number of threads per worker process.
    memory_limit : str
        Per-worker memory limit.
    local_directory : str, optional
        Optional local spill/scratch directory.
    """

    n_workers: Optional[int] = None
    threads_per_worker: int = 1
    memory_limit: str = "auto"
    local_directory: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate local-cluster configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Values are normalized in-place.

        Raises
        ------
        ValueError
            If any numeric field is invalid.
        """
        object.__setattr__(
            self,
            "n_workers",
            _normalize_optional_positive_int(
                self.n_workers,
                field_name="LocalCluster n_workers",
            ),
        )
        object.__setattr__(
            self,
            "threads_per_worker",
            _normalize_optional_positive_int(
                self.threads_per_worker,
                field_name="LocalCluster threads_per_worker",
            )
            or 1,
        )
        object.__setattr__(
            self,
            "memory_limit",
            _normalize_timeout_text(
                self.memory_limit,
                field_name="LocalCluster memory_limit",
            ),
        )
        object.__setattr__(
            self,
            "local_directory",
            _normalize_optional_text(self.local_directory),
        )


@dataclass(frozen=True)
class SlurmRunnerConfig:
    """Runtime options for connecting through ``dask_jobqueue.SLURMRunner``.

    Attributes
    ----------
    scheduler_file : str, optional
        Scheduler file path used by ``SLURMRunner``.
    wait_for_workers : int, optional
        Explicit worker count to await after client connection. ``None`` uses
        runner defaults.
    """

    scheduler_file: Optional[str] = None
    wait_for_workers: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate SLURM runner configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Values are normalized in-place.

        Raises
        ------
        ValueError
            If numeric values are invalid.
        """
        object.__setattr__(
            self,
            "scheduler_file",
            _normalize_optional_text(self.scheduler_file),
        )
        object.__setattr__(
            self,
            "wait_for_workers",
            _normalize_optional_positive_int(
                self.wait_for_workers,
                field_name="SLURMRunner wait_for_workers",
            ),
        )


@dataclass(frozen=True)
class SlurmClusterConfig:
    """Runtime options for launching a Dask ``SLURMCluster``.

    Attributes
    ----------
    workers : int
        Number of worker jobs to request.
    cores : int
        Worker cores/threads setting.
    processes : int
        Number of Python processes per job.
    memory : str
        Worker memory request (for example ``"220GB"``).
    local_directory : str, optional
        Worker local spill/scratch directory.
    interface : str
        Network interface for worker communications.
    walltime : str
        Walltime in Slurm format (for example ``"01:00:00"``).
    job_name : str
        Slurm job name.
    queue : str
        Slurm partition/queue.
    death_timeout : str
        Worker death timeout text (for example ``"600s"``).
    mail_user : str, optional
        Email recipient for Slurm notifications.
    job_extra_directives : tuple[str, ...]
        Additional Slurm directives passed through ``job_extra_directives``.
    dashboard_address : str
        Dashboard bind address for scheduler options.
    scheduler_interface : str
        Network interface for scheduler options.
    idle_timeout : str
        Scheduler idle timeout (for example ``"3600s"``).
    allowed_failures : int
        Scheduler allowed worker failures before shutdown.
    """

    workers: int = 1
    cores: int = 28
    processes: int = 1
    memory: str = "220GB"
    local_directory: Optional[str] = None
    interface: str = "ib0"
    walltime: str = "01:00:00"
    job_name: str = "clearex"
    queue: str = "256GB"
    death_timeout: str = "600s"
    mail_user: Optional[str] = None
    job_extra_directives: Tuple[str, ...] = DEFAULT_SLURM_CLUSTER_JOB_EXTRA_DIRECTIVES
    dashboard_address: str = ":9000"
    scheduler_interface: str = "ib0"
    idle_timeout: str = "3600s"
    allowed_failures: int = 10

    def __post_init__(self) -> None:
        """Validate SLURM cluster configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Values are normalized in-place.

        Raises
        ------
        ValueError
            If fields are invalid.
        """
        object.__setattr__(
            self,
            "workers",
            _normalize_optional_positive_int(
                self.workers,
                field_name="SLURMCluster workers",
            )
            or 1,
        )
        object.__setattr__(
            self,
            "cores",
            _normalize_optional_positive_int(
                self.cores,
                field_name="SLURMCluster cores",
            )
            or 1,
        )
        object.__setattr__(
            self,
            "processes",
            _normalize_optional_positive_int(
                self.processes,
                field_name="SLURMCluster processes",
            )
            or 1,
        )
        object.__setattr__(
            self,
            "memory",
            _normalize_timeout_text(
                self.memory,
                field_name="SLURMCluster memory",
            ),
        )
        object.__setattr__(
            self,
            "local_directory",
            _normalize_optional_text(self.local_directory),
        )
        object.__setattr__(
            self,
            "interface",
            _normalize_timeout_text(
                self.interface,
                field_name="SLURMCluster interface",
            ),
        )
        object.__setattr__(
            self,
            "walltime",
            _normalize_timeout_text(
                self.walltime,
                field_name="SLURMCluster walltime",
            ),
        )
        object.__setattr__(
            self,
            "job_name",
            _normalize_timeout_text(
                self.job_name,
                field_name="SLURMCluster job_name",
            ),
        )
        object.__setattr__(
            self,
            "queue",
            _normalize_timeout_text(
                self.queue,
                field_name="SLURMCluster queue",
            ),
        )
        object.__setattr__(
            self,
            "death_timeout",
            _normalize_timeout_text(
                self.death_timeout,
                field_name="SLURMCluster death_timeout",
            ),
        )
        object.__setattr__(
            self,
            "mail_user",
            _normalize_optional_text(self.mail_user),
        )

        directives = tuple(
            directive.strip()
            for directive in self.job_extra_directives
            if str(directive).strip()
        )
        if not directives:
            directives = DEFAULT_SLURM_CLUSTER_JOB_EXTRA_DIRECTIVES
        object.__setattr__(
            self,
            "job_extra_directives",
            directives,
        )
        object.__setattr__(
            self,
            "dashboard_address",
            _normalize_timeout_text(
                self.dashboard_address,
                field_name="SLURMCluster dashboard_address",
            ),
        )
        object.__setattr__(
            self,
            "scheduler_interface",
            _normalize_timeout_text(
                self.scheduler_interface,
                field_name="SLURMCluster scheduler_interface",
            ),
        )
        object.__setattr__(
            self,
            "idle_timeout",
            _normalize_timeout_text(
                self.idle_timeout,
                field_name="SLURMCluster idle_timeout",
            ),
        )
        object.__setattr__(
            self,
            "allowed_failures",
            _normalize_optional_positive_int(
                self.allowed_failures,
                field_name="SLURMCluster allowed_failures",
            )
            or 1,
        )


@dataclass(frozen=True)
class DaskBackendConfig:
    """Shared Dask backend configuration for GUI and headless execution.

    Attributes
    ----------
    mode : {"local_cluster", "slurm_runner", "slurm_cluster"}
        Selected Dask backend mode.
    local_cluster : LocalClusterConfig
        Settings for local distributed execution.
    slurm_runner : SlurmRunnerConfig
        Settings for scheduler-file-based SLURM runner execution.
    slurm_cluster : SlurmClusterConfig
        Settings for launching worker jobs with ``SLURMCluster``.
    """

    mode: DaskBackendMode = DASK_BACKEND_LOCAL_CLUSTER
    local_cluster: LocalClusterConfig = field(default_factory=LocalClusterConfig)
    slurm_runner: SlurmRunnerConfig = field(default_factory=SlurmRunnerConfig)
    slurm_cluster: SlurmClusterConfig = field(default_factory=SlurmClusterConfig)

    def __post_init__(self) -> None:
        """Validate selected backend mode.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Values are validated in-place.

        Raises
        ------
        ValueError
            If mode is unknown.
        """
        mode = str(self.mode).strip().lower()
        if mode not in DASK_BACKEND_MODE_LABELS:
            valid_modes = ", ".join(DASK_BACKEND_MODE_LABELS.keys())
            raise ValueError(f"Dask backend mode must be one of: {valid_modes}.")
        object.__setattr__(self, "mode", mode)


def dask_backend_to_dict(config: DaskBackendConfig) -> Dict[str, Any]:
    """Serialize Dask backend config into JSON-friendly mappings.

    Parameters
    ----------
    config : DaskBackendConfig
        Backend configuration to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable backend mapping.
    """
    return {
        "mode": config.mode,
        "local_cluster": {
            "n_workers": config.local_cluster.n_workers,
            "threads_per_worker": config.local_cluster.threads_per_worker,
            "memory_limit": config.local_cluster.memory_limit,
            "local_directory": config.local_cluster.local_directory,
        },
        "slurm_runner": {
            "scheduler_file": config.slurm_runner.scheduler_file,
            "wait_for_workers": config.slurm_runner.wait_for_workers,
        },
        "slurm_cluster": {
            "workers": config.slurm_cluster.workers,
            "cores": config.slurm_cluster.cores,
            "processes": config.slurm_cluster.processes,
            "memory": config.slurm_cluster.memory,
            "local_directory": config.slurm_cluster.local_directory,
            "interface": config.slurm_cluster.interface,
            "walltime": config.slurm_cluster.walltime,
            "job_name": config.slurm_cluster.job_name,
            "queue": config.slurm_cluster.queue,
            "death_timeout": config.slurm_cluster.death_timeout,
            "mail_user": config.slurm_cluster.mail_user,
            "job_extra_directives": list(config.slurm_cluster.job_extra_directives),
            "scheduler_options": {
                "dashboard_address": config.slurm_cluster.dashboard_address,
                "interface": config.slurm_cluster.scheduler_interface,
                "idle_timeout": config.slurm_cluster.idle_timeout,
                "allowed_failures": config.slurm_cluster.allowed_failures,
            },
        },
    }


def format_dask_backend_summary(config: DaskBackendConfig) -> str:
    """Format a compact summary of the selected Dask backend.

    Parameters
    ----------
    config : DaskBackendConfig
        Backend configuration.

    Returns
    -------
    str
        Human-readable one-line summary.
    """
    mode_label = DASK_BACKEND_MODE_LABELS.get(config.mode, config.mode)
    if config.mode == DASK_BACKEND_LOCAL_CLUSTER:
        workers_text = (
            str(config.local_cluster.n_workers)
            if config.local_cluster.n_workers is not None
            else "auto"
        )
        return (
            f"{mode_label} "
            f"(workers={workers_text}, threads={config.local_cluster.threads_per_worker}, "
            f"memory={config.local_cluster.memory_limit})"
        )
    if config.mode == DASK_BACKEND_SLURM_RUNNER:
        scheduler_file = config.slurm_runner.scheduler_file or "not set"
        return f"{mode_label} (scheduler_file={scheduler_file})"
    return (
        f"{mode_label} "
        f"(workers={config.slurm_cluster.workers}, "
        f"queue={config.slurm_cluster.queue}, walltime={config.slurm_cluster.walltime})"
    )


@dataclass
class WorkflowConfig:
    """Runtime workflow options shared by GUI and headless entrypoints.

    Attributes
    ----------
    file : str, optional
        Input image path for processing.
    prefer_dask : bool
        Whether to open data using lazy Dask-backed arrays when supported.
    dask_backend : DaskBackendConfig
        Backend orchestration mode and runtime settings for Dask execution.
    chunks : int or tuple of int, optional
        Chunking configuration used for Dask reads.
    deconvolution : bool
        Flag indicating whether deconvolution workflow should run.
    particle_detection : bool
        Flag indicating whether particle detection workflow should run.
    registration : bool
        Flag indicating whether registration workflow should run.
    visualization : bool
        Flag indicating whether visualization workflow should run.
    zarr_save : ZarrSaveConfig
        Analysis-store chunking and pyramid configuration for saved Zarr data.
    analysis_parameters : dict[str, dict[str, Any]]
        Per-analysis runtime parameters keyed by analysis name.
    """

    file: Optional[str] = None
    prefer_dask: bool = True
    dask_backend: DaskBackendConfig = field(default_factory=DaskBackendConfig)
    chunks: ChunkSpec = None
    deconvolution: bool = False
    particle_detection: bool = False
    registration: bool = False
    visualization: bool = False
    zarr_save: ZarrSaveConfig = field(default_factory=ZarrSaveConfig)
    analysis_parameters: Dict[str, Dict[str, Any]] = field(
        default_factory=default_analysis_operation_parameters
    )

    def __post_init__(self) -> None:
        """Normalize analysis-parameter mappings.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Values are normalized in-place on the dataclass.

        Raises
        ------
        ValueError
            If analysis parameter mappings are invalid.
        """
        self.analysis_parameters = normalize_analysis_operation_parameters(
            self.analysis_parameters
        )

    def has_analysis_selection(self) -> bool:
        """Return whether at least one analysis operation is selected.

        Returns
        -------
        bool
            ``True`` if any analysis flag is enabled, otherwise ``False``.
        """
        return any(
            (
                self.deconvolution,
                self.particle_detection,
                self.registration,
                self.visualization,
            )
        )


def parse_chunks(chunks: Optional[str]) -> ChunkSpec:
    """Parse chunk spec from CLI/GUI text.

    Parameters
    ----------
    chunks : str, optional
        A single integer (e.g., ``"256"``) or comma-separated tuple
        (e.g., ``"1,256,256"``). Empty strings are treated as ``None``.

    Returns
    -------
    Optional[int | Tuple[int, ...]]
        Parsed chunk specification or ``None``.

    Raises
    ------
    ValueError
        If ``chunks`` cannot be parsed as integers or contains non-positive
        values.
    """
    if chunks is None:
        return None

    text = chunks.strip()
    if not text:
        return None

    if "," not in text:
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                "Chunks must be a positive integer or comma-separated integers."
            ) from exc
        if value <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        return value

    try:
        parts = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(
            "Chunks must be a positive integer or comma-separated integers."
        ) from exc
    if not parts:
        return None
    if any(value <= 0 for value in parts):
        raise ValueError("Chunk sizes must be greater than zero.")
    return parts


def format_chunks(chunks: ChunkSpec) -> str:
    """Format a chunk specification for display.

    Parameters
    ----------
    chunks : int or tuple of int, optional
        Chunk specification to serialize.

    Returns
    -------
    str
        Empty string when ``chunks`` is ``None``. Otherwise returns a single
        integer or comma-separated integer list.
    """
    if chunks is None:
        return ""
    if isinstance(chunks, tuple):
        return ",".join(str(part) for part in chunks)
    return str(chunks)
