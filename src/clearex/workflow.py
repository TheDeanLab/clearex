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
import math
import os
import subprocess
from typing import Any, Collection, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union


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
    "flatfield",
    "deconvolution",
    "shear_transform",
    "particle_detection",
    "usegment3d",
    "registration",
    "visualization",
    "mip_export",
)

ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS: Dict[str, str] = {
    "data": "data",
    "flatfield": "results/flatfield/latest/data",
    "deconvolution": "results/deconvolution/latest/data",
    "shear_transform": "results/shear_transform/latest/data",
    "usegment3d": "results/usegment3d/latest/data",
}
ANALYSIS_KNOWN_OUTPUT_COMPONENTS: Dict[str, str] = {
    "data": "data",
    "flatfield": "results/flatfield/latest/data",
    "deconvolution": "results/deconvolution/latest/data",
    "shear_transform": "results/shear_transform/latest/data",
    "particle_detection": "results/particle_detection/latest/detections",
    "usegment3d": "results/usegment3d/latest/data",
    "registration": "results/registration/latest/data",
    "visualization": "results/visualization/latest",
    "mip_export": "results/mip_export/latest",
}
_OUTPUT_COMPONENT_TO_OPERATION: Dict[str, str] = {
    str(component): str(operation_name)
    for operation_name, component in ANALYSIS_KNOWN_OUTPUT_COMPONENTS.items()
}
SPATIAL_CALIBRATION_SCHEMA = "clearex.spatial_calibration.v1"
SPATIAL_CALIBRATION_WORLD_AXES = ("z", "y", "x")
SPATIAL_CALIBRATION_SOURCE_AXES = ("x", "y", "z", "f")
SPATIAL_CALIBRATION_ALLOWED_BINDINGS = frozenset(
    {
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
        "+f",
        "-f",
        "none",
    }
)
SPATIAL_CALIBRATION_DEFAULT_STAGE_AXIS_MAP_ZYX = ("+z", "+y", "+x")
SPATIAL_CALIBRATION_DEFAULT_THETA_MODE = "rotate_zy_about_x"


@dataclass(frozen=True)
class AnalysisInputReference:
    """One analysis input reference used for dependency validation.

    Parameters
    ----------
    consumer_operation : str
        Analysis operation consuming the reference.
    field_name : str
        Parameter field name carrying the reference.
    requested_source : str
        Requested alias or component path.
    """

    consumer_operation: str
    field_name: str
    requested_source: str


@dataclass(frozen=True)
class AnalysisInputDependencyIssue:
    """Description of one invalid analysis input dependency.

    Parameters
    ----------
    consumer_operation : str
        Analysis operation consuming the reference.
    field_name : str
        Parameter field name carrying the reference.
    requested_source : str
        Requested alias or component path.
    reason : str
        Machine-readable reason key.
    message : str
        Human-readable validation message.
    producer_operation : str, optional
        Upstream producer operation when the reference maps to one.
    resolved_component : str, optional
        Expected component path for the requested source when known.
    """

    consumer_operation: str
    field_name: str
    requested_source: str
    reason: str
    message: str
    producer_operation: Optional[str] = None
    resolved_component: Optional[str] = None


def _analysis_operation_display_name(operation_name: str) -> str:
    """Return a human-readable analysis operation name.

    Parameters
    ----------
    operation_name : str
        Operation key.

    Returns
    -------
    str
        Title-cased operation label for diagnostics.
    """
    return str(operation_name).strip().replace("_", " ").title() or "Analysis"


def analysis_chainable_output_component(operation_name: str) -> Optional[str]:
    """Return the reusable latest-output component for one operation.

    Parameters
    ----------
    operation_name : str
        Candidate operation key.

    Returns
    -------
    str, optional
        Reusable component path when the operation exposes one, otherwise
        ``None``.
    """
    key = str(operation_name).strip()
    if not key:
        return None
    return ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS.get(key)


def analysis_operation_for_output_component(component: str) -> Optional[str]:
    """Return the producing operation for one known output component.

    Parameters
    ----------
    component : str
        Output component path.

    Returns
    -------
    str, optional
        Operation key when the component is a known latest-output location,
        otherwise ``None``.
    """
    key = str(component).strip()
    if not key:
        return None
    return _OUTPUT_COMPONENT_TO_OPERATION.get(key)


def resolve_analysis_input_component(
    requested_source: str,
    produced_components: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve an analysis input alias to a component path.

    Parameters
    ----------
    requested_source : str
        Requested alias or explicit component path.
    produced_components : mapping[str, str], optional
        Components produced earlier in the current workflow run.

    Returns
    -------
    str
        Resolved component path suitable for store lookup.
    """
    source = str(requested_source).strip() or "data"
    if produced_components is not None and source in produced_components:
        return str(produced_components[source])
    component = analysis_chainable_output_component(source)
    if component is not None:
        return str(component)
    return source


def collect_analysis_input_references(
    *,
    execution_sequence: Sequence[str],
    analysis_parameters: Mapping[str, Mapping[str, Any]],
) -> Tuple[AnalysisInputReference, ...]:
    """Collect input references from the selected analysis sequence.

    Parameters
    ----------
    execution_sequence : sequence[str]
        Selected operations in execution order.
    analysis_parameters : mapping[str, mapping[str, Any]]
        Candidate normalized or denormalized per-operation parameters.

    Returns
    -------
    tuple[AnalysisInputReference, ...]
        Collected references, including visualization volume-layer components.
    """
    normalized = normalize_analysis_operation_parameters(dict(analysis_parameters))
    references: list[AnalysisInputReference] = []
    for operation_name in execution_sequence:
        params = dict(normalized.get(str(operation_name), {}))
        references.append(
            AnalysisInputReference(
                consumer_operation=str(operation_name),
                field_name="input_source",
                requested_source=str(params.get("input_source", "data")).strip()
                or "data",
            )
        )
        if str(operation_name).strip() != "visualization":
            continue
        raw_rows = params.get("volume_layers", [])
        if not isinstance(raw_rows, (tuple, list)):
            continue
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                continue
            component = str(
                raw_row.get("component", raw_row.get("source_component", ""))
            ).strip()
            if not component:
                continue
            references.append(
                AnalysisInputReference(
                    consumer_operation="visualization",
                    field_name=f"volume_layers[{index}].component",
                    requested_source=component,
                )
            )
    return tuple(references)


def _validate_analysis_input_reference(
    *,
    reference: AnalysisInputReference,
    execution_sequence: Sequence[str],
    available_components: Collection[str],
) -> Optional[AnalysisInputDependencyIssue]:
    """Validate one analysis input reference against sequence and store state.

    Parameters
    ----------
    reference : AnalysisInputReference
        Input reference to validate.
    execution_sequence : sequence[str]
        Selected operations in execution order.
    available_components : collection[str]
        Components currently available in the active analysis store.

    Returns
    -------
    AnalysisInputDependencyIssue, optional
        Validation issue when the reference is invalid, otherwise ``None``.
    """
    requested_source = str(reference.requested_source).strip() or "data"
    if requested_source == "data":
        return None

    available = {
        str(component).strip()
        for component in available_components
        if str(component).strip()
    }
    available.add("data")
    order_map = {
        str(operation_name).strip(): index
        for index, operation_name in enumerate(execution_sequence)
        if str(operation_name).strip()
    }
    consumer_name = _analysis_operation_display_name(reference.consumer_operation)
    field_name = str(reference.field_name).strip() or "input_source"

    producer_operation: Optional[str] = None
    resolved_component: Optional[str] = None
    if requested_source in ANALYSIS_KNOWN_OUTPUT_COMPONENTS:
        producer_operation = requested_source
        resolved_component = ANALYSIS_KNOWN_OUTPUT_COMPONENTS[requested_source]
    else:
        producer_operation = analysis_operation_for_output_component(requested_source)
        resolved_component = (
            requested_source if producer_operation is not None else None
        )

    if producer_operation is not None:
        producer_name = _analysis_operation_display_name(producer_operation)
        if producer_operation not in ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS:
            return AnalysisInputDependencyIssue(
                consumer_operation=reference.consumer_operation,
                field_name=field_name,
                requested_source=requested_source,
                reason="producer_not_chainable",
                message=(
                    f"{consumer_name} {field_name} references {producer_name} output "
                    f"({resolved_component or requested_source}), but {producer_name} "
                    "does not expose a reusable downstream input for generic chaining."
                ),
                producer_operation=producer_operation,
                resolved_component=resolved_component,
            )
        if producer_operation in order_map:
            consumer_index = order_map.get(str(reference.consumer_operation).strip(), -1)
            producer_index = order_map[producer_operation]
            if producer_index >= consumer_index:
                return AnalysisInputDependencyIssue(
                    consumer_operation=reference.consumer_operation,
                    field_name=field_name,
                    requested_source=requested_source,
                    reason="producer_scheduled_after_consumer",
                    message=(
                        f"{consumer_name} {field_name} references {producer_name} "
                        f"output ({resolved_component or requested_source}), but "
                        f"{producer_name} is scheduled after {consumer_name}. "
                        "Reorder the analyses so the producer runs earlier."
                    ),
                    producer_operation=producer_operation,
                    resolved_component=resolved_component,
                )
            return None
        if (resolved_component or "") in available:
            return None
        return AnalysisInputDependencyIssue(
            consumer_operation=reference.consumer_operation,
            field_name=field_name,
            requested_source=requested_source,
            reason="producer_not_selected_or_available",
            message=(
                f"{consumer_name} {field_name} references {producer_name} output "
                f"({resolved_component or requested_source}), but {producer_name} is "
                "not scheduled earlier in this run and that component is not "
                "available in the current analysis store."
            ),
            producer_operation=producer_operation,
            resolved_component=resolved_component,
        )

    if requested_source in available:
        return None

    return AnalysisInputDependencyIssue(
        consumer_operation=reference.consumer_operation,
        field_name=field_name,
        requested_source=requested_source,
        reason="missing_component",
        message=(
            f"{consumer_name} {field_name} references component "
            f"'{requested_source}', but it is not available in the current "
            "analysis store and is not produced earlier in this run."
        ),
        resolved_component=requested_source,
    )


def validate_analysis_input_references(
    *,
    execution_sequence: Sequence[str],
    analysis_parameters: Mapping[str, Mapping[str, Any]],
    available_components: Optional[Collection[str]] = None,
) -> Tuple[AnalysisInputDependencyIssue, ...]:
    """Validate selected analysis input references.

    Parameters
    ----------
    execution_sequence : sequence[str]
        Selected operations in execution order.
    analysis_parameters : mapping[str, mapping[str, Any]]
        Candidate normalized or denormalized per-operation parameters.
    available_components : collection[str], optional
        Components currently available in the active analysis store.

    Returns
    -------
    tuple[AnalysisInputDependencyIssue, ...]
        Validation issues for invalid references. Empty when configuration is
        dependency-safe under the current sequence and available components.
    """
    references = collect_analysis_input_references(
        execution_sequence=execution_sequence,
        analysis_parameters=analysis_parameters,
    )
    available = {
        str(component).strip()
        for component in (available_components or ())
        if str(component).strip()
    }
    available.add("data")

    issues: list[AnalysisInputDependencyIssue] = []
    seen: set[tuple[str, str, str, str]] = set()
    for reference in references:
        issue = _validate_analysis_input_reference(
            reference=reference,
            execution_sequence=execution_sequence,
            available_components=available,
        )
        if issue is None:
            continue
        key = (
            str(issue.consumer_operation),
            str(issue.field_name),
            str(issue.requested_source),
            str(issue.reason),
        )
        if key in seen:
            continue
        seen.add(key)
        issues.append(issue)
    return tuple(issues)


class WorkflowExecutionCancelled(RuntimeError):
    """Raised when a running workflow is cancelled by the operator."""


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
    "flatfield": {
        "execution_order": 1,
        "input_source": "data",
        "force_rerun": False,
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": True,
        "overlap_zyx": [0, 32, 32],
        "memory_overhead_factor": 2.0,
        "fit_mode": "tiled",
        "fit_tile_shape_yx": [256, 256],
        "blend_tiles": False,
        "get_darkfield": False,
        "smoothness_flatfield": 1.0,
        "working_size": 128,
        "is_timelapse": False,
    },
    "deconvolution": {
        "execution_order": 2,
        "input_source": "data",
        "force_rerun": False,
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
    "shear_transform": {
        "execution_order": 3,
        "input_source": "data",
        "force_rerun": False,
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 2.0,
        "shear_xy": 0.0,
        "shear_xz": 0.0,
        "shear_yz": 0.0,
        "rotation_deg_x": 0.0,
        "rotation_deg_y": 0.0,
        "rotation_deg_z": 0.0,
        "auto_rotate_from_shear": False,
        "auto_estimate_shear_yz": False,
        "auto_estimate_extreme_fraction_x": 0.03,
        "auto_estimate_zy_stride": 4,
        "auto_estimate_signal_fraction": 0.10,
        "auto_estimate_t_index": 0,
        "auto_estimate_p_index": 0,
        "auto_estimate_c_index": 0,
        "interpolation": "linear",
        "fill_value": 0.0,
        "output_dtype": "float32",
        "roi_padding_zyx": [2, 2, 2],
    },
    "particle_detection": {
        "execution_order": 4,
        "input_source": "data",
        "force_rerun": False,
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
    "usegment3d": {
        "execution_order": 5,
        "input_source": "data",
        "force_rerun": False,
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [12, 24, 24],
        "memory_overhead_factor": 3.0,
        "all_channels": False,
        "channel_indices": [0],
        "channel_index": 0,
        "use_views": ["xy", "xz", "yz"],
        "input_resolution_level": 0,
        "output_reference_space": "level0",
        "save_native_labels": False,
        "gpu": True,
        "require_gpu": False,
        "preprocess_factor": 1.0,
        "preprocess_voxel_res_zyx": [1.0, 1.0, 1.0],
        "preprocess_do_bg_correction": True,
        "preprocess_bg_ds": 16,
        "preprocess_bg_sigma": 5.0,
        "preprocess_normalize_min": 2.0,
        "preprocess_normalize_max": 99.8,
        "cellpose_model_name": "cyto",
        "cellpose_channels": "grayscale",
        "cellpose_hist_norm": False,
        "cellpose_ksize": 15,
        "cellpose_use_auto_diameter": False,
        "cellpose_best_diameter": None,
        "cellpose_diameter_range": [10.0, 120.0, 2.5],
        "cellpose_use_edge": True,
        "cellpose_model_invert": False,
        "cellpose_test_slice": None,
        "aggregation_gradient_decay": 0.0,
        "aggregation_n_iter": 200,
        "aggregation_momenta": 0.98,
        "aggregation_prob_threshold": None,
        "aggregation_threshold_n_levels": 3,
        "aggregation_threshold_level": 1,
        "aggregation_min_prob_threshold": 0.0,
        "aggregation_connected_min_area": 5,
        "aggregation_connected_smooth_sigma": 1.0,
        "aggregation_connected_thresh_factor": 0.0,
        "aggregation_binary_fill_holes": False,
        "aggregation_tile_mode": False,
        "aggregation_tile_shape_zyx": [128, 256, 256],
        "aggregation_tile_overlap_ratio": 0.25,
        "n_cpu": None,
        "postprocess_enable": True,
        "postprocess_min_size": 200,
        "postprocess_do_flow_remove": True,
        "postprocess_flow_threshold": 0.85,
        "postprocess_dtform_method": "cellpose_improve",
        "postprocess_edt_fixed_point_percentile": 0.01,
        "output_dtype": "uint32",
    },
    "registration": {
        "execution_order": 6,
        "input_source": "data",
        "force_rerun": False,
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": True,
        "overlap_zyx": [8, 32, 32],
        "memory_overhead_factor": 2.5,
    },
    "visualization": {
        "execution_order": 7,
        "input_source": "data",
        "chunk_basis": "2d",
        "detect_2d_per_slice": True,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 1.0,
        "show_all_positions": False,
        "position_index": 0,
        "use_multiscale": True,
        "use_3d_view": True,
        "overlay_particle_detections": True,
        "particle_detection_component": "results/particle_detection/latest/detections",
        "launch_mode": "auto",
        "require_gpu_rendering": True,
        "capture_keyframes": True,
        "keyframe_manifest_path": "",
        "keyframe_layer_overrides": [],
        "volume_layers": [],
    },
    "mip_export": {
        "execution_order": 8,
        "input_source": "data",
        "force_rerun": False,
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": 1.0,
        "position_mode": "multi_position",
        "export_format": "ome-tiff",
        "output_directory": "",
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
    normalized["force_rerun"] = bool(normalized.get("force_rerun", False))

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


def _normalize_parameter_int_pair(
    value: Any,
    *,
    field_name: str,
    default: tuple[int, int],
) -> list[int]:
    """Normalize parameter value into a positive integer pair.

    Parameters
    ----------
    value : Any
        Candidate value to parse.
    field_name : str
        Field name used in validation errors.
    default : tuple[int, int]
        Default pair when value is empty.

    Returns
    -------
    list[int]
        Positive integer pair.

    Raises
    ------
    ValueError
        If value does not define exactly two positive integers.
    """
    tokens = _normalize_parameter_string_list(value)
    if not tokens:
        return [int(default[0]), int(default[1])]
    if len(tokens) != 2:
        raise ValueError(f"{field_name} must define exactly two values.")
    parsed = [int(tokens[0]), int(tokens[1])]
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


def _normalize_flatfield_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize flatfield-correction runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate flatfield parameters.

    Returns
    -------
    dict[str, Any]
        Normalized flatfield parameter mapping.

    Raises
    ------
    ValueError
        If required values are invalid.
    """
    normalized = _normalize_common_operation_parameters("flatfield", params)
    normalized["get_darkfield"] = bool(normalized.get("get_darkfield", False))
    normalized["is_timelapse"] = bool(normalized.get("is_timelapse", False))
    normalized["smoothness_flatfield"] = float(
        normalized.get("smoothness_flatfield", 1.0)
    )
    if normalized["smoothness_flatfield"] <= 0:
        raise ValueError("flatfield smoothness_flatfield must be greater than zero.")
    normalized["working_size"] = int(normalized.get("working_size", 128))
    if normalized["working_size"] <= 0:
        raise ValueError("flatfield working_size must be greater than zero.")
    fit_mode = (
        str(normalized.get("fit_mode", "tiled")).strip().lower().replace("-", "_")
    )
    if fit_mode not in {"tiled", "full_volume"}:
        raise ValueError("flatfield fit_mode must be 'tiled' or 'full_volume'.")
    normalized["fit_mode"] = fit_mode
    normalized["fit_tile_shape_yx"] = _normalize_parameter_int_pair(
        normalized.get("fit_tile_shape_yx", [256, 256]),
        field_name="flatfield fit_tile_shape_yx",
        default=(256, 256),
    )
    normalized["blend_tiles"] = bool(normalized.get("blend_tiles", False))
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


def _normalize_usegment3d_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize u-segment3d runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate usegment3d parameters.

    Returns
    -------
    dict[str, Any]
        Normalized usegment3d parameter mapping.

    Raises
    ------
    ValueError
        If required values are invalid.
    """
    normalized = _normalize_common_operation_parameters("usegment3d", params)
    normalized["detect_2d_per_slice"] = False
    normalized["all_channels"] = bool(normalized.get("all_channels", False))

    default_usegment3d = DEFAULT_ANALYSIS_OPERATION_PARAMETERS.get("usegment3d", {})
    default_channel_indices = [
        max(0, int(value))
        for value in _normalize_parameter_float_list(
            default_usegment3d.get("channel_indices", [0])
        )
    ]
    if not default_channel_indices:
        default_channel_indices = [0]
    default_primary_channel = max(
        0,
        int(default_usegment3d.get("channel_index", default_channel_indices[0])),
    )
    primary_channel_value = normalized.get(
        "channel_index",
        normalized.get("channel", default_primary_channel),
    )
    primary_channel_index = max(0, int(primary_channel_value))

    raw_channel_indices = _normalize_parameter_float_list(
        normalized.get("channel_indices", normalized.get("channels", []))
    )
    normalized_raw_indices = [max(0, int(value)) for value in raw_channel_indices]
    if not normalized_raw_indices or (
        normalized_raw_indices == default_channel_indices
        and primary_channel_index != default_primary_channel
    ):
        raw_channel_indices = [float(primary_channel_index)]
    normalized_channel_indices: list[int] = []
    seen_channel_indices: set[int] = set()
    for value in raw_channel_indices:
        parsed_value = max(0, int(value))
        if parsed_value in seen_channel_indices:
            continue
        normalized_channel_indices.append(parsed_value)
        seen_channel_indices.add(parsed_value)
    if not normalized_channel_indices:
        normalized_channel_indices = [0]
    normalized["channel_indices"] = list(normalized_channel_indices)
    normalized["channel_index"] = int(normalized_channel_indices[0])

    input_resolution_level = int(
        normalized.get(
            "input_resolution_level",
            normalized.get("resolution_level", 0),
        )
    )
    if input_resolution_level < 0:
        raise ValueError("usegment3d input_resolution_level must be >= 0.")
    normalized["input_resolution_level"] = input_resolution_level

    output_space_raw = (
        str(
            normalized.get(
                "output_reference_space",
                normalized.get("output_space", "level0"),
            )
        )
        .strip()
        .lower()
        or "level0"
    )
    if output_space_raw in {"level0", "full", "full_resolution", "original"}:
        output_reference_space = "level0"
    elif output_space_raw in {"native", "native_level", "selected_level"}:
        output_reference_space = "native_level"
    else:
        raise ValueError(
            "usegment3d output_reference_space must be one of level0 or native_level."
        )
    normalized["output_reference_space"] = output_reference_space
    normalized["save_native_labels"] = bool(
        bool(normalized.get("save_native_labels", False))
        or bool(normalized.get("save_native", False))
    )

    raw_views = normalized.get("use_views", normalized.get("views", ["xy", "xz", "yz"]))
    if isinstance(raw_views, str):
        raw_tokens = _normalize_parameter_string_list(raw_views)
    elif isinstance(raw_views, (tuple, list)):
        raw_tokens = [str(value).strip() for value in raw_views]
    else:
        raw_tokens = []
    view_order = ("xy", "xz", "yz")
    seen_views: set[str] = set()
    selected_views: list[str] = []
    for token in raw_tokens:
        view = str(token).strip().lower()
        if view not in view_order or view in seen_views:
            continue
        selected_views.append(view)
        seen_views.add(view)
    if not selected_views:
        raise ValueError(
            "usegment3d use_views must include at least one of xy, xz, yz."
        )
    normalized["use_views"] = selected_views

    gpu_value = normalized.get("gpu", normalized.get("use_gpu", True))
    normalized["gpu"] = bool(gpu_value)
    normalized["require_gpu"] = bool(normalized.get("require_gpu", False))

    normalized["preprocess_factor"] = float(normalized.get("preprocess_factor", 1.0))
    if normalized["preprocess_factor"] <= 0:
        raise ValueError("usegment3d preprocess_factor must be greater than zero.")

    voxel_res = normalized.get("preprocess_voxel_res_zyx", [1.0, 1.0, 1.0])
    if not isinstance(voxel_res, (tuple, list)) or len(voxel_res) != 3:
        raise ValueError(
            "usegment3d preprocess_voxel_res_zyx must define three values in (z, y, x) order."
        )
    voxel_res_zyx = [float(voxel_res[0]), float(voxel_res[1]), float(voxel_res[2])]
    if any(value <= 0 for value in voxel_res_zyx):
        raise ValueError(
            "usegment3d preprocess_voxel_res_zyx values must be greater than zero."
        )
    normalized["preprocess_voxel_res_zyx"] = voxel_res_zyx

    bg_correction = normalized.get(
        "preprocess_do_bg_correction",
        normalized.get(
            "preprocess_bg_correction",
            normalized.get("bg_correction", True),
        ),
    )
    normalized["preprocess_do_bg_correction"] = bool(bg_correction)
    normalized["preprocess_bg_ds"] = max(1, int(normalized.get("preprocess_bg_ds", 16)))
    normalized["preprocess_bg_sigma"] = float(
        normalized.get("preprocess_bg_sigma", 5.0)
    )
    if normalized["preprocess_bg_sigma"] < 0:
        raise ValueError("usegment3d preprocess_bg_sigma cannot be negative.")
    normalized["preprocess_normalize_min"] = float(
        normalized.get(
            "preprocess_normalize_min",
            normalized.get("preprocess_min", 2.0),
        )
    )
    normalized["preprocess_normalize_max"] = float(
        normalized.get(
            "preprocess_normalize_max",
            normalized.get("preprocess_max", 99.8),
        )
    )
    if (
        normalized["preprocess_normalize_min"] < 0
        or normalized["preprocess_normalize_max"] > 100
        or normalized["preprocess_normalize_min"]
        >= normalized["preprocess_normalize_max"]
    ):
        raise ValueError(
            "usegment3d preprocess normalization bounds must satisfy 0 <= min < max <= 100."
        )

    normalized["cellpose_model_name"] = (
        str(
            normalized.get(
                "cellpose_model_name",
                normalized.get("cellpose_model", "cyto"),
            )
        ).strip()
        or "cyto"
    )
    cellpose_channels_raw = normalized.get("cellpose_channels", "grayscale")
    if isinstance(cellpose_channels_raw, (tuple, list)):
        if len(cellpose_channels_raw) == 2:
            try:
                parsed_pair = [
                    int(cellpose_channels_raw[0]),
                    int(cellpose_channels_raw[1]),
                ]
            except (TypeError, ValueError):
                parsed_pair = []
            if parsed_pair == [0, 0]:
                cellpose_channels = "grayscale"
            elif len(parsed_pair) == 2:
                cellpose_channels = "color"
            else:
                cellpose_channels = "grayscale"
        else:
            cellpose_channels = "grayscale"
    else:
        cellpose_channels_text = (
            str(cellpose_channels_raw).strip().lower() or "grayscale"
        )
        if cellpose_channels_text in {"grayscale", "gray", "mono", "0,0", "[0,0]"}:
            cellpose_channels = "grayscale"
        elif cellpose_channels_text in {"color", "rgb", "2,1", "[2,1]"}:
            cellpose_channels = "color"
        else:
            tokens = _normalize_parameter_string_list(cellpose_channels_text)
            if len(tokens) == 2:
                try:
                    parsed_pair = [int(tokens[0]), int(tokens[1])]
                except (TypeError, ValueError):
                    parsed_pair = []
                if parsed_pair == [0, 0]:
                    cellpose_channels = "grayscale"
                elif len(parsed_pair) == 2:
                    cellpose_channels = "color"
                else:
                    raise ValueError(
                        "usegment3d cellpose_channels must be 'grayscale' or 'color'."
                    )
            else:
                raise ValueError(
                    "usegment3d cellpose_channels must be 'grayscale' or 'color'."
                )
    normalized["cellpose_channels"] = cellpose_channels
    normalized["cellpose_hist_norm"] = bool(normalized.get("cellpose_hist_norm", False))
    normalized["cellpose_ksize"] = max(3, int(normalized.get("cellpose_ksize", 15)))

    best_diameter = normalized.get(
        "cellpose_best_diameter",
        normalized.get("cellpose_diameter", None),
    )
    use_auto_diameter = bool(normalized.get("cellpose_use_auto_diameter", False))
    if best_diameter in (None, "", 0, 0.0):
        normalized["cellpose_best_diameter"] = None
        if "cellpose_diameter" in normalized and not use_auto_diameter:
            use_auto_diameter = True
    else:
        parsed_best_diameter = float(best_diameter)
        if parsed_best_diameter <= 0:
            raise ValueError("usegment3d cellpose_best_diameter must be positive.")
        normalized["cellpose_best_diameter"] = parsed_best_diameter
    normalized["cellpose_use_auto_diameter"] = bool(use_auto_diameter)

    diameter_range = normalized.get("cellpose_diameter_range", [10.0, 120.0, 2.5])
    if isinstance(diameter_range, str):
        diameter_tokens = _normalize_parameter_float_list(diameter_range)
    elif isinstance(diameter_range, (tuple, list)):
        diameter_tokens = [float(value) for value in diameter_range]
    else:
        diameter_tokens = []
    if len(diameter_tokens) != 3:
        raise ValueError(
            "usegment3d cellpose_diameter_range must define [min, max, step]."
        )
    diameter_min, diameter_max, diameter_step = diameter_tokens
    if diameter_min <= 0 or diameter_max < diameter_min or diameter_step <= 0:
        raise ValueError(
            "usegment3d cellpose_diameter_range must satisfy min>0, max>=min, step>0."
        )
    normalized["cellpose_diameter_range"] = [
        float(diameter_min),
        float(diameter_max),
        float(diameter_step),
    ]
    normalized["cellpose_use_edge"] = bool(
        normalized.get("cellpose_use_edge", normalized.get("cellpose_edge", True))
    )
    normalized["cellpose_model_invert"] = bool(
        normalized.get("cellpose_model_invert", False)
    )
    test_slice = normalized.get("cellpose_test_slice", None)
    if test_slice in (None, ""):
        normalized["cellpose_test_slice"] = None
    else:
        parsed_test_slice = int(test_slice)
        if parsed_test_slice < 0:
            raise ValueError("usegment3d cellpose_test_slice cannot be negative.")
        normalized["cellpose_test_slice"] = parsed_test_slice

    normalized["aggregation_gradient_decay"] = float(
        normalized.get(
            "aggregation_gradient_decay",
            normalized.get("aggregation_decay", 0.0),
        )
    )
    if normalized["aggregation_gradient_decay"] < 0:
        raise ValueError("usegment3d aggregation_gradient_decay cannot be negative.")
    normalized["aggregation_n_iter"] = max(
        1,
        int(
            normalized.get(
                "aggregation_n_iter",
                normalized.get("aggregation_iterations", 200),
            )
        ),
    )
    normalized["aggregation_momenta"] = float(
        normalized.get(
            "aggregation_momenta", normalized.get("aggregation_momentum", 0.98)
        )
    )
    if normalized["aggregation_momenta"] < 0 or normalized["aggregation_momenta"] > 1:
        raise ValueError("usegment3d aggregation_momenta must be within [0, 1].")
    probability_threshold = normalized.get("aggregation_prob_threshold", None)
    if probability_threshold in (None, ""):
        normalized["aggregation_prob_threshold"] = None
    else:
        parsed_probability_threshold = float(probability_threshold)
        if parsed_probability_threshold <= 0 or parsed_probability_threshold >= 1:
            raise ValueError(
                "usegment3d aggregation_prob_threshold must satisfy 0 < value < 1."
            )
        normalized["aggregation_prob_threshold"] = parsed_probability_threshold
    threshold_n_levels = max(
        2, int(normalized.get("aggregation_threshold_n_levels", 3))
    )
    threshold_level = int(normalized.get("aggregation_threshold_level", 1))
    if threshold_level < 0 or threshold_level >= threshold_n_levels - 1:
        raise ValueError(
            "usegment3d aggregation_threshold_level must be within [0, n_levels - 2]."
        )
    normalized["aggregation_threshold_n_levels"] = threshold_n_levels
    normalized["aggregation_threshold_level"] = threshold_level
    normalized["aggregation_min_prob_threshold"] = float(
        normalized.get("aggregation_min_prob_threshold", 0.0)
    )
    if (
        normalized["aggregation_min_prob_threshold"] < 0
        or normalized["aggregation_min_prob_threshold"] >= 1
    ):
        raise ValueError(
            "usegment3d aggregation_min_prob_threshold must satisfy 0 <= value < 1."
        )
    normalized["aggregation_connected_min_area"] = max(
        1, int(normalized.get("aggregation_connected_min_area", 5))
    )
    normalized["aggregation_connected_smooth_sigma"] = float(
        normalized.get("aggregation_connected_smooth_sigma", 1.0)
    )
    if normalized["aggregation_connected_smooth_sigma"] < 0:
        raise ValueError(
            "usegment3d aggregation_connected_smooth_sigma cannot be negative."
        )
    normalized["aggregation_connected_thresh_factor"] = float(
        normalized.get("aggregation_connected_thresh_factor", 0.0)
    )
    normalized["aggregation_binary_fill_holes"] = bool(
        normalized.get("aggregation_binary_fill_holes", False)
    )

    tile_mode_raw = str(normalized.get("tile_mode", "")).strip().lower()
    if tile_mode_raw == "none":
        aggregation_tile_mode = False
    elif tile_mode_raw in {"auto", "manual"}:
        aggregation_tile_mode = True
    else:
        aggregation_tile_mode = bool(normalized.get("aggregation_tile_mode", False))
    normalized["aggregation_tile_mode"] = aggregation_tile_mode
    normalized["aggregation_tile_shape_zyx"] = _normalize_parameter_int_triplet(
        normalized.get(
            "aggregation_tile_shape_zyx",
            normalized.get(
                "tile_shape_zyx",
                normalized.get("tile_shape", [128, 256, 256]),
            ),
        ),
        field_name="usegment3d aggregation_tile_shape_zyx",
        default=(128, 256, 256),
    )
    tile_overlap_zyx = normalized.get(
        "tile_overlap_zyx",
        normalized.get("tile_overlap", [0, 0, 0]),
    )
    tile_overlap_ratio = normalized.get("aggregation_tile_overlap_ratio", None)
    if tile_overlap_ratio in (None, ""):
        if (
            isinstance(tile_overlap_zyx, (tuple, list))
            and len(tile_overlap_zyx) == 3
            and all(int(v) >= 0 for v in tile_overlap_zyx)
        ):
            shape_zyx = normalized["aggregation_tile_shape_zyx"]
            overlap_values = [max(0, int(v)) for v in tile_overlap_zyx]
            ratio_candidates = [
                (float(overlap_values[idx]) / float(shape_zyx[idx]))
                if int(shape_zyx[idx]) > 0
                else 0.0
                for idx in range(3)
            ]
            tile_overlap_ratio = max(ratio_candidates) if ratio_candidates else 0.0
        else:
            tile_overlap_ratio = 0.25
    normalized["aggregation_tile_overlap_ratio"] = float(tile_overlap_ratio)
    if (
        normalized["aggregation_tile_overlap_ratio"] < 0
        or normalized["aggregation_tile_overlap_ratio"] >= 1
    ):
        raise ValueError(
            "usegment3d aggregation_tile_overlap_ratio must satisfy 0 <= value < 1."
        )
    n_cpu = normalized.get("n_cpu", None)
    if n_cpu in (None, ""):
        normalized["n_cpu"] = None
    else:
        parsed_n_cpu = int(n_cpu)
        if parsed_n_cpu <= 0:
            raise ValueError("usegment3d n_cpu must be greater than zero when set.")
        normalized["n_cpu"] = parsed_n_cpu

    normalized["postprocess_enable"] = bool(
        normalized.get(
            "postprocess_enable",
            normalized.get("postprocess", normalized.get("postprocess_enabled", True)),
        )
    )
    normalized["postprocess_min_size"] = max(
        0, int(normalized.get("postprocess_min_size", 200))
    )
    normalized["postprocess_do_flow_remove"] = bool(
        normalized.get(
            "postprocess_do_flow_remove",
            normalized["postprocess_enable"],
        )
    )
    normalized["postprocess_flow_threshold"] = float(
        normalized.get(
            "postprocess_flow_threshold",
            normalized.get("flow_threshold", 0.85),
        )
    )
    if (
        normalized["postprocess_do_flow_remove"]
        and normalized["postprocess_flow_threshold"] <= 0
    ):
        raise ValueError("usegment3d postprocess_flow_threshold must be positive.")
    dtform_method = (
        str(
            normalized.get(
                "postprocess_dtform_method",
                normalized.get("postprocess_dtform", "cellpose_improve"),
            )
        )
        .strip()
        .lower()
        or "cellpose_improve"
    )
    if dtform_method == "cellpose":
        dtform_method = "cellpose_improve"
    elif dtform_method == "none":
        normalized["postprocess_do_flow_remove"] = False
        dtform_method = "edt"
    if dtform_method not in {
        "cellpose_improve",
        "edt",
        "fmm",
        "fmm_skel",
        "cellpose_skel",
    }:
        raise ValueError(
            "usegment3d postprocess_dtform_method must be one of "
            "cellpose_improve, edt, fmm, fmm_skel, cellpose_skel."
        )
    normalized["postprocess_dtform_method"] = dtform_method
    normalized["postprocess_edt_fixed_point_percentile"] = float(
        normalized.get("postprocess_edt_fixed_point_percentile", 0.01)
    )
    if (
        normalized["postprocess_edt_fixed_point_percentile"] < 0
        or normalized["postprocess_edt_fixed_point_percentile"] > 1
    ):
        raise ValueError(
            "usegment3d postprocess_edt_fixed_point_percentile must be within [0, 1]."
        )

    output_dtype = (
        str(normalized.get("output_dtype", "uint32")).strip().lower() or "uint32"
    )
    if output_dtype not in {"uint16", "uint32", "int32"}:
        raise ValueError("usegment3d output_dtype must be uint16, uint32, or int32.")
    normalized["output_dtype"] = output_dtype
    return normalized


def _normalize_shear_transform_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize shear-transform runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate shear-transform parameters.

    Returns
    -------
    dict[str, Any]
        Normalized shear-transform parameters.

    Raises
    ------
    ValueError
        If required values are invalid.
    """
    normalized = _normalize_common_operation_parameters("shear_transform", params)
    normalized["detect_2d_per_slice"] = False
    for axis_suffix in ("xy", "xz", "yz"):
        shear_key = f"shear_{axis_suffix}"
        shear_deg_key = f"shear_{axis_suffix}_deg"
        if shear_deg_key in normalized:
            angle_deg = float(normalized.get(shear_deg_key, 0.0))
            normalized[shear_deg_key] = angle_deg
            normalized[shear_key] = float(math.tan(math.radians(angle_deg)))
            continue
        shear_value = float(normalized.get(shear_key, 0.0))
        normalized[shear_key] = shear_value
        normalized[shear_deg_key] = float(math.degrees(math.atan(shear_value)))
    normalized["rotation_deg_x"] = float(normalized.get("rotation_deg_x", 0.0))
    normalized["rotation_deg_y"] = float(normalized.get("rotation_deg_y", 0.0))
    normalized["rotation_deg_z"] = float(normalized.get("rotation_deg_z", 0.0))
    normalized["auto_rotate_from_shear"] = bool(
        normalized.get("auto_rotate_from_shear", False)
    )
    normalized["auto_estimate_shear_yz"] = bool(
        normalized.get("auto_estimate_shear_yz", False)
    )
    normalized["auto_estimate_extreme_fraction_x"] = float(
        normalized.get("auto_estimate_extreme_fraction_x", 0.03)
    )
    if (
        normalized["auto_estimate_extreme_fraction_x"] <= 0.0
        or normalized["auto_estimate_extreme_fraction_x"] > 0.5
    ):
        raise ValueError(
            "shear_transform auto_estimate_extreme_fraction_x must be within (0, 0.5]."
        )
    normalized["auto_estimate_zy_stride"] = max(
        1, int(normalized.get("auto_estimate_zy_stride", 4))
    )
    normalized["auto_estimate_signal_fraction"] = float(
        normalized.get("auto_estimate_signal_fraction", 0.10)
    )
    if (
        normalized["auto_estimate_signal_fraction"] < 0.0
        or normalized["auto_estimate_signal_fraction"] > 1.0
    ):
        raise ValueError(
            "shear_transform auto_estimate_signal_fraction must be within [0, 1]."
        )
    normalized["auto_estimate_t_index"] = max(
        0, int(normalized.get("auto_estimate_t_index", 0))
    )
    normalized["auto_estimate_p_index"] = max(
        0, int(normalized.get("auto_estimate_p_index", 0))
    )
    normalized["auto_estimate_c_index"] = max(
        0, int(normalized.get("auto_estimate_c_index", 0))
    )
    interpolation = (
        str(normalized.get("interpolation", "linear")).strip().lower() or "linear"
    )
    if interpolation not in {"linear", "nearestneighbor", "bspline"}:
        raise ValueError(
            "shear_transform interpolation must be one of "
            "linear, nearestneighbor, or bspline."
        )
    normalized["interpolation"] = interpolation
    normalized["fill_value"] = float(normalized.get("fill_value", 0.0))
    normalized["output_dtype"] = (
        str(normalized.get("output_dtype", "float32")).strip().lower() or "float32"
    )
    roi_padding = normalized.get("roi_padding_zyx", [2, 2, 2])
    if not isinstance(roi_padding, (tuple, list)) or len(roi_padding) != 3:
        raise ValueError(
            "shear_transform roi_padding_zyx must define three integers in (z, y, x) order."
        )
    normalized["roi_padding_zyx"] = [max(0, int(v)) for v in roi_padding]
    return normalized


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    """Coerce a value into ``True``/``False``/``None``.

    Parameters
    ----------
    value : Any
        Candidate boolean-like value.

    Returns
    -------
    bool, optional
        Parsed boolean, or ``None`` when unspecified.

    Raises
    ------
    None
        Invalid values return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if not text or text == "auto":
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_keyframe_layer_overrides(value: Any) -> list[Dict[str, Any]]:
    """Normalize visualization keyframe layer override rows.

    Parameters
    ----------
    value : Any
        Candidate list of row mappings.

    Returns
    -------
    list[dict[str, Any]]
        Normalized rows containing ``layer_name``, ``visible``, ``colormap``,
        ``rendering``, and ``annotation``.

    Raises
    ------
    None
        Invalid rows are skipped.
    """
    if not isinstance(value, (tuple, list)):
        return []

    rows: list[Dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, dict):
            continue
        layer_name = str(raw_row.get("layer_name", raw_row.get("layer", ""))).strip()
        visible = _coerce_optional_bool(raw_row.get("visible"))
        colormap = str(raw_row.get("colormap", raw_row.get("lut", ""))).strip()
        rendering = str(raw_row.get("rendering", "")).strip()
        annotation = str(raw_row.get("annotation", "")).strip()
        if not any(
            (
                layer_name,
                colormap,
                rendering,
                annotation,
                visible is not None,
            )
        ):
            continue
        rows.append(
            {
                "layer_name": layer_name,
                "visible": visible,
                "colormap": colormap,
                "rendering": rendering,
                "annotation": annotation,
            }
        )
    return rows


def _normalize_visualization_channels(value: Any) -> list[int]:
    """Normalize channel-selection values for visualization layer rows.

    Parameters
    ----------
    value : Any
        Candidate channel-selection value.

    Returns
    -------
    list[int]
        Unique non-negative channel indices in sorted order. Empty means
        "all available channels".

    Raises
    ------
    None
        Invalid values return an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"all", "auto"}:
            return []
        items = [part.strip() for part in text.split(",")]
    elif isinstance(value, (tuple, list)):
        items = list(value)
    else:
        items = [value]

    parsed: list[int] = []
    for item in items:
        if item is None:
            continue
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        parsed.append(int(index))
    return sorted(set(parsed))


def _coerce_optional_unit_interval_float(value: Any) -> Optional[float]:
    """Coerce optional layer-opacity values into ``[0, 1]`` floats.

    Parameters
    ----------
    value : Any
        Candidate opacity value.

    Returns
    -------
    float, optional
        Parsed opacity value, or ``None`` when unspecified/invalid.

    Raises
    ------
    None
        Invalid values return ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        opacity = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(opacity):
        return None
    return float(min(1.0, max(0.0, opacity)))


def _normalize_visualization_volume_layers(value: Any) -> list[Dict[str, Any]]:
    """Normalize visualization volume-layer configuration rows.

    Parameters
    ----------
    value : Any
        Candidate list-like value containing layer rows.

    Returns
    -------
    list[dict[str, Any]]
        Normalized layer rows with component, display, and multiscale policy
        fields suitable for runtime use.

    Raises
    ------
    None
        Invalid rows are skipped.
    """
    if not isinstance(value, (tuple, list)):
        return []

    rows: list[Dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, Mapping):
            continue
        component = str(
            raw_row.get("component", raw_row.get("source_component", ""))
        ).strip()
        if not component:
            continue
        layer_type = str(raw_row.get("layer_type", "image")).strip().lower() or "image"
        if layer_type not in {"image", "labels"}:
            layer_type = "image"
        multiscale_policy = (
            str(raw_row.get("multiscale_policy", "inherit")).strip().lower()
            or "inherit"
        )
        if multiscale_policy not in {"inherit", "require", "auto_build", "off"}:
            multiscale_policy = "inherit"
        blending = str(raw_row.get("blending", "")).strip().lower()
        if blending in {"auto", "default"}:
            blending = ""
        colormap = str(raw_row.get("colormap", raw_row.get("lut", ""))).strip()
        if colormap.strip().lower() in {"auto", "default"}:
            colormap = ""
        rendering = str(raw_row.get("rendering", "")).strip().lower()
        if rendering in {"auto", "default"}:
            rendering = ""
        rows.append(
            {
                "component": component,
                "name": str(raw_row.get("name", "")).strip(),
                "layer_type": layer_type,
                "channels": _normalize_visualization_channels(raw_row.get("channels")),
                "visible": _coerce_optional_bool(raw_row.get("visible")),
                "opacity": _coerce_optional_unit_interval_float(raw_row.get("opacity")),
                "blending": blending,
                "colormap": colormap,
                "rendering": rendering,
                "multiscale_policy": multiscale_policy,
            }
        )
    return rows


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
    normalized["show_all_positions"] = bool(normalized.get("show_all_positions", False))
    normalized["position_index"] = max(0, int(normalized.get("position_index", 0)))
    normalized["use_multiscale"] = bool(normalized.get("use_multiscale", True))
    normalized["use_3d_view"] = bool(normalized.get("use_3d_view", True))
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
    normalized["require_gpu_rendering"] = bool(
        normalized.get("require_gpu_rendering", True)
    )
    normalized["capture_keyframes"] = bool(normalized.get("capture_keyframes", True))
    normalized["keyframe_manifest_path"] = str(
        normalized.get("keyframe_manifest_path", "")
    ).strip()
    normalized["keyframe_layer_overrides"] = _normalize_keyframe_layer_overrides(
        normalized.get("keyframe_layer_overrides", [])
    )
    normalized["volume_layers"] = _normalize_visualization_volume_layers(
        normalized.get("volume_layers", [])
    )
    if not normalized["volume_layers"]:
        default_source = str(normalized.get("input_source", "data")).strip() or "data"
        normalized["volume_layers"] = [
            {
                "component": default_source,
                "name": "",
                "layer_type": "image",
                "channels": [],
                "visible": None,
                "opacity": None,
                "blending": "",
                "colormap": "",
                "rendering": "",
                "multiscale_policy": (
                    "inherit" if bool(normalized.get("use_multiscale", True)) else "off"
                ),
            }
        ]
    launch_mode = str(normalized.get("launch_mode", "auto")).strip().lower() or "auto"
    if launch_mode not in {"auto", "in_process", "subprocess"}:
        raise ValueError(
            "visualization launch_mode must be one of auto, in_process, subprocess."
        )
    normalized["launch_mode"] = launch_mode
    return normalized


def _normalize_mip_export_parameters(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize maximum-intensity projection export runtime parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Candidate MIP-export parameters.

    Returns
    -------
    dict[str, Any]
        Normalized MIP-export parameters.

    Raises
    ------
    ValueError
        If export mode values are unsupported.
    """
    normalized = _normalize_common_operation_parameters("mip_export", params)
    normalized["chunk_basis"] = "3d"
    normalized["detect_2d_per_slice"] = False
    normalized["use_map_overlap"] = False
    normalized["overlap_zyx"] = [0, 0, 0]

    position_mode = (
        str(normalized.get("position_mode", "multi_position")).strip().lower()
        or "multi_position"
    )
    if position_mode not in {"multi_position", "per_position"}:
        raise ValueError(
            "mip_export position_mode must be 'multi_position' or 'per_position'."
        )
    normalized["position_mode"] = position_mode

    export_format_raw = (
        str(normalized.get("export_format", "ome-tiff")).strip().lower() or "ome-tiff"
    )
    if export_format_raw in {"tiff", "ome-tiff", "ome_tiff", "ome.tiff"}:
        export_format = "ome-tiff"
    elif export_format_raw == "zarr":
        export_format = "zarr"
    else:
        raise ValueError(
            "mip_export export_format must be one of: ome-tiff, tiff, zarr."
        )
    normalized["export_format"] = export_format
    normalized["output_directory"] = str(normalized.get("output_directory", "")).strip()
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
        if operation_name == "flatfield":
            merged[operation_name] = _normalize_flatfield_parameters(
                merged[operation_name]
            )
        elif operation_name == "deconvolution":
            merged[operation_name] = _normalize_deconvolution_parameters(
                merged[operation_name]
            )
        elif operation_name == "shear_transform":
            merged[operation_name] = _normalize_shear_transform_parameters(
                merged[operation_name]
            )
        elif operation_name == "particle_detection":
            merged[operation_name] = _normalize_particle_detection_parameters(
                merged[operation_name]
            )
        elif operation_name == "usegment3d":
            merged[operation_name] = _normalize_usegment3d_parameters(
                merged[operation_name]
            )
        elif operation_name == "visualization":
            merged[operation_name] = _normalize_visualization_parameters(
                merged[operation_name]
            )
        elif operation_name == "mip_export":
            merged[operation_name] = _normalize_mip_export_parameters(
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
    flatfield: bool,
    deconvolution: bool,
    shear_transform: bool,
    particle_detection: bool,
    usegment3d: bool,
    registration: bool,
    visualization: bool,
    mip_export: bool,
) -> Tuple[str, ...]:
    """Collect enabled analysis operation names.

    Parameters
    ----------
    flatfield : bool
        Whether flatfield correction is enabled.
    deconvolution : bool
        Whether deconvolution is enabled.
    shear_transform : bool
        Whether shear transform is enabled.
    particle_detection : bool
        Whether particle detection is enabled.
    usegment3d : bool
        Whether usegment3d segmentation is enabled.
    registration : bool
        Whether registration is enabled.
    visualization : bool
        Whether visualization is enabled.
    mip_export : bool
        Whether MIP export is enabled.

    Returns
    -------
    tuple[str, ...]
        Selected operations in canonical declaration order.
    """
    selected: list[str] = []
    if flatfield:
        selected.append("flatfield")
    if deconvolution:
        selected.append("deconvolution")
    if shear_transform:
        selected.append("shear_transform")
    if particle_detection:
        selected.append("particle_detection")
    if usegment3d:
        selected.append("usegment3d")
    if registration:
        selected.append("registration")
    if visualization:
        selected.append("visualization")
    if mip_export:
        selected.append("mip_export")
    return tuple(selected)


def resolve_analysis_execution_sequence(
    *,
    flatfield: bool,
    deconvolution: bool,
    shear_transform: bool,
    particle_detection: bool,
    usegment3d: bool,
    registration: bool,
    visualization: bool,
    mip_export: bool,
    analysis_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, ...]:
    """Resolve selected analyses into execution order.

    Parameters
    ----------
    flatfield : bool
        Whether flatfield correction is enabled.
    deconvolution : bool
        Whether deconvolution is enabled.
    shear_transform : bool
        Whether shear transform is enabled.
    particle_detection : bool
        Whether particle detection is enabled.
    usegment3d : bool
        Whether usegment3d segmentation is enabled.
    registration : bool
        Whether registration is enabled.
    visualization : bool
        Whether visualization is enabled.
    mip_export : bool
        Whether MIP export is enabled.
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
        flatfield=flatfield,
        deconvolution=deconvolution,
        shear_transform=shear_transform,
        particle_detection=particle_detection,
        usegment3d=usegment3d,
        registration=registration,
        visualization=visualization,
        mip_export=mip_export,
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


def zarr_save_to_dict(config: ZarrSaveConfig) -> Dict[str, Any]:
    """Serialize a Zarr save configuration to a JSON-safe mapping.

    Parameters
    ----------
    config : ZarrSaveConfig
        Zarr save configuration to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-safe mapping containing chunk and pyramid settings in
        ``(p, t, c, z, y, x)`` order.
    """
    return {
        "chunks_ptczyx": [int(value) for value in config.chunks_ptczyx],
        "pyramid_ptczyx": [
            [int(value) for value in levels] for levels in config.pyramid_ptczyx
        ],
    }


def zarr_save_from_dict(payload: Any) -> ZarrSaveConfig:
    """Deserialize a Zarr save configuration from a mapping payload.

    Parameters
    ----------
    payload : Any
        Mapping payload produced by :func:`zarr_save_to_dict`.

    Returns
    -------
    ZarrSaveConfig
        Parsed Zarr save configuration.

    Raises
    ------
    ValueError
        If the payload is not a mapping containing valid Zarr save settings.
    """
    if not isinstance(payload, dict):
        raise ValueError("Zarr save settings must be a JSON object.")
    return ZarrSaveConfig(
        chunks_ptczyx=payload.get("chunks_ptczyx", DEFAULT_ZARR_CHUNKS_PTCZYX),
        pyramid_ptczyx=payload.get("pyramid_ptczyx", DEFAULT_ZARR_PYRAMID_PTCZYX),
    )


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


@dataclass(frozen=True)
class LocalClusterRecommendation:
    """Recommended LocalCluster settings derived from host and data context.

    Attributes
    ----------
    config : LocalClusterConfig
        Recommended LocalCluster configuration.
    detected_cpu_count : int
        Effective CPU count detected for the current process.
    detected_memory_bytes : int
        Effective memory budget detected for the current process.
    detected_gpu_count : int
        Number of locally visible GPUs.
    detected_gpu_memory_bytes : int, optional
        Aggregate GPU memory in bytes when detectable.
    estimated_chunk_bytes : int
        Estimated bytes per canonical chunk using the configured save chunks.
    estimated_dataset_bytes : int, optional
        Estimated canonical dataset size in bytes when shape metadata is known.
    estimated_chunk_count : int, optional
        Estimated number of canonical chunks when shape metadata is known.
    """

    config: LocalClusterConfig
    detected_cpu_count: int
    detected_memory_bytes: int
    detected_gpu_count: int
    detected_gpu_memory_bytes: Optional[int]
    estimated_chunk_bytes: int
    estimated_dataset_bytes: Optional[int]
    estimated_chunk_count: Optional[int]


def _detect_local_cpu_count() -> int:
    """Return the effective CPU count for local scheduling decisions.

    Parameters
    ----------
    None

    Returns
    -------
    int
        CPU count respecting process affinity when available.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, int(os.cpu_count() or 1))


def _detect_cgroup_memory_limit_bytes() -> Optional[int]:
    """Return Linux cgroup memory limit when one is enforced.

    Parameters
    ----------
    None

    Returns
    -------
    int, optional
        Positive cgroup limit in bytes when detectable, otherwise ``None``.
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in candidates:
        try:
            raw = open(path, "r", encoding="utf-8").read().strip()
        except Exception:
            continue
        if not raw or raw.lower() == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= (1 << 60):
            continue
        return value
    return None


def _detect_local_memory_bytes() -> int:
    """Return effective memory budget for local scheduling heuristics.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Effective memory bytes, constrained by cgroups when detectable.
    """
    physical_total: Optional[int] = None
    try:
        import psutil

        physical_total = int(psutil.virtual_memory().total)
    except Exception:
        physical_total = None

    if physical_total is None and os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                physical_total = int(status.ullTotalPhys)
        except Exception:
            physical_total = None

    if physical_total is None and hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and phys_pages > 0:
                physical_total = page_size * phys_pages
        except Exception:
            physical_total = None

    if physical_total is None or physical_total <= 0:
        physical_total = 8 << 30

    cgroup_limit = _detect_cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        return max(1 << 30, min(int(physical_total), int(cgroup_limit)))
    return max(1 << 30, int(physical_total))


def _detect_local_gpu_info() -> tuple[int, Optional[int]]:
    """Return local GPU count and aggregate memory when detectable.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[int, int | None]
        ``(gpu_count, total_gpu_memory_bytes)``.
        Memory is ``None`` when not available.

    Notes
    -----
    Detection prefers NVML via ``pynvml`` and falls back to ``nvidia-smi``,
    ``torch.cuda``, and ``CUDA_VISIBLE_DEVICES``.
    Failure to query GPU information is non-fatal and returns ``(0, None)``.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            count = max(0, int(pynvml.nvmlDeviceGetCount()))
            total_bytes = 0
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_bytes += max(0, int(getattr(memory_info, "total", 0)))
            return count, (int(total_bytes) if total_bytes > 0 else None)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        result = None

    if result is not None and int(result.returncode) == 0:
        memories_mib: list[int] = []
        for line in result.stdout.splitlines():
            text = str(line).strip()
            if not text:
                continue
            try:
                parsed = int(float(text))
            except ValueError:
                continue
            if parsed > 0:
                memories_mib.append(parsed)

        if memories_mib:
            total_bytes = int(sum(memories_mib)) << 20
            return len(memories_mib), (total_bytes if total_bytes > 0 else None)

    try:
        import torch

        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is not None and callable(
            getattr(cuda_module, "is_available", None)
        ):
            if bool(cuda_module.is_available()):
                count = max(0, int(cuda_module.device_count()))
                if count > 0:
                    total_bytes = 0
                    get_props = getattr(cuda_module, "get_device_properties", None)
                    if callable(get_props):
                        for index in range(count):
                            try:
                                props = get_props(index)
                                total_bytes += max(
                                    0,
                                    int(getattr(props, "total_memory", 0)),
                                )
                            except Exception:
                                continue
                    return count, (int(total_bytes) if total_bytes > 0 else None)
    except Exception:
        pass

    raw_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if raw_visible_devices:
        lowered = raw_visible_devices.lower()
        if lowered not in {"-1", "none", "void", "nodevfiles"}:
            visible_devices = [
                token.strip()
                for token in raw_visible_devices.split(",")
                if token.strip()
            ]
            if visible_devices:
                return len(visible_devices), None

    return 0, None


def _format_binary_size(num_bytes: int) -> str:
    """Format byte counts in binary units for GUI display.

    Parameters
    ----------
    num_bytes : int
        Byte count.

    Returns
    -------
    str
        Human-readable binary-size string.
    """
    value = max(1, int(num_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{value}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def _format_worker_memory_limit(num_bytes: int) -> str:
    """Format worker memory limits in Dask-friendly units.

    Parameters
    ----------
    num_bytes : int
        Byte count to format.

    Returns
    -------
    str
        Dask-friendly memory-limit string.
    """
    rounded = max(512 << 20, (int(num_bytes) // (256 << 20)) * (256 << 20))
    gib = 1 << 30
    mib = 1 << 20
    if rounded % gib == 0:
        return f"{rounded // gib}GiB"
    return f"{rounded // mib}MiB"


def recommend_local_cluster_config(
    *,
    shape_tpczyx: Optional[Tuple[int, int, int, int, int, int]] = None,
    chunks_tpczyx: Optional[Tuple[int, int, int, int, int, int]] = None,
    dtype_itemsize: Optional[int] = None,
    cpu_count: Optional[int] = None,
    memory_bytes: Optional[int] = None,
    gpu_count: Optional[int] = None,
    gpu_memory_bytes: Optional[int] = None,
) -> LocalClusterRecommendation:
    """Recommend aggressive LocalCluster settings from host/data characteristics.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int], optional
        Canonical dataset shape in ``(t, p, c, z, y, x)`` order.
    chunks_tpczyx : tuple[int, int, int, int, int, int], optional
        Canonical chunk shape in ``(t, p, c, z, y, x)`` order.
    dtype_itemsize : int, optional
        Bytes per voxel. Defaults to ``2`` when unknown.
    cpu_count : int, optional
        Explicit effective CPU count for deterministic testing.
    memory_bytes : int, optional
        Explicit effective memory budget in bytes for deterministic testing.
    gpu_count : int, optional
        Explicit GPU count for deterministic testing.
    gpu_memory_bytes : int, optional
        Explicit aggregate GPU memory in bytes for deterministic testing.

    Returns
    -------
    LocalClusterRecommendation
        Recommended LocalCluster configuration plus sizing diagnostics.

    Notes
    -----
    Recommendations are intentionally aggressive by default: they target high
    CPU utilization while respecting chunk-memory pressure and available RAM.
    GPU presence is incorporated as additional diagnostic context and can raise
    target worker counts when resources permit.
    """
    detected_cpus = max(1, int(cpu_count or _detect_local_cpu_count()))
    detected_memory = max(1 << 30, int(memory_bytes or _detect_local_memory_bytes()))
    probed_gpu_count: int = 0
    probed_gpu_memory_bytes: Optional[int] = None
    if gpu_count is None or gpu_memory_bytes is None:
        probed_gpu_count, probed_gpu_memory_bytes = _detect_local_gpu_info()
    detected_gpu_count = max(
        0,
        int(probed_gpu_count if gpu_count is None else gpu_count),
    )
    if detected_gpu_count == 0:
        detected_gpu_memory_bytes: Optional[int] = None
    else:
        detected_gpu_memory_bytes = (
            max(1, int(gpu_memory_bytes))
            if gpu_memory_bytes is not None
            else (
                None
                if probed_gpu_memory_bytes is None
                else max(1, int(probed_gpu_memory_bytes))
            )
        )

    itemsize = max(1, int(dtype_itemsize or 2))
    effective_chunks = tuple(
        int(v) for v in (chunks_tpczyx or (1, 1, 1, 256, 256, 256))
    )
    estimated_chunk_bytes = max(1, math.prod(effective_chunks) * itemsize)
    estimated_dataset_bytes: Optional[int] = None
    estimated_chunk_count: Optional[int] = None

    if shape_tpczyx is not None:
        effective_shape = tuple(int(v) for v in shape_tpczyx)
        estimated_dataset_bytes = max(1, math.prod(effective_shape) * itemsize)
        chunk_counts = [
            max(1, math.ceil(int(dim) / max(1, int(chunk))))
            for dim, chunk in zip(effective_shape, effective_chunks, strict=False)
        ]
        estimated_chunk_count = max(1, math.prod(chunk_counts))

    reserve_bytes = min(
        max(2 << 30, detected_memory // 10),
        max(1 << 30, detected_memory // 6),
    )
    usable_bytes = max(1 << 30, detected_memory - reserve_bytes)

    threads_per_worker = 1
    if detected_cpus >= 32 and estimated_chunk_bytes < (16 << 20):
        threads_per_worker = 2

    max_workers_by_cpu = max(1, detected_cpus // threads_per_worker)
    min_bytes_per_worker = max(1 << 30, estimated_chunk_bytes * 3)
    max_workers_by_memory = max(1, usable_bytes // min_bytes_per_worker)

    recommended_workers = min(max_workers_by_cpu, max_workers_by_memory, 64)
    if estimated_chunk_count is not None and detected_gpu_count == 0:
        recommended_workers = min(recommended_workers, estimated_chunk_count)
    if detected_gpu_count > 0:
        workers_per_gpu = 4 if estimated_chunk_bytes < (256 << 20) else 2
        gpu_target_workers = max(1, int(detected_gpu_count) * int(workers_per_gpu))
        recommended_workers = max(
            int(recommended_workers),
            min(max_workers_by_cpu, max_workers_by_memory, gpu_target_workers),
        )
    recommended_workers = max(1, int(recommended_workers))

    if recommended_workers == 1:
        threads_per_worker = max(
            1,
            min(
                detected_cpus,
                2 if estimated_chunk_bytes >= (512 << 20) else 6,
            ),
        )

    if recommended_workers * threads_per_worker > detected_cpus:
        threads_per_worker = max(1, detected_cpus // recommended_workers)

    worker_memory_bytes = max(
        min_bytes_per_worker,
        usable_bytes // max(1, recommended_workers),
    )
    config = LocalClusterConfig(
        n_workers=int(recommended_workers),
        threads_per_worker=int(threads_per_worker),
        memory_limit=_format_worker_memory_limit(worker_memory_bytes),
    )
    return LocalClusterRecommendation(
        config=config,
        detected_cpu_count=int(detected_cpus),
        detected_memory_bytes=int(detected_memory),
        detected_gpu_count=int(detected_gpu_count),
        detected_gpu_memory_bytes=(
            None
            if detected_gpu_memory_bytes is None
            else int(detected_gpu_memory_bytes)
        ),
        estimated_chunk_bytes=int(estimated_chunk_bytes),
        estimated_dataset_bytes=(
            None if estimated_dataset_bytes is None else int(estimated_dataset_bytes)
        ),
        estimated_chunk_count=(
            None if estimated_chunk_count is None else int(estimated_chunk_count)
        ),
    )


def format_local_cluster_recommendation_summary(
    recommendation: LocalClusterRecommendation,
) -> str:
    """Format a human-readable summary of local-cluster recommendations.

    Parameters
    ----------
    recommendation : LocalClusterRecommendation
        Recommendation payload to summarize.

    Returns
    -------
    str
        Concise operator-facing summary string.
    """
    config = recommendation.config
    parts = [
        f"Detected {recommendation.detected_cpu_count} CPUs",
        f"{_format_binary_size(recommendation.detected_memory_bytes)} memory",
        f"~{_format_binary_size(recommendation.estimated_chunk_bytes)} per chunk",
        f"recommend {config.n_workers} worker(s)",
        f"{config.threads_per_worker} thread(s)/worker",
        f"{config.memory_limit} memory/worker",
    ]
    if recommendation.detected_gpu_count > 0:
        gpu_text = f"{recommendation.detected_gpu_count} GPU(s)"
        if recommendation.detected_gpu_memory_bytes is not None:
            gpu_text += (
                " "
                f"({_format_binary_size(recommendation.detected_gpu_memory_bytes)} total)"
            )
        parts.insert(2, gpu_text)
    if recommendation.estimated_dataset_bytes is not None:
        parts.insert(
            3,
            f"~{_format_binary_size(recommendation.estimated_dataset_bytes)} dataset",
        )
    return " | ".join(parts)


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


def dask_backend_from_dict(payload: Any) -> DaskBackendConfig:
    """Deserialize a backend mapping into ``DaskBackendConfig``.

    Parameters
    ----------
    payload : Any
        JSON-like mapping, typically produced by :func:`dask_backend_to_dict`.

    Returns
    -------
    DaskBackendConfig
        Parsed backend configuration. Invalid or partial payloads fall back to
        default values for affected sections.

    Notes
    -----
    This parser is intentionally tolerant for user-level persisted settings.
    Unknown keys are ignored and malformed subsections are replaced with
    defaults so GUI startup can always proceed.
    """
    defaults = DaskBackendConfig()
    if not isinstance(payload, dict):
        return defaults

    local_payload = payload.get("local_cluster")
    runner_payload = payload.get("slurm_runner")
    cluster_payload = payload.get("slurm_cluster")
    scheduler_options_payload: Any = (
        cluster_payload.get("scheduler_options")
        if isinstance(cluster_payload, dict)
        else None
    )

    try:
        local_cluster = (
            LocalClusterConfig(
                n_workers=local_payload.get("n_workers"),
                threads_per_worker=local_payload.get(
                    "threads_per_worker",
                    defaults.local_cluster.threads_per_worker,
                ),
                memory_limit=local_payload.get(
                    "memory_limit",
                    defaults.local_cluster.memory_limit,
                ),
                local_directory=local_payload.get(
                    "local_directory",
                    defaults.local_cluster.local_directory,
                ),
            )
            if isinstance(local_payload, dict)
            else defaults.local_cluster
        )
    except Exception:
        local_cluster = defaults.local_cluster

    try:
        slurm_runner = (
            SlurmRunnerConfig(
                scheduler_file=runner_payload.get("scheduler_file"),
                wait_for_workers=runner_payload.get("wait_for_workers"),
            )
            if isinstance(runner_payload, dict)
            else defaults.slurm_runner
        )
    except Exception:
        slurm_runner = defaults.slurm_runner

    try:
        slurm_cluster = (
            SlurmClusterConfig(
                workers=cluster_payload.get("workers", defaults.slurm_cluster.workers),
                cores=cluster_payload.get("cores", defaults.slurm_cluster.cores),
                processes=cluster_payload.get(
                    "processes", defaults.slurm_cluster.processes
                ),
                memory=cluster_payload.get("memory", defaults.slurm_cluster.memory),
                local_directory=cluster_payload.get(
                    "local_directory",
                    defaults.slurm_cluster.local_directory,
                ),
                interface=cluster_payload.get(
                    "interface", defaults.slurm_cluster.interface
                ),
                walltime=cluster_payload.get(
                    "walltime", defaults.slurm_cluster.walltime
                ),
                job_name=cluster_payload.get(
                    "job_name", defaults.slurm_cluster.job_name
                ),
                queue=cluster_payload.get("queue", defaults.slurm_cluster.queue),
                death_timeout=cluster_payload.get(
                    "death_timeout",
                    defaults.slurm_cluster.death_timeout,
                ),
                mail_user=cluster_payload.get(
                    "mail_user", defaults.slurm_cluster.mail_user
                ),
                job_extra_directives=tuple(
                    cluster_payload.get(
                        "job_extra_directives",
                        defaults.slurm_cluster.job_extra_directives,
                    )
                ),
                dashboard_address=(
                    scheduler_options_payload.get(
                        "dashboard_address",
                        defaults.slurm_cluster.dashboard_address,
                    )
                    if isinstance(scheduler_options_payload, dict)
                    else defaults.slurm_cluster.dashboard_address
                ),
                scheduler_interface=(
                    scheduler_options_payload.get(
                        "interface",
                        defaults.slurm_cluster.scheduler_interface,
                    )
                    if isinstance(scheduler_options_payload, dict)
                    else defaults.slurm_cluster.scheduler_interface
                ),
                idle_timeout=(
                    scheduler_options_payload.get(
                        "idle_timeout",
                        defaults.slurm_cluster.idle_timeout,
                    )
                    if isinstance(scheduler_options_payload, dict)
                    else defaults.slurm_cluster.idle_timeout
                ),
                allowed_failures=(
                    scheduler_options_payload.get(
                        "allowed_failures",
                        defaults.slurm_cluster.allowed_failures,
                    )
                    if isinstance(scheduler_options_payload, dict)
                    else defaults.slurm_cluster.allowed_failures
                ),
            )
            if isinstance(cluster_payload, dict)
            else defaults.slurm_cluster
        )
    except Exception:
        slurm_cluster = defaults.slurm_cluster

    mode_value = str(payload.get("mode", defaults.mode)).strip().lower()
    mode = (
        mode_value
        if mode_value in DASK_BACKEND_MODE_LABELS
        else DASK_BACKEND_LOCAL_CLUSTER
    )

    try:
        return DaskBackendConfig(
            mode=mode,
            local_cluster=local_cluster,
            slurm_runner=slurm_runner,
            slurm_cluster=slurm_cluster,
        )
    except Exception:
        return defaults


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


def _normalize_spatial_calibration_binding(
    value: Any,
    *,
    axis_name: str,
) -> str:
    """Normalize one world-axis spatial-calibration binding.

    Parameters
    ----------
    value : Any
        Candidate binding value.
    axis_name : str
        World axis receiving the binding for error context.

    Returns
    -------
    str
        Canonical lowercase binding.

    Raises
    ------
    ValueError
        If the binding is empty or unsupported.
    """
    text = str(value).strip().lower()
    if not text:
        raise ValueError(
            f"Spatial calibration binding for world axis '{axis_name}' cannot be empty."
        )
    if text in SPATIAL_CALIBRATION_SOURCE_AXES:
        text = f"+{text}"
    if text not in SPATIAL_CALIBRATION_ALLOWED_BINDINGS:
        allowed = ", ".join(sorted(SPATIAL_CALIBRATION_ALLOWED_BINDINGS))
        raise ValueError(
            f"Spatial calibration binding for world axis '{axis_name}' must be one "
            f"of: {allowed}."
        )
    return text


def _normalize_spatial_calibration_stage_axis_map(
    stage_axis_map_zyx: Any,
) -> tuple[str, str, str]:
    """Normalize world ``z/y/x`` axis bindings for spatial calibration.

    Parameters
    ----------
    stage_axis_map_zyx : Any
        Candidate binding payload. Accepts a sequence in ``(z, y, x)`` order or
        a mapping with ``z``, ``y``, and ``x`` keys.

    Returns
    -------
    tuple[str, str, str]
        Canonical world-axis bindings in ``(z, y, x)`` order.

    Raises
    ------
    ValueError
        If the mapping is malformed or reuses one source axis more than once.
    """
    if isinstance(stage_axis_map_zyx, Mapping):
        missing_axes = [
            axis_name
            for axis_name in SPATIAL_CALIBRATION_WORLD_AXES
            if axis_name not in stage_axis_map_zyx
        ]
        if missing_axes:
            raise ValueError(
                "Spatial calibration mappings must define z, y, and x bindings."
            )
        normalized = (
            _normalize_spatial_calibration_binding(
                stage_axis_map_zyx["z"],
                axis_name="z",
            ),
            _normalize_spatial_calibration_binding(
                stage_axis_map_zyx["y"],
                axis_name="y",
            ),
            _normalize_spatial_calibration_binding(
                stage_axis_map_zyx["x"],
                axis_name="x",
            ),
        )
    elif isinstance(stage_axis_map_zyx, Sequence) and not isinstance(
        stage_axis_map_zyx, (str, bytes)
    ):
        values = tuple(stage_axis_map_zyx)
        if len(values) != len(SPATIAL_CALIBRATION_WORLD_AXES):
            raise ValueError(
                "Spatial calibration stage_axis_map_zyx must define three "
                "entries in (z, y, x) order."
            )
        normalized = (
            _normalize_spatial_calibration_binding(values[0], axis_name="z"),
            _normalize_spatial_calibration_binding(values[1], axis_name="y"),
            _normalize_spatial_calibration_binding(values[2], axis_name="x"),
        )
    else:
        raise ValueError(
            "Spatial calibration stage_axis_map_zyx must be a mapping or "
            "three-entry sequence."
        )

    seen_sources: set[str] = set()
    for axis_name, binding in zip(
        SPATIAL_CALIBRATION_WORLD_AXES,
        normalized,
        strict=False,
    ):
        if binding == "none":
            continue
        source_axis = binding[1:]
        if source_axis in seen_sources:
            raise ValueError(
                "Spatial calibration cannot map one stage axis to multiple world "
                f"axes. Duplicate source axis '{source_axis}' detected at world "
                f"axis '{axis_name}'."
            )
        seen_sources.add(source_axis)
    return normalized


@dataclass(frozen=True)
class SpatialCalibrationConfig:
    """Store-level stage-to-world axis mapping for multiposition placement.

    Attributes
    ----------
    stage_axis_map_zyx : tuple[str, str, str]
        World-axis bindings in ``(z, y, x)`` order. Allowed values are
        ``+x``, ``-x``, ``+y``, ``-y``, ``+z``, ``-z``, ``+f``, ``-f``, and
        ``none``.
    theta_mode : str
        Rotation interpretation for Navigate ``THETA`` values.
    """

    stage_axis_map_zyx: tuple[str, str, str] = (
        SPATIAL_CALIBRATION_DEFAULT_STAGE_AXIS_MAP_ZYX
    )
    theta_mode: str = SPATIAL_CALIBRATION_DEFAULT_THETA_MODE

    def __post_init__(self) -> None:
        """Normalize and validate spatial-calibration fields.

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
            If bindings or theta mode are invalid.
        """
        normalized_bindings = _normalize_spatial_calibration_stage_axis_map(
            self.stage_axis_map_zyx
        )
        theta_mode = (
            str(self.theta_mode).strip().lower()
            or SPATIAL_CALIBRATION_DEFAULT_THETA_MODE
        )
        if theta_mode != SPATIAL_CALIBRATION_DEFAULT_THETA_MODE:
            raise ValueError(
                "Spatial calibration theta_mode must be "
                f"'{SPATIAL_CALIBRATION_DEFAULT_THETA_MODE}'."
            )
        object.__setattr__(self, "stage_axis_map_zyx", normalized_bindings)
        object.__setattr__(self, "theta_mode", theta_mode)

    def stage_axis_map_by_world_axis(self) -> Dict[str, str]:
        """Return the world-axis binding mapping.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, str]
            Mapping of world ``z/y/x`` axes to canonical binding strings.
        """
        return {
            axis_name: binding
            for axis_name, binding in zip(
                SPATIAL_CALIBRATION_WORLD_AXES,
                self.stage_axis_map_zyx,
                strict=False,
            )
        }


def parse_spatial_calibration(
    mapping: Optional[str],
) -> SpatialCalibrationConfig:
    """Parse CLI/GUI text into a spatial calibration configuration.

    Parameters
    ----------
    mapping : str, optional
        Canonical text form such as ``"z=+x,y=none,x=+y"``.

    Returns
    -------
    SpatialCalibrationConfig
        Parsed and validated calibration. Empty input resolves to identity.

    Raises
    ------
    ValueError
        If the text is malformed or reuses a non-``none`` stage axis.
    """
    if mapping is None:
        return SpatialCalibrationConfig()

    text = str(mapping).strip()
    if not text:
        return SpatialCalibrationConfig()

    assignments: Dict[str, str] = {}
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Spatial calibration must use 'world_axis=binding' assignments."
            )
        axis_name, binding = item.split("=", 1)
        key = str(axis_name).strip().lower()
        if key not in SPATIAL_CALIBRATION_WORLD_AXES:
            raise ValueError(
                "Spatial calibration world axes must be z, y, or x."
            )
        if key in assignments:
            raise ValueError(
                f"Spatial calibration world axis '{key}' is assigned more than once."
            )
        assignments[key] = str(binding).strip()

    if set(assignments) != set(SPATIAL_CALIBRATION_WORLD_AXES):
        raise ValueError(
            "Spatial calibration must define exactly z, y, and x assignments."
        )
    return SpatialCalibrationConfig(
        stage_axis_map_zyx=(
            assignments["z"],
            assignments["y"],
            assignments["x"],
        )
    )


def normalize_spatial_calibration(
    value: Any,
) -> SpatialCalibrationConfig:
    """Normalize flexible spatial-calibration payloads.

    Parameters
    ----------
    value : Any
        Candidate calibration payload. Accepts
        :class:`SpatialCalibrationConfig`, canonical text, or metadata mappings.

    Returns
    -------
    SpatialCalibrationConfig
        Normalized calibration configuration.

    Raises
    ------
    ValueError
        If the payload cannot be interpreted as a valid calibration.
    """
    if value is None:
        return SpatialCalibrationConfig()
    if isinstance(value, SpatialCalibrationConfig):
        return value
    if isinstance(value, str):
        return parse_spatial_calibration(value)
    if not isinstance(value, Mapping):
        raise ValueError(
            "Spatial calibration must be a SpatialCalibrationConfig, string, or mapping."
        )

    theta_mode = (
        str(
            value.get("theta_mode", SPATIAL_CALIBRATION_DEFAULT_THETA_MODE)
        ).strip()
        or SPATIAL_CALIBRATION_DEFAULT_THETA_MODE
    )
    schema = str(value.get("schema", SPATIAL_CALIBRATION_SCHEMA)).strip()
    if schema and schema != SPATIAL_CALIBRATION_SCHEMA:
        raise ValueError(
            f"Unsupported spatial calibration schema '{schema}'."
        )

    if "stage_axis_map_zyx" in value:
        stage_axis_payload = value.get("stage_axis_map_zyx")
    elif any(axis_name in value for axis_name in SPATIAL_CALIBRATION_WORLD_AXES):
        missing_axes = [
            axis_name
            for axis_name in SPATIAL_CALIBRATION_WORLD_AXES
            if axis_name not in value
        ]
        if missing_axes:
            raise ValueError(
                "Spatial calibration mappings must define z, y, and x bindings."
            )
        stage_axis_payload = {
            axis_name: value.get(axis_name)
            for axis_name in SPATIAL_CALIBRATION_WORLD_AXES
        }
    else:
        raise ValueError(
            "Spatial calibration mappings must provide stage_axis_map_zyx or z/y/x keys."
        )

    if isinstance(stage_axis_payload, str):
        parsed = parse_spatial_calibration(stage_axis_payload)
        return SpatialCalibrationConfig(
            stage_axis_map_zyx=parsed.stage_axis_map_zyx,
            theta_mode=theta_mode,
        )

    return SpatialCalibrationConfig(
        stage_axis_map_zyx=_normalize_spatial_calibration_stage_axis_map(
            stage_axis_payload
        ),
        theta_mode=theta_mode,
    )


def spatial_calibration_to_dict(
    config: SpatialCalibrationConfig,
) -> Dict[str, Any]:
    """Serialize spatial calibration for Zarr attrs and provenance.

    Parameters
    ----------
    config : SpatialCalibrationConfig
        Calibration to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-compatible payload with schema, bindings, and theta mode.
    """
    normalized = normalize_spatial_calibration(config)
    return {
        "schema": SPATIAL_CALIBRATION_SCHEMA,
        "stage_axis_map_zyx": normalized.stage_axis_map_by_world_axis(),
        "theta_mode": normalized.theta_mode,
    }


def spatial_calibration_from_dict(
    payload: Any,
) -> SpatialCalibrationConfig:
    """Deserialize spatial calibration from metadata payloads.

    Parameters
    ----------
    payload : Any
        Stored calibration payload. Missing values resolve to identity.

    Returns
    -------
    SpatialCalibrationConfig
        Parsed calibration configuration.
    """
    return normalize_spatial_calibration(payload)


def format_spatial_calibration(
    config: Any,
) -> str:
    """Format a spatial calibration in canonical text form.

    Parameters
    ----------
    config : Any
        Calibration payload accepted by :func:`normalize_spatial_calibration`.

    Returns
    -------
    str
        Canonical text form ``z=...,y=...,x=...``.
    """
    normalized = normalize_spatial_calibration(config)
    return ",".join(
        f"{axis_name}={binding}"
        for axis_name, binding in zip(
            SPATIAL_CALIBRATION_WORLD_AXES,
            normalized.stage_axis_map_zyx,
            strict=False,
        )
    )


@dataclass(frozen=True)
class AnalysisTarget:
    """Resolved experiment/store pair available to the analysis dialog.

    Attributes
    ----------
    experiment_path : str
        Navigate ``experiment.yml`` path shown to operators in batch/single
        analysis scope controls.
    store_path : str
        Canonical analysis-store path used when executing this target.
    """

    experiment_path: str
    store_path: str


def normalize_analysis_targets(
    targets: Optional[Sequence[Union[AnalysisTarget, Mapping[str, Any]]]],
) -> tuple[AnalysisTarget, ...]:
    """Normalize workflow analysis targets.

    Parameters
    ----------
    targets : sequence[AnalysisTarget or mapping], optional
        Candidate analysis targets. Mapping entries must provide
        ``experiment_path`` and ``store_path`` keys.

    Returns
    -------
    tuple[AnalysisTarget, ...]
        Normalized targets in first-seen order. Duplicate experiment paths are
        collapsed to one entry.

    Raises
    ------
    ValueError
        If a target is malformed or missing required paths.
    """
    if targets is None:
        return tuple()

    normalized: list[AnalysisTarget] = []
    seen_experiment_paths: set[str] = set()
    for target in targets:
        if isinstance(target, AnalysisTarget):
            experiment_path = str(target.experiment_path).strip()
            store_path = str(target.store_path).strip()
        elif isinstance(target, Mapping):
            experiment_path = str(target.get("experiment_path", "")).strip()
            store_path = str(target.get("store_path", "")).strip()
        else:
            raise ValueError(
                "Analysis targets must be AnalysisTarget instances or mappings."
            )
        if not experiment_path:
            raise ValueError("Each analysis target must define an experiment_path.")
        if not store_path:
            raise ValueError("Each analysis target must define a store_path.")
        if experiment_path in seen_experiment_paths:
            continue
        seen_experiment_paths.add(experiment_path)
        normalized.append(
            AnalysisTarget(
                experiment_path=experiment_path,
                store_path=store_path,
            )
        )
    return tuple(normalized)


@dataclass
class WorkflowConfig:
    """Runtime workflow options shared by GUI and headless entrypoints.

    Attributes
    ----------
    file : str, optional
        Input image path for processing.
    analysis_targets : tuple[AnalysisTarget, ...]
        Ordered experiment/store pairs available for single-experiment or
        batch analysis selection in the GUI.
    analysis_selected_experiment_path : str, optional
        Currently selected Navigate ``experiment.yml`` path within
        ``analysis_targets``.
    analysis_apply_to_all : bool
        Whether analysis should run across every entry in
        ``analysis_targets`` instead of only the selected one.
    prefer_dask : bool
        Whether to open data using lazy Dask-backed arrays when supported.
    dask_backend : DaskBackendConfig
        Backend orchestration mode and runtime settings for Dask execution.
    chunks : int or tuple of int, optional
        Chunking configuration used for Dask reads.
    flatfield : bool
        Flag indicating whether flatfield-correction workflow should run.
    deconvolution : bool
        Flag indicating whether deconvolution workflow should run.
    shear_transform : bool
        Flag indicating whether shear-transform workflow should run.
    particle_detection : bool
        Flag indicating whether particle detection workflow should run.
    usegment3d : bool
        Flag indicating whether usegment3d workflow should run.
    registration : bool
        Flag indicating whether registration workflow should run.
    visualization : bool
        Flag indicating whether visualization workflow should run.
    mip_export : bool
        Flag indicating whether MIP-export workflow should run.
    zarr_save : ZarrSaveConfig
        Analysis-store chunking and pyramid configuration for saved Zarr data.
    spatial_calibration : SpatialCalibrationConfig
        Store-level Navigate stage-to-world axis mapping used for multiposition
        placement metadata.
    spatial_calibration_explicit : bool
        Whether the current spatial calibration was explicitly supplied by the
        operator rather than inherited as the identity default.
    analysis_parameters : dict[str, dict[str, Any]]
        Per-analysis runtime parameters keyed by analysis name.
    """

    file: Optional[str] = None
    analysis_targets: tuple[AnalysisTarget, ...] = field(default_factory=tuple)
    analysis_selected_experiment_path: Optional[str] = None
    analysis_apply_to_all: bool = False
    prefer_dask: bool = True
    dask_backend: DaskBackendConfig = field(default_factory=DaskBackendConfig)
    chunks: ChunkSpec = None
    flatfield: bool = False
    deconvolution: bool = False
    shear_transform: bool = False
    particle_detection: bool = False
    usegment3d: bool = False
    registration: bool = False
    visualization: bool = False
    mip_export: bool = False
    zarr_save: ZarrSaveConfig = field(default_factory=ZarrSaveConfig)
    spatial_calibration: SpatialCalibrationConfig = field(
        default_factory=SpatialCalibrationConfig
    )
    spatial_calibration_explicit: bool = False
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
        self.analysis_targets = normalize_analysis_targets(self.analysis_targets)
        if not isinstance(self.spatial_calibration, SpatialCalibrationConfig):
            self.spatial_calibration = normalize_spatial_calibration(
                self.spatial_calibration
            )
        self.spatial_calibration_explicit = bool(self.spatial_calibration_explicit)
        selected_experiment_path = (
            str(self.analysis_selected_experiment_path).strip()
            if self.analysis_selected_experiment_path is not None
            else ""
        )
        if self.analysis_targets:
            selected_target = None
            if selected_experiment_path:
                selected_target = next(
                    (
                        target
                        for target in self.analysis_targets
                        if target.experiment_path == selected_experiment_path
                    ),
                    None,
                )
                if selected_target is None:
                    raise ValueError(
                        "analysis_selected_experiment_path must match one of the "
                        "configured analysis targets."
                    )
            else:
                current_file = str(self.file or "").strip()
                selected_target = next(
                    (
                        target
                        for target in self.analysis_targets
                        if target.store_path == current_file
                    ),
                    None,
                )
                if selected_target is None:
                    selected_target = self.analysis_targets[0]
            self.analysis_selected_experiment_path = (
                selected_target.experiment_path
            )
            self.file = selected_target.store_path
        else:
            self.analysis_selected_experiment_path = (
                selected_experiment_path if selected_experiment_path else None
            )
        self.analysis_apply_to_all = bool(self.analysis_apply_to_all)
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
                self.flatfield,
                self.deconvolution,
                self.shear_transform,
                self.particle_detection,
                self.usegment3d,
                self.registration,
                self.visualization,
                self.mip_export,
            )
        )

    def selected_analysis_target(self) -> Optional[AnalysisTarget]:
        """Return the currently selected analysis target, when configured.

        Returns
        -------
        AnalysisTarget, optional
            Selected target resolved from ``analysis_targets`` and
            ``analysis_selected_experiment_path``.
        """
        selected_experiment_path = (
            str(self.analysis_selected_experiment_path or "").strip()
        )
        if not selected_experiment_path:
            return None
        for target in self.analysis_targets:
            if target.experiment_path == selected_experiment_path:
                return target
        return None


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
