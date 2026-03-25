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

"""uSegment3D segmentation workflow on canonical 6D Zarr stores."""

from __future__ import annotations

# Standard Library Imports
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
)

# Third Party Imports
import dask
from dask import delayed
import numpy as np
from skimage.transform import resize
import zarr

# Local Imports
from clearex.io.ome_store import (
    analysis_auxiliary_root,
    analysis_cache_data_component,
    analysis_cache_root,
    public_analysis_root,
    resolve_voxel_size_um_zyx_with_source,
)
from clearex.io.provenance import register_latest_output_reference
from clearex.workflow import (
    default_analysis_operation_parameters,
    normalize_analysis_operation_parameters,
)

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]

_USEGMENT3D_RUNTIME_STDOUT_REMAP: Dict[str, str] = {
    "no CUDA. trying torch for resizing": (
        "uSegment3D: CuPy CUDA resizing path unavailable; trying Torch backend."
    ),
    "no CUDA. trying torch for normalizing": (
        "uSegment3D: CuPy CUDA normalization path unavailable; trying Torch backend."
    ),
    "no gpu. falling back to CPU for resizing": (
        "uSegment3D: GPU resizing backends unavailable; falling back to CPU."
    ),
    "no gpu. falling back to CPU for normalizing": (
        "uSegment3D: GPU normalization backends unavailable; falling back to CPU."
    ),
}


@dataclass(frozen=True)
class Usegment3dSummary:
    """Summary metadata for one uSegment3D run.

    Attributes
    ----------
    component : str
        Output latest component group path.
    data_component : str
        Output latest data-array component path.
    source_component : str
        Input source component used for segmentation.
    volumes_processed : int
        Number of submitted segmentation tasks (``t``/``p``/selected-channel).
    channel_indices : tuple[int, ...]
        Source channel indices selected for this run.
    channel_index : int
        Backward-compatible primary channel index (first selected channel).
    views : tuple[str, ...]
        Views used for 2D-to-3D aggregation (subset of ``xy``, ``xz``, ``yz``).
    output_shape_tpczyx : tuple[int, int, int, int, int, int]
        Saved output shape in canonical axis order.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Saved output chunk shape in canonical axis order.
    gpu_requested : bool
        Whether GPU execution was requested.
    gpu_enabled : bool
        Whether GPU execution was enabled at runtime.
    require_gpu : bool
        Whether GPU was required by configuration.
    """

    component: str
    data_component: str
    source_component: str
    volumes_processed: int
    channel_indices: tuple[int, ...]
    channel_index: int
    views: tuple[str, ...]
    output_shape_tpczyx: tuple[int, int, int, int, int, int]
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]
    gpu_requested: bool
    gpu_enabled: bool
    require_gpu: bool


def _remap_usegment3d_runtime_stdout_line(line: str) -> str:
    """Normalize uSegment3D runtime stdout lines for clearer GPU diagnostics.

    Parameters
    ----------
    line : str
        One stdout line emitted by ``segment3D`` runtime code.

    Returns
    -------
    str
        Clarified line text. Empty string indicates the input line was empty.
    """
    stripped = str(line).strip()
    if not stripped:
        return ""
    return str(_USEGMENT3D_RUNTIME_STDOUT_REMAP.get(stripped, stripped))


def _run_usegment3d_preprocess_with_stdout_bridge(
    *,
    runtime_module: Any,
    image_zyx: np.ndarray,
    preprocess_params: Mapping[str, Any],
) -> Any:
    """Run uSegment3D preprocessing and replay stdout with clarified messages.

    Parameters
    ----------
    runtime_module : Any
        Imported ``segment3D.usegment3d`` runtime module.
    image_zyx : numpy.ndarray
        Input source volume in ``(z, y, x)`` order.
    preprocess_params : mapping[str, Any]
        Parameter payload for ``runtime_module.preprocess_imgs``.

    Returns
    -------
    Any
        Preprocessed output returned by ``runtime_module.preprocess_imgs``.
    """
    stdout_buffer = StringIO()
    with redirect_stdout(stdout_buffer):
        preprocessed = runtime_module.preprocess_imgs(
            np.asarray(image_zyx),
            params=dict(preprocess_params),
        )
    captured_stdout = stdout_buffer.getvalue()
    if captured_stdout:
        for raw_line in captured_stdout.splitlines():
            rendered_line = _remap_usegment3d_runtime_stdout_line(raw_line)
            if rendered_line:
                print(rendered_line, flush=True)
    return preprocessed


def _to_json_parameter_value(value: Any, *, field_name: str) -> Any:
    """Convert parameter values to JSON-serializable scalar/list payloads.

    Parameters
    ----------
    value : Any
        Candidate parameter value.
    field_name : str
        Parameter key used for error context.

    Returns
    -------
    Any
        JSON-safe scalar, ``None``, or list of JSON-safe values.

    Raises
    ------
    ValueError
        If value type is unsupported for parameter serialization.
    """
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [
            _to_json_parameter_value(item, field_name=field_name)
            for item in list(value)
        ]
    raise ValueError(
        "uSegment3D parameter "
        f"'{field_name}' has unsupported type {type(value).__name__}."
    )


def _normalize_parameters_with_dropped_keys(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize and sanitize uSegment3D parameters for distributed transport.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate uSegment3D parameter mapping.

    Returns
    -------
    tuple[dict[str, Any], tuple[str, ...]]
        Sanitized parameter mapping and unsupported normalized keys that were
        dropped before task submission.

    Raises
    ------
    ValueError
        If normalization fails due to invalid parameter values.
    """
    normalized_map = normalize_analysis_operation_parameters(
        {"usegment3d": dict(parameters)}
    )
    normalized = dict(normalized_map.get("usegment3d", {}))
    default_params = dict(default_analysis_operation_parameters().get("usegment3d", {}))
    allowed_keys = set(default_params.keys())
    dropped_keys = tuple(
        sorted(str(key) for key in normalized.keys() if str(key) not in allowed_keys)
    )
    sanitized: dict[str, Any] = {}
    for key in default_params.keys():
        sanitized[str(key)] = _to_json_parameter_value(
            normalized.get(str(key), default_params[str(key)]),
            field_name=str(key),
        )
    return sanitized, dropped_keys


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize uSegment3D parameters using workflow canonical rules.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate uSegment3D parameter mapping.

    Returns
    -------
    dict[str, Any]
        Canonical normalized uSegment3D parameter mapping.

    Raises
    ------
    ValueError
        If normalization fails due to invalid parameter values.
    """
    normalized, _ = _normalize_parameters_with_dropped_keys(parameters)
    return normalized


def _load_usegment3d_runtime() -> tuple[Any, Any]:
    """Import optional uSegment3D runtime modules.

    Returns
    -------
    tuple[Any, Any]
        ``(segment3D.parameters, segment3D.usegment3d)`` modules.

    Raises
    ------
    RuntimeError
        If the optional runtime dependency cannot be imported.
    """
    try:
        import segment3D.parameters as usegment3d_parameters  # type: ignore[import-untyped]
        import segment3D.usegment3d as usegment3d_runtime  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "uSegment3D runtime is unavailable. Install optional dependencies with "
            '`uv pip install -e ".[usegment3d]"` or `pip install u-Segment3D`.'
        ) from exc
    return usegment3d_parameters, usegment3d_runtime


def _is_gpu_available() -> bool:
    """Return whether a CUDA-capable torch runtime is available.

    Returns
    -------
    bool
        ``True`` when torch is importable and ``torch.cuda.is_available()``.
    """
    try:
        import torch  # type: ignore[import-untyped]
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _summarize_client_worker_state(client: "Client") -> str:
    """Summarize current distributed worker state for failure diagnostics.

    Parameters
    ----------
    client : dask.distributed.Client
        Active distributed client.

    Returns
    -------
    str
        Concise worker/resource summary string.
    """
    try:
        scheduler_info = client.scheduler_info()
    except Exception as exc:
        return "scheduler_info_unavailable=" f"{type(exc).__name__}: {exc}"

    worker_infos = dict(scheduler_info.get("workers", {}))
    active_workers = len(worker_infos)
    gpu_workers = sum(
        1
        for worker in worker_infos.values()
        if float(dict(worker.get("resources", {})).get("GPU", 0.0)) >= 1.0
    )
    total_threads = sum(
        max(0, int(worker.get("nthreads", 0))) for worker in worker_infos.values()
    )
    return (
        f"active_workers={active_workers}, "
        f"gpu_workers={gpu_workers}, "
        f"total_threads={total_threads}"
    )


def _extract_base_voxel_size_um_zyx(
    root: zarr.hierarchy.Group,
    *,
    source_component: Optional[str] = None,
) -> tuple[float, float, float]:
    """Extract base-level physical voxel size from canonical store metadata.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Root Zarr group.
    source_component : str, optional
        Preferred source component used to follow source-component provenance
        back to the full-resolution voxel-size metadata.

    Returns
    -------
    tuple[float, float, float]
        Base voxel size in ``(z, y, x)`` microns.
    """
    resolved, _ = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component=source_component,
    )
    return tuple(float(value) for value in resolved)


def _pyramid_factor_zyx_for_level(
    root: zarr.hierarchy.Group,
    *,
    level: int,
    source_component: Optional[str] = None,
) -> tuple[float, float, float]:
    """Return pyramid downsampling factors for one level in ``(z, y, x)``.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Root Zarr group.
    level : int
        Requested pyramid level.

    Returns
    -------
    tuple[float, float, float]
        Downsampling factors in ``(z, y, x)`` order.
    """
    if level <= 0:
        return (1.0, 1.0, 1.0)

    candidate_factors: list[Any] = []
    if source_component:
        try:
            source_attrs = dict(root[str(source_component)].attrs)
        except Exception:
            source_attrs = {}
        candidate_factors.extend(
            [
                source_attrs.get("pyramid_factors_tpczyx"),
                source_attrs.get("resolution_pyramid_factors_tpczyx"),
            ]
        )
    candidate_factors.append(root.attrs.get("data_pyramid_factors_tpczyx"))

    for factors in candidate_factors:
        if isinstance(factors, (tuple, list)) and len(factors) > level:
            level_entry = factors[level]
            if isinstance(level_entry, (tuple, list)) and len(level_entry) >= 6:
                try:
                    parsed = (
                        max(1.0, float(level_entry[3])),
                        max(1.0, float(level_entry[4])),
                        max(1.0, float(level_entry[5])),
                    )
                    return parsed
                except Exception:
                    pass

    uniform = float(2 ** int(level))
    return (uniform, uniform, uniform)


def _component_level_suffix(component: str) -> Optional[int]:
    """Parse trailing ``level_<n>`` suffix from a component path.

    Parameters
    ----------
    component : str
        Component path.

    Returns
    -------
    int, optional
        Parsed level integer if present and valid.
    """
    token = str(component).strip().split("/")[-1]
    if not token.startswith("level_"):
        return None
    try:
        level = int(token.split("_", maxsplit=1)[1])
    except Exception:
        return None
    if level < 0:
        return None
    return level


def _resolve_source_component_for_level(
    *,
    root: zarr.hierarchy.Group,
    source_component: str,
    input_resolution_level: int,
) -> tuple[str, int]:
    """Resolve effective input component for requested pyramid level.

    Parameters
    ----------
    root : zarr.hierarchy.Group
        Root Zarr group.
    source_component : str
        Requested base component path.
    input_resolution_level : int
        Requested resolution level (0 = full resolution).

    Returns
    -------
    tuple[str, int]
        ``(effective_component, effective_level)``.

    Raises
    ------
    ValueError
        If requested level/component cannot be resolved.
    """
    requested_component = str(source_component).strip() or "data"
    requested_level = max(0, int(input_resolution_level))

    if requested_component not in root:
        raise ValueError(
            f"uSegment3D input component '{requested_component}' was not found."
        )
    if requested_level <= 0:
        return requested_component, 0

    direct_level = _component_level_suffix(requested_component)
    if direct_level is not None and direct_level == requested_level:
        return requested_component, requested_level

    candidate_components: list[str] = []
    if requested_component == "data":
        candidate_components.append(f"data_pyramid/level_{requested_level}")
    elif requested_component.startswith("data_pyramid/level_"):
        candidate_components.append(f"data_pyramid/level_{requested_level}")
    candidate_components.append(
        f"{requested_component}_pyramid/level_{requested_level}"
    )

    for candidate in candidate_components:
        if candidate in root:
            return candidate, requested_level

    raise ValueError(
        "uSegment3D input_resolution_level="
        f"{requested_level} was requested for '{requested_component}', "
        "but no matching pyramid component exists."
    )


def _prepare_output_array(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    output_reference_component: str,
    output_channel_count: int,
    source_channel_indices: Sequence[int],
    save_native_labels: bool,
    parameters: Mapping[str, Any],
) -> tuple[
    str,
    str,
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int, int, int],
    Optional[str],
    Optional[tuple[int, int, int, int, int, int]],
    Optional[tuple[int, int, int, int, int, int]],
]:
    """Create latest uSegment3D output dataset in canonical store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis store.
    source_component : str
        Effective source component path used for segmentation.
    output_reference_component : str
        Component path used to define final output shape/chunks.
    output_channel_count : int
        Number of output label channels (one per selected source channel).
    source_channel_indices : sequence[int]
        Source channel indices mapped to output channel slots.
    save_native_labels : bool
        Whether to also persist labels at the native input resolution.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    tuple[str, str, tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int], str, tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int]]
        ``(component, data_component, shape_tpczyx, chunks_tpczyx,``
        ``native_data_component, native_shape_tpczyx, native_chunks_tpczyx)``.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    source = root[source_component]
    reference = root[output_reference_component]
    source_shape = tuple(int(v) for v in source.shape)
    source_chunks = tuple(int(v) for v in source.chunks)
    reference_shape = tuple(int(v) for v in reference.shape)
    reference_chunks = tuple(int(v) for v in reference.chunks)

    channel_count = max(1, int(output_channel_count))
    output_shape = (
        int(reference_shape[0]),
        int(reference_shape[1]),
        channel_count,
        int(reference_shape[3]),
        int(reference_shape[4]),
        int(reference_shape[5]),
    )
    output_chunks = (
        1,
        1,
        1,
        max(1, min(int(reference_chunks[3]), int(output_shape[3]))),
        max(1, min(int(reference_chunks[4]), int(output_shape[4]))),
        max(1, min(int(reference_chunks[5]), int(output_shape[5]))),
    )
    output_dtype = np.dtype(str(parameters.get("output_dtype", "uint32")))

    cache_root = analysis_cache_root("usegment3d")
    auxiliary_root = analysis_auxiliary_root("usegment3d")
    if cache_root in root:
        del root[cache_root]
    if auxiliary_root in root:
        del root[auxiliary_root]
    latest = root.require_group(cache_root)
    latest.create_dataset(
        name="data",
        shape=output_shape,
        chunks=output_chunks,
        dtype=output_dtype,
        overwrite=True,
    )
    native_data_component: Optional[str] = None
    native_shape: Optional[tuple[int, int, int, int, int, int]] = None
    native_chunks: Optional[tuple[int, int, int, int, int, int]] = None
    should_save_native = bool(
        save_native_labels and str(source_component) != str(output_reference_component)
    )
    if should_save_native:
        native_shape = (
            int(source_shape[0]),
            int(source_shape[1]),
            channel_count,
            int(source_shape[3]),
            int(source_shape[4]),
            int(source_shape[5]),
        )
        native_chunks = (
            1,
            1,
            1,
            max(1, min(int(source_chunks[3]), int(native_shape[3]))),
            max(1, min(int(source_chunks[4]), int(native_shape[4]))),
            max(1, min(int(source_chunks[5]), int(native_shape[5]))),
        )
        latest.create_dataset(
            name="data_native",
            shape=native_shape,
            chunks=native_chunks,
            dtype=output_dtype,
            overwrite=True,
        )
        native_data_component = f"{cache_root}/data_native"
    latest.attrs.update(
        {
            "storage_policy": "latest_only",
            "source_component": str(source_component),
            "output_reference_component": str(output_reference_component),
            "native_data_component": native_data_component,
            "source_channel_indices": [int(value) for value in source_channel_indices],
            "parameters": {str(key): value for key, value in dict(parameters).items()},
            "run_id": None,
        }
    )
    root.require_group(auxiliary_root).attrs.update(dict(latest.attrs))

    return (
        public_analysis_root("usegment3d"),
        analysis_cache_data_component("usegment3d"),
        (
            int(output_shape[0]),
            int(output_shape[1]),
            int(output_shape[2]),
            int(output_shape[3]),
            int(output_shape[4]),
            int(output_shape[5]),
        ),
        (
            int(output_chunks[0]),
            int(output_chunks[1]),
            int(output_chunks[2]),
            int(output_chunks[3]),
            int(output_chunks[4]),
            int(output_chunks[5]),
        ),
        native_data_component,
        native_shape,
        native_chunks,
    )


def _prepare_preprocess_params(
    parameters_module: Any, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Build preprocess parameter payload for uSegment3D.

    Parameters
    ----------
    parameters_module : Any
        Imported ``segment3D.parameters`` module.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    dict[str, Any]
        Parameters compatible with ``segment3D.usegment3d.preprocess_imgs``.
    """
    preprocess_params = dict(parameters_module.get_preprocess_params())
    preprocess_params["factor"] = float(parameters.get("preprocess_factor", 1.0))
    preprocess_params["voxel_res"] = [
        float(value)
        for value in list(parameters.get("preprocess_voxel_res_zyx", [1.0, 1.0, 1.0]))
    ]
    preprocess_params["do_bg_correction"] = bool(
        parameters.get("preprocess_do_bg_correction", True)
    )
    preprocess_params["bg_ds"] = int(parameters.get("preprocess_bg_ds", 16))
    preprocess_params["bg_sigma"] = float(parameters.get("preprocess_bg_sigma", 5.0))
    preprocess_params["normalize_min"] = float(
        parameters.get("preprocess_normalize_min", 2.0)
    )
    preprocess_params["normalize_max"] = float(
        parameters.get("preprocess_normalize_max", 99.8)
    )
    preprocess_params["do_avg_imgs"] = False
    return preprocess_params


def _prepare_cellpose_params(
    parameters_module: Any, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Build Cellpose runtime parameter payload.

    Parameters
    ----------
    parameters_module : Any
        Imported ``segment3D.parameters`` module.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    dict[str, Any]
        Parameters compatible with ``segment3D.usegment3d.Cellpose2D_model_auto``.
    """
    cellpose_params = dict(parameters_module.get_Cellpose_autotune_params())
    cellpose_params["cellpose_modelname"] = str(
        parameters.get("cellpose_model_name", "cyto")
    )
    cellpose_params["cellpose_channels"] = str(
        parameters.get("cellpose_channels", "grayscale")
    )
    cellpose_params["hist_norm"] = bool(parameters.get("cellpose_hist_norm", False))
    cellpose_params["ksize"] = int(parameters.get("cellpose_ksize", 15))
    cellpose_params["use_Cellpose_auto_diameter"] = bool(
        parameters.get("cellpose_use_auto_diameter", False)
    )
    cellpose_params["gpu"] = bool(parameters.get("gpu", False))
    cellpose_params["best_diam"] = parameters.get("cellpose_best_diameter", None)
    cellpose_params["model_invert"] = bool(
        parameters.get("cellpose_model_invert", False)
    )
    cellpose_params["test_slice"] = parameters.get("cellpose_test_slice", None)
    diameter_range = list(parameters.get("cellpose_diameter_range", [10.0, 120.0, 2.5]))
    diameter_min, diameter_max, diameter_step = (
        float(diameter_range[0]),
        float(diameter_range[1]),
        float(diameter_range[2]),
    )
    diameter_stop = diameter_max + (diameter_step * 0.5)
    cellpose_params["diam_range"] = np.arange(
        diameter_min,
        diameter_stop,
        diameter_step,
        dtype=np.float32,
    )
    cellpose_params["use_edge"] = bool(parameters.get("cellpose_use_edge", True))
    cellpose_params["debug_viz"] = False
    cellpose_params["saveplotsfolder"] = None
    return cellpose_params


def _prepare_aggregation_params(
    parameters_module: Any, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Build aggregation parameter payload for direct-method uSegment3D.

    Parameters
    ----------
    parameters_module : Any
        Imported ``segment3D.parameters`` module.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    dict[str, Any]
        Parameters compatible with
        ``segment3D.usegment3d.aggregate_2D_to_3D_segmentation_direct_method``.
    """
    aggregation_params = dict(parameters_module.get_2D_to_3D_aggregation_params())

    combine_cell_probs = dict(aggregation_params.get("combine_cell_probs", {}))
    combine_cell_probs["prob_thresh"] = parameters.get(
        "aggregation_prob_threshold", None
    )
    combine_cell_probs["threshold_n_levels"] = int(
        parameters.get("aggregation_threshold_n_levels", 3)
    )
    combine_cell_probs["threshold_level"] = int(
        parameters.get("aggregation_threshold_level", 1)
    )
    combine_cell_probs["min_prob_thresh"] = float(
        parameters.get("aggregation_min_prob_threshold", 0.0)
    )
    aggregation_params["combine_cell_probs"] = combine_cell_probs

    connected_component = dict(aggregation_params.get("connected_component", {}))
    connected_component["min_area"] = int(
        parameters.get("aggregation_connected_min_area", 5)
    )
    connected_component["smooth_sigma"] = float(
        parameters.get("aggregation_connected_smooth_sigma", 1.0)
    )
    connected_component["thresh_factor"] = float(
        parameters.get("aggregation_connected_thresh_factor", 0.0)
    )
    aggregation_params["connected_component"] = connected_component

    postprocess_binary = dict(aggregation_params.get("postprocess_binary", {}))
    postprocess_binary["binary_fill_holes"] = bool(
        parameters.get("aggregation_binary_fill_holes", False)
    )
    aggregation_params["postprocess_binary"] = postprocess_binary

    gradient_descent = dict(aggregation_params.get("gradient_descent", {}))
    gradient_descent["gradient_decay"] = float(
        parameters.get("aggregation_gradient_decay", 0.0)
    )
    gradient_descent["n_iter"] = int(parameters.get("aggregation_n_iter", 200))
    gradient_descent["momenta"] = float(parameters.get("aggregation_momenta", 0.98))
    gradient_descent["do_mp"] = bool(parameters.get("aggregation_tile_mode", False))
    gradient_descent["tile_shape"] = tuple(
        int(value)
        for value in list(parameters.get("aggregation_tile_shape_zyx", [128, 256, 256]))
    )
    gradient_descent["tile_overlap_ratio"] = float(
        parameters.get("aggregation_tile_overlap_ratio", 0.25)
    )
    aggregation_params["gradient_descent"] = gradient_descent

    indirect_method = dict(aggregation_params.get("indirect_method", {}))
    indirect_method["n_cpu"] = parameters.get("n_cpu", None)
    indirect_method["dtform_method"] = str(
        parameters.get("postprocess_dtform_method", "cellpose_improve")
    )
    indirect_method["edt_fixed_point_percentile"] = float(
        parameters.get("postprocess_edt_fixed_point_percentile", 0.01)
    )
    aggregation_params["indirect_method"] = indirect_method

    return aggregation_params


def _prepare_postprocess_params(
    parameters_module: Any, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Build postprocess parameter payload for uSegment3D.

    Parameters
    ----------
    parameters_module : Any
        Imported ``segment3D.parameters`` module.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    dict[str, Any]
        Parameters compatible with
        ``segment3D.usegment3d.postprocess_3D_cell_segmentation``.
    """
    postprocess_params = dict(parameters_module.get_postprocess_segmentation_params())

    size_filters = dict(postprocess_params.get("size_filters", {}))
    size_filters["min_size"] = int(parameters.get("postprocess_min_size", 200))
    postprocess_params["size_filters"] = size_filters

    flow_consistency = dict(postprocess_params.get("flow_consistency", {}))
    flow_consistency["do_flow_remove"] = bool(
        parameters.get("postprocess_do_flow_remove", True)
    )
    flow_consistency["flow_threshold"] = float(
        parameters.get("postprocess_flow_threshold", 0.85)
    )
    flow_consistency["dtform_method"] = str(
        parameters.get("postprocess_dtform_method", "cellpose_improve")
    )
    flow_consistency["edt_fixed_point_percentile"] = float(
        parameters.get("postprocess_edt_fixed_point_percentile", 0.01)
    )
    flow_consistency["n_cpu"] = parameters.get("n_cpu", None)
    postprocess_params["flow_consistency"] = flow_consistency

    return postprocess_params


def _ordered_view_payload(
    *,
    selected_views: tuple[str, ...],
    probabilities_by_view: Mapping[str, np.ndarray],
    gradients_by_view: Mapping[str, np.ndarray],
) -> tuple[list[Any], list[Any]]:
    """Build ordered ``[xy, xz, yz]`` payload lists required by uSegment3D.

    Parameters
    ----------
    selected_views : tuple[str, ...]
        Views selected in runtime parameters.
    probabilities_by_view : mapping[str, numpy.ndarray]
        Probability maps keyed by view.
    gradients_by_view : mapping[str, numpy.ndarray]
        Flow/gradient maps keyed by view.

    Returns
    -------
    tuple[list[Any], list[Any]]
        ``(ordered_probabilities, ordered_gradients)`` where missing views are
        represented by empty lists.
    """
    del selected_views
    ordered_probs: list[Any] = []
    ordered_gradients: list[Any] = []
    for view in ("xy", "xz", "yz"):
        ordered_probs.append(probabilities_by_view.get(view, []))
        ordered_gradients.append(gradients_by_view.get(view, []))
    return ordered_probs, ordered_gradients


def _resize_labels_to_shape(
    labels_zyx: np.ndarray,
    *,
    target_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Resize labels to target shape using nearest-neighbor interpolation.

    Parameters
    ----------
    labels_zyx : numpy.ndarray
        Source label volume.
    target_shape_zyx : tuple[int, int, int]
        Desired target shape.

    Returns
    -------
    numpy.ndarray
        Resized label volume.
    """
    resized = resize(
        labels_zyx.astype(np.float32, copy=False),
        output_shape=target_shape_zyx,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
        mode="edge",
    )
    return np.rint(resized).astype(np.int64, copy=False)


def _segment_volume(
    *,
    image_zyx: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Run uSegment3D segmentation for one 3D source volume.

    Parameters
    ----------
    image_zyx : numpy.ndarray
        Source intensity volume in ``(z, y, x)`` order.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    numpy.ndarray
        Segmentation label volume in ``(z, y, x)`` order.

    Raises
    ------
    ValueError
        If intermediate volume dimensionality is incompatible.
    """
    parameters_module, runtime_module = _load_usegment3d_runtime()

    preprocess_params = _prepare_preprocess_params(parameters_module, parameters)
    cellpose_params = _prepare_cellpose_params(parameters_module, parameters)
    aggregation_params = _prepare_aggregation_params(parameters_module, parameters)
    postprocess_params = _prepare_postprocess_params(parameters_module, parameters)

    preprocessed = _run_usegment3d_preprocess_with_stdout_bridge(
        runtime_module=runtime_module,
        image_zyx=np.asarray(image_zyx),
        preprocess_params=preprocess_params,
    )
    preprocessed_array = np.asarray(preprocessed)

    if preprocessed_array.ndim == 4 and preprocessed_array.shape[0] == 1:
        preprocessed_array = np.squeeze(preprocessed_array, axis=0)
    elif preprocessed_array.ndim == 4:
        preprocessed_array = np.moveaxis(preprocessed_array, 0, -1)

    if preprocessed_array.ndim == 3:
        cellpose_input = preprocessed_array[..., np.newaxis]
    elif preprocessed_array.ndim == 4:
        cellpose_input = preprocessed_array
    else:
        raise ValueError(
            "uSegment3D preprocessing returned an incompatible volume shape "
            f"{tuple(preprocessed_array.shape)}."
        )

    views = tuple(
        str(value) for value in list(parameters.get("use_views", ["xy", "xz", "yz"]))
    )
    probabilities_by_view: dict[str, np.ndarray] = {}
    gradients_by_view: dict[str, np.ndarray] = {}
    for view in views:
        try:
            _, probability_map, flow_map, _ = runtime_module.Cellpose2D_model_auto(
                cellpose_input,
                view=view,
                params=cellpose_params,
                basename=None,
                savefolder=None,
            )
        except ValueError as exc:
            # Some cellpose/uSegment3D combinations return 3 values from model eval
            # instead of the 4-value shape expected by uSegment3D auto-diameter logic.
            # Retry once with explicit diameter to preserve compatibility.
            if "not enough values to unpack" not in str(exc).lower() or not bool(
                cellpose_params.get("use_Cellpose_auto_diameter", False)
            ):
                raise
            fallback_params = dict(cellpose_params)
            fallback_params["use_Cellpose_auto_diameter"] = False
            best_diam = fallback_params.get("best_diam", None)
            if best_diam in (None, "", 0, 0.0):
                diam_range = np.asarray(
                    fallback_params.get("diam_range", np.asarray([30.0])),
                    dtype=np.float32,
                ).ravel()
                fallback_best_diam = (
                    float(np.nanmedian(diam_range)) if diam_range.size > 0 else 30.0
                )
                if not np.isfinite(fallback_best_diam) or fallback_best_diam <= 0:
                    fallback_best_diam = 30.0
                fallback_params["best_diam"] = float(fallback_best_diam)
            _, probability_map, flow_map, _ = runtime_module.Cellpose2D_model_auto(
                cellpose_input,
                view=view,
                params=fallback_params,
                basename=None,
                savefolder=None,
            )
            cellpose_params = fallback_params
        probabilities_by_view[str(view)] = np.asarray(probability_map, dtype=np.float32)
        gradients_by_view[str(view)] = np.asarray(flow_map, dtype=np.float32)

    ordered_probs, ordered_gradients = _ordered_view_payload(
        selected_views=views,
        probabilities_by_view=probabilities_by_view,
        gradients_by_view=gradients_by_view,
    )
    segmentation, (_, gradients3d) = (
        runtime_module.aggregate_2D_to_3D_segmentation_direct_method(
            probs=ordered_probs,
            gradients=ordered_gradients,
            params=aggregation_params,
            savefolder=None,
            basename=None,
        )
    )
    labels = np.asarray(segmentation)

    if bool(parameters.get("postprocess_enable", True)):
        labels, _ = runtime_module.postprocess_3D_cell_segmentation(
            labels,
            aggregation_params=aggregation_params,
            postprocess_params=postprocess_params,
            cell_gradients=np.asarray(gradients3d),
            savefolder=None,
            basename=None,
        )
        labels = np.asarray(labels)

    if labels.ndim != 3:
        raise ValueError(
            "uSegment3D segmentation returned an incompatible volume shape "
            f"{tuple(labels.shape)}."
        )

    target_shape = (
        int(image_zyx.shape[0]),
        int(image_zyx.shape[1]),
        int(image_zyx.shape[2]),
    )
    if tuple(int(v) for v in labels.shape) != target_shape:
        labels = _resize_labels_to_shape(labels, target_shape_zyx=target_shape)

    labels = np.maximum(labels, 0)
    output_dtype = np.dtype(str(parameters.get("output_dtype", "uint32")))
    return labels.astype(output_dtype, copy=False)


def _run_usegment3d_for_volume(
    *,
    zarr_path: str,
    source_component: str,
    output_data_component: str,
    output_native_data_component: Optional[str],
    output_target_shape_zyx: tuple[int, int, int],
    t_index: int,
    p_index: int,
    source_channel_index: int,
    output_channel_index: int,
    parameters: Mapping[str, Any],
) -> int:
    """Run uSegment3D segmentation for one ``(t, p)`` volume and write output.

    Parameters
    ----------
    zarr_path : str
        Canonical store path.
    source_component : str
        Input source component path.
    output_data_component : str
        Output label-array component path.
    output_native_data_component : str, optional
        Optional native-resolution label-array component path.
    output_target_shape_zyx : tuple[int, int, int]
        Final output target shape in ``(z, y, x)`` order.
    t_index : int
        Time index.
    p_index : int
        Position index.
    source_channel_index : int
        Source channel index used as segmentation input.
    output_channel_index : int
        Channel index in the output label array.
    parameters : mapping[str, Any]
        Normalized runtime parameters.

    Returns
    -------
    int
        Always returns ``1`` for task-level completion accounting.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    source = root[source_component]
    output = root[output_data_component]
    output_native = (
        root[output_native_data_component]
        if output_native_data_component is not None
        else None
    )

    volume_zyx = np.asarray(
        source[int(t_index), int(p_index), int(source_channel_index), :, :, :],
        dtype=np.float32,
    )
    labels_native_zyx = _segment_volume(image_zyx=volume_zyx, parameters=parameters)
    if output_native is not None:
        output_native[
            int(t_index), int(p_index), int(output_channel_index), :, :, :
        ] = np.asarray(
            labels_native_zyx,
            dtype=output_native.dtype,
        )

    target_shape = (
        int(output_target_shape_zyx[0]),
        int(output_target_shape_zyx[1]),
        int(output_target_shape_zyx[2]),
    )
    if tuple(int(v) for v in labels_native_zyx.shape) != target_shape:
        labels_zyx = _resize_labels_to_shape(
            np.asarray(labels_native_zyx),
            target_shape_zyx=target_shape,
        )
    else:
        labels_zyx = np.asarray(labels_native_zyx)
    output[int(t_index), int(p_index), int(output_channel_index), :, :, :] = np.asarray(
        labels_zyx,
        dtype=output.dtype,
    )
    return 1


def run_usegment3d_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Usegment3dSummary:
    """Run uSegment3D segmentation for canonical 6D data and persist latest output.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Candidate uSegment3D parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    Usegment3dSummary
        Summary metadata for the completed uSegment3D run.

    Raises
    ------
    ValueError
        If source component is missing or incompatible.
    RuntimeError
        If runtime dependency import fails, or if ``require_gpu`` is true and no
        GPU is available.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized, dropped_parameter_keys = _normalize_parameters_with_dropped_keys(
        parameters
    )
    if dropped_parameter_keys:
        preview = ", ".join(dropped_parameter_keys[:4])
        remaining = int(len(dropped_parameter_keys) - 4)
        if remaining > 0:
            preview = f"{preview}, +{remaining} more"
        _emit(3, f"Ignoring unsupported uSegment3D keys: {preview}")

    gpu_requested = bool(normalized.get("gpu", False))
    require_gpu = bool(normalized.get("require_gpu", False))
    gpu_available = _is_gpu_available()
    gpu_enabled = bool(gpu_requested and gpu_available)
    if require_gpu and not gpu_available:
        raise RuntimeError(
            "uSegment3D require_gpu=True but no CUDA-capable GPU was detected."
        )
    if gpu_requested and not gpu_enabled:
        _emit(2, "GPU requested but unavailable; running uSegment3D on CPU")
    normalized["gpu"] = gpu_enabled

    aggregation_tile_mode_requested = bool(
        normalized.get("aggregation_tile_mode", False)
    )
    if client is not None and aggregation_tile_mode_requested:
        normalized["aggregation_tile_mode"] = False
        _emit(
            12,
            "Disabled uSegment3D tile multiprocessing under distributed execution",
        )
    aggregation_tile_mode_effective = bool(
        normalized.get("aggregation_tile_mode", False)
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    requested_source_component = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    if requested_source_component not in root:
        raise ValueError(
            "uSegment3D input component "
            f"'{requested_source_component}' was not found in {zarr_path}."
        )

    requested_resolution_level = max(
        0,
        int(normalized.get("input_resolution_level", 0)),
    )
    source_component, effective_resolution_level = _resolve_source_component_for_level(
        root=root,
        source_component=requested_source_component,
        input_resolution_level=requested_resolution_level,
    )
    source = root[source_component]
    source_shape = tuple(int(v) for v in source.shape)
    if len(source_shape) != 6:
        raise ValueError(
            "uSegment3D requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    output_reference_space = (
        str(normalized.get("output_reference_space", "level0")).strip().lower()
        or "level0"
    )
    if output_reference_space not in {"level0", "native_level"}:
        raise ValueError(
            "uSegment3D output_reference_space must be 'level0' or 'native_level'."
        )
    if output_reference_space == "level0" and effective_resolution_level > 0:
        output_reference_component = requested_source_component
        output_resolution_level = 0
    else:
        output_reference_component = source_component
        output_resolution_level = effective_resolution_level

    if output_reference_component not in root:
        raise ValueError(
            "uSegment3D output_reference_component "
            f"'{output_reference_component}' was not found in {zarr_path}."
        )
    output_reference = root[output_reference_component]
    output_reference_shape = tuple(int(v) for v in output_reference.shape)
    if len(output_reference_shape) != 6:
        raise ValueError(
            "uSegment3D output reference must be canonical 6D data (t,p,c,z,y,x). "
            f"Component '{output_reference_component}' is incompatible."
        )
    if int(source_shape[0]) != int(output_reference_shape[0]) or int(
        source_shape[1]
    ) != int(output_reference_shape[1]):
        raise ValueError(
            "uSegment3D source/output-reference (t,p) dimensions must match. "
            f"source={source_shape[:2]}, reference={output_reference_shape[:2]}."
        )

    requested_channel_indices = list(
        normalized.get("channel_indices", [normalized.get("channel_index", 0)])
    )
    if not requested_channel_indices:
        requested_channel_indices = [int(normalized.get("channel_index", 0))]
    all_channels = bool(normalized.get("all_channels", False))
    channel_indices: list[int] = []
    if all_channels:
        channel_indices = list(range(int(source_shape[2])))
    else:
        seen_channel_indices: set[int] = set()
        for value in requested_channel_indices:
            parsed = max(0, int(value))
            if parsed in seen_channel_indices:
                continue
            if parsed >= int(source_shape[2]):
                raise ValueError(
                    f"uSegment3D channel_index={parsed} is out of bounds "
                    f"for channel axis size {source_shape[2]}."
                )
            channel_indices.append(parsed)
            seen_channel_indices.add(parsed)
    if not channel_indices:
        channel_indices = [0]
        all_channels = False
    normalized["channel_indices"] = list(channel_indices)
    normalized["channel_index"] = int(channel_indices[0])
    normalized["all_channels"] = bool(all_channels)
    channel_index = int(channel_indices[0])

    base_voxel_size_um_zyx = _extract_base_voxel_size_um_zyx(
        root,
        source_component=source_component,
    )
    source_factor_zyx = _pyramid_factor_zyx_for_level(
        root,
        level=effective_resolution_level,
        source_component=source_component,
    )
    output_factor_zyx = _pyramid_factor_zyx_for_level(
        root,
        level=output_resolution_level,
        source_component=output_reference_component,
    )
    source_voxel_size_um_zyx = (
        float(base_voxel_size_um_zyx[0] * source_factor_zyx[0]),
        float(base_voxel_size_um_zyx[1] * source_factor_zyx[1]),
        float(base_voxel_size_um_zyx[2] * source_factor_zyx[2]),
    )
    output_voxel_size_um_zyx = (
        float(base_voxel_size_um_zyx[0] * output_factor_zyx[0]),
        float(base_voxel_size_um_zyx[1] * output_factor_zyx[1]),
        float(base_voxel_size_um_zyx[2] * output_factor_zyx[2]),
    )

    preprocess_voxel_res = list(
        normalized.get("preprocess_voxel_res_zyx", [1.0, 1.0, 1.0])
    )
    if len(preprocess_voxel_res) == 3 and all(
        abs(float(value) - 1.0) <= 1e-8 for value in preprocess_voxel_res
    ):
        normalized["preprocess_voxel_res_zyx"] = [
            float(source_voxel_size_um_zyx[0]),
            float(source_voxel_size_um_zyx[1]),
            float(source_voxel_size_um_zyx[2]),
        ]

    save_native_labels_requested = bool(normalized.get("save_native_labels", False))
    save_native_labels_effective = bool(
        save_native_labels_requested and output_reference_component != source_component
    )

    _emit(
        5,
        "Prepared uSegment3D inputs "
        "(source="
        f"{source_component}, level={effective_resolution_level}, "
        f"channels={channel_indices})",
    )
    (
        component,
        data_component,
        output_shape,
        output_chunks,
        native_data_component,
        native_shape,
        native_chunks,
    ) = _prepare_output_array(
        zarr_path=zarr_path,
        source_component=source_component,
        output_reference_component=output_reference_component,
        output_channel_count=len(channel_indices),
        source_channel_indices=channel_indices,
        save_native_labels=save_native_labels_effective,
        parameters=normalized,
    )
    try:
        output_array = root[data_component]
        output_array.attrs.update(
            {
                "voxel_size_um_zyx": [
                    float(value) for value in output_voxel_size_um_zyx
                ],
                "scale_tczyx": [
                    1.0,
                    1.0,
                    float(output_voxel_size_um_zyx[0]),
                    float(output_voxel_size_um_zyx[1]),
                    float(output_voxel_size_um_zyx[2]),
                ],
                "source_component": str(source_component),
                "output_reference_component": str(output_reference_component),
                "input_resolution_level": int(effective_resolution_level),
                "output_resolution_level": int(output_resolution_level),
                "output_reference_space": str(output_reference_space),
                "downsample_factors_tpczyx": [
                    1,
                    1,
                    1,
                    int(output_factor_zyx[0]),
                    int(output_factor_zyx[1]),
                    int(output_factor_zyx[2]),
                ],
            }
        )
    except Exception:
        pass
    if native_data_component:
        try:
            native_array = root[native_data_component]
            native_array.attrs.update(
                {
                    "voxel_size_um_zyx": [
                        float(value) for value in source_voxel_size_um_zyx
                    ],
                    "scale_tczyx": [
                        1.0,
                        1.0,
                        float(source_voxel_size_um_zyx[0]),
                        float(source_voxel_size_um_zyx[1]),
                        float(source_voxel_size_um_zyx[2]),
                    ],
                    "source_component": str(source_component),
                    "input_resolution_level": int(effective_resolution_level),
                    "downsample_factors_tpczyx": [
                        1,
                        1,
                        1,
                        int(source_factor_zyx[0]),
                        int(source_factor_zyx[1]),
                        int(source_factor_zyx[2]),
                    ],
                }
            )
        except Exception:
            pass
    _emit(10, "Initialized latest uSegment3D output dataset")

    work_items = [
        (
            int(t_index),
            int(p_index),
            int(source_channel_index),
            int(output_channel_index),
        )
        for output_channel_index, source_channel_index in enumerate(channel_indices)
        for t_index in range(int(output_shape[0]))
        for p_index in range(int(output_shape[1]))
    ]
    total = int(len(work_items))
    if total == 0:
        register_latest_output_reference(
            zarr_path=zarr_path,
            analysis_name="usegment3d",
            component=component,
            metadata={
                "data_component": data_component,
                "native_data_component": native_data_component,
                "source_component": source_component,
                "requested_source_component": requested_source_component,
                "output_reference_component": output_reference_component,
                "volumes_processed": 0,
                "channels_processed": 0,
                "channel_indices": list(channel_indices),
                "channel_index": channel_index,
                "input_resolution_level": effective_resolution_level,
                "requested_input_resolution_level": requested_resolution_level,
                "output_resolution_level": output_resolution_level,
                "output_reference_space": output_reference_space,
                "save_native_labels_requested": save_native_labels_requested,
                "save_native_labels_effective": save_native_labels_effective,
                "views": list(normalized.get("use_views", ["xy", "xz", "yz"])),
                "output_shape_tpczyx": list(output_shape),
                "output_chunks_tpczyx": list(output_chunks),
                "native_shape_tpczyx": list(native_shape) if native_shape else None,
                "native_chunks_tpczyx": list(native_chunks) if native_chunks else None,
                "base_voxel_size_um_zyx": list(base_voxel_size_um_zyx),
                "source_voxel_size_um_zyx": list(source_voxel_size_um_zyx),
                "output_voxel_size_um_zyx": list(output_voxel_size_um_zyx),
                "gpu_requested": gpu_requested,
                "gpu_enabled": gpu_enabled,
                "require_gpu": require_gpu,
                "aggregation_tile_mode_requested": aggregation_tile_mode_requested,
                "aggregation_tile_mode_effective": aggregation_tile_mode_effective,
                "dropped_parameter_keys": list(dropped_parameter_keys),
                "parameters": {str(key): value for key, value in normalized.items()},
            },
        )
        _emit(100, "No uSegment3D tasks to run.")
        return Usegment3dSummary(
            component=component,
            data_component=data_component,
            source_component=source_component,
            volumes_processed=0,
            channel_indices=tuple(int(value) for value in channel_indices),
            channel_index=channel_index,
            views=tuple(str(value) for value in list(normalized.get("use_views", []))),
            output_shape_tpczyx=output_shape,
            output_chunks_tpczyx=output_chunks,
            gpu_requested=gpu_requested,
            gpu_enabled=gpu_enabled,
            require_gpu=require_gpu,
        )

    if client is None:
        tasks = [
            delayed(_run_usegment3d_for_volume)(
                zarr_path=str(zarr_path),
                source_component=source_component,
                output_data_component=data_component,
                output_native_data_component=native_data_component,
                output_target_shape_zyx=(
                    int(output_shape[3]),
                    int(output_shape[4]),
                    int(output_shape[5]),
                ),
                t_index=t_index,
                p_index=p_index,
                source_channel_index=source_channel_index,
                output_channel_index=output_channel_index,
                parameters=normalized,
            )
            for t_index, p_index, source_channel_index, output_channel_index in work_items
        ]
        _emit(15, "Running uSegment3D tasks with local process scheduler")
        _ = dask.compute(*tasks, scheduler="processes")
        _emit(95, f"Completed {total} uSegment3D tasks")
    else:
        from dask.distributed import as_completed
        from distributed.client import FutureCancelledError

        _emit(15, f"Submitting {total} uSegment3D tasks to Dask client")
        submit_resources: Optional[Dict[str, float]] = None
        if gpu_enabled:
            try:
                scheduler_info = client.scheduler_info()
                worker_infos = scheduler_info.get("workers", {})
                has_gpu_resources = any(
                    float(dict(worker_info.get("resources", {})).get("GPU", 0.0)) >= 1.0
                    for worker_info in worker_infos.values()
                )
            except Exception:
                has_gpu_resources = False
            if has_gpu_resources:
                submit_resources = {"GPU": 1.0}
        futures = [
            client.submit(
                _run_usegment3d_for_volume,
                zarr_path=str(zarr_path),
                source_component=source_component,
                output_data_component=data_component,
                output_native_data_component=native_data_component,
                output_target_shape_zyx=(
                    int(output_shape[3]),
                    int(output_shape[4]),
                    int(output_shape[5]),
                ),
                t_index=t_index,
                p_index=p_index,
                source_channel_index=source_channel_index,
                output_channel_index=output_channel_index,
                parameters=dict(normalized),
                pure=False,
                resources=submit_resources,
            )
            for t_index, p_index, source_channel_index, output_channel_index in work_items
        ]
        completed = 0
        for future in as_completed(futures):
            try:
                _ = int(future.result())
            except FutureCancelledError as exc:
                raise RuntimeError(
                    "uSegment3D distributed task was cancelled before completion. "
                    "One or more Dask workers exited unexpectedly. "
                    f"Current scheduler state: {_summarize_client_worker_state(client)}."
                ) from exc
            completed += 1
            progress = 15 + int((completed / max(1, total)) * 80)
            _emit(progress, f"Processed uSegment3D volume {completed}/{total}")

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="usegment3d",
        component=component,
        metadata={
            "data_component": data_component,
            "native_data_component": native_data_component,
            "source_component": source_component,
            "requested_source_component": requested_source_component,
            "output_reference_component": output_reference_component,
            "volumes_processed": total,
            "channels_processed": len(channel_indices),
            "channel_indices": list(channel_indices),
            "channel_index": channel_index,
            "input_resolution_level": effective_resolution_level,
            "requested_input_resolution_level": requested_resolution_level,
            "output_resolution_level": output_resolution_level,
            "output_reference_space": output_reference_space,
            "save_native_labels_requested": save_native_labels_requested,
            "save_native_labels_effective": save_native_labels_effective,
            "views": list(normalized.get("use_views", ["xy", "xz", "yz"])),
            "output_shape_tpczyx": list(output_shape),
            "output_chunks_tpczyx": list(output_chunks),
            "native_shape_tpczyx": list(native_shape) if native_shape else None,
            "native_chunks_tpczyx": list(native_chunks) if native_chunks else None,
            "base_voxel_size_um_zyx": list(base_voxel_size_um_zyx),
            "source_voxel_size_um_zyx": list(source_voxel_size_um_zyx),
            "output_voxel_size_um_zyx": list(output_voxel_size_um_zyx),
            "gpu_requested": gpu_requested,
            "gpu_enabled": gpu_enabled,
            "require_gpu": require_gpu,
            "aggregation_tile_mode": aggregation_tile_mode_effective,
            "aggregation_tile_mode_requested": aggregation_tile_mode_requested,
            "aggregation_tile_mode_effective": aggregation_tile_mode_effective,
            "aggregation_tile_shape_zyx": list(
                normalized.get("aggregation_tile_shape_zyx", [128, 256, 256])
            ),
            "aggregation_tile_overlap_ratio": float(
                normalized.get("aggregation_tile_overlap_ratio", 0.25)
            ),
            "dropped_parameter_keys": list(dropped_parameter_keys),
            "parameters": {str(key): value for key, value in normalized.items()},
        },
    )
    _emit(100, "uSegment3D segmentation complete")

    return Usegment3dSummary(
        component=component,
        data_component=data_component,
        source_component=source_component,
        volumes_processed=total,
        channel_indices=tuple(int(value) for value in channel_indices),
        channel_index=channel_index,
        views=tuple(str(value) for value in list(normalized.get("use_views", []))),
        output_shape_tpczyx=output_shape,
        output_chunks_tpczyx=output_chunks,
        gpu_requested=gpu_requested,
        gpu_enabled=gpu_enabled,
        require_gpu=require_gpu,
    )
