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
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence
import argparse
import json
import logging
import math
import os


# Third Party Imports
import zarr

# Local Imports
from clearex.io.read import ImageInfo, ImageOpener
from clearex.io.experiment import (
    create_dask_client,
    is_navigate_experiment_file,
    load_navigate_experiment,
    load_store_spatial_calibration,
    materialize_experiment_data_store,
    resolve_data_store_path,
    resolve_experiment_data_path,
    save_store_spatial_calibration,
)
from clearex.io.cli import create_parser, display_logo
from clearex.io.log import initiate_logger
from clearex.io.provenance import (
    is_zarr_store_path,
    persist_run_provenance,
    register_latest_output_reference,
    summarize_analysis_history,
)
from clearex.detect.pipeline import (
    run_particle_detection_analysis,
)
from clearex.deconvolution.pipeline import (
    run_deconvolution_analysis,
)
from clearex.shear.pipeline import (
    run_shear_transform_analysis,
)
from clearex.flatfield.pipeline import (
    run_flatfield_analysis,
)
from clearex.visualization.pipeline import (
    run_display_pyramid_analysis,
    run_visualization_analysis,
)
from clearex.mip_export.pipeline import (
    run_mip_export_analysis,
)

try:
    from clearex.usegment3d.pipeline import (
        run_usegment3d_analysis,
    )
except ImportError:

    def run_usegment3d_analysis(*, zarr_path, parameters, client, progress_callback):
        """Fallback when the optional usegment3d runtime module is unavailable.

        Parameters
        ----------
        zarr_path : str
            Canonical analysis-store path.
        parameters : dict[str, Any]
            Runtime parameter mapping.
        client : Any
            Dask client handle.
        progress_callback : callable
            Progress callback.

        Returns
        -------
        None
            This fallback always raises before returning.

        Raises
        ------
        RuntimeError
            Always raised to indicate missing usegment3d implementation.
        """
        del zarr_path, parameters, client, progress_callback
        raise RuntimeError(
            "usegment3d analysis is unavailable: "
            "could not import clearex.usegment3d.pipeline."
        )


from clearex.workflow import (
    ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS,
    DASK_BACKEND_LOCAL_CLUSTER,
    DASK_BACKEND_SLURM_CLUSTER,
    DASK_BACKEND_SLURM_RUNNER,
    AnalysisInputReference,
    DaskBackendConfig,
    LocalClusterConfig,
    SpatialCalibrationConfig,
    WorkflowConfig,
    WorkflowExecutionCancelled,
    analysis_chainable_output_component,
    collect_analysis_input_references,
    dask_backend_from_dict,
    dask_backend_to_dict,
    format_spatial_calibration,
    format_dask_backend_summary,
    format_chunks,
    normalize_analysis_operation_parameters,
    parse_spatial_calibration,
    recommend_local_cluster_config,
    resolve_analysis_input_component,
    resolve_analysis_execution_sequence,
    validate_analysis_input_references,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    parse_chunks,
)

_CLEAREX_SETTINGS_DIR_NAME = ".clearex"
_CLEAREX_DASK_BACKEND_SETTINGS_FILE = "dask_backend_settings.json"

_ANALYSIS_OPERATIONS_REQUIRING_DASK_CLIENT = frozenset(
    {
        "flatfield",
        "deconvolution",
        "shear_transform",
        "particle_detection",
        "usegment3d",
        "display_pyramid",
        "mip_export",
    }
)
_ANALYSIS_OPERATIONS_WITH_PROVENANCE_DEDUP = frozenset(
    {
        "flatfield",
        "deconvolution",
        "shear_transform",
        "particle_detection",
        "usegment3d",
        "registration",
        "mip_export",
    }
)
_ANALYSIS_PROVENANCE_REQUIRED_COMPONENTS: Dict[str, tuple[str, ...]] = {
    "flatfield": (
        "results/flatfield/latest/data",
        "results/flatfield/latest/data_pyramid",
        "results/flatfield/latest/flatfield_pcyx",
        "results/flatfield/latest/darkfield_pcyx",
        "results/flatfield/latest/baseline_pctz",
    ),
    "deconvolution": ("results/deconvolution/latest/data",),
    "shear_transform": ("results/shear_transform/latest/data",),
    "particle_detection": ("results/particle_detection/latest/detections",),
    "usegment3d": ("results/usegment3d/latest/data",),
    "registration": ("results/registration/latest/data",),
    "display_pyramid": ("results/display_pyramid/latest",),
    "mip_export": ("results/mip_export/latest",),
}


class AnalysisDependencyError(RuntimeError):
    """Raised when a workflow requests an unavailable or invalid input."""


def _is_zarr_like_path(path: Path) -> bool:
    """Return whether a path represents a Zarr/N5 directory-style store name.

    Parameters
    ----------
    path : pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` when suffix is ``.zarr`` or ``.n5``.
    """
    return path.suffix.lower() in {".zarr", ".n5"}


def _resolve_log_directory_for_workflow(workflow: WorkflowConfig) -> Path:
    """Resolve directory path for runtime log file placement.

    Parameters
    ----------
    workflow : WorkflowConfig
        Effective workflow selected from CLI/GUI.

    Returns
    -------
    pathlib.Path
        Directory where runtime logs should be initialized.

    Raises
    ------
    Exception
        Propagates path-resolution errors from experiment metadata parsing.

    Notes
    -----
    For Navigate ``experiment.yml`` inputs, logs are colocated with the
    canonical data store path resolved by ingestion policy. When the resolved
    store path is Zarr/N5 but does not exist yet, logs are written to the
    parent directory so logging does not pre-create or depend on store layout.
    """
    if not workflow.file:
        return Path.cwd().resolve()

    selected = Path(workflow.file).expanduser().resolve()
    if is_navigate_experiment_file(selected):
        experiment = load_navigate_experiment(selected)
        source_data_path = resolve_experiment_data_path(experiment)
        store_path = resolve_data_store_path(experiment, source_data_path)
        if _is_zarr_like_path(store_path) and not store_path.exists():
            return (
                store_path.parent
                if store_path.parent != Path("")
                else Path.cwd().resolve()
            )
        return store_path if _is_zarr_like_path(store_path) else store_path.parent

    if _is_zarr_like_path(selected):
        if selected.exists():
            return selected
        return selected.parent if selected.parent != Path("") else Path.cwd().resolve()
    return selected.parent if selected.parent != Path("") else Path.cwd().resolve()


def _collect_available_analysis_components(
    zarr_path: Optional[str],
    references: Sequence[AnalysisInputReference],
) -> set[str]:
    """Collect store components available for dependency validation.

    Parameters
    ----------
    zarr_path : str, optional
        Active canonical analysis-store path.
    references : sequence[AnalysisInputReference]
        Requested input references to validate.

    Returns
    -------
    set[str]
        Components known to exist in the current store.
    """
    available_components = {"data"}
    if not zarr_path or not is_zarr_store_path(zarr_path):
        return available_components

    for operation_name, component in ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS.items():
        if operation_name == "data":
            continue
        if _zarr_component_exists(zarr_path, component):
            available_components.add(str(component))

    for reference in references:
        requested_source = str(reference.requested_source).strip()
        if (
            not requested_source
            or requested_source == "data"
            or requested_source in ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS
            or requested_source in available_components
        ):
            continue
        if _zarr_component_exists(zarr_path, requested_source):
            available_components.add(requested_source)
    return available_components


def _require_analysis_input_component(
    *,
    zarr_path: str,
    operation_name: str,
    field_name: str,
    requested_source: str,
    resolved_component: str,
) -> None:
    """Raise when a required analysis input component is unavailable.

    Parameters
    ----------
    zarr_path : str
        Active canonical analysis-store path.
    operation_name : str
        Operation consuming the input.
    field_name : str
        Parameter field name carrying the reference.
    requested_source : str
        Original requested alias or component path.
    resolved_component : str
        Resolved component path expected at runtime.

    Returns
    -------
    None
        Validation side effects only.

    Raises
    ------
    AnalysisDependencyError
        If the resolved component does not exist in the active store.
    """
    if _zarr_component_exists(zarr_path, resolved_component):
        return

    operation_label = str(operation_name).strip().replace("_", " ").title()
    requested_text = str(requested_source).strip() or "data"
    resolved_text = str(resolved_component).strip() or "data"
    raise AnalysisDependencyError(
        f"{operation_label} {field_name} requested '{requested_text}' and resolved "
        f"to '{resolved_text}', but that component is unavailable. ClearEx stops "
        "instead of falling back to raw data."
    )


def _analysis_execution_requires_dask_client(
    execution_sequence: tuple[str, ...],
) -> bool:
    """Return whether selected operations require a distributed Dask client.

    Parameters
    ----------
    execution_sequence : tuple[str, ...]
        Ordered analysis operations selected for execution.

    Returns
    -------
    bool
        ``True`` when at least one operation depends on a Dask client.
    """
    return any(
        operation in _ANALYSIS_OPERATIONS_REQUIRING_DASK_CLIENT
        for operation in execution_sequence
    )


def _zarr_component_exists(zarr_path: str, component: str) -> bool:
    """Return whether a component path exists inside a Zarr store.

    Parameters
    ----------
    zarr_path : str
        Zarr store path.
    component : str
        Group/array component path within the store.

    Returns
    -------
    bool
        ``True`` if the component can be resolved, otherwise ``False``.
    """
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        _ = root[component]
    except Exception:
        return False
    return True


def _create_bootstrap_logger() -> logging.Logger:
    """Create a minimal stdout-only logger used before workflow is known.

    Parameters
    ----------
    None

    Returns
    -------
    logging.Logger
        Logger configured with a single stream handler.
    """
    logger = logging.getLogger("clearex.bootstrap")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _resolve_effective_store_spatial_calibration(
    *,
    store_path: str,
    desired_calibration: SpatialCalibrationConfig,
    persist: bool,
) -> SpatialCalibrationConfig:
    """Load or persist the effective spatial calibration for one store.

    Parameters
    ----------
    store_path : str
        Canonical analysis-store path.
    desired_calibration : SpatialCalibrationConfig
        Candidate calibration selected for the current workflow.
    persist : bool
        Whether to write ``desired_calibration`` back to the store before
        returning it.

    Returns
    -------
    SpatialCalibrationConfig
        Effective store calibration used for the current run.
    """
    if persist:
        return save_store_spatial_calibration(store_path, desired_calibration)
    return load_store_spatial_calibration(store_path)


def _build_workflow_config(args: argparse.Namespace) -> WorkflowConfig:
    """Translate parsed CLI arguments into a workflow configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments from :func:`clearex.io.cli.create_parser`.

    Returns
    -------
    WorkflowConfig
        Workflow configuration used by GUI/headless execution.

    Raises
    ------
    ValueError
        If the ``--chunks`` option cannot be parsed as valid positive integers
        or uSegment3D CLI overrides contain invalid values.
    """
    usegment3d_analysis_parameters: Dict[str, Dict[str, Any]] = {}
    input_resolution_level_arg = getattr(args, "input_resolution_level", None)
    if input_resolution_level_arg is not None:
        parsed_input_resolution_level = int(input_resolution_level_arg)
        if parsed_input_resolution_level < 0:
            raise ValueError(
                "Invalid --input-resolution-level value. Level must be >= 0."
            )
        usegment3d_analysis_parameters.setdefault("usegment3d", {})[
            "input_resolution_level"
        ] = parsed_input_resolution_level

    channel_indices_arg = getattr(args, "channel_indices", None)
    if channel_indices_arg is not None and str(channel_indices_arg).strip():
        channel_tokens = [
            token.strip()
            for token in str(channel_indices_arg).replace("\n", ",").split(",")
            if token.strip()
        ]
        if len(channel_tokens) == 1 and channel_tokens[0].lower() in {"all", "*"}:
            usegment3d_analysis_parameters.setdefault("usegment3d", {})[
                "all_channels"
            ] = True
        else:
            parsed_indices: list[int] = []
            seen_indices: set[int] = set()
            for token in channel_tokens:
                try:
                    index = int(token)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid --channel-indices value. Use comma-separated "
                        "integers (for example '0,1,2') or 'all'."
                    ) from exc
                if index < 0:
                    raise ValueError(
                        "Invalid --channel-indices value. Channel indices must "
                        "be non-negative."
                    )
                if index in seen_indices:
                    continue
                parsed_indices.append(index)
                seen_indices.add(index)
            if parsed_indices:
                usegment3d_analysis_parameters.setdefault("usegment3d", {}).update(
                    {
                        "channel_indices": parsed_indices,
                        "channel_index": int(parsed_indices[0]),
                    }
                )

    persisted_dask_backend = _load_persisted_dask_backend_config()
    stage_axis_map_arg = getattr(args, "stage_axis_map", None)
    spatial_calibration_explicit = bool(
        stage_axis_map_arg is not None and str(stage_axis_map_arg).strip()
    )
    spatial_calibration = (
        parse_spatial_calibration(stage_axis_map_arg)
        if spatial_calibration_explicit
        else SpatialCalibrationConfig()
    )

    return WorkflowConfig(
        file=args.file,
        prefer_dask=args.dask,
        dask_backend=(
            persisted_dask_backend
            if persisted_dask_backend is not None
            else DaskBackendConfig()
        ),
        chunks=parse_chunks(args.chunks),
        flatfield=args.flatfield,
        deconvolution=args.deconvolution,
        shear_transform=args.shear_transform,
        particle_detection=args.particle_detection,
        usegment3d=args.usegment3d,
        registration=args.registration,
        display_pyramid=bool(getattr(args, "display_pyramid", False)),
        visualization=args.visualization,
        mip_export=args.mip_export,
        spatial_calibration=spatial_calibration,
        spatial_calibration_explicit=spatial_calibration_explicit,
        analysis_parameters=usegment3d_analysis_parameters,
    )


def _resolve_persisted_dask_backend_settings_path() -> Path:
    """Resolve the user settings JSON path for persisted backend config.

    Parameters
    ----------
    None

    Returns
    -------
    pathlib.Path
        Full JSON path used for persisted Dask backend settings.
    """
    return (
        Path.home()
        / _CLEAREX_SETTINGS_DIR_NAME
        / _CLEAREX_DASK_BACKEND_SETTINGS_FILE
    ).expanduser()


def _load_persisted_dask_backend_config() -> Optional[DaskBackendConfig]:
    """Load persisted Dask backend settings for CLI/headless execution.

    Parameters
    ----------
    None

    Returns
    -------
    DaskBackendConfig, optional
        Persisted backend configuration when available and valid.

    Notes
    -----
    Set ``CLEAREX_DISABLE_PERSISTED_DASK_BACKEND=1`` to ignore persisted
    backend settings for the current process.
    """
    disabled = str(os.environ.get("CLEAREX_DISABLE_PERSISTED_DASK_BACKEND", "")).strip()
    if disabled.lower() in {"1", "true", "yes", "on"}:
        return None

    settings_path = _resolve_persisted_dask_backend_settings_path()
    if not settings_path.exists():
        return None

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict) or not payload:
        return None
    return dask_backend_from_dict(payload)


def _extract_axis_map(info: ImageInfo) -> Dict[str, int]:
    """Map axis labels to corresponding dimension sizes.

    Parameters
    ----------
    info : ImageInfo
        Metadata returned by an image reader.

    Returns
    -------
    dict[str, int]
        Lowercase axis names mapped to sizes. Returns an empty dictionary if
        axis information is missing or incompatible with shape length.
    """
    if not info.axes or len(info.axes) != len(info.shape):
        return {}
    return {
        axis.lower(): int(size)
        for axis, size in zip(info.axes, info.shape, strict=False)
        if isinstance(axis, str)
    }


def _coerce_positive_float(value: Any) -> Optional[float]:
    """Parse a value into a finite positive float.

    Parameters
    ----------
    value : Any
        Candidate numeric value.

    Returns
    -------
    float, optional
        Parsed finite positive float, otherwise ``None``.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return float(parsed)


def _extract_pixel_size_um_zyx(info: ImageInfo) -> Optional[tuple[float, float, float]]:
    """Extract physical voxel size from ``ImageInfo``.

    Parameters
    ----------
    info : ImageInfo
        Loaded image metadata.

    Returns
    -------
    tuple[float, float, float], optional
        Physical voxel size in ``(z, y, x)`` microns.
    """
    raw_pixel_size = getattr(info, "pixel_size", None)
    if isinstance(raw_pixel_size, (tuple, list)) and len(raw_pixel_size) >= 3:
        z_um = _coerce_positive_float(raw_pixel_size[0])
        y_um = _coerce_positive_float(raw_pixel_size[1])
        x_um = _coerce_positive_float(raw_pixel_size[2])
        if z_um is not None and y_um is not None and x_um is not None:
            return (z_um, y_um, x_um)

    metadata = info.metadata or {}
    candidate_keys = (
        "voxel_size_um_zyx",
        "pixel_size_um_zyx",
        "physical_pixel_size_zyx",
        "pixel_size_zyx",
    )
    for key in candidate_keys:
        candidate = metadata.get(key)
        if not isinstance(candidate, (tuple, list)) or len(candidate) < 3:
            continue
        z_um = _coerce_positive_float(candidate[0])
        y_um = _coerce_positive_float(candidate[1])
        x_um = _coerce_positive_float(candidate[2])
        if z_um is None or y_um is None or x_um is None:
            continue
        return (z_um, y_um, x_um)

    navigate = metadata.get("navigate_experiment")
    if isinstance(navigate, dict):
        xy_um = _coerce_positive_float(navigate.get("xy_pixel_size_um"))
        z_um = _coerce_positive_float(navigate.get("z_step_um"))
        if xy_um is not None and z_um is not None:
            return (z_um, xy_um, xy_um)
    return None


def _log_loaded_image(info: ImageInfo, logger: logging.Logger) -> None:
    """Log key image metadata after successful load.

    Parameters
    ----------
    info : ImageInfo
        Loaded image metadata.
    logger : logging.Logger
        Logger used to emit metadata summary fields.

    Returns
    -------
    None
        This function logs side effects only.
    """
    axis_map = _extract_axis_map(info)
    channels = axis_map.get("c")
    positions = axis_map.get("p") or axis_map.get("s")
    time_points = axis_map.get("t")

    x_size = axis_map.get("x")
    y_size = axis_map.get("y")
    z_size = axis_map.get("z")
    if x_size is not None and y_size is not None:
        image_size = f"{x_size} x {y_size}"
        if z_size is not None:
            image_size += f" x {z_size}"
    else:
        fallback_spatial = info.shape[-3:] if len(info.shape) >= 3 else info.shape
        image_size = " x ".join(str(v) for v in fallback_spatial)

    logger.info(f"Image shape: {info.shape}")
    logger.info(f"Image dtype: {info.dtype}")
    logger.info(f"Image axes: {info.axes or 'n/a'}")
    logger.info(f"Channels: {channels if channels is not None else 'n/a'}")
    logger.info(f"Positions: {positions if positions is not None else 'n/a'}")
    logger.info(f"Time points: {time_points if time_points is not None else 'n/a'}")
    logger.info(f"Image size (XxYxZ): {image_size}")
    voxel_size_um_zyx = _extract_pixel_size_um_zyx(info)
    if voxel_size_um_zyx is None:
        logger.info("Pixel size (um): n/a")
    else:
        z_um, y_um, x_um = voxel_size_um_zyx
        logger.info(f"Pixel size (um): z={z_um:.6g}, y={y_um:.6g}, x={x_um:.6g}")

    if info.metadata:
        metadata_types = {
            key: type(value).__name__ for key, value in info.metadata.items()
        }
        logger.info(f"Image metadata keys/types: {metadata_types}")


def _apply_gui_if_requested(
    workflow: WorkflowConfig, args: argparse.Namespace, logger: logging.Logger
) -> Optional[WorkflowConfig]:
    """Launch GUI when enabled and return effective workflow configuration.

    Parameters
    ----------
    workflow : WorkflowConfig
        Baseline configuration derived from CLI values.
    args : argparse.Namespace
        Parsed command-line arguments controlling GUI/headless behavior.
    logger : logging.Logger
        Logger used for fallback and state messages.

    Returns
    -------
    WorkflowConfig or None
        Selected workflow configuration. Returns ``None`` when the GUI is opened
        and then cancelled by the user.

    Notes
    -----
    GUI initialization/runtime errors are caught and downgraded to headless
    fallback behavior to keep the entrypoint usable on servers/HPC nodes.
    """
    if args.headless:
        return workflow
    if not args.gui:
        return workflow

    try:
        from clearex.gui import GuiUnavailableError, launch_gui

        selected = launch_gui(initial=workflow)
        if selected is None:
            logger.info("GUI closed without running workflow.")
            return None
        return selected
    except Exception as exc:
        try:
            from clearex.gui import GuiUnavailableError

            if isinstance(exc, GuiUnavailableError):
                logger.warning(
                    f"GUI unavailable: {exc}. Falling back to headless mode."
                )
                return workflow
        except Exception:
            pass

        logger.warning(f"GUI launch failed ({type(exc).__name__}: {exc}).")
        logger.warning("Falling back to headless mode.")
        return workflow


def _configure_dask_backend(
    workflow: WorkflowConfig,
    logger: logging.Logger,
    exit_stack: ExitStack,
    *,
    workload: str = "io",
) -> Optional[Any]:
    """Initialize and register the configured Dask backend.

    Parameters
    ----------
    workflow : WorkflowConfig
        Effective workflow configuration including backend settings.
    logger : logging.Logger
        Logger used for status and fallback messages.
    exit_stack : contextlib.ExitStack
        Exit stack used to manage backend resource teardown.
    workload : str, default="io"
        Workload profile. ``"analysis"`` configures local clusters with
        processes. ``"io"`` uses processes when multiple workers are active to
        isolate worker memory accounting, and threads for single-worker mode.

    Returns
    -------
    Any, optional
        Active Dask client-like object when configured successfully;
        otherwise ``None``.

    Notes
    -----
    Backend initialization errors are converted into warnings and the workflow
    continues without a distributed client. This keeps local/headless paths
    operational even when optional Dask distributed backends are unavailable.
    When LocalCluster ``n_workers`` is unset, runtime applies aggressive
    host/data-aware defaults from
    :func:`clearex.workflow.recommend_local_cluster_config`, including worker
    count and, when left at defaults, thread and memory settings.
    """
    if not workflow.prefer_dask:
        logger.info("Dask lazy loading disabled; skipping backend startup.")
        return None

    backend = workflow.dask_backend
    workload_name = workload.strip().lower()
    logger.info(
        "Dask backend selection: "
        f"{format_dask_backend_summary(backend)} (workload={workload_name})"
    )

    try:
        if backend.mode == DASK_BACKEND_LOCAL_CLUSTER:
            local_cfg = backend.local_cluster
            requested_processes = workload_name == "analysis"
            default_local_cfg = LocalClusterConfig()
            effective_n_workers = local_cfg.n_workers
            effective_threads_per_worker = local_cfg.threads_per_worker
            effective_memory_limit = local_cfg.memory_limit
            if effective_n_workers is None:
                recommendation = recommend_local_cluster_config(
                    chunks_tpczyx=workflow.zarr_save.chunks_tpczyx(),
                )
                effective_n_workers = recommendation.config.n_workers
                if local_cfg.threads_per_worker == default_local_cfg.threads_per_worker:
                    effective_threads_per_worker = (
                        recommendation.config.threads_per_worker
                    )
                if (
                    str(local_cfg.memory_limit).strip().lower()
                    == str(default_local_cfg.memory_limit).strip().lower()
                ):
                    effective_memory_limit = recommendation.config.memory_limit
                logger.info(
                    "Auto-selected aggressive LocalCluster settings from "
                    "host/data recommendation: "
                    f"workers={effective_n_workers}, "
                    f"threads_per_worker={effective_threads_per_worker}, "
                    f"memory_limit={effective_memory_limit}, "
                    f"gpus={recommendation.detected_gpu_count}."
                )

            if workload_name == "analysis":
                gpu_worker_cap: Optional[int] = None
                use_gpu_local_cluster = False
                if bool(getattr(workflow, "usegment3d", False)):
                    try:
                        normalized_params = normalize_analysis_operation_parameters(
                            workflow.analysis_parameters
                        )
                    except Exception:
                        normalized_params = {}
                    usegment3d_params = dict(normalized_params.get("usegment3d", {}))
                    gpu_requested = bool(
                        usegment3d_params.get("gpu", False)
                        or usegment3d_params.get("require_gpu", False)
                    )
                    if gpu_requested:
                        gpu_recommendation = recommend_local_cluster_config(
                            chunks_tpczyx=workflow.zarr_save.chunks_tpczyx(),
                        )
                        detected_gpu_count = int(gpu_recommendation.detected_gpu_count)
                        if detected_gpu_count > 0:
                            gpu_worker_cap = max(1, detected_gpu_count)
                            use_gpu_local_cluster = True

                if gpu_worker_cap is not None:
                    requested_workers = (
                        int(effective_n_workers)
                        if effective_n_workers is not None
                        else int(gpu_worker_cap)
                    )
                    if requested_workers > int(gpu_worker_cap):
                        logger.info(
                            "GPU-aware LocalCluster cap applied for analysis: "
                            f"requested_workers={requested_workers}, "
                            f"capped_workers={int(gpu_worker_cap)}."
                        )
                        effective_n_workers = int(gpu_worker_cap)
            else:
                use_gpu_local_cluster = False

            effective_worker_count = (
                int(effective_n_workers) if effective_n_workers is not None else 1
            )
            use_processes = bool(requested_processes or effective_worker_count > 1)
            if not requested_processes and use_processes:
                logger.info(
                    "Using process-based LocalCluster for multi-worker I/O "
                    "execution (memory isolation enabled)."
                )
            client = create_dask_client(
                n_workers=effective_n_workers,
                threads_per_worker=effective_threads_per_worker,
                processes=use_processes,
                memory_limit=effective_memory_limit,
                local_directory=local_cfg.local_directory,
                gpu_enabled=use_gpu_local_cluster,
            )
            exit_stack.callback(client.close)
            logger.info(
                "Connected to LocalCluster backend "
                f"(processes={use_processes}, gpu_mode={use_gpu_local_cluster})."
            )
            return client

        if backend.mode == DASK_BACKEND_SLURM_RUNNER:
            scheduler_file = backend.slurm_runner.scheduler_file
            if not scheduler_file:
                raise ValueError("SLURMRunner backend requires a scheduler file path.")

            from dask.distributed import Client
            from dask_jobqueue.slurm import SLURMRunner

            runner = exit_stack.enter_context(
                SLURMRunner(scheduler_file=scheduler_file)
            )
            client = exit_stack.enter_context(Client(runner))

            wait_for_workers = backend.slurm_runner.wait_for_workers
            if wait_for_workers is None:
                runner_workers = getattr(runner, "n_workers", None)
                if isinstance(runner_workers, int) and runner_workers > 0:
                    wait_for_workers = runner_workers
            if wait_for_workers is not None:
                client.wait_for_workers(wait_for_workers)

            logger.info(
                f"Connected to SLURMRunner backend (scheduler_file={scheduler_file})."
            )
            return client

        if backend.mode == DASK_BACKEND_SLURM_CLUSTER:
            from dask.distributed import Client
            from dask_jobqueue import SLURMCluster

            cluster_cfg = backend.slurm_cluster
            if (
                workload.strip().lower() == "analysis"
                and int(cluster_cfg.processes) == 1
                and int(cluster_cfg.cores) > 1
            ):
                logger.warning(
                    "SLURMCluster is configured with processes=1 and cores=%s. "
                    "CPU-bound Python analyses (for example shear transform) may "
                    "underutilize allocated CPUs with this layout. "
                    "For maximum process-level parallelism, increase processes "
                    "toward cores in the Dask backend configuration.",
                    cluster_cfg.cores,
                )
            extra_directives = [
                directive.strip()
                for directive in cluster_cfg.job_extra_directives
                if directive.strip()
            ]
            if cluster_cfg.mail_user:
                extra_directives = [
                    directive
                    for directive in extra_directives
                    if not directive.startswith("--mail-user=")
                ]
                extra_directives.append(f"--mail-user={cluster_cfg.mail_user}")

            cluster_kwargs = {
                "cores": cluster_cfg.cores,
                "processes": cluster_cfg.processes,
                "memory": cluster_cfg.memory,
                "local_directory": cluster_cfg.local_directory,
                "interface": cluster_cfg.interface,
                "walltime": cluster_cfg.walltime,
                "job_name": cluster_cfg.job_name,
                "queue": cluster_cfg.queue,
                "death_timeout": cluster_cfg.death_timeout,
                "job_extra_directives": extra_directives,
                "scheduler_options": {
                    "dashboard_address": cluster_cfg.dashboard_address,
                    "interface": cluster_cfg.scheduler_interface,
                    "idle_timeout": cluster_cfg.idle_timeout,
                    "allowed_failures": cluster_cfg.allowed_failures,
                },
            }
            cluster = SLURMCluster(**cluster_kwargs)
            exit_stack.callback(cluster.close)
            cluster.scale(jobs=cluster_cfg.workers)

            client = Client(cluster)
            exit_stack.callback(client.close)
            client.wait_for_workers(cluster_cfg.workers)
            logger.info(
                "Connected to SLURMCluster backend "
                f"(workers={cluster_cfg.workers}, queue={cluster_cfg.queue})."
            )
            return client

        raise ValueError(f"Unsupported Dask backend mode: {backend.mode}")
    except Exception as exc:
        logger.warning(
            f"Failed to initialize Dask backend "
            f"({format_dask_backend_summary(backend)}): {exc}. "
            "Continuing without distributed client."
        )
        return None


def _run_workflow(
    workflow: WorkflowConfig,
    logger: logging.Logger,
    *,
    analysis_progress_callback: Optional[Callable[[int, str], None]] = None,
) -> None:
    """Execute a configured workflow in headless mode.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow parameters including file path and selected analyses.
    logger : logging.Logger
        Logger used for progress and status messages.
    analysis_progress_callback : callable, optional
        Optional callback used to report analysis progress as
        ``callback(percent, message)``.

    Returns
    -------
    None
        This function coordinates workflow side effects only.

    Raises
    ------
    FileNotFoundError
        If the configured image file path does not exist.
    ValueError
        If no reader can open the configured file.
    """

    def _emit_analysis_progress(percent: int, message: str) -> None:
        """Emit analysis progress callback updates when configured.

        Parameters
        ----------
        percent : int
            Progress percentage.
        message : str
            Human-readable stage text.

        Returns
        -------
        None
            Callback side effects only.
        """
        if analysis_progress_callback is None:
            return
        clamped = max(0, min(100, int(percent)))
        analysis_progress_callback(clamped, str(message))

    _emit_analysis_progress(1, "Preparing workflow execution.")

    # TODO: Persist workflow configuration for provenance/replay support.
    run_started_at = datetime.now(tz=timezone.utc)
    image_info: Optional[ImageInfo] = None
    step_records: list[Dict[str, object]] = []
    output_records: Dict[str, Dict[str, object]] = {}
    input_path = workflow.file
    provenance_store_path: Optional[str] = None
    run_status = "completed"
    cancellation_exc: Optional[WorkflowExecutionCancelled] = None
    failure_exc: Optional[Exception] = None
    runtime_analysis_parameters = normalize_analysis_operation_parameters(
        workflow.analysis_parameters
    )
    runtime_spatial_calibration = workflow.spatial_calibration

    if workflow.file:
        is_experiment_input = is_navigate_experiment_file(workflow.file)
        with ExitStack() as io_stack:
            io_client = (
                _configure_dask_backend(
                    workflow=workflow,
                    logger=logger,
                    exit_stack=io_stack,
                    workload="io",
                )
                if is_experiment_input
                else None
            )

            if is_experiment_input:
                experiment = load_navigate_experiment(workflow.file)
                resolved_data_path = resolve_experiment_data_path(experiment)
                input_path = str(resolved_data_path)
                zarr_chunks_tpczyx = workflow.zarr_save.chunks_tpczyx()
                zarr_pyramid_tpczyx = workflow.zarr_save.pyramid_tpczyx()

                logger.info(
                    f"Loaded experiment metadata from {workflow.file}: "
                    f"file_type={experiment.file_type}, "
                    f"timepoints={experiment.timepoints}, "
                    f"positions={experiment.multiposition_count}, "
                    f"channels={experiment.channel_count}, "
                    f"z_steps={experiment.number_z_steps}, "
                    f"zarr_chunks_ptczyx={format_zarr_chunks_ptczyx(workflow.zarr_save.chunks_ptczyx)}, "
                    "zarr_pyramid_ptczyx="
                    f"{format_zarr_pyramid_ptczyx(workflow.zarr_save.pyramid_ptczyx)}."
                )
                step_records.append(
                    {
                        "name": "load_experiment",
                        "parameters": {
                            "experiment_path": workflow.file,
                            "resolved_data_path": str(resolved_data_path),
                            "file_type": experiment.file_type,
                            "timepoints": experiment.timepoints,
                            "positions": experiment.multiposition_count,
                            "channels": experiment.channel_count,
                            "z_steps": experiment.number_z_steps,
                            "zarr_chunks_ptczyx": list(
                                workflow.zarr_save.chunks_ptczyx
                            ),
                            "zarr_pyramid_ptczyx": [
                                list(levels)
                                for levels in workflow.zarr_save.pyramid_ptczyx
                            ],
                        },
                    }
                )

                materialized = materialize_experiment_data_store(
                    experiment=experiment,
                    source_path=resolved_data_path,
                    chunks=zarr_chunks_tpczyx,
                    pyramid_factors=zarr_pyramid_tpczyx,
                    client=io_client,
                )
                image_info = materialized.source_image_info
                provenance_store_path = str(materialized.store_path)
                runtime_spatial_calibration = (
                    _resolve_effective_store_spatial_calibration(
                        store_path=provenance_store_path,
                        desired_calibration=workflow.spatial_calibration,
                        persist=workflow.spatial_calibration_explicit,
                    )
                )
                _log_loaded_image(image_info, logger)
                logger.info(
                    "Materialized source data to Zarr store "
                    f"{materialized.store_path} (component=data, "
                    f"chunks_tpczyx={materialized.chunks_tpczyx}, "
                    "spatial_calibration="
                    f"{format_spatial_calibration(runtime_spatial_calibration)})."
                )
                step_records.append(
                    {
                        "name": "materialize_data_store",
                        "parameters": {
                            "source_path": str(materialized.source_path),
                            "source_component": materialized.source_component,
                            "store_path": str(materialized.store_path),
                            "target_component": "data",
                            "canonical_shape_tpczyx": list(
                                materialized.data_image_info.shape
                            ),
                            "chunks_tpczyx": list(materialized.chunks_tpczyx),
                            "zarr_chunks_ptczyx": list(
                                workflow.zarr_save.chunks_ptczyx
                            ),
                            "zarr_pyramid_ptczyx": [
                                list(levels)
                                for levels in workflow.zarr_save.pyramid_ptczyx
                            ],
                            "spatial_calibration": format_spatial_calibration(
                                runtime_spatial_calibration
                            ),
                        },
                    }
                )
            else:
                opener = ImageOpener()
                _, info = opener.open(
                    input_path,
                    prefer_dask=workflow.prefer_dask,
                    chunks=workflow.chunks,
                )
                image_info = info
                _log_loaded_image(info, logger)

                if input_path and is_zarr_store_path(input_path):
                    provenance_store_path = input_path
                    runtime_spatial_calibration = (
                        _resolve_effective_store_spatial_calibration(
                            store_path=input_path,
                            desired_calibration=workflow.spatial_calibration,
                            persist=workflow.spatial_calibration_explicit,
                        )
                    )

        step_records.append(
            {
                "name": "load_data",
                "parameters": {
                    "source_path": input_path,
                    "prefer_dask": workflow.prefer_dask,
                    "chunks": format_chunks(workflow.chunks) or None,
                    "dask_backend": dask_backend_to_dict(workflow.dask_backend),
                    "spatial_calibration": format_spatial_calibration(
                        runtime_spatial_calibration
                    ),
                },
            }
        )
        _emit_analysis_progress(5, "Loaded source data and metadata.")

    with ExitStack() as analysis_stack:
        execution_sequence = resolve_analysis_execution_sequence(
            flatfield=workflow.flatfield,
            deconvolution=workflow.deconvolution,
            shear_transform=workflow.shear_transform,
            particle_detection=workflow.particle_detection,
            usegment3d=workflow.usegment3d,
            registration=workflow.registration,
            display_pyramid=workflow.display_pyramid,
            visualization=workflow.visualization,
            mip_export=workflow.mip_export,
            analysis_parameters=runtime_analysis_parameters,
        )

        if execution_sequence:
            logger.info(
                "Analysis execution sequence: %s",
                " -> ".join(execution_sequence),
            )
            _emit_analysis_progress(
                10,
                "Analysis sequence: " + " -> ".join(execution_sequence),
            )
        else:
            _emit_analysis_progress(100, "No analysis operations selected.")

        current_operation_name: Optional[str] = None
        current_requested_source = "data"
        current_resolved_source = "data"
        current_operation_parameters: Dict[str, Any] = {}
        references = collect_analysis_input_references(
            execution_sequence=execution_sequence,
            analysis_parameters=runtime_analysis_parameters,
        )
        available_components = _collect_available_analysis_components(
            provenance_store_path,
            references,
        )
        dependency_issues = validate_analysis_input_references(
            execution_sequence=execution_sequence,
            analysis_parameters=runtime_analysis_parameters,
            available_components=available_components,
        )
        if dependency_issues:
            first_issue = dependency_issues[0]
            run_status = "failed"
            current_operation_name = str(first_issue.consumer_operation)
            current_requested_source = str(first_issue.requested_source)
            current_resolved_source = (
                str(first_issue.resolved_component).strip()
                if first_issue.resolved_component is not None
                else current_requested_source
            ) or "data"
            current_operation_parameters = dict(
                runtime_analysis_parameters.get(first_issue.consumer_operation, {})
            )
            current_operation_parameters.update(
                {
                    "status": "failed",
                    "reason": str(first_issue.reason),
                    "dependency_message": str(first_issue.message),
                    "requested_input": current_requested_source,
                    "resolved_input": current_resolved_source,
                }
            )
            step_records.append(
                {
                    "name": current_operation_name,
                    "parameters": dict(current_operation_parameters),
                }
            )
            failure_exc = AnalysisDependencyError(str(first_issue.message))
            logger.error(
                "Invalid analysis dependency for '%s': %s",
                current_operation_name,
                first_issue.message,
            )
            _emit_analysis_progress(100, str(first_issue.message))

        analysis_client = (
            _configure_dask_backend(
                workflow=workflow,
                logger=logger,
                exit_stack=analysis_stack,
                workload="analysis",
            )
            if failure_exc is None
            and _analysis_execution_requires_dask_client(execution_sequence)
            else None
        )

        produced_components: Dict[str, str] = {"data": "data"}
        total_operations = max(1, len(execution_sequence))
        try:
            if failure_exc is not None:
                raise failure_exc
            for operation_index, operation_name in enumerate(execution_sequence):
                current_operation_name = str(operation_name)
                current_requested_source = "data"
                current_resolved_source = "data"
                current_operation_parameters = {}
                operation_start = 10 + int((operation_index / total_operations) * 85)
                operation_end = 10 + int(
                    ((operation_index + 1) / total_operations) * 85
                )

                operation_parameters = dict(
                    runtime_analysis_parameters.get(operation_name, {})
                )
                requested_source = (
                    str(operation_parameters.get("input_source", "data")).strip()
                    or "data"
                )
                resolved_source = resolve_analysis_input_component(
                    requested_source=requested_source,
                    produced_components=produced_components,
                )
                operation_parameters["input_source"] = resolved_source
                runtime_analysis_parameters[operation_name] = operation_parameters
                current_requested_source = str(requested_source)
                current_resolved_source = str(resolved_source)
                current_operation_parameters = dict(operation_parameters)
                logger.info(
                    "Starting %s analysis (requested_input=%s, resolved_input=%s, store=%s).",
                    operation_name,
                    requested_source,
                    resolved_source,
                    provenance_store_path,
                )

                force_rerun = bool(operation_parameters.get("force_rerun", False))
                if (
                    operation_name in _ANALYSIS_OPERATIONS_WITH_PROVENANCE_DEDUP
                    and provenance_store_path
                    and is_zarr_store_path(provenance_store_path)
                    and not force_rerun
                ):
                    try:
                        history = summarize_analysis_history(
                            provenance_store_path,
                            operation_name,
                            parameters=operation_parameters,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Could not read provenance history for %s: %s",
                            operation_name,
                            exc,
                        )
                    else:
                        if bool(history.get("matches_parameters", False)):
                            required_components = (
                                _ANALYSIS_PROVENANCE_REQUIRED_COMPONENTS.get(
                                    operation_name
                                )
                            )
                            missing_components: list[str] = []
                            for component in required_components or ():
                                if _zarr_component_exists(
                                    provenance_store_path, component
                                ):
                                    continue
                                else:
                                    missing_components.append(component)
                            output_available = not missing_components
                            if output_available:
                                matching_run_id = str(
                                    history.get("matching_run_id") or ""
                                )
                                logger.info(
                                    "Skipping %s because a successful matching run "
                                    "already exists in provenance%s.",
                                    operation_name,
                                    (
                                        f" (run_id={matching_run_id})"
                                        if matching_run_id
                                        else ""
                                    ),
                                )
                                chainable_component = (
                                    analysis_chainable_output_component(operation_name)
                                )
                                if (
                                    chainable_component is not None
                                    and operation_name != "data"
                                ):
                                    produced_components[operation_name] = (
                                        chainable_component
                                    )
                                step_records.append(
                                    {
                                        "name": operation_name,
                                        "parameters": {
                                            **operation_parameters,
                                            "status": "skipped",
                                            "reason": "provenance_parameter_match",
                                            "matching_run_id": (
                                                matching_run_id or None
                                            ),
                                            "matching_ended_utc": history.get(
                                                "matching_ended_utc"
                                            ),
                                        },
                                    }
                                )
                                _emit_analysis_progress(
                                    operation_end,
                                    f"{operation_name.replace('_', ' ').title()} skipped "
                                    "(matching provenance run).",
                                )
                                continue
                            logger.info(
                                "Matching provenance run found for %s, but required output "
                                "components are missing (%s). Re-running operation.",
                                operation_name,
                                ", ".join(missing_components)
                                if missing_components
                                else "none",
                            )

                if (
                    operation_name != "visualization"
                    and provenance_store_path
                    and is_zarr_store_path(provenance_store_path)
                ):
                    _require_analysis_input_component(
                        zarr_path=provenance_store_path,
                        operation_name=str(operation_name),
                        field_name="input_source",
                        requested_source=str(requested_source),
                        resolved_component=str(resolved_source),
                    )

                if operation_name == "flatfield":
                    flatfield_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        progress_state = {"last_percent": -5}

                        def _flatfield_progress(percent: int, message: str) -> None:
                            """Throttle flatfield progress logs.

                            Parameters
                            ----------
                            percent : int
                                Progress percent.
                            message : str
                                Progress message.

                            Returns
                            -------
                            None
                                Logger side effects only.
                            """
                            last_percent = int(progress_state["last_percent"])
                            if percent >= 100 or percent - last_percent >= 5:
                                progress_state["last_percent"] = int(percent)
                                logger.info(f"[flatfield] {int(percent)}% - {message}")
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            _emit_analysis_progress(
                                mapped,
                                f"flatfield: {message}",
                            )

                        summary = run_flatfield_analysis(
                            zarr_path=provenance_store_path,
                            parameters=flatfield_parameters,
                            client=analysis_client,
                            progress_callback=_flatfield_progress,
                        )
                        flatfield_source_component = str(
                            getattr(
                                summary,
                                "source_component",
                                flatfield_parameters.get("input_source", "data"),
                            )
                        )
                        flatfield_basicpy_version = getattr(
                            summary,
                            "basicpy_version",
                            None,
                        )
                        produced_components["flatfield"] = summary.data_component
                        output_records["flatfield"] = {
                            "component": summary.component,
                            "data_component": summary.data_component,
                            "flatfield_component": summary.flatfield_component,
                            "darkfield_component": summary.darkfield_component,
                            "baseline_component": summary.baseline_component,
                            "source_component": flatfield_source_component,
                            "profile_count": summary.profile_count,
                            "transformed_volumes": summary.transformed_volumes,
                            "output_chunks_tpczyx": list(summary.output_chunks_tpczyx),
                            "output_dtype": summary.output_dtype,
                            "basicpy_version": flatfield_basicpy_version,
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "Flatfield correction completed: "
                            f"profiles={summary.profile_count}, "
                            f"transformed_volumes={summary.transformed_volumes}, "
                            f"component={summary.component}."
                        )
                        step_records.append(
                            {
                                "name": "flatfield",
                                "parameters": {
                                    **flatfield_parameters,
                                    "component": summary.component,
                                    "data_component": summary.data_component,
                                    "flatfield_component": summary.flatfield_component,
                                    "darkfield_component": summary.darkfield_component,
                                    "baseline_component": summary.baseline_component,
                                    "source_component": flatfield_source_component,
                                    "profile_count": summary.profile_count,
                                    "transformed_volumes": summary.transformed_volumes,
                                    "output_dtype": summary.output_dtype,
                                    "basicpy_version": flatfield_basicpy_version,
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Flatfield correction complete.",
                        )
                    else:
                        logger.warning(
                            "Flatfield correction requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "flatfield",
                                "parameters": {
                                    **flatfield_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Flatfield correction skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "deconvolution":
                    decon_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        progress_state = {"last_percent": -5}

                        def _decon_progress(percent: int, message: str) -> None:
                            """Throttle deconvolution progress logs.

                            Parameters
                            ----------
                            percent : int
                                Progress percent.
                            message : str
                                Progress message.

                            Returns
                            -------
                            None
                                Logger side effects only.
                            """
                            last_percent = int(progress_state["last_percent"])
                            if percent >= 100 or percent - last_percent >= 5:
                                progress_state["last_percent"] = int(percent)
                                logger.info(
                                    f"[deconvolution] {int(percent)}% - {message}"
                                )
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            _emit_analysis_progress(
                                mapped,
                                f"deconvolution: {message}",
                            )

                        summary = run_deconvolution_analysis(
                            zarr_path=provenance_store_path,
                            parameters=decon_parameters,
                            client=analysis_client,
                            progress_callback=_decon_progress,
                        )
                        produced_components["deconvolution"] = summary.data_component
                        output_records["deconvolution"] = {
                            "component": summary.component,
                            "data_component": summary.data_component,
                            "volumes_processed": summary.volumes_processed,
                            "channel_count": summary.channel_count,
                            "psf_mode": summary.psf_mode,
                            "output_chunks_tpczyx": list(summary.output_chunks_tpczyx),
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "Deconvolution completed: "
                            f"volumes_processed={summary.volumes_processed}, "
                            f"channels={summary.channel_count}, "
                            f"psf_mode={summary.psf_mode}, "
                            f"component={summary.component}."
                        )
                        step_records.append(
                            {
                                "name": "deconvolution",
                                "parameters": {
                                    **decon_parameters,
                                    "component": summary.component,
                                    "data_component": summary.data_component,
                                    "volumes_processed": summary.volumes_processed,
                                    "channel_count": summary.channel_count,
                                    "psf_mode": summary.psf_mode,
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Deconvolution complete.",
                        )
                    else:
                        logger.warning(
                            "Deconvolution requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "deconvolution",
                                "parameters": {
                                    **decon_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Deconvolution skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "shear_transform":
                    shear_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        progress_state = {"last_percent": -5}

                        def _shear_progress(percent: int, message: str) -> None:
                            """Throttle shear-transform progress logs.

                            Parameters
                            ----------
                            percent : int
                                Progress percent.
                            message : str
                                Progress message.

                            Returns
                            -------
                            None
                                Logger side effects only.
                            """
                            last_percent = int(progress_state["last_percent"])
                            if percent >= 100 or percent - last_percent >= 5:
                                progress_state["last_percent"] = int(percent)
                                logger.info(
                                    f"[shear_transform] {int(percent)}% - {message}"
                                )
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            _emit_analysis_progress(
                                mapped,
                                f"shear_transform: {message}",
                            )

                        summary = run_shear_transform_analysis(
                            zarr_path=provenance_store_path,
                            parameters=shear_parameters,
                            client=analysis_client,
                            progress_callback=_shear_progress,
                        )
                        produced_components["shear_transform"] = summary.data_component
                        output_records["shear_transform"] = {
                            "component": summary.component,
                            "data_component": summary.data_component,
                            "volumes_processed": summary.volumes_processed,
                            "output_shape_tpczyx": list(summary.output_shape_tpczyx),
                            "output_chunks_tpczyx": list(summary.output_chunks_tpczyx),
                            "voxel_size_um_zyx": list(summary.voxel_size_um_zyx),
                            "applied_shear": {
                                "xy": summary.applied_shear_xy,
                                "xz": summary.applied_shear_xz,
                                "yz": summary.applied_shear_yz,
                            },
                            "applied_rotation_deg_xyz": list(
                                summary.applied_rotation_deg_xyz
                            ),
                            "interpolation": summary.interpolation,
                            "output_dtype": summary.output_dtype,
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "Shear transform completed: "
                            f"volumes_processed={summary.volumes_processed}, "
                            f"shape={summary.output_shape_tpczyx}, "
                            f"component={summary.component}."
                        )
                        step_records.append(
                            {
                                "name": "shear_transform",
                                "parameters": {
                                    **shear_parameters,
                                    "component": summary.component,
                                    "data_component": summary.data_component,
                                    "volumes_processed": summary.volumes_processed,
                                    "output_shape_tpczyx": list(
                                        summary.output_shape_tpczyx
                                    ),
                                    "output_chunks_tpczyx": list(
                                        summary.output_chunks_tpczyx
                                    ),
                                    "voxel_size_um_zyx": list(
                                        summary.voxel_size_um_zyx
                                    ),
                                    "applied_shear_xy": summary.applied_shear_xy,
                                    "applied_shear_xz": summary.applied_shear_xz,
                                    "applied_shear_yz": summary.applied_shear_yz,
                                    "applied_rotation_deg_xyz": list(
                                        summary.applied_rotation_deg_xyz
                                    ),
                                    "interpolation": summary.interpolation,
                                    "output_dtype": summary.output_dtype,
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Shear transform complete.",
                        )
                    else:
                        logger.warning(
                            "Shear transform requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "shear_transform",
                                "parameters": {
                                    **shear_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Shear transform skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "particle_detection":
                    particle_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        progress_state = {"last_percent": -5}

                        def _particle_progress(percent: int, message: str) -> None:
                            """Throttle particle-detection progress logs.

                            Parameters
                            ----------
                            percent : int
                                Progress percent.
                            message : str
                                Progress message.

                            Returns
                            -------
                            None
                                Logger side effects only.
                            """
                            last_percent = int(progress_state["last_percent"])
                            if percent >= 100 or percent - last_percent >= 5:
                                progress_state["last_percent"] = int(percent)
                                logger.info(
                                    f"[particle_detection] {int(percent)}% - {message}"
                                )
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            _emit_analysis_progress(
                                mapped,
                                f"particle_detection: {message}",
                            )

                        summary = run_particle_detection_analysis(
                            zarr_path=provenance_store_path,
                            parameters=particle_parameters,
                            client=analysis_client,
                            progress_callback=_particle_progress,
                        )
                        output_records["particle_detection"] = {
                            "component": summary.component,
                            "detections": summary.detections,
                            "chunks_processed": summary.chunks_processed,
                            "channel_index": summary.channel_index,
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "Particle detection completed: "
                            f"detections={summary.detections}, "
                            f"chunks_processed={summary.chunks_processed}, "
                            f"channel={summary.channel_index}, "
                            f"component={summary.component}."
                        )
                        step_records.append(
                            {
                                "name": "particle_detection",
                                "parameters": {
                                    **particle_parameters,
                                    "detections": summary.detections,
                                    "chunks_processed": summary.chunks_processed,
                                    "component": summary.component,
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Particle detection complete.",
                        )
                    else:
                        logger.warning(
                            "Particle detection requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "particle_detection",
                                "parameters": {
                                    **particle_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Particle detection skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "usegment3d":
                    usegment3d_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        progress_state = {"last_percent": -5}

                        def _usegment3d_progress(percent: int, message: str) -> None:
                            """Throttle usegment3d progress logs.

                            Parameters
                            ----------
                            percent : int
                                Progress percent.
                            message : str
                                Progress message.

                            Returns
                            -------
                            None
                                Logger side effects only.
                            """
                            last_percent = int(progress_state["last_percent"])
                            if percent >= 100 or percent - last_percent >= 5:
                                progress_state["last_percent"] = int(percent)
                                logger.info(f"[usegment3d] {int(percent)}% - {message}")
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            _emit_analysis_progress(
                                mapped,
                                f"usegment3d: {message}",
                            )

                        summary = run_usegment3d_analysis(
                            zarr_path=provenance_store_path,
                            parameters=usegment3d_parameters,
                            client=analysis_client,
                            progress_callback=_usegment3d_progress,
                        )
                        usegment3d_component = str(
                            getattr(summary, "component", "results/usegment3d/latest")
                        )
                        usegment3d_data_component = str(
                            getattr(
                                summary,
                                "data_component",
                                f"{usegment3d_component}/data",
                            )
                        )
                        usegment3d_source_component = str(
                            getattr(
                                summary,
                                "source_component",
                                usegment3d_parameters.get("input_source", "data"),
                            )
                        )
                        usegment3d_channel_indices = [
                            int(value)
                            for value in list(
                                getattr(
                                    summary,
                                    "channel_indices",
                                    [getattr(summary, "channel_index", 0)],
                                )
                            )
                        ]
                        if not usegment3d_channel_indices:
                            usegment3d_channel_indices = [
                                int(getattr(summary, "channel_index", 0))
                            ]
                        produced_components["usegment3d"] = usegment3d_data_component
                        output_records["usegment3d"] = {
                            "component": usegment3d_component,
                            "data_component": usegment3d_data_component,
                            "source_component": usegment3d_source_component,
                            "channel_indices": list(usegment3d_channel_indices),
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "usegment3d completed: "
                            f"component={usegment3d_component}, "
                            f"data_component={usegment3d_data_component}, "
                            f"source={usegment3d_source_component}, "
                            f"channels={usegment3d_channel_indices}."
                        )
                        step_records.append(
                            {
                                "name": "usegment3d",
                                "parameters": {
                                    **usegment3d_parameters,
                                    "component": usegment3d_component,
                                    "data_component": usegment3d_data_component,
                                    "source_component": usegment3d_source_component,
                                    "channel_indices": list(usegment3d_channel_indices),
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "usegment3d complete.",
                        )
                    else:
                        logger.warning(
                            "usegment3d requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "usegment3d",
                                "parameters": {
                                    **usegment3d_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "usegment3d skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "registration":
                    _emit_analysis_progress(
                        operation_start,
                        "Running registration workflow.",
                    )
                    logger.info(
                        "Running registration workflow (input=%s).",
                        resolved_source,
                    )
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        logger.warning(
                            "Registration is enabled but is not yet integrated with "
                            "canonical 6D store inputs. Skipping registration."
                        )
                        step_records.append(
                            {
                                "name": "registration",
                                "parameters": {
                                    **operation_parameters,
                                    "status": "skipped",
                                    "reason": "not_integrated_with_canonical_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Registration skipped (not yet integrated with canonical store).",
                        )
                    else:
                        logger.warning(
                            "Registration requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "registration",
                                "parameters": {
                                    **operation_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Registration skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "display_pyramid":
                    display_pyramid_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        def _display_pyramid_progress(
                            percent: int, message: str
                        ) -> None:
                            """Map display-pyramid progress into workflow-scale progress.

                            Parameters
                            ----------
                            percent : int
                                Display-pyramid progress percent.
                            message : str
                                Progress status text.

                            Returns
                            -------
                            None
                                Logger and progress-callback side effects only.
                            """
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            logger.info(
                                f"[display_pyramid] {int(percent)}% - {message}"
                            )
                            _emit_analysis_progress(
                                mapped,
                                f"display_pyramid: {message}",
                            )

                        summary = run_display_pyramid_analysis(
                            zarr_path=provenance_store_path,
                            parameters=display_pyramid_parameters,
                            client=analysis_client,
                            progress_callback=_display_pyramid_progress,
                        )
                        output_records["display_pyramid"] = {
                            "component": summary.component,
                            "source_component": summary.source_component,
                            "source_components": list(summary.source_components),
                            "channel_count": summary.channel_count,
                            "contrast_limits_by_channel": [
                                [float(row[0]), float(row[1])]
                                for row in summary.contrast_limits_by_channel
                            ],
                            "reused_existing_levels": summary.reused_existing_levels,
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "Display pyramid preparation completed: "
                            f"component={summary.component}, "
                            f"source={summary.source_component}, "
                            f"levels={len(summary.source_components)}, "
                            f"channels={summary.channel_count}, "
                            f"reused_existing_levels={summary.reused_existing_levels}."
                        )
                        step_records.append(
                            {
                                "name": "display_pyramid",
                                "parameters": {
                                    **display_pyramid_parameters,
                                    "component": summary.component,
                                    "source_component": summary.source_component,
                                    "source_components": list(
                                        summary.source_components
                                    ),
                                    "channel_count": summary.channel_count,
                                    "contrast_limits_by_channel": [
                                        [float(row[0]), float(row[1])]
                                        for row in summary.contrast_limits_by_channel
                                    ],
                                    "reused_existing_levels": summary.reused_existing_levels,
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Display pyramid preparation complete.",
                        )
                    else:
                        logger.warning(
                            "Display pyramid preparation requires a canonical Zarr/N5 "
                            "data store."
                        )
                        step_records.append(
                            {
                                "name": "display_pyramid",
                                "parameters": {
                                    **display_pyramid_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Display pyramid preparation skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "visualization":
                    visualization_parameters = dict(operation_parameters)
                    raw_volume_layers = visualization_parameters.get(
                        "volume_layers", []
                    )
                    resolved_volume_layers: list[dict[str, Any]] = []
                    if isinstance(raw_volume_layers, (tuple, list)):
                        for raw_row in raw_volume_layers:
                            if not isinstance(raw_row, dict):
                                continue
                            row = dict(raw_row)
                            requested_component = str(
                                row.get("component", row.get("source_component", ""))
                            ).strip()
                            if not requested_component:
                                continue
                            resolved_component = resolve_analysis_input_component(
                                requested_source=requested_component,
                                produced_components=produced_components,
                            )
                            row["component"] = str(resolved_component)
                            if "source_component" in row:
                                row["source_component"] = str(resolved_component)
                            resolved_volume_layers.append(row)
                    if resolved_volume_layers:
                        visualization_parameters["volume_layers"] = (
                            resolved_volume_layers
                        )
                        first_component = (
                            str(
                                resolved_volume_layers[0].get("component", "data")
                            ).strip()
                            or "data"
                        )
                        visualization_parameters["input_source"] = first_component
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        _require_analysis_input_component(
                            zarr_path=provenance_store_path,
                            operation_name="visualization",
                            field_name="input_source",
                            requested_source=str(current_requested_source),
                            resolved_component=str(
                                visualization_parameters.get("input_source", "data")
                            ),
                        )
                        volume_layers_value = visualization_parameters.get(
                            "volume_layers"
                        )
                        if isinstance(volume_layers_value, (tuple, list)):
                            sanitized_volume_layers: list[dict[str, Any]] = []
                            for index, row in enumerate(volume_layers_value):
                                if not isinstance(row, dict):
                                    continue
                                row_copy = dict(row)
                                resolved_component = (
                                    str(
                                        row_copy.get(
                                            "component",
                                            row_copy.get("source_component", ""),
                                        )
                                    ).strip()
                                    or "data"
                                )
                                _require_analysis_input_component(
                                    zarr_path=provenance_store_path,
                                    operation_name="visualization",
                                    field_name=f"volume_layers[{index}].component",
                                    requested_source=str(
                                        row.get(
                                            "component",
                                            row.get("source_component", ""),
                                        )
                                    ).strip()
                                    or "data",
                                    resolved_component=resolved_component,
                                )
                                row_copy["component"] = resolved_component
                                if "source_component" in row_copy:
                                    row_copy["source_component"] = resolved_component
                                sanitized_volume_layers.append(row_copy)
                            if sanitized_volume_layers:
                                visualization_parameters["volume_layers"] = (
                                    sanitized_volume_layers
                                )
                                visualization_parameters["input_source"] = (
                                    str(
                                        sanitized_volume_layers[0].get(
                                            "component", "data"
                                        )
                                    ).strip()
                                    or "data"
                                )
                            runtime_analysis_parameters["visualization"] = dict(
                                visualization_parameters
                            )

                        def _visualization_progress(percent: int, message: str) -> None:
                            """Map visualization progress into workflow-scale progress.

                            Parameters
                            ----------
                            percent : int
                                Visualization progress percent.
                            message : str
                                Progress status text.

                            Returns
                            -------
                            None
                                Logger and progress-callback side effects only.
                            """
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            logger.info(f"[visualization] {int(percent)}% - {message}")
                            _emit_analysis_progress(
                                mapped,
                                f"visualization: {message}",
                            )

                        summary = run_visualization_analysis(
                            zarr_path=provenance_store_path,
                            parameters=visualization_parameters,
                            progress_callback=_visualization_progress,
                        )
                        viewer_ndisplay_requested = int(
                            getattr(summary, "viewer_ndisplay_requested", 3)
                        )
                        viewer_ndisplay_effective = int(
                            getattr(
                                summary,
                                "viewer_ndisplay_effective",
                                viewer_ndisplay_requested,
                            )
                        )
                        display_mode_fallback_reason = getattr(
                            summary,
                            "display_mode_fallback_reason",
                            None,
                        )
                        output_records["visualization"] = {
                            "component": summary.component,
                            "source_component": summary.source_component,
                            "source_components": list(summary.source_components),
                            "volume_layers": list(
                                visualization_parameters.get("volume_layers", [])
                            ),
                            "position_index": summary.position_index,
                            "overlay_points_count": summary.overlay_points_count,
                            "launch_mode": summary.launch_mode,
                            "viewer_pid": summary.viewer_pid,
                            "viewer_ndisplay_requested": viewer_ndisplay_requested,
                            "viewer_ndisplay_effective": viewer_ndisplay_effective,
                            "display_mode_fallback_reason": display_mode_fallback_reason,
                            "renderer": dict(getattr(summary, "renderer", None) or {}),
                            "keyframe_manifest_path": summary.keyframe_manifest_path,
                            "keyframe_count": summary.keyframe_count,
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "Visualization workflow completed: "
                            f"component={summary.component}, "
                            f"source={summary.source_component}, "
                            f"position={summary.position_index}, "
                            f"multiscale_levels={len(summary.source_components)}, "
                            f"viewer_ndisplay={viewer_ndisplay_effective}, "
                            f"overlay_points={summary.overlay_points_count}, "
                            f"launch_mode={summary.launch_mode}, "
                            f"keyframes={summary.keyframe_count}, "
                            f"keyframe_manifest={summary.keyframe_manifest_path}."
                        )
                        step_records.append(
                            {
                                "name": "visualization",
                                "parameters": {
                                    **visualization_parameters,
                                    "component": summary.component,
                                    "source_component": summary.source_component,
                                    "source_components": list(
                                        summary.source_components
                                    ),
                                    "position_index": summary.position_index,
                                    "overlay_points_count": summary.overlay_points_count,
                                    "launch_mode": summary.launch_mode,
                                    "viewer_pid": summary.viewer_pid,
                                    "viewer_ndisplay_requested": viewer_ndisplay_requested,
                                    "viewer_ndisplay_effective": viewer_ndisplay_effective,
                                    "display_mode_fallback_reason": display_mode_fallback_reason,
                                    "keyframe_manifest_path": summary.keyframe_manifest_path,
                                    "keyframe_count": summary.keyframe_count,
                                },
                            }
                        )
                        if summary.launch_mode == "subprocess":
                            _emit_analysis_progress(
                                operation_end,
                                "Visualization launched in a separate napari process.",
                            )
                        else:
                            _emit_analysis_progress(
                                operation_end,
                                "Visualization viewer closed; workflow continuing.",
                            )
                    else:
                        logger.warning(
                            "Visualization requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "visualization",
                                "parameters": {
                                    **visualization_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "Visualization skipped (no Zarr/N5 store).",
                        )
                    continue

                if operation_name == "mip_export":
                    mip_parameters = dict(operation_parameters)
                    if provenance_store_path and is_zarr_store_path(
                        provenance_store_path
                    ):
                        def _mip_export_progress(percent: int, message: str) -> None:
                            """Map MIP-export progress into workflow-scale progress.

                            Parameters
                            ----------
                            percent : int
                                MIP-export progress percent.
                            message : str
                                Progress status text.

                            Returns
                            -------
                            None
                                Logger and progress-callback side effects only.
                            """
                            mapped = operation_start + int(
                                (max(0, min(100, int(percent))) / 100)
                                * max(1, operation_end - operation_start)
                            )
                            logger.info(f"[mip_export] {int(percent)}% - {message}")
                            _emit_analysis_progress(
                                mapped,
                                f"mip_export: {message}",
                            )

                        summary = run_mip_export_analysis(
                            zarr_path=provenance_store_path,
                            parameters=mip_parameters,
                            client=analysis_client,
                            progress_callback=_mip_export_progress,
                        )
                        output_records["mip_export"] = {
                            "component": summary.component,
                            "source_component": summary.source_component,
                            "output_directory": summary.output_directory,
                            "export_format": summary.export_format,
                            "position_mode": summary.position_mode,
                            "task_count": summary.task_count,
                            "exported_files": summary.exported_files,
                            "storage_policy": "latest_only",
                        }
                        logger.info(
                            "MIP export completed: "
                            f"component={summary.component}, "
                            f"source={summary.source_component}, "
                            f"format={summary.export_format}, "
                            f"position_mode={summary.position_mode}, "
                            f"files={summary.exported_files}, "
                            f"output_directory={summary.output_directory}."
                        )
                        step_records.append(
                            {
                                "name": "mip_export",
                                "parameters": {
                                    **mip_parameters,
                                    "component": summary.component,
                                    "source_component": summary.source_component,
                                    "output_directory": summary.output_directory,
                                    "export_format": summary.export_format,
                                    "position_mode": summary.position_mode,
                                    "task_count": summary.task_count,
                                    "exported_files": summary.exported_files,
                                    "projections": list(summary.projections),
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "MIP export complete.",
                        )
                    else:
                        logger.warning(
                            "MIP export requires a canonical Zarr/N5 data store."
                        )
                        step_records.append(
                            {
                                "name": "mip_export",
                                "parameters": {
                                    **mip_parameters,
                                    "status": "skipped",
                                    "reason": "no_zarr_store",
                                },
                            }
                        )
                        _emit_analysis_progress(
                            operation_end,
                            "MIP export skipped (no Zarr/N5 store).",
                        )
                    continue
        except WorkflowExecutionCancelled as exc:
            run_status = "cancelled"
            cancellation_exc = exc
            logger.info(
                "Analysis execution cancelled by user during '%s' (store=%s).",
                current_operation_name,
                provenance_store_path,
            )
            if (
                current_operation_name
                and (
                    not step_records
                    or str(step_records[-1].get("name")) != current_operation_name
                )
            ):
                step_records.append(
                    {
                        "name": current_operation_name,
                        "parameters": {
                            **current_operation_parameters,
                            "status": "cancelled",
                            "requested_input": current_requested_source,
                            "resolved_input": current_resolved_source,
                        },
                    }
                )
        except Exception as exc:
            run_status = "failed"
            failure_exc = exc
            logger.exception(
                "Analysis operation '%s' failed (store=%s, requested_input=%s, "
                "resolved_input=%s, parameters=%s).",
                current_operation_name,
                provenance_store_path,
                current_requested_source,
                current_resolved_source,
                current_operation_parameters,
            )
            if (
                current_operation_name
                and (
                    not step_records
                    or str(step_records[-1].get("name")) != current_operation_name
                    or str(step_records[-1].get("parameters", {}).get("status", ""))
                    not in {"failed", "cancelled"}
                )
            ):
                reason = (
                    "missing_input_dependency"
                    if isinstance(exc, AnalysisDependencyError)
                    else "runtime_error"
                )
                step_records.append(
                    {
                        "name": current_operation_name,
                        "parameters": {
                            **current_operation_parameters,
                            "status": "failed",
                            "reason": reason,
                            "error": str(exc),
                            "requested_input": current_requested_source,
                            "resolved_input": current_resolved_source,
                        },
                    }
                )

    if provenance_store_path and is_zarr_store_path(provenance_store_path):
        provenance_workflow = WorkflowConfig(
            file=input_path,
            prefer_dask=workflow.prefer_dask,
            dask_backend=workflow.dask_backend,
            chunks=workflow.chunks,
            flatfield=workflow.flatfield,
            deconvolution=workflow.deconvolution,
            shear_transform=workflow.shear_transform,
            particle_detection=workflow.particle_detection,
            usegment3d=workflow.usegment3d,
            registration=workflow.registration,
            display_pyramid=workflow.display_pyramid,
            visualization=workflow.visualization,
            mip_export=workflow.mip_export,
            zarr_save=workflow.zarr_save,
            spatial_calibration=runtime_spatial_calibration,
            spatial_calibration_explicit=workflow.spatial_calibration_explicit,
            analysis_parameters=runtime_analysis_parameters,
        )
        try:
            run_id = persist_run_provenance(
                zarr_path=provenance_store_path,
                workflow=provenance_workflow,
                image_info=image_info,
                steps=step_records or None,
                outputs=output_records or None,
                status=run_status,
                started_at_utc=run_started_at,
                ended_at_utc=datetime.now(tz=timezone.utc),
                repo_root=Path(__file__).resolve().parents[2],
            )
            logger.info(
                f"Persisted provenance run to store {provenance_store_path} "
                f"with run_id={run_id}."
            )

            for analysis_name, metadata in output_records.items():
                try:
                    root = zarr.open_group(str(provenance_store_path), mode="a")
                    component = str(metadata.get("component", "")).strip()
                    if component:
                        root[component].attrs["run_id"] = run_id
                except Exception:
                    pass
                register_latest_output_reference(
                    zarr_path=provenance_store_path,
                    analysis_name=analysis_name,
                    component=str(metadata["component"]),
                    run_id=run_id,
                    metadata=metadata,
                )
        except Exception as exc:
            logger.warning(
                f"Failed to persist provenance in Zarr store "
                f"{provenance_store_path}: {exc}"
            )

    if cancellation_exc is not None:
        raise cancellation_exc

    if failure_exc is not None:
        raise failure_exc

    _emit_analysis_progress(100, "Workflow execution complete.")


def main() -> None:
    """Run the ClearEx entrypoint.

    The entrypoint defaults to GUI mode. If GUI cannot be launched (for example
    in headless environments), execution falls back to non-interactive mode.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Process behavior is controlled through side effects and exit status.

    Raises
    ------
    SystemExit
        Raised by :mod:`argparse` on invalid arguments.
    """
    display_logo()

    # Parse command line arguments.
    parser = create_parser()
    args = parser.parse_args()
    bootstrap_logger = _create_bootstrap_logger()

    try:
        workflow = _build_workflow_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    if args.gui and not args.headless:
        try:
            from clearex.gui import GuiUnavailableError, launch_gui

            def _run_from_gui(
                selected_workflow: WorkflowConfig,
                progress_callback: Callable[[int, str], None],
            ) -> None:
                """Execute one GUI-selected workflow with per-run logger setup.

                Parameters
                ----------
                selected_workflow : WorkflowConfig
                    Workflow selected in GUI.
                progress_callback : callable
                    Progress callback used by GUI progress dialog.

                Returns
                -------
                None
                    Side-effect execution only.
                """
                try:
                    run_log_directory = _resolve_log_directory_for_workflow(
                        selected_workflow
                    )
                except Exception as exc:
                    run_log_directory = Path.cwd().resolve()
                    bootstrap_logger.warning(
                        "Failed to resolve workflow log directory "
                        f"({type(exc).__name__}: {exc}); using {run_log_directory}."
                    )

                run_logger = initiate_logger(run_log_directory)
                run_logger.info("Starting ClearEx")
                run_logger.info(f"Command line arguments: {args}")
                run_logger.info(f"Log directory: {run_log_directory}")
                _run_workflow(
                    workflow=selected_workflow,
                    logger=run_logger,
                    analysis_progress_callback=progress_callback,
                )

            _ = launch_gui(initial=workflow, run_callback=_run_from_gui)
            bootstrap_logger.info("GUI session closed by user.")
            return
        except Exception as exc:
            try:
                from clearex.gui import GuiUnavailableError

                if isinstance(exc, GuiUnavailableError):
                    bootstrap_logger.warning(
                        f"GUI unavailable: {exc}. Falling back to headless mode."
                    )
                else:
                    bootstrap_logger.warning(
                        f"GUI launch failed ({type(exc).__name__}: {exc}). "
                        "Falling back to headless mode."
                    )
            except Exception:
                bootstrap_logger.warning(
                    f"GUI launch failed ({type(exc).__name__}: {exc}). "
                    "Falling back to headless mode."
                )
            args.gui = False

    workflow = _apply_gui_if_requested(
        workflow=workflow,
        args=args,
        logger=bootstrap_logger,
    )
    if workflow is None:
        bootstrap_logger.info("No workflow selected. Exiting.")
        return

    try:
        log_directory = _resolve_log_directory_for_workflow(workflow)
    except Exception as exc:
        log_directory = Path.cwd().resolve()
        bootstrap_logger.warning(
            "Failed to resolve workflow log directory "
            f"({type(exc).__name__}: {exc}); using {log_directory}."
        )

    logger = initiate_logger(log_directory)
    logger.info("Starting ClearEx")
    logger.info(f"Command line arguments: {args}")
    logger.info(f"Log directory: {log_directory}")

    _run_workflow(workflow=workflow, logger=logger)


if __name__ == "__main__":
    main()
