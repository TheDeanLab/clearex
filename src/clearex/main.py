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
from typing import Any, Dict, Optional
import argparse
import logging
import os


# Third Party Imports

# Local Imports
from clearex.io.read import ImageInfo, ImageOpener
from clearex.io.experiment import (
    create_dask_client,
    is_navigate_experiment_file,
    load_navigate_experiment,
    materialize_experiment_data_store,
    resolve_experiment_data_path,
)
from clearex.io.cli import create_parser, display_logo
from clearex.io.log import initiate_logger
from clearex.io.provenance import is_zarr_store_path, persist_run_provenance
from clearex.workflow import (
    DASK_BACKEND_LOCAL_CLUSTER,
    DASK_BACKEND_SLURM_CLUSTER,
    DASK_BACKEND_SLURM_RUNNER,
    WorkflowConfig,
    dask_backend_to_dict,
    format_dask_backend_summary,
    format_chunks,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
    parse_chunks,
)


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
        If the ``--chunks`` option cannot be parsed as valid positive integers.
    """
    return WorkflowConfig(
        file=args.file,
        prefer_dask=args.dask,
        chunks=parse_chunks(args.chunks),
        deconvolution=args.deconvolution,
        particle_detection=args.particle_detection,
        registration=args.registration,
        visualization=args.visualization,
    )


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
    logger.info(f"Pixel size (um): {getattr(info, 'pixel_size', None) or 'n/a'}")

    if info.metadata:
        metadata_types = {key: type(value).__name__ for key, value in info.metadata.items()}
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
                logger.warning(f"GUI unavailable: {exc}. Falling back to headless mode.")
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
    """
    if not workflow.prefer_dask:
        logger.info("Dask lazy loading disabled; skipping backend startup.")
        return None

    backend = workflow.dask_backend
    logger.info(f"Dask backend selection: {format_dask_backend_summary(backend)}")

    try:
        if backend.mode == DASK_BACKEND_LOCAL_CLUSTER:
            client = create_dask_client(
                n_workers=backend.local_cluster.n_workers,
                threads_per_worker=backend.local_cluster.threads_per_worker,
                processes=False,
                memory_limit=backend.local_cluster.memory_limit,
                local_directory=backend.local_cluster.local_directory,
            )
            exit_stack.callback(client.close)
            logger.info("Connected to LocalCluster backend.")
            return client

        if backend.mode == DASK_BACKEND_SLURM_RUNNER:
            scheduler_file = backend.slurm_runner.scheduler_file
            if not scheduler_file:
                raise ValueError(
                    "SLURMRunner backend requires a scheduler file path."
                )

            from dask.distributed import Client
            from dask_jobqueue.slurm import SLURMRunner

            runner = exit_stack.enter_context(SLURMRunner(scheduler_file=scheduler_file))
            client = exit_stack.enter_context(Client(runner))

            wait_for_workers = backend.slurm_runner.wait_for_workers
            if wait_for_workers is None:
                runner_workers = getattr(runner, "n_workers", None)
                if isinstance(runner_workers, int) and runner_workers > 0:
                    wait_for_workers = runner_workers
            if wait_for_workers is not None:
                client.wait_for_workers(wait_for_workers)

            logger.info(
                "Connected to SLURMRunner backend "
                f"(scheduler_file={scheduler_file})."
            )
            return client

        if backend.mode == DASK_BACKEND_SLURM_CLUSTER:
            from dask.distributed import Client
            from dask_jobqueue import SLURMCluster

            cluster_cfg = backend.slurm_cluster
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


def _run_workflow(workflow: WorkflowConfig, logger: logging.Logger) -> None:
    """Execute a configured workflow in headless mode.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow parameters including file path and selected analyses.
    logger : logging.Logger
        Logger used for progress and status messages.

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
    # TODO: Persist workflow configuration for provenance/replay support.
    run_started_at = datetime.now(tz=timezone.utc)
    image_info: Optional[ImageInfo] = None
    step_records: list[Dict[str, object]] = []
    input_path = workflow.file
    provenance_store_path: Optional[str] = None

    with ExitStack() as exit_stack:
        dask_client = _configure_dask_backend(
            workflow=workflow,
            logger=logger,
            exit_stack=exit_stack,
        )

        if workflow.file:
            if is_navigate_experiment_file(workflow.file):
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
                            "zarr_chunks_ptczyx": list(workflow.zarr_save.chunks_ptczyx),
                            "zarr_pyramid_ptczyx": [
                                list(levels) for levels in workflow.zarr_save.pyramid_ptczyx
                            ],
                        },
                    }
                )

                materialized = materialize_experiment_data_store(
                    experiment=experiment,
                    source_path=resolved_data_path,
                    chunks=zarr_chunks_tpczyx,
                    pyramid_factors=zarr_pyramid_tpczyx,
                    client=dask_client,
                )
                image_info = materialized.source_image_info
                provenance_store_path = str(materialized.store_path)
                _log_loaded_image(image_info, logger)
                logger.info(
                    "Materialized source data to Zarr store "
                    f"{materialized.store_path} (component=data, "
                    f"chunks_tpczyx={materialized.chunks_tpczyx})."
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
                            "zarr_chunks_ptczyx": list(workflow.zarr_save.chunks_ptczyx),
                            "zarr_pyramid_ptczyx": [
                                list(levels) for levels in workflow.zarr_save.pyramid_ptczyx
                            ],
                        },
                    }
                )
            else:
                experiment = None

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

            step_records.append(
                {
                    "name": "load_data",
                    "parameters": {
                        "source_path": input_path,
                        "prefer_dask": workflow.prefer_dask,
                        "chunks": format_chunks(workflow.chunks) or None,
                        "dask_backend": dask_backend_to_dict(workflow.dask_backend),
                    },
                }
            )

        if workflow.deconvolution:
            logger.info(
                "Deconvolution selected. Workflow hook is reserved; implementation pending."
            )
            step_records.append({"name": "deconvolution", "parameters": {}})

        if workflow.particle_detection:
            logger.info(
                "Particle detection selected. Workflow hook is reserved; implementation pending."
            )
            step_records.append({"name": "particle_detection", "parameters": {}})

        if workflow.registration:
            logger.info("Running registration workflow.")
            from clearex.registration import ImageRegistration

            ImageRegistration()
            step_records.append({"name": "registration", "parameters": {}})

        if workflow.visualization:
            logger.info("Launching visualization workflow.")
            print("Launching visualization")
            step_records.append({"name": "visualization", "parameters": {}})

        if provenance_store_path and is_zarr_store_path(provenance_store_path):
            provenance_workflow = WorkflowConfig(
                file=input_path,
                prefer_dask=workflow.prefer_dask,
                dask_backend=workflow.dask_backend,
                chunks=workflow.chunks,
                deconvolution=workflow.deconvolution,
                particle_detection=workflow.particle_detection,
                registration=workflow.registration,
                visualization=workflow.visualization,
                zarr_save=workflow.zarr_save,
            )
            try:
                run_id = persist_run_provenance(
                    zarr_path=provenance_store_path,
                    workflow=provenance_workflow,
                    image_info=image_info,
                    steps=step_records or None,
                    started_at_utc=run_started_at,
                    ended_at_utc=datetime.now(tz=timezone.utc),
                    repo_root=Path(__file__).resolve().parents[2],
                )
                logger.info(
                    f"Persisted provenance run to store {provenance_store_path} "
                    f"with run_id={run_id}."
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to persist provenance in Zarr store "
                    f"{provenance_store_path}: {exc}"
                )


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

    # Initialize Logging
    logger = initiate_logger(os.getcwd())
    logger.info("Starting ClearEx")

    # Parse command line arguments.
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"Command line arguments: {args}")

    try:
        workflow = _build_workflow_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    workflow = _apply_gui_if_requested(workflow=workflow, args=args, logger=logger)
    if workflow is None:
        logger.info("No workflow selected. Exiting.")
        return

    _run_workflow(workflow=workflow, logger=logger)


if __name__ == "__main__":
    main()
