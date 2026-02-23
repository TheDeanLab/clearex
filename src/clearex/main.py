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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import argparse
import logging
import os


# Third Party Imports

# Local Imports
from clearex.io.read import ImageInfo, ImageOpener
from clearex.io.experiment import (
    default_analysis_store_path,
    initialize_analysis_store,
    is_navigate_experiment_file,
    load_navigate_experiment,
    resolve_experiment_data_path,
)
from clearex.io.cli import create_parser, display_logo
from clearex.io.log import initiate_logger
from clearex.io.provenance import is_zarr_store_path, persist_run_provenance
from clearex.workflow import WorkflowConfig, format_chunks, parse_chunks


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

    if workflow.file:
        if is_navigate_experiment_file(workflow.file):
            experiment = load_navigate_experiment(workflow.file)
            resolved_data_path = resolve_experiment_data_path(experiment)
            input_path = str(resolved_data_path)

            logger.info(
                f"Loaded experiment metadata from {workflow.file}: "
                f"file_type={experiment.file_type}, "
                f"timepoints={experiment.timepoints}, "
                f"positions={experiment.multiposition_count}, "
                f"channels={experiment.channel_count}, "
                f"z_steps={experiment.number_z_steps}."
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

        if experiment is not None:
            analysis_store = default_analysis_store_path(experiment)
            initialize_analysis_store(
                experiment=experiment,
                zarr_path=analysis_store,
                image_info=info,
                overwrite=False,
                dtype=str(info.dtype),
            )
            provenance_store_path = str(analysis_store)
            logger.info(
                f"Initialized/validated 6D analysis store at {analysis_store} "
                "(axes=t,p,c,z,y,x)."
            )
        elif input_path and is_zarr_store_path(input_path):
            provenance_store_path = input_path

        step_records.append(
            {
                "name": "load_data",
                "parameters": {
                    "source_path": input_path,
                    "prefer_dask": workflow.prefer_dask,
                    "chunks": format_chunks(workflow.chunks) or None,
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
            chunks=workflow.chunks,
            deconvolution=workflow.deconvolution,
            particle_detection=workflow.particle_detection,
            registration=workflow.registration,
            visualization=workflow.visualization,
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
