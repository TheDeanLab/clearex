#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

"""Metadata-only repair helpers for relocated multiposition stores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zarr

from clearex.io.experiment import load_navigate_experiment
from clearex.io.ome_store import (
    CLEAREX_RUNTIME_RESULTS_ROOT,
    SOURCE_CACHE_COMPONENT,
    compute_position_translations_zyx_um,
    get_node,
    load_store_metadata,
    update_store_metadata,
)
from clearex.io.stage_metadata import (
    load_multiposition_stage_rows_from_directory,
    parse_multiposition_stage_rows,
)
from clearex.workflow import spatial_calibration_from_dict


@dataclass(frozen=True)
class MultipositionStageMetadataRepairSummary:
    """Summary of one metadata-only multiposition stage repair."""

    store_path: Path
    experiment_path: Path
    position_count: int
    stage_row_count: int
    position_translations_zyx_um: list[list[float]]


def _infer_store_position_count(
    root: zarr.Group,
    metadata: Mapping[str, Any],
    *,
    fallback: int,
) -> int:
    """Infer the canonical position count from known store components."""
    for component in (
        SOURCE_CACHE_COMPONENT,
        f"{CLEAREX_RUNTIME_RESULTS_ROOT}/shear_transform/latest/data",
        f"{CLEAREX_RUNTIME_RESULTS_ROOT}/flatfield/latest/data",
        f"{CLEAREX_RUNTIME_RESULTS_ROOT}/registration/latest/data",
        "data",
    ):
        try:
            node = get_node(root, component)
        except Exception:
            continue
        shape = getattr(node, "shape", None)
        if isinstance(shape, tuple) and len(shape) == 6:
            return max(1, int(shape[1]))
        if isinstance(shape, list) and len(shape) == 6:
            return max(1, int(shape[1]))

    shapes = metadata.get("data_pyramid_shapes_tpczyx")
    if isinstance(shapes, list) and shapes:
        first_shape = shapes[0]
        if isinstance(first_shape, (list, tuple)) and len(first_shape) == 6:
            return max(1, int(first_shape[1]))
    return max(1, int(fallback))


def repair_multiposition_stage_metadata(
    store_path: str | Path,
    experiment_path: str | Path,
) -> MultipositionStageMetadataRepairSummary:
    """Repair relocated-store multiposition stage metadata from an experiment.

    Parameters
    ----------
    store_path : str or pathlib.Path
        Existing canonical ClearEx OME-Zarr store to patch.
    experiment_path : str or pathlib.Path
        Relocated Navigate ``experiment.yml`` or ``experiment.yaml`` path.

    Returns
    -------
    MultipositionStageMetadataRepairSummary
        Summary of the metadata values written.

    Raises
    ------
    ValueError
        If the experiment path is invalid or does not provide enough stage rows.
    """
    resolved_store_path = Path(store_path).expanduser().resolve()
    resolved_experiment_path = Path(experiment_path).expanduser().resolve()
    if resolved_experiment_path.name.lower() not in {
        "experiment.yml",
        "experiment.yaml",
    }:
        raise ValueError("Select a Navigate experiment.yml or experiment.yaml file.")
    if not resolved_experiment_path.exists():
        raise FileNotFoundError(resolved_experiment_path)

    experiment = load_navigate_experiment(resolved_experiment_path)
    stage_rows = load_multiposition_stage_rows_from_directory(
        resolved_experiment_path.parent
    )
    if not stage_rows:
        stage_rows = load_multiposition_stage_rows_from_directory(
            experiment.save_directory
        )
    if not stage_rows:
        stage_rows = parse_multiposition_stage_rows(
            experiment.raw.get("MultiPositions")
        )

    root = zarr.open_group(str(resolved_store_path), mode="a")
    metadata = load_store_metadata(root)
    position_count = _infer_store_position_count(
        root,
        metadata,
        fallback=int(experiment.multiposition_count),
    )
    if len(stage_rows) < position_count:
        message = (
            "Selected experiment metadata does not contain enough multiposition "
            + f"stage rows for this store: found {len(stage_rows)}, "
            + f"required {position_count}."
        )
        raise ValueError(message)

    spatial_calibration = spatial_calibration_from_dict(
        metadata.get("spatial_calibration")
    )
    translations = compute_position_translations_zyx_um(
        stage_rows,
        spatial_calibration,
        position_count=position_count,
    )
    navigate_payload = dict(metadata.get("navigate_experiment") or {})
    navigate_payload.update(
        {
            "experiment_path": str(resolved_experiment_path),
            "save_directory": str(experiment.save_directory),
            "file_type": experiment.file_type,
            "microscope_name": experiment.microscope_name,
            "image_mode": experiment.image_mode,
            "timepoints": int(experiment.timepoints),
            "number_z_steps": int(experiment.number_z_steps),
            "multiposition_count": int(experiment.multiposition_count),
            "channel_count": len(experiment.selected_channels),
            "xy_pixel_size_um": experiment.xy_pixel_size_um,
            "z_step_um": experiment.z_step_um,
            "selected_channels": [
                {"name": channel.name, "laser": channel.laser}
                for channel in experiment.selected_channels
            ],
        }
    )
    _ = update_store_metadata(
        root,
        source_experiment=str(resolved_experiment_path),
        navigate_experiment=navigate_payload,
        stage_rows=stage_rows,
        position_translations_zyx_um=translations,
    )
    return MultipositionStageMetadataRepairSummary(
        store_path=resolved_store_path,
        experiment_path=resolved_experiment_path,
        position_count=position_count,
        stage_row_count=len(stage_rows),
        position_translations_zyx_um=translations,
    )
