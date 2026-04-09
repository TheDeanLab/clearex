from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import json

import dask.array as da
import zarr

from clearex.io.ome_store import (
    analysis_auxiliary_root,
    analysis_cache_data_component,
    analysis_cache_root,
    delete_path,
    ensure_group,
    get_node,
    resolve_voxel_size_um_zyx_with_source,
)
from clearex.io.provenance import register_latest_output_reference
from clearex.workflow import (
    normalize_analysis_operation_parameters,
    resolve_analysis_input_component,
)

ProgressCallback = Callable[[int, str], None]

_AXES_TPCZYX = ("t", "p", "c", "z", "y", "x")
_AXES_TCZYX = ("t", "c", "z", "y", "x")
_ANALYSIS_NAME = "volume_export"


@dataclass(frozen=True)
class VolumeExportSummary:
    """Summary metadata for one volume-export run."""

    component: str
    data_component: str
    source_component: str
    resolved_resolution_component: str
    export_scope: str
    resolution_level: int
    generated_resolution_level: bool
    export_format: str
    tiff_file_layout: str
    artifact_paths: tuple[str, ...]


def _jsonable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of one mapping."""
    return json.loads(json.dumps(dict(payload)))


def _emit(
    progress_callback: Optional[ProgressCallback], percent: int, message: str
) -> None:
    if progress_callback is None:
        return
    progress_callback(int(percent), str(message))


def _source_component_for_input(input_source: str) -> str:
    """Resolve the requested source against the current store layout."""
    requested = str(input_source).strip() or "data"
    return resolve_analysis_input_component(requested)


def run_volume_export_analysis(
    *,
    zarr_path: str | Path,
    parameters: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> VolumeExportSummary:
    """Export the current selection as a singleton runtime-cache volume."""
    _emit(progress_callback, 0, "Normalizing volume export parameters")
    normalized = dict(
        normalize_analysis_operation_parameters({"volume_export": dict(parameters)})[
            "volume_export"
        ]
    )

    export_scope = str(normalized.get("export_scope", "current_selection")).strip()
    if export_scope != "current_selection":
        raise ValueError(
            "volume_export currently supports only export_scope=current_selection."
        )

    resolution_level = int(normalized.get("resolution_level", 0))
    if resolution_level != 0:
        raise ValueError("volume_export currently supports only resolution_level=0.")

    export_format = str(normalized.get("export_format", "ome-zarr")).strip()
    if export_format != "ome-zarr":
        raise ValueError(
            "volume_export currently supports only export_format=ome-zarr."
        )

    root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="a")
    source_component = _source_component_for_input(
        str(normalized.get("input_source", "data"))
    )
    try:
        source_node = get_node(root, source_component)
    except Exception as exc:
        raise ValueError(
            f"volume_export source component '{source_component}' was not found in the store."
        ) from exc
    if not isinstance(source_node, zarr.Array):
        raise ValueError(
            f"volume_export expected a 6D array at '{source_component}', "
            f"but found {type(source_node).__name__}."
        )

    source_shape = tuple(int(value) for value in source_node.shape)
    if len(source_shape) != 6:
        raise ValueError(
            f"volume_export expected canonical 6D input at '{source_component}', "
            f"but found shape {source_shape}."
        )

    t_index = int(normalized.get("t_index", 0))
    p_index = int(normalized.get("p_index", 0))
    c_index = int(normalized.get("c_index", 0))
    if t_index < 0 or t_index >= source_shape[0]:
        raise ValueError(
            f"volume_export t_index={t_index} is out of bounds for shape {source_shape}."
        )
    if p_index < 0 or p_index >= source_shape[1]:
        raise ValueError(
            f"volume_export p_index={p_index} is out of bounds for shape {source_shape}."
        )
    if c_index < 0 or c_index >= source_shape[2]:
        raise ValueError(
            f"volume_export c_index={c_index} is out of bounds for shape {source_shape}."
        )

    delete_path(root, analysis_cache_root(_ANALYSIS_NAME))
    data_component = analysis_cache_data_component(_ANALYSIS_NAME)
    parent_path, _, leaf = data_component.rpartition("/")
    parent_group = ensure_group(root, parent_path) if parent_path else root
    target = parent_group.create_array(
        leaf,
        shape=(1, 1, 1, source_shape[3], source_shape[4], source_shape[5]),
        chunks=(1, 1, 1, source_shape[3], source_shape[4], source_shape[5]),
        dtype=source_node.dtype,
        overwrite=True,
        dimension_names=_AXES_TPCZYX,
    )

    _emit(progress_callback, 40, "Writing current-selection volume to cache")
    selected = da.from_zarr(source_node)[
        t_index : t_index + 1,
        p_index : p_index + 1,
        c_index : c_index + 1,
        :,
        :,
        :,
    ]
    da.store(selected, target, lock=False, compute=True)

    voxel_size_um_zyx, voxel_size_source = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component=source_component,
    )
    resolved_resolution_component = source_component
    generated_resolution_level = False
    auxiliary_root = analysis_auxiliary_root(_ANALYSIS_NAME)
    delete_path(root, auxiliary_root)
    auxiliary_group = ensure_group(root, auxiliary_root)

    selection_payload = {
        "t_index": t_index,
        "p_index": p_index,
        "c_index": c_index,
        "resolution_level": resolution_level,
    }
    metadata = {
        "analysis_name": _ANALYSIS_NAME,
        "component": auxiliary_root,
        "data_component": data_component,
        "source_component": source_component,
        "resolved_resolution_component": resolved_resolution_component,
        "export_scope": export_scope,
        "resolution_level": resolution_level,
        "generated_resolution_level": generated_resolution_level,
        "export_format": export_format,
        "tiff_file_layout": str(normalized.get("tiff_file_layout", "single_file")),
        "input_source": str(normalized.get("input_source", "data")).strip() or "data",
        "selection": selection_payload,
        "selected_shape_tpczyx": [
            1,
            1,
            1,
            source_shape[3],
            source_shape[4],
            source_shape[5],
        ],
        "source_shape_tpczyx": list(source_shape),
        "source_dtype": str(source_node.dtype),
        "axes_tpczyx": list(_AXES_TPCZYX),
        "axes_tczyx": list(_AXES_TCZYX),
        "dimension_names_tpczyx": list(_AXES_TPCZYX),
        "voxel_size_um_zyx": [float(value) for value in voxel_size_um_zyx],
        "voxel_size_resolution_source": str(voxel_size_source),
        "artifact_paths": [data_component, auxiliary_root],
    }
    target.attrs.update(
        _jsonable(
            {
                "component": data_component,
                "analysis_name": _ANALYSIS_NAME,
                "source_component": source_component,
                "resolved_resolution_component": resolved_resolution_component,
                "export_scope": export_scope,
                "resolution_level": resolution_level,
                "generated_resolution_level": generated_resolution_level,
                "export_format": export_format,
                "tiff_file_layout": str(
                    normalized.get("tiff_file_layout", "single_file")
                ),
                "input_source": str(normalized.get("input_source", "data")).strip()
                or "data",
                "selection": selection_payload,
                "selected_shape_tpczyx": [
                    1,
                    1,
                    1,
                    source_shape[3],
                    source_shape[4],
                    source_shape[5],
                ],
                "source_shape_tpczyx": list(source_shape),
                "source_dtype": str(source_node.dtype),
                "axes_tpczyx": list(_AXES_TPCZYX),
                "axes_tczyx": list(_AXES_TCZYX),
                "dimension_names_tpczyx": list(_AXES_TPCZYX),
                "voxel_size_um_zyx": [float(value) for value in voxel_size_um_zyx],
                "voxel_size_resolution_source": str(voxel_size_source),
            }
        )
    )
    auxiliary_group.attrs.update(_jsonable(metadata))

    register_latest_output_reference(
        zarr_path,
        _ANALYSIS_NAME,
        auxiliary_root,
        run_id=run_id,
        metadata=metadata,
    )

    _emit(progress_callback, 100, "Volume export complete")
    return VolumeExportSummary(
        component=auxiliary_root,
        data_component=data_component,
        source_component=source_component,
        resolved_resolution_component=resolved_resolution_component,
        export_scope=export_scope,
        resolution_level=resolution_level,
        generated_resolution_level=generated_resolution_level,
        export_format=export_format,
        tiff_file_layout=str(normalized.get("tiff_file_layout", "single_file")),
        artifact_paths=(data_component, auxiliary_root),
    )
