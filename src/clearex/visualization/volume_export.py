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
from clearex.visualization.pipeline import run_display_pyramid_analysis
from clearex.workflow import (
    normalize_analysis_operation_parameters,
    resolve_analysis_input_component,
)

ProgressCallback = Callable[[int, str], None]

_AXES_TPCZYX = ("t", "p", "c", "z", "y", "x")
_AXES_TCZYX = ("t", "c", "z", "y", "x")
_ANALYSIS_NAME = "volume_export"
_DISPLAY_PYRAMID_LEVELS_ATTR = "display_pyramid_levels"
_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "display_pyramid_levels_by_component"
_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR = "visualization_pyramid_levels"
_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "visualization_pyramid_levels_by_component"


def _display_pyramid_level_component(*, source_component: str, level: int) -> str:
    """Return the canonical source-adjacent component path for one level."""
    level_index = int(level)
    if level_index < 1:
        raise ValueError("volume_export resolution levels must be >= 1.")
    component = str(source_component).strip() or "data"
    if component == "data":
        return f"data_pyramid/level_{level_index}"
    return f"{component}_pyramid/level_{level_index}"


def _discover_source_adjacent_resolution_components(
    *,
    root: zarr.Group,
    source_component: str,
) -> tuple[str, ...]:
    """Return known 6D source-adjacent resolution components for one source."""
    try:
        source_node = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"volume_export source component '{source_component}' was not found in the store."
        ) from exc
    assert isinstance(source_node, zarr.Array)

    if len(tuple(source_node.shape)) != 6:
        raise ValueError(
            f"volume_export expected canonical 6D input at '{source_component}', "
            f"but found shape {tuple(source_node.shape)}."
        )

    candidates: list[str] = [str(source_component)]
    source_attrs = dict(source_node.attrs)
    for key in (
        _DISPLAY_PYRAMID_LEVELS_ATTR,
        "pyramid_levels",
        _LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR,
    ):
        values = source_attrs.get(key)
        if isinstance(values, (tuple, list)):
            candidates.extend(str(item) for item in values)

    root_attrs = dict(root.attrs)
    for key in (_DISPLAY_PYRAMID_ROOT_MAP_ATTR, _LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR):
        root_level_map = root_attrs.get(key)
        if not isinstance(root_level_map, Mapping):
            continue
        mapped = root_level_map.get(str(source_component))
        if isinstance(mapped, (tuple, list)):
            candidates.extend(str(item) for item in mapped)

    if str(source_component).endswith("/data") or str(source_component) == "data":
        root_levels = root_attrs.get("data_pyramid_levels")
        if isinstance(root_levels, (tuple, list)):
            candidates.extend(str(item) for item in root_levels)

    ordered: list[str] = []
    for candidate in candidates:
        component = str(candidate).strip()
        if not component or component in ordered:
            continue
        try:
            node = root[component]
        except Exception:
            continue
        assert isinstance(node, zarr.Array)
        if len(tuple(node.shape)) != 6:
            continue
        ordered.append(component)
    return tuple(ordered) if ordered else (str(source_component),)


def _resolve_volume_export_resolution_component(
    *,
    zarr_path: str | Path,
    root: zarr.Group,
    source_component: str,
    resolution_level: int,
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> tuple[str, bool]:
    """Resolve or generate one requested source-adjacent resolution component."""
    level_index = int(resolution_level)
    if level_index <= 0:
        return str(source_component), False

    requested_component = _display_pyramid_level_component(
        source_component=source_component,
        level=level_index,
    )
    discovered_components = _discover_source_adjacent_resolution_components(
        root=root,
        source_component=source_component,
    )
    if requested_component in discovered_components:
        return requested_component, False

    _ = run_display_pyramid_analysis(
        zarr_path=zarr_path,
        parameters={"input_source": source_component},
        progress_callback=progress_callback,
        run_id=run_id,
    )

    refreshed_components = _discover_source_adjacent_resolution_components(
        root=zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="a"),
        source_component=source_component,
    )
    if requested_component not in refreshed_components:
        raise ValueError(
            f"volume_export could not resolve resolution_level={level_index} "
            f"for source component '{source_component}'."
        )
    return requested_component, True


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
    """Export the resolved source component into the runtime-cache volume."""
    _emit(progress_callback, 0, "Normalizing volume export parameters")
    normalized = dict(
        normalize_analysis_operation_parameters({"volume_export": dict(parameters)})[
            "volume_export"
        ]
    )

    export_scope = str(normalized.get("export_scope", "current_selection")).strip()
    if export_scope not in {"current_selection", "all_indices"}:
        raise ValueError(
            "volume_export currently supports export_scope=current_selection or all_indices."
        )

    resolution_level = int(normalized.get("resolution_level", 0))
    if resolution_level < 0:
        raise ValueError("volume_export resolution_level must be >= 0.")

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

    resolved_resolution_component, generated_resolution_level = (
        _resolve_volume_export_resolution_component(
            zarr_path=zarr_path,
            root=root,
            source_component=source_component,
            resolution_level=resolution_level,
            progress_callback=progress_callback,
            run_id=run_id,
        )
    )
    if generated_resolution_level:
        root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="a")

    try:
        resolved_node = get_node(root, resolved_resolution_component)
    except Exception as exc:
        raise ValueError(
            f"volume_export resolved component '{resolved_resolution_component}' was not found in the store."
        ) from exc
    if not isinstance(resolved_node, zarr.Array):
        raise ValueError(
            f"volume_export expected a 6D array at '{resolved_resolution_component}', "
            f"but found {type(resolved_node).__name__}."
        )

    resolved_shape = tuple(int(value) for value in resolved_node.shape)
    if len(resolved_shape) != 6:
        raise ValueError(
            f"volume_export expected canonical 6D input at '{resolved_resolution_component}', "
            f"but found shape {resolved_shape}."
        )

    t_index = int(normalized.get("t_index", 0))
    p_index = int(normalized.get("p_index", 0))
    c_index = int(normalized.get("c_index", 0))
    if t_index < 0 or t_index >= resolved_shape[0]:
        raise ValueError(
            f"volume_export t_index={t_index} is out of bounds for shape {resolved_shape}."
        )
    if p_index < 0 or p_index >= resolved_shape[1]:
        raise ValueError(
            f"volume_export p_index={p_index} is out of bounds for shape {resolved_shape}."
        )
    if c_index < 0 or c_index >= resolved_shape[2]:
        raise ValueError(
            f"volume_export c_index={c_index} is out of bounds for shape {resolved_shape}."
        )

    delete_path(root, analysis_cache_root(_ANALYSIS_NAME))
    data_component = analysis_cache_data_component(_ANALYSIS_NAME)
    parent_path, _, leaf = data_component.rpartition("/")
    parent_group = ensure_group(root, parent_path) if parent_path else root
    if export_scope == "all_indices":
        export_array = da.from_zarr(resolved_node)
        export_shape = resolved_shape
        export_chunks = tuple(
            int(value) for value in (resolved_node.chunks or resolved_shape)
        )
        _emit(progress_callback, 40, "Writing resolved volume to cache")
    else:
        export_array = da.from_zarr(resolved_node)[
            t_index : t_index + 1,
            p_index : p_index + 1,
            c_index : c_index + 1,
            :,
            :,
            :,
        ]
        export_shape = (
            1,
            1,
            1,
            resolved_shape[3],
            resolved_shape[4],
            resolved_shape[5],
        )
        export_chunks = (
            1,
            1,
            1,
            resolved_shape[3],
            resolved_shape[4],
            resolved_shape[5],
        )
        _emit(progress_callback, 40, "Writing current-selection volume to cache")
    target = parent_group.create_array(
        leaf,
        shape=export_shape,
        chunks=export_chunks,
        dtype=resolved_node.dtype,
        overwrite=True,
        dimension_names=_AXES_TPCZYX,
    )

    da.store(export_array, target, lock=False, compute=True)

    voxel_size_um_zyx, voxel_size_source = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component=resolved_resolution_component,
    )
    auxiliary_root = analysis_auxiliary_root(_ANALYSIS_NAME)
    delete_path(root, auxiliary_root)
    auxiliary_group = ensure_group(root, auxiliary_root)

    selection_payload = {
        "t_index": t_index,
        "p_index": p_index,
        "c_index": c_index,
        "resolution_level": resolution_level,
    }
    export_shape_tpczyx = [int(value) for value in tuple(export_array.shape)]
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
        "selected_shape_tpczyx": export_shape_tpczyx,
        "source_shape_tpczyx": list(source_shape),
        "resolved_shape_tpczyx": list(resolved_shape),
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
                "selected_shape_tpczyx": export_shape_tpczyx,
                "source_shape_tpczyx": list(source_shape),
                "resolved_shape_tpczyx": list(resolved_shape),
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
