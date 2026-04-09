from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import json
import shutil

import dask.array as da
import numpy as np
import tifffile
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
from clearex.visualization import pipeline as visualization_pipeline
from clearex.workflow import (
    normalize_analysis_operation_parameters,
    resolve_analysis_input_component,
)

ProgressCallback = Callable[[int, str], None]

_AXES_TPCZYX = ("t", "p", "c", "z", "y", "x")
_AXES_TCZYX = ("t", "c", "z", "y", "x")
_ANALYSIS_NAME = "volume_export"
_DISPLAY_PYRAMID_LEVELS_ATTR = "display_pyramid_levels"
_DISPLAY_PYRAMID_FACTORS_ATTR = "display_pyramid_factors_tpczyx"
_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "display_pyramid_levels_by_component"
_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR = "visualization_pyramid_levels"
_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR = "visualization_pyramid_factors_tpczyx"
_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR = "visualization_pyramid_levels_by_component"
_ARTIFACTS_ROOT = f"{analysis_auxiliary_root(_ANALYSIS_NAME)}/files"


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
    return visualization_pipeline._collect_existing_multiscale_components(
        root=root,
        source_component=source_component,
    )


def _extend_level_factors_tpczyx(
    level_factors_tpczyx: tuple[tuple[int, int, int, int, int, int], ...],
    required_level_index: int,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Extend absolute pyramid factors until a requested level index exists."""
    factors = [tuple(int(value) for value in row) for row in level_factors_tpczyx]
    if not factors:
        factors = [(1, 1, 1, 1, 1, 1)]
    while len(factors) <= int(required_level_index):
        previous = factors[-1]
        factors.append(
            (
                int(previous[0]),
                int(previous[1]),
                int(previous[2]),
                int(max(1, previous[3] * 2)),
                int(max(1, previous[4] * 2)),
                int(max(1, previous[5] * 2)),
            )
        )
    return tuple(factors)


def _generate_missing_volume_export_resolution_components(
    *,
    zarr_path: str | Path,
    root: zarr.Group,
    source_component: str,
    resolution_level: int,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[str, bool]:
    """Materialize missing source-adjacent levels up to ``resolution_level``."""
    source_node = root[source_component]
    if not isinstance(source_node, zarr.Array):
        raise ValueError(
            f"volume_export expected a 6D array at '{source_component}', "
            f"but found {type(source_node).__name__}."
        )

    source_attrs = dict(source_node.attrs)
    root_attrs = dict(root.attrs)
    level_factors = (
        visualization_pipeline._resolve_visualization_pyramid_factors_tpczyx(
            root_attrs=root_attrs,
            source_attrs=source_attrs,
        )
    )
    level_factors = _extend_level_factors_tpczyx(level_factors, resolution_level)

    discovered_components = list(
        _discover_source_adjacent_resolution_components(
            root=root,
            source_component=source_component,
        )
    )
    if int(resolution_level) < len(discovered_components):
        return str(discovered_components[int(resolution_level)]), False

    source_chunks = (
        tuple(source_node.chunks) if source_node.chunks is not None else None
    )
    prior_component = (
        str(discovered_components[-1])
        if discovered_components
        else str(source_component)
    )
    prior_factors = tuple(
        int(value) for value in level_factors[len(discovered_components) - 1]
    )
    level_paths = list(discovered_components)
    factor_payload = [list(row) for row in level_factors[: len(level_paths)]]

    for level_index in range(len(level_paths), int(resolution_level) + 1):
        absolute_factors = tuple(int(value) for value in level_factors[level_index])
        all_relative = all(
            int(current) % int(previous) == 0
            for current, previous in zip(absolute_factors, prior_factors, strict=False)
        )
        if all_relative:
            downsample_factors = (
                int(absolute_factors[0] // prior_factors[0]),
                int(absolute_factors[1] // prior_factors[1]),
                int(absolute_factors[2] // prior_factors[2]),
                int(absolute_factors[3] // prior_factors[3]),
                int(absolute_factors[4] // prior_factors[4]),
                int(absolute_factors[5] // prior_factors[5]),
            )
            source_level_component = str(prior_component)
        else:
            downsample_factors = absolute_factors
            source_level_component = str(source_component)

        target_component = _display_pyramid_level_component(
            source_component=source_component,
            level=int(level_index),
        )
        source_level = da.from_zarr(str(zarr_path), component=source_level_component)
        downsampled = visualization_pipeline._downsample_tpczyx_by_stride(
            source_level,
            downsample_factors,
        )
        level_shape = tuple(int(size) for size in tuple(downsampled.shape))
        level_chunks = visualization_pipeline._resolve_level_chunks_tpczyx(
            source_chunks=source_chunks,
            level_shape=level_shape,
            level_array=downsampled,
        )
        if not visualization_pipeline._component_matches_shape_chunks(
            root=root,
            component=target_component,
            shape_tpczyx=(
                int(level_shape[0]),
                int(level_shape[1]),
                int(level_shape[2]),
                int(level_shape[3]),
                int(level_shape[4]),
                int(level_shape[5]),
            ),
            chunks_tpczyx=level_chunks,
        ):
            parent_path, _, leaf = target_component.rpartition("/")
            parent_group = ensure_group(root, parent_path) if parent_path else root
            target = parent_group.create_array(
                leaf,
                shape=level_shape,
                chunks=level_chunks,
                dtype=source_node.dtype,
                overwrite=True,
                dimension_names=_AXES_TPCZYX,
            )
            da.store(downsampled, target, lock=False, compute=True)
        root[target_component].attrs.update(
            {
                "axes": list(_AXES_TPCZYX),
                "pyramid_level": int(level_index),
                "downsample_factors_tpczyx": [int(value) for value in absolute_factors],
                "chunk_shape_tpczyx": [int(value) for value in level_chunks],
                "source_component": str(source_level_component),
                "generated_by": "volume_export",
            }
        )
        level_paths.append(target_component)
        factor_payload.append([int(value) for value in absolute_factors])
        prior_component = target_component
        prior_factors = absolute_factors

    root[str(source_component)].attrs[_DISPLAY_PYRAMID_LEVELS_ATTR] = [
        str(item) for item in level_paths
    ]
    root[str(source_component)].attrs[_DISPLAY_PYRAMID_FACTORS_ATTR] = [
        list(row) for row in factor_payload
    ]
    root[str(source_component)].attrs[_LEGACY_DISPLAY_PYRAMID_LEVELS_ATTR] = [
        str(item) for item in level_paths
    ]
    root[str(source_component)].attrs[_LEGACY_DISPLAY_PYRAMID_FACTORS_ATTR] = [
        list(row) for row in factor_payload
    ]
    component_map = root.attrs.get(_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if not isinstance(component_map, dict):
        component_map = {}
    component_map[str(source_component)] = [str(item) for item in level_paths]
    root.attrs[_DISPLAY_PYRAMID_ROOT_MAP_ATTR] = component_map
    legacy_component_map = root.attrs.get(_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR)
    if not isinstance(legacy_component_map, dict):
        legacy_component_map = {}
    legacy_component_map[str(source_component)] = [str(item) for item in level_paths]
    root.attrs[_LEGACY_DISPLAY_PYRAMID_ROOT_MAP_ATTR] = legacy_component_map
    return str(level_paths[int(resolution_level)]), True


def _resolve_volume_export_voxel_size_um_zyx(
    *,
    root: zarr.Group,
    source_component: str,
    resolved_resolution_component: str,
) -> tuple[tuple[float, float, float], str]:
    """Resolve voxel size for export with a fallback to the base source."""
    voxel_size_um_zyx, voxel_size_source = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component=resolved_resolution_component,
    )
    resolved_component = (
        str(resolved_resolution_component).strip() or str(source_component).strip()
    )
    direct_resolution_sources = {
        f"component:{resolved_component}",
        f"component_navigate:{resolved_component}",
    }
    if (
        voxel_size_source in direct_resolution_sources
        or resolved_component == str(source_component).strip()
    ):
        return voxel_size_um_zyx, voxel_size_source

    fallback_voxel_size_um_zyx, fallback_source = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component=source_component,
    )
    if fallback_source == "default":
        return voxel_size_um_zyx, voxel_size_source

    downsample_factors_tpczyx = _resolve_volume_export_level_factors_tpczyx(
        root=root,
        source_component=source_component,
        resolved_resolution_component=resolved_component,
    )
    return (
        (
            float(fallback_voxel_size_um_zyx[0]) * float(downsample_factors_tpczyx[3]),
            float(fallback_voxel_size_um_zyx[1]) * float(downsample_factors_tpczyx[4]),
            float(fallback_voxel_size_um_zyx[2]) * float(downsample_factors_tpczyx[5]),
        ),
        fallback_source,
    )


def _resolve_volume_export_level_factors_tpczyx(
    *,
    root: zarr.Group,
    source_component: str,
    resolved_resolution_component: str,
) -> tuple[int, int, int, int, int, int]:
    """Resolve absolute downsample factors for one export level."""
    resolved_component = (
        str(resolved_resolution_component).strip() or str(source_component).strip()
    )
    if resolved_component == str(source_component).strip():
        return (1, 1, 1, 1, 1, 1)

    try:
        resolved_node = get_node(root, resolved_component)
    except Exception:
        resolved_node = None
    payload = getattr(resolved_node, "attrs", {}).get("downsample_factors_tpczyx")
    if isinstance(payload, (tuple, list)) and len(payload) == 6:
        try:
            return tuple(int(value) for value in payload)  # type: ignore[return-value]
        except Exception:
            pass

    try:
        source_node = get_node(root, source_component)
    except Exception:
        source_node = None
    source_attrs = dict(getattr(source_node, "attrs", {}))
    level_factors_tpczyx = (
        visualization_pipeline._resolve_visualization_pyramid_factors_tpczyx(
            root_attrs=dict(root.attrs),
            source_attrs=source_attrs,
        )
    )
    discovered_components = _discover_source_adjacent_resolution_components(
        root=root,
        source_component=source_component,
    )
    try:
        level_index = next(
            index
            for index, component in enumerate(discovered_components)
            if str(component).strip() == resolved_component
        )
    except StopIteration:
        return (1, 1, 1, 1, 1, 1)

    extended_factors_tpczyx = _extend_level_factors_tpczyx(
        level_factors_tpczyx,
        level_index,
    )
    return tuple(
        int(value) for value in extended_factors_tpczyx[int(level_index)]
    )  # type: ignore[return-value]


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

    discovered_components = _discover_source_adjacent_resolution_components(
        root=root,
        source_component=source_component,
    )
    if level_index < len(discovered_components):
        return str(discovered_components[level_index]), False

    return _generate_missing_volume_export_resolution_components(
        zarr_path=zarr_path,
        root=root,
        source_component=source_component,
        resolution_level=level_index,
        progress_callback=progress_callback,
    )


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


def _artifact_directory(zarr_path: str | Path) -> Path:
    """Return the filesystem directory for TIFF artifacts."""
    return Path(zarr_path).expanduser().resolve() / _ARTIFACTS_ROOT


def _reset_artifact_directory(zarr_path: str | Path) -> Path:
    """Remove any prior TIFF artifacts and return a fresh directory."""
    artifact_dir = _artifact_directory(zarr_path)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _artifact_relative_path(filename: str) -> str:
    """Return the store-relative artifact path for one filename."""
    return f"{_ARTIFACTS_ROOT}/{filename}"


def _ome_tiff_metadata(
    *,
    axes: str,
    voxel_size_um_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    """Return OME-TIFF metadata for one exported volume payload."""
    return {
        "axes": str(axes),
        "PhysicalSizeZ": float(voxel_size_um_zyx[0]),
        "PhysicalSizeY": float(voxel_size_um_zyx[1]),
        "PhysicalSizeX": float(voxel_size_um_zyx[2]),
    }


def _write_single_volume_tiff(
    *,
    output_path: Path,
    volume: np.ndarray[Any, Any],
    voxel_size_um_zyx: tuple[float, float, float],
) -> None:
    """Write one OME-TIFF volume with ``ZYX`` axes."""
    tifffile.imwrite(
        str(output_path),
        np.asarray(volume),
        photometric="minisblack",
        ome=True,
        bigtiff=True,
        metadata=_ome_tiff_metadata(
            axes="ZYX",
            voxel_size_um_zyx=voxel_size_um_zyx,
        ),
    )


def _write_volume_export_tiffs(
    *,
    zarr_path: str | Path,
    export_array: da.Array,
    export_scope: str,
    tiff_file_layout: str,
    selection_payload: Mapping[str, int],
    voxel_size_um_zyx: tuple[float, float, float],
) -> tuple[str, ...]:
    """Write TIFF artifacts for one resolved export payload."""
    artifact_dir = _reset_artifact_directory(zarr_path)
    artifact_paths: list[str] = []

    if str(export_scope) == "current_selection":
        filename = (
            "volume_export_"
            f"t{int(selection_payload['t_index']):04d}_"
            f"p{int(selection_payload['p_index']):04d}_"
            f"c{int(selection_payload['c_index']):04d}.ome.tif"
        )
        output_path = artifact_dir / filename
        _write_single_volume_tiff(
            output_path=output_path,
            volume=np.asarray(export_array[0, 0, 0].compute()),
            voxel_size_um_zyx=voxel_size_um_zyx,
        )
        artifact_paths.append(_artifact_relative_path(filename))
        return tuple(artifact_paths)

    if str(tiff_file_layout) == "single_file":
        filename = "volume_export_all_positions.ome.tif"
        output_path = artifact_dir / filename
        with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
            for position_index in range(int(export_array.shape[1])):
                position_payload = np.asarray(
                    export_array[:, position_index, :, :, :, :].compute()
                )
                tif.write(
                    position_payload,
                    photometric="minisblack",
                    metadata=_ome_tiff_metadata(
                        axes="TCZYX",
                        voxel_size_um_zyx=voxel_size_um_zyx,
                    ),
                )
        artifact_paths.append(_artifact_relative_path(filename))
        return tuple(artifact_paths)

    for time_index in range(int(export_array.shape[0])):
        for position_index in range(int(export_array.shape[1])):
            for channel_index in range(int(export_array.shape[2])):
                filename = (
                    "volume_export_"
                    f"t{int(time_index):04d}_"
                    f"p{int(position_index):04d}_"
                    f"c{int(channel_index):04d}.ome.tif"
                )
                output_path = artifact_dir / filename
                _write_single_volume_tiff(
                    output_path=output_path,
                    volume=np.asarray(
                        export_array[
                            time_index,
                            position_index,
                            channel_index,
                            :,
                            :,
                            :,
                        ].compute()
                    ),
                    voxel_size_um_zyx=voxel_size_um_zyx,
                )
                artifact_paths.append(_artifact_relative_path(filename))
    return tuple(artifact_paths)


def run_volume_export_analysis(
    *,
    zarr_path: str | Path,
    parameters: Mapping[str, Any],
    client: Optional[Any] = None,
    progress_callback: Optional[ProgressCallback] = None,
    run_id: Optional[str] = None,
) -> VolumeExportSummary:
    """Export the resolved source component into the runtime-cache volume."""
    del client
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
    if export_format not in {"ome-zarr", "ome-tiff"}:
        raise ValueError(
            "volume_export currently supports only export_format=ome-zarr or ome-tiff."
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
    if export_scope == "current_selection":
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

    voxel_size_um_zyx, voxel_size_source = _resolve_volume_export_voxel_size_um_zyx(
        root=root,
        source_component=source_component,
        resolved_resolution_component=resolved_resolution_component,
    )
    auxiliary_root = analysis_auxiliary_root(_ANALYSIS_NAME)
    delete_path(root, auxiliary_root)
    auxiliary_group = ensure_group(root, auxiliary_root)

    selection_payload: dict[str, int] = {
        "resolution_level": resolution_level,
    }
    if export_scope == "current_selection":
        selection_payload.update(
            {
                "t_index": t_index,
                "p_index": p_index,
                "c_index": c_index,
            }
        )
    export_shape_tpczyx = [int(value) for value in tuple(export_array.shape)]
    artifact_paths: tuple[str, ...]
    if export_format == "ome-tiff":
        artifact_paths = _write_volume_export_tiffs(
            zarr_path=zarr_path,
            export_array=export_array,
            export_scope=export_scope,
            tiff_file_layout=str(normalized.get("tiff_file_layout", "single_file")),
            selection_payload=selection_payload,
            voxel_size_um_zyx=voxel_size_um_zyx,
        )
    else:
        artifact_paths = (data_component, auxiliary_root)
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
        "artifact_paths": [str(path) for path in artifact_paths],
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
                "artifact_paths": [str(path) for path in artifact_paths],
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
        artifact_paths=tuple(str(path) for path in artifact_paths),
    )
