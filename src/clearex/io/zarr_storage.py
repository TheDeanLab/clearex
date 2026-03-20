"""Shared Zarr storage helpers for v2/v3 compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
import inspect
import json
import shutil

import dask.array as da
import numpy as np
import zarr


_OPEN_GROUP_SIGNATURE = inspect.signature(zarr.open_group)
_OPEN_GROUP_FORMAT_KEY = (
    "zarr_format"
    if "zarr_format" in _OPEN_GROUP_SIGNATURE.parameters
    else "zarr_version"
    if "zarr_version" in _OPEN_GROUP_SIGNATURE.parameters
    else None
)


def to_jsonable(value: Any) -> Any:
    """Round-trip a value through JSON-compatible types."""
    return json.loads(json.dumps(value))


def open_group(
    zarr_path: str | Path,
    *,
    mode: str = "a",
    zarr_format: Optional[int] = None,
) -> Any:
    """Open a Zarr group, forwarding format selection when supported."""
    kwargs: dict[str, Any] = {}
    if zarr_format is not None and _OPEN_GROUP_FORMAT_KEY is not None:
        kwargs[_OPEN_GROUP_FORMAT_KEY] = int(zarr_format)
    return zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode=mode, **kwargs)


def detect_store_format(zarr_path: str | Path) -> Optional[int]:
    """Return the on-disk Zarr format version when it can be inferred."""
    path = Path(zarr_path).expanduser().resolve()
    if not path.exists():
        return None
    if (path / "zarr.json").exists():
        return 3
    if (path / ".zgroup").exists() or (path / ".zarray").exists():
        return 2
    if path.suffix.lower() == ".n5":
        return 2
    return None


def is_clearex_analysis_store(zarr_path: str | Path) -> bool:
    """Return whether a store looks like a ClearEx-managed analysis store."""
    try:
        root = open_group(zarr_path, mode="r")
    except Exception:
        return False
    schema = str(root.attrs.get("schema", "")).strip()
    return schema.startswith("clearex.analysis_store")


def resolve_external_analysis_store_path(source_path: str | Path) -> Path:
    """Return the sibling ClearEx-managed store path for an external source store."""
    source = Path(source_path).expanduser().resolve()
    if source.name.endswith(".clearex.zarr"):
        return source
    return source.with_name(f"{source.name}.clearex.zarr")


def resolve_staging_store_path(target_store_path: str | Path) -> Path:
    """Return the stable sibling staging path for a target analysis store."""
    target = Path(target_store_path).expanduser().resolve()
    return target.with_name(f"{target.name}.staging")


def resolve_legacy_v2_store_path(target_store_path: str | Path) -> Path:
    """Return the sibling legacy-v2 handoff path used by the N5 helper."""
    target = Path(target_store_path).expanduser().resolve()
    return target.with_name(f"{target.name}.legacy-v2.zarr")


def replace_store_path(
    *,
    staging_path: str | Path,
    target_path: str | Path,
    keep_backup: bool = False,
) -> Optional[Path]:
    """Replace a target directory store with a fully written staging store."""
    staging = Path(staging_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    if staging == target:
        return None
    if not staging.exists():
        raise FileNotFoundError(staging)

    backup = target.with_name(f"{target.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)

    renamed_target = False
    if target.exists():
        target.rename(backup)
        renamed_target = True

    try:
        staging.rename(target)
    except Exception:
        if renamed_target and backup.exists() and not target.exists():
            backup.rename(target)
        raise

    if renamed_target and backup.exists() and not keep_backup:
        shutil.rmtree(backup)
        return None
    return backup if renamed_target and backup.exists() else None


def clear_component(root: Any, component: str) -> None:
    """Delete a component from a group when present."""
    if str(component) in root:
        del root[str(component)]


def create_or_overwrite_array(
    *,
    root: Any,
    name: str,
    shape: tuple[int, ...] | list[int] | None = None,
    chunks: Any = None,
    dtype: Any = None,
    data: Any = None,
    overwrite: bool = True,
    zarr_format: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Create an array across Zarr v2/v3, deleting any existing target when requested."""
    del zarr_format
    if overwrite:
        clear_component(root, name)
    prepared_data = (
        np.asarray(data, dtype=dtype) if data is not None and dtype is not None else data
    )
    effective_shape = (
        tuple(int(v) for v in prepared_data.shape)
        if prepared_data is not None and shape is None
        else shape
    )
    effective_chunks = (
        tuple(int(v) for v in effective_shape)
        if prepared_data is not None and chunks is None and effective_shape is not None
        else chunks
    )
    if hasattr(root, "create_array"):
        if prepared_data is not None:
            return root.create_array(
                name=str(name),
                data=prepared_data,
                chunks=effective_chunks,
                overwrite=False,
                **kwargs,
            )
        return root.create_array(
            name=str(name),
            shape=effective_shape,
            chunks=effective_chunks,
            dtype=dtype,
            overwrite=False,
            **kwargs,
        )
    return root.create_dataset(
        name=str(name),
        shape=effective_shape,
        chunks=effective_chunks,
        dtype=dtype,
        data=prepared_data,
        overwrite=False,
        **kwargs,
    )


def _default_chunks_for_array(array: da.Array) -> tuple[int, ...]:
    """Return a concrete chunk tuple from a Dask array."""
    return tuple(int(axis_chunks[0]) for axis_chunks in array.chunks)


def write_dask_array(
    *,
    zarr_path: str | Path,
    component: str,
    array: da.Array,
    overwrite: bool = True,
    chunks: Optional[tuple[int, ...]] = None,
    compute: bool = True,
    zarr_format: Optional[int] = None,
) -> Any:
    """Write a Dask array to an explicitly created Zarr array object."""
    root = open_group(zarr_path, mode="a", zarr_format=zarr_format)
    target = create_or_overwrite_array(
        root=root,
        name=str(component),
        shape=tuple(int(v) for v in array.shape),
        chunks=tuple(int(v) for v in (chunks or _default_chunks_for_array(array))),
        dtype=array.dtype,
        overwrite=overwrite,
    )
    return da.to_zarr(array, target, compute=compute)


def _extract_ome_multiscales(attrs: Mapping[str, Any]) -> Any:
    """Return OME multiscales metadata from either v0.4 or v0.5 layouts."""
    ome_payload = attrs.get("ome")
    if isinstance(ome_payload, Mapping):
        multiscales = ome_payload.get("multiscales")
        if multiscales:
            return multiscales
    return attrs.get("multiscales")


def _normalize_axis_payload(value: Any) -> Any:
    """Collapse OME axis dictionaries into plain axis-name tokens."""
    if not isinstance(value, (list, tuple)):
        return value
    normalized: list[Any] = []
    changed = False
    for axis in value:
        if isinstance(axis, Mapping):
            normalized.append(axis.get("name"))
            changed = True
        else:
            normalized.append(axis)
    return normalized if changed else value


def extract_raw_axes_metadata(array: Any, group_attrs: Mapping[str, Any]) -> Any:
    """Return the best raw axis descriptor available for a Zarr array."""
    attrs = dict(getattr(array, "attrs", {}))
    metadata = getattr(array, "metadata", None)
    dimension_names = getattr(metadata, "dimension_names", None)
    if dimension_names:
        return _normalize_axis_payload(dimension_names)

    raw_axes = (
        _extract_ome_multiscales(attrs)[0].get("axes")
        if isinstance(_extract_ome_multiscales(attrs), list)
        and _extract_ome_multiscales(attrs)
        else _extract_ome_multiscales(group_attrs)[0].get("axes")
        if isinstance(_extract_ome_multiscales(group_attrs), list)
        and _extract_ome_multiscales(group_attrs)
        else attrs.get("_ARRAY_DIMENSIONS")
        or group_attrs.get("_ARRAY_DIMENSIONS")
        or attrs.get("axes")
        or group_attrs.get("axes")
    )
    return _normalize_axis_payload(raw_axes)
