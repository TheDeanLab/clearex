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

"""OME-Zarr v3 storage helpers for canonical ClearEx stores."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import json
import shutil

import dask.array as da
import zarr

from ome_zarr_models.common.plate import Column, Row, WellInPlate
from ome_zarr_models.v05.hcs import HCSAttrs
from ome_zarr_models.v05.image import ImageAttrs
from ome_zarr_models.v05.multiscales import Dataset, Multiscale
from ome_zarr_models.v05.plate import Plate
from ome_zarr_models.v05.well import WellAttrs
from ome_zarr_models.v05.well_types import WellImage, WellMeta

from clearex.workflow import (
    SpatialCalibrationConfig,
    spatial_calibration_from_dict,
    spatial_calibration_to_dict,
)

PUBLIC_WELL_ROW = "A"
PUBLIC_WELL_COLUMN = "1"
OME_ZARR_STORE_SUFFIX = ".ome.zarr"

CLEAREX_ROOT_GROUP = "clearex"
CLEAREX_METADATA_GROUP = f"{CLEAREX_ROOT_GROUP}/metadata"
CLEAREX_PROVENANCE_GROUP = f"{CLEAREX_ROOT_GROUP}/provenance"
CLEAREX_GUI_STATE_GROUP = f"{CLEAREX_ROOT_GROUP}/gui_state"
CLEAREX_RESULTS_GROUP = f"{CLEAREX_ROOT_GROUP}/results"
CLEAREX_RUNTIME_CACHE_ROOT = f"{CLEAREX_ROOT_GROUP}/runtime_cache"
CLEAREX_RUNTIME_SOURCE_ROOT = f"{CLEAREX_RUNTIME_CACHE_ROOT}/source"
CLEAREX_RUNTIME_RESULTS_ROOT = f"{CLEAREX_RUNTIME_CACHE_ROOT}/results"

SOURCE_CACHE_COMPONENT = f"{CLEAREX_RUNTIME_SOURCE_ROOT}/data"
SOURCE_CACHE_PYRAMID_ROOT = f"{CLEAREX_RUNTIME_SOURCE_ROOT}/data_pyramid"
STORE_METADATA_SCHEMA = "clearex.ome_store.v1"
LEGACY_STORE_MIGRATION_HINT = (
    "Legacy ClearEx stores are no longer treated as canonical runtime inputs. "
    "Run `clearex --migrate-store <legacy_store>` to convert them to "
    "OME-Zarr v3."
)


def default_ome_store_path(experiment_directory: Path) -> Path:
    """Return the canonical OME-Zarr v3 output path beside ``experiment.yml``.

    Parameters
    ----------
    experiment_directory : pathlib.Path
        Parent directory for the source experiment file.

    Returns
    -------
    pathlib.Path
        Canonical output path.
    """
    return experiment_directory.resolve() / f"data_store{OME_ZARR_STORE_SUFFIX}"


def is_ome_zarr_path(path: Path) -> bool:
    """Return whether a path is an OME-Zarr directory-style store."""
    normalized = str(path).lower()
    return path.is_dir() and normalized.endswith(OME_ZARR_STORE_SUFFIX)


def public_analysis_root(analysis_name: str) -> str:
    """Return the public OME image-collection root for one analysis."""
    key = str(analysis_name).strip()
    return f"results/{key}/latest"


def analysis_cache_root(analysis_name: str) -> str:
    """Return the runtime-cache root for one analysis output."""
    key = str(analysis_name).strip()
    return f"{CLEAREX_RUNTIME_RESULTS_ROOT}/{key}/latest"


def analysis_cache_data_component(analysis_name: str, *, level_index: int = 0) -> str:
    """Return one runtime-cache image component path for an analysis output."""
    root = analysis_cache_root(analysis_name)
    if level_index <= 0:
        return f"{root}/data"
    return f"{root}/data_pyramid/level_{int(level_index)}"


def analysis_auxiliary_root(analysis_name: str) -> str:
    """Return the ClearEx-owned auxiliary artifact root for one analysis."""
    key = str(analysis_name).strip()
    return f"{CLEAREX_RESULTS_GROUP}/{key}/latest"


def analysis_auxiliary_component(analysis_name: str, name: str) -> str:
    """Return one ClearEx-owned auxiliary artifact component path."""
    return f"{analysis_auxiliary_root(analysis_name)}/{str(name).strip()}"


def source_cache_component(*, level_index: int = 0) -> str:
    """Return one runtime-cache source image component path."""
    if level_index <= 0:
        return SOURCE_CACHE_COMPONENT
    return f"{SOURCE_CACHE_PYRAMID_ROOT}/level_{int(level_index)}"


def _split_component(path: str) -> tuple[str, ...]:
    return tuple(part for part in str(path).strip("/").split("/") if part)


def ensure_group(root: zarr.Group, path: str) -> zarr.Group:
    """Ensure a nested group path exists below ``root``."""
    group = root
    for token in _split_component(path):
        group = group.require_group(token)
    return group


def get_node(root: zarr.Group, path: str) -> Any:
    """Return a nested group or array node."""
    node: Any = root
    for token in _split_component(path):
        node = node[token]
    return node


def delete_path(root: zarr.Group, path: str) -> None:
    """Delete a nested path when it exists."""
    tokens = list(_split_component(path))
    if not tokens:
        return
    parent = root
    for token in tokens[:-1]:
        if token not in parent:
            return
        parent = parent[token]
    leaf = tokens[-1]
    if leaf in parent:
        del parent[leaf]


def load_store_metadata(path_or_root: str | Path | zarr.Group) -> dict[str, Any]:
    """Load ClearEx namespaced store metadata."""
    root = (
        path_or_root
        if isinstance(path_or_root, zarr.Group)
        else zarr.open_group(str(Path(path_or_root).expanduser().resolve()), mode="a")
    )
    group = ensure_group(root, CLEAREX_METADATA_GROUP)
    payload = dict(group.attrs)
    if payload:
        return payload
    return {"schema": STORE_METADATA_SCHEMA}


def update_store_metadata(
    path_or_root: str | Path | zarr.Group, **payload: Any
) -> dict[str, Any]:
    """Merge and persist ClearEx namespaced store metadata."""
    root = (
        path_or_root
        if isinstance(path_or_root, zarr.Group)
        else zarr.open_group(str(Path(path_or_root).expanduser().resolve()), mode="a")
    )
    group = ensure_group(root, CLEAREX_METADATA_GROUP)
    current = {"schema": STORE_METADATA_SCHEMA}
    current.update(dict(group.attrs))
    current.update(json.loads(json.dumps(payload)))
    group.attrs.update(current)
    return dict(group.attrs)


def store_has_public_ome_metadata(path_or_root: str | Path | zarr.Group) -> bool:
    """Return whether a store/root advertises public OME metadata at the root."""
    root = (
        path_or_root
        if isinstance(path_or_root, zarr.Group)
        else zarr.open_group(str(Path(path_or_root).expanduser().resolve()), mode="r")
    )
    payload = getattr(root, "attrs", {}).get("ome")
    return isinstance(payload, Mapping)


def store_has_valid_public_source_collection(
    path_or_root: str | Path | zarr.Group,
) -> bool:
    """Return whether the root source HCS collection validates via OME models.

    Parameters
    ----------
    path_or_root : str or pathlib.Path or zarr.Group
        Store path or opened root group.

    Returns
    -------
    bool
        ``True`` when root/well/image OME metadata validates and every
        declared multiscale dataset path exists.
    """
    root = (
        path_or_root
        if isinstance(path_or_root, zarr.Group)
        else zarr.open_group(str(Path(path_or_root).expanduser().resolve()), mode="r")
    )
    well_path = f"{PUBLIC_WELL_ROW}/{PUBLIC_WELL_COLUMN}"
    try:
        root_ome = HCSAttrs.model_validate(getattr(root, "attrs", {}).get("ome"))
        well_group = get_node(root, well_path)
        well_ome = WellAttrs.model_validate(getattr(well_group, "attrs", {}).get("ome"))
    except Exception:
        return False

    declared_wells = tuple(
        str(entry.path).strip() for entry in root_ome.plate.wells if entry is not None
    )
    if well_path not in declared_wells:
        return False

    for image in well_ome.well.images:
        image_token = str(image.path).strip()
        if not image_token:
            return False
        image_component = f"{well_path}/{image_token}"
        try:
            image_group = get_node(root, image_component)
            image_ome = ImageAttrs.model_validate(
                getattr(image_group, "attrs", {}).get("ome")
            )
        except Exception:
            return False
        if not image_ome.multiscales:
            return False
        for multiscale in image_ome.multiscales:
            for dataset in multiscale.datasets:
                dataset_path = str(dataset.path).strip()
                if not dataset_path:
                    return False
                if dataset_path not in image_group:
                    return False

    return True


def is_legacy_clearex_store(path_or_root: str | Path | zarr.Group) -> bool:
    """Return whether a store still follows the legacy pre-OME ClearEx layout."""
    root = (
        path_or_root
        if isinstance(path_or_root, zarr.Group)
        else zarr.open_group(str(Path(path_or_root).expanduser().resolve()), mode="r")
    )
    if store_has_public_ome_metadata(root):
        return False
    legacy_markers = (
        "data" in root,
        "data_pyramid" in root,
        "provenance" in root,
        "results" in root and CLEAREX_ROOT_GROUP not in root,
        root.attrs.get("schema") == "clearex.analysis_store.v1",
        root.attrs.get("data_pyramid_levels") is not None,
    )
    return any(bool(marker) for marker in legacy_markers)


def load_store_spatial_calibration(
    path_or_root: str | Path | zarr.Group,
) -> SpatialCalibrationConfig:
    """Load store-level spatial calibration from the namespaced metadata group."""
    metadata = load_store_metadata(path_or_root)
    return spatial_calibration_from_dict(metadata.get("spatial_calibration"))


def save_store_spatial_calibration(
    path_or_root: str | Path | zarr.Group,
    calibration: SpatialCalibrationConfig | Mapping[str, Any] | str | None,
) -> SpatialCalibrationConfig:
    """Persist store-level spatial calibration in the namespaced metadata group."""
    normalized = spatial_calibration_from_dict(calibration)
    update_store_metadata(
        path_or_root,
        spatial_calibration=spatial_calibration_to_dict(normalized),
    )
    return normalized


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe shallow mapping copy."""
    return json.loads(json.dumps(dict(payload)))


def _copy_array(
    *,
    source: zarr.Array,
    dest_root: zarr.Group,
    dest_component: str,
) -> None:
    """Copy one array into the destination store."""
    parent_path, _, leaf = str(dest_component).rpartition("/")
    parent = ensure_group(dest_root, parent_path) if parent_path else dest_root
    if leaf in parent:
        del parent[leaf]
    shape = tuple(int(value) for value in source.shape)
    source_chunks = getattr(source, "chunks", None)
    chunks = (
        tuple(int(value) for value in source_chunks)
        if isinstance(source_chunks, (tuple, list))
        else shape
    )
    target = parent.create_array(
        leaf,
        shape=shape,
        chunks=chunks,
        dtype=source.dtype,
        overwrite=True,
    )
    da.store(da.from_zarr(source), target, lock=False, compute=True)
    target.attrs.update(_jsonable_mapping(dict(source.attrs)))


def _copy_group_contents(
    *,
    source: zarr.Group,
    dest_root: zarr.Group,
    dest_prefix: str,
    skip_children: Optional[set[str]] = None,
) -> None:
    """Recursively copy one group tree into the destination store."""
    skip = {str(token) for token in (skip_children or set())}
    target_group = ensure_group(dest_root, dest_prefix)
    target_group.attrs.update(_jsonable_mapping(dict(source.attrs)))
    for key in sorted(source.array_keys()):
        if str(key) in skip:
            continue
        _copy_array(
            source=source[key],
            dest_root=dest_root,
            dest_component=f"{dest_prefix}/{key}" if dest_prefix else str(key),
        )
    for key in sorted(source.group_keys()):
        if str(key) in skip:
            continue
        _copy_group_contents(
            source=source[key],
            dest_root=dest_root,
            dest_prefix=f"{dest_prefix}/{key}" if dest_prefix else str(key),
        )


def compute_position_translations_zyx_um(
    stage_rows: Sequence[Mapping[str, Any]] | None,
    spatial_calibration: SpatialCalibrationConfig | Mapping[str, Any] | str | None,
    *,
    position_count: int,
) -> list[list[float]]:
    """Resolve one world-space ``(z, y, x)`` translation per position."""
    normalized = spatial_calibration_from_dict(spatial_calibration)
    rows = list(stage_rows or [])
    if not rows:
        return [[0.0, 0.0, 0.0] for _ in range(max(1, int(position_count)))]

    def _row_value(row: Mapping[str, Any], axis_name: str) -> float:
        axis = str(axis_name).strip().lower()
        for key, value in row.items():
            if str(key).strip().lower() == axis:
                try:
                    return float(value)
                except Exception:
                    return 0.0
        return 0.0

    def _binding_value(row: Mapping[str, Any], binding: str) -> float:
        token = str(binding).strip().lower()
        if token == "none":
            return 0.0
        sign = -1.0 if token.startswith("-") else 1.0
        axis_name = token[1:] if token[:1] in {"+", "-"} else token
        return sign * _row_value(row, axis_name)

    stage_axis_map = normalized.stage_axis_map_by_world_axis()
    reference = rows[0]
    translations: list[list[float]] = []
    for position_index in range(max(1, int(position_count))):
        row = rows[position_index] if position_index < len(rows) else reference
        translations.append(
            [
                float(
                    _binding_value(row, stage_axis_map["z"])
                    - _binding_value(reference, stage_axis_map["z"])
                ),
                float(
                    _binding_value(row, stage_axis_map["y"])
                    - _binding_value(reference, stage_axis_map["y"])
                ),
                float(
                    _binding_value(row, stage_axis_map["x"])
                    - _binding_value(reference, stage_axis_map["x"])
                ),
            ]
        )
    return translations


def _model_payload(model: Any) -> dict[str, Any]:
    return json.loads(model.model_dump_json(by_alias=True, exclude_none=True))


def _level_component_paths(cache_root: str, root: zarr.Group) -> list[str]:
    components = [f"{cache_root}/data"]
    pyramid_root = f"{cache_root}/data_pyramid"
    try:
        pyramid_group = get_node(root, pyramid_root)
    except Exception:
        pyramid_group = None
    if isinstance(pyramid_group, zarr.Group):
        level_tokens = sorted(
            (
                token
                for token in pyramid_group.group_keys()
                if str(token).startswith("level_")
            ),
            key=lambda token: int(str(token).split("level_", maxsplit=1)[1]),
        )
        components.extend(f"{pyramid_root}/{token}" for token in level_tokens)
    return components


def _level_downsample_factors(
    level_array: Any, *, level_index: int
) -> tuple[int, int, int, int, int, int]:
    payload = getattr(level_array, "attrs", {}).get("downsample_factors_tpczyx")
    if isinstance(payload, (tuple, list)) and len(payload) == 6:
        try:
            return tuple(int(value) for value in payload)  # type: ignore[return-value]
        except Exception:
            pass
    fallback = 2 ** max(0, int(level_index))
    return (1, 1, 1, fallback, fallback, fallback)


def _set_hcs_group_attrs(
    collection_group: zarr.Group, *, name: str, field_count: int
) -> None:
    payload = HCSAttrs(
        version="0.5",
        plate=Plate(
            version="0.5",
            name=str(name),
            rows=[Row(name=PUBLIC_WELL_ROW)],
            columns=[Column(name=PUBLIC_WELL_COLUMN)],
            wells=[
                WellInPlate(
                    path=f"{PUBLIC_WELL_ROW}/{PUBLIC_WELL_COLUMN}",
                    rowIndex=0,
                    columnIndex=0,
                )
            ],
            field_count=max(1, int(field_count)),
        ),
    )
    collection_group.attrs["ome"] = _model_payload(payload)


def _set_well_group_attrs(
    well_group: zarr.Group, *, field_paths: Sequence[str]
) -> None:
    payload = WellAttrs(
        version="0.5",
        well=WellMeta(images=[WellImage(path=str(path)) for path in field_paths]),
    )
    well_group.attrs["ome"] = _model_payload(payload)


def _set_image_group_attrs(
    image_group: zarr.Group,
    *,
    level_count: int,
    voxel_size_um_zyx: Sequence[float] | None,
    position_translation_zyx_um: Sequence[float] | None,
    level_factors_tpczyx: Sequence[tuple[int, int, int, int, int, int]],
) -> None:
    voxel = tuple(float(value) for value in (voxel_size_um_zyx or (1.0, 1.0, 1.0)))
    translation = tuple(
        float(value) for value in (position_translation_zyx_um or (0.0, 0.0, 0.0))
    )
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = []
    for level_index in range(max(1, int(level_count))):
        factors = level_factors_tpczyx[level_index]
        scale = [
            1.0,
            1.0,
            float(voxel[0]) * float(factors[3]),
            float(voxel[1]) * float(factors[4]),
            float(voxel[2]) * float(factors[5]),
        ]
        datasets.append(
            Dataset(
                path=str(level_index),
                coordinateTransformations=[
                    {"type": "scale", "scale": scale},
                    {
                        "type": "translation",
                        "translation": [
                            0.0,
                            0.0,
                            translation[0],
                            translation[1],
                            translation[2],
                        ],
                    },
                ],
            )
        )
    payload = ImageAttrs(
        version="0.5",
        multiscales=[Multiscale(axes=axes, datasets=tuple(datasets))],
    )
    image_group.attrs["ome"] = _model_payload(payload)


def _prepare_public_collection_root(
    root: zarr.Group,
    *,
    public_root: str,
    name: str,
    field_count: int,
) -> zarr.Group:
    if str(public_root).strip():
        delete_path(root, public_root)
        collection_group = ensure_group(root, public_root)
    else:
        if PUBLIC_WELL_ROW in root:
            del root[PUBLIC_WELL_ROW]
        collection_group = root
    _set_hcs_group_attrs(collection_group, name=name, field_count=field_count)
    ensure_group(collection_group, f"{PUBLIC_WELL_ROW}/{PUBLIC_WELL_COLUMN}")
    well_group = get_node(collection_group, f"{PUBLIC_WELL_ROW}/{PUBLIC_WELL_COLUMN}")
    _set_well_group_attrs(
        well_group,
        field_paths=[str(index) for index in range(max(1, int(field_count)))],
    )
    return collection_group


def publish_image_collection_from_cache(
    zarr_path: str | Path,
    *,
    cache_root: str,
    public_root: str,
    name: str,
) -> str:
    """Publish one runtime-cache 6D image collection as public OME-Zarr HCS data."""
    store_path = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(store_path), mode="a")
    cache_components = _level_component_paths(cache_root, root)
    if not cache_components:
        raise ValueError(f"No runtime-cache data was found at {cache_root}.")

    base_array = get_node(root, cache_components[0])
    if len(tuple(base_array.shape)) != 6:
        raise ValueError(
            f"Expected runtime-cache image data at {cache_components[0]} to be 6D."
        )

    shape_tpczyx = tuple(int(value) for value in base_array.shape)
    position_count = max(1, int(shape_tpczyx[1]))
    voxel_size_um_zyx = base_array.attrs.get("voxel_size_um_zyx")
    metadata = load_store_metadata(root)
    translations = metadata.get("position_translations_zyx_um")
    if not isinstance(translations, list):
        translations = [[0.0, 0.0, 0.0] for _ in range(position_count)]

    level_arrays = [get_node(root, component) for component in cache_components]
    level_factors = [
        (1, 1, 1, 1, 1, 1),
        *[
            _level_downsample_factors(level_array, level_index=index)
            for index, level_array in enumerate(level_arrays[1:], start=1)
        ],
    ]

    collection_group = _prepare_public_collection_root(
        root,
        public_root=public_root,
        name=name,
        field_count=position_count,
    )
    well_group = get_node(collection_group, f"{PUBLIC_WELL_ROW}/{PUBLIC_WELL_COLUMN}")

    for position_index in range(position_count):
        field_name = str(position_index)
        image_group = ensure_group(well_group, field_name)
        _set_image_group_attrs(
            image_group,
            level_count=len(level_arrays),
            voxel_size_um_zyx=voxel_size_um_zyx,
            position_translation_zyx_um=(
                translations[position_index]
                if position_index < len(translations)
                else (0.0, 0.0, 0.0)
            ),
            level_factors_tpczyx=level_factors,
        )
        for level_index, level_array in enumerate(level_arrays):
            level_shape_tczyx = (
                int(level_array.shape[0]),
                int(level_array.shape[2]),
                int(level_array.shape[3]),
                int(level_array.shape[4]),
                int(level_array.shape[5]),
            )
            level_chunks_tczyx = (
                (
                    int(level_array.chunks[0])
                    if level_array.chunks is not None
                    else level_shape_tczyx[0]
                ),
                (
                    int(level_array.chunks[2])
                    if level_array.chunks is not None
                    else level_shape_tczyx[1]
                ),
                (
                    int(level_array.chunks[3])
                    if level_array.chunks is not None
                    else level_shape_tczyx[2]
                ),
                (
                    int(level_array.chunks[4])
                    if level_array.chunks is not None
                    else level_shape_tczyx[3]
                ),
                (
                    int(level_array.chunks[5])
                    if level_array.chunks is not None
                    else level_shape_tczyx[4]
                ),
            )
            if str(level_index) in image_group:
                del image_group[str(level_index)]
            target = image_group.create_array(
                str(level_index),
                shape=level_shape_tczyx,
                chunks=level_chunks_tczyx,
                dtype=level_array.dtype,
                overwrite=True,
                dimension_names=("t", "c", "z", "y", "x"),
            )
            source = da.from_zarr(level_array)[:, position_index, :, :, :, :]
            da.store(source, target, lock=False, compute=True)

    return public_root


def publish_source_collection_from_cache(zarr_path: str | Path) -> str:
    """Publish the canonical source runtime-cache image tree at the store root."""
    return publish_image_collection_from_cache(
        zarr_path,
        cache_root=CLEAREX_RUNTIME_SOURCE_ROOT,
        public_root="",
        name="clearex-source",
    )


def publish_analysis_collection_from_cache(
    zarr_path: str | Path,
    *,
    analysis_name: str,
) -> str:
    """Publish one analysis runtime-cache image collection under ``results/``."""
    key = str(analysis_name).strip()
    return publish_image_collection_from_cache(
        zarr_path,
        cache_root=analysis_cache_root(key),
        public_root=public_analysis_root(key),
        name=key,
    )


def default_migrated_ome_store_path(legacy_store_path: str | Path) -> Path:
    """Return the default destination path for legacy-store migration."""
    source_path = Path(legacy_store_path).expanduser().resolve()
    name = source_path.name
    if name.endswith(".zarr"):
        stem = name[: -len(".zarr")]
        return source_path.with_name(f"{stem}{OME_ZARR_STORE_SUFFIX}")
    if name.endswith(".n5"):
        stem = name[: -len(".n5")]
        return source_path.with_name(f"{stem}{OME_ZARR_STORE_SUFFIX}")
    return source_path.with_name(f"{name}{OME_ZARR_STORE_SUFFIX}")


def migrate_legacy_store(
    legacy_store_path: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Migrate one legacy ClearEx store into canonical OME-Zarr layout."""
    source_path = Path(legacy_store_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_root = zarr.open_group(str(source_path), mode="r")
    if not is_legacy_clearex_store(source_root):
        raise ValueError(f"Path is not a legacy ClearEx store: {source_path}")

    dest_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_migrated_ome_store_path(source_path)
    )
    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(dest_path)
        if dest_path.is_dir():
            shutil.rmtree(dest_path)
        else:
            dest_path.unlink()

    dest_root = zarr.open_group(str(dest_path), mode="a")
    ensure_group(dest_root, CLEAREX_ROOT_GROUP)
    ensure_group(dest_root, CLEAREX_METADATA_GROUP)
    ensure_group(dest_root, CLEAREX_PROVENANCE_GROUP)
    ensure_group(dest_root, CLEAREX_RESULTS_GROUP)
    ensure_group(dest_root, CLEAREX_RUNTIME_SOURCE_ROOT)
    ensure_group(dest_root, CLEAREX_RUNTIME_RESULTS_ROOT)

    if "data" not in source_root:
        raise ValueError(
            f"Legacy source store '{source_path}' does not contain a root 'data' array."
        )
    _copy_array(
        source=source_root["data"],
        dest_root=dest_root,
        dest_component=SOURCE_CACHE_COMPONENT,
    )
    if "data_pyramid" in source_root:
        _copy_group_contents(
            source=source_root["data_pyramid"],
            dest_root=dest_root,
            dest_prefix=SOURCE_CACHE_PYRAMID_ROOT,
        )

    root_attrs = _jsonable_mapping(dict(source_root.attrs))
    source_metadata = {
        str(key): value
        for key, value in root_attrs.items()
        if str(key) not in {"ome", "multiscales"}
    }
    source_metadata.update(
        {
            "migrated_from_store": str(source_path),
            "migrated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "legacy_layout": True,
        }
    )
    update_store_metadata(dest_root, **source_metadata)

    if "provenance" in source_root:
        _copy_group_contents(
            source=source_root["provenance"],
            dest_root=dest_root,
            dest_prefix=CLEAREX_PROVENANCE_GROUP,
        )

    if "results" in source_root:
        results_group = source_root["results"]
        for analysis_name in sorted(results_group.group_keys()):
            analysis_group = results_group[analysis_name]
            latest_group = analysis_group.get("latest")
            if not isinstance(latest_group, zarr.Group):
                continue

            if "data" in latest_group:
                _copy_array(
                    source=latest_group["data"],
                    dest_root=dest_root,
                    dest_component=analysis_cache_data_component(analysis_name),
                )
                if "data_pyramid" in latest_group:
                    _copy_group_contents(
                        source=latest_group["data_pyramid"],
                        dest_root=dest_root,
                        dest_prefix=f"{analysis_cache_root(analysis_name)}/data_pyramid",
                    )
                _copy_group_contents(
                    source=latest_group,
                    dest_root=dest_root,
                    dest_prefix=analysis_auxiliary_root(analysis_name),
                    skip_children={"data", "data_pyramid"},
                )
                publish_analysis_collection_from_cache(
                    dest_path,
                    analysis_name=analysis_name,
                )
                continue

            _copy_group_contents(
                source=latest_group,
                dest_root=dest_root,
                dest_prefix=analysis_auxiliary_root(analysis_name),
            )

    publish_source_collection_from_cache(dest_path)
    return dest_path
