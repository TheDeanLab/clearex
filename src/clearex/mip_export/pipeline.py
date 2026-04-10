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

"""Maximum-intensity projection export workflow for canonical 6D Zarr stores."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union

# Third Party Imports
import dask
from dask import delayed
import numpy as np
import tifffile
import zarr

# Local Imports
from clearex.io.ome_store import (
    analysis_auxiliary_root,
    resolve_voxel_size_um_zyx_with_source,
)
from clearex.io.provenance import register_latest_output_reference

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]
_PROJECTIONS = ("xy", "xz", "yz")
_MAX_REDUCTION_READ_BYTES = 128 * 1024 * 1024
_MAX_INTERPOLATION_BLOCK_BYTES = 64 * 1024 * 1024
_OME_ZARR_DATASET_NAME = "0"


def _coerce_bool(value: Any) -> bool:
    """Coerce CLI/UI boolean-like values into ``bool``.

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    bool
        Parsed boolean value.

    Raises
    ------
    None
        Invalid values are coerced with Python truthiness rules.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalize_export_format(value: Any) -> str:
    """Normalize MIP-export format aliases.

    Parameters
    ----------
    value : Any
        Candidate format text.

    Returns
    -------
    str
        Canonical export format (`ome-tiff` or `zarr`).

    Raises
    ------
    ValueError
        If format text is unsupported.
    """
    text = str(value).strip().lower() or "ome-tiff"
    if text in {"tiff", "ome-tiff", "ome_tiff", "ome.tiff"}:
        return "ome-tiff"
    if text == "zarr":
        return "zarr"
    raise ValueError("mip_export export_format must be one of: ome-tiff, tiff, zarr.")


def _is_tiff_export_format(export_format: str) -> bool:
    """Return whether export format should be written as OME-TIFF."""
    return _normalize_export_format(export_format) == "ome-tiff"


@dataclass(frozen=True)
class _MipExportTask:
    """One MIP-export work item.

    Attributes
    ----------
    projection : str
        Projection identifier (``xy``, ``xz``, or ``yz``).
    t_index : int
        Time index.
    c_index : int
        Channel index.
    p_index : int, optional
        Position index for ``per_position`` mode.
    """

    projection: str
    t_index: int
    c_index: int
    p_index: Optional[int]


@dataclass(frozen=True)
class _PlannedOutput:
    """Execution plan metadata for one projection output file.

    Attributes
    ----------
    task_index : int
        Stable index of the projection task in submission order.
    task : _MipExportTask
        Projection export task definition.
    axes : tuple[str, ...]
        Output axis labels.
    output_shape : tuple[int, ...]
        Output array shape.
    preserved_axes : tuple[int, ...]
        Source-axis indices preserved in output order.
    reduction_axis : int
        Source-axis index reduced by max-projection.
    reduction_block_size : int
        Reduction-axis block length used for source reads.
    tile_shape : tuple[int, ...]
        Output tile shape used for distributed tile tasks.
    output_tiles : tuple[tuple[slice, ...], ...]
        Output tile slices.
    output_path : pathlib.Path
        Destination output path.
    """

    task_index: int
    task: _MipExportTask
    axes: tuple[str, ...]
    output_shape: tuple[int, ...]
    preserved_axes: tuple[int, ...]
    reduction_axis: int
    reduction_block_size: int
    tile_shape: tuple[int, ...]
    output_tiles: tuple[tuple[slice, ...], ...]
    output_path: Path


@dataclass(frozen=True)
class MipExportSummary:
    """Summary metadata for one MIP-export run.

    Attributes
    ----------
    component : str
        Output latest group component path.
    source_component : str
        Source data component used as input.
    output_directory : str
        Filesystem directory containing exported projection files.
    export_format : str
        Export file format (``ome-tiff`` or ``zarr``).
    position_mode : str
        Export mode (``multi_position`` or ``per_position``).
    task_count : int
        Number of projection tasks executed.
    exported_files : int
        Number of files written.
    projections : tuple[str, str, str]
        Exported projection identifiers.
    """

    component: str
    source_component: str
    output_directory: str
    export_format: str
    position_mode: str
    task_count: int
    exported_files: int
    projections: tuple[str, str, str]


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize MIP-export runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized parameter mapping.

    Raises
    ------
    ValueError
        If mode or format values are unsupported.
    """
    normalized = dict(parameters)
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    normalized["force_rerun"] = bool(normalized.get("force_rerun", False))
    normalized["chunk_basis"] = "3d"
    normalized["detect_2d_per_slice"] = False
    normalized["use_map_overlap"] = False
    normalized["overlap_zyx"] = [0, 0, 0]
    normalized["memory_overhead_factor"] = float(
        normalized.get("memory_overhead_factor", 1.0)
    )

    position_mode = (
        str(normalized.get("position_mode", "multi_position")).strip().lower()
        or "multi_position"
    )
    if position_mode not in {"multi_position", "per_position"}:
        raise ValueError(
            "mip_export position_mode must be 'multi_position' or 'per_position'."
        )
    normalized["position_mode"] = position_mode

    normalized["export_format"] = _normalize_export_format(
        normalized.get("export_format", "ome-tiff")
    )
    normalized["resample_z_to_lateral"] = _coerce_bool(
        normalized.get("resample_z_to_lateral", True)
    )
    normalized["output_directory"] = str(normalized.get("output_directory", "")).strip()
    return normalized


def _extract_voxel_size_um_zyx(
    *,
    root: zarr.Group,
    source_component: str,
) -> tuple[float, float, float]:
    """Extract voxel size metadata in ``(z, y, x)`` order.

    Parameters
    ----------
    root : zarr.Group
        Open analysis-store root group.
    source_component : str
        Source component path.

    Returns
    -------
    tuple[float, float, float]
        Voxel sizes in microns for ``(z, y, x)``. Falls back to ``(1, 1, 1)``
        when metadata is unavailable.
    """
    voxel_size_um_zyx, _ = resolve_voxel_size_um_zyx_with_source(
        root,
        source_component=source_component,
    )
    return voxel_size_um_zyx


def _projection_pixel_size_um(
    *,
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
) -> tuple[float, float]:
    """Return output pixel calibration ``(PhysicalSizeY, PhysicalSizeX)``.

    Parameters
    ----------
    axes : tuple[str, ...]
        Output axis labels in array order.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel size in ``(z, y, x)`` order.

    Returns
    -------
    tuple[float, float]
        Physical pixel sizes for output row/column axes in microns.
    """
    z_um, y_um, x_um = voxel_size_um_zyx
    axis_to_um = {"z": float(z_um), "y": float(y_um), "x": float(x_um)}
    if len(axes) < 2:
        return 1.0, 1.0
    row_axis = str(axes[-2]).lower()
    col_axis = str(axes[-1]).lower()
    physical_size_y = float(axis_to_um.get(row_axis, 1.0))
    physical_size_x = float(axis_to_um.get(col_axis, 1.0))
    return physical_size_y, physical_size_x


def _effective_voxel_size_um_zyx_for_projection(
    *,
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
    resample_z_to_lateral: bool,
) -> tuple[float, float, float]:
    """Return effective voxel spacing after optional projection resampling.

    Parameters
    ----------
    axes : tuple[str, ...]
        Output axis labels in array order.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel spacing in microns.
    resample_z_to_lateral : bool
        Whether projected outputs containing ``z`` should resample the ``z`` axis
        to the in-plane lateral spacing.

    Returns
    -------
    tuple[float, float, float]
        Effective voxel spacing in ``(z, y, x)`` after projection-space
        isotropic-lateral normalization.

    Raises
    ------
    None
        Non-positive spacing values disable resampling adjustments.
    """
    z_um, y_um, x_um = (float(v) for v in voxel_size_um_zyx)
    if not bool(resample_z_to_lateral):
        return (z_um, y_um, x_um)

    labels = [str(axis).strip().lower() for axis in axes]
    if "z" not in labels:
        return (z_um, y_um, x_um)

    lateral_labels = [label for label in labels if label in {"x", "y"}]
    if not lateral_labels:
        return (z_um, y_um, x_um)

    target_label = str(lateral_labels[-1])
    target_um = y_um if target_label == "y" else x_um
    if target_um <= 0.0:
        return (z_um, y_um, x_um)
    return (float(target_um), y_um, x_um)


def _resampled_axis_length(
    *,
    axis_length: int,
    source_spacing_um: float,
    target_spacing_um: float,
) -> int:
    """Return output axis length that preserves physical extent.

    Parameters
    ----------
    axis_length : int
        Source axis length.
    source_spacing_um : float
        Source pixel spacing along the axis.
    target_spacing_um : float
        Target pixel spacing along the axis.

    Returns
    -------
    int
        Target axis length.

    Raises
    ------
    None
        Invalid spacing inputs fall back to the source length.
    """
    length = max(1, int(axis_length))
    source_um = float(source_spacing_um)
    target_um = float(target_spacing_um)
    if length <= 1 or source_um <= 0.0 or target_um <= 0.0:
        return length
    physical_extent_um = float(length - 1) * source_um
    return max(1, int(np.round(physical_extent_um / target_um)) + 1)


def _resample_axis_linear_to_uint16(
    *,
    source: np.ndarray,
    destination: np.ndarray,
    axis: int,
) -> None:
    """Linearly resample one axis and write into a uint16 destination array.

    Parameters
    ----------
    source : numpy.ndarray
        Source array.
    destination : numpy.ndarray
        Destination array. Must match ``source`` rank.
    axis : int
        Axis index to resample.

    Returns
    -------
    None
        Destination is written in place.

    Raises
    ------
    ValueError
        If source and destination ranks differ.
    """
    src = np.asarray(source)
    dst = np.asarray(destination)
    if src.ndim != dst.ndim:
        raise ValueError("mip_export resampling requires matching array ranks.")

    axis_index = int(axis)
    src_len = int(src.shape[axis_index])
    dst_len = int(dst.shape[axis_index])
    src_moved = np.moveaxis(src, axis_index, 0)
    dst_moved = np.moveaxis(dst, axis_index, 0)
    src_flat = src_moved.reshape(src_len, -1)
    dst_flat = dst_moved.reshape(dst_len, -1)
    dst_requires_copy_back = not np.shares_memory(dst_flat, dst_moved)

    if dst_flat.size == 0:
        return

    if src_len <= 1:
        repeated = np.repeat(src_flat[:1, :], repeats=dst_len, axis=0)
        dst_flat[:, :] = _to_uint16(repeated)
        if dst_requires_copy_back:
            dst_moved[...] = dst_flat.reshape(dst_moved.shape)
        return

    sample_positions = np.linspace(
        0.0, float(src_len - 1), num=dst_len, dtype=np.float64
    )
    lower = np.floor(sample_positions).astype(np.int64)
    upper = np.minimum(lower + 1, src_len - 1)
    weight_upper = (sample_positions - lower).astype(np.float32)
    weight_lower = (1.0 - weight_upper).astype(np.float32)

    estimated_cols = int(
        max(
            1,
            _MAX_INTERPOLATION_BLOCK_BYTES
            // max(1, 3 * dst_len * np.dtype(np.float32).itemsize),
        )
    )
    block_cols = max(1, min(int(estimated_cols), int(src_flat.shape[1])))
    for start in range(0, int(src_flat.shape[1]), int(block_cols)):
        stop = min(int(src_flat.shape[1]), int(start + block_cols))
        lower_block = src_flat[lower, start:stop].astype(np.float32, copy=False)
        upper_block = src_flat[upper, start:stop].astype(np.float32, copy=False)
        interpolated = (
            lower_block * weight_lower[:, None] + upper_block * weight_upper[:, None]
        )
        dst_flat[:, start:stop] = _to_uint16(interpolated)

    if dst_requires_copy_back:
        dst_moved[...] = dst_flat.reshape(dst_moved.shape)


def _realized_tiff_axes(
    *,
    axes: tuple[str, ...],
    realized_ndim: int,
) -> tuple[str, ...]:
    """Align symbolic output axes to the rank realized by ``tifffile``.

    ``tifffile.memmap(...)`` can squeeze leading singleton non-spatial axes
    from OME-TIFF outputs. This occurs for ``multi_position`` exports when the
    leading ``p`` axis has length ``1``, so a planned ``(p, z, x)`` output is
    reopened as a 2D ``(z, x)`` array. Resampling must target the realized
    ``z`` axis, not the squeezed-out symbolic index.
    """
    labels = tuple(str(axis).strip().lower() for axis in axes)
    ndim = max(0, int(realized_ndim))
    if len(labels) <= ndim:
        return labels
    return labels[-ndim:]


def _resample_tiff_projection_z_axis_if_needed(
    *,
    output_path: Path,
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
    resample_z_to_lateral: bool,
) -> tuple[float, float, float]:
    """Resample TIFF projection ``z`` axis to lateral spacing when required.

    Parameters
    ----------
    output_path : pathlib.Path
        Projection TIFF path.
    axes : tuple[str, ...]
        Output axis labels.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel spacing in microns.
    resample_z_to_lateral : bool
        Whether to resample projections containing ``z``.

    Returns
    -------
    tuple[float, float, float]
        Effective voxel spacing in ``(z, y, x)`` after optional resampling.

    Raises
    ------
    Exception
        Propagates filesystem or TIFF I/O failures during rewrite.
    """
    effective_voxel_size_um_zyx = _effective_voxel_size_um_zyx_for_projection(
        axes=tuple(str(axis) for axis in axes),
        voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
        resample_z_to_lateral=bool(resample_z_to_lateral),
    )
    labels = [str(axis).strip().lower() for axis in axes]
    if "z" not in labels:
        return effective_voxel_size_um_zyx

    source_z_um = float(voxel_size_um_zyx[0])
    target_z_um = float(effective_voxel_size_um_zyx[0])
    if np.isclose(source_z_um, target_z_um, rtol=1e-6, atol=1e-6):
        return effective_voxel_size_um_zyx

    source = tifffile.memmap(str(output_path), mode="r")
    temp_path = output_path.with_name(f"{output_path.name}.tmp_resample.tif")
    try:
        source_shape = tuple(int(v) for v in np.asarray(source).shape)
        realized_axes = _realized_tiff_axes(
            axes=tuple(str(axis) for axis in axes),
            realized_ndim=len(source_shape),
        )
        realized_z_axis_index = int(realized_axes.index("z"))
        output_z_len = _resampled_axis_length(
            axis_length=int(source_shape[int(realized_z_axis_index)]),
            source_spacing_um=float(source_z_um),
            target_spacing_um=float(target_z_um),
        )
        if output_z_len == int(source_shape[int(realized_z_axis_index)]):
            del source
            return effective_voxel_size_um_zyx

        output_shape = list(source_shape)
        output_shape[int(realized_z_axis_index)] = int(output_z_len)
        if temp_path.exists():
            temp_path.unlink()
        target = tifffile.memmap(
            str(temp_path),
            shape=tuple(int(v) for v in output_shape),
            dtype=np.uint16,
            bigtiff=True,
            photometric="minisblack",
            ome=True,
            metadata=_ome_metadata_for_projection_output(
                output_shape=tuple(int(v) for v in output_shape),
                axes=tuple(str(axis) for axis in realized_axes),
                voxel_size_um_zyx=tuple(float(v) for v in effective_voxel_size_um_zyx),
                image_name=str(output_path.stem),
            ),
        )
        try:
            _resample_axis_linear_to_uint16(
                source=np.asarray(source),
                destination=target,
                axis=int(realized_z_axis_index),
            )
            target.flush()
        finally:
            del target
        del source
        output_path.unlink()
        temp_path.replace(output_path)
        return effective_voxel_size_um_zyx
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def _ome_axes_for_output_shape(output_shape: tuple[int, ...]) -> str:
    """Return OME-compatible axis string for projection outputs."""
    if len(output_shape) == 2:
        return "YX"
    if len(output_shape) == 3:
        return "QYX"
    raise ValueError("mip_export OME-TIFF output must be 2D or 3D.")


def _ome_metadata_for_projection_output(
    *,
    output_shape: tuple[int, ...],
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
    image_name: str,
) -> dict[str, Any]:
    """Build OME metadata payload for one projection output.

    Parameters
    ----------
    output_shape : tuple[int, ...]
        Projection output shape.
    axes : tuple[str, ...]
        Output axis labels.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel spacing in microns.
    image_name : str
        Image name for OME metadata.

    Returns
    -------
    dict[str, Any]
        Metadata mapping for ``tifffile`` OME serialization.
    """
    physical_size_y, physical_size_x = _projection_pixel_size_um(
        axes=axes,
        voxel_size_um_zyx=voxel_size_um_zyx,
    )
    return {
        "axes": _ome_axes_for_output_shape(tuple(int(v) for v in output_shape)),
        "Name": str(image_name),
        "PhysicalSizeX": float(physical_size_x),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": float(physical_size_y),
        "PhysicalSizeYUnit": "µm",
        "SignificantBits": 16,
    }


def _ome_zarr_axis_metadata(axis: str) -> dict[str, Any]:
    """Return one OME-Zarr axis metadata mapping."""
    token = str(axis).strip().lower()
    if token == "t":
        return {"name": "t", "type": "time"}
    if token == "c":
        return {"name": "c", "type": "channel"}
    if token in {"z", "y", "x"}:
        return {"name": token, "type": "space", "unit": "micrometer"}
    return {"name": str(axis).strip() or token or "axis"}


def _ome_zarr_scale_for_projection_output(
    *,
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
) -> list[float]:
    """Return per-axis OME-Zarr scale values for one projection output."""
    z_um, y_um, x_um = (float(value) for value in voxel_size_um_zyx)
    axis_to_scale = {"z": z_um, "y": y_um, "x": x_um}
    return [float(axis_to_scale.get(str(axis).strip().lower(), 1.0)) for axis in axes]


def _ome_zarr_metadata_for_projection_output(
    *,
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
    image_name: str,
) -> dict[str, Any]:
    """Build standalone OME-Zarr image metadata for one projection output."""
    multiscales = [
        {
            "name": str(image_name),
            "axes": [_ome_zarr_axis_metadata(axis) for axis in axes],
            "datasets": [
                {
                    "path": _OME_ZARR_DATASET_NAME,
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": _ome_zarr_scale_for_projection_output(
                                axes=axes,
                                voxel_size_um_zyx=voxel_size_um_zyx,
                            ),
                        }
                    ],
                }
            ],
        }
    ]
    return {"version": "0.5", "multiscales": multiscales}


def _initialize_projection_ome_zarr(
    *,
    output_path: Path,
    output_shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype[Any],
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float],
) -> tuple[zarr.Group, Any]:
    """Create one standalone OME-Zarr output and return its writable dataset."""
    if output_path.exists():
        try:
            shutil.rmtree(output_path)
        except Exception:
            output_path.unlink()
    root = zarr.open_group(str(output_path), mode="w")
    metadata = _ome_zarr_metadata_for_projection_output(
        axes=tuple(str(axis) for axis in axes),
        voxel_size_um_zyx=tuple(float(value) for value in voxel_size_um_zyx),
        image_name=str(output_path.stem).replace(".ome", ""),
    )
    root.attrs.update(
        {
            "ome": metadata,
            "multiscales": metadata["multiscales"],
            "axes": [str(axis) for axis in axes],
        }
    )
    target = root.create_array(
        name=_OME_ZARR_DATASET_NAME,
        shape=tuple(int(value) for value in output_shape),
        chunks=tuple(int(value) for value in chunks),
        dtype=np.dtype(dtype),
        overwrite=True,
        dimension_names=tuple(str(axis) for axis in axes),
    )
    target.attrs.update(
        {
            "axes": [str(axis) for axis in axes],
            "_ARRAY_DIMENSIONS": [str(axis) for axis in axes],
        }
    )
    return root, target


def _estimate_worker_thread_capacity(client: "Client") -> int:
    """Estimate total worker-thread capacity from Dask scheduler metadata.

    Parameters
    ----------
    client : dask.distributed.Client
        Active Dask client.

    Returns
    -------
    int
        Sum of worker thread counts, clamped to at least ``1``.
    """
    try:
        scheduler_info = client.scheduler_info()
    except Exception:
        return 1

    workers = scheduler_info.get("workers", {})
    if not isinstance(workers, dict):
        return 1

    capacity = 0
    for metadata in workers.values():
        if not isinstance(metadata, dict):
            continue
        nthreads = metadata.get("nthreads", 1)
        try:
            capacity += max(1, int(nthreads))
        except (TypeError, ValueError):
            capacity += 1
    return max(1, int(capacity))


def _resolve_output_base_directory(
    *,
    zarr_path: Union[str, Path],
    output_directory: str,
) -> Path:
    """Resolve MIP-export base directory.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Canonical analysis-store path.
    output_directory : str
        Requested output directory path, possibly empty.

    Returns
    -------
    pathlib.Path
        Resolved base directory path.
    """
    store_path = Path(zarr_path).expanduser().resolve()
    candidate = str(output_directory).strip()
    if candidate:
        resolved = Path(candidate).expanduser()
        if not resolved.is_absolute():
            resolved = (store_path.parent / resolved).resolve()
        return resolved
    return (store_path.parent / f"{store_path.name}_mip_export").resolve()


def _prepare_output_directory(base_directory: Path) -> Path:
    """Create a clean latest-output directory for this run.

    Parameters
    ----------
    base_directory : pathlib.Path
        Base output directory.

    Returns
    -------
    pathlib.Path
        Prepared ``latest`` output directory.
    """
    latest_directory = (base_directory / "latest").resolve()
    if latest_directory.exists():
        shutil.rmtree(latest_directory)
    latest_directory.mkdir(parents=True, exist_ok=True)
    return latest_directory


def _iter_export_tasks(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    position_mode: str,
) -> list[_MipExportTask]:
    """Enumerate projection-export tasks.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source data shape in canonical axis order.
    position_mode : str
        Export mode.

    Returns
    -------
    list[_MipExportTask]
        Ordered list of export tasks.
    """
    t_count = int(shape_tpczyx[0])
    p_count = int(shape_tpczyx[1])
    c_count = int(shape_tpczyx[2])
    tasks: list[_MipExportTask] = []

    if position_mode == "multi_position":
        for t_index, c_index, projection in product(
            range(t_count), range(c_count), _PROJECTIONS
        ):
            tasks.append(
                _MipExportTask(
                    projection=str(projection),
                    t_index=int(t_index),
                    c_index=int(c_index),
                    p_index=None,
                )
            )
        return tasks

    for p_index, t_index, c_index, projection in product(
        range(p_count), range(t_count), range(c_count), _PROJECTIONS
    ):
        tasks.append(
            _MipExportTask(
                projection=str(projection),
                t_index=int(t_index),
                c_index=int(c_index),
                p_index=int(p_index),
            )
        )
    return tasks


def _projection_axes(
    *,
    projection: str,
    position_mode: str,
) -> tuple[str, ...]:
    """Return axis labels for one projection output.

    Parameters
    ----------
    projection : str
        Projection identifier.
    position_mode : str
        Export mode.

    Returns
    -------
    tuple[str, ...]
        Axis labels in output array order.
    """
    if projection == "xy":
        spatial_axes = ("y", "x")
    elif projection == "xz":
        spatial_axes = ("z", "x")
    else:
        spatial_axes = ("z", "y")

    if position_mode == "multi_position":
        return ("p", *spatial_axes)
    return spatial_axes


def _projection_layout(
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    projection: str,
    position_mode: str,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return output shape and source-axis mapping for one projection.

    Parameters
    ----------
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Source shape in canonical axis order.
    projection : str
        Projection identifier.
    position_mode : str
        Export mode.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...], int]
        ``(output_shape, preserved_axes, reduction_axis)`` where
        ``preserved_axes`` are source-axis indices in output order.

    Raises
    ------
    ValueError
        If projection or position-mode values are unsupported.
    """
    _, p_count, _, z_count, y_count, x_count = shape_tpczyx
    mode = str(position_mode).strip().lower()
    proj = str(projection).strip().lower()

    if mode not in {"multi_position", "per_position"}:
        raise ValueError(
            "mip_export position_mode must be 'multi_position' or 'per_position'."
        )
    if proj not in set(_PROJECTIONS):
        raise ValueError(f"Unsupported MIP projection: {projection!r}.")

    if mode == "per_position":
        if proj == "xy":
            return ((int(y_count), int(x_count)), (4, 5), 3)
        if proj == "xz":
            return ((int(z_count), int(x_count)), (3, 5), 4)
        return ((int(z_count), int(y_count)), (3, 4), 5)

    if proj == "xy":
        return ((int(p_count), int(y_count), int(x_count)), (1, 4, 5), 3)
    if proj == "xz":
        return ((int(p_count), int(z_count), int(x_count)), (1, 3, 5), 4)
    return ((int(p_count), int(z_count), int(y_count)), (1, 3, 4), 5)


def _projection_from_single_volume(
    *,
    volume_zyx: np.ndarray,
    projection: str,
) -> np.ndarray:
    """Compute one MIP from a single-position volume.

    Parameters
    ----------
    volume_zyx : numpy.ndarray
        Volume data in ``(z, y, x)`` order.
    projection : str
        Projection identifier.

    Returns
    -------
    numpy.ndarray
        Projection array.
    """
    if projection == "xy":
        return np.max(volume_zyx, axis=0)
    if projection == "xz":
        return np.max(volume_zyx, axis=1)
    return np.max(volume_zyx, axis=2)


def _projection_from_position_stack(
    *,
    volume_pzyx: np.ndarray,
    projection: str,
) -> np.ndarray:
    """Compute one MIP from a multi-position stack.

    Parameters
    ----------
    volume_pzyx : numpy.ndarray
        Position stack in ``(p, z, y, x)`` order.
    projection : str
        Projection identifier.

    Returns
    -------
    numpy.ndarray
        Projection array preserving the ``p`` axis.
    """
    if projection == "xy":
        return np.max(volume_pzyx, axis=1)
    if projection == "xz":
        return np.max(volume_pzyx, axis=2)
    return np.max(volume_pzyx, axis=3)


def _iter_block_ranges(
    *,
    length: int,
    block_size: int,
) -> list[tuple[int, int]]:
    """Return contiguous half-open block ranges for one axis length.

    Parameters
    ----------
    length : int
        Axis length.
    block_size : int
        Requested block size.

    Returns
    -------
    list[tuple[int, int]]
        Half-open index ranges ``(start, stop)``.

    Raises
    ------
    ValueError
        If ``length`` or ``block_size`` are not positive.
    """
    axis_length = int(length)
    step = int(block_size)
    if axis_length <= 0:
        raise ValueError("mip_export requires non-empty reduction axes.")
    if step <= 0:
        raise ValueError("mip_export reduction block size must be positive.")
    return [
        (int(start), int(min(axis_length, start + step)))
        for start in range(0, axis_length, step)
    ]


def _coerce_chunk_length(chunk_value: Any) -> int:
    """Normalize one Zarr chunk-size value to a positive integer.

    Parameters
    ----------
    chunk_value : Any
        Candidate chunk-size value.

    Returns
    -------
    int
        Positive chunk length.

    Raises
    ------
    None
        Invalid values fall back to ``1``.
    """
    if isinstance(chunk_value, tuple):
        if not chunk_value:
            return 1
        chunk_value = chunk_value[0]
    try:
        parsed = int(chunk_value)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def _close_zarr_store(container: Any) -> None:
    """Close a Zarr store handle when the backing store supports it.

    Parameters
    ----------
    container : Any
        Zarr group/array-like object exposing a ``store`` attribute.

    Returns
    -------
    None
        Store closure is best-effort.

    Raises
    ------
    None
        Errors during closure are intentionally ignored.
    """
    store = getattr(container, "store", None)
    close_method = getattr(store, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            return


def _choose_reduction_block_size(
    *,
    source_array: Any,
    reduction_axis: int,
    preserved_shape: tuple[int, ...],
    max_read_bytes: int = _MAX_REDUCTION_READ_BYTES,
) -> int:
    """Choose a safe read block size for chunkwise projection reduction.

    Parameters
    ----------
    source_array : Any
        Source Zarr-like array exposing ``chunks`` and ``dtype`` attributes.
    reduction_axis : int
        Axis index in ``source_array`` reduced by max-projection.
    preserved_shape : tuple[int, ...]
        Shape of output-preserved axes (not reduced).
    max_read_bytes : int, default=268435456
        Maximum in-memory byte budget for one read block.

    Returns
    -------
    int
        Positive reduction block size.

    Raises
    ------
    None
        Missing metadata falls back to conservative defaults.
    """
    preferred_block = 1
    chunks = getattr(source_array, "chunks", None)
    if isinstance(chunks, tuple) and len(chunks) > int(reduction_axis):
        preferred_block = _coerce_chunk_length(chunks[int(reduction_axis)])

    try:
        source_dtype = np.dtype(getattr(source_array, "dtype"))
    except Exception:
        source_dtype = np.dtype(np.float32)

    try:
        preserved_elements = int(np.prod(np.asarray(preserved_shape, dtype=np.int64)))
    except Exception:
        preserved_elements = 0
    if preserved_elements <= 0:
        return max(1, int(preferred_block))

    bytes_per_slice = preserved_elements * max(1, int(source_dtype.itemsize))
    if bytes_per_slice <= 0:
        return max(1, int(preferred_block))

    max_slices = max(1, int(max_read_bytes // bytes_per_slice))
    return max(1, min(int(preferred_block), int(max_slices)))


def _choose_preserved_tile_shape(
    *,
    source_array: Any,
    preserved_axes: tuple[int, ...],
    preserved_shape: tuple[int, ...],
    reduction_block_size: int,
    max_read_bytes: int = _MAX_REDUCTION_READ_BYTES,
) -> tuple[int, ...]:
    """Choose tile sizes for preserved output axes.

    Parameters
    ----------
    source_array : Any
        Source array exposing ``chunks`` and ``dtype``.
    preserved_axes : tuple[int, ...]
        Source-axis indices preserved in the projection output.
    preserved_shape : tuple[int, ...]
        Output shape for preserved axes.
    reduction_block_size : int
        Selected reduction-axis block size.
    max_read_bytes : int, default=134217728
        Maximum in-memory byte budget for one source read.

    Returns
    -------
    tuple[int, ...]
        Tile lengths for each preserved axis in output order.

    Raises
    ------
    ValueError
        If preserved-axis metadata is inconsistent.
    """
    if len(preserved_axes) != len(preserved_shape):
        raise ValueError(
            "mip_export preserved-axis metadata is inconsistent with output shape."
        )

    chunks = getattr(source_array, "chunks", None)
    tile_lengths: list[int] = []
    for axis_index, axis_length in zip(preserved_axes, preserved_shape):
        axis_size = max(1, int(axis_length))
        preferred = 1
        if isinstance(chunks, tuple) and len(chunks) > int(axis_index):
            preferred = _coerce_chunk_length(chunks[int(axis_index)])
        tile_lengths.append(max(1, min(axis_size, preferred)))

    try:
        source_dtype = np.dtype(getattr(source_array, "dtype"))
    except Exception:
        source_dtype = np.dtype(np.float32)
    itemsize = max(1, int(source_dtype.itemsize))

    read_bytes = int(
        int(max(1, int(reduction_block_size)))
        * int(np.prod(np.asarray(tile_lengths, dtype=np.int64)))
        * itemsize
    )
    budget = max(1, int(max_read_bytes))
    while read_bytes > budget:
        largest_axis = int(np.argmax(np.asarray(tile_lengths, dtype=np.int64)))
        if tile_lengths[largest_axis] <= 1:
            break
        tile_lengths[largest_axis] = max(1, tile_lengths[largest_axis] // 2)
        read_bytes = int(
            int(max(1, int(reduction_block_size)))
            * int(np.prod(np.asarray(tile_lengths, dtype=np.int64)))
            * itemsize
        )
    return tuple(int(value) for value in tile_lengths)


def _iter_tile_slices(
    *,
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
) -> list[tuple[slice, ...]]:
    """Enumerate output tile slices for a projection array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Output array shape.
    tile_shape : tuple[int, ...]
        Tile lengths for each output axis.

    Returns
    -------
    list[tuple[slice, ...]]
        Output tile slices in row-major order.

    Raises
    ------
    ValueError
        If shape and tile metadata lengths do not match.
    """
    if len(shape) != len(tile_shape):
        raise ValueError("mip_export tile metadata is inconsistent with shape.")
    axis_ranges = [
        _iter_block_ranges(length=int(axis_length), block_size=int(axis_tile))
        for axis_length, axis_tile in zip(shape, tile_shape)
    ]
    return [
        tuple(slice(int(start), int(stop)) for start, stop in range_tuple)
        for range_tuple in product(*axis_ranges)
    ]


def _tile_slices_to_bounds(
    tile_slices: tuple[slice, ...],
) -> tuple[tuple[int, int], ...]:
    """Convert tile slices into serializable integer bounds.

    Parameters
    ----------
    tile_slices : tuple[slice, ...]
        Output tile slices.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Tile bounds as ``((start, stop), ...)``.
    """
    return tuple(
        (int(tile_slice.start), int(tile_slice.stop)) for tile_slice in tile_slices
    )


def _tile_bounds_to_slices(
    tile_bounds: tuple[tuple[int, int], ...],
) -> tuple[slice, ...]:
    """Convert serialized tile bounds back into slices.

    Parameters
    ----------
    tile_bounds : tuple[tuple[int, int], ...]
        Tile bounds as ``((start, stop), ...)``.

    Returns
    -------
    tuple[slice, ...]
        Tile slices.
    """
    return tuple(slice(int(start), int(stop)) for start, stop in tile_bounds)


def _max_reduce_into(
    current: Optional[np.ndarray],
    update: np.ndarray,
) -> np.ndarray:
    """Accumulate max-projection updates into an output array.

    Parameters
    ----------
    current : numpy.ndarray, optional
        Existing reduction output.
    update : numpy.ndarray
        New partial reduction result.

    Returns
    -------
    numpy.ndarray
        Updated reduction output.

    Raises
    ------
    None
        Input arrays are coerced to NumPy arrays internally.
    """
    update_arr = np.asarray(update)
    if current is None:
        return update_arr
    np.maximum(current, update_arr, out=current)
    return current


def _plan_projection_output(
    *,
    task_index: int,
    task: _MipExportTask,
    source_array: Any,
    output_directory: Path,
    export_format: str,
    position_mode: str,
) -> _PlannedOutput:
    """Build execution metadata for one projection output.

    Parameters
    ----------
    task_index : int
        Stable projection-task index.
    task : _MipExportTask
        Projection export task.
    source_array : Any
        Source array in canonical ``(t, p, c, z, y, x)`` order.
    output_directory : pathlib.Path
        Base output directory for this run.
    export_format : str
        Export format identifier.
    position_mode : str
        Position mode identifier.

    Returns
    -------
    _PlannedOutput
        Planned output metadata.
    """
    axes = _projection_axes(
        projection=str(task.projection),
        position_mode=str(position_mode),
    )
    source_shape_tpczyx = tuple(int(value) for value in source_array.shape)
    output_shape, preserved_axes, reduction_axis = _projection_layout(
        shape_tpczyx=source_shape_tpczyx,
        projection=str(task.projection),
        position_mode=str(position_mode),
    )
    reduction_block_size = _choose_reduction_block_size(
        source_array=source_array,
        reduction_axis=int(reduction_axis),
        preserved_shape=tuple(int(v) for v in output_shape),
    )
    tile_shape = _choose_preserved_tile_shape(
        source_array=source_array,
        preserved_axes=tuple(int(v) for v in preserved_axes),
        preserved_shape=tuple(int(v) for v in output_shape),
        reduction_block_size=int(reduction_block_size),
    )
    output_tiles = tuple(
        _iter_tile_slices(
            shape=tuple(int(v) for v in output_shape),
            tile_shape=tuple(int(v) for v in tile_shape),
        )
    )
    output_path = _build_output_path(
        output_directory=Path(output_directory),
        export_format=str(export_format),
        task=task,
    )
    return _PlannedOutput(
        task_index=int(task_index),
        task=task,
        axes=tuple(str(axis) for axis in axes),
        output_shape=tuple(int(v) for v in output_shape),
        preserved_axes=tuple(int(v) for v in preserved_axes),
        reduction_axis=int(reduction_axis),
        reduction_block_size=int(reduction_block_size),
        tile_shape=tuple(int(v) for v in tile_shape),
        output_tiles=output_tiles,
        output_path=output_path,
    )


def _run_export_tile_task(
    *,
    zarr_path: str,
    source_component: str,
    position_mode: str,
    task_index: int,
    task: _MipExportTask,
    tile_bounds: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    """Compute one projection tile for a projection-export task.

    Parameters
    ----------
    zarr_path : str
        Canonical analysis-store path.
    source_component : str
        Source data component path.
    position_mode : str
        Position mode.
    task_index : int
        Stable projection-task index.
    task : _MipExportTask
        Projection task for this tile.
    tile_bounds : tuple[tuple[int, int], ...]
        Tile bounds as ``((start, stop), ...)``.

    Returns
    -------
    dict[str, Any]
        Tile payload metadata including task index and output bounds.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        source_array = root[source_component]
        source_shape_tpczyx = tuple(int(value) for value in source_array.shape)
        output_shape, preserved_axes, reduction_axis = _projection_layout(
            shape_tpczyx=source_shape_tpczyx,
            projection=str(task.projection),
            position_mode=str(position_mode),
        )
        output_slices = _tile_bounds_to_slices(tile_bounds)
        tile_preserved_shape = tuple(
            max(1, int(tile_slice.stop) - int(tile_slice.start))
            for tile_slice in output_slices
        )
        # DO NOT REGRESS: size reduction blocks from tile shape, not full output
        # shape. Full-shape sizing can collapse to tiny blocks and stall throughput.
        reduction_block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=int(reduction_axis),
            preserved_shape=tuple(int(v) for v in tile_preserved_shape),
        )
        reduction_ranges = _iter_block_ranges(
            length=int(source_shape_tpczyx[int(reduction_axis)]),
            block_size=int(reduction_block_size),
        )
        fixed_indices: dict[int, int] = {
            0: int(task.t_index),
            2: int(task.c_index),
        }
        if task.p_index is not None:
            fixed_indices[1] = int(task.p_index)

        chunk_slice_axes = tuple(
            sorted(
                (
                    int(reduction_axis),
                    *tuple(int(axis) for axis in preserved_axes),
                )
            )
        )
        reduce_axis_in_chunk = int(chunk_slice_axes.index(int(reduction_axis)))

        reduced_tile: Optional[np.ndarray] = None
        for red_start, red_stop in reduction_ranges:
            source_selection: list[object] = [slice(None)] * 6
            for axis_index, fixed_value in fixed_indices.items():
                source_selection[int(axis_index)] = int(fixed_value)
            source_selection[int(reduction_axis)] = slice(int(red_start), int(red_stop))
            for source_axis, output_slice in zip(preserved_axes, output_slices):
                source_selection[int(source_axis)] = slice(
                    int(output_slice.start),
                    int(output_slice.stop),
                )
            source_chunk = np.asarray(source_array[tuple(source_selection)])
            reduced_tile = _max_reduce_into(
                reduced_tile,
                np.max(source_chunk, axis=int(reduce_axis_in_chunk)),
            )

        return {
            "task_index": int(task_index),
            "tile_bounds": [tuple(int(v) for v in bounds) for bounds in tile_bounds],
            "tile": np.asarray(reduced_tile),
        }
    finally:
        _close_zarr_store(root)


def _chunkwise_projection_from_single_position(
    *,
    source_array: Any,
    t_index: int,
    p_index: int,
    c_index: int,
    projection: str,
) -> np.ndarray:
    """Compute one projection from a single position via chunkwise reads.

    Parameters
    ----------
    source_array : Any
        Canonical source data array in ``(t, p, c, z, y, x)`` order.
    t_index : int
        Time index.
    p_index : int
        Position index.
    c_index : int
        Channel index.
    projection : str
        Projection identifier (``xy``, ``xz``, or ``yz``).

    Returns
    -------
    numpy.ndarray
        Projection result.

    Raises
    ------
    ValueError
        If projection value is unsupported.
    """
    shape_tpczyx = tuple(int(value) for value in source_array.shape)
    _, _, _, z_count, y_count, x_count = shape_tpczyx

    if projection == "xy":
        block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=3,
            preserved_shape=(y_count, x_count),
        )
        reduced: Optional[np.ndarray] = None
        for start, stop in _iter_block_ranges(length=z_count, block_size=block_size):
            chunk = np.asarray(
                source_array[
                    int(t_index),
                    int(p_index),
                    int(c_index),
                    int(start) : int(stop),
                    :,
                    :,
                ]
            )
            reduced = _max_reduce_into(reduced, np.max(chunk, axis=0))
        return np.asarray(reduced)

    if projection == "xz":
        block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=4,
            preserved_shape=(z_count, x_count),
        )
        reduced = None
        for start, stop in _iter_block_ranges(length=y_count, block_size=block_size):
            chunk = np.asarray(
                source_array[
                    int(t_index),
                    int(p_index),
                    int(c_index),
                    :,
                    int(start) : int(stop),
                    :,
                ]
            )
            reduced = _max_reduce_into(reduced, np.max(chunk, axis=1))
        return np.asarray(reduced)

    if projection == "yz":
        block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=5,
            preserved_shape=(z_count, y_count),
        )
        reduced = None
        for start, stop in _iter_block_ranges(length=x_count, block_size=block_size):
            chunk = np.asarray(
                source_array[
                    int(t_index),
                    int(p_index),
                    int(c_index),
                    :,
                    :,
                    int(start) : int(stop),
                ]
            )
            reduced = _max_reduce_into(reduced, np.max(chunk, axis=2))
        return np.asarray(reduced)

    raise ValueError(f"Unsupported MIP projection: {projection!r}.")


def _chunkwise_projection_from_multi_position(
    *,
    source_array: Any,
    t_index: int,
    c_index: int,
    projection: str,
) -> np.ndarray:
    """Compute one projection preserving all positions via chunkwise reads.

    Parameters
    ----------
    source_array : Any
        Canonical source data array in ``(t, p, c, z, y, x)`` order.
    t_index : int
        Time index.
    c_index : int
        Channel index.
    projection : str
        Projection identifier (``xy``, ``xz``, or ``yz``).

    Returns
    -------
    numpy.ndarray
        Projection result preserving the ``p`` axis.

    Raises
    ------
    ValueError
        If projection value is unsupported.
    """
    shape_tpczyx = tuple(int(value) for value in source_array.shape)
    _, p_count, _, z_count, y_count, x_count = shape_tpczyx

    if projection == "xy":
        block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=3,
            preserved_shape=(p_count, y_count, x_count),
        )
        reduced: Optional[np.ndarray] = None
        for start, stop in _iter_block_ranges(length=z_count, block_size=block_size):
            chunk = np.asarray(
                source_array[
                    int(t_index),
                    :,
                    int(c_index),
                    int(start) : int(stop),
                    :,
                    :,
                ]
            )
            reduced = _max_reduce_into(reduced, np.max(chunk, axis=1))
        return np.asarray(reduced)

    if projection == "xz":
        block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=4,
            preserved_shape=(p_count, z_count, x_count),
        )
        reduced = None
        for start, stop in _iter_block_ranges(length=y_count, block_size=block_size):
            chunk = np.asarray(
                source_array[
                    int(t_index),
                    :,
                    int(c_index),
                    :,
                    int(start) : int(stop),
                    :,
                ]
            )
            reduced = _max_reduce_into(reduced, np.max(chunk, axis=2))
        return np.asarray(reduced)

    if projection == "yz":
        block_size = _choose_reduction_block_size(
            source_array=source_array,
            reduction_axis=5,
            preserved_shape=(p_count, z_count, y_count),
        )
        reduced = None
        for start, stop in _iter_block_ranges(length=x_count, block_size=block_size):
            chunk = np.asarray(
                source_array[
                    int(t_index),
                    :,
                    int(c_index),
                    :,
                    :,
                    int(start) : int(stop),
                ]
            )
            reduced = _max_reduce_into(reduced, np.max(chunk, axis=3))
        return np.asarray(reduced)

    raise ValueError(f"Unsupported MIP projection: {projection!r}.")


def _to_uint16(data: np.ndarray) -> np.ndarray:
    """Convert projection data to TIFF-compatible uint16."""
    clipped = np.clip(np.asarray(data, dtype=np.float64), 0.0, 65535.0)
    return np.rint(clipped).astype(np.uint16, copy=False)


def _default_projection_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose conservative chunk sizes for projection Zarr files."""
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        return (1, int(shape[1]), int(shape[2]))
    return tuple(int(value) for value in shape)


def _build_output_path(
    *,
    output_directory: Path,
    export_format: str,
    task: _MipExportTask,
) -> Path:
    """Build output file path for one export task.

    Parameters
    ----------
    output_directory : pathlib.Path
        Base output directory for this run.
    export_format : str
        Export file format.
    task : _MipExportTask
        Projection task.

    Returns
    -------
    pathlib.Path
        Output file path.
    """
    suffix = ".tif" if _is_tiff_export_format(str(export_format)) else ".ome.zarr"
    if task.p_index is None:
        filename = (
            f"mip_{task.projection}_t{int(task.t_index):04d}_c{int(task.c_index):04d}"
            f"{suffix}"
        )
    else:
        filename = (
            f"mip_{task.projection}_p{int(task.p_index):04d}_t{int(task.t_index):04d}"
            f"_c{int(task.c_index):04d}{suffix}"
        )
    return (output_directory / filename).resolve()


def _write_projection_output(
    *,
    output_path: Path,
    projection: np.ndarray,
    export_format: str,
    axes: tuple[str, ...],
    voxel_size_um_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """Persist one projection output file.

    Parameters
    ----------
    output_path : pathlib.Path
        Output file path.
    projection : numpy.ndarray
        Projection data.
    export_format : str
        Export format.
    axes : tuple[str, ...]
        Axis labels for the output data.
    voxel_size_um_zyx : tuple[float, float, float], default=(1.0, 1.0, 1.0)
        Source voxel spacing used for OME-TIFF physical pixel calibration.

    Returns
    -------
    str
        Stored dtype name.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_tiff_export_format(export_format):
        payload = _to_uint16(np.asarray(projection))
        tifffile.imwrite(
            str(output_path),
            payload,
            photometric="minisblack",
            ome=True,
            metadata=_ome_metadata_for_projection_output(
                output_shape=tuple(int(v) for v in payload.shape),
                axes=tuple(str(axis) for axis in axes),
                voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
                image_name=str(output_path.stem),
            ),
        )
        return str(payload.dtype)

    if output_path.exists():
        shutil.rmtree(output_path)
    root, target = _initialize_projection_ome_zarr(
        output_path=output_path,
        output_shape=tuple(int(v) for v in np.asarray(projection).shape),
        chunks=_default_projection_chunks(
            tuple(int(v) for v in np.asarray(projection).shape)
        ),
        dtype=np.asarray(projection).dtype,
        axes=tuple(str(axis) for axis in axes),
        voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
    )
    try:
        payload = np.asarray(projection)
        target[...] = payload
        return str(payload.dtype)
    finally:
        _close_zarr_store(root)


def _run_export_task(
    *,
    zarr_path: str,
    source_component: str,
    output_directory: str,
    export_format: str,
    position_mode: str,
    task: _MipExportTask,
    voxel_size_um_zyx: tuple[float, float, float],
    resample_z_to_lateral: bool,
) -> dict[str, Any]:
    """Execute one projection task and write output.

    Parameters
    ----------
    zarr_path : str
        Canonical analysis-store path.
    source_component : str
        Source data component.
    output_directory : str
        Export directory path.
    export_format : str
        Export file format.
    position_mode : str
        Export mode.
    task : _MipExportTask
        Projection task to execute.
    voxel_size_um_zyx : tuple[float, float, float]
        Source voxel spacing in ``(z, y, x)`` order.
    resample_z_to_lateral : bool
        Whether to resample projection ``z`` axis to in-plane lateral spacing.

    Returns
    -------
    dict[str, Any]
        Result metadata for manifest/provenance.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    output_root: Optional[Any] = None
    try:
        source_array = root[source_component]
        axes = _projection_axes(
            projection=str(task.projection),
            position_mode=str(position_mode),
        )
        source_shape_tpczyx = tuple(int(value) for value in source_array.shape)
        output_shape, preserved_axes, reduction_axis = _projection_layout(
            shape_tpczyx=source_shape_tpczyx,
            projection=str(task.projection),
            position_mode=str(position_mode),
        )
        tile_shape = _choose_preserved_tile_shape(
            source_array=source_array,
            preserved_axes=tuple(int(v) for v in preserved_axes),
            preserved_shape=tuple(int(v) for v in output_shape),
            reduction_block_size=_coerce_chunk_length(
                getattr(source_array, "chunks", (1, 1, 1, 1, 1, 1))[int(reduction_axis)]
            ),
        )
        output_tiles = _iter_tile_slices(
            shape=tuple(int(v) for v in output_shape),
            tile_shape=tuple(int(v) for v in tile_shape),
        )
        fixed_indices: dict[int, int] = {
            0: int(task.t_index),
            2: int(task.c_index),
        }
        if task.p_index is not None:
            fixed_indices[1] = int(task.p_index)

        chunk_slice_axes = tuple(
            sorted(
                (
                    int(reduction_axis),
                    *tuple(int(axis) for axis in preserved_axes),
                )
            )
        )
        reduce_axis_in_chunk = int(chunk_slice_axes.index(int(reduction_axis)))

        output_path = _build_output_path(
            output_directory=Path(output_directory),
            export_format=str(export_format),
            task=task,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if _is_tiff_export_format(export_format):
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    shutil.rmtree(output_path)
            output_target = tifffile.memmap(
                str(output_path),
                shape=tuple(int(v) for v in output_shape),
                dtype=np.uint16,
                bigtiff=True,
                photometric="minisblack",
                ome=True,
                metadata=_ome_metadata_for_projection_output(
                    output_shape=tuple(int(v) for v in output_shape),
                    axes=tuple(str(axis) for axis in axes),
                    voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
                    image_name=str(output_path.stem),
                ),
            )
        else:
            output_root, output_target = _initialize_projection_ome_zarr(
                output_path=output_path,
                output_shape=tuple(int(v) for v in output_shape),
                chunks=_default_projection_chunks(tuple(int(v) for v in output_shape)),
                dtype=np.dtype(getattr(source_array, "dtype")),
                axes=tuple(str(axis) for axis in axes),
                voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
            )

        for output_slices in output_tiles:
            tile_preserved_shape = tuple(
                max(1, int(output_slice.stop) - int(output_slice.start))
                for output_slice in output_slices
            )
            # DO NOT REGRESS: tile-local preserved shape is required for stable
            # reduction block sizing and prevents apparent distributed stalls.
            reduction_block_size = _choose_reduction_block_size(
                source_array=source_array,
                reduction_axis=int(reduction_axis),
                preserved_shape=tuple(int(v) for v in tile_preserved_shape),
            )
            reduction_ranges = _iter_block_ranges(
                length=int(source_shape_tpczyx[int(reduction_axis)]),
                block_size=int(reduction_block_size),
            )
            reduced_tile: Optional[np.ndarray] = None
            for red_start, red_stop in reduction_ranges:
                source_selection: list[object] = [slice(None)] * 6
                for axis_index, fixed_value in fixed_indices.items():
                    source_selection[int(axis_index)] = int(fixed_value)
                source_selection[int(reduction_axis)] = slice(
                    int(red_start), int(red_stop)
                )
                for source_axis, output_slice in zip(preserved_axes, output_slices):
                    source_selection[int(source_axis)] = slice(
                        int(output_slice.start),
                        int(output_slice.stop),
                    )

                source_chunk = np.asarray(source_array[tuple(source_selection)])
                reduced_tile = _max_reduce_into(
                    reduced_tile,
                    np.max(source_chunk, axis=int(reduce_axis_in_chunk)),
                )

            tile_payload = np.asarray(reduced_tile)
            if _is_tiff_export_format(export_format):
                output_target[output_slices] = _to_uint16(tile_payload)
            else:
                output_target[output_slices] = tile_payload

        if _is_tiff_export_format(export_format):
            output_target.flush()
            effective_voxel_size_um_zyx = _resample_tiff_projection_z_axis_if_needed(
                output_path=output_path,
                axes=tuple(str(axis) for axis in axes),
                voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
                resample_z_to_lateral=bool(resample_z_to_lateral),
            )
            stored_dtype = str(np.dtype(np.uint16))
        else:
            stored_dtype = str(
                np.dtype(getattr(output_target, "dtype", source_array.dtype))
            )
            effective_voxel_size_um_zyx = tuple(float(v) for v in voxel_size_um_zyx)

        physical_size_y_um, physical_size_x_um = _projection_pixel_size_um(
            axes=tuple(str(axis) for axis in axes),
            voxel_size_um_zyx=tuple(float(v) for v in effective_voxel_size_um_zyx),
        )
        return {
            "path": str(output_path),
            "projection": str(task.projection),
            "t_index": int(task.t_index),
            "c_index": int(task.c_index),
            "p_index": int(task.p_index) if task.p_index is not None else None,
            "axes": list(axes),
            "dtype": str(stored_dtype),
            "physical_size_y_um": float(physical_size_y_um),
            "physical_size_x_um": float(physical_size_x_um),
        }
    finally:
        if output_root is not None:
            _close_zarr_store(output_root)
        _close_zarr_store(root)


def run_mip_export_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> MipExportSummary:
    """Run MIP export and persist latest metadata in the analysis store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        MIP-export parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    MipExportSummary
        Summary including output component, mode, and file counts.

    Raises
    ------
    ValueError
        If source component is missing or shape is incompatible.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_parameters(parameters)
    root = zarr.open_group(str(zarr_path), mode="r")
    source_component = str(normalized.get("input_source", "data")).strip() or "data"
    try:
        source_array = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"mip_export input component '{source_component}' was not found in {zarr_path}."
        ) from exc

    source_shape_tpczyx = tuple(int(value) for value in source_array.shape)
    if len(source_shape_tpczyx) != 6:
        raise ValueError(
            "mip_export requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )
    voxel_size_um_zyx = _extract_voxel_size_um_zyx(
        root=root,
        source_component=source_component,
    )

    output_base_directory = _resolve_output_base_directory(
        zarr_path=zarr_path,
        output_directory=str(normalized.get("output_directory", "")),
    )
    output_directory = _prepare_output_directory(output_base_directory)
    position_mode = str(normalized["position_mode"])
    export_format = str(normalized["export_format"])
    tasks = _iter_export_tasks(
        shape_tpczyx=source_shape_tpczyx,
        position_mode=position_mode,
    )
    total = int(len(tasks))

    _emit(5, f"Preparing MIP-export output directory: {output_directory}")
    _emit(10, f"Prepared {total} projection export tasks")

    task_results: list[dict[str, Any]] = []
    try:
        if client is None:
            delayed_tasks = [
                delayed(_run_export_task)(
                    zarr_path=str(zarr_path),
                    source_component=source_component,
                    output_directory=str(output_directory),
                    export_format=export_format,
                    position_mode=position_mode,
                    task=task,
                    voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
                    resample_z_to_lateral=bool(
                        normalized.get("resample_z_to_lateral", True)
                    ),
                )
                for task in tasks
            ]
            _emit(15, "Running MIP-export tasks with local process scheduler")
            if delayed_tasks:
                computed = dask.compute(*delayed_tasks, scheduler="processes")
                task_results = [dict(item) for item in computed]
            _emit(95, "Completed MIP-export task execution")
        else:
            from dask.distributed import as_completed

            thread_capacity = _estimate_worker_thread_capacity(client)
            max_in_flight = max(64, int(thread_capacity) * 2)

            planned_outputs = [
                _plan_projection_output(
                    task_index=int(task_index),
                    task=task,
                    source_array=source_array,
                    output_directory=Path(output_directory),
                    export_format=export_format,
                    position_mode=position_mode,
                )
                for task_index, task in enumerate(tasks)
            ]
            tile_work_items: list[tuple[int, tuple[tuple[int, int], ...]]] = []
            tiff_targets: dict[int, Any] = {}
            stored_dtypes: dict[int, str] = {}

            for planned in planned_outputs:
                planned.output_path.parent.mkdir(parents=True, exist_ok=True)
                if _is_tiff_export_format(export_format):
                    if planned.output_path.exists():
                        try:
                            planned.output_path.unlink()
                        except Exception:
                            shutil.rmtree(planned.output_path)
                    tiff_targets[int(planned.task_index)] = tifffile.memmap(
                        str(planned.output_path),
                        shape=tuple(int(v) for v in planned.output_shape),
                        dtype=np.uint16,
                        bigtiff=True,
                        photometric="minisblack",
                        ome=True,
                        metadata=_ome_metadata_for_projection_output(
                            output_shape=tuple(int(v) for v in planned.output_shape),
                            axes=tuple(str(axis) for axis in planned.axes),
                            voxel_size_um_zyx=tuple(
                                float(v) for v in voxel_size_um_zyx
                            ),
                            image_name=str(planned.output_path.stem),
                        ),
                    )
                    stored_dtypes[int(planned.task_index)] = str(np.dtype(np.uint16))
                else:
                    output_root, output_target = _initialize_projection_ome_zarr(
                        output_path=planned.output_path,
                        output_shape=tuple(int(v) for v in planned.output_shape),
                        chunks=tuple(
                            max(1, min(int(length), int(chunk)))
                            for length, chunk in zip(
                                planned.output_shape, planned.tile_shape
                            )
                        ),
                        dtype=np.dtype(getattr(source_array, "dtype")),
                        axes=tuple(str(axis) for axis in planned.axes),
                        voxel_size_um_zyx=tuple(float(v) for v in voxel_size_um_zyx),
                    )
                    try:
                        stored_dtypes[int(planned.task_index)] = str(
                            np.dtype(
                                getattr(output_target, "dtype", source_array.dtype)
                            )
                        )
                    finally:
                        _close_zarr_store(output_root)

                tile_work_items.extend(
                    [
                        (
                            int(planned.task_index),
                            _tile_slices_to_bounds(tile_slices),
                        )
                        for tile_slices in planned.output_tiles
                    ]
                )
            total_tile_tasks = int(len(tile_work_items))
            _emit(
                15,
                "Submitting MIP-export tile tasks to Dask client "
                f"(outputs={total}, tiles={total_tile_tasks}, "
                f"worker_threads={thread_capacity}, max_in_flight={max_in_flight})",
            )
            completed_tiles = 0
            pending_count = 0
            completion_queue = as_completed()

            for task_index, tile_bounds in tile_work_items:
                planned = planned_outputs[int(task_index)]
                completion_queue.add(
                    client.submit(
                        _run_export_tile_task,
                        zarr_path=str(zarr_path),
                        source_component=source_component,
                        position_mode=position_mode,
                        task_index=int(planned.task_index),
                        task=planned.task,
                        tile_bounds=tile_bounds,
                        pure=False,
                    )
                )
                pending_count += 1
                if pending_count < max_in_flight:
                    continue
                completed_future = next(completion_queue)
                tile_result = dict(completed_future.result())
                result_task_index = int(tile_result["task_index"])
                tile_payload = np.asarray(tile_result["tile"])
                tile_slices = _tile_bounds_to_slices(
                    tuple(
                        (int(bounds[0]), int(bounds[1]))
                        for bounds in tile_result["tile_bounds"]
                    )
                )
                if _is_tiff_export_format(export_format):
                    tiff_targets[int(result_task_index)][tile_slices] = _to_uint16(
                        tile_payload
                    )
                else:
                    planned = planned_outputs[int(result_task_index)]
                    output_root = zarr.open_group(str(planned.output_path), mode="a")
                    try:
                        output_root[_OME_ZARR_DATASET_NAME][tile_slices] = tile_payload
                    finally:
                        _close_zarr_store(output_root)
                pending_count -= 1
                completed_tiles += 1
                progress = 15 + int((completed_tiles / max(1, total_tile_tasks)) * 80)
                _emit(progress, f"Exported tile {completed_tiles}/{total_tile_tasks}")

            for completed_future in completion_queue:
                tile_result = dict(completed_future.result())
                result_task_index = int(tile_result["task_index"])
                tile_payload = np.asarray(tile_result["tile"])
                tile_slices = _tile_bounds_to_slices(
                    tuple(
                        (int(bounds[0]), int(bounds[1]))
                        for bounds in tile_result["tile_bounds"]
                    )
                )
                if _is_tiff_export_format(export_format):
                    tiff_targets[int(result_task_index)][tile_slices] = _to_uint16(
                        tile_payload
                    )
                else:
                    planned = planned_outputs[int(result_task_index)]
                    output_root = zarr.open_group(str(planned.output_path), mode="a")
                    try:
                        output_root[_OME_ZARR_DATASET_NAME][tile_slices] = tile_payload
                    finally:
                        _close_zarr_store(output_root)
                completed_tiles += 1
                progress = 15 + int((completed_tiles / max(1, total_tile_tasks)) * 80)
                _emit(progress, f"Exported tile {completed_tiles}/{total_tile_tasks}")

            for planned in planned_outputs:
                task = planned.task
                effective_voxel_size_um_zyx = tuple(float(v) for v in voxel_size_um_zyx)
                if _is_tiff_export_format(export_format):
                    tiff_targets[int(planned.task_index)].flush()
                    effective_voxel_size_um_zyx = (
                        _resample_tiff_projection_z_axis_if_needed(
                            output_path=Path(planned.output_path),
                            axes=tuple(str(axis) for axis in planned.axes),
                            voxel_size_um_zyx=tuple(
                                float(v) for v in voxel_size_um_zyx
                            ),
                            resample_z_to_lateral=bool(
                                normalized.get("resample_z_to_lateral", True)
                            ),
                        )
                    )
                    physical_size_y_um, physical_size_x_um = _projection_pixel_size_um(
                        axes=tuple(str(axis) for axis in planned.axes),
                        voxel_size_um_zyx=tuple(
                            float(v) for v in effective_voxel_size_um_zyx
                        ),
                    )
                else:
                    physical_size_y_um, physical_size_x_um = _projection_pixel_size_um(
                        axes=tuple(str(axis) for axis in planned.axes),
                        voxel_size_um_zyx=tuple(
                            float(v) for v in effective_voxel_size_um_zyx
                        ),
                    )
                task_results.append(
                    {
                        "path": str(planned.output_path),
                        "projection": str(task.projection),
                        "t_index": int(task.t_index),
                        "c_index": int(task.c_index),
                        "p_index": (
                            int(task.p_index) if task.p_index is not None else None
                        ),
                        "axes": [str(axis) for axis in planned.axes],
                        "dtype": str(stored_dtypes[int(planned.task_index)]),
                        "physical_size_y_um": float(physical_size_y_um),
                        "physical_size_x_um": float(physical_size_x_um),
                    }
                )
    finally:
        _close_zarr_store(root)
    task_results = sorted(task_results, key=lambda item: str(item.get("path", "")))
    exported_files = int(len(task_results))

    component = analysis_auxiliary_root("mip_export")
    root_w = zarr.open_group(str(zarr_path), mode="a")
    try:
        if component in root_w:
            del root_w[component]
        latest_group = root_w.require_group(component)
        latest_group.attrs.update(
            {
                "storage_policy": "latest_only",
                "source_component": source_component,
                "export_format": export_format,
                "position_mode": position_mode,
                "output_directory": str(output_directory),
                "task_count": total,
                "exported_files": exported_files,
                "projections": [str(value) for value in _PROJECTIONS],
                "voxel_size_um_zyx": [float(v) for v in voxel_size_um_zyx],
                "parameters": {
                    str(key): value for key, value in dict(normalized).items()
                },
                "manifest_preview": [
                    str(item.get("path", "")) for item in task_results[:128]
                ],
            }
        )
    finally:
        _close_zarr_store(root_w)

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="mip_export",
        component=component,
        metadata={
            "source_component": source_component,
            "export_format": export_format,
            "position_mode": position_mode,
            "output_directory": str(output_directory),
            "task_count": total,
            "exported_files": exported_files,
            "projections": [str(value) for value in _PROJECTIONS],
            "voxel_size_um_zyx": [float(v) for v in voxel_size_um_zyx],
        },
    )
    _emit(100, "MIP export complete")
    return MipExportSummary(
        component=component,
        source_component=source_component,
        output_directory=str(output_directory),
        export_format=export_format,
        position_mode=position_mode,
        task_count=total,
        exported_files=exported_files,
        projections=tuple(str(value) for value in _PROJECTIONS),
    )
