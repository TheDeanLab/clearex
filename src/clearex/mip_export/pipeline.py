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
from clearex.io.provenance import register_latest_output_reference

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]
_PROJECTIONS = ("xy", "xz", "yz")
_MAX_REDUCTION_READ_BYTES = 256 * 1024 * 1024


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
        Export file format (``tiff`` or ``zarr``).
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

    export_format = (
        str(normalized.get("export_format", "tiff")).strip().lower() or "tiff"
    )
    if export_format not in {"tiff", "zarr"}:
        raise ValueError("mip_export export_format must be 'tiff' or 'zarr'.")
    normalized["export_format"] = export_format
    normalized["output_directory"] = str(
        normalized.get("output_directory", "")
    ).strip()
    return normalized


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
    suffix = ".tif" if export_format == "tiff" else ".zarr"
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

    Returns
    -------
    str
        Stored dtype name.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "tiff":
        payload = _to_uint16(np.asarray(projection))
        tifffile.imwrite(str(output_path), payload, photometric="minisblack")
        return str(payload.dtype)

    if output_path.exists():
        shutil.rmtree(output_path)
    root = zarr.open_group(str(output_path), mode="w")
    payload = np.asarray(projection)
    root.create_dataset(
        name="data",
        data=payload,
        chunks=_default_projection_chunks(tuple(int(v) for v in payload.shape)),
        overwrite=True,
    )
    root["data"].attrs.update({"axes": [str(axis) for axis in axes]})
    root.attrs.update({"axes": [str(axis) for axis in axes]})
    return str(payload.dtype)


def _run_export_task(
    *,
    zarr_path: str,
    source_component: str,
    output_directory: str,
    export_format: str,
    position_mode: str,
    task: _MipExportTask,
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

    Returns
    -------
    dict[str, Any]
        Result metadata for manifest/provenance.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source_array = root[source_component]

    if task.p_index is None:
        projection = _chunkwise_projection_from_multi_position(
            source_array=source_array,
            t_index=int(task.t_index),
            c_index=int(task.c_index),
            projection=str(task.projection),
        )
    else:
        projection = _chunkwise_projection_from_single_position(
            source_array=source_array,
            t_index=int(task.t_index),
            p_index=int(task.p_index),
            c_index=int(task.c_index),
            projection=str(task.projection),
        )

    axes = _projection_axes(
        projection=str(task.projection),
        position_mode=str(position_mode),
    )
    output_path = _build_output_path(
        output_directory=Path(output_directory),
        export_format=str(export_format),
        task=task,
    )
    stored_dtype = _write_projection_output(
        output_path=output_path,
        projection=np.asarray(projection),
        export_format=str(export_format),
        axes=axes,
    )
    return {
        "path": str(output_path),
        "projection": str(task.projection),
        "t_index": int(task.t_index),
        "c_index": int(task.c_index),
        "p_index": int(task.p_index) if task.p_index is not None else None,
        "axes": list(axes),
        "dtype": str(stored_dtype),
    }


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
    if client is None:
        delayed_tasks = [
            delayed(_run_export_task)(
                zarr_path=str(zarr_path),
                source_component=source_component,
                output_directory=str(output_directory),
                export_format=export_format,
                position_mode=position_mode,
                task=task,
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
        max_in_flight = max(16, int(thread_capacity) * 4)
        _emit(
            15,
            "Submitting MIP-export tasks to Dask client "
            f"(worker_threads={thread_capacity}, max_in_flight={max_in_flight})",
        )
        completed = 0
        pending_count = 0
        completion_queue = as_completed()

        for task in tasks:
            completion_queue.add(
                client.submit(
                    _run_export_task,
                    zarr_path=str(zarr_path),
                    source_component=source_component,
                    output_directory=str(output_directory),
                    export_format=export_format,
                    position_mode=position_mode,
                    task=task,
                    pure=False,
                )
            )
            pending_count += 1
            if pending_count < max_in_flight:
                continue
            completed_future = next(completion_queue)
            task_results.append(dict(completed_future.result()))
            pending_count -= 1
            completed += 1
            progress = 15 + int((completed / max(1, total)) * 80)
            _emit(progress, f"Exported projection {completed}/{total}")

        for completed_future in completion_queue:
            task_results.append(dict(completed_future.result()))
            completed += 1
            progress = 15 + int((completed / max(1, total)) * 80)
            _emit(progress, f"Exported projection {completed}/{total}")

    task_results = sorted(task_results, key=lambda item: str(item.get("path", "")))
    exported_files = int(len(task_results))

    component = "results/mip_export/latest"
    root_w = zarr.open_group(str(zarr_path), mode="a")
    results_group = root_w.require_group("results")
    mip_group = results_group.require_group("mip_export")
    if "latest" in mip_group:
        del mip_group["latest"]
    latest_group = mip_group.create_group("latest")
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
            "parameters": {str(key): value for key, value in dict(normalized).items()},
            "manifest_preview": [
                str(item.get("path", "")) for item in task_results[:128]
            ],
        }
    )

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
