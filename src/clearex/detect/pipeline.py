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

"""Chunk-parallel particle-detection workflow on canonical 6D Zarr stores."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union

# Third Party Imports
import dask
from dask import delayed
import numpy as np
import zarr

# Local Imports
from clearex.detect.particles import (
    detect_particles,
    eliminate_insignificant_point_sources,
    preprocess,
    remove_close_blobs,
)
from clearex.io.provenance import register_latest_output_reference

if TYPE_CHECKING:
    from dask.distributed import Client


DetectionsArray = np.ndarray
RegionBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
ProgressCallback = Callable[[int, str], None]

_PARTICLE_COLUMNS = (
    "t",
    "p",
    "c",
    "z",
    "y",
    "x",
    "sigma",
    "intensity",
)


@dataclass(frozen=True)
class ParticleDetectionSummary:
    """Summary metadata for one particle-detection run.

    Attributes
    ----------
    component : str
        Output component path inside the Zarr store.
    detections : int
        Total number of detections written.
    chunks_processed : int
        Number of chunk tasks that were processed.
    channel_index : int
        Selected source channel index.
    """

    component: str
    detections: int
    chunks_processed: int
    channel_index: int


def _normalize_particle_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize particle-detection runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate parameter dictionary.

    Returns
    -------
    dict[str, Any]
        Normalized parameter mapping with concrete numeric/bool values.

    Raises
    ------
    ValueError
        If overlap parameters are malformed.
    """
    normalized = dict(parameters)
    normalized["channel_index"] = max(0, int(normalized.get("channel_index", 0)))
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    normalized["use_map_overlap"] = bool(normalized.get("use_map_overlap", False))
    normalized["detect_2d_per_slice"] = bool(
        normalized.get("detect_2d_per_slice", True)
    )
    normalized["bg_sigma"] = float(normalized.get("bg_sigma", 20.0))
    normalized["fwhm_px"] = float(normalized.get("fwhm_px", 3.0))
    normalized["sigma_min_factor"] = float(normalized.get("sigma_min_factor", 1.0))
    normalized["sigma_max_factor"] = float(normalized.get("sigma_max_factor", 3.0))
    normalized["threshold"] = float(normalized.get("threshold", 0.1))
    normalized["overlap"] = float(normalized.get("overlap", 0.5))
    normalized["exclude_border"] = max(0, int(normalized.get("exclude_border", 5)))
    normalized["eliminate_insignificant_particles"] = bool(
        normalized.get("eliminate_insignificant_particles", False)
    )
    normalized["remove_close_particles"] = bool(
        normalized.get("remove_close_particles", False)
    )
    normalized["min_distance_sigma"] = float(
        normalized.get("min_distance_sigma", 10.0)
    )
    overlap_zyx = normalized.get("overlap_zyx", [0, 0, 0])
    if not isinstance(overlap_zyx, (tuple, list)) or len(overlap_zyx) != 3:
        raise ValueError(
            "particle_detection overlap_zyx must define (z, y, x) overlap values."
        )
    normalized["overlap_zyx"] = tuple(max(0, int(v)) for v in overlap_zyx)
    return normalized


def _axis_chunk_bounds(size: int, chunk_size: int) -> list[tuple[int, int]]:
    """Build contiguous chunk bounds for one axis.

    Parameters
    ----------
    size : int
        Axis length.
    chunk_size : int
        Chunk size for the axis.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(start, stop)`` bounds covering the full axis.
    """
    return [
        (start, min(start + chunk_size, size))
        for start in range(0, size, chunk_size)
    ]


def _region_to_slices(region: RegionBounds) -> tuple[slice, slice, slice, slice, slice, slice]:
    """Convert six-axis integer bounds into Python slices.

    Parameters
    ----------
    region : RegionBounds
        Axis bounds in canonical ``(t, p, c, z, y, x)`` order.

    Returns
    -------
    tuple[slice, slice, slice, slice, slice, slice]
        Equivalent slicing tuple.
    """
    return (
        slice(int(region[0][0]), int(region[0][1])),
        slice(int(region[1][0]), int(region[1][1])),
        slice(int(region[2][0]), int(region[2][1])),
        slice(int(region[3][0]), int(region[3][1])),
        slice(int(region[4][0]), int(region[4][1])),
        slice(int(region[5][0]), int(region[5][1])),
    )


def _expand_region_with_overlap(
    core_region: RegionBounds,
    *,
    shape_tpczyx: tuple[int, int, int, int, int, int],
    overlap_zyx: tuple[int, int, int],
    use_overlap: bool,
) -> RegionBounds:
    """Expand a core region by overlap margins in spatial axes.

    Parameters
    ----------
    core_region : RegionBounds
        Core chunk bounds in canonical order.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full array shape.
    overlap_zyx : tuple[int, int, int]
        Requested overlap depth for ``(z, y, x)`` axes.
    use_overlap : bool
        Whether overlap expansion should be applied.

    Returns
    -------
    RegionBounds
        Region bounds with overlap margins clipped to array bounds.
    """
    if not use_overlap:
        return core_region

    z_overlap, y_overlap, x_overlap = overlap_zyx
    t_bounds, p_bounds, c_bounds, z_bounds, y_bounds, x_bounds = core_region

    return (
        t_bounds,
        p_bounds,
        c_bounds,
        (
            max(0, int(z_bounds[0]) - z_overlap),
            min(int(shape_tpczyx[3]), int(z_bounds[1]) + z_overlap),
        ),
        (
            max(0, int(y_bounds[0]) - y_overlap),
            min(int(shape_tpczyx[4]), int(y_bounds[1]) + y_overlap),
        ),
        (
            max(0, int(x_bounds[0]) - x_overlap),
            min(int(shape_tpczyx[5]), int(x_bounds[1]) + x_overlap),
        ),
    )


def _apply_spatial_core_mask(
    detections: DetectionsArray,
    core_region: RegionBounds,
) -> DetectionsArray:
    """Filter detections to core chunk spatial bounds.

    Parameters
    ----------
    detections : numpy.ndarray
        Detection rows with columns ``(t, p, c, z, y, x, sigma, intensity)``.
    core_region : RegionBounds
        Core chunk bounds in canonical order.

    Returns
    -------
    numpy.ndarray
        Detections constrained to core-region ``z/y/x`` bounds.
    """
    if detections.size == 0:
        return detections.reshape(0, len(_PARTICLE_COLUMNS))

    z_start, z_stop = core_region[3]
    y_start, y_stop = core_region[4]
    x_start, x_stop = core_region[5]
    mask = (
        (detections[:, 3] >= float(z_start))
        & (detections[:, 3] < float(z_stop))
        & (detections[:, 4] >= float(y_start))
        & (detections[:, 4] < float(y_stop))
        & (detections[:, 5] >= float(x_start))
        & (detections[:, 5] < float(x_stop))
    )
    return detections[mask]


def _chunk_detection_rows(
    *,
    chunk_zyx: np.ndarray,
    read_region: RegionBounds,
    params: Mapping[str, Any],
) -> DetectionsArray:
    """Detect particles in one 3D chunk and return global-coordinate rows.

    Parameters
    ----------
    chunk_zyx : numpy.ndarray
        Source image chunk in ``(z, y, x)`` order.
    read_region : RegionBounds
        Read bounds in canonical ``(t, p, c, z, y, x)`` order.
    params : mapping[str, Any]
        Normalized particle-detection parameters.

    Returns
    -------
    numpy.ndarray
        Detection rows with columns ``(t, p, c, z, y, x, sigma, intensity)``.
    """
    t_start, _ = read_region[0]
    p_start, _ = read_region[1]
    c_start, _ = read_region[2]
    z_start, _ = read_region[3]
    y_start, _ = read_region[4]
    x_start, _ = read_region[5]

    rows: list[list[float]] = []
    for z_local in range(int(chunk_zyx.shape[0])):
        plane = chunk_zyx[z_local, :, :]
        preprocessed = preprocess(plane, bg_sigma=float(params["bg_sigma"]))
        detections = detect_particles(
            img=preprocessed,
            fwhm_px=float(params["fwhm_px"]),
            sigma_min_factor=float(params["sigma_min_factor"]),
            sigma_max_factor=float(params["sigma_max_factor"]),
            threshold=float(params["threshold"]),
            overlap=float(params["overlap"]),
            exclude_border=int(params["exclude_border"]),
        )
        if detections.size == 0:
            continue

        for y_local, x_local, sigma in detections:
            y_idx = int(np.clip(int(np.round(y_local)), 0, chunk_zyx.shape[1] - 1))
            x_idx = int(np.clip(int(np.round(x_local)), 0, chunk_zyx.shape[2] - 1))
            intensity = float(chunk_zyx[z_local, y_idx, x_idx])
            rows.append(
                [
                    float(t_start),
                    float(p_start),
                    float(c_start),
                    float(z_start + z_local),
                    float(y_start + y_local),
                    float(x_start + x_local),
                    float(sigma),
                    intensity,
                ]
            )

    if not rows:
        return np.empty((0, len(_PARTICLE_COLUMNS)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _apply_optional_detection_filters(
    detections: DetectionsArray,
    *,
    chunk_zyx: np.ndarray,
    read_region: RegionBounds,
    params: Mapping[str, Any],
) -> DetectionsArray:
    """Apply optional significance and proximity filters in chunk-local space.

    Parameters
    ----------
    detections : numpy.ndarray
        Detection rows in global coordinates.
    chunk_zyx : numpy.ndarray
        Source 3D chunk in ``(z, y, x)`` order.
    read_region : RegionBounds
        Read bounds in canonical order.
    params : mapping[str, Any]
        Normalized particle-detection parameters.

    Returns
    -------
    numpy.ndarray
        Filtered detection rows in global coordinates.
    """
    if detections.size == 0:
        return detections.reshape(0, len(_PARTICLE_COLUMNS))
    if not bool(params["eliminate_insignificant_particles"]) and not bool(
        params["remove_close_particles"]
    ):
        return detections

    z_start, _ = read_region[3]
    y_start, _ = read_region[4]
    x_start, _ = read_region[5]
    local_blobs = np.column_stack(
        (
            detections[:, 3] - float(z_start),
            detections[:, 4] - float(y_start),
            detections[:, 5] - float(x_start),
            detections[:, 6],
        )
    )

    if bool(params["eliminate_insignificant_particles"]):
        local_blobs = eliminate_insignificant_point_sources(chunk_zyx, local_blobs)
    if local_blobs.size == 0:
        return np.empty((0, len(_PARTICLE_COLUMNS)), dtype=np.float64)

    if bool(params["remove_close_particles"]):
        filtered = np.asarray(local_blobs, dtype=np.float64)
        while True:
            before = int(filtered.shape[0])
            filtered = remove_close_blobs(
                blobs=filtered,
                image=chunk_zyx,
                min_dist=float(params["min_distance_sigma"]),
            )
            if int(filtered.shape[0]) >= before:
                break
        local_blobs = filtered

    if local_blobs.size == 0:
        return np.empty((0, len(_PARTICLE_COLUMNS)), dtype=np.float64)

    t_value = float(detections[0, 0])
    p_value = float(detections[0, 1])
    c_value = float(detections[0, 2])
    filtered_rows: list[list[float]] = []
    for z_local, y_local, x_local, sigma in np.asarray(local_blobs, dtype=np.float64):
        z_idx = int(np.clip(int(np.round(z_local)), 0, chunk_zyx.shape[0] - 1))
        y_idx = int(np.clip(int(np.round(y_local)), 0, chunk_zyx.shape[1] - 1))
        x_idx = int(np.clip(int(np.round(x_local)), 0, chunk_zyx.shape[2] - 1))
        intensity = float(chunk_zyx[z_idx, y_idx, x_idx])
        filtered_rows.append(
            [
                t_value,
                p_value,
                c_value,
                float(z_start) + float(z_local),
                float(y_start) + float(y_local),
                float(x_start) + float(x_local),
                float(sigma),
                intensity,
            ]
        )

    if not filtered_rows:
        return np.empty((0, len(_PARTICLE_COLUMNS)), dtype=np.float64)
    return np.asarray(filtered_rows, dtype=np.float64)


def _detect_particles_for_region(
    *,
    zarr_path: str,
    source_component: str,
    read_region: RegionBounds,
    core_region: RegionBounds,
    parameters: Mapping[str, Any],
) -> DetectionsArray:
    """Run particle detection for one chunk region.

    Parameters
    ----------
    zarr_path : str
        Source analysis-store path.
    source_component : str
        Source array component path within the Zarr store.
    read_region : RegionBounds
        Read bounds in canonical order.
    core_region : RegionBounds
        Core chunk bounds used for overlap de-duplication.
    parameters : mapping[str, Any]
        Normalized particle-detection parameters.

    Returns
    -------
    numpy.ndarray
        Detection rows in global coordinates.
    """
    root = zarr.open_group(zarr_path, mode="r")
    data = root[source_component]
    chunk_6d = np.asarray(data[_region_to_slices(read_region)])
    chunk_zyx = np.asarray(chunk_6d[0, 0, 0, :, :, :])

    detections = _chunk_detection_rows(
        chunk_zyx=chunk_zyx,
        read_region=read_region,
        params=parameters,
    )
    detections = _apply_optional_detection_filters(
        detections,
        chunk_zyx=chunk_zyx,
        read_region=read_region,
        params=parameters,
    )
    detections = _apply_spatial_core_mask(detections, core_region=core_region)
    return detections


def _compose_detection_output(
    arrays: Sequence[DetectionsArray],
) -> DetectionsArray:
    """Concatenate and sort detection rows.

    Parameters
    ----------
    arrays : sequence of numpy.ndarray
        Chunk-level detection arrays.

    Returns
    -------
    numpy.ndarray
        Consolidated detection table sorted by ``(t, p, z, y, x)``.
    """
    valid = [
        np.asarray(array, dtype=np.float64)
        for array in arrays
        if array is not None and int(np.asarray(array).size) > 0
    ]
    if not valid:
        return np.empty((0, len(_PARTICLE_COLUMNS)), dtype=np.float64)

    merged = np.concatenate(valid, axis=0)
    order = np.lexsort(
        (
            merged[:, 5],
            merged[:, 4],
            merged[:, 3],
            merged[:, 1],
            merged[:, 0],
        )
    )
    return merged[order]


def save_particle_detections_to_store(
    *,
    zarr_path: Union[str, Path],
    detections: DetectionsArray,
    parameters: Mapping[str, Any],
    run_id: Optional[str] = None,
) -> str:
    """Store particle detections under ``results/particle_detection/latest``.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis store path.
    detections : numpy.ndarray
        Detection table with columns
        ``(t, p, c, z, y, x, sigma, intensity)``.
    parameters : mapping[str, Any]
        Effective particle-detection parameter mapping.
    run_id : str, optional
        Provenance run identifier associated with this result.

    Returns
    -------
    str
        Latest component path for particle-detection results.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    results_group = root.require_group("results")
    particle_group = results_group.require_group("particle_detection")
    if "latest" in particle_group:
        del particle_group["latest"]
    latest_group = particle_group.create_group("latest")

    detection_array = np.asarray(detections, dtype=np.float32)
    row_chunks = int(min(max(1, detection_array.shape[0]), 16384))
    latest_group.create_dataset(
        name="detections",
        data=detection_array,
        chunks=(row_chunks, len(_PARTICLE_COLUMNS)),
        overwrite=True,
    )
    latest_group["detections"].attrs.update(
        {
            "columns": list(_PARTICLE_COLUMNS),
            "axes_points": ["t", "z", "y", "x"],
        }
    )

    napari_points = (
        detection_array[:, [0, 3, 4, 5]]
        if detection_array.shape[0] > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    points_chunks = int(min(max(1, napari_points.shape[0]), 16384))
    latest_group.create_dataset(
        name="points_tzyx",
        data=napari_points,
        chunks=(points_chunks, 4),
        overwrite=True,
    )
    latest_group.attrs.update(
        {
            "storage_policy": "latest_only",
            "channel_index": int(parameters.get("channel_index", 0)),
            "detection_count": int(detection_array.shape[0]),
            "parameters": {str(k): v for k, v in dict(parameters).items()},
            "napari_points_component": "results/particle_detection/latest/points_tzyx",
            "run_id": run_id,
        }
    )

    component = "results/particle_detection/latest"
    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="particle_detection",
        component=component,
        run_id=run_id,
        metadata={
            "detections_component": f"{component}/detections",
            "points_component": f"{component}/points_tzyx",
            "detection_count": int(detection_array.shape[0]),
            "parameters": {str(k): v for k, v in dict(parameters).items()},
        },
    )
    return component


def run_particle_detection_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> ParticleDetectionSummary:
    """Run chunk-parallel particle detection and persist latest result output.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Particle-detection parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    ParticleDetectionSummary
        Summary including output component and detection count.

    Raises
    ------
    ValueError
        If required source data are missing or channel index is out of bounds.
    """
    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_particle_parameters(parameters)
    root = zarr.open_group(str(zarr_path), mode="r")
    source_component = str(normalized.get("input_source", "data")).strip() or "data"
    try:
        data = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"Particle-detection input component '{source_component}' was not found in {zarr_path}."
        ) from exc
    shape = tuple(int(v) for v in data.shape)
    chunks = tuple(int(v) for v in data.chunks)
    if len(shape) != 6 or len(chunks) != 6:
        raise ValueError(
            "Particle detection requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    channel_index = int(normalized["channel_index"])
    if channel_index >= int(shape[2]):
        raise ValueError(
            f"Particle-detection channel_index={channel_index} is out of bounds "
            f"for channel axis size {shape[2]}."
        )

    t_bounds = _axis_chunk_bounds(shape[0], chunks[0])
    p_bounds = _axis_chunk_bounds(shape[1], chunks[1])
    z_bounds = _axis_chunk_bounds(shape[3], chunks[3])
    y_bounds = _axis_chunk_bounds(shape[4], chunks[4])
    x_bounds = _axis_chunk_bounds(shape[5], chunks[5])

    core_regions: list[RegionBounds] = []
    for t_chunk, p_chunk, z_chunk, y_chunk, x_chunk in product(
        t_bounds, p_bounds, z_bounds, y_bounds, x_bounds
    ):
        core_regions.append(
            (
                t_chunk,
                p_chunk,
                (channel_index, channel_index + 1),
                z_chunk,
                y_chunk,
                x_chunk,
            )
        )

    _emit(5, f"Prepared {len(core_regions)} particle-detection chunk tasks")
    if not core_regions:
        component = save_particle_detections_to_store(
            zarr_path=zarr_path,
            detections=np.empty((0, len(_PARTICLE_COLUMNS)), dtype=np.float32),
            parameters=normalized,
        )
        return ParticleDetectionSummary(
            component=component,
            detections=0,
            chunks_processed=0,
            channel_index=channel_index,
        )

    delayed_tasks = [
        delayed(_detect_particles_for_region)(
            zarr_path=str(zarr_path),
            source_component=source_component,
            read_region=_expand_region_with_overlap(
                core_region,
                shape_tpczyx=(
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    shape[4],
                    shape[5],
                ),
                overlap_zyx=normalized["overlap_zyx"],
                use_overlap=bool(normalized["use_map_overlap"]),
            ),
            core_region=core_region,
            parameters=normalized,
        )
        for core_region in core_regions
    ]

    results: list[DetectionsArray] = []
    if client is None:
        _emit(10, "Running particle detection with local process scheduler")
        computed = dask.compute(*delayed_tasks, scheduler="processes")
        results = [np.asarray(arr, dtype=np.float64) for arr in computed]
        _emit(90, "Completed particle-detection chunk computation")
    else:
        from dask.distributed import as_completed

        _emit(10, "Submitting particle-detection chunk tasks to Dask client")
        futures = client.compute(delayed_tasks)
        total = int(len(futures))
        completed = 0
        for future in as_completed(futures):
            results.append(np.asarray(future.result(), dtype=np.float64))
            completed += 1
            progress = 10 + int((completed / total) * 80)
            _emit(progress, f"Processed chunk {completed}/{total}")

    merged = _compose_detection_output(results)
    _emit(95, f"Writing {int(merged.shape[0])} detections to Zarr store")
    component = save_particle_detections_to_store(
        zarr_path=zarr_path,
        detections=merged,
        parameters=normalized,
    )
    _emit(100, "Particle detection complete")
    return ParticleDetectionSummary(
        component=component,
        detections=int(merged.shape[0]),
        chunks_processed=int(len(core_regions)),
        channel_index=channel_index,
    )
