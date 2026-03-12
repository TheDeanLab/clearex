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

"""BaSiCPy-backed flatfield correction on canonical 6D Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union, cast

import dask
from dask import delayed
import numpy as np
from numpy.typing import NDArray
import zarr

from clearex.io.provenance import register_latest_output_reference

if TYPE_CHECKING:
    from dask.distributed import Client


ProgressCallback = Callable[[int, str], None]
Float32Array = NDArray[np.float32]
RegionBounds = tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]


@dataclass(frozen=True)
class FlatfieldProfileResult:
    """Fitted BaSiC profile for one ``(position, channel)`` pair.

    Attributes
    ----------
    position_index : int
        Zero-based position index.
    channel_index : int
        Zero-based channel index.
    flatfield_yx : numpy.ndarray
        Estimated flatfield image in ``(y, x)`` order.
    darkfield_yx : numpy.ndarray
        Estimated darkfield image in ``(y, x)`` order.
    baseline_tz : numpy.ndarray
        Estimated baseline values reshaped into ``(t, z)`` order.
    """

    position_index: int
    channel_index: int
    flatfield_yx: Float32Array
    darkfield_yx: Float32Array
    baseline_tz: Float32Array


@dataclass(frozen=True)
class FlatfieldSummary:
    """Summary metadata for one flatfield-correction run.

    Attributes
    ----------
    component : str
        Output latest group component path.
    data_component : str
        Corrected data array component path.
    flatfield_component : str
        Flatfield artifact component path.
    darkfield_component : str
        Darkfield artifact component path.
    baseline_component : str
        Baseline artifact component path.
    source_component : str
        Source data component used as input.
    profile_count : int
        Number of fitted ``(position, channel)`` profiles.
    transformed_volumes : int
        Number of transformed ``(time, position, channel)`` volumes.
    output_chunks_tpczyx : tuple[int, int, int, int, int, int]
        Chunk shape used for corrected output.
    output_dtype : str
        Saved corrected output dtype.
    basicpy_version : str, optional
        Installed BaSiCPy package version when available.
    """

    component: str
    data_component: str
    flatfield_component: str
    darkfield_component: str
    baseline_component: str
    source_component: str
    profile_count: int
    transformed_volumes: int
    output_chunks_tpczyx: tuple[int, int, int, int, int, int]
    output_dtype: str
    basicpy_version: Optional[str]


def _basicpy_version() -> Optional[str]:
    """Return installed BaSiCPy package version when available.

    Parameters
    ----------
    None

    Returns
    -------
    str, optional
        Installed version string, otherwise ``None``.
    """
    for package_name in ("BaSiCPy", "basicpy"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue
    return None


def _load_basic_class() -> Any:
    """Import and return the BaSiCPy ``BaSiC`` class.

    Parameters
    ----------
    None

    Returns
    -------
    Any
        Imported ``BaSiC`` class.

    Raises
    ------
    ImportError
        If BaSiCPy is unavailable in the current environment.
    """
    try:
        module = import_module("basicpy")
    except ImportError as exc:
        raise ImportError(
            "Flatfield correction requires the 'basicpy' package, but it is not "
            "available in the current environment."
        ) from exc
    return getattr(module, "BaSiC")


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flatfield runtime parameters.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate flatfield parameter mapping.

    Returns
    -------
    dict[str, Any]
        Normalized parameter mapping with concrete types.

    Raises
    ------
    ValueError
        If overlap shape or scalar values are invalid.
    """
    normalized = dict(parameters)
    normalized["input_source"] = (
        str(normalized.get("input_source", "data")).strip() or "data"
    )
    normalized["execution_order"] = max(1, int(normalized.get("execution_order", 1)))
    normalized["force_rerun"] = bool(normalized.get("force_rerun", False))
    normalized["chunk_basis"] = str(normalized.get("chunk_basis", "3d")).strip() or "3d"
    normalized["detect_2d_per_slice"] = bool(
        normalized.get("detect_2d_per_slice", False)
    )
    normalized["use_map_overlap"] = bool(normalized.get("use_map_overlap", True))
    overlap_zyx = normalized.get("overlap_zyx", [0, 32, 32])
    if not isinstance(overlap_zyx, (tuple, list)) or len(overlap_zyx) != 3:
        raise ValueError("flatfield overlap_zyx must define (z, y, x) overlap values.")
    normalized["overlap_zyx"] = tuple(max(0, int(v)) for v in overlap_zyx)
    normalized["memory_overhead_factor"] = float(
        normalized.get("memory_overhead_factor", 2.0)
    )
    if normalized["memory_overhead_factor"] <= 0:
        raise ValueError("flatfield memory_overhead_factor must be greater than zero.")
    normalized["get_darkfield"] = bool(normalized.get("get_darkfield", True))
    normalized["smoothness_flatfield"] = float(
        normalized.get("smoothness_flatfield", 1.0)
    )
    if normalized["smoothness_flatfield"] <= 0:
        raise ValueError("flatfield smoothness_flatfield must be greater than zero.")
    normalized["working_size"] = max(1, int(normalized.get("working_size", 128)))
    normalized["is_timelapse"] = bool(normalized.get("is_timelapse", False))
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
    """Convert integer region bounds into six-axis Python slices.

    Parameters
    ----------
    region : RegionBounds
        Bounds in canonical ``(t, p, c, z, y, x)`` order.

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
        Core bounds in canonical axis order.
    shape_tpczyx : tuple[int, int, int, int, int, int]
        Full array shape.
    overlap_zyx : tuple[int, int, int]
        Overlap depth in ``(z, y, x)`` order.
    use_overlap : bool
        Whether overlap expansion should be applied.

    Returns
    -------
    RegionBounds
        Expanded bounds clipped to array limits.
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


def _crop_slices_for_core(
    *,
    read_region: RegionBounds,
    core_region: RegionBounds,
) -> tuple[slice, slice, slice, slice, slice, slice]:
    """Build crop slices that trim a read region back to its core bounds.

    Parameters
    ----------
    read_region : RegionBounds
        Expanded region used for reading.
    core_region : RegionBounds
        Core output region.

    Returns
    -------
    tuple[slice, slice, slice, slice, slice, slice]
        Crop slices in canonical ``(t, p, c, z, y, x)`` order.
    """
    return (
        slice(
            int(core_region[0][0]) - int(read_region[0][0]),
            int(core_region[0][1]) - int(read_region[0][0]),
        ),
        slice(
            int(core_region[1][0]) - int(read_region[1][0]),
            int(core_region[1][1]) - int(read_region[1][0]),
        ),
        slice(
            int(core_region[2][0]) - int(read_region[2][0]),
            int(core_region[2][1]) - int(read_region[2][0]),
        ),
        slice(
            int(core_region[3][0]) - int(read_region[3][0]),
            int(core_region[3][1]) - int(read_region[3][0]),
        ),
        slice(
            int(core_region[4][0]) - int(read_region[4][0]),
            int(core_region[4][1]) - int(read_region[4][0]),
        ),
        slice(
            int(core_region[5][0]) - int(read_region[5][0]),
            int(core_region[5][1]) - int(read_region[5][0]),
        ),
    )


def _copy_source_array_attrs(
    *,
    root: zarr.Group,
    source_component: str,
    target_array: zarr.Array,
    output_chunks: tuple[int, int, int, int, int, int],
) -> None:
    """Copy key visualization and layout metadata from the source array.

    Parameters
    ----------
    root : zarr.Group
        Open Zarr root group.
    source_component : str
        Source array component path.
    target_array : zarr.Array
        Newly created corrected output array.
    output_chunks : tuple[int, int, int, int, int, int]
        Chunk shape for the corrected array.

    Returns
    -------
    None
        Target attrs are updated in-place.
    """
    source_attrs = dict(root[source_component].attrs)
    root_attrs = dict(root.attrs)
    copied: dict[str, Any] = {
        "source_component": str(source_component),
        "chunk_shape_tpczyx": [int(v) for v in output_chunks],
        "pyramid_levels": ["results/flatfield/latest/data"],
    }
    for key in (
        "scale_tpczyx",
        "physical_scale_tpczyx",
        "resolution_pyramid_factors_tpczyx",
    ):
        if key in source_attrs:
            copied[key] = source_attrs.get(key)
        elif key in root_attrs:
            copied[key] = root_attrs.get(key)
    target_array.attrs.update(copied)


def _prepare_output_arrays(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    parameters: Mapping[str, Any],
    basicpy_version: Optional[str],
) -> tuple[str, str, str, str, str, tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int]]:
    """Prepare latest flatfield output datasets in the Zarr store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Zarr analysis store path.
    source_component : str
        Source component path.
    parameters : mapping[str, Any]
        Normalized flatfield parameters.
    basicpy_version : str, optional
        Installed BaSiCPy version string.

    Returns
    -------
    tuple[str, str, str, str, str, tuple[int, ...], tuple[int, ...]]
        Component paths for latest group, data, flatfield, darkfield, baseline,
        followed by source shape and output chunks in ``(t, p, c, z, y, x)`` order.

    Raises
    ------
    ValueError
        If the source component is incompatible with canonical 6D data.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    source = root[source_component]
    shape = tuple(int(v) for v in source.shape)
    chunks = tuple(int(v) for v in source.chunks)
    if len(shape) != 6 or len(chunks) != 6:
        raise ValueError(
            "Flatfield correction requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    results_group = root.require_group("results")
    flatfield_group = results_group.require_group("flatfield")
    if "latest" in flatfield_group:
        del flatfield_group["latest"]
    latest_group = flatfield_group.create_group("latest")

    data_component = "results/flatfield/latest/data"
    flatfield_component = "results/flatfield/latest/flatfield_pcyx"
    darkfield_component = "results/flatfield/latest/darkfield_pcyx"
    baseline_component = "results/flatfield/latest/baseline_pctz"

    data_array = latest_group.create_dataset(
        name="data",
        shape=shape,
        chunks=chunks,
        dtype=np.float32,
        overwrite=True,
    )
    _copy_source_array_attrs(
        root=root,
        source_component=source_component,
        target_array=data_array,
        output_chunks=chunks,
    )

    artifact_chunks_pcyx = (
        1,
        1,
        max(1, min(chunks[4], shape[4])),
        max(1, min(chunks[5], shape[5])),
    )
    latest_group.create_dataset(
        name="flatfield_pcyx",
        shape=(shape[1], shape[2], shape[4], shape[5]),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="darkfield_pcyx",
        shape=(shape[1], shape[2], shape[4], shape[5]),
        chunks=artifact_chunks_pcyx,
        dtype=np.float32,
        overwrite=True,
    )
    latest_group.create_dataset(
        name="baseline_pctz",
        shape=(shape[1], shape[2], shape[0], shape[3]),
        chunks=(
            1,
            1,
            1,
            max(1, min(shape[3], chunks[3])),
        ),
        dtype=np.float32,
        overwrite=True,
    )

    latest_group.attrs.update(
        {
            "storage_policy": "latest_only",
            "run_id": None,
            "source_component": str(source_component),
            "data_component": data_component,
            "flatfield_component": flatfield_component,
            "darkfield_component": darkfield_component,
            "baseline_component": baseline_component,
            "parameters": {str(k): v for k, v in dict(parameters).items()},
            "output_dtype": "float32",
            "output_chunks_tpczyx": [int(v) for v in chunks],
            "basicpy_version": basicpy_version,
        }
    )

    return (
        "results/flatfield/latest",
        data_component,
        flatfield_component,
        darkfield_component,
        baseline_component,
        (
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            int(shape[3]),
            int(shape[4]),
            int(shape[5]),
        ),
        (
            int(chunks[0]),
            int(chunks[1]),
            int(chunks[2]),
            int(chunks[3]),
            int(chunks[4]),
            int(chunks[5]),
        ),
    )


def _fit_profile(
    *,
    zarr_path: str,
    source_component: str,
    position_index: int,
    channel_index: int,
    parameters: Mapping[str, Any],
) -> FlatfieldProfileResult:
    """Fit a BaSiC profile for one ``(position, channel)`` input subset.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component.
    position_index : int
        Position index on the ``p`` axis.
    channel_index : int
        Channel index on the ``c`` axis.
    parameters : mapping[str, Any]
        Normalized flatfield parameter mapping.

    Returns
    -------
    FlatfieldProfileResult
        Fitted profile artifacts for the requested ``(position, channel)`` pair.

    Raises
    ------
    ValueError
        If the fitted baseline length cannot be reshaped back to ``(t, z)``.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    source = root[source_component]
    volume_tzyx = np.asarray(
        source[:, position_index, channel_index, :, :, :],
        dtype=np.float32,
    )
    t_count, z_count, y_count, x_count = volume_tzyx.shape
    fit_images = volume_tzyx.reshape(t_count * z_count, y_count, x_count)

    basic_class = _load_basic_class()
    basic = basic_class(
        fitting_mode="approximate",
        get_darkfield=bool(parameters.get("get_darkfield", True)),
        smoothness_flatfield=float(parameters.get("smoothness_flatfield", 1.0)),
        working_size=int(parameters.get("working_size", 128)),
        device="cpu",
    )
    basic.fit(fit_images, skip_shape_warning=True)

    flatfield_yx = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield_yx = np.asarray(basic.darkfield, dtype=np.float32)
    baseline = np.asarray(basic.baseline, dtype=np.float32)
    if int(baseline.size) != int(t_count * z_count):
        raise ValueError(
            "Flatfield baseline shape mismatch: "
            f"expected {t_count * z_count} values, got {baseline.size}."
        )

    return FlatfieldProfileResult(
        position_index=int(position_index),
        channel_index=int(channel_index),
        flatfield_yx=flatfield_yx,
        darkfield_yx=darkfield_yx,
        baseline_tz=baseline.reshape(t_count, z_count),
    )


def _persist_profile_artifacts(
    *,
    zarr_path: str,
    flatfield_component: str,
    darkfield_component: str,
    baseline_component: str,
    profile: FlatfieldProfileResult,
) -> None:
    """Persist one fitted profile artifact bundle to the Zarr store.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    flatfield_component : str
        Flatfield artifact component path.
    darkfield_component : str
        Darkfield artifact component path.
    baseline_component : str
        Baseline artifact component path.
    profile : FlatfieldProfileResult
        Fitted profile payload.

    Returns
    -------
    None
        Artifacts are written in-place to the store.
    """
    root = zarr.open_group(str(zarr_path), mode="a")
    root[flatfield_component][
        profile.position_index,
        profile.channel_index,
        :,
        :,
    ] = profile.flatfield_yx
    root[darkfield_component][
        profile.position_index,
        profile.channel_index,
        :,
        :,
    ] = profile.darkfield_yx
    root[baseline_component][
        profile.position_index,
        profile.channel_index,
        :,
        :,
    ] = profile.baseline_tz


def _transform_region(
    *,
    zarr_path: str,
    source_component: str,
    output_data_component: str,
    read_region: RegionBounds,
    core_region: RegionBounds,
    flatfield_yx: Float32Array,
    darkfield_yx: Float32Array,
    baseline_tz: Float32Array,
    is_timelapse: bool,
) -> int:
    """Apply flatfield correction for one chunk region and write the core.

    Parameters
    ----------
    zarr_path : str
        Zarr analysis store path.
    source_component : str
        Source data component path.
    output_data_component : str
        Corrected output data component path.
    read_region : RegionBounds
        Expanded read bounds including overlap halo.
    core_region : RegionBounds
        Non-overlapping write bounds.
    flatfield_yx : numpy.ndarray
        Fitted flatfield image in ``(y, x)`` order.
    darkfield_yx : numpy.ndarray
        Fitted darkfield image in ``(y, x)`` order.
    baseline_tz : numpy.ndarray
        Fitted baseline in ``(t, z)`` order.
    is_timelapse : bool
        Whether baseline subtraction should be applied.

    Returns
    -------
    int
        Constant ``1`` used for task completion accounting.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    chunk = np.asarray(
        root[source_component][_region_to_slices(read_region)],
        dtype=np.float32,
    )

    y_start, y_stop = int(read_region[4][0]), int(read_region[4][1])
    x_start, x_stop = int(read_region[5][0]), int(read_region[5][1])
    denominator = np.maximum(
        np.asarray(flatfield_yx[y_start:y_stop, x_start:x_stop], dtype=np.float32),
        np.float32(1e-6),
    )
    darkfield = np.asarray(
        darkfield_yx[y_start:y_stop, x_start:x_stop],
        dtype=np.float32,
    )
    corrected = (chunk - darkfield[None, None, None, None, :, :]) / denominator[
        None,
        None,
        None,
        None,
        :,
        :,
    ]

    if is_timelapse:
        baseline = np.asarray(
            baseline_tz[
                read_region[0][0] : read_region[0][1],
                read_region[3][0] : read_region[3][1],
            ],
            dtype=np.float32,
        )
        corrected = corrected - baseline[:, None, None, :, None, None]

    corrected_core = corrected[_crop_slices_for_core(
        read_region=read_region,
        core_region=core_region,
    )].astype(np.float32, copy=False)

    write_root = zarr.open_group(str(zarr_path), mode="a")
    write_root[output_data_component][_region_to_slices(core_region)] = corrected_core
    return 1


def run_flatfield_analysis(
    *,
    zarr_path: Union[str, Path],
    parameters: Mapping[str, Any],
    client: Optional["Client"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> FlatfieldSummary:
    """Run BaSiCPy flatfield correction and persist latest outputs.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to canonical analysis-store Zarr object.
    parameters : mapping[str, Any]
        Flatfield-correction parameters.
    client : dask.distributed.Client, optional
        Active Dask client for distributed execution.
    progress_callback : callable, optional
        Progress callback invoked as ``callback(percent, message)``.

    Returns
    -------
    FlatfieldSummary
        Summary metadata for the completed flatfield run.

    Raises
    ------
    ValueError
        If the source component is missing or incompatible.
    ImportError
        If BaSiCPy is unavailable in the current environment.
    """

    def _emit(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(int(percent), str(message))

    normalized = _normalize_parameters(parameters)
    basicpy_version = _basicpy_version()
    root = zarr.open_group(str(zarr_path), mode="r")
    source_component = str(normalized.get("input_source", "data")).strip() or "data"
    try:
        source = root[source_component]
    except Exception as exc:
        raise ValueError(
            f"Flatfield input component '{source_component}' was not found in {zarr_path}."
        ) from exc

    shape = tuple(int(v) for v in source.shape)
    chunks = tuple(int(v) for v in source.chunks)
    if len(shape) != 6 or len(chunks) != 6:
        raise ValueError(
            "Flatfield correction requires canonical 6D data (t,p,c,z,y,x). "
            f"Input component '{source_component}' is incompatible."
        )

    _emit(5, "Preparing flatfield output datasets")
    (
        component,
        data_component,
        flatfield_component,
        darkfield_component,
        baseline_component,
        shape_tpczyx,
        output_chunks,
    ) = _prepare_output_arrays(
        zarr_path=zarr_path,
        source_component=source_component,
        parameters=normalized,
        basicpy_version=basicpy_version,
    )

    fit_tasks = [
        delayed(_fit_profile)(
            zarr_path=str(zarr_path),
            source_component=source_component,
            position_index=int(position_index),
            channel_index=int(channel_index),
            parameters=normalized,
        )
        for position_index, channel_index in product(
            range(shape_tpczyx[1]),
            range(shape_tpczyx[2]),
        )
    ]

    _emit(10, f"Prepared {len(fit_tasks)} flatfield profile fit tasks")
    profile_results: list[FlatfieldProfileResult] = []
    if client is None:
        computed = dask.compute(*fit_tasks, scheduler="processes")
        profile_results = [item for item in computed]
    else:
        from dask.distributed import as_completed

        futures = cast(list[Any], client.compute(fit_tasks))
        total = max(1, int(len(futures)))
        completed = 0
        for future in as_completed(futures):
            profile_results.append(future.result())
            completed += 1
            progress = 10 + int((completed / total) * 25)
            _emit(progress, f"Fitted flatfield profile {completed}/{total}")

    profile_results.sort(
        key=lambda item: (int(item.position_index), int(item.channel_index))
    )
    for profile in profile_results:
        _persist_profile_artifacts(
            zarr_path=str(zarr_path),
            flatfield_component=flatfield_component,
            darkfield_component=darkfield_component,
            baseline_component=baseline_component,
            profile=profile,
        )
    _emit(40, f"Persisted {len(profile_results)} fitted flatfield profiles")

    profile_map = {
        (int(item.position_index), int(item.channel_index)): item
        for item in profile_results
    }
    y_bounds = _axis_chunk_bounds(shape_tpczyx[4], output_chunks[4])
    x_bounds = _axis_chunk_bounds(shape_tpczyx[5], output_chunks[5])
    transform_tasks = []
    for t_index, p_index, c_index, y_bounds_chunk, x_bounds_chunk in product(
        range(shape_tpczyx[0]),
        range(shape_tpczyx[1]),
        range(shape_tpczyx[2]),
        y_bounds,
        x_bounds,
    ):
        profile = profile_map[(int(p_index), int(c_index))]
        core_region: RegionBounds = (
            (int(t_index), int(t_index) + 1),
            (int(p_index), int(p_index) + 1),
            (int(c_index), int(c_index) + 1),
            (0, int(shape_tpczyx[3])),
            (int(y_bounds_chunk[0]), int(y_bounds_chunk[1])),
            (int(x_bounds_chunk[0]), int(x_bounds_chunk[1])),
        )
        transform_tasks.append(
            delayed(_transform_region)(
                zarr_path=str(zarr_path),
                source_component=source_component,
                output_data_component=data_component,
                read_region=_expand_region_with_overlap(
                    core_region,
                    shape_tpczyx=shape_tpczyx,
                    overlap_zyx=normalized["overlap_zyx"],
                    use_overlap=bool(normalized["use_map_overlap"]),
                ),
                core_region=core_region,
                flatfield_yx=profile.flatfield_yx,
                darkfield_yx=profile.darkfield_yx,
                baseline_tz=profile.baseline_tz,
                is_timelapse=bool(normalized["is_timelapse"]),
            )
        )

    _emit(45, f"Prepared {len(transform_tasks)} flatfield transform tasks")
    if client is None:
        dask.compute(*transform_tasks, scheduler="processes")
        _emit(95, "Completed flatfield transform tasks")
    else:
        from dask.distributed import as_completed

        futures = cast(list[Any], client.compute(transform_tasks))
        total = max(1, int(len(futures)))
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            progress = 45 + int((completed / total) * 50)
            _emit(progress, f"Corrected chunk {completed}/{total}")

    root = zarr.open_group(str(zarr_path), mode="a")
    root[component].attrs.update(
        {
            "profile_count": int(len(profile_results)),
            "transformed_volumes": int(
                shape_tpczyx[0] * shape_tpczyx[1] * shape_tpczyx[2]
            ),
            "source_component": str(source_component),
            "data_component": data_component,
            "flatfield_component": flatfield_component,
            "darkfield_component": darkfield_component,
            "baseline_component": baseline_component,
            "output_dtype": "float32",
            "output_chunks_tpczyx": [int(v) for v in output_chunks],
            "basicpy_version": basicpy_version,
        }
    )
    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name="flatfield",
        component=component,
        metadata={
            "data_component": data_component,
            "flatfield_component": flatfield_component,
            "darkfield_component": darkfield_component,
            "baseline_component": baseline_component,
            "source_component": source_component,
            "profile_count": int(len(profile_results)),
            "transformed_volumes": int(
                shape_tpczyx[0] * shape_tpczyx[1] * shape_tpczyx[2]
            ),
            "output_dtype": "float32",
            "output_chunks_tpczyx": [int(v) for v in output_chunks],
            "basicpy_version": basicpy_version,
        },
    )
    _emit(100, "Flatfield correction complete")
    return FlatfieldSummary(
        component=component,
        data_component=data_component,
        flatfield_component=flatfield_component,
        darkfield_component=darkfield_component,
        baseline_component=baseline_component,
        source_component=source_component,
        profile_count=int(len(profile_results)),
        transformed_volumes=int(shape_tpczyx[0] * shape_tpczyx[1] * shape_tpczyx[2]),
        output_chunks_tpczyx=output_chunks,
        output_dtype="float32",
        basicpy_version=basicpy_version,
    )
