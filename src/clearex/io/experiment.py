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

"""Navigate experiment ingestion and 6D Zarr analysis-store utilities."""

from __future__ import annotations

# Standard Library Imports
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import re

# Third Party Imports
import dask.array as da
import numpy as np
import zarr

# Local Imports
from clearex.io.read import ImageInfo

# YAML parsing is optional because many Navigate ``experiment.yml`` files are JSON.
try:
    import yaml

    HAS_PYYAML = True
except Exception:
    HAS_PYYAML = False


ArrayLike = Union[np.ndarray, da.Array]


@dataclass
class NavigateChannel:
    """Selected channel metadata from a Navigate experiment.

    Attributes
    ----------
    name : str
        Channel key in the experiment state (e.g., ``"channel_1"``).
    laser : str, optional
        Laser label assigned for the channel.
    laser_index : int, optional
        Laser index used by acquisition.
    exposure_ms : float, optional
        Camera exposure time in milliseconds.
    is_selected : bool
        Whether the channel was selected for acquisition.
    """

    name: str
    laser: Optional[str]
    laser_index: Optional[int]
    exposure_ms: Optional[float]
    is_selected: bool


@dataclass
class NavigateExperiment:
    """Parsed Navigate experiment metadata required by ClearEx.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to the source ``experiment.yml``.
    raw : dict[str, Any]
        Full parsed experiment mapping.
    save_directory : pathlib.Path
        Acquisition output directory.
    file_type : str
        Declared output file type from acquisition settings.
    microscope_name : str, optional
        Active microscope profile name used during acquisition.
    image_mode : str, optional
        Acquisition mode (e.g., ``"z-stack"``).
    timepoints : int
        Number of timepoints captured.
    number_z_steps : int
        Number of z slices per stack.
    y_pixels : int
        Image height in pixels.
    x_pixels : int
        Image width in pixels.
    multiposition_count : int
        Number of multiposition entries recorded.
    selected_channels : list[NavigateChannel]
        Channels marked as selected in acquisition state.
    """

    path: Path
    raw: Dict[str, Any]
    save_directory: Path
    file_type: str
    microscope_name: Optional[str]
    image_mode: Optional[str]
    timepoints: int
    number_z_steps: int
    y_pixels: int
    x_pixels: int
    multiposition_count: int
    selected_channels: list[NavigateChannel]

    @property
    def channel_count(self) -> int:
        """Return number of selected channels.

        Returns
        -------
        int
            Count of selected channels, defaulting to ``1``.
        """
        return max(1, len(self.selected_channels))

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert parsed experiment into JSON-friendly metadata.

        Returns
        -------
        dict[str, Any]
            Compact metadata mapping suitable for Zarr attrs/provenance.
        """
        return {
            "experiment_path": str(self.path),
            "save_directory": str(self.save_directory),
            "file_type": self.file_type,
            "microscope_name": self.microscope_name,
            "image_mode": self.image_mode,
            "timepoints": self.timepoints,
            "number_z_steps": self.number_z_steps,
            "multiposition_count": self.multiposition_count,
            "channel_count": self.channel_count,
            "selected_channels": [
                {
                    "name": channel.name,
                    "laser": channel.laser,
                    "laser_index": channel.laser_index,
                    "exposure_ms": channel.exposure_ms,
                    "is_selected": channel.is_selected,
                }
                for channel in self.selected_channels
            ],
        }


def is_navigate_experiment_file(path: Union[str, Path]) -> bool:
    """Return whether a path is a Navigate experiment descriptor file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` when basename is ``experiment.yml`` or ``experiment.yaml``.
    """
    name = Path(path).name.lower()
    return name in {"experiment.yml", "experiment.yaml"}


def _parse_serialized_text(text: str, *, context: str) -> Any:
    """Parse JSON text with optional YAML fallback.

    Parameters
    ----------
    text : str
        Input file content.
    context : str
        Human-readable context used in parse error messages.

    Returns
    -------
    Any
        Parsed Python object.

    Raises
    ------
    ValueError
        If text cannot be parsed.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not HAS_PYYAML:
            raise ValueError(
                f"{context} is not valid JSON and PyYAML is unavailable."
            )
        return yaml.safe_load(text)


def _parse_experiment_text(text: str) -> Dict[str, Any]:
    """Parse Navigate experiment text as JSON with YAML fallback.

    Parameters
    ----------
    text : str
        File content.

    Returns
    -------
    dict[str, Any]
        Parsed mapping.

    Raises
    ------
    ValueError
        If text cannot be parsed into a mapping.
    """
    parsed = _parse_serialized_text(text=text, context="experiment.yml")

    if not isinstance(parsed, dict):
        raise ValueError("Parsed experiment must be a mapping.")
    return parsed


def _looks_like_position_header(row: Any) -> bool:
    """Return whether a multiposition row is a header row.

    Parameters
    ----------
    row : Any
        Candidate row value.

    Returns
    -------
    bool
        ``True`` when the row appears to be a list/tuple of header strings.
    """
    if not isinstance(row, (list, tuple)):
        return False
    if not row:
        return False
    return all(isinstance(value, str) for value in row)


def _extract_position_rows_from_payload(payload: Any) -> Optional[list[Any]]:
    """Extract position rows from serialized multiposition payload.

    Parameters
    ----------
    payload : Any
        Parsed payload from ``multi_positions.yml``.

    Returns
    -------
    list[Any], optional
        Position rows with any header row removed, or ``None`` when payload
        does not contain an expected structure.
    """
    rows: Optional[list[Any]]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("positions", "MultiPositions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            rows = None
    else:
        rows = None

    if rows is None:
        return None
    if rows and _looks_like_position_header(rows[0]):
        return rows[1:]
    return rows


def _load_multiposition_rows(save_directory: Path) -> Optional[list[Any]]:
    """Load multiposition rows from ``multi_positions.yml`` when available.

    Parameters
    ----------
    save_directory : pathlib.Path
        Acquisition save directory containing experiment sidecar files.

    Returns
    -------
    list[Any], optional
        Parsed position rows without header row when available; otherwise
        ``None`` if file is missing or not parseable.
    """
    path = save_directory / "multi_positions.yml"
    if not path.exists():
        return None
    try:
        payload = _parse_serialized_text(
            text=path.read_text(), context="multi_positions.yml"
        )
        return _extract_position_rows_from_payload(payload)
    except Exception:
        return None


def _infer_multiposition_count(
    raw: Dict[str, Any],
    state: Dict[str, Any],
    save_directory: Path,
) -> int:
    """Infer position count from sidecar metadata and experiment payload.

    Parameters
    ----------
    raw : dict[str, Any]
        Parsed experiment payload.
    state : dict[str, Any]
        ``MicroscopeState`` mapping.
    save_directory : pathlib.Path
        Acquisition save directory.

    Returns
    -------
    int
        Position count with a minimum of ``1``.
    """
    is_multiposition = bool(state.get("is_multiposition", False))

    # Navigate records detailed position lists in the sidecar file.
    if is_multiposition:
        rows = _load_multiposition_rows(save_directory=save_directory)
        if rows is not None:
            return max(1, len(rows))

    fallback = raw.get("MultiPositions", [])
    if isinstance(fallback, list) and fallback:
        if _looks_like_position_header(fallback[0]):
            fallback = fallback[1:]
        return max(1, len(fallback))
    return 1


def load_navigate_experiment(path: Union[str, Path]) -> NavigateExperiment:
    """Load and normalize Navigate ``experiment.yml`` metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``experiment.yml``.

    Returns
    -------
    NavigateExperiment
        Parsed experiment metadata.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If file content is invalid or missing required structure.
    """
    experiment_path = Path(path).expanduser().resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(experiment_path)

    raw = _parse_experiment_text(experiment_path.read_text())

    saving = raw.get("Saving", {}) if isinstance(raw.get("Saving", {}), dict) else {}
    state = (
        raw.get("MicroscopeState", {})
        if isinstance(raw.get("MicroscopeState", {}), dict)
        else {}
    )
    camera = (
        raw.get("CameraParameters", {})
        if isinstance(raw.get("CameraParameters", {}), dict)
        else {}
    )

    save_directory = Path(
        saving.get("save_directory", str(experiment_path.parent))
    ).expanduser()
    if not save_directory.is_absolute():
        save_directory = (experiment_path.parent / save_directory).resolve()

    selected_channels: list[NavigateChannel] = []
    channels_obj = state.get("channels", {})
    if isinstance(channels_obj, dict):
        for channel_name in sorted(channels_obj.keys()):
            channel_value = channels_obj.get(channel_name, {})
            if not isinstance(channel_value, dict):
                continue
            if not bool(channel_value.get("is_selected", False)):
                continue
            selected_channels.append(
                NavigateChannel(
                    name=channel_name,
                    laser=(
                        str(channel_value.get("laser"))
                        if channel_value.get("laser") is not None
                        else None
                    ),
                    laser_index=(
                        int(channel_value["laser_index"])
                        if channel_value.get("laser_index") is not None
                        else None
                    ),
                    exposure_ms=(
                        float(channel_value["camera_exposure_time"])
                        if channel_value.get("camera_exposure_time") is not None
                        else None
                    ),
                    is_selected=True,
                )
            )

    multiposition_count = _infer_multiposition_count(
        raw=raw,
        state=state,
        save_directory=save_directory,
    )

    return NavigateExperiment(
        path=experiment_path,
        raw=raw,
        save_directory=save_directory,
        file_type=str(saving.get("file_type", "UNKNOWN")).upper(),
        microscope_name=(
            str(state.get("microscope_name"))
            if state.get("microscope_name") is not None
            else None
        ),
        image_mode=str(state.get("image_mode")) if state.get("image_mode") else None,
        timepoints=max(1, int(state.get("timepoints", 1))),
        number_z_steps=max(1, int(state.get("number_z_steps", 1))),
        y_pixels=max(
            1, int(camera.get("img_y_pixels", camera.get("y_pixels", 1)))
        ),
        x_pixels=max(
            1, int(camera.get("img_x_pixels", camera.get("x_pixels", 1)))
        ),
        multiposition_count=max(1, multiposition_count),
        selected_channels=selected_channels,
    )


def _h5_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort H5 acquisition files by channel and time when available.

    Parameters
    ----------
    path : pathlib.Path
        H5 file path.

    Returns
    -------
    tuple[int, int, str]
        Sort key tuple ``(channel_index, time_index, name)``.
    """
    match = re.match(r"CH(\d+)_(\d+)\.(?:h5|hdf5|hdf)$", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    return 9999, 9999, path.name


def _normalize_file_type(file_type: str) -> str:
    """Normalize acquisition file-type labels to canonical format tokens.

    Parameters
    ----------
    file_type : str
        Raw file-type value from acquisition metadata.

    Returns
    -------
    str
        Canonical token in ``{"TIFF", "H5", "N5", "ZARR", "UNKNOWN"}``.
    """
    token = str(file_type or "").strip().upper().replace("_", "-")
    token = token.replace(" ", "")

    if token in {
        "TIFF",
        "TIF",
        ".TIFF",
        ".TIF",
        "OME-TIFF",
        "OME-TIF",
        "OMETIFF",
        "OMETIF",
    }:
        return "TIFF"
    if token in {"H5", "HDF5", "HDF", ".H5", ".HDF5", ".HDF"}:
        return "H5"
    if token in {"N5", ".N5", "OME-N5", "OMEN5"}:
        return "N5"
    if token in {
        "ZARR",
        ".ZARR",
        "OME-ZARR",
        "OMEZARR",
        "OME-NGFF",
        "OMENGFF",
        "NGFF",
    }:
        return "ZARR"
    return "UNKNOWN"


def _is_mip_path(path: Path) -> bool:
    """Return whether a path appears to be a MIP/preview image.

    Parameters
    ----------
    path : pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` when path is located in a ``MIP`` directory or filename
        indicates a MIP artifact.
    """
    if "MIP" in path.name.upper():
        return True
    return any(part.upper() == "MIP" for part in path.parts)


def find_experiment_data_candidates(experiment: NavigateExperiment) -> list[Path]:
    """Find candidate acquisition data files/stores for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.

    Returns
    -------
    list[pathlib.Path]
        Candidate paths sorted by deterministic preference.
    """
    base = experiment.save_directory
    file_type = _normalize_file_type(experiment.file_type)

    if file_type == "H5":
        candidates = sorted(
            [p for p in base.glob("*") if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}],
            key=_h5_sort_key,
        )
        return candidates

    if file_type == "ZARR":
        return sorted([p for p in base.glob("*.zarr") if p.is_dir()])

    if file_type == "N5":
        return sorted([p for p in base.glob("*.n5") if p.is_dir()])

    if file_type == "TIFF":
        all_tiffs = sorted(
            {*base.rglob("*.tif"), *base.rglob("*.tiff")},
            key=lambda p: str(p),
        )
        primary = [p for p in all_tiffs if not _is_mip_path(p)]
        return primary or all_tiffs

    # Fallback: search known formats in priority order.
    fallback_tiffs = sorted(
        [p for p in {*base.rglob("*.tif"), *base.rglob("*.tiff")} if not _is_mip_path(p)],
        key=lambda p: str(p),
    )
    if not fallback_tiffs:
        fallback_tiffs = sorted(
            {*base.rglob("*.tif"), *base.rglob("*.tiff")},
            key=lambda p: str(p),
        )

    fallback = [
        *sorted([p for p in base.glob("*.zarr") if p.is_dir()]),
        *sorted([p for p in base.glob("*.n5") if p.is_dir()]),
        *sorted(
            [p for p in base.glob("*") if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}],
            key=_h5_sort_key,
        ),
        *fallback_tiffs,
    ]
    return fallback


def resolve_experiment_data_path(experiment: NavigateExperiment) -> Path:
    """Resolve primary acquisition data path from experiment metadata.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.

    Returns
    -------
    pathlib.Path
        Selected source data path.

    Raises
    ------
    FileNotFoundError
        If no compatible source data can be found.
    """
    candidates = find_experiment_data_candidates(experiment)
    if not candidates:
        raise FileNotFoundError(
            f"No source data candidates found in {experiment.save_directory} "
            f"for file_type={experiment.file_type}."
        )
    return candidates[0]


def default_analysis_store_path(experiment: NavigateExperiment) -> Path:
    """Return canonical 6D analysis Zarr store path for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.

    Returns
    -------
    pathlib.Path
        Path to canonical analysis store (``analysis_6d.zarr``).
    """
    return experiment.save_directory / "analysis_6d.zarr"


def infer_zyx_shape(
    experiment: NavigateExperiment, image_info: Optional[ImageInfo]
) -> tuple[int, int, int]:
    """Infer ``(z, y, x)`` shape for 6D analysis store allocation.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    image_info : ImageInfo, optional
        Metadata from source image reader.

    Returns
    -------
    tuple[int, int, int]
        Inferred spatial dimensions ``(z, y, x)``.
    """
    if image_info is not None:
        shape = tuple(int(v) for v in image_info.shape)
        if len(shape) >= 3:
            return shape[-3], shape[-2], shape[-1]
    return experiment.number_z_steps, experiment.y_pixels, experiment.x_pixels


def initialize_analysis_store(
    experiment: NavigateExperiment,
    zarr_path: Union[str, Path],
    *,
    image_info: Optional[ImageInfo] = None,
    overwrite: bool = False,
    chunks: tuple[int, int, int, int, int, int] = (1, 1, 1, 8, 256, 256),
    dtype: Optional[str] = None,
) -> Path:
    """Initialize canonical 6D analysis Zarr store for an experiment.

    Parameters
    ----------
    experiment : NavigateExperiment
        Parsed experiment metadata.
    zarr_path : str or pathlib.Path
        Output Zarr store path.
    image_info : ImageInfo, optional
        Source image metadata used for dtype/shape inference.
    overwrite : bool, default=False
        Whether to overwrite existing ``data`` array when present.
    chunks : tuple[int, int, int, int, int, int], default=(1,1,1,8,256,256)
        Target 6D chunking in ``(t, p, c, z, y, x)`` order.
    dtype : str, optional
        Explicit output dtype. Defaults to source dtype or ``uint16``.

    Returns
    -------
    pathlib.Path
        Resolved Zarr store path.
    """
    output_path = Path(zarr_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_size, y_size, x_size = infer_zyx_shape(experiment=experiment, image_info=image_info)
    shape = (
        int(experiment.timepoints),
        int(experiment.multiposition_count),
        int(experiment.channel_count),
        int(z_size),
        int(y_size),
        int(x_size),
    )

    if dtype is None:
        if image_info is not None:
            dtype = np.dtype(image_info.dtype).name
        else:
            dtype = "uint16"
    else:
        dtype = np.dtype(dtype).name

    normalized_chunks = tuple(
        min(int(chunk), int(dim)) for chunk, dim in zip(chunks, shape, strict=False)
    )

    root = zarr.open_group(str(output_path), mode="a")
    if "data" in root:
        if overwrite:
            del root["data"]
        else:
            existing = root["data"]
            existing.attrs.update(
                {
                    "axes": ["t", "p", "c", "z", "y", "x"],
                    "storage_policy": "latest_only",
                }
            )
            return output_path

    root.create_dataset(
        name="data",
        shape=shape,
        chunks=normalized_chunks,
        dtype=dtype,
        overwrite=True,
    )
    root["data"].attrs.update(
        {
            "axes": ["t", "p", "c", "z", "y", "x"],
            "storage_policy": "latest_only",
        }
    )
    root.require_group("results")
    root.require_group("provenance")
    root.attrs.update(
        {
            "schema": "clearex.analysis_store.v1",
            "axes": ["t", "p", "c", "z", "y", "x"],
            "source_experiment": str(experiment.path),
            "navigate_experiment": experiment.to_metadata_dict(),
            "storage_policy_analysis_outputs": "latest_only",
            "storage_policy_provenance": "append_only",
        }
    )
    return output_path


def create_dask_client(
    *,
    scheduler_address: Optional[str] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    memory_limit: Union[str, float] = "auto",
    local_directory: Optional[Union[str, Path]] = None,
):
    """Create Dask distributed client (local default, cluster optional).

    Parameters
    ----------
    scheduler_address : str, optional
        Existing Dask scheduler address for multi-node operation.
    n_workers : int, optional
        Number of local workers when using local mode.
    threads_per_worker : int, default=1
        Threads per worker for local mode.
    memory_limit : str or float, default="auto"
        Memory limit per worker for local mode.
    local_directory : str or pathlib.Path, optional
        Worker local directory for spills/temp files.

    Returns
    -------
    dask.distributed.Client
        Connected Dask client.

    Raises
    ------
    ImportError
        If ``dask.distributed`` is not available.
    """
    from dask.distributed import Client, LocalCluster

    if scheduler_address:
        return Client(scheduler_address)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit=memory_limit,
        local_directory=str(local_directory) if local_directory is not None else None,
    )
    return Client(cluster)


def write_zyx_block(
    zarr_path: Union[str, Path],
    block: ArrayLike,
    *,
    t_index: int,
    p_index: int,
    c_index: int,
    compute: bool = True,
):
    """Write one non-overlapping ``(z, y, x)`` block into 6D analysis store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Analysis Zarr store path.
    block : numpy.ndarray or dask.array.Array
        ``(z, y, x)`` volume to write.
    t_index : int
        Time index.
    p_index : int
        Position index.
    c_index : int
        Channel index.
    compute : bool, default=True
        If ``block`` is Dask, whether to execute the write immediately.

    Returns
    -------
    Any
        ``None`` for NumPy writes. For Dask writes, returns a delayed object
        when ``compute=False``.

    Raises
    ------
    ValueError
        If ``block`` is not 3D.
    TypeError
        If ``block`` type is unsupported.
    """
    if block.ndim != 3:
        raise ValueError(f"Expected 3D block (z, y, x), got shape={block.shape}.")

    z_size, y_size, x_size = (int(v) for v in block.shape)
    region = (
        slice(int(t_index), int(t_index) + 1),
        slice(int(p_index), int(p_index) + 1),
        slice(int(c_index), int(c_index) + 1),
        slice(0, z_size),
        slice(0, y_size),
        slice(0, x_size),
    )

    if isinstance(block, da.Array):
        block_6d = block[None, None, None, :, :, :]
        return da.to_zarr(
            block_6d,
            url=str(zarr_path),
            component="data",
            region=region,
            overwrite=False,
            compute=compute,
        )

    if isinstance(block, np.ndarray):
        root = zarr.open_group(str(zarr_path), mode="a")
        root["data"][region] = block[None, None, None, :, :, :]
        return None

    raise TypeError(
        f"Unsupported block type {type(block).__name__}; expected ndarray or Dask Array."
    )
