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

"""Utilities for FAIR-aligned provenance and latest-only outputs in Zarr stores."""

from __future__ import annotations

# Standard Library Imports
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import hashlib
import json
import platform
import re
import subprocess
import sys
import uuid

# Third Party Imports
import dask.array as da
import numpy as np
import zarr

# Local Imports
from clearex.io.read import ImageInfo
from clearex.workflow import (
    WorkflowConfig,
    dask_backend_to_dict,
    format_dask_backend_summary,
    format_chunks,
    format_zarr_chunks_ptczyx,
    format_zarr_pyramid_ptczyx,
)

ArrayLike = Union[np.ndarray, da.Array]
_PROVENANCE_PARAMETER_COMPARE_EXCLUDE_KEYS = frozenset(
    {"execution_order", "force_rerun"}
)


def is_zarr_store_path(path: Union[str, Path]) -> bool:
    """Return whether a path appears to reference a Zarr/N5 store.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to evaluate.

    Returns
    -------
    bool
        ``True`` if the path suffix is ``.zarr`` or ``.n5``.
    """
    suffix = Path(path).suffix.lower()
    return suffix in {".zarr", ".n5"}


def _normalize_analysis_name(name: str) -> str:
    """Normalize analysis names for use as Zarr group keys.

    Parameters
    ----------
    name : str
        Analysis name.

    Returns
    -------
    str
        Lowercase key containing only ``[a-z0-9_.-]``.
    """
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip().lower())
    return normalized or "analysis"


def _to_jsonable(value: Any) -> Any:
    """Convert values to JSON-serializable structures.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    Any
        JSON-serializable value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Encode a mapping into canonical JSON for stable hashing.

    Parameters
    ----------
    value : Mapping[str, Any]
        Mapping to encode.

    Returns
    -------
    str
        Canonical JSON string with sorted keys.
    """
    return json.dumps(
        _to_jsonable(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _sha256_text(text: str) -> str:
    """Compute SHA-256 hash for text content.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> Optional[str]:
    """Hash a file when it exists.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    str, optional
        Hex digest if file exists, otherwise ``None``.
    """
    if not path.exists() or not path.is_file():
        return None

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_command(repo_root: Path, args: Sequence[str]) -> Optional[str]:
    """Run a Git command and return stdout.

    Parameters
    ----------
    repo_root : pathlib.Path
        Repository root to execute Git command in.
    args : sequence of str
        Arguments to pass after ``git``.

    Returns
    -------
    str, optional
        Stripped stdout if command succeeds, otherwise ``None``.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    """Collect Git metadata used in provenance records.

    Parameters
    ----------
    repo_root : pathlib.Path
        Repository root.

    Returns
    -------
    dict[str, Any]
        Dictionary with commit, branch, dirty state, and remote URL.
    """
    commit = _git_command(repo_root=repo_root, args=["rev-parse", "HEAD"])
    branch = _git_command(repo_root=repo_root, args=["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git_command(repo_root=repo_root, args=["status", "--porcelain"])
    remote = _git_command(repo_root=repo_root, args=["config", "--get", "remote.origin.url"])

    dirty: Optional[bool]
    if status is None:
        dirty = None
    else:
        dirty = bool(status)

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "remote": remote,
    }


def _clearex_version() -> Optional[str]:
    """Return installed ClearEx package version.

    Parameters
    ----------
    None

    Returns
    -------
    str, optional
        Installed package version, or ``None`` when unavailable.
    """
    try:
        return version("clearex")
    except PackageNotFoundError:
        return None


def _input_summary(workflow: WorkflowConfig, image_info: Optional[ImageInfo]) -> Dict[str, Any]:
    """Build input-summary metadata for provenance records.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration.
    image_info : ImageInfo, optional
        Loaded image metadata.

    Returns
    -------
    dict[str, Any]
        Input summary with fingerprint material and fingerprint hash.
    """
    axes_value = None
    if image_info and getattr(image_info, "axes", None) is not None:
        axes = getattr(image_info, "axes", None)
        axes_value = list(axes) if isinstance(axes, str) else axes

    pixel_size = getattr(image_info, "pixel_size", None) if image_info else None

    summary: Dict[str, Any] = {
        "path": workflow.file,
        "shape": [int(v) for v in (image_info.shape if image_info else ())],
        "dtype": str(image_info.dtype) if image_info else None,
        "axes": axes_value,
        "pixel_size": [float(v) for v in pixel_size] if pixel_size else None,
    }
    fingerprint = _sha256_text(_canonical_json(summary))
    summary["fingerprint_sha256"] = fingerprint
    return summary


def _selected_analyses(workflow: WorkflowConfig) -> list[str]:
    """Collect selected analysis names from workflow flags.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration.

    Returns
    -------
    list[str]
        Ordered list of selected analyses.
    """
    selected: list[str] = []
    if workflow.deconvolution:
        selected.append("deconvolution")
    if workflow.particle_detection:
        selected.append("particle_detection")
    if workflow.registration:
        selected.append("registration")
    if workflow.visualization:
        selected.append("visualization")
    return selected


def _default_steps(workflow: WorkflowConfig) -> list[Dict[str, Any]]:
    """Generate default step list for run records.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration.

    Returns
    -------
    list of dict[str, Any]
        Ordered workflow steps.
    """
    steps: list[Dict[str, Any]] = []
    if workflow.file:
        steps.append(
            {
                "name": "load_data",
                "parameters": {
                    "prefer_dask": workflow.prefer_dask,
                    "chunks": format_chunks(workflow.chunks) or None,
                    "dask_backend_summary": format_dask_backend_summary(
                        workflow.dask_backend
                    ),
                    "dask_backend": dask_backend_to_dict(workflow.dask_backend),
                    "zarr_chunks_ptczyx": format_zarr_chunks_ptczyx(
                        workflow.zarr_save.chunks_ptczyx
                    ),
                    "zarr_pyramid_ptczyx": format_zarr_pyramid_ptczyx(
                        workflow.zarr_save.pyramid_ptczyx
                    ),
                },
            }
        )
    for analysis_name in _selected_analyses(workflow):
        steps.append({"name": analysis_name, "parameters": {}})
    return steps


def _default_outputs(workflow: WorkflowConfig) -> Dict[str, Any]:
    """Build default output references for selected analyses.

    Parameters
    ----------
    workflow : WorkflowConfig
        Workflow configuration.

    Returns
    -------
    dict[str, Any]
        Output references using a latest-only storage policy.
    """
    outputs: Dict[str, Any] = {}
    for analysis_name in _selected_analyses(workflow):
        key = _normalize_analysis_name(analysis_name)
        outputs[key] = {
            "component": f"results/{key}/latest",
            "storage_policy": "latest_only",
        }
    return outputs


def _analysis_parameters_from_run_record(
    record: Mapping[str, Any],
    *,
    analysis_name: str,
) -> Optional[Dict[str, Any]]:
    """Extract per-analysis parameter mapping from one run record.

    Parameters
    ----------
    record : mapping[str, Any]
        Provenance run record payload.
    analysis_name : str
        Analysis operation name.

    Returns
    -------
    dict[str, Any], optional
        Analysis parameter mapping when present.
    """
    workflow = record.get("workflow")
    if not isinstance(workflow, Mapping):
        return None
    analysis_parameters = workflow.get("analysis_parameters")
    if not isinstance(analysis_parameters, Mapping):
        return None
    params = analysis_parameters.get(str(analysis_name))
    if not isinstance(params, Mapping):
        return None
    return dict(params)


def _comparison_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Prepare analysis parameters for provenance equality comparison.

    Parameters
    ----------
    parameters : mapping[str, Any]
        Candidate parameter mapping.

    Returns
    -------
    dict[str, Any]
        Comparison-safe parameter mapping with non-output-affecting controls removed.
    """
    return {
        str(key): value
        for key, value in dict(parameters).items()
        if str(key) not in _PROVENANCE_PARAMETER_COMPARE_EXCLUDE_KEYS
    }


def _run_record_analysis_step_completed(
    record: Mapping[str, Any],
    *,
    analysis_name: str,
) -> bool:
    """Return whether a run record completed a specific analysis operation.

    Parameters
    ----------
    record : mapping[str, Any]
        Provenance run record payload.
    analysis_name : str
        Analysis operation name.

    Returns
    -------
    bool
        ``True`` when run status is completed and the step is not skipped.
    """
    if str(record.get("status", "")).strip().lower() != "completed":
        return False

    normalized_name = _normalize_analysis_name(str(analysis_name))
    steps = record.get("steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_name = _normalize_analysis_name(str(step.get("name", "")))
        if step_name != normalized_name:
            continue
        params = step.get("parameters")
        if isinstance(params, Mapping):
            if str(params.get("status", "")).strip().lower() == "skipped":
                return False
        return True
    return False


def summarize_analysis_history(
    zarr_path: Union[str, Path],
    analysis_name: str,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize successful-run history for an analysis operation.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to a Zarr/N5 store.
    analysis_name : str
        Analysis operation name.
    parameters : mapping[str, Any], optional
        Candidate parameter mapping used to detect exact prior matches.

    Returns
    -------
    dict[str, Any]
        History summary with keys:
        ``has_successful_run``, ``latest_success_run_id``,
        ``latest_success_ended_utc``, ``matches_parameters``,
        ``matching_run_id``, ``matching_ended_utc``.

    Raises
    ------
    ValueError
        If ``zarr_path`` is not a Zarr/N5 path.
    """
    if not is_zarr_store_path(zarr_path):
        raise ValueError(f"Path is not a Zarr/N5 store: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="r")
    provenance_group = root.get("provenance")
    if provenance_group is None or "runs" not in provenance_group:
        return {
            "has_successful_run": False,
            "latest_success_run_id": None,
            "latest_success_ended_utc": None,
            "matches_parameters": False,
            "matching_run_id": None,
            "matching_ended_utc": None,
        }

    runs_group = provenance_group["runs"]
    run_entries: list[tuple[int, str, Dict[str, Any]]] = []
    for run_id in runs_group.group_keys():
        record = dict(runs_group[run_id].attrs.get("record", {}))
        run_index = int(record.get("run_index", 0))
        run_entries.append((run_index, str(run_id), record))
    run_entries.sort(key=lambda item: item[0], reverse=True)

    candidate_json: Optional[str] = None
    if parameters is not None:
        candidate_json = _canonical_json(
            _to_jsonable(_comparison_parameters(dict(parameters)))
        )

    latest_success_run_id: Optional[str] = None
    latest_success_ended_utc: Optional[str] = None
    matching_run_id: Optional[str] = None
    matching_ended_utc: Optional[str] = None

    for _, run_id, record in run_entries:
        if not _run_record_analysis_step_completed(
            record,
            analysis_name=analysis_name,
        ):
            continue

        ended_utc = None
        timestamps = record.get("timestamps")
        if isinstance(timestamps, Mapping):
            ended_value = timestamps.get("ended_utc")
            if ended_value is not None:
                ended_utc = str(ended_value)

        if latest_success_run_id is None:
            latest_success_run_id = str(run_id)
            latest_success_ended_utc = ended_utc

        if candidate_json is not None:
            record_parameters = _analysis_parameters_from_run_record(
                record,
                analysis_name=analysis_name,
            )
            if record_parameters is not None:
                record_json = _canonical_json(
                    _to_jsonable(_comparison_parameters(record_parameters))
                )
                if record_json == candidate_json:
                    matching_run_id = str(run_id)
                    matching_ended_utc = ended_utc
                    break

    return {
        "has_successful_run": latest_success_run_id is not None,
        "latest_success_run_id": latest_success_run_id,
        "latest_success_ended_utc": latest_success_ended_utc,
        "matches_parameters": matching_run_id is not None,
        "matching_run_id": matching_run_id,
        "matching_ended_utc": matching_ended_utc,
    }


def register_latest_output_reference(
    zarr_path: Union[str, Path],
    analysis_name: str,
    component: str,
    *,
    run_id: Optional[str] = None,
    output_hash: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register metadata for latest output location of an analysis.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to Zarr/N5 store.
    analysis_name : str
        Analysis name.
    component : str
        Zarr component path that points to latest output.
    run_id : str, optional
        Provenance run identifier that produced this output.
    output_hash : str, optional
        Optional hash for output integrity tracking.
    metadata : mapping, optional
        Additional JSON-serializable metadata.

    Returns
    -------
    None
        Metadata is written in-place to the Zarr store.

    Raises
    ------
    ValueError
        If ``zarr_path`` is not a Zarr/N5 path.
    """
    if not is_zarr_store_path(zarr_path):
        raise ValueError(f"Path is not a Zarr/N5 store: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="a")
    provenance_group = root.require_group("provenance")
    latest_outputs_group = provenance_group.require_group("latest_outputs")
    key = _normalize_analysis_name(analysis_name)
    latest_group = latest_outputs_group.require_group(key)

    payload: Dict[str, Any] = {
        "analysis_name": analysis_name,
        "component": component,
        "storage_policy": "latest_only",
        "updated_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    if run_id:
        payload["run_id"] = run_id
    if output_hash:
        payload["output_hash"] = output_hash
    if metadata:
        payload["metadata"] = _to_jsonable(dict(metadata))

    latest_group.attrs.update(_to_jsonable(payload))


def store_latest_analysis_output(
    zarr_path: Union[str, Path],
    analysis_name: str,
    output_array: ArrayLike,
    *,
    run_id: Optional[str] = None,
    chunks: Optional[Union[int, tuple[int, ...]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Store only the latest version of a large analysis output array.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to Zarr/N5 store.
    analysis_name : str
        Analysis name used for output namespace.
    output_array : numpy.ndarray or dask.array.Array
        Array to write as latest output.
    run_id : str, optional
        Provenance run identifier that produced this output.
    chunks : int or tuple of int, optional
        Chunk specification. Applied for NumPy writes and optional for Dask
        writes through rechunking.
    metadata : mapping, optional
        Additional metadata to attach to output reference record.

    Returns
    -------
    str
        Zarr component path where latest output was written.

    Raises
    ------
    ValueError
        If ``zarr_path`` is not a Zarr/N5 path.
    TypeError
        If ``output_array`` is neither NumPy nor Dask array.

    Notes
    -----
    Output path is always ``results/<analysis>/latest`` and is overwritten,
    ensuring only one large output version is retained.
    """
    if not is_zarr_store_path(zarr_path):
        raise ValueError(f"Path is not a Zarr/N5 store: {zarr_path}")
    if not isinstance(output_array, (np.ndarray, da.Array)):
        raise TypeError(
            "output_array must be a NumPy ndarray or Dask Array, "
            f"got {type(output_array).__name__}."
        )

    key = _normalize_analysis_name(analysis_name)
    component = f"results/{key}/latest"

    if isinstance(output_array, da.Array):
        data = output_array.rechunk(chunks) if chunks is not None else output_array
        da.to_zarr(data, url=str(zarr_path), component=component, overwrite=True)
    else:
        root = zarr.open_group(str(zarr_path), mode="a")
        results_group = root.require_group("results")
        analysis_group = results_group.require_group(key)
        if "latest" in analysis_group:
            del analysis_group["latest"]
        analysis_group.create_dataset(
            name="latest",
            data=output_array,
            chunks=chunks,
            overwrite=True,
        )

    root = zarr.open_group(str(zarr_path), mode="a")
    analysis_group = root.require_group("results").require_group(key)
    analysis_group.attrs.update(
        _to_jsonable(
            {
                "storage_policy": "latest_only",
                "latest_component": component,
                "latest_shape": [int(v) for v in output_array.shape],
                "latest_dtype": str(output_array.dtype),
                "latest_updated_utc": datetime.now(tz=timezone.utc).isoformat(),
                "latest_run_id": run_id,
            }
        )
    )

    register_latest_output_reference(
        zarr_path=zarr_path,
        analysis_name=analysis_name,
        component=component,
        run_id=run_id,
        metadata=metadata,
    )
    return component


def persist_run_provenance(
    zarr_path: Union[str, Path],
    workflow: WorkflowConfig,
    image_info: Optional[ImageInfo],
    *,
    steps: Optional[Sequence[Mapping[str, Any]]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    status: str = "completed",
    started_at_utc: Optional[datetime] = None,
    ended_at_utc: Optional[datetime] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> str:
    """Append a provenance run record to a Zarr/N5 store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to Zarr/N5 store.
    workflow : WorkflowConfig
        Workflow configuration used for the run.
    image_info : ImageInfo, optional
        Input image metadata loaded for the run.
    steps : sequence of mapping, optional
        Explicit ordered analysis steps. Default steps are derived from
        workflow flags when omitted.
    outputs : mapping, optional
        Output references. Defaults to latest-only output components for
        selected analyses.
    status : str, default="completed"
        Run terminal status.
    started_at_utc : datetime, optional
        Run start timestamp in UTC.
    ended_at_utc : datetime, optional
        Run end timestamp in UTC.
    repo_root : str or pathlib.Path, optional
        Repository root for Git/environment metadata resolution.

    Returns
    -------
    str
        Newly created run identifier.

    Raises
    ------
    ValueError
        If ``zarr_path`` is not a Zarr/N5 path.
    """
    if not is_zarr_store_path(zarr_path):
        raise ValueError(f"Path is not a Zarr/N5 store: {zarr_path}")

    started = started_at_utc or datetime.now(tz=timezone.utc)
    ended = ended_at_utc or datetime.now(tz=timezone.utc)
    repository_root = Path(repo_root) if repo_root is not None else Path.cwd()

    root = zarr.open_group(str(zarr_path), mode="a")
    provenance_group = root.require_group("provenance")
    runs_group = provenance_group.require_group("runs")

    run_id = uuid.uuid4().hex
    previous_hash = provenance_group.attrs.get("latest_hash")
    run_count = int(provenance_group.attrs.get("run_count", 0))
    run_index = run_count + 1

    step_records = (
        [_to_jsonable(dict(step)) for step in steps]
        if steps is not None
        else _default_steps(workflow)
    )
    output_records = (
        _to_jsonable(dict(outputs)) if outputs is not None else _default_outputs(workflow)
    )

    workflow_payload = {
        "file": workflow.file,
        "prefer_dask": workflow.prefer_dask,
        "dask_backend_summary": format_dask_backend_summary(workflow.dask_backend),
        "dask_backend": dask_backend_to_dict(workflow.dask_backend),
        "chunks": format_chunks(workflow.chunks) or None,
        "zarr_chunks_ptczyx": format_zarr_chunks_ptczyx(
            workflow.zarr_save.chunks_ptczyx
        ),
        "zarr_pyramid_ptczyx": format_zarr_pyramid_ptczyx(
            workflow.zarr_save.pyramid_ptczyx
        ),
        "deconvolution": workflow.deconvolution,
        "particle_detection": workflow.particle_detection,
        "registration": workflow.registration,
        "visualization": workflow.visualization,
        "selected_analyses": _selected_analyses(workflow),
        "analysis_parameters": _to_jsonable(workflow.analysis_parameters),
        "analysis_output_policy": "latest_only",
    }

    run_record: Dict[str, Any] = {
        "schema": "clearex.provenance.v1",
        "run_id": run_id,
        "run_index": run_index,
        "status": status,
        "timestamps": {
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
        },
        "input": _input_summary(workflow=workflow, image_info=image_info),
        "workflow": _to_jsonable(workflow_payload),
        "steps": step_records,
        "outputs": output_records,
        "software": {
            "package": "clearex",
            "version": _clearex_version(),
            "git": _git_metadata(repo_root=repository_root),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "lockfile_sha256": _file_sha256(repository_root / "uv.lock"),
            "argv": list(sys.argv),
        },
        "hash_chain": {
            "prev_hash": previous_hash,
        },
        "storage_policy": {
            "provenance": "append_only",
            "analysis_outputs": "latest_only",
        },
    }

    record_to_hash = _to_jsonable(run_record)
    self_hash = _sha256_text(_canonical_json(record_to_hash))
    run_record["hash_chain"]["self_hash"] = self_hash

    run_group = runs_group.require_group(run_id)
    run_group.attrs.update({"record": _to_jsonable(run_record)})

    provenance_group.attrs.update(
        {
            "schema": "clearex.provenance.v1",
            "latest_run_id": run_id,
            "latest_hash": self_hash,
            "run_count": run_index,
            "latest_updated_utc": ended.isoformat(),
        }
    )
    return run_id


def verify_provenance_chain(zarr_path: Union[str, Path]) -> tuple[bool, list[str]]:
    """Verify run-record hash chain stored in a Zarr/N5 store.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to Zarr/N5 store.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, issues)`` where ``issues`` is empty when valid.

    Raises
    ------
    ValueError
        If ``zarr_path`` is not a Zarr/N5 path.
    """
    if not is_zarr_store_path(zarr_path):
        raise ValueError(f"Path is not a Zarr/N5 store: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="r")
    provenance_group = root.get("provenance")
    if provenance_group is None or "runs" not in provenance_group:
        return True, []

    runs_group = provenance_group["runs"]
    run_entries: list[tuple[int, str, Dict[str, Any]]] = []

    for run_id in runs_group.group_keys():
        record = dict(runs_group[run_id].attrs.get("record", {}))
        run_index = int(record.get("run_index", 0))
        run_entries.append((run_index, run_id, record))

    run_entries.sort(key=lambda item: item[0])
    issues: list[str] = []
    previous_hash: Optional[str] = None

    for _, run_id, record in run_entries:
        hash_chain = dict(record.get("hash_chain", {}))
        stored_self_hash = hash_chain.get("self_hash")
        stored_prev_hash = hash_chain.get("prev_hash")

        if stored_prev_hash != previous_hash:
            issues.append(
                f"Run {run_id}: prev_hash mismatch "
                f"(expected {previous_hash}, found {stored_prev_hash})."
            )

        hash_chain.pop("self_hash", None)
        record_for_hash = dict(record)
        record_for_hash["hash_chain"] = hash_chain
        recomputed = _sha256_text(_canonical_json(_to_jsonable(record_for_hash)))
        if recomputed != stored_self_hash:
            issues.append(
                f"Run {run_id}: self_hash mismatch "
                f"(expected {stored_self_hash}, recomputed {recomputed})."
            )
        previous_hash = stored_self_hash

    return len(issues) == 0, issues
