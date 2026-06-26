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

"""Structured audit-event logging utilities for ClearEx workflow runs."""

from __future__ import annotations

# Standard Library Imports
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO
import hashlib
import json
import logging
import os
import shutil
import socket
import uuid

AUDIT_EVENT_SCHEMA = "clearex.audit_event.v1"
AUDIT_LOG_MANIFEST_SCHEMA = "clearex.audit_log_manifest.v1"
REDACTED_VALUE = "[REDACTED]"
SENSITIVE_KEY_FRAGMENTS = (
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "access_key",
    "private_key",
    "credential",
)


def audit_log_path_for_text_log(text_log_path: os.PathLike[str] | str) -> Path:
    """Return the sibling structured event-log path for a text log.

    Parameters
    ----------
    text_log_path : os.PathLike or str
        Existing plain-text log file path.

    Returns
    -------
    pathlib.Path
        Sibling path using the same stem and an ``.events.jsonl`` suffix.
    """
    path = Path(text_log_path)
    return path.with_name(f"{path.stem}.events.jsonl")


def audit_log_path_for_logger(logger: logging.Logger) -> Optional[Path]:
    """Resolve a structured event-log path from a logger file handler.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for the workflow run.

    Returns
    -------
    pathlib.Path, optional
        Sibling JSONL path when a ``logging.FileHandler`` is attached to the
        provided logger or the root logger.
    """
    candidates = [logger]
    root_logger = logging.getLogger()
    if root_logger is not logger:
        candidates.append(root_logger)

    for candidate in candidates:
        for handler in candidate.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            base_filename = getattr(handler, "baseFilename", None)
            if base_filename:
                return audit_log_path_for_text_log(base_filename)
    return None


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 text form."""
    return datetime.now(tz=timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    """Compute the SHA-256 digest for a file.

    Parameters
    ----------
    path : pathlib.Path
        File path to hash.

    Returns
    -------
    str
        Hex digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sensitive_key(key: object) -> bool:
    """Return whether a mapping key should have its value redacted."""
    normalized = str(key).strip().lower().replace("-", "_")
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def _jsonable(value: Any) -> Any:
    """Convert values into JSON-serializable structures.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    Any
        JSON-compatible value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {
            "shape": [int(axis) for axis in shape],
            "dtype": str(dtype),
        }
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def redact_for_audit(value: Any) -> Any:
    """Return a JSON-compatible copy with sensitive values removed.

    Parameters
    ----------
    value : Any
        Value to sanitize for audit logging.

    Returns
    -------
    Any
        Redacted, JSON-compatible value.
    """
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            redacted[text_key] = (
                REDACTED_VALUE
                if _is_sensitive_key(text_key)
                else redact_for_audit(item)
            )
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [redact_for_audit(item) for item in value]
    return _jsonable(value)


def _error_payload(error: BaseException | Mapping[str, Any] | str) -> dict[str, Any]:
    """Build a redacted error payload for an audit event."""
    if isinstance(error, BaseException):
        return {
            "type": type(error).__name__,
            "message": str(error),
        }
    if isinstance(error, Mapping):
        redacted = redact_for_audit(error)
        return redacted if isinstance(redacted, dict) else {"message": str(redacted)}
    return {"message": str(error)}


def _count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL records in a file."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


class AuditEventWriter:
    """Write structured ClearEx audit events to a JSONL file.

    Parameters
    ----------
    path : os.PathLike or str
        Destination JSONL file.
    execution_id : str, optional
        Stable identifier for this process-local workflow execution. A random
        UUID hex value is generated when omitted.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        execution_id: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.execution_id = str(execution_id or uuid.uuid4().hex)
        self._sequence = 0
        self._event_count = 0
        self._run_id: Optional[str] = None
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("w", encoding="utf-8")

    def __enter__(self) -> "AuditEventWriter":
        """Return the active writer for context-manager use."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the writer when leaving a context manager."""
        self.close()

    @property
    def event_count(self) -> int:
        """Return the number of events emitted by this writer."""
        return self._event_count

    @property
    def run_id(self) -> Optional[str]:
        """Return the provenance run identifier linked to this writer."""
        return self._run_id

    def set_run_id(self, run_id: Optional[str]) -> None:
        """Associate later events and manifests with a provenance run.

        Parameters
        ----------
        run_id : str, optional
            Provenance run identifier.

        Returns
        -------
        None
            The writer's run identifier is updated in-place.
        """
        self._run_id = str(run_id) if run_id else None

    def emit(
        self,
        event_type: str,
        *,
        level: str = "INFO",
        operation: Optional[str] = None,
        status: Optional[str] = None,
        percent: Optional[int] = None,
        message: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        error: Optional[BaseException | Mapping[str, Any] | str] = None,
    ) -> dict[str, Any]:
        """Write one structured audit event.

        Parameters
        ----------
        event_type : str
            Event type name, for example ``"analysis.step.started"``.
        level : str, default="INFO"
            Logging severity.
        operation : str, optional
            Analysis operation associated with the event.
        status : str, optional
            Operation or workflow status.
        percent : int, optional
            Progress percentage, clamped to ``0..100``.
        message : str, optional
            Human-readable status message.
        metadata : mapping, optional
            Additional JSON-compatible event context. Sensitive keys are
            redacted recursively.
        error : exception, mapping, or str, optional
            Error context for failure events.

        Returns
        -------
        dict[str, Any]
            The record that was written.
        """
        if self._closed:
            raise RuntimeError("Cannot emit audit events after the writer is closed.")

        self._sequence += 1
        record: dict[str, Any] = {
            "schema": AUDIT_EVENT_SCHEMA,
            "sequence": self._sequence,
            "timestamp_utc": _utc_now(),
            "level": str(level).upper() or "INFO",
            "event_type": str(event_type),
            "execution_id": self.execution_id,
            "run_id": self._run_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        if operation is not None:
            record["operation"] = str(operation)
        if status is not None:
            record["status"] = str(status)
        if percent is not None:
            record["percent"] = max(0, min(100, int(percent)))
        if message is not None:
            record["message"] = str(message)
        if metadata is not None:
            record["metadata"] = redact_for_audit(dict(metadata))
        if error is not None:
            record["error"] = redact_for_audit(_error_payload(error))

        self._handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )
        self._handle.write("\n")
        self._handle.flush()
        self._event_count += 1
        return record

    def close(self) -> None:
        """Close the underlying JSONL file handle."""
        if self._closed:
            return
        self._handle.flush()
        self._handle.close()
        self._closed = True

    def finish(self, *, run_id: Optional[str] = None) -> dict[str, Any]:
        """Close the writer and return an audit-log manifest.

        Parameters
        ----------
        run_id : str, optional
            Provenance run identifier linked to the event log.

        Returns
        -------
        dict[str, Any]
            Manifest describing the structured event log.
        """
        if run_id:
            self.set_run_id(run_id)
        self.close()
        return build_audit_log_manifest(
            audit_log_path=self.path,
            execution_id=self.execution_id,
            run_id=self._run_id,
            event_count=self._event_count,
        )


def build_audit_log_manifest(
    *,
    audit_log_path: os.PathLike[str] | str,
    execution_id: str,
    run_id: Optional[str] = None,
    event_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build a manifest for a structured audit JSONL file.

    Parameters
    ----------
    audit_log_path : os.PathLike or str
        Audit JSONL path.
    execution_id : str
        Execution identifier recorded in the event log.
    run_id : str, optional
        Linked provenance run identifier.
    event_count : int, optional
        Known event count. The file is counted when omitted.

    Returns
    -------
    dict[str, Any]
        JSON-compatible audit-log manifest.
    """
    path = Path(audit_log_path)
    return {
        "schema": AUDIT_LOG_MANIFEST_SCHEMA,
        "execution_id": str(execution_id),
        "run_id": str(run_id) if run_id else None,
        "event_log_path": str(path),
        "event_count": int(
            event_count if event_count is not None else _count_jsonl_records(path)
        ),
        "sha256": _file_sha256(path),
        "redaction_policy": {
            "redacted_value": REDACTED_VALUE,
            "sensitive_key_fragments": list(SENSITIVE_KEY_FRAGMENTS),
        },
    }


def persist_audit_log_file_to_store(
    *,
    zarr_path: os.PathLike[str] | str,
    audit_log_path: os.PathLike[str] | str,
    run_id: str,
    execution_id: str,
) -> dict[str, Any]:
    """Copy an audit JSONL file into the ClearEx provenance namespace.

    Parameters
    ----------
    zarr_path : os.PathLike or str
        Local canonical store path.
    audit_log_path : os.PathLike or str
        Source JSONL audit file.
    run_id : str
        Provenance run identifier used for the copied filename.
    execution_id : str
        Execution identifier recorded in the event log.

    Returns
    -------
    dict[str, Any]
        Manifest including external and store-local paths.
    """
    source = Path(audit_log_path)
    store = Path(zarr_path)
    relative = Path("clearex") / "provenance" / "event_logs" / f"{run_id}.jsonl"
    destination = store / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

    manifest = build_audit_log_manifest(
        audit_log_path=source,
        execution_id=execution_id,
        run_id=run_id,
    )
    manifest["store_relative_path"] = relative.as_posix()
    manifest["store_path"] = str(destination)
    return manifest
