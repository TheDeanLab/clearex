#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path
import hashlib
import json

# Third Party Imports
import zarr

# Local Imports
from clearex.io.audit import (
    AuditEventWriter,
    audit_log_path_for_text_log,
    persist_audit_log_file_to_store,
)


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_audit_event_writer_writes_jsonl_manifest_and_redacts_sensitive_values(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "run.events.jsonl"

    with AuditEventWriter(event_path, execution_id="exec-1") as writer:
        writer.emit(
            "analysis.step.started",
            operation="deconvolution",
            message="Starting deconvolution",
            metadata={
                "parameters": {
                    "input_source": "data",
                    "authorization_token": "raw-token-value",
                    "nested": {"api_key": "raw-api-key"},
                },
                "path": tmp_path / "input.ome.zarr",
            },
        )
        manifest = writer.finish(run_id="run-1")

    records = _jsonl_records(event_path)

    assert len(records) == 1
    assert records[0]["schema"] == "clearex.audit_event.v1"
    assert records[0]["sequence"] == 1
    assert records[0]["execution_id"] == "exec-1"
    assert records[0]["run_id"] is None
    assert records[0]["event_type"] == "analysis.step.started"
    assert records[0]["operation"] == "deconvolution"
    metadata = records[0]["metadata"]
    assert isinstance(metadata, dict)
    parameters = metadata["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["authorization_token"] == "[REDACTED]"
    nested = parameters["nested"]
    assert isinstance(nested, dict)
    assert nested["api_key"] == "[REDACTED]"
    assert metadata["path"] == str(tmp_path / "input.ome.zarr")

    assert manifest["schema"] == "clearex.audit_log_manifest.v1"
    assert manifest["execution_id"] == "exec-1"
    assert manifest["run_id"] == "run-1"
    assert manifest["event_count"] == 1
    assert manifest["event_log_path"] == str(event_path)
    assert manifest["sha256"] == hashlib.sha256(event_path.read_bytes()).hexdigest()
    assert "secret" in manifest["redaction_policy"]["sensitive_key_fragments"]


def test_audit_log_path_for_text_log_uses_sibling_jsonl_path(tmp_path: Path) -> None:
    text_log = tmp_path / "2026-06-26-host-123.log"

    assert audit_log_path_for_text_log(text_log) == (
        tmp_path / "2026-06-26-host-123.events.jsonl"
    )


def test_persist_audit_log_file_to_store_copies_jsonl_under_provenance(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "audit_store.ome.zarr"
    event_path = tmp_path / "run.events.jsonl"
    event_path.write_text('{"event_type":"workflow.started"}\n')
    zarr.open_group(str(store_path), mode="w")

    manifest = persist_audit_log_file_to_store(
        zarr_path=store_path,
        audit_log_path=event_path,
        run_id="run-1",
        execution_id="exec-1",
    )

    copied_path = store_path / "clearex" / "provenance" / "event_logs" / "run-1.jsonl"
    assert copied_path.read_text() == event_path.read_text()
    assert (
        manifest["store_relative_path"] == "clearex/provenance/event_logs/run-1.jsonl"
    )
    assert manifest["store_path"] == str(copied_path)
    assert manifest["event_count"] == 1
    assert manifest["sha256"] == hashlib.sha256(event_path.read_bytes()).hexdigest()
