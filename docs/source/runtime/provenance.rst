Provenance and Reproducibility
==============================

Provenance Goals
----------------

ClearEx provenance is designed to answer:

- what was run,
- with which exact parameters and backend settings,
- against which input data,
- producing which latest output references.

Storage Layout
--------------

Within a canonical OME-Zarr store:

- run records: ``clearex/provenance/runs/<run_id>``
- latest output references: ``clearex/provenance/latest_outputs/<analysis>``
- structured audit logs:
  ``clearex/provenance/event_logs/<run_id>.jsonl``
- public image collections: root source HCS collection and
  ``results/<analysis>/latest`` for image-producing analyses
- internal image outputs:
  ``clearex/runtime_cache/results/<analysis>/latest``
- ClearEx-owned non-image latest outputs:
  ``clearex/results/<analysis>/latest``

Run records are append-only. Large image outputs remain latest-only in the
runtime-cache/public-output split to control storage growth.

Run Record Content
------------------

``persist_run_provenance`` stores:

- run identifiers, index, status, and timestamps,
- input summary and input fingerprint hash,
- normalized workflow settings,
- effective Dask backend payload and chunk/pyramid settings,
- effective spatial-calibration payload and canonical text form,
- selected analyses and per-analysis parameters,
- ordered step records and output references,
- software metadata (package version, git commit/branch/dirty),
- environment fingerprint (Python/platform/lockfile hash/argv).
- structured audit-log manifest when a file-backed audit log was written.

Structured Audit Events
-----------------------

For file-backed workflow runs, ClearEx writes a sibling ``*.events.jsonl`` file
next to the plain text log. Each JSONL record contains a schema/version,
sequence number, UTC timestamp, execution id, optional run id, event type,
status, operation, progress, message, and redacted metadata.

The workflow dispatcher emits events for workflow start/end, input resolution,
store materialization, analysis sequence resolution, per-analysis
start/progress/skip/complete/failure/cancel states, and provenance persistence.
Sensitive credential-like keys such as tokens, passwords, secrets, API keys,
private keys, and authorization headers are replaced with ``[REDACTED]`` before
the record is written.

After provenance persistence succeeds, the final JSONL file is copied into the
canonical store under ``clearex/provenance/event_logs/<run_id>.jsonl``. The
latest run record is updated with the audit manifest and its hash is recomputed
so ``verify_provenance_chain`` still validates the run history.

Hash Chain Integrity
--------------------

Each run includes chained integrity values:

- ``prev_hash``: prior run hash,
- ``self_hash``: hash of current run record payload.

``verify_provenance_chain`` recomputes hashes and validates chain continuity,
returning ``(is_valid, issues)``.

Latest Output References
------------------------

``register_latest_output_reference`` writes lightweight metadata for discovery:

- analysis name,
- latest component path,
- update timestamp,
- optional producing ``run_id``,
- optional output metadata payload.

This decouples large arrays from the append-only provenance history while
keeping latest output pointers searchable.

Registration latest output references point to metadata-only results under
``clearex/results/registration/latest``. Affine transform/layout arrays remain
the stable compatibility contract; opt-in deformable runs also record the
deformable parameter payload and the optional displacement-lattice metadata
needed for reproducible fusion.

Spatial Placement Reproducibility
---------------------------------

Store-level Navigate placement metadata is part of the reproducibility record:

- canonical store metadata persists ``spatial_calibration`` in
  ``clearex/metadata``,
- workflow provenance stores the effective payload, canonical text form, and
  whether it was explicitly supplied by the operator,
- visualization latest metadata stores the effective spatial calibration used
  for multiposition placement.

This keeps historical runs interpretable even when microscope stage axes do not
match camera/world axes.

History Summaries and Dedup-Aware Execution
-------------------------------------------

``summarize_analysis_history`` provides per-analysis history summaries and
parameter-match checks. Runtime orchestration uses this for dedup-aware
execution:

- if a matching successful run exists and required output components are
  present, the operation can be skipped;
- if required outputs are missing, the operation is re-run.

Current runtime applies this matching logic to flatfield, deconvolution,
particle detection, registration, and other latest-only analysis steps.
