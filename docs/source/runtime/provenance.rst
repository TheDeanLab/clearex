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

Within a canonical Zarr/N5 store:

- run records: ``provenance/runs/<run_id>``
- latest output references: ``provenance/latest_outputs/<analysis>``

Run records are append-only. Large analysis outputs remain latest-only under
``results/<analysis>/latest`` to control storage growth.

Run Record Content
------------------

``persist_run_provenance`` stores:

- run identifiers/index/status/timestamps,
- input summary and input fingerprint hash,
- normalized workflow settings,
- effective Dask backend payload and chunk/pyramid settings,
- selected analyses and per-analysis parameters,
- ordered step records and output references,
- software metadata (package version, git commit/branch/dirty),
- environment fingerprint (Python/platform/lockfile hash/argv).

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

History Summaries and Dedup-Aware Execution
-------------------------------------------

``summarize_analysis_history`` provides per-analysis history summaries and
parameter-match checks. Runtime orchestration uses this for dedup-aware
execution:

- if a matching successful run exists and required output components are
  present, the operation can be skipped;
- if required outputs are missing, the operation is re-run.

Current runtime applies this matching logic to flatfield, deconvolution,
particle detection, and registration steps.
