# ClearEx Agent Strategy

This file summarizes the current engineering strategy for agent-driven changes in this repository.

## Scope

- Primary entrypoint: `src/clearex/main.py`
- Shared runtime config/parsing: `src/clearex/workflow.py`
- GUI layer: `src/clearex/gui/app.py`
- CLI parser: `src/clearex/io/cli.py`
- Navigate experiment ingestion + 6D analysis store: `src/clearex/io/experiment.py`
- Provenance + latest-output policy: `src/clearex/io/provenance.py`
- Tests for strategy-critical behavior:
  - `tests/test_workflow.py`
  - `tests/io/test_cli.py`
  - `tests/io/test_experiment.py`
  - `tests/io/test_provenance.py`

## Execution Model

- Default mode is GUI-first.
- Headless operation must remain first-class and easy to run.
- GUI-selected Dask backend is the execution engine for ingestion, loading, and all analysis operations.
- CLI controls:
  - `--gui` / `--no-gui`
  - `--headless` (overrides GUI launch)
  - `--file`, `--dask`, `--chunks`
  - `--deconvolution`, `--particle-detection`, `--registration`, `--visualization`
- GUI launch failures (missing display, missing Qt bindings, runtime issues) should gracefully fall back to headless mode with clear logs.
- `experiment.yml` is a first-class input path:
  - user may select `experiment.yml` in GUI/CLI,
  - backend resolves source acquisition data path from experiment metadata,
  - backend initializes/uses canonical 6D analysis Zarr store.
- First operation for acquisition inputs is canonical Zarr establishment:
  - ingest/convert acquisition data into canonical Zarr analysis store,
  - apply GUI-selected chunking + pyramid configuration during ingest,
  - then run analysis steps against that established store.
- If source is already Zarr/N5, prefer reusing that existing directory/object store path and augment it with ClearEx analysis/provenance layout (do not force duplicate copies).

## `workflow.py` Contract

- `WorkflowConfig` is the shared schema used by both GUI and headless paths.
- `WorkflowConfig` carries both storage config (`ZarrSaveConfig`) and backend config (`DaskBackendConfig`) so GUI and runtime stay aligned.
- `parse_chunks` is the single source of truth for validating chunk specs.
- `format_chunks` is the single source of truth for rendering chunk specs back into UI text.
- Dask backend and Zarr-save helpers (validation, summary, serialization) should stay centralized in `workflow.py`.
- Keep parsing/formatting logic centralized here to avoid drift between CLI and GUI behavior.

## GUI Strategy

- GUI is implemented in `src/clearex/gui/app.py`.
- UI responsibilities:
  - data path selection (file/folder),
  - support direct selection of Navigate `experiment.yml`,
  - metadata loading via `ImageOpener`,
  - metadata display (path, shape, dtype, axes, channels, positions, image size, time points, pixel size, metadata keys),
  - Dask backend configuration popup:
    - `LocalCluster`, `SLURMRunner`, `SLURMCluster`,
    - mode guidance text for operators,
    - required scheduler file for `SLURMRunner`,
    - required operator email input for `SLURMCluster` notifications,
  - Zarr save configuration popup:
    - chunk sizes in `(p, t, c, z, y, x)`,
    - pyramid factors per axis,
  - workflow selection toggles.
- Metadata panel should stay operator-friendly and compact. Current design uses a 2-column key/value layout.
- Visual direction: clean modern dark theme.
- GUI layer should only collect/validate user intent and return a `WorkflowConfig`; core execution remains in `main.py`.

## Navigate Experiment Strategy

- Prefer using Navigate `experiment.yml` as acquisition entrypoint metadata.
- Normalize acquisition file-type aliases (for example `OME-TIFF`, `OME-ZARR`) to canonical resolver targets.
- Parse and persist key fields:
  - file type (`Saving.file_type`)
  - save directory (`Saving.save_directory`)
  - timepoints, z steps, selected channels, multiposition count
  - camera image dimensions
- For multiposition acquisitions (`MicroscopeState.is_multiposition == true`), use `multi_positions.yml` in the same directory as `experiment.yml` as the authoritative source of position rows/count.
- Resolve source data path from experiment metadata and on-disk candidates.
- For TIFF-family acquisitions, prefer primary stack files and avoid selecting `MIP` preview outputs as source data.
- Zarr/N5 ingestion should support nested group layouts (not only root-level arrays).
- Ingestion should target directory/object-store-friendly Zarr layouts (many chunk objects/files) to maximize safe parallel writes without single-file-handle contention.
- Canonical analysis store path: `analysis_6d.zarr` in experiment save directory.
- Canonical array layout: `(t, p, c, z, y, x)`.
- For new conversion, chunking and pyramid factors come from GUI/backend `WorkflowConfig`.
- After canonical data array ingest/establishment, treat base image data as read-only for downstream analysis stages.

## Parallelism Strategy

- Supported Dask backend modes:
  - `LocalCluster` for local distributed execution,
  - `SLURMRunner` for attaching to scheduler-file-driven cluster sessions,
  - `SLURMCluster` for direct Slurm jobqueue worker launch from ClearEx.
- `SLURMCluster` configuration should expose and persist key job/scheduler fields (cores/processes/memory/interface/walltime/queue/job directives/dashboard/idle timeout/allowed failures).
- Operator-provided email must be used for Slurm notification directives (never hardcode personal addresses).
- Backend selection should be honored for ingestion, loading, and analysis execution.
- Zarr writes should be parallelized only for non-overlapping regions/chunks.
- Large outputs follow latest-only replacement semantics (`results/<analysis>/latest`), while provenance remains append-only.

## Analysis Hooks

- Current selectable operations:
  - deconvolution,
  - particle detection,
  - registration,
  - visualization.
- Registration and visualization have runtime hooks.
- Deconvolution and particle detection are currently scaffolded as hooks with logging placeholders.

## Provenance Strategy (FAIR)

- Goal: persist exactly how an analysis was run so it can be inspected and reproduced.
- Default behavior: provenance-first, only write the most recent version of the large derived image arrays. Do not continuously append each iteration of the derived arrays to provenance records.

### Required record content per run

- run identifier (`run_id`) and UTC timestamps (start/end)
- ordered analysis steps and exact parameter values
- input data locator and input fingerprint/hash
- effective Dask backend mode + parameters
- effective Zarr ingest/save chunk + pyramid settings
- software identity:
  - git commit SHA
  - git branch
  - git dirty state (uncommitted local changes)
- runtime environment fingerprint (for example Python version + lockfile hash)
- random seeds (where relevant)

### Storage policy

- Co-locate provenance with dataset when possible (for example in Zarr metadata paths), while keeping records lightweight.
- Prefer append-only provenance records; do not overwrite prior run records.
- Keep a `latest` pointer/reference for convenience, but preserve full history.
- Base ingested source image data in canonical store is immutable after ingest; analysis products evolve independently.
- Large analysis arrays are not append-only by default; they are overwritten at canonical latest paths to control storage growth.

### Integrity policy

- Use tamper-evident records:
  - content hash per record (`self_hash`)
  - chained hash to prior record (`prev_hash`) when applicable
- Verification should be available during load/replay and emit clear warnings on mismatch.

### Reproducibility policy

- Replay should reconstruct the workflow from provenance without requiring manual parameter recovery.
- If outputs are stored, keep them reference-based (URI/path + hash) rather than duplicating full arrays by default.

## Documentation and Typing Standards

For all new or changed public/internal functions in this workflow stack:

- Use explicit type hints for parameters and return values.
- Use numpydoc docstrings with, at minimum:
  - `Parameters`
  - `Returns`
  - `Raises` (or explicit note when exceptions are handled internally)
- Include `Notes` when behavior has fallback logic, side effects, or environment constraints.

## Quality Gate

Before finishing changes in this area, run:

- `ruff check src/clearex/main.py src/clearex/workflow.py src/clearex/gui/app.py src/clearex/io/cli.py src/clearex/io/experiment.py src/clearex/io/provenance.py`
- `pytest -q tests/test_workflow.py tests/io/test_cli.py tests/io/test_experiment.py tests/io/test_provenance.py`

If a change touches metadata parsing or UI rendering logic, add/adjust targeted tests accordingly.

## Dependency Guidance

- GUI implementation targets `PyQt6`.
- Avoid introducing `PyQt5` requirements for the GUI layer.
- If dependency resolution breaks on macOS arm64, verify lockfile resolution for Qt-related transitive dependencies before changing runtime code.

## Future Roadmap (Not Implemented Yet)

- Provenance replay UX: load a prior run record directly into GUI/CLI form controls.
- End-to-end ingestion: robust conversion/mapping from all Navigate acquisition layouts into canonical `(t, p, c, z, y, x)` analysis data.
- Distributed orchestration: scheduler-aware chunk planning and retries for large multi-node runs.
- Keep this future capability in mind when evolving `WorkflowConfig` and execution wiring.
