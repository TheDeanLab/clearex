# IO Agent Notes

This folder contains ingestion, CLI, logging, and provenance logic.

## Key Files

- `experiment.py`: Navigate `experiment.yml` parsing, source resolution, canonical store materialization.
- `provenance.py`: run record persistence and latest-output references.
- `cli.py`: headless flags and parser behavior.
- `log.py`: logger setup and output location.

## Data-Store Rules

- For non-Zarr/N5 sources: create `data_store.zarr` beside `experiment.yml`.
- For Zarr/N5 sources: reuse existing store path (no duplicate copy path).
- Canonical base array component is `data` with shape `(t, p, c, z, y, x)`.

## Materialization Rules

- Use GUI/Workflow chunk and pyramid configuration as the source of truth.
- Prefer chunk-parallel, non-overlapping writes.
- Preserve metadata required for downstream analysis/provenance.

## Dask Client Defaults

- `create_dask_client(...)` local-mode startup now defaults to `dashboard_address=":0"`.
- This avoids noisy `Port 8787 is already in use` warnings when multiple local clusters are started over time.
- Callers can still override `dashboard_address` explicitly when fixed dashboard ports are required.

## Provenance Rules

- Provenance is append-only.
- Large analysis outputs use latest-only storage under `results/<analysis>/latest`.
- Persist effective backend config and analysis parameters.
- Register `latest` output references for discoverability.

## Logging Rules

- Runtime logs should initialize in the canonical store directory context.
- Keep failure messages explicit (source path, component, reason).

## Validation

- `uv run ruff check src/clearex/io/cli.py src/clearex/io/experiment.py src/clearex/io/provenance.py`
- `uv run --with pytest --with requests python -m pytest -q tests/io/test_cli.py tests/io/test_experiment.py tests/io/test_provenance.py`
