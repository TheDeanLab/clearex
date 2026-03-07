# Zarr Materialization Workflow and Verification

## Implemented workflow

The ingestion path for Navigate `experiment.yml` now materializes source data into a canonical 6D Zarr `data` array using Dask-native readers and parallel writes.

1. Load and normalize experiment metadata from `experiment.yml`.
2. Resolve source data path from acquisition outputs.
3. Open source directly with format-specific, Dask-compatible loaders:
   - TIFF/OME-TIFF via `tifffile` + `da.from_zarr(...)`
   - H5/HDF5 via `h5py` + `da.from_array(...)`
   - Zarr/N5 via `da.from_zarr(...)`
   - NPY/NPZ via `numpy` + `da.from_array(...)`
4. Infer/normalize axis order and coerce to canonical `(t, p, c, z, y, x)`.
5. Materialize with parallel Dask writes using GUI-configured chunking and pyramid metadata.
6. Persist canonical store metadata and source provenance fields in Zarr attributes.

## Store path behavior

- If source is already Zarr/N5, the same store path is reused (no duplicate store path is created).
- If source is non-Zarr, output store is written next to `experiment.yml` as:
  - `data_store.zarr`

## Dask backend behavior

- Runtime now uses the configured GUI backend as before.
- Local cluster startup for this I/O-heavy workflow is now thread-oriented (`processes=False`).
- Materialization compute executes on the active Dask client when available.
- For source graphs that cannot be serialized to distributed workers (for example some lock-backed TIFF/HDF inputs), execution automatically falls back to local threaded compute.

## Safety for in-place Zarr conversion

When source and destination are the same store and source component is already `data`, conversion stages into a temporary component and then swaps into `data` to avoid read/write self-conflicts.

## Automated verification

### Unit tests

Executed:

```bash
uv run --with pytest --with requests python -m pytest -q tests/test_workflow.py tests/io/test_cli.py tests/io/test_experiment.py tests/io/test_provenance.py
```

Result: `40 passed`

Added coverage validates:

- Non-Zarr source writes to `data_store.zarr` in `experiment.yml` directory.
- Existing Zarr source reuses same store path.
- Same-component (`data`) in-place Zarr rewrite path works.
- Canonical shape/chunk expectations and value integrity.

### Lint checks

Executed:

```bash
ruff check src/clearex/main.py src/clearex/workflow.py src/clearex/gui/app.py src/clearex/io/cli.py src/clearex/io/experiment.py src/clearex/io/provenance.py
```

Result: `All checks passed`

### Real-data run (provided dataset)

Dataset root:

- `/Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/`

Materialization run on representative cells (`cell_001` to `cell_005`) across TIFF, OME-TIFF, H5, N5, and OME-Zarr with chunks `(1, 1, 1, 8, 128, 128)`:

- `cell_001` TIFF -> `data_store.zarr`, shape `(1, 1, 1, 100, 2048, 2048)`, elapsed `1.97s`
- `cell_002` OME-TIFF -> `data_store.zarr`, shape `(1, 1, 1, 100, 2048, 2048)`, elapsed `1.85s`
- `cell_003` H5 -> `data_store.zarr`, shape `(1, 1, 1, 100, 2048, 2048)`, elapsed `26.02s`
- `cell_004` N5 -> same store `CH00_000000.n5`, shape `(1, 1, 1, 100, 2048, 2048)`, elapsed `3.01s`
- `cell_005` OME-Zarr -> same store `CH00_000000.ome.zarr`, shape `(2, 1, 1, 100, 2048, 2048)`, elapsed `3.33s`

This confirms output-path policy, canonical layout, and parallel read/write behavior on heterogeneous acquisition formats.

### Headless workflow smoke test

Executed:

```bash
uv run python -m clearex.main --headless --no-gui --file /Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/cell_001/experiment.yml --dask
```

Result:

- Workflow completed successfully.
- Source TIFF was materialized to `/Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/cell_001/data_store.zarr`.
- Provenance run record was persisted in the same store.
