# MIP Export Agent Notes

This folder owns maximum-intensity projection (MIP) export from canonical
ClearEx OME-Zarr stores.

## Key Files

- `pipeline.py`
  - `run_mip_export_analysis`: top-level orchestration.
  - `_run_export_task`: one `(projection, t, c, [p])` export unit.
  - `_run_export_tile_task`: distributed tile-reduction worker task.
  - chunk/tile helpers: reduction- and preserved-axis memory controls.

## Input Contract

- Source data must be canonical 6D `(t, p, c, z, y, x)`.
- `input_source` selects a logical source alias or explicit internal image
  component (default: `data`).
- `position_mode`:
  - `per_position`: one file per `(projection, p, t, c)`.
  - `multi_position`: one file per `(projection, t, c)` with leading `p` axis.
- `export_format`:
  - `ome-tiff` (or legacy alias `tiff`): output stored as `uint16` with OME
    physical pixel-size metadata (`PhysicalSizeX/Y`).
  - `zarr`: output dtype follows source/projection dtype.

## Execution Model

- Tasks are enumerated across projections `xy/xz/yz`, time, channel, and
  optionally position.
- Local mode (`client=None`) uses delayed tasks with local scheduler.
- Distributed mode plans output tiles and submits `_run_export_tile_task` for
  each tile to the provided Dask client with bounded in-flight work.
- Driver process streams completed tiles into final output files (TIFF/Zarr)
  incrementally; this increases parallel work beyond one-future-per-output.

## Performance-Sensitive Rule

- In tile execution, reduction block sizing must be based on the current tile
  preserved-shape, not full projection output-shape.
- Reason: using full projection shape can force tiny reduction blocks (for
  example 1-4 slices) and produce apparent stalls where progress remains on a
  low tile index for a long time.
- Keep this behavior in both:
  - `_run_export_tile_task` (distributed tile execution), and
  - `_run_export_task` (local/process scheduler execution).
- If this regresses, first symptoms are:
  - extremely slow progress per tile despite active workers,
  - low CPU utilization with many queued tile tasks.

## Memory Model (Critical)

- `_run_export_task` must not read full 3D volumes into NumPy.
- Both `_run_export_task` and `_run_export_tile_task` read chunkwise along the
  reduced axis and tile across preserved axes.
- Read budget is controlled by `_MAX_REDUCTION_READ_BYTES` in `pipeline.py`.
- Projection outputs are written incrementally:
  - OME-TIFF uses `tifffile.memmap(..., ome=True, metadata=...)`.
  - Zarr writes into a pre-created dataset by tile via reopened group handles
    to avoid excessive simultaneously open files.
- This design prevents `np.asarray(source_array[..., :, :, :])` style
  allocations that can trigger worker deaths on large volumes.
- Zarr stores should be closed explicitly after read/write operations where
  possible (best-effort) to reduce file-descriptor pressure.

## Output + Provenance Contract

- Latest analysis metadata path: `clearex/results/mip_export/latest`.
- Large projection files are stored outside the analysis store in a `latest`
  output directory (configured or auto-generated).
- `register_latest_output_reference(...)` must be called with analysis key
  `mip_export` and component `clearex/results/mip_export/latest`.
- MIP export is metadata-only inside the OME-Zarr store; it does not publish a
  public OME image collection under `results/mip_export/latest`.

## Failure Patterns

- `numpy.core._exceptions._ArrayMemoryError`: usually indicates accidental
  full-volume materialization.
- `distributed.scheduler.KilledWorker`: typically worker OOM; verify chunk/tile
  budgeting and inspect worker memory limits/logs.
- `OSError: [Errno 24] Too many open files` (including `/proc/<pid>/stat`):
  indicates process-level FD exhaustion; check store-handle lifetime, process
  count, and OS/Slurm `nofile` limits.

## Safe Modification Checklist

1. Preserve axis mapping between projection type and source axes.
2. Keep chunkwise/tiled reads for all projections and both position modes.
3. Keep output writes non-overlapping and incremental.
4. If adding parameters, update normalization + GUI/workflow defaults and tests.

## Validation

- `uv run --extra dev ruff check src/clearex/mip_export/pipeline.py tests/mip_export/test_pipeline.py`
- `uv run --extra dev pytest -q tests/mip_export/test_pipeline.py`
- `uv run --extra dev pytest -q tests/test_main.py`
