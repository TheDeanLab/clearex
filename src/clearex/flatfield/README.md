# Flatfield Agent Notes

This folder owns BaSiCPy-driven flatfield correction for canonical ClearEx
stores.

## Key File

- `pipeline.py`
  - `run_flatfield_analysis`: top-level orchestration.
  - Fit stage: full-volume or tiled profile estimation.
  - Transform stage: chunked correction writes to latest output.
  - Resume/checkpoint + pyramid materialization + provenance registration.

## Input Contract

- Source must be canonical 6D `(t, p, c, z, y, x)`.
- `input_source` defaults to `data`.
- Core params:
  - `fit_mode`: `tiled` or `full_volume`.
  - `fit_tile_shape_yx`: tile size for tiled fitting.
  - `use_map_overlap`, `overlap_zyx`: halo behavior.
  - `blend_tiles`: feathered stitched tile artifacts (effective only when
    overlap is enabled).
  - `is_timelapse`: enables baseline subtraction in transform stage.

## Output Contract

- Latest output root: `results/flatfield/latest`
- Datasets:
  - `data` (corrected output, float32)
  - `flatfield_pcyx` (float32)
  - `darkfield_pcyx` (float32)
  - `baseline_pctz` (float32)
- Storage policy is latest-only for large arrays.
- `run_id` is backfilled later by main workflow/provenance path.

## Correction Formula

- Per transform chunk:
  - `corrected = (source - darkfield) / max(flatfield, 1e-6)`
  - If `is_timelapse=True`, subtract `baseline[:, z]` broadcast over `y,x`.
- Writes only non-overlapping core region for each transform task.

## Fit Modes

- `full_volume`:
  - Materializes one full `(t, z, y, x)` block per `(p, c)` profile.
  - Simpler path, higher memory risk for large volumes.
- `tiled`:
  - Fits per-tile on `(t, z, y_read, x_read)`.
  - Supports overlap halo and optional feather blending to reduce seams.
  - Baseline is aggregated from tile fits (`baseline_sum / tile_count`).

## Tiled-Fit Fallback Behavior (Important)

- If a tile fit fails with recoverable SVD convergence errors:
  - Retry that profile via full-volume fit.
  - If retry fails, use identity profile (`flatfield=1`, `darkfield=0`,
    `baseline=0`).
- Fallback metadata is tracked in checkpoint attrs:
  - `fit_warning_count`
  - `fit_fallback_records`
- Completed run status becomes `complete_with_warnings` when fallbacks occur.

## Resume / Checkpoint Contract

- Checkpoint group: `results/flatfield/latest/checkpoint`
- Schema guard: `clearex.flatfield.resume.v1`
- Resume is allowed only when all are compatible:
  - source component, shape, chunks
  - fit mode + tile/transform grid
  - blend mode
  - parameter fingerprint (`force_rerun` excluded from fingerprint payload)
  - expected checkpoint datasets with correct shape/dtype
  - representative chunk decode probes pass (including `(p,c)` scans for
    profile datasets) to catch malformed N5 chunks.
- `force_rerun=True` skips resume and reinitializes latest/checkpoint outputs.

## Pyramid Materialization

- Builds multiscale corrected output under
  `results/flatfield/latest/data_pyramid/level_*`.
- Factors are resolved from source/root attrs when available; otherwise base
  level only.
- Uses Dask array slicing/rechunk/to_zarr.
- Writes pyramid metadata to both:
  - `results/flatfield/latest/data` attrs
  - `results/flatfield/latest` attrs (`data_pyramid_*`).

## Provenance / Latest Output Reference

- `register_latest_output_reference(...)` is called with `analysis_name="flatfield"`
  and component `results/flatfield/latest`.
- Metadata includes components, output chunking/dtype, pyramid info, resume
  fingerprint/state, and warning counts.

## Operational Caveats

- `full_volume` fit can be memory-heavy on large datasets.
- Distributed path currently computes all stage futures at once; if scaling
  issues appear, consider bounded in-flight scheduling.
- Keep transform writes non-overlapping and preserve checkpoint update semantics.
- Preserve chunk-probe resume guards; they protect against malformed checkpoint
  chunks in N5/Zarr stores.

## Validation

- `uv run --extra dev ruff check src/clearex/flatfield/pipeline.py tests/flatfield/test_pipeline.py`
- `uv run --extra dev pytest -q tests/flatfield/test_pipeline.py`
