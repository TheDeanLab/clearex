# ClearEx Core Agent Notes

This directory contains the runtime orchestration surface for ClearEx.

## Primary Files

- `main.py`: GUI/headless entrypoint, Dask backend initialization, step execution, provenance persistence.
- `workflow.py`: shared config schema and normalization for GUI + CLI/headless.

## Agent Notes Convention

- This file is the top-level agent reference for `src/clearex`.
- For subdirectories under `src/clearex`, treat `README.md` as the canonical
  agent-reference file for that folder.
- If both `README.md` and `CODEX.md` exist in a subdirectory, prefer
  `README.md`.
- `CODEX.md` files may remain temporarily for compatibility but should be kept
  aligned with their corresponding `README.md` files.

## Runtime Invariants

- Canonical analysis image layout is `(t, p, c, z, y, x)`.
- Canonical source array component is `data`.
- Analysis outputs use `results/<analysis>/latest/...` (latest-only replacement).
- Provenance records are append-only and include workflow + runtime parameters.
- Root store attr `spatial_calibration` is the canonical store-level
  stage-to-world axis mapping for Navigate multiposition placement; missing
  attrs mean identity mapping.

## Dask Workload Policy

- I/O-heavy steps (loading/materialization): prefer threaded local workers (`processes=False`).
- CPU-heavy analysis steps: prefer process-based local workers (`processes=True`).
- Always honor user-selected backend mode from GUI/CLI config.

## Recent Runtime Updates (2026-03-09)

- `_run_workflow(...)` now starts Dask backends lazily:
  - I/O backend starts only for Navigate `experiment.yml` materialization paths.
  - Analysis backend starts only when selected operations need a client (`deconvolution`, `particle_detection`, `usegment3d`).
- Visualization-only workflows should not create extra LocalCluster instances.
- Local Dask dashboard binding now defaults to an ephemeral port (`dashboard_address=":0"`) to avoid recurring `8787` conflicts.
- Navigate TIFF ingestion now stacks `Position*/CH*` collections into canonical
  `(t, p, c, z, y, x)` data instead of materializing only the first TIFF file.
- Navigate BDV `H5`/`N5` ingestion now stacks setup collections using companion
  XML `ViewSetup` metadata so channels/positions are preserved across formats.
- Napari multiposition affine translations are now treated in world-space
  microns, with scale preferring stored `voxel_size_um_zyx` attrs.

## Recent Runtime Updates (2026-03-15)

- `mip_export` distributed execution now relies on tile-parallel reduction and
  tile-shaped reduction block sizing for stable throughput on very large
  datasets.
- `mip_export` now uses explicit best-effort Zarr-store closure around worker
  and driver read/write operations to reduce file-descriptor pressure.
- `mip_export` TIFF outputs are now written as OME-TIFF with projection-aware
  physical pixel calibration (`PhysicalSizeX/Y`) derived from source
  `voxel_size_um_zyx` metadata.
- Detailed operational guidance for this area lives in
  `src/clearex/mip_export/README.md`.

## Recent Runtime Updates (2026-03-16)

- Added `uSegment3D` segmentation operation:
  - workflow normalization + GUI controls now map to canonical runtime keys,
  - optional `input_resolution_level` supports segmentation on pyramid levels,
  - `output_reference_space` can upsample labels back to level 0,
  - optional `save_native_labels` stores downsampled native labels alongside upsampled output,
  - distributed execution runs one task per `(t, p, selected channel)` volume,
  - GUI runtime controls now expose channel checkboxes and emit `channel_indices`,
  - headless CLI supports `--channel-indices` with `all` for full-channel runs,
  - headless CLI supports `--input-resolution-level` for pyramid-level selection,
  - latest output is persisted to `results/usegment3d/latest/data`,
  - provenance references include GPU/tiling configuration, resolution/output-space metadata, and selected views.
- Runtime uses optional dependency loading (`u-Segment3D`) and supports
  `require_gpu` fail-fast behavior when CUDA is unavailable.
- Local GPU execution now supports a GPU-pinned `LocalCluster` mode via
  `create_dask_client(..., gpu_enabled=True)`, launching one worker per
  CUDA device with `GPU=1` worker resources for explicit task placement.
- Visualization now supports multi-volume napari overlays:
  - configurable `volume_layers` rows support image/labels overlays with
    channel/display controls,
  - GUI includes a popup `Volume Layers...` table for these rows,
  - per-layer multiscale policies are now `inherit`, `require`, and `off`,
  - legacy saved `auto_build` values are normalized to `inherit`,
  - reusable display pyramids are prepared explicitly via the
    `display_pyramid` analysis task under
    `results/display_pyramid/by_component/...`.

## Recent Runtime Updates (2026-03-20)

- Added store-level spatial calibration for Navigate multiposition datasets:
  - `WorkflowConfig` now carries `SpatialCalibrationConfig`,
  - canonical text form is `z=...,y=...,x=...`,
  - allowed bindings are `+/-x`, `+/-y`, `+/-z`, `+/-f`, and `none`,
  - the root store attr `spatial_calibration` persists schema, mapping, and
    `theta_mode`,
  - missing attrs resolve to identity instead of requiring backfilled config.
- Setup flow now exposes a lightweight `Spatial Calibration` control per
  experiment:
  - one draft is kept per experiment while setup is open,
  - existing stores prefill the current mapping,
  - `Next` writes the resolved mapping to every reused or newly prepared store.
- Headless workflows now accept `--stage-axis-map` for Navigate
  `experiment.yml` inputs and existing Zarr/N5 stores.
- Visualization position affines now derive world `z/y/x` translations from
  the stored calibration:
  - Navigate `F` is available as a placement source,
  - `none` zeroes a world axis translation,
  - sign inversion is supported,
  - `THETA` remains rotation of the `z/y` plane about world `x`.
- Provenance now records the effective spatial calibration used by the run.

## Recent Runtime Updates (2026-03-21)

- Added `display_pyramid` as a first-class analysis operation between
  `registration` and `visualization`.
- `display_pyramid` prepares reusable display pyramids for one selected source
  component and stores per-channel `1/95` display contrast metadata.
- Visualization no longer performs viewer-time pyramid auto-build, launch-time
  percentile/min/max reductions, or hidden coarse-level/stride downsampling.
- Napari launch policy is now:
  - 2D uses lazy arrays directly and can use prepared display pyramids,
  - 3D always uses the base single-scale array,
  - oversized 3D image layers trigger a warning and force 2D instead of
    silent downsampling.
- Multiposition image layers now default to `translucent` blending when the
  user leaves blending blank/auto.
- Detailed operational guidance for this area now lives in
  `src/clearex/visualization/README.md`.

## Sequencing and Inputs

- Operation order is driven by `analysis_parameters[<op>]["execution_order"]`.
- Per-step input source comes from `analysis_parameters[<op>]["input_source"]`.
- `workflow.resolve_analysis_execution_sequence(...)` is the canonical order resolver.

## Implementation Rules

- Keep parsing/normalization centralized in `workflow.py`; avoid duplicating logic in GUI/runtime.
- Keep function signatures type hinted.
- Use numpydoc docstrings for new/changed functions.

## Ongoing GUI And Docs Hygiene

- For every GUI change, run a theme audit and confirm all visible text elements use the ClearEx theme (labels, form labels, tab text, table headers/items, tooltips, and popup/dialog text).
- Explicitly verify operation-parameter panels and their popup editors for text-theme consistency; do not rely on platform default colors.
- Update user and developer documentation in the same change set whenever behavior, parameters, or workflow/provenance expectations change.
- Treat documentation refresh as recurring maintenance, and regularly update docs to match the shipped UI/runtime behavior.

## Validation

- `uv run ruff check src/clearex/main.py src/clearex/workflow.py`
- `uv run pytest -q tests/test_main.py`
- `uv run --with pytest --with requests python -m pytest -q tests/test_workflow.py tests/io/test_provenance.py tests/usegment3d/test_pipeline.py`
