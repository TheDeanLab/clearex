# ClearEx Core Agent Notes

This directory contains the runtime orchestration surface for ClearEx.

## Primary Files

- `main.py`: GUI/headless entrypoint, Dask backend initialization, step execution, provenance persistence.
- `workflow.py`: shared config schema and normalization for GUI + CLI/headless.

## Agent Notes Convention

- This file is the authoritative package-level guide for `src/clearex`.
- The repository root `AGENTS.md` covers repo-wide conventions only and should
  not duplicate package runtime strategy.
- For subdirectories under `src/clearex`, treat `README.md` as the canonical
  subsystem-level reference file for that folder.
- If guidance conflicts, the more specific file wins:
  `src/clearex/<subsystem>/README.md` over this file, and this file over the
  repository-root `AGENTS.md`.
- Keep package-level rules here and move subsystem-only detail into the
  corresponding `README.md`.

## Runtime Invariants

- Canonical persisted stores are OME-Zarr v3 `*.ome.zarr` objects.
- Canonical analysis image layout remains `(t, p, c, z, y, x)` for ClearEx
  internal execution arrays.
- Public source data is exposed as a single-well OME-Zarr HCS collection at the
  store root (`A/1/<field>/<level>` with `TCZYX` arrays).
- Public image-producing analysis outputs are exposed as OME-Zarr HCS
  collections under `results/<analysis>/latest`.
- ClearEx internal execution arrays live under
  `clearex/runtime_cache/source/...` and
  `clearex/runtime_cache/results/<analysis>/latest/...`.
- ClearEx-owned metadata and non-image artifacts live under namespaced paths:
  `clearex/metadata`, `clearex/provenance`, `clearex/gui_state`, and
  `clearex/results/<analysis>/latest`.
- Workflow input aliases such as `data`, `flatfield`, `deconvolution`,
  `shear_transform`, `usegment3d`, and `registration` are logical names that
  resolve to runtime-cache components. New runtime code must not hard-code old
  public array paths.
- Provenance records are append-only and include workflow + runtime parameters.
- Store-level spatial calibration is persisted in `clearex/metadata` and
  missing values resolve to the identity mapping.
- Legacy root `data`, root `data_pyramid`, and
  `results/<analysis>/latest/data` layouts are migration-only and must not be
  reintroduced as canonical outputs.
- Legacy `.n5` inputs are source-only. For Navigate BDV N5, agents must use
  TensorStore-backed reads from `setup*/timepoint*/s0` plus companion XML
  `ViewSetup` metadata; do not reintroduce `zarr.open_group(...)` or
  `da.from_zarr(...)` on raw `.n5` paths.
- Bare standalone `.n5` runtime input remains unsupported in phase 1. If the
  source is a Navigate N5 acquisition, route it through `experiment.yml`
  materialization into canonical `.ome.zarr`.

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
- Navigate BDV `N5` now reads through TensorStore because `zarr>=3` no longer
  provides `N5Store`; legacy ClearEx groups inside source `.n5` trees must be
  ignored during source selection.
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
  - image output is written to the runtime cache and published as the public
    OME collection `results/usegment3d/latest`,
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
    `display_pyramid` analysis task and summarized under
    `clearex/results/display_pyramid/latest`.

## Recent Runtime Updates (2026-03-20)

- Added store-level spatial calibration for Navigate multiposition datasets:
  - `WorkflowConfig` now carries `SpatialCalibrationConfig`,
  - canonical text form is `z=...,y=...,x=...`,
  - allowed bindings are `+/-x`, `+/-y`, `+/-z`, `+/-f`, and `none`,
  - `clearex/metadata` persists schema, mapping, and `theta_mode`,
  - missing attrs resolve to identity instead of requiring backfilled config.
- Setup flow now exposes a lightweight `Spatial Calibration` control per
  experiment:
  - one draft is kept per experiment while setup is open,
  - existing stores prefill the current mapping,
  - `Next` writes the resolved mapping to every reused or newly prepared store.
- Headless workflows now accept `--stage-axis-map` for Navigate
  `experiment.yml` inputs and existing canonical OME-Zarr stores.
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

## Recent Runtime Updates (2026-03-23)

- ClearEx adopted OME-Zarr v3 as the canonical store contract:
  - materialization now targets `data_store.ome.zarr`,
  - public source data is published as a root OME HCS collection,
  - public image-analysis outputs are published as OME HCS collections under
    `results/<analysis>/latest`,
  - internal execution arrays moved under `clearex/runtime_cache/...`,
  - store metadata, provenance, GUI state, and non-image artifacts moved under
    `clearex/...`.
- The runtime now refuses legacy canonical ClearEx stores as direct canonical
  inputs and requires migration via `clearex --migrate-store`.
- Reader selection now prefers validated/public OME metadata instead of
  “largest array wins” heuristics.

## Recent Runtime Updates (2026-03-24)

- Generic Zarr / OME-Zarr ingestion and canonical-store reuse rules were tightened:
  - only canonical ClearEx OME-Zarr stores are reused in place,
  - generic source ``.ome.zarr`` inputs are materialized into a fresh
    ``data_store.ome.zarr`` beside ``experiment.yml``,
  - reader selection prefers validated/public OME arrays and falls back to
    largest-array discovery only when no public OME contract is available.
- GUI/store metadata discovery and uSegment3D voxel-size resolution now follow
  the canonical runtime source component plus ``source_component`` ancestry
  instead of assuming root ``data`` arrays or root voxel attrs.
- Registration and fusion are now separate analysis operations:
  - `registration` is metadata-only and writes transforms/layout artifacts
    under `clearex/results/registration/latest`,
  - `fusion` consumes the latest registration result and writes stitched image
    output under `clearex/runtime_cache/results/fusion/latest/data`,
  - this split exists so users can run registration and fusion in separate
    workflow executions with different Dask worker counts and memory limits.
- Fusion now supports richer overlap blending controls:
  - `blend_mode` accepts `average`, `feather`, `center_weighted`,
    `content_aware`, and `gain_compensated_feather`,
  - `blend_exponent` tunes feather/center-weighted falloff strength,
  - `gain_clip_range` constrains fitted overlap gain during
    gain-compensated fusion,
  - `blend_overlap_zyx` controls fusion ramp width independently from
    registration `pairwise_overlap_zyx`.
- Gain-compensated feather now estimates per-edge moving-to-fixed linear
  intensity maps during fusion, solves per-position gain/offset corrections,
  and records `intensity_gains_tp` / `intensity_offsets_tp` under
  `clearex/results/fusion/latest`.
- Blend-weight storage now persists separable 1D profiles plus blend metadata
  under `clearex/results/fusion/latest/blend_weights` instead of assuming a
  materialized 3D volume.
- ``mip_export export_format=zarr`` now writes standalone OME-Zarr v3 image
  stores (``.ome.zarr``) with axes and physical-scale metadata so the exported
  projections remain directly viewable in OME-aware tools outside the main
  ClearEx store.

## Recent Runtime Updates (2026-03-25)

- Visualization now includes a two-stage movie workflow:
  - `render_movie` consumes captured napari keyframes, reconstructs the
    visualization scene, interpolates camera/layer state, and exports PNG
    frames at one or more selected resolution levels.
  - `compile_movie` consumes a rendered frame set and encodes MP4 and/or
    ProRes movies through `ffmpeg` without rerendering napari screenshots.
- The movie workflow writes latest metadata under:
  - `clearex/results/render_movie/latest`
  - `clearex/results/compile_movie/latest`
- Default movie artifacts now stay inside the canonical store under:
  - `clearex/results/visualization/latest/keyframes.json`
  - `clearex/results/render_movie/latest/...`
  - `clearex/results/compile_movie/latest/...`
- `output_directory` remains available only when a user explicitly wants an
  external movie-export tree.
- Visualization keyframe manifests (`schema_version=2`) now persist stable
  keyframe ids/order indices and enough overlay metadata to rebuild captured
  `Points` and `Tracks` layers during offline movie rendering.
- 3D movie camera interpolation now uses quaternion SLERP for orientation
  changes rather than naive Euler-angle interpolation. This is the intended
  path for avoiding the nonlinear jump seen in older notebook-based movie
  workflows.
- Movie rendering now uses a visible napari viewer by default and still
  verifies that the first screenshot is non-empty before continuing.

## Recent Runtime Updates (2026-04-08)

- Added `volume_export` as a visualization-family analysis operation between
  `compile_movie` and `mip_export`.
- `volume_export` exports one selected image-producing source component at an
  explicit `(t, p, c)` selection or across all available indices.
- Runtime now supports source-selectable resolution-level export:
  - existing discovered source-adjacent pyramid levels are reused when present,
  - missing deeper levels can be generated during export,
  - exported metadata preserves voxel calibration through source ancestry.
- Storage contract for `volume_export` is:
  - runtime-cache image data at
    `clearex/runtime_cache/results/volume_export/latest/data`
  - latest metadata at `clearex/results/volume_export/latest`
  - public OME image collection at `results/volume_export/latest` only when
    `export_format=ome-zarr`
  - in-store TIFF artifacts at
    `clearex/results/volume_export/latest/files/*.ome.tif` when
    `export_format=ome-tiff`
- `volume_export` is not a preferred chainable scientific upstream alias; it is
  an export workflow that records latest outputs and provenance.
- When `render_movie` is launched from the GUI or any other Qt-active context,
  it should run that visible napari capture inside a dedicated subprocess
  rather than constructing the viewer on a worker thread.
- `render_movie` and `compile_movie` are metadata/export analyses only. They do
  not publish public OME image collections and they do not become chainable
  scientific image sources for downstream analysis.

## Recent Runtime Updates (2026-04-09)

- GUI-triggered Dask dashboard launch now uses a localhost-only tokenized
  relay managed by `clearex.gui.dask_dashboard_proxy` instead of opening the
  scheduler URL directly.
- The relay is available only while a ClearEx-managed Dask client is alive and
  registered for the current GUI workload.
- Both the main analysis dialog and the `Running Analysis` dialog can launch
  the relay-backed dashboard.
- If ClearEx cannot reach the upstream dashboard or cannot start the relay,
  the GUI shows a warning and does not fall back to the raw dashboard URL.

## Recent Runtime Updates (2026-03-22)

- Registration pipeline (`pipeline.py`) performance optimizations:
  - Default `registration_type` changed from `rigid` to `translation` for
    maximum throughput; rigid/similarity remain available.
  - FFT phase-correlation initial alignment (`use_fft_initial_alignment=True`
    by default) pre-aligns the moving crop before ANTs, so ANTs starts near
    the solution and converges with fewer iterations.  The ANTs result is
    composed with the FFT correction automatically.
  - Pairwise and fusion source reads now use `_source_subvolume_for_overlap` to
    load only the minimal Zarr slice covering each overlap or output chunk,
    instead of reading full tile volumes.
  - Large overlap crops are optionally downsampled to a configurable voxel
    budget (`max_pairwise_voxels`, default 500 000) before ANTs estimation.
  - ANTs iteration counts reduced from `(2000, 1000, 500, 250)` to
    `(200, 100, 50, 25)` by default; legacy values available as
    `_ANTS_AFF_ITERATIONS_LEGACY`.  ANTs parameters are now configurable via
    `ants_iterations` and `ants_sampling_rate` in the parameters dict.
  - Optional FFT-only fast path for translation-only registration
    (`use_phase_correlation=True`), using
    `skimage.registration.phase_cross_correlation`.
  - Blend-weight profiles are pre-computed once per fusion run and persisted as
    separable 1D Zarr datasets under `clearex/results/fusion/latest/blend_weights`
    so fusion workers can reconstruct separable weights lazily without
    materializing a dense 3D blend volume.
  - GPU TODO: `_resample_source_to_world_grid` annotated for future
    `cupyx.scipy.ndimage.affine_transform` integration.
  - Progress emission now covers the pairwise registration and fusion phases
    separately: registration reports metadata loading, transform building, edge
    preparation, per-edge pairwise progress, solving, layout computation, and
    metadata writing; fusion reports per-chunk render progress. The progress
    bar no longer appears stalled during long runs.
  - All new parameters recorded in provenance metadata and group attrs.
- Detailed operational guidance in `src/clearex/registration/README.md`.

## Sequencing and Inputs

- Operation order is driven by `analysis_parameters[<op>]["execution_order"]`.
- Per-step input source comes from `analysis_parameters[<op>]["input_source"]`.
- `workflow.resolve_analysis_execution_sequence(...)` is the canonical order resolver.
- Input-source UI and workflow defaults should present logical aliases rather
  than raw internal component paths whenever possible.

## Implementation Rules

- Keep parsing/normalization centralized in `workflow.py`; avoid duplicating logic in GUI/runtime.
- Keep function signatures type hinted.
- Use numpydoc docstrings for new/changed functions.
- New image-producing analyses must:
  - write executable arrays into `clearex/runtime_cache/results/<analysis>/latest`,
  - publish a public OME image collection under `results/<analysis>/latest`,
  - keep auxiliary arrays / metadata under `clearex/results/<analysis>/latest`.
- New metadata/table-only analyses must write to `clearex/results/...` and
  must not allocate fake public image collections.
- Readers, GUI discovery, and visualization helpers must prefer public OME
  metadata and OME coordinate transforms. Do not reintroduce root-array
  heuristics as the canonical path.

## Ongoing GUI And Docs Hygiene

- For every GUI change, run a theme audit and confirm all visible text elements use the ClearEx theme (labels, form labels, tab text, table headers/items, tooltips, and popup/dialog text).
- Explicitly verify operation-parameter panels and their popup editors for text-theme consistency; do not rely on platform default colors.
- Update user and developer documentation in the same change set whenever behavior, parameters, or workflow/provenance expectations change.
- Treat documentation refresh as recurring maintenance, and regularly update docs to match the shipped UI/runtime behavior.

## Validation

- `uv run ruff check src/clearex/main.py src/clearex/workflow.py`
- `uv run pytest -q tests/test_main.py`
- `uv run --with pytest --with requests python -m pytest -q tests/test_workflow.py tests/io/test_provenance.py tests/usegment3d/test_pipeline.py`
