# Visualization Strategy

This `README.md` is the canonical agent-reference and runtime-strategy file for
`src/clearex/visualization`.

## Scope

- Main runtime file: `src/clearex/visualization/pipeline.py`
- Primary analysis hooks:
  - `run_display_pyramid_analysis`
  - `run_visualization_analysis`
- Subprocess entrypoint:
  - `python -m clearex.visualization.pipeline`

## Large-Data Principles

ClearEx follows napari's lazy-array guidance for large datasets.

- Napari should receive lazy Dask-backed arrays whenever possible.
- ClearEx must not mutate image payloads at viewer launch to force them under a
  render budget.
- ClearEx must not compute percentiles or min/max reductions on lazy arrays
  during viewer launch.
- ClearEx only uses multiscale when an explicit pyramid already exists.
- Viewer launch is not allowed to auto-build pyramids.

The practical consequence is that visualization is now split into:

- `display_pyramid`: an explicit preparation task
- `visualization`: a viewer-launch task

## Display Pyramid Task

`display_pyramid` is a first-class analysis operation that prepares reusable
display pyramids for one selected source component.

### Input contract

- Input source is one canonical 6D component in `(t, p, c, z, y, x)` order.
- The operation uses `analysis_parameters["display_pyramid"]["input_source"]`.
- The task is explicit and per-component. ClearEx does not auto-build display
  pyramids for every downstream result.

### Storage contract

- Latest task metadata is stored at:
  - `clearex/results/display_pyramid/latest`
- Prepared levels are written as ClearEx-owned helper arrays adjacent to the
  selected internal source component:
  - base-source helper levels use source-adjacent pyramid naming
  - derived-source helper levels use `<source_component>_pyramid/level_n`
- Source-component attrs store lookup metadata:
  - `display_pyramid_levels`
  - `display_pyramid_factors_tpczyx`
- Root attrs store lookup metadata:
  - `display_pyramid_levels_by_component`

These helper levels are not public OME image collections. They are internal
visualization aids that should remain separate from the canonical OME source
and analysis image collections.

Compatibility attrs from older visualization-driven behavior are still written
and still read:

- `visualization_pyramid_levels`
- `visualization_pyramid_factors_tpczyx`
- `visualization_pyramid_levels_by_component`

### Performance invariants (do not regress)

The `display_pyramid` pipeline is intentionally optimized to avoid expensive
Dask shuffles/rechunk graphs.

- Do not force `downsampled.rechunk(...)` before `to_zarr` writes.
- Do not set `array.rechunk.method="tasks"` in this write path.
- Level chunking should prefer the generated level array's existing chunk
  geometry (`chunksize`) and only clamp by array bounds.
- If chunk-policy changes are required, they must be benchmarked against the
  large sheared-liver datasets and reviewed for shuffle warnings in logs.

This follows napari + Dask best-practice intent: keep data lazy, chunk-aligned,
and avoid global graph reshapes for interactive workflows.

### Reuse policy

- If matching pyramid levels already exist for the selected component, ClearEx
  reuses them unless `force_rerun=True`.
- Existing ingest-time raw-data pyramids are still valid display sources.
- Re-running `display_pyramid` only overwrites the selected component's explicit
  display-pyramid namespace.

### Contrast metadata

`display_pyramid` also computes and stores per-channel display contrast limits.

- Fixed percentiles: `1` and `95`
- Stored on the source component attrs:
  - `display_contrast_limits_by_channel`
  - `display_contrast_percentiles`
  - `display_contrast_source_component`
- These values are intended for later napari launch and should be reused
  instead of recomputing viewer-time display ranges.

#### Contrast performance invariants

- Contrast computation should use bounded sampling, not full-data percentiles.
- Preferred source for contrast estimation is the coarsest prepared display
  level (when present), recorded in `display_contrast_source_component`.
- Do not use global `reshape(-1)` + `dask.percentile` on full-resolution arrays
  for display metadata.
- Keep viewer launch reduction-free: visualization should consume stored
  contrast limits or dtype fallbacks only.

## Visualization Policy

### 2D behavior

- ClearEx passes lazy arrays directly to napari.
- If `use_multiscale=True` and explicit pyramid levels exist, ClearEx passes
  the prepared levels as a multiscale image.
- If no explicit pyramid exists, ClearEx passes the single base lazy array.

### 3D behavior

- ClearEx always uses the base single-scale array in 3D.
- ClearEx does not switch to a coarser level for 3D.
- ClearEx does not stride/downsample the array for 3D.
- If any image layer exceeds the ClearEx 3D safety limit
  (`_MAX_LAYER_DISPLAY_VOXELS` in `pipeline.py`), ClearEx emits a warning and
  forces the effective display mode to 2D.

This is intentional. The policy is:

- warn
- persist the fallback reason
- force 2D

not:

- downsample silently
- auto-build pyramids
- compute a launch-time surrogate

### Multiscale policy values

Supported runtime/UI policies are now:

- `inherit`
- `require`
- `off`

Legacy saved `auto_build` values are normalized to `inherit`.

## Contrast-Limit Policy

Visualization resolves image contrast limits in this order:

1. stored `display_contrast_limits_by_channel`
2. dtype fallback with no compute

Fallback behavior:

- integer dtype: full dtype range
- bool/float dtype: `[0.0, 1.0]`

ClearEx should not run percentile, min, or max reductions on Dask-backed image
data during napari launch.

When display metadata must be regenerated, use the sampled pipeline from
`run_display_pyramid_analysis` rather than launch-time reductions.

## Multiposition Policy

Multiposition rendering continues to use store-level spatial calibration from
`clearex/metadata` and Navigate position metadata for affine placement.

Default image blending has changed:

- single-position image layers with no explicit blending: `additive`
- multiposition image layers with no explicit blending: `translucent`
- labels layers with no explicit blending: `translucent`

This avoids stripe artifacts from overlapping multiposition regions.

## Napari Metadata Contract

Visualization metadata is stored at:

- `clearex/results/visualization/latest`

The latest metadata must include:

- requested vs effective viewer mode:
  - `viewer_ndisplay_requested`
  - `viewer_ndisplay_effective`
- `display_mode_fallback_reason` when 3D is forced to 2D
- display-pyramid lookup references:
  - `display_pyramid_lookup`
- display-contrast metadata references:
  - `display_contrast_metadata_reference`
- rendered source component information:
  - `source_component`
  - `source_components`
- rendered position information:
  - `position_index`
  - `selected_positions`
  - `show_all_positions`

Image-layer metadata built in `_build_napari_layer_payload(...)` should carry
the same display-policy information so downstream inspection can reconstruct why
napari opened in 2D or 3D.

## GUI / Workflow Notes

- `display_pyramid` is ordered between `registration` and `visualization`.
- The visualization GUI should describe `use_multiscale` as using existing
  display pyramids, not auto-building them.
- The visualization GUI should describe 3D as a request that may fall back to
  2D for oversized image layers.

## Agent Expectations

When editing this area:

- treat this `README.md` as the authoritative file for visualization strategy
- keep `CODEX.md` as a short compatibility pointer only
- update tests with behavior changes in the same change set
- preserve compatibility reads for older stored visualization pyramid attrs
  until migration is intentional and explicit
- do not document display-pyramid helper arrays as the canonical public image
  contract; public OME collections remain the root source image and
  `results/<analysis>/latest` image outputs
