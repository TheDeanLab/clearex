# Visualization Strategy

This `README.md` is the canonical agent-reference and runtime-strategy file for
`src/clearex/visualization`.

## Scope

- Main runtime file: `src/clearex/visualization/pipeline.py`
- Primary analysis hooks:
  - `run_display_pyramid_analysis`
  - `run_visualization_analysis`
  - `run_render_movie_analysis`
  - `run_compile_movie_analysis`
- Movie encoding helpers:
  - `src/clearex/visualization/export.py`
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
- `volume_export`: a reproducible 3D volume export task for one selected image source
- `render_movie`: an offline frame-render task driven by captured keyframes
- `compile_movie`: a fast encode task driven by rendered frame sets

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
- If a prior `display_pyramid` write left behind an incomplete source-adjacent
  pyramid subtree, the next retry deletes that subtree and rebuilds from the
  full-resolution source before reusing anything.
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

## Render Movie Task

`render_movie` consumes a captured visualization keyframe manifest and exports
offline PNG frame sequences for one or more selected pyramid levels.

### Input contract

- The operation reads visualization metadata from:
  - `clearex/results/visualization/latest`
- The authoritative keyframe manifest comes from:
  - `visualization.latest.keyframe_manifest_path`, or
  - explicit `render_movie.keyframe_manifest_path`
- Rendering reconstructs the prepared visualization scene, reapplies captured
  camera/dims/layer state, and then emits screenshots from napari.
- Captured `Points` and `Tracks` layers are serialized into the keyframe
  manifest and rebuilt during offline rendering so particle/track overlays can
  survive beyond the interactive viewer session.
- `render_movie` now uses a visible napari viewer by default and validates the
  first captured screenshot before continuing.
- When a Qt application is already active, `render_movie` launches that visible
  capture viewer in a dedicated subprocess so GUI-triggered runs do not create
  Qt/OpenGL objects from a worker thread.

### Storage contract

- Default keyframe manifests are written inside the store at:
  - `clearex/results/visualization/latest/keyframes.json`
- Latest task metadata is stored at:
  - `clearex/results/render_movie/latest`
- Default rendered frames are written inside the store under:
  - `clearex/results/render_movie/latest/level_<nn>_frames/frame_000000.png`
- Each run also writes a render manifest at:
  - `clearex/results/render_movie/latest/render_manifest.json`
- Users can still override `output_directory` when they explicitly want an
  external export tree.

### Parameter guidance

- `resolution_levels`
  - Use `[1]` or `[2]` for quick previews and timing checks.
  - Use `[0]` for final publication frames.
  - Rendering multiple levels in one run is supported when you want both a
    fast preview set and a final master set.
- `render_size_xy`
  - Good preview sizes: `[1280, 720]` or `[1600, 900]`
  - Common final sizes: `[1920, 1080]`, `[2560, 1440]`, `[3840, 2160]`
  - Large 4K renders are best reserved for the final pass because napari
    screenshot capture cost scales with output size.
- `default_transition_frames`
  - Good range: `12` to `96`
  - `24` gives brisk cuts, `48` is a balanced default, `72+` gives slower
    cinematic motion.
- `transition_frames_by_gap`
  - Optional per-gap overrides. Use this when one camera move needs a much
    longer dwell or orbit than the rest of the sequence.
- `hold_frames`
  - Good range: `0` to `24`
  - Use `0` for continuous motion or `6` to `12` when key poses need a brief
    pause for the viewer to register labels and context.
- `interpolation_mode`
  - `ease_in_out` is the default and should be preferred for most camera moves.
  - `linear` is mainly useful for technical/measurement movies where constant
    motion speed matters more than cinematic easing.
- `camera_effect`
  - `none`: plain interpolation between captured keyframes.
  - `orbit`: adds a controlled yaw sweep across a transition.
    Good `orbit_degrees` range: `10` to `90`.
  - `flythrough`: adds extra travel along the motion vector.
    Good `flythrough_distance_factor` range: `0.02` to `0.20`.
  - `zoom_fx`: adds a push-in/pull-back on top of captured zoom.
    Good `zoom_effect_factor` range: `0.05` to `0.25`.
- Overlays
  - `overlay_title` / `overlay_subtitle` are intended for figure-style titles
    and panel labels.
  - `overlay_frame_text_mode=frame_number` is useful for review cuts.
  - `overlay_frame_text_mode=keyframe_annotations` is useful when keyframes
    carry scene annotations that should appear in the final movie.
  - `overlay_scalebar` should be enabled for publication movies whenever a
    physical spatial reference is important.
  - Good `overlay_scalebar_length_um` values are dataset-dependent, but use a
    bar that occupies roughly `10%` to `25%` of the frame width.

### Interpolation policy

- Camera orientation interpolation in 3D uses quaternion SLERP rather than
  independent Euler interpolation. This is the intended fix for the nonlinear
  camera jump seen in older notebook-based movie workflows.
- Continuous layer properties such as opacity, contrast, scale, translate, and
  track tail lengths are interpolated when both endpoints define them.
- Discrete layer properties such as visibility toggles, blending, rendering
  mode, and colormap snap at the midpoint of the transition.

## Compile Movie Task

`compile_movie` consumes one rendered frame set and encodes it with `ffmpeg`
into review and/or delivery movies.

### Input contract

- The operation reads render metadata from:
  - `clearex/results/render_movie/latest`
- The authoritative render manifest comes from:
  - `render_movie.latest.render_manifest_path`, or
  - explicit `compile_movie.render_manifest_path`
- One rendered level is compiled per run through `rendered_level`.

### Storage contract

- Latest task metadata is stored at:
  - `clearex/results/compile_movie/latest`
- Default compiled files are written inside the store under:
  - `clearex/results/compile_movie/latest/*.mp4`
  - `clearex/results/compile_movie/latest/*.mov`
- Users can still override `output_directory` when they explicitly want an
  external export tree.

### Parameter guidance

- `output_format`
  - `mp4`: small review/share files.
  - `prores`: high-quality editorial or submission master.
  - `both`: useful when one render pass should immediately produce both.
- `fps`
  - Common values: `24`, `30`, `60`
  - Use the same FPS as `render_movie` unless you intentionally want a timing
    reinterpretation.
- `mp4_crf`
  - Good range: `16` to `24`
  - Lower is larger and cleaner. `18` is a strong default.
- `mp4_preset`
  - `medium` or `slow` are the practical defaults.
  - Faster presets are fine for iteration, but `slow` is usually better for
    final delivery size/quality.
- `prores_profile`
  - `3` is a good default.
  - `4` or `5` are appropriate when you want a heavier master for grading or
    archival handoff.
- `resize_xy`
  - Leave empty to preserve rendered-frame size.
  - Use this when you want to iterate on encoded file size without rerendering
    the PNG frames.

### Recommended workflow

1. Capture keyframes in `visualization`.
2. Run `render_movie` at a preview level and moderate frame size.
3. Iterate quickly with `compile_movie` until the timing/codec/file-size trade
   is acceptable.
4. Re-run `render_movie` at level `0` and the final output size.
5. Run `compile_movie` again for the publication deliverables.

## Volume Export Task

`volume_export` is a visualization-family export workflow for one selected
image-producing source component.

### Input contract

- Input source is one canonical 6D component in `(t, p, c, z, y, x)` order.
- The operation uses `analysis_parameters["volume_export"]["input_source"]`.
- `force_rerun` preserves the standard latest-only rerun semantics for the
  export result.
- `export_scope=current_selection` exports one explicit `(t, p, c)` volume.
- `export_scope=all_indices` exports every available `(t, p, c)` volume for
  the selected source.
- `resolution_level` reuses discovered source-adjacent pyramid levels when
  available and can generate deeper missing levels during export.
- When an analysis Dask backend is active, missing-level generation and the
  exported runtime-cache write execute through that client instead of the local
  scheduler.
- `current_selection` cache writes preserve the selected source Dask chunk
  geometry, clamped to the selected `(1, 1, 1, z, y, x)` shape. Do not collapse
  the selected volume into one full-spatial target chunk; partial writes into a
  huge target chunk can force large read/modify/write operations and kill Dask
  workers.

### Parameter reference

- `execution_order=12` places `volume_export` after `compile_movie` and before
  `mip_export`.
- `input_source` selects the image-producing upstream component or explicit
  internal component path to export.
- `force_rerun` bypasses provenance-based reuse of the latest export result.
- `memory_overhead_factor` is carried through normalized workflow payloads as a
  scheduler hint; the current export path does not consume it directly.
- `export_scope` chooses `current_selection` or `all_indices`.
- `t_index`, `p_index`, and `c_index` are used only for
  `export_scope=current_selection`.
- `resolution_level` selects the source-adjacent pyramid level to export,
  generating missing deeper levels when needed.
- `export_format` chooses between `ome-zarr` and `ome-tiff`.
- `tiff_file_layout` chooses `single_file` or `per_volume_files` for TIFF
  output.
- `chunk_basis`, `detect_2d_per_slice`, `use_map_overlap`, and `overlap_zyx`
  are normalized compatibility fields inherited from other volume analyses and
  remain pinned to 3D/no-overlap behavior for `volume_export`.

### Storage contract

- Latest task metadata is stored at:
  - `clearex/results/volume_export/latest`
- Exported runtime-cache image data is written at:
  - `clearex/runtime_cache/results/volume_export/latest/data`
- `export_format=ome-zarr` publishes a public OME image collection at:
  - `results/volume_export/latest`
- `export_format=ome-tiff` keeps file artifacts inside the store under:
  - `clearex/results/volume_export/latest/files/*.ome.tif`

### TIFF layout policy

- `current_selection` writes one `ZYX` OME-TIFF/BigTIFF volume.
- `all_indices + single_file` writes one BigTIFF with one `TCZYX` series per
  position field.
- `all_indices + per_volume_files` writes one `ZYX` OME-TIFF per exported
  `(t, p, c)` volume.
- TIFF metadata must preserve resolved `PhysicalSizeZ/Y/X`.

## GUI / Workflow Notes

- `display_pyramid` is ordered between `registration` and `visualization`.
- `volume_export` is ordered after `compile_movie` and before `mip_export`.
- `render_movie` is ordered after `visualization`.
- `compile_movie` is ordered after `render_movie`.
- The visualization GUI should describe `use_multiscale` as using existing
  display pyramids, not auto-building them.
- The visualization GUI should describe 3D as a request that may fall back to
  2D for oversized image layers.
- The Visualization tab now owns three related workflows:
  - `visualization` for napari launch and keyframe capture,
  - `volume_export` for source-selectable 3D OME-Zarr / OME-TIFF export,
  - `render_movie` for offline frame generation,
  - `compile_movie` for ffmpeg encoding.
- `launch_mode=auto` should resolve to `subprocess` whenever a Qt application
  instance is active, so GUI workflows can continue and complete while napari
  remains open.
- `launch_mode=auto` in non-Qt contexts retains thread-based behavior:
  main-thread runs stay `in_process`; non-main-thread runs use `subprocess`.
- Detailed movie timing/overlay knobs are currently exposed in the GUI and in
  normalized workflow parameters. The CLI currently exposes operation flags but
  not dedicated per-parameter movie arguments.

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
