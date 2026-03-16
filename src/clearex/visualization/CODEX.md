# Visualization Agent Notes

This folder owns napari-facing visualization workflows.

## Runtime Entry Points

- Main analysis hook: `src/clearex/visualization/pipeline.py::run_visualization_analysis`
- Subprocess viewer entry: `python -m clearex.visualization.pipeline`

## Data Expectations

- Visualization reads canonical 6D arrays in `(t, p, c, z, y, x)` order.
- Default source is `data`.
- Visualization can render multiple volume sources in one scene (`volume_layers`),
  including image and labels layers.
- Multiposition/channel rendering is only as complete as canonical `data`
  materialization. For Navigate TIFF inputs, verify ingestion stacked
  `Position*/CH*` files (otherwise visualization will only show one `p/c` slot).
- Multiscale defaults to enabled and resolves levels from:
  - `data.attrs["pyramid_levels"]`, and
  - root attrs `data_pyramid_levels` for canonical base data.
- Per-layer multiscale policy options:
  - `inherit`: use existing levels when available (fallback single-scale),
  - `require`: raise if multiscale levels are missing,
  - `auto_build`: generate levels under `results/visualization_cache/pyramids/...`,
  - `off`: force single-scale rendering.

## Overlay Behavior

- Particle overlays are read from `results/particle_detection/latest/detections` by default.
- Overlay points are emitted in napari coordinate order `(t, c, z, y, x)`.
- Position filtering is applied when detection tables include `(t, p, c, z, y, x, ...)`.
- Detection layer defaults:
  - `border_color="white"`,
  - `opacity=0.5`,
  - `blending="translucent"`,
  - transparent faces.

## Napari Metadata Contract

- `run_visualization_analysis` builds a layer payload in `pipeline.py::_build_napari_layer_payload`.
- Image and points layers both receive:
  - axis labels `(t, c, z, y, x)`,
  - matched `scale` values for aligned physical/index coordinates,
  - metadata dictionaries with source/store/pyramid details.
- Image channel rendering defaults in napari:
  - separate image layers per channel (`c` sliced as singleton per layer),
  - no synthetic channel-axis affine offsets (avoids napari status-thread
    out-of-bounds lookups when the cursor axis position changes),
  - `blending="additive"` and `rendering="attenuated_mip"`,
  - per-channel contrast limits from `1st/95th` percentiles,
  - colormap order starts with `green`, `magenta`, `bop orange`, then high-contrast additions (`cyan`, `yellow`, `blue`, ...),
  - channel opacity decreases with channel count (1: `1.0`, 2: `0.9`, 3: `0.8`, floor `0.55`).
- Labels layers are supported through napari `add_labels` and can be mixed with
  image layers in the same viewer.
- Scale resolution precedence is:
  1. explicit Zarr attrs (for example `scale_tpczyx`, `scale_tczyx`, `voxel_size_um_zyx`, `voxel_size_*`),
  2. `source_experiment` metadata (`timepoint_interval`, `step_size`, camera profile `fov_* / img_*`, then `pixel_size / zoom` with binning fallback),
  3. fallback to `(1, 1, 1, 1, 1)`.

## Multiposition Affine Contract

- Visualization supports:
  - single-position mode (`show_all_positions=False`, use `position_index`), and
  - multiposition mode (`show_all_positions=True`, render all positions).
- Stage coordinates are resolved from `multi_positions.yml` adjacent to `source_experiment` when available (fallback: `MultiPositions` in experiment metadata).
- Parsed stage rows use fields `X`, `Y`, `Z`, and `THETA` (`F` is ignored for visualization transforms).
- Per-position napari affine uses homogeneous `6x6` matrix in `(t, c, z, y, x)` coordinates:
  - `THETA` rotates the `z/y` plane (sample rotation around x axis).
  - stage coordinates are in microns and affine translations are applied directly in world-space microns.
- Persisted visualization metadata includes:
  - `selected_positions`,
  - `show_all_positions`,
  - `position_affines_tczyx`,
  - `stage_positions_xyztheta`.

## GUI/Threading Contract

- GUI analysis execution runs in a worker thread.
- `launch_mode=auto` must choose subprocess mode off the main thread to avoid Qt/napari thread violations.
- In CLI/main-thread contexts, `launch_mode=auto` should run in-process.


## Keyframe Capture Contract

- Visualization keyframe capture is enabled by default (`capture_keyframes=True`).
- In the napari viewer:
  - press `K` to capture a keyframe,
  - press `Shift-K` to remove the most recent keyframe.
- Keyframe manifests now record reproducible viewer state, including:
  - camera angles/zoom/center/perspective,
  - dims state (`current_step`, axis labels, order, and 2D/3D view type),
  - layer order and selected/active layer names,
  - per-layer display settings (visibility, opacity, blending, LUT/colormap, contrast limits, rendering mode, interpolation/depiction fields when available),
  - optional per-layer `annotation` text from GUI overrides.
- Default manifest path is `<analysis_store>.visualization_keyframes.json`; override with `keyframe_manifest_path`.
- GUI provides a popup keyframe layer table (`Layer/View Table...`) with columns:
  - `Layer`, `Visible`, `LUT/Colormap`, `Rendering`, `Annotation`.
- GUI provides a popup volume-layer table (`Volume Layers...`) for component/type/channel/display/multiscale configuration.
- Visualization probes napari OpenGL runtime and records renderer metadata (`vendor`, `renderer`, `version`).
- `require_gpu_rendering=True` (default) raises when renderer hints indicate software GL (for example `llvmpipe`, `swiftshader`) or when GPU-backed OpenGL cannot be confirmed.
- Latest visualization metadata includes:
  - `capture_keyframes`,
  - `keyframe_manifest_path` (when set),
  - `keyframe_count`,
  - `keyframe_layer_overrides`.

## Provenance/Metadata

- Visualization metadata is stored at `results/visualization/latest`.
- Auto-built visualization pyramids are stored at `results/visualization_cache/pyramids`.
- Latest-output provenance registration uses analysis key `visualization`.
- Main workflow later backfills `run_id` after provenance run persistence.
