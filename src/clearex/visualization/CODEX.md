# Visualization Agent Notes

This folder owns napari-facing visualization workflows.

## Runtime Entry Points

- Main analysis hook: `src/clearex/visualization/pipeline.py::run_visualization_analysis`
- Subprocess viewer entry: `python -m clearex.visualization.pipeline`

## Data Expectations

- Visualization reads canonical 6D arrays in `(t, p, c, z, y, x)` order.
- Default source is `data`.
- Multiscale defaults to enabled and resolves levels from:
  - `data.attrs["pyramid_levels"]`, and
  - root attrs `data_pyramid_levels` for canonical base data.

## Overlay Behavior

- Particle overlays are read from `results/particle_detection/latest/detections` by default.
- Overlay points are emitted in napari coordinate order `(t, c, z, y, x)`.
- Position filtering is applied when detection tables include `(t, p, c, z, y, x, ...)`.

## Napari Metadata Contract

- `run_visualization_analysis` builds a layer payload in `pipeline.py::_build_napari_layer_payload`.
- Image and points layers both receive:
  - axis labels `(t, c, z, y, x)`,
  - matched `scale` values for aligned physical/index coordinates,
  - metadata dictionaries with source/store/pyramid details.
- Scale resolution precedence is:
  1. explicit Zarr attrs (for example `scale_tpczyx`, `scale_tczyx`, `voxel_size_*`),
  2. `source_experiment` metadata (`timepoint_interval`, `step_size`, camera `pixel_size`),
  3. fallback to `(1, 1, 1, 1, 1)`.

## GUI/Threading Contract

- GUI analysis execution runs in a worker thread.
- `launch_mode=auto` must choose subprocess mode off the main thread to avoid Qt/napari thread violations.
- In CLI/main-thread contexts, `launch_mode=auto` should run in-process.

## Provenance/Metadata

- Visualization metadata is stored at `results/visualization/latest`.
- Latest-output provenance registration uses analysis key `visualization`.
- Main workflow later backfills `run_id` after provenance run persistence.
