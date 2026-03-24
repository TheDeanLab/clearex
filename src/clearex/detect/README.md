# Detection Agent Notes

This folder contains particle-detection logic and chunk-parallel analysis execution.

## Key Files

- `pipeline.py`: chunk orchestration, Dask execution, coordinate stitching, Zarr result writes.
- `particles.py`: algorithm primitives (`preprocess`, `detect_particles`, post-filters).

## Particle Pipeline Contract

- Input is a canonical 6D component `(t, p, c, z, y, x)`.
- `input_source` parameter selects a logical source alias or explicit internal
  component path (default: `data`).
- Channel selection is done with `channel_index`.
- Chunk tasks run in parallel and return global-coordinate rows.

## Output Contract

- Latest result root: `clearex/results/particle_detection/latest`
- Datasets:
  - `detections` columns: `t,p,c,z,y,x,sigma,intensity`
  - `points_tzyx` for Napari points rendering
- Particle detection is a metadata/table output, not a public OME image
  collection. Keep it in the ClearEx namespace.
- Latest-output reference must be registered in provenance metadata.

## Performance Notes

- Keep I/O access chunk-local and avoid overlapping writes.
- Favor Dask distributed execution for large runs.
- Use overlap mode only when seam artifacts require it.

## Parameter and API Changes

When adding/changing parameters:

1. Normalize in `workflow.py` and `pipeline.py`.
2. Keep defaults aligned with GUI collection/hydration.
3. Ensure provenance captures effective values.
4. Update tests under `tests/detect/`.

## Validation

- `uv run ruff check src/clearex/detect/pipeline.py tests/detect/test_pipeline.py`
- `uv run --with pytest --with requests python -m pytest -q tests/detect/test_pipeline.py`
