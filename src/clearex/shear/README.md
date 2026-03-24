# Shear Transform Strategy

This `README.md` is the canonical subsystem reference for
`src/clearex/shear`.

## Scope

- Main runtime file: `src/clearex/shear/pipeline.py`
- Primary entrypoint: `run_shear_transform_analysis`
- Runtime output: `clearex/runtime_cache/results/shear_transform/latest/data`
- Public OME publication target: `results/shear_transform/latest`

## Coordinate And Geometry Contract

- Shear/rotation geometry is solved in physical-space XYZ coordinates.
- Source data axes remain canonical `(t, p, c, z, y, x)`.
- Voxel spacing is interpreted as `voxel_size_um_zyx` and used to map index
  space to physical space.
- `auto_rotate_from_shear` applies axis-coupled Euler adjustments:
  - `rotation_deg_x += -atan(shear_yz)`
  - `rotation_deg_y += +atan(shear_xz)`
  - `rotation_deg_z += -atan(shear_xy)`

## Scale Preservation Invariants

The shear pipeline must preserve physical pixel size metadata end-to-end.

Required invariants:

- Upstream operations must preserve or restate physical voxel size metadata on
  their output arrays.
- Shear voxel-size resolution must not rely on a single location only.
- Shear outputs must persist:
  - `voxel_size_um_zyx`
  - `voxel_size_resolution_source`
- Downstream registration, visualization, OME publication, and MIP export must
  consume the same resolved voxel spacing.

## Voxel-Size Resolution Policy

Shear now relies on shared resolver logic in `clearex.io.ome_store`:

- `resolve_voxel_size_um_zyx_with_source(...)`
- `resolve_voxel_size_um_zyx(...)`

Resolution order:

1. source component attrs
2. source-component ancestry via `source_component`
3. namespaced store metadata (`clearex/metadata`)
4. root attrs
5. Navigate metadata fallbacks (`navigate_experiment`)
6. default `(1.0, 1.0, 1.0)` only when all metadata is missing

This policy exists to prevent regressions where intermediate runtime arrays
accidentally drop voxel-size attrs.

Read-only safety requirement:

- Metadata reads must not create `clearex/metadata` when the store is opened
  in read-only mode.
- Resolver helpers must remain side-effect free for `mode="r"` roots.

Downsampled-layer requirement:

- Global/root/store voxel metadata describes base-resolution physical spacing.
- For non-base components that do not define component-local scale metadata,
  visualization must apply integer shape-ratio scaling from base data shape to
  preserve physical spacing in napari.

## Regression Guardrails

When editing this subsystem:

- Do not remove `voxel_size_um_zyx` persistence from shear outputs.
- Do not replace shared voxel-size resolution with local single-source lookups.
- Keep `voxel_size_resolution_source` on shear outputs for diagnostics.
- If output metadata keys change, update downstream readers and tests in the
  same change set.

## Validation

Run at minimum after scale-related changes:

- `uv run ruff check src/clearex/shear/pipeline.py src/clearex/io/ome_store.py src/clearex/visualization/pipeline.py src/clearex/registration/pipeline.py src/clearex/flatfield/pipeline.py src/clearex/mip_export/pipeline.py`
- `uv run --with pytest --with requests python -m pytest -q tests/shear/test_pipeline.py tests/io/test_ome_store_scale.py tests/flatfield/test_pipeline.py::test_copy_source_array_attrs_preserves_voxel_size tests/registration/test_pipeline.py::test_extract_voxel_size_uses_source_component_chain tests/visualization/test_pipeline.py::test_run_visualization_analysis_prefers_voxel_size_um_attrs tests/visualization/test_pipeline.py::test_run_visualization_analysis_resolves_scale_from_source_chain tests/visualization/test_pipeline.py::test_launch_napari_viewer_resolves_per_layer_scale_for_downsampled_components`

For store-level verification on real data, confirm scale lineage and OME
multiscale scale transforms remain non-isotropic where expected.
