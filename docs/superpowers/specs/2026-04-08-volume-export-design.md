# 2026-04-08 Volume Export Design

## Status

Approved for implementation planning.

## Summary

Add a new `volume_export` analysis in the visualization workflow family that
exports one selected image-producing ClearEx source at a user-selected
resolution. The operation supports:

- preview export of one explicit `(t, p, c)` volume
- batch export of all available `(t, p, c)` indices
- in-store OME-Zarr publication under a namespaced results path
- optional TIFF artifact emission with selectable TIFF layout

The operation is visualization-driven in scope, but it remains a reproducible
pipeline step with normalized parameters, persisted provenance, and testable
runtime behavior.

## Goals

- Allow the user to export any image-producing source currently selectable from
  the visualization workflow, including `data`, `flatfield`,
  `deconvolution`, `fusion`, and other image outputs.
- Support two export scopes:
  - one explicit `(t, p, c)` preview volume
  - all available `(t, p, c)` indices for the chosen source
- Support a simple single-resolution selector rather than independent `z/y/x`
  downsampling controls.
- Publish the exported volume back into the ClearEx store under a namespaced
  results path.
- Support TIFF export in both requested layouts:
  - one single-file export
  - one file per exported `(t, p, c)` volume
- Keep the existing `mip_export` workflow unchanged as the projection-only
  export path.

## Non-goals

- Do not export every active visualization layer in one combined operation.
  `volume_export` targets one selected image source at a time.
- Do not depend on transient live napari viewer state for `t` or `c`
  selection.
- Do not replace or broaden `mip_export`.
- Do not introduce independent anisotropic export controls for `z`, `y`, and
  `x` in the first version.

## Context

The current visualization family contains:

- `display_pyramid`
- `visualization`
- `render_movie`
- `compile_movie`
- `mip_export`

The existing `visualization` workflow owns a position selector, volume-layer
configuration, and napari launch metadata, but it does not own persistent tab
selectors for the currently active `t` or `c` values. Scraping those values
from a live viewer would be brittle and hard to test. The export workflow
therefore needs its own explicit `t/p/c` selectors for preview exports.

ClearEx runtime policy requires new image-producing analyses to:

- write executable arrays into
  `clearex/runtime_cache/results/<analysis>/latest`
- publish a public OME image collection under `results/<analysis>/latest`
- keep auxiliary metadata under `clearex/results/<analysis>/latest`

That policy drives the storage contract below.

## Chosen Approach

Add a dedicated `volume_export` operation in the visualization workflow family.

This is preferred over extending `mip_export` because volume export and
projection export have materially different products, metadata, and UI. It is
also preferred over a viewer-only export action because the new workflow needs
to support batch export, provenance, normalization, and automated tests.

## Operation Contract

### Operation name and placement

- New operation key: `volume_export`
- Workflow family: visualization/export
- Recommended ordering: after `compile_movie` and before `mip_export`

### Input source

- `input_source` follows the same image-source chooser pattern used by other
  analysis operations.
- The operation targets one image-producing source component or logical alias
  per run.
- `volume_export` should not attempt to export all configured visualization
  volume layers at once.

### Parameters

- `input_source`
  - selected image-producing component or alias
- `export_scope`
  - `current_selection`
  - `all_indices`
- `t_index`
  - used only for `current_selection`
- `p_index`
  - used only for `current_selection`
- `c_index`
  - used only for `current_selection`
- `resolution_level`
  - integer, `0` means full resolution
- `export_format`
  - `ome-zarr`
  - `ome-tiff`
- `tiff_file_layout`
  - `single_file`
  - `per_volume_files`
  - only relevant when `export_format=ome-tiff`

### Default values

- `input_source=data`
- `export_scope=current_selection`
- `t_index=0`
- `p_index=0`
- `c_index=0`
- `resolution_level=0`
- `export_format=ome-zarr`
- `tiff_file_layout=single_file`

## Selection Semantics

### Current selection

`current_selection` exports one explicit source volume chosen by the export
panel's own selectors.

- one selected `t`
- one selected `p`
- one selected `c`
- output logical extent is one `ZYX` volume

The persisted in-store runtime payload is still canonical `TPCZYX`, but with
singleton `t`, `p`, and `c` dimensions:

- shape `(1, 1, 1, z, y, x)`

### All indices

`all_indices` exports the full selected source at the requested resolution
across all available `t/p/c` indices.

- shape remains canonical `TPCZYX`
- no individual selectors are used

## Resolution Policy

- `resolution_level=0` uses the base source component.
- Higher levels first reuse any existing source-adjacent pyramid or
  display-pyramid component for the selected source.
- If the requested level is unavailable, `volume_export` builds the missing
  downsampled helper level as part of the export workflow.
- Missing-level generation must reuse the same downsampling conventions as
  `display_pyramid`.
- Generated helper levels should be stored in the same source-adjacent helper
  layout already used by display-pyramid behavior so they can be reused later.
- Export metadata and provenance must record whether export reused an existing
  level or generated one on demand.

`volume_export` is allowed to auto-build helper levels. `visualization`
continues to avoid auto-building them at viewer-launch time.

## Storage Contract

### Runtime cache

Every `volume_export` run materializes the selected export payload into:

- `clearex/runtime_cache/results/volume_export/latest/data`

This runtime array is always canonical `TPCZYX` and carries explicit axes
metadata compatible with existing ClearEx runtime conventions.

For `current_selection`, the runtime cache contains singleton `t/p/c`
dimensions. For `all_indices`, it contains the full selected dataset at the
requested export resolution.

### Public OME image collection

Every `volume_export` run publishes a public OME image collection under:

- `results/volume_export/latest`

This satisfies the requirement that the exported volume be written back into
the store under a namespaced results path.

Public publication rules:

- the public side uses the existing ClearEx HCS contract
- each field is `TCZYX`
- OME metadata version remains `0.5`
- physical scale is derived from the selected export resolution

Field layout:

- `current_selection`
  - publish one field only
  - public field array shape is `(1, 1, z, y, x)`
- `all_indices`
  - publish one field per exported position
  - each field array shape is `(t, c, z, y, x)`

### Auxiliary metadata and artifacts

Auxiliary metadata is stored at:

- `clearex/results/volume_export/latest`

TIFF artifacts, when requested, are stored under that namespaced subtree as
regular files inside the store directory, for example:

- `clearex/results/volume_export/latest/files/...`

## OME-Zarr Contract

The exported in-store OME image must remain consistent with existing ClearEx
publication behavior and with `d6d` expectations.

Consistency points:

- internal runtime arrays remain canonical `TPCZYX`
- internal array attrs include explicit
  `axes=["t", "p", "c", "z", "y", "x"]`
- public published fields remain `TCZYX`
- published OME metadata continues to use standard multiscales `0.5`

`volume_export` publishes a single-resolution exported dataset. The selected
`resolution_level` controls which source resolution is exported; it does not
require exporting a full multiscale pyramid for the result.

## TIFF Contract

TIFF emission is optional and controlled by `export_format=ome-tiff`.

### Current selection TIFF

For `current_selection`, TIFF export writes one 3D OME-TIFF/BigTIFF volume:

- one file
- one selected `(t, p, c)`
- array logical content is `ZYX`
- physical calibration includes `PhysicalSizeZ`, `PhysicalSizeY`, and
  `PhysicalSizeX`

### All indices, single file

OME-TIFF does not have a clean standard `p` axis. Therefore the
`single_file` contract for `all_indices` is:

- one BigTIFF file
- one image series per exported position
- each series stores `TCZYX` for that position

This preserves position identity without inventing a non-standard TIFF `p`
axis contract.

### All indices, per-volume files

For `per_volume_files`, TIFF export writes one file per exported `(t, p, c)`
volume:

- array logical content is `ZYX`
- position, time, and channel are encoded in the filename and manifest
- physical calibration includes `PhysicalSizeZ`, `PhysicalSizeY`, and
  `PhysicalSizeX`

### TIFF naming

Representative filename patterns:

- current selection:
  - `<source>_lvl<nn>_t<tttt>_p<pppp>_c<cc>.ome.tif`
- all indices single file:
  - `<source>_lvl<nn>_all_indices.ome.tif`
- all indices per-volume:
  - `<source>_lvl<nn>_t<tttt>_p<pppp>_c<cc>.ome.tif`

## Execution Design

### High-level flow

1. Normalize parameters.
2. Resolve the selected source component.
3. Resolve or generate the requested resolution component.
4. Materialize the selected export payload into
   `clearex/runtime_cache/results/volume_export/latest/data`.
5. Publish the public OME image collection under `results/volume_export/latest`.
6. If TIFF export is requested, emit TIFF artifact(s) under
   `clearex/results/volume_export/latest/files`.
7. Persist latest metadata and provenance references.

### Data movement and memory policy

- Avoid full-volume `np.asarray(...)` on batch export paths.
- Local mode may stream chunkwise writes directly from Dask-backed source
  arrays.
- With a Dask client, use chunked reads and bounded writes similar in spirit to
  `mip_export`, but without axis reduction.
- OME-Zarr writes should stream chunkwise into the runtime-cache export target.
- TIFF writes should use BigTIFF when needed and avoid accidental full-memory
  staging for large exports.

### Helper-level generation

- Requested missing helper levels should be generated before payload export.
- Generation is scoped to the selected source component only.
- Helper-level generation should not mutate unrelated components.

## Metadata And Provenance

`clearex/results/volume_export/latest` attrs should include at least:

- `source_component`
- `resolved_resolution_component`
- `resolution_level`
- `generated_resolution_level`
- `export_scope`
- `selection`
  - explicit `t/p/c` selection for `current_selection`
- `export_format`
- `tiff_file_layout`
- `runtime_cache_component`
- `public_root`
- `artifact_paths`
- `storage_policy=latest_only`
- `parameters`
- `run_id`

Provenance behavior:

- register latest output reference with analysis name `volume_export`
- include whether a helper level was reused or generated during the run
- include the chosen source component, export scope, resolution level, and
  TIFF layout

## GUI Design

Add a dedicated `volume_export` parameter panel in the visualization/export
area with the following controls:

- export scope selector
- `t_index` selector
- `p_index` selector
- `c_index` selector
- resolution level selector
- export format selector
- TIFF file layout selector

Enablement rules:

- `t/p/c` selectors are enabled only for `current_selection`
- TIFF layout is enabled only for `export_format=ome-tiff`

The selected source component continues to come from the standard
operation-level input-source chooser rather than a second dedicated selector
inside the `volume_export` panel.

The preview-export selectors are owned by the `volume_export` panel and are not
derived from the live napari viewer state.

## Workflow Integration

Required changes:

- add `volume_export` defaults and normalization in `workflow.py`
- add dispatch and progress plumbing in `main.py`
- add GUI controls, collection, restore, and enable/disable logic in
  `src/clearex/gui/app.py`
- add a visualization subsystem implementation module for `volume_export`
- keep `mip_export` unchanged as the projection-only export path

Input-source behavior:

- `volume_export` should accept previously run image-producing outputs as
  inputs
- `volume_export` itself should not be presented as a preferred upstream
  scientific analysis source by default, even though it writes an image result
  into the store

## Testing Plan

### Workflow normalization

- default parameter coverage for `volume_export`
- validation of `export_scope`
- validation of `export_format`
- validation of `tiff_file_layout`
- clamping and normalization of `t/p/c` selectors
- normalization of `resolution_level`

### GUI tests

- widget creation and parameter collection
- parameter restore from normalized workflow config
- enable/disable rules for:
  - `t/p/c` selectors
  - TIFF file layout control

### Pipeline tests

- current-selection OME-Zarr export
- all-indices OME-Zarr export
- current-selection TIFF export
- all-indices TIFF single-file export
- all-indices TIFF per-volume export
- requested-resolution reuse when helper levels already exist
- on-demand helper-level generation when the requested level is missing
- metadata/provenance payload contents

### Main workflow tests

- dispatch of the new operation
- progress callback mapping
- latest-output registration
- chaining from prior image-producing outputs

### Documentation updates

Update in the same implementation change set:

- `README.md`
- `src/clearex/AGENTS.md`
- `src/clearex/visualization/README.md`
- relevant runtime docs under `docs/source/runtime/`

## Risks And Mitigations

### Large TIFF exports

Risk:
- large all-indices TIFF emission can be slow and file-size heavy

Mitigation:
- support per-volume TIFF layout
- use BigTIFF
- stream writes instead of full-memory staging

### Resolution mismatch confusion

Risk:
- users may assume export resolution always maps to an existing prepared level

Mitigation:
- record whether the level was reused or generated
- expose the selected resolution level clearly in metadata and the GUI

### OME-TIFF position semantics

Risk:
- a literal `TPCZYX` TIFF contract would be poorly standardized

Mitigation:
- use one `TCZYX` series per position for all-position single-file TIFF export

## Final Decision

Implement `volume_export` as a dedicated visualization-family analysis that:

- exports one selected image-producing source at a user-selected resolution
- supports explicit preview export of one `(t, p, c)` volume
- supports batch export of all available indices
- always writes the exported result back into the store under
  `results/volume_export/latest`
- optionally emits TIFF artifacts inside
  `clearex/results/volume_export/latest/files`
- preserves the existing `mip_export` workflow as the separate projection-only
  export path
