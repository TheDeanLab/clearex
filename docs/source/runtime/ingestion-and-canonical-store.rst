Ingestion and Canonical Store
=============================

Navigate Metadata as Acquisition Entrypoint
-------------------------------------------

For Navigate runs, ``experiment.yml``/``experiment.yaml`` is the first-class
entrypoint. ``clearex.io.experiment`` parses:

- save directory and declared file type,
- timepoints, z steps, channels, and positions,
- camera dimensions and pixel-size metadata,
- multiposition metadata (including ``multi_positions.yml`` and its
  ``X/Y/Z/F/THETA`` stage rows when available).

Source Data Resolution
----------------------

Source candidates are resolved from ordered search roots:

1. optional user override directory,
2. ``Saving.save_directory``,
3. directory containing ``experiment.yml``.

File-type aliases are normalized (for example OME-TIFF / OME-Zarr aliases), and
TIFF discovery prefers primary stack files over MIP preview artifacts.

Supported Source Inputs
-----------------------

Materialization supports:

- TIFF/OME-TIFF,
- H5/HDF5/HDF,
- NumPy ``.npy`` and ``.npz``,
- generic Zarr stores,
- Navigate BDV N5 acquisitions routed through ``experiment.yml``,
- canonical OME-Zarr stores.

Special collection logic is implemented for:

- Navigate TIFF ``Position*/CH*`` collections (stacked into canonical
  dimensions),
- Navigate BDV H5/N5 setup collections (mapped with companion XML metadata).

Navigate BDV N5 sources are source-only and are not opened through Zarr APIs.
ClearEx reads ``setup*/timepoint*/s0`` datasets through TensorStore so Dask
ingestion remains parallelized on ``zarr>=3``. Standalone bare ``.n5`` runtime
input remains unsupported in this phase; use ``experiment.yml`` materialization
to convert the source into canonical ``*.ome.zarr``.
If stale legacy ClearEx groups such as ``data`` or ``results`` exist inside the
source ``.n5`` tree, they are ignored for source selection.

Materialization Execution Model
-------------------------------

Pressing ``Next`` in the setup flow starts
``materialize_experiment_data_store(...)`` and enters the Dask-backed ingestion
path. Setup metadata selection itself does not materialize TIFF payloads.

Current execution behavior is format-dependent:

- TIFF/OME-TIFF, HDF5, ``.npy``, and ``.npz`` sources are opened as lazy Dask
  arrays and written in bounded parallel batches, but the write graph currently
  executes through Dask's local threaded scheduler.
- Generic Zarr and canonical OME-Zarr sources can use the active distributed
  Dask ``Client`` for canonical writes.
- Navigate BDV ``.n5`` sources remain Dask-parallel through TensorStore-backed
  reads and the active client path.

This distinction is intentional in the current implementation because some
file-backed graphs, especially TIFF/HDF-backed graphs with locks or
non-serializable handles, do not reliably serialize to distributed workers.

Canonical Store Path Policy
---------------------------

``resolve_data_store_path`` follows this policy:

- Existing canonical OME-Zarr store: reuse the ``*.ome.zarr`` path in place.
- Non-canonical source input: materialize ``data_store.ome.zarr`` beside
  ``experiment.yml``.
- Legacy ClearEx canonical stores (root ``data`` / ``data_pyramid`` layout):
  migrate them first with ``clearex --migrate-store`` before treating them as a
  canonical runtime input.

Canonical Layout Contract
-------------------------

Canonical ClearEx stores have two layers of structure:

Public OME image contract
~~~~~~~~~~~~~~~~~~~~~~~~~

- Root source data is published as a synthetic single-well HCS collection:
  ``A/1/<field>/<level>``.
- Image-producing analyses publish sibling HCS collections under
  ``results/<analysis>/latest``.
- Each field image is ``TCZYX`` with OME multiscale metadata and coordinate
  transforms.

Internal ClearEx execution contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Source runtime array:
  ``clearex/runtime_cache/source/data``
- Source internal pyramids:
  ``clearex/runtime_cache/source/data_pyramid/level_<n>``
- Image-analysis runtime outputs:
  ``clearex/runtime_cache/results/<analysis>/latest/data``
- Image-analysis runtime pyramids:
  ``clearex/runtime_cache/results/<analysis>/latest/data_pyramid/level_<n>``
- ClearEx-owned metadata and non-image artifacts:
  ``clearex/metadata``, ``clearex/provenance``, ``clearex/gui_state``,
  ``clearex/results/<analysis>/latest``

Store-Level Spatial Calibration
-------------------------------

Canonical analysis stores persist optional placement metadata for Navigate
multiposition datasets in ``clearex/metadata["spatial_calibration"]``.

- Schema payload is ``{schema, stage_axis_map_zyx, theta_mode}``.
- Missing metadata resolves to the identity mapping
  ``z=+z,y=+y,x=+x``.
- Calibration is metadata-only and does not rewrite source image payloads.
- GUI setup writes the resolved mapping on ``Next`` for every prepared or
  reused store in the experiment list.
- Headless ``--stage-axis-map`` writes an explicit override after
  materialization for ``experiment.yml`` inputs and before analysis for
  existing canonical OME-Zarr stores.
- Legacy stores without this metadata are backfilled logically as identity
  during migration, while stores that already have a mapping keep it unless the
  operator explicitly overrides it.

Materialization Lifecycle
-------------------------

``materialize_experiment_data_store`` performs:

1. source open and metadata extraction,
2. axis coercion to canonical ``(t, p, c, z, y, x)``,
3. chunk normalization,
4. internal source-array writes to
   ``clearex/runtime_cache/source/data``,
5. internal source-pyramid materialization,
6. stage-to-world translation computation from Navigate stage rows and spatial
   calibration,
7. publication of the root public OME HCS source collection,
8. namespaced metadata update for reproducibility and resume checks.

If a store is already complete for expected chunks and pyramid settings,
materialization returns quickly without rewriting image data.

Ingestion Progress and Resume
-----------------------------

Ingestion progress is tracked in namespaced store metadata and validated via
completion checks over the runtime-cache source component and public OME
metadata.

Progress records include completion counters for base and pyramid regions. This
enables robust completion checks and resume-aware writes for interrupted runs.

Operational Rule for Downstream Analyses
----------------------------------------

After canonical source data is established, downstream analyses should resolve
logical input aliases to internal runtime-cache components. New code should not
treat root arrays or legacy ``results/.../data`` paths as the canonical public
interface.
