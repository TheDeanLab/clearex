Ingestion and Canonical Store
=============================

Navigate Metadata as Acquisition Entrypoint
-------------------------------------------

For Navigate runs, ``experiment.yml``/``experiment.yaml`` is the first-class
entrypoint. ``clearex.io.experiment`` parses:

- save directory and declared file type,
- timepoints/z steps/channels/positions,
- camera dimensions,
- pixel size metadata,
- multiposition metadata (including ``multi_positions.yml`` when available).

Source Data Resolution
----------------------

Source candidates are resolved from ordered search roots:

1. optional user override directory,
2. ``Saving.save_directory``,
3. directory containing ``experiment.yml``.

File-type aliases are normalized (for example OME-TIFF/OME-ZARR aliases), and
TIFF discovery prefers primary stack files over MIP preview artifacts.

Supported Source Inputs
-----------------------

Materialization supports:

- TIFF/OME-TIFF,
- H5/HDF5/HDF,
- Zarr/N5 (including nested group layouts),
- NumPy ``.npy`` and ``.npz``.

Special collection logic is implemented for:

- Navigate TIFF ``Position*/CH*`` collections (stacked into canonical
  dimensions),
- Navigate BDV H5/N5 setup collections (mapped with companion XML metadata).

Canonical Store Path Policy
---------------------------

``resolve_data_store_path`` follows this policy:

- Source already Zarr/N5: reuse source store path in place.
- Source not Zarr/N5: write canonical store to ``data_store.zarr`` beside
  ``experiment.yml``.

Canonical Layout Contract
-------------------------

Base analysis data:

- component: ``data``
- axis order: ``(t, p, c, z, y, x)``

Pyramid levels:

- ``data_pyramid/level_1``, ``data_pyramid/level_2``, ...
- per-axis factors normalized from workflow save settings

Store metadata captures source path/component/axes and effective write strategy
for reproducibility.

Materialization Lifecycle
-------------------------

``materialize_experiment_data_store`` performs:

1. source open + metadata extraction,
2. axis coercion to canonical order,
3. chunk normalization,
4. canonical base data writes,
5. pyramid level materialization,
6. ingestion completion metadata update.

If a store is already complete for expected chunks/pyramid settings,
materialization returns quickly without rewriting data.

Ingestion Progress and Resume
-----------------------------

Ingestion progress is tracked in store metadata and validated via:

- ``has_canonical_data_component``
- ``has_complete_canonical_data_store``

Progress records include completion counters for base/pyramid regions. This
enables robust completion checks and resume-aware writes for interrupted runs.

Operational Rule for Downstream Analyses
----------------------------------------

After canonical ``data`` is established, downstream analyses treat base source
data as immutable and write derived outputs under ``results/<analysis>/latest``.
