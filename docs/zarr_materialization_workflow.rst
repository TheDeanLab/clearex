OME-Zarr Materialization and Migration Workflow
===============================================

Implemented workflow
--------------------

The ingestion path for Navigate ``experiment.yml`` materializes source data
into a canonical OME-Zarr v3 store and preserves ClearEx's internal execution
arrays alongside public OME image collections.

1. Load and normalize experiment metadata from ``experiment.yml``.
2. Resolve source data path from acquisition outputs.
3. Open source directly with format-specific loaders:

   - TIFF/OME-TIFF
   - H5/HDF5
   - generic Zarr/N5
   - NumPy ``.npy`` / ``.npz``
   - canonical OME-Zarr

4. Infer and normalize axis order, coercing to canonical
   ``(t, p, c, z, y, x)`` for internal execution.
5. Materialize the internal source array at
   ``clearex/runtime_cache/source/data`` using GUI-configured chunks.
6. Build internal source pyramid levels at
   ``clearex/runtime_cache/source/data_pyramid/level_<n>`` according to GUI
   pyramid factors.
7. Persist namespaced store metadata in ``clearex/metadata``.
8. Compute multiposition translations from Navigate stage rows plus the
   effective spatial calibration.
9. Publish the public root OME HCS source collection
   ``A/1/<field>/<level>``.

Canonical store path behavior
-----------------------------

- Existing canonical OME-Zarr stores are reused in place.
- Non-canonical inputs materialize to ``data_store.ome.zarr`` beside
  ``experiment.yml``.
- Legacy ClearEx ``.zarr`` / ``.n5`` stores are not reused as canonical inputs
  directly. They must be migrated.

Public vs internal layout
-------------------------

Public OME contract
~~~~~~~~~~~~~~~~~~~

- Root source image collection: ``A/1/<field>/<level>``
- Public image-analysis collections: ``results/<analysis>/latest``

Internal ClearEx contract
~~~~~~~~~~~~~~~~~~~~~~~~~

- Source runtime array: ``clearex/runtime_cache/source/data``
- Source runtime pyramid: ``clearex/runtime_cache/source/data_pyramid/level_*``
- Image-analysis runtime outputs:
  ``clearex/runtime_cache/results/<analysis>/latest/data``
- Non-image artifacts and metadata:
  ``clearex/results/<analysis>/latest``
- Provenance:
  ``clearex/provenance/runs/<run_id>``

Legacy-store migration
----------------------

Use the explicit migration command to upgrade pre-OME ClearEx stores:

.. code-block:: bash

   clearex --migrate-store /path/to/legacy_store.zarr

Optional destination and overwrite flags are available:

.. code-block:: bash

   clearex --migrate-store /path/to/legacy_store.n5 \
     --migrate-output /path/to/migrated_store.ome.zarr \
     --migrate-overwrite

Migration copies legacy source/runtime arrays into the new namespaced layout,
copies provenance and auxiliary analysis artifacts, and republishes public OME
image collections.

Verification expectations
-------------------------

When this workflow changes, validation should cover:

- canonical path selection to ``data_store.ome.zarr``,
- internal source-array and pyramid completeness,
- public OME root metadata and HCS publication,
- preservation of spatial calibration / position translations,
- migration of legacy source data and representative analysis outputs,
- OME-aware reader selection of the public collection.
