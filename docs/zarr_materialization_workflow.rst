Zarr Materialization Workflow and Verification
==============================================

Implemented workflow
--------------------

The ingestion path for Navigate ``experiment.yml`` now materializes
source data into a canonical 6D Zarr ``data`` array and writes
configured resolution-pyramid levels with Dask-parallel writes.

1. Load and normalize experiment metadata from ``experiment.yml``.
2. Resolve source data path from acquisition outputs.
3. Open source directly with format-specific, Dask-compatible loaders:

   -  TIFF/OME-TIFF via ``tifffile`` + ``da.from_zarr(...)``
   -  H5/HDF5 via ``h5py`` + ``da.from_array(...)``
   -  Zarr/N5 via ``da.from_zarr(...)``
   -  NPY/NPZ via ``numpy`` + ``da.from_array(...)``

4. Infer/normalize axis order and coerce to canonical
   ``(t, p, c, z, y, x)``.
5. Materialize base ``data`` using GUI-configured chunks.
6. Build downsampled levels in ``data_pyramid/level_<n>`` according to
   GUI pyramid factors.
7. Persist canonical store metadata and source provenance fields in Zarr
   attributes.

Pyramid downsampling behavior
-----------------------------

-  Pyramid levels are generated in the same materialization run,
   immediately after base ``data`` write.
-  Writes run through Dask and Zarr for chunk-parallel persistence.
-  Levels are generated with stride-based decimation (nearest-neighbor)
   for speed and dtype preservation.
-  Metadata is stored at root and base-array attrs:

   -  ``data_pyramid_levels``
   -  ``data_pyramid_factors_tpczyx``
   -  ``data_pyramid_shapes_tpczyx``
   -  ``data.attrs["pyramid_levels"]``
   -  ``data.attrs["pyramid_factors_tpczyx"]``

GUI workflow (two-step)
-----------------------

The GUI now runs in two windows:

1. **Setup window**

   -  User selects ``experiment.yml``.
   -  User configures Dask backend and Zarr save settings.
   -  User loads and reviews image metadata.
   -  User clicks **Next**.

2. **Analysis window**

   -  Opens after canonical store readiness is confirmed.
   -  User selects analysis operations (deconvolution, particle
      detection, registration, visualization).

Setup-window behavior on **Next**:

-  If target canonical store already exists, setup proceeds directly to
   analysis selection.
-  If target store does not exist, GUI creates it first and shows a
   styled progress dialog with stage updates.
-  After store creation completes, setup closes and analysis-selection
   window opens.

Store path behavior
-------------------

-  If source is already Zarr/N5, the same store path is reused (no
   duplicate store path is created).
-  If source is non-Zarr, output store is written next to
   ``experiment.yml`` as:

   -  ``data_store.zarr``

Dask backend behavior
---------------------

-  Runtime now uses the configured GUI backend as before.
-  Local cluster startup for this I/O-heavy workflow is now
   thread-oriented (``processes=False``).
-  Materialization compute executes on the active Dask client when
   available.
-  For source graphs that cannot be serialized to distributed workers
   (for example some lock-backed TIFF/HDF inputs), execution
   automatically falls back to local threaded compute.

Safety for in-place Zarr conversion
-----------------------------------

When source and destination are the same store and source component is
already ``data``, conversion stages into a temporary component and then
swaps into ``data`` to avoid read/write self-conflicts.

Automated verification
----------------------

Unit tests
~~~~~~~~~~

Executed:

.. code:: bash

   uv run --with pytest --with requests python -m pytest -q tests/test_workflow.py tests/io/test_cli.py tests/io/test_experiment.py tests/io/test_provenance.py

Result: ``40 passed``

Added coverage validates:

-  Non-Zarr source writes to ``data_store.zarr`` in ``experiment.yml``
   directory.
-  Existing Zarr source reuses same store path.
-  Same-component (``data``) in-place Zarr rewrite path works.
-  Canonical shape/chunk expectations and value integrity.
-  Downsampled pyramid levels are written and have expected
   shapes/values.

Lint checks
~~~~~~~~~~~

Executed:

.. code:: bash

   ruff check src/clearex/main.py src/clearex/workflow.py src/clearex/gui/app.py src/clearex/io/cli.py src/clearex/io/experiment.py src/clearex/io/provenance.py

Result: ``All checks passed``

Real-data run (provided dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset root:

-  ``/Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/``

Materialization run on representative cells (``cell_006`` to
``cell_010``) across TIFF, OME-TIFF, H5, N5, and OME-Zarr with chunks
``(1, 1, 1, 8, 128, 128)`` and pyramid factors
``((1,), (1,), (1,), (1, 2, 4), (1, 2, 4), (1, 2, 4))``:

-  ``cell_006`` TIFF -> ``data_store.zarr``, shape
   ``(1, 1, 1, 100, 2048, 2048)``, pyramid shapes
   ``[(1, 1, 1, 50, 1024, 1024), (1, 1, 1, 25, 512, 512)]``, elapsed
   ``6.63s``
-  ``cell_007`` OME-TIFF -> ``data_store.zarr``, shape
   ``(1, 1, 1, 100, 2048, 2048)``, pyramid shapes
   ``[(1, 1, 1, 50, 1024, 1024), (1, 1, 1, 25, 512, 512)]``, elapsed
   ``6.27s``
-  ``cell_008`` H5 -> ``data_store.zarr``, shape
   ``(1, 1, 1, 100, 2048, 2048)``, pyramid shapes
   ``[(1, 1, 1, 50, 1024, 1024), (1, 1, 1, 25, 512, 512)]``, elapsed
   ``29.76s``
-  ``cell_009`` N5 -> same store ``CH00_000000.n5``, shape
   ``(1, 1, 1, 100, 512, 512)``, pyramid shapes
   ``[(1, 1, 1, 50, 256, 256), (1, 1, 1, 25, 128, 128)]``, elapsed
   ``0.69s``
-  ``cell_010`` OME-Zarr -> same store ``CH00_000000.ome.zarr``, shape
   ``(2, 1, 2, 100, 512, 512)``, pyramid shapes
   ``[(2, 1, 2, 50, 256, 256), (2, 1, 2, 25, 128, 128)]``, elapsed
   ``1.46s``

This confirms output-path policy, canonical layout, and actual persisted
downsample pyramid levels on heterogeneous acquisition formats.

Headless workflow smoke test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Executed:

.. code:: bash

   uv run python -m clearex.main --headless --no-gui --file /Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/cell_001/experiment.yml --dask

Result:

-  Workflow completed successfully.
-  Source TIFF was materialized to
   ``/Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/cell_001/data_store.zarr``.
-  Provenance run record was persisted in the same store.
