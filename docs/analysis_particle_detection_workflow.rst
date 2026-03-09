.. _analysis-workflow-particle-detection-dask--zarr:

Analysis Workflow: Particle Detection (Dask + Zarr)
===================================================

Implemented behavior
--------------------

This implementation adds a runnable analysis path for particle detection
on canonical ClearEx stores (``data_store.zarr``, ``.n5``,
``.ome.zarr``) with CPU-oriented Dask execution and persisted outputs in
``results/``.

Dask backend policy by workload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **I/O-heavy ingestion/materialization**: local Dask uses
   ``processes=False`` (threads).
-  **CPU-heavy analysis**: local Dask uses ``processes=True`` (worker
   processes).
-  Backend mode still follows the GUI-selected backend configuration.

Default analysis parameter dictionary
-------------------------------------

``WorkflowConfig`` now carries ``analysis_parameters``, keyed by
analysis method, with defaults in
``DEFAULT_ANALYSIS_OPERATION_PARAMETERS``.

Current keys include:

-  ``deconvolution``
-  ``particle_detection``
-  ``registration``
-  ``visualization``

All operation dictionaries now include:

-  ``execution_order`` (int): defines run order when multiple routines
   are selected.
-  ``input_source`` (str): source dataset key/component (default
   ``data``).

The ``particle_detection`` defaults include:

-  channel selection (``channel_index``)
-  chunk/detection profile metadata (``chunk_basis``,
   ``detect_2d_per_slice``)
-  overlap controls (``use_map_overlap``, ``overlap_zyx``)
-  memory hint (``memory_overhead_factor``)
-  detection parameters (``bg_sigma``, ``fwhm_px``,
   ``sigma_min_factor``, ``sigma_max_factor``, ``threshold``,
   ``overlap``, ``exclude_border``)
-  optional post-filters (``eliminate_insignificant_particles``,
   ``remove_close_particles``, ``min_distance_sigma``)

GUI updates (second window)
---------------------------

The analysis-selection dialog remains split into two halves:

-  **Left panel**: operation selection rows with:

   -  enable checkbox,
   -  execution-order spinner,
   -  ``Configure`` button.

-  **Right panel**: stacked operation-parameter panel.

   -  Only one operation panel is visible at a time.
   -  Clicking ``Configure`` on one operation hides the previous
      operation panel.
   -  Unselected operations cannot be configured.

Implemented controls:

-  Per-operation ``Input source`` selector (raw ``data`` or prior
   selected upstream output).
-  Particle-detection parameter form:

   -  channel index,
   -  preprocessing + detector numeric parameters,
   -  overlap toggle and ``overlap_zyx``,
   -  optional significance/proximity filtering toggles,
   -  minimum-distance setting.

-  Parameter-help panel:

   -  Updates when hovering/focusing a specific parameter.
   -  Includes verbose hints for fields such as ``fwhm_px``.

-  Theme consistency:

   -  Operation parameter controls use the same dark theme.
   -  Dropdown item views use dark backgrounds and readable selected
      text.

Execution sequence and upstream input behavior
----------------------------------------------

-  Runtime now resolves selected analyses by ``execution_order`` instead
   of fixed hard-coded ordering.
-  Per-step ``input_source`` is resolved at runtime:

   -  ``data`` maps to canonical ``data`` array.
   -  Operation keys map to canonical latest component paths.
   -  Explicit Zarr component paths are also supported.

-  Particle detection now reads from configurable ``input_source``.
-  If requested component is missing, particle detection falls back to
   ``data`` with a warning.

Particle detection execution workflow
-------------------------------------

1. Read configured input component from selected store
   (``t,p,c,z,y,x``).
2. Select user-configured channel.
3. Build chunk tasks at native 3D chunk boundaries.
4. For each chunk:

   -  preprocess each z-slice (``preprocess(..., bg_sigma=...)``)
   -  detect blobs per slice (``detect_particles(...)``)
   -  convert chunk-local detections to global ``(t,p,c,z,y,x)``
      coordinates
   -  optionally apply significance/proximity filters
   -  optionally apply core-region masking for overlap mode

5. Merge and sort detections globally.
6. Save latest outputs under:

   -  ``results/particle_detection/latest/detections`` (columns:
      ``t,p,c,z,y,x,sigma,intensity``)
   -  ``results/particle_detection/latest/points_tzyx`` (Napari-friendly
      points)

7. Register latest-output reference and persist provenance.

Output format for Napari
------------------------

Napari-friendly point coordinates are persisted at:

-  ``results/particle_detection/latest/points_tzyx``

Detection table metadata includes column names and points-axis metadata
in Zarr attrs.

Provenance integration
----------------------

-  Particle-detection step parameters and run summary are included in
   provenance steps.
-  Workflow provenance now includes ``analysis_parameters``.
-  Output records include latest component and detection summary.
-  Latest output reference for particle detection is registered in
   provenance metadata.

Verification
------------

Unit tests
~~~~~~~~~~

Executed:

.. code:: bash

   uv run --with pytest --with requests python -m pytest -q tests/detect/test_pipeline.py tests/test_workflow.py tests/io/test_cli.py tests/io/test_experiment.py tests/io/test_provenance.py

Result:

-  ``48 passed``

Lint checks
~~~~~~~~~~~

Executed:

.. code:: bash

   uv run ruff check src/clearex/main.py src/clearex/workflow.py src/clearex/gui/app.py src/clearex/io/cli.py src/clearex/io/experiment.py src/clearex/io/provenance.py src/clearex/detect/pipeline.py tests/detect/test_pipeline.py tests/test_workflow.py

Result:

-  ``All checks passed``

Headless real-data smoke test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Executed:

.. code:: bash

   uv run python -m clearex.main --headless --no-gui --file /Users/Dean/Desktop/kevin/20260307_lung_mv3_488nm/cell_009/CH00_000000.n5 --particle-detection --dask

Observed:

-  Analysis backend ran with ``LocalCluster`` in process mode
   (``processes=True``).
-  Progress updates were emitted for chunk completion.
-  Results were written to ``results/particle_detection/latest``.
-  Provenance run record was persisted
   (``run_id=77d6e00e9ade47b0a524b964a6bc435d`` in this verification
   run).

Post-check of stored outputs:

-  ``detections`` shape: ``(10663, 8)``
-  ``points_tzyx`` shape: ``(10663, 4)``
-  ``channel_index``: ``0``
