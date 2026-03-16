Module Map
==========

This page tracks the currently implemented module surface and runtime role.

Core Orchestration
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Area
     - Primary modules
     - Current role
   * - Entrypoint
     - ``clearex.main``
     - GUI/headless launch, ingestion + analysis orchestration, provenance
       persistence.
   * - Shared runtime schema
     - ``clearex.workflow``
     - Typed workflow config, operation parameter normalization, execution
       sequence resolution, Dask backend config/serialization.
   * - GUI
     - ``clearex.gui.app``, ``clearex.gui.spacing``
     - Setup dialog, backend/Zarr settings dialogs, analysis selection and
       parameter collection, workflow progress UI.
   * - CLI and logging
     - ``clearex.io.cli``, ``clearex.io.log``
     - Command-line parser and runtime logger setup.

Data and Metadata
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Area
     - Primary modules
     - Current role
   * - Experiment ingestion
     - ``clearex.io.experiment``
     - Navigate experiment parsing, source discovery, canonical Zarr/N5
       materialization, pyramid generation, completion checks.
   * - Provenance
     - ``clearex.io.provenance``
     - Append-only run records, latest-output references, history summaries,
       hash-chain verification.
   * - Data reading
     - ``clearex.io.read``
     - Reader abstraction used for metadata loading and non-Navigate inputs.

Analysis Modules
----------------

The following operations are wired into ``clearex.main``:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Operation
     - Primary module
     - Runtime status
   * - Flatfield correction
     - ``clearex.flatfield.pipeline``
     - Integrated and executable; writes latest outputs in canonical results
       namespace.
   * - Deconvolution
     - ``clearex.deconvolution.pipeline``
     - Integrated and executable with canonical-store inputs.
   * - Particle detection
     - ``clearex.detect.pipeline``
     - Integrated and executable with canonical-store inputs.
   * - Visualization
     - ``clearex.visualization.pipeline``
     - Integrated and executable; supports napari launch behavior and optional
       overlays, interactive keyframe capture, and persisted movie-ready
       keyframe manifests.
   * - Registration
     - Registration modules under ``clearex.registration``
     - Selectable in workflow config, but canonical 6D-store integration in
       ``main.py`` is currently marked as not integrated and is skipped.

Supporting Algorithm Packages
-----------------------------

ClearEx also includes reusable lower-level modules used by analysis routines:

- ``clearex.filter``
- ``clearex.fit``
- ``clearex.preprocess``
- ``clearex.segmentation``
- ``clearex.stats``
- ``clearex.file_operations``
- ``clearex.context``
- ``clearex.plot``

These modules are available for internal reuse and future analysis expansion,
even when they are not direct top-level workflow steps.
