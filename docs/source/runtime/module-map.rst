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
     - GUI/headless launch, ingestion + analysis orchestration, OME output
       publication, and provenance persistence.
   * - Shared runtime schema
     - ``clearex.workflow``
     - Typed workflow config, operation parameter normalization, execution
       sequence resolution, and logical input-source mapping.
   * - GUI
     - ``clearex.gui.app``, ``clearex.gui.spacing``
     - Setup dialog, backend / OME-Zarr settings dialogs, analysis selection,
       parameter collection, and workflow progress UI.
   * - CLI and logging
     - ``clearex.io.cli``, ``clearex.io.log``
     - Command-line parser, migration entrypoints, and runtime logger setup.

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
     - Navigate experiment parsing, source discovery, canonical OME-Zarr
       materialization, runtime-cache generation, and completion checks.
   * - OME store helpers
     - ``clearex.io.ome_store``
     - OME-Zarr path helpers, namespaced metadata helpers, public HCS
       publication, and legacy-store migration.
   * - Provenance
     - ``clearex.io.provenance``
     - Append-only run records, latest-output references, history summaries,
       and hash-chain verification.
   * - Data reading
     - ``clearex.io.read``
     - Reader abstraction used for metadata loading and non-Navigate inputs,
       with OME-aware array selection.

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
     - Integrated and executable; writes runtime-cache image outputs and
       publishes public OME results.
   * - Deconvolution
     - ``clearex.deconvolution.pipeline``
     - Integrated and executable with OME-Zarr runtime inputs and public OME
       output publication.
   * - Shear transform
     - ``clearex.shear.pipeline``
     - Integrated and executable with runtime-cache output publication.
   * - Registration
     - ``clearex.registration.pipeline``
     - Integrated and executable; writes fused runtime-cache outputs, publishes
       public OME results, and stores auxiliary transforms in ``clearex/results``.
   * - uSegment3D
     - ``clearex.usegment3d.pipeline``
     - Integrated and executable with runtime-cache image outputs and public
       OME publication.
   * - Particle detection
     - ``clearex.detect.pipeline``
     - Integrated and executable; writes metadata/table outputs under
       ``clearex/results``.
   * - Display pyramid / visualization
     - ``clearex.visualization.pipeline``
     - Integrated and executable; prepares internal display pyramids,
       launches napari, and persists visualization metadata.
   * - MIP export
     - ``clearex.mip_export.pipeline``
     - Integrated and executable; exports files outside the store and records
       metadata in ``clearex/results``.

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
