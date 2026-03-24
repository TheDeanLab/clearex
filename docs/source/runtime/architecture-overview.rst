Architecture Overview
=====================

Design Goals
------------

The runtime architecture is built around these constraints:

- GUI-first operator workflow, with fully supported headless execution.
- Canonical persisted store format is OME-Zarr v3 (``*.ome.zarr``).
- Public image interoperability uses OME-Zarr HCS collections.
- ClearEx internal analysis arrays keep canonical ``(t, p, c, z, y, x)``
  layout for execution kernels and workflow chaining.
- Metadata-only spatial calibration for Navigate multiposition placement.
- Deterministic latest-output publication for image and non-image artifacts.
- Append-only, FAIR-oriented provenance records.
- Shared configuration model between GUI and headless paths.

Layered Runtime Model
---------------------

ClearEx is intentionally split into layers that can evolve independently:

1. Interface layer:
   GUI in ``clearex.gui.app`` and CLI parsing in ``clearex.io.cli``.
2. Configuration layer:
   typed runtime schema in ``clearex.workflow.WorkflowConfig`` (plus
   ``DaskBackendConfig``, ``ZarrSaveConfig``, and
   ``SpatialCalibrationConfig``).
3. Orchestration layer:
   workflow entrypoint and execution coordinator in ``clearex.main``.
4. Data and metadata layer:
   ingestion in ``clearex.io.experiment``, OME publication and migration in
   ``clearex.io.ome_store``, and provenance persistence in
   ``clearex.io.provenance``.
5. Analysis layer:
   analysis routines (flatfield, deconvolution, shear transform, registration,
   particle detection, uSegment3D, visualization, MIP export) via
   ``clearex.<analysis>.pipeline``.

End-to-End Execution Flow
-------------------------

At runtime, control flows through one orchestrator path:

1. Build a ``WorkflowConfig`` from CLI arguments and/or GUI state.
2. Optionally launch GUI and let the operator finalize settings.
3. Resolve acquisition source data or target store.
4. Materialize or validate canonical OME-Zarr store state.
5. Resolve analysis sequence and per-operation logical inputs.
6. Run selected analyses against runtime-cache image components.
7. Publish public OME image outputs and append one provenance run record.

Operational Invariants
----------------------

These contracts are stable and expected by multiple modules:

- Canonical public source image collection is the root OME HCS layout:
  ``A/1/<field>/<level>``.
- Canonical internal source component is
  ``clearex/runtime_cache/source/data``.
- Canonical internal image shape is always six-dimensional in
  ``(t, p, c, z, y, x)`` order.
- Internal multiscale source levels are stored under
  ``clearex/runtime_cache/source/data_pyramid/level_<n>``.
- Public image-producing analysis outputs are published under
  ``results/<analysis>/latest``.
- Internal image-producing analysis arrays live under
  ``clearex/runtime_cache/results/<analysis>/latest``.
- ClearEx-owned metadata, provenance, GUI state, and non-image artifacts live
  under ``clearex/...``.
- Store-level placement metadata is persisted in
  ``clearex/metadata["spatial_calibration"]``.
- Provenance run history is append-only under ``clearex/provenance/runs``.
- Legacy root ``data``, root ``data_pyramid``, and
  ``results/<analysis>/latest/data`` layouts are migration-only and are not the
  canonical public contract.

Analysis Composition Model
--------------------------

Selected operations are not hard-coded into one fixed pipeline. Composition is
driven by normalized per-operation parameters in ``analysis_parameters``:

- ``execution_order`` controls relative ordering between selected operations.
- ``input_source`` controls which logical upstream image source an operation
  reads.
- ``force_rerun`` lets operators bypass provenance-based dedup logic.

This allows one run to execute only one step, or a custom chain of steps,
without changing orchestration code.
