Architecture Overview
=====================

Design Goals
------------

The runtime architecture is built around these constraints:

- GUI-first operator workflow, with fully supported headless execution.
- Canonical analysis data layout in ``(t, p, c, z, y, x)``.
- Deterministic latest-output paths for large derived arrays.
- Append-only, FAIR-oriented provenance records.
- Shared configuration model between GUI and headless paths.

Layered Runtime Model
---------------------

ClearEx is intentionally split into layers that can evolve independently:

1. Interface layer:
   GUI in ``clearex.gui.app`` and CLI parsing in ``clearex.io.cli``.
2. Configuration layer:
   typed runtime schema in ``clearex.workflow.WorkflowConfig`` (plus
   ``DaskBackendConfig`` and ``ZarrSaveConfig``).
3. Orchestration layer:
   workflow entrypoint and execution coordinator in ``clearex.main``.
4. Data and metadata layer:
   ingestion/canonical store logic in ``clearex.io.experiment`` and provenance
   persistence in ``clearex.io.provenance``.
5. Analysis layer:
   analysis routines (flatfield, deconvolution, particle detection,
   visualization via ``clearex.<analysis>.pipeline``, with registration hooks
   currently under ``clearex.registration``.

End-to-End Execution Flow
-------------------------

At runtime, control flows through one orchestrator path:

1. Build a ``WorkflowConfig`` from CLI arguments and/or GUI state.
2. Optionally launch GUI and let the operator finalize settings.
3. If input is Navigate ``experiment.yml``, resolve acquisition source data.
4. Materialize or validate canonical Zarr/N5 store.
5. Resolve analysis sequence and per-operation effective inputs.
6. Run selected analyses in sequence.
7. Write latest outputs and append one provenance run record.

Operational Invariants
----------------------

These contracts are stable and expected by multiple modules:

- Canonical base image component is ``data``.
- Canonical base image shape is always six-dimensional in
  ``(t, p, c, z, y, x)`` order.
- Multiscale levels are stored under ``data_pyramid/level_<n>``.
- Large analysis outputs are latest-only under ``results/<analysis>/latest``.
- Provenance run history is append-only under ``provenance/runs``.
- Provenance includes hash chaining for tamper-evident verification.

Analysis Composition Model
--------------------------

Selected operations are not hard-coded into one fixed pipeline. Composition is
driven by normalized per-operation parameters in ``analysis_parameters``:

- ``execution_order`` controls relative ordering between selected operations.
- ``input_source`` controls which prior/latest component an operation reads.
- ``force_rerun`` lets operators bypass provenance-based dedup logic.

This allows one run to execute only one step, or a custom chain of steps,
without changing orchestration code.
