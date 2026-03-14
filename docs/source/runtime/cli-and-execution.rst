CLI and Execution Modes
=======================

Command Surface
---------------

ClearEx installs the ``clearex`` command.

Current primary arguments are:

- ``--flatfield``
- ``--deconvolution``
- ``--particle-detection``
- ``--registration``
- ``--visualization``
- ``--file``
- ``--dask`` / ``--no-dask``
- ``--chunks``
- ``--gui`` / ``--no-gui``
- ``--headless``

Execution Modes
---------------

The entrypoint is GUI-first by default:

- ``clearex`` attempts GUI launch.
- ``--headless`` forces non-interactive mode.
- ``--no-gui`` disables GUI launch attempts.
- GUI launch failures (missing display/runtime issues) fall back to headless
  mode with warnings.

Workflow Selection
------------------

Each analysis flag is independent. You can run a single operation or multiple
operations in one run.

Examples:

.. code-block:: bash

   # GUI-first default
   clearex

.. code-block:: bash

   # Headless chained run
   clearex --headless \
     --file /path/to/experiment.yml \
     --flatfield --deconvolution --particle-detection

.. code-block:: bash

   # Headless visualization only
   clearex --headless \
     --file /path/to/data_store.zarr \
     --visualization

Interchangeable Routine Composition
-----------------------------------

In orchestration, routines are composed from normalized
``analysis_parameters`` rather than hard-coded fixed order:

- ``execution_order`` decides sequence among selected routines.
- ``input_source`` decides which component each routine reads.
- ``force_rerun`` can override provenance-based skip behavior.

This allows operators to rerun one stage, swap stage order, or run partial
chains without changing the code path in ``main.py``.

Input Source Resolution
-----------------------

Runtime source aliases currently include:

- ``data`` -> ``data``
- ``flatfield`` -> ``results/flatfield/latest/data``
- ``deconvolution`` -> ``results/deconvolution/latest/data``
- ``registration`` -> ``results/registration/latest/data``

When a requested source component does not exist, operation-specific fallback
logic can revert to ``data`` to keep workflows operable.

Progress and Run Lifecycle
--------------------------

Execution progresses through these coarse stages:

1. Resolve workflow and inputs.
2. Materialize canonical store when needed.
3. Execute selected analyses in resolved order.
4. Persist latest outputs and append provenance run record.

GUI execution uses explicit progress callbacks and per-run logging in the
resolved workflow log directory.
