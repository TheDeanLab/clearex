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

GUI Setup Flow
--------------

The first GUI window is an experiment-list driven setup flow:

- ``Load Experiment`` adds one Navigate ``experiment.yml``/``experiment.yaml``.
- ``Create Experiment List`` can either:
  - recursively scan a folder for Navigate experiment descriptors, or
  - reload a saved ClearEx list file (``.clearex-experiment-list.json``).
- Drag and drop accepts individual experiment descriptors, folders to scan,
  and saved list files.
- Selecting an entry in the list automatically loads the metadata panel.
- Double-clicking a list item reloads that experiment's metadata explicitly.
- The current ordered list can be saved back to a reusable
  ``.clearex-experiment-list.json`` file.
- Pressing ``Next`` batch-prepares canonical stores for every listed
  experiment that still needs ingestion, then opens analysis selection for the
  currently selected experiment.

.. figure:: ../../screenshots/setup_dialog_experiment_list.png
   :alt: ClearEx setup dialog showing the themed experiment list pane and metadata panel
   :width: 100%

   Setup dialog with the experiment list pane, automatic metadata loading, and
   themed list-management controls.

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

Visualization Keyframe Capture
------------------------------

When visualization launches napari, keyframe capture is enabled by default:

- Press ``K`` to capture a keyframe.
- Press ``Shift-K`` to remove the most recent keyframe.

The keyframe manifest path defaults to:

- ``<analysis_store>.visualization_keyframes.json``

and can be overridden with ``keyframe_manifest_path`` in visualization
parameters.

Each keyframe stores enough state to recreate the current scene for movie
generation, including:

- camera values (angles, zoom, center, perspective),
- dims state (current step, axis labels, order, and 2D/3D mode),
- layer order and selected/active layers,
- per-layer display configuration (visibility, LUT/colormap, rendering mode,
  blending, opacity, contrast, and transforms when available).

The GUI provides a popup editor (``Layer/View Table...``) for optional
per-layer overrides with columns:

- ``Layer``, ``Visible``, ``LUT/Colormap``, ``Rendering``, ``Annotation``.
