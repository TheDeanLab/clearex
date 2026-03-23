.. _analysis-workflow-particle-detection-dask--ome-zarr:

Analysis Workflow: Particle Detection (OME-Zarr + Dask)
=======================================================

Implemented behavior
--------------------

This analysis path runs particle detection on canonical ClearEx OME-Zarr stores
(``data_store.ome.zarr`` or another canonical ``*.ome.zarr`` input) with
CPU-oriented Dask execution and persisted outputs in the ClearEx namespace.

Dask backend policy by workload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **I/O-heavy ingestion/materialization**: local Dask uses
  ``processes=False`` (threads).
- **CPU-heavy analysis**: local Dask uses ``processes=True`` (worker
  processes).
- Backend mode still follows the GUI-selected backend configuration.

Default analysis parameter dictionary
-------------------------------------

``WorkflowConfig`` carries ``analysis_parameters``, keyed by analysis method,
with defaults in ``DEFAULT_ANALYSIS_OPERATION_PARAMETERS``.

All operation dictionaries include:

- ``execution_order`` (int): defines run order when multiple routines are
  selected.
- ``input_source`` (str): logical source alias or explicit internal component
  path (default ``data``).

The ``particle_detection`` defaults include:

- channel selection (``channel_index``),
- chunk/detection profile metadata (``chunk_basis``,
  ``detect_2d_per_slice``),
- overlap controls (``use_map_overlap``, ``overlap_zyx``),
- memory hint (``memory_overhead_factor``),
- detection parameters (``bg_sigma``, ``fwhm_px``,
  ``sigma_min_factor``, ``sigma_max_factor``, ``threshold``,
  ``overlap``, ``exclude_border``),
- optional post-filters (``eliminate_insignificant_particles``,
  ``remove_close_particles``, ``min_distance_sigma``).

Execution sequence and upstream input behavior
----------------------------------------------

- Runtime resolves selected analyses by ``execution_order`` instead of fixed
  hard-coded ordering.
- Per-step ``input_source`` is resolved at runtime:

  - ``data`` maps to ``clearex/runtime_cache/source/data``
  - operation keys map to internal runtime-cache result components
  - explicit internal component paths are also supported

- Particle detection reads from the resolved source component.
- If the requested component is missing, particle detection raises an input
  dependency error instead of silently falling back.

Particle detection execution workflow
-------------------------------------

1. Read the configured input component from the selected canonical store
   (``t,p,c,z,y,x``).
2. Select the configured channel.
3. Build chunk tasks at native 3D chunk boundaries.
4. For each chunk:

   - preprocess each z-slice (``preprocess(..., bg_sigma=...)``),
   - detect blobs per slice (``detect_particles(...)``),
   - convert chunk-local detections to global ``(t,p,c,z,y,x)``
     coordinates,
   - optionally apply significance/proximity filters,
   - optionally apply core-region masking for overlap mode.

5. Merge and sort detections globally.
6. Save latest outputs under:

   - ``clearex/results/particle_detection/latest/detections``
   - ``clearex/results/particle_detection/latest/points_tzyx``

7. Register the latest-output reference and persist provenance.

Output format for Napari
------------------------

Napari-friendly point coordinates are persisted at:

- ``clearex/results/particle_detection/latest/points_tzyx``

Detection table metadata includes column names and points-axis metadata in Zarr
attrs.

Provenance integration
----------------------

- Particle-detection step parameters and run summary are included in
  provenance steps.
- Workflow provenance includes ``analysis_parameters``.
- Output records include latest component and detection summary.
- Latest output reference for particle detection is registered under
  ``clearex/provenance/latest_outputs/particle_detection``.

Verification
------------

When this workflow changes, validation should cover:

- logical input-source resolution,
- chunk task planning and global coordinate stitching,
- correct write locations in ``clearex/results/particle_detection/latest``,
- provenance latest-output registration,
- headless CLI execution against a canonical ``*.ome.zarr`` store.
