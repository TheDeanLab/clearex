Dask Backends
=============

Backend Modes
-------------

ClearEx runtime supports three Dask backend modes via
``workflow.DaskBackendConfig``:

1. ``local_cluster`` (``LocalCluster``)
2. ``slurm_runner`` (``SLURMRunner`` via scheduler file)
3. ``slurm_cluster`` (``SLURMCluster`` worker launch)

Backend Configuration Model
---------------------------

Backend config is structured as:

- ``LocalClusterConfig``:
  ``n_workers``, ``threads_per_worker``, ``memory_limit``,
  ``local_directory``.
- ``SlurmRunnerConfig``:
  ``scheduler_file``, ``wait_for_workers``.
- ``SlurmClusterConfig``:
  workers/cores/processes/memory/interface/walltime/queue/job directives plus
  scheduler options and operator email for notifications.

Execution Policy
----------------

Backend startup is lazy and workload-aware:

- I/O backend is started only when needed for Navigate materialization.
- Analysis backend is started only when selected operations require it.
- Visualization-only and non-Dask-required runs avoid unnecessary cluster
  startup.

For local mode, runtime can auto-recommend aggressive settings based on host
CPU/memory/GPU context and canonical chunk sizing.

For GPU-enabled uSegment3D analysis on ``local_cluster``, runtime applies an
additional safety cap: worker count is limited to visible GPU count (1 worker
per GPU) to reduce GPU overcommit and worker restarts.

Where Backend Selection Happens
-------------------------------

- GUI: backend is configured in the setup flow.
- Headless CLI: current CLI flags do not expose backend mode selection; default
  workflow backend is local cluster.

The selected backend config is still captured in runtime provenance payloads.

Persistence of Last-Used Backend Settings
-----------------------------------------

GUI startup persists backend settings for operator convenience:

- Settings directory: ``~/.clearex``
- File: ``~/.clearex/dask_backend_settings.json``
- On launch: directory is created if missing.
- On load: missing/invalid settings safely fall back to software defaults.
- On successful setup acceptance: current backend settings are saved.

Serialization helpers live in ``workflow.py``:

- ``dask_backend_to_dict``
- ``dask_backend_from_dict``

This keeps persistence, provenance payloads, and GUI hydration aligned.
