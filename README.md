<h1 align="center">ClearEx</h1>
<h2 align="center">Scalable Analytics for Cleared and Expanded Tissue Imaging.</h2>

[![Tests](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml/badge.svg)](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/TheDeanLab/clearex/graph/badge.svg?token=ONldpMpLse)](https://codecov.io/gh/TheDeanLab/clearex)

ClearEx is an open source Python package for scalable analytics of cleared and expanded tissue imaging data.

## Cyclic Immunofluorescence Resubmission Scope

For the cyclic immunofluorescence resubmission, ClearEx is being supplied for
the registration code used in that work. This workflow uses the Python
registration API under `clearex.registration` and the registration examples
under `examples/`; it is separate from the main ClearEx CLI and GUI workflows
described later in this README.

Do not use `clearex`, `clearex --registration`, or the GUI to reproduce the
cyclic immunofluorescence registration workflow. Those entrypoints drive the
newer OME-Zarr tile-registration and fusion runtime. The cyclic
immunofluorescence registration path is accessed programmatically through the
examples and module-level API.

Features used for the cyclic immunofluorescence work:

- Linear/affine and nonlinear/deformable registration of 3D TIFF image volumes
  with ANTsPyX.
- Round-to-reference registration through `ImageRegistration` and the
  `register_round(...)` convenience function.
- Reuse of saved transforms so interrupted or repeated runs can skip completed
  registration stages unless `force_override=True`.
- Optional chunked and parallel chunked registration classes for large volumes.
- Transform application utilities for applying saved affine/nonlinear
  transforms to additional channels or rounds.

Reviewer-facing system requirements for this registration path:

- Python `3.12`.
- Tested ClearEx version: `0.1.1`.
- Tested dependency set: the versions resolved in `uv.lock`; key registration
  dependencies include `antspyx 0.6.2`, `numpy 1.26.4`, `scipy 1.12.0`,
  `tifffile 2025.1.10`, and `scikit-image 0.26.0`.
- Operating systems: macOS, Linux, or Windows where the Python dependencies
  install successfully.
- Non-standard hardware: none required. CPU core count and RAM determine
  practical runtime for large volumes; a GPU is not required for the cyclic
  immunofluorescence registration examples.

Typical installation time on a normal desktop computer is about 5-15 minutes
with `uv` when binary wheels are available. It may take longer on systems that
must compile scientific Python dependencies from source.

To install the software for reviewer use:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

To run a small registration demo on the supplied test dataset:

```bash
uv run python examples/scripts/registration/linear_registration.py
```

The demo downloads a small test dataset from Zenodo into `downloaded_data/`,
runs affine registration, and writes a registered TIFF plus an ANTs transform
file into that same directory. Expected terminal output includes the fixed and
moving image paths, ANTs transform inspection, export messages, and `Done.`.
Typical runtime for the small linear demo is minutes on a normal desktop after
the dataset has downloaded. Full nonlinear or chunked registration of
publication-scale volumes can take tens of minutes to hours depending on image
size, registration accuracy, CPU count, RAM, and storage speed.

To run the cyclic immunofluorescence registration API on your own data:

```python
from clearex.registration import register_round

register_round(
    fixed_image_path="path/to/reference_round.tif",
    moving_image_path="path/to/moving_round.tif",
    save_directory="path/to/registration_output",
    imaging_round=1,
    crop=False,
    enable_logging=True,
)
```

Expected outputs include `linear_transform_<round>.mat`,
`nonlinear_warp_<round>.nii.gz`, registration metric text files, and log output
in the selected save directory. See `examples/scripts/registration/` and
`examples/notebooks/registration/` for the scripts and notebooks covering the
class-based API, functional API, linear registration, nonlinear registration,
transform application, and chunked registration.

## Current Functionality
- GUI-first entrypoint with headless fallback.
- Headless CLI for scripted runs.
- Input support for TIFF/OME-TIFF, generic Zarr / OME-Zarr, HDF5 (`.h5/.hdf5/.hdf`), NumPy (`.npy/.npz`), and Navigate BDV N5 acquisitions through `experiment.yml`.
- Navigate experiment ingestion from `experiment.yml` / `experiment.yaml`.
- Canonical persisted store format is OME-Zarr v3 (`*.ome.zarr`).
- Legacy `.n5` remains a source-only input format. Navigate BDV N5 materialization requires companion `*.xml` metadata and now reads `setup*/timepoint*/s0` datasets through TensorStore so Dask ingestion stays parallelized on `zarr>=3`.
- Public microscopy-facing image data is published as OME-Zarr HCS collections, while ClearEx execution caches and non-image artifacts live under namespaced `clearex/...` groups.
- Internal analysis image layout remains `(t, p, c, z, y, x)` for runtime-cache arrays and analysis kernels.
- Store-level spatial calibration for Navigate multiposition data is persisted in `clearex/metadata` and applied to physical placement metadata without rewriting image data.
- Analysis operations available from the main entrypoint:
  - flatfield (`results/flatfield/latest`)
  - deconvolution (`results/deconvolution/latest`)
  - shear transform (`results/shear_transform/latest`)
  - registration metadata (`clearex/results/registration/latest`)
  - fusion (`results/fusion/latest`)
  - uSegment3D segmentation (`results/usegment3d/latest`)
  - particle detection (`clearex/results/particle_detection/latest`)
  - display-pyramid metadata (`clearex/results/display_pyramid/latest`)
  - visualization metadata (`clearex/results/visualization/latest`) with napari launch
  - volume-export metadata (`clearex/results/volume_export/latest`) with in-store OME-Zarr / OME-TIFF outputs
  - render-movie metadata (`clearex/results/render_movie/latest`) with external PNG frame sets
  - compile-movie metadata (`clearex/results/compile_movie/latest`) with external MP4 / ProRes files
  - MIP export metadata (`clearex/results/mip_export/latest`) with external OME-TIFF / OME-Zarr files
- FAIR-style provenance records are persisted in `clearex/provenance/runs` with append-only run history and hash chaining.

## Canonical Store Contract
- `data_store.ome.zarr` is the canonical materialized store beside `experiment.yml`.
- The public source image collection is a synthetic single-well HCS layout at the store root: `A/1/<field>/<level>`.
- Public image-producing analysis outputs are sibling HCS collections under `results/<analysis>/latest`.
- `registration` is metadata-only under `clearex/results/registration/latest`; `fusion` owns the stitched image output under `results/fusion/latest`.
- `volume_export` writes canonical cache data under `clearex/runtime_cache/results/volume_export/latest/data`; OME-Zarr exports also publish `results/volume_export/latest`, while OME-TIFF artifacts stay in-store under `clearex/results/volume_export/latest/files`.
- Standalone projection exports from `mip_export` are written outside the store as OME-TIFF (`.tif`) or standalone OME-Zarr (`.ome.zarr`) files.
- ClearEx internal execution data lives under:
  - `clearex/runtime_cache/source/...`
  - `clearex/runtime_cache/results/<analysis>/latest/...`
- ClearEx-owned metadata, provenance, GUI state, and non-image artifacts live under:
  - `clearex/metadata`
  - `clearex/provenance`
  - `clearex/gui_state`
  - `clearex/results/<analysis>/latest`
- Legacy root `data`, root `data_pyramid`, and `results/<analysis>/latest/data` layouts are migration-only and are no longer the canonical public contract.

## Installation

### Requirements
- Python `>=3.12,<3.13` (use Python `3.12`)
- macOS/Linux/Windows supported by Python stack
- For GUI usage: display server + `PyQt6` (installed in base dependencies)

Flatfield correction depends on BaSiCPy, which currently pins `scipy<1.13`.
SciPy `1.12.x` does not publish Python `3.13` wheels, so Python `3.13`
installations may fall back to unsupported source builds on Linux clusters.

### Recommended: install with `uv`
Install `uv`:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

#### macOS / Linux

```bash
# Ensure a supported Python version is installed and used
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install ClearEx (editable install)
uv pip install -e .

# Optional extras
uv pip install -e ".[decon]"      # deconvolution Python wrappers (PyPetaKit5D + PSFmodels)
uv pip install -e ".[usegment3d]" # uSegment3D segmentation stack
uv pip install -e ".[dev]"        # tests/lint/dev tools
uv pip install -e ".[docs]"       # docs build stack
uv pip install -e ".[dev,docs,decon,usegment3d]"
```

The `decon` extra installs the Python wrappers only. ClearEx intentionally does
not run the upstream PyPetaKit5D `setup.py` runtime download during `uv`
installation. Deconvolution also requires PetaKit5D MCC assets and MATLAB
Runtime installed on a shared filesystem. On BioHPC, install them once under
`/project` and source the generated environment file before launching ClearEx:

```bash
scripts/install_petakit_runtime.sh
source /project/bioinformatics/Danuser_lab/Dean/dean/matlab_runtime/clearex_petakit_env.sh
```

That file exports:

```bash
export CLEAREX_PETAKIT5D_ROOT=/project/bioinformatics/Danuser_lab/Dean/dean/matlab_runtime/PetaKit5D
export CLEAREX_MATLAB_RUNTIME_ROOT=/project/bioinformatics/Danuser_lab/Dean/dean/matlab_runtime/MATLAB_Runtime/R2023a
```

If these variables or files are missing, ClearEx fails deconvolution preflight
before submitting Dask work and points back to `scripts/install_petakit_runtime.sh`.

#### Windows (PowerShell)

```powershell
# Ensure a supported Python version is installed and used
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\Activate.ps1

# Install ClearEx (editable install)
uv pip install -e .

# Optional extras
uv pip install -e ".[decon]"
uv pip install -e ".[usegment3d]"
uv pip install -e ".[dev]"
uv pip install -e ".[docs]"
uv pip install -e ".[dev,docs,decon,usegment3d]"
```

Alternative repo-aware install with lockfile:

```bash
uv sync --python 3.12
```

If you previously created `.venv` with Python `3.13` or `3.14`, remove it and recreate:

```bash
# macOS / Linux
rm -rf .venv
uv venv --python 3.12
```

```powershell
# Windows (PowerShell)
Remove-Item -Recurse -Force .venv
uv venv --python 3.12
```

### Install with pip (alternative)
#### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### HPC Notes
For HPC-specific setup guidance, see [INSTALLATION_NOTES.md](INSTALLATION_NOTES.md).

## Command Line Interface

ClearEx installs the `clearex` command:

```bash
clearex --help
```

Current CLI usage:

```text
usage: clearex [-h] [--flatfield] [--deconvolution] [--particle-detection]
               [--usegment3d] [--channel-indices CHANNEL_INDICES]
               [--input-resolution-level INPUT_RESOLUTION_LEVEL]
               [--shear-transform] [-r] [--fusion] [--display-pyramid] [-v]
               [--render-movie] [--compile-movie] [--volume-export]
               [--mip-export] [-f FILE]
               [--migrate-store MIGRATE_STORE]
               [--migrate-output MIGRATE_OUTPUT] [--migrate-overwrite]
               [--dask | --no-dask] [--chunks CHUNKS]
               [--stage-axis-map STAGE_AXIS_MAP] [--gui | --no-gui]
               [--headless]
```

### Options
- `--flatfield`: Run flatfield-correction workflow.
- `-f, --file`: Path to input image/store or Navigate `experiment.yml`.
- `--deconvolution`: Run deconvolution workflow.
- `--particle-detection`: Run particle detection workflow.
- `--usegment3d`: Run uSegment3D segmentation workflow.
- `--channel-indices`: uSegment3D channels to process (`0,1,2` or `all`).
- `--input-resolution-level`: uSegment3D input pyramid level (`0`, `1`, ...).
- `--shear-transform`: Run shear-transform workflow.
- `-r, --registration`: Run metadata-only registration workflow.
- `--fusion`: Run stitched-volume fusion from the latest registration result.
- `--display-pyramid`: Prepare reusable display pyramids for visualization.
- `-v, --visualization`: Run visualization workflow.
- `--render-movie`: Render PNG movie frames from captured visualization keyframes.
- `--compile-movie`: Compile rendered PNG frames into MP4 and/or ProRes movies.
- `--volume-export`: Export full 3D volumes from one selected source as OME-Zarr or OME-TIFF.
- `--mip-export`: Export XY/XZ/YZ maximum-intensity projections as OME-TIFF or standalone OME-Zarr.
- `--migrate-store`: Convert one legacy ClearEx `.zarr` / `.n5` store into canonical OME-Zarr v3.
- `--migrate-output`: Optional destination path for `--migrate-store`.
- `--migrate-overwrite`: Overwrite the migration destination.
- `--dask / --no-dask`: Enable/disable Dask-backed reading.
- `--chunks`: Chunk spec for Dask reads, for example `256` or `1,256,256`.
- `--stage-axis-map`: Store-level world `z/y/x` mapping for Navigate multiposition stage coordinates, for example `z=+x,y=none,x=+y`.
- `--gui / --no-gui`: Enable/disable GUI launch (default is `--gui`).
- `--headless`: Force non-interactive mode (overrides `--gui`).

### Common Commands

Launch GUI (default behavior):

```bash
clearex
```

Run headless on a Navigate experiment:

```bash
clearex --headless \
  --file /path/to/experiment.yml \
  --deconvolution --usegment3d --particle-detection
```

Run registration and fusion as separate headless passes so each phase can use a
different Dask worker profile:

```bash
clearex --headless \
  --file /path/to/experiment.yml \
  --registration

clearex --headless \
  --file /path/to/data_store.ome.zarr \
  --fusion
```

Run headless with an explicit stage-to-world axis mapping for Navigate multiposition placement:

```bash
clearex --headless \
  --file /path/to/experiment.yml \
  --visualization \
  --stage-axis-map z=+x,y=none,x=+y
```

Run headless uSegment3D on all channels:

```bash
clearex --headless \
  --file /path/to/experiment.yml \
  --usegment3d \
  --channel-indices all \
  --input-resolution-level 1
```

Run headless particle detection on an existing canonical OME-Zarr store:

```bash
clearex --headless \
  --file /path/to/data_store.ome.zarr \
  --particle-detection
```

Disable Dask lazy loading:

```bash
clearex --headless --no-dask --file /path/to/data_store.ome.zarr --particle-detection
```

Run the movie workflow in separate passes against an existing canonical store:

```bash
clearex --headless \
  --file /path/to/data_store.ome.zarr \
  --visualization

clearex --headless \
  --file /path/to/data_store.ome.zarr \
  --render-movie

clearex --headless \
  --file /path/to/data_store.ome.zarr \
  --compile-movie
```

Run headless volume export against an existing canonical store:

```bash
clearex --headless \
  --file /path/to/data_store.ome.zarr \
  --volume-export
```

Detailed movie timing, overlay, and codec parameters are currently configured
through the GUI or programmatic `WorkflowConfig.analysis_parameters`; the CLI
currently exposes the movie operations as top-level flags.

Migrate one legacy ClearEx store into canonical OME-Zarr v3:

```bash
clearex --migrate-store /path/to/legacy_store.zarr
```

## Runtime Behavior Notes
- If `--file` points to Navigate `experiment.yml`, ClearEx resolves acquisition data and materializes a canonical store first.
- Existing canonical OME-Zarr stores are reused in place only when they already match the ClearEx canonical layout.
- Non-canonical acquisition inputs, including TIFF/OME-TIFF, HDF5, NumPy, generic Zarr, generic source OME-Zarr, Navigate BDV N5 sources, and other Navigate source layouts, materialize to `data_store.ome.zarr` beside `experiment.yml`.
- Bare direct source `.n5` runtime input is not a supported phase-1 workflow. For N5 acquisitions, point `--file` at Navigate `experiment.yml` so ClearEx can resolve BDV XML metadata and materialize canonical `.ome.zarr`.
- Legacy ClearEx `.zarr` / `.n5` stores are not treated as canonical runtime inputs. Migrate them first with `clearex --migrate-store`.
- Generic Zarr-like source selection now prefers validated public OME arrays when metadata is present and falls back to largest-array discovery only when necessary.
- Canonical stores persist `spatial_calibration = {schema, stage_axis_map_zyx, theta_mode}` inside `clearex/metadata`. Missing metadata resolves to the identity mapping `z=+z,y=+y,x=+x`.
- In the setup window, `Spatial Calibration` is configured per listed experiment. Draft mappings are tracked per experiment while the dialog is open, existing stores prefill the control, and `Next` writes the resolved mapping to every reused or newly prepared store before analysis selection opens.
- In headless mode, `--stage-axis-map` writes the supplied mapping to materialized experiment stores and existing canonical OME-Zarr stores before analysis starts. If the flag is omitted, existing store calibration is preserved.
- Deconvolution, particle detection, uSegment3D, fusion, and visualization operations run against canonical OME-Zarr stores.
- Registration and fusion can be run in separate executions. `registration` writes transform/layout metadata only, while `fusion` consumes that metadata and writes the stitched image result.
- Visualization supports multi-volume overlays using logical sources and/or public OME image collections (for example source data plus `results/usegment3d/latest`) with per-layer image/labels display controls.
- Multiposition visualization placement now resolves world `z/y/x` translations from the store-level spatial calibration. Bindings support `X`, `Y`, `Z`, and Navigate focus axis `F` with sign inversion or `none`; `THETA` remains a rotation of the `z/y` plane about world `x`.
- Visualization now probes napari OpenGL renderer info (`vendor`/`renderer`/`version`) and can fail fast when software rendering is detected or GPU rendering cannot be confirmed (`require_gpu_rendering=True`).
- `render_movie` consumes visualization keyframes and writes both latest
  metadata and default PNG frame sets inside the canonical store under
  `clearex/results/render_movie/latest`.
- `compile_movie` consumes the latest render manifest and writes both latest
  metadata and default movie files inside the canonical store under
  `clearex/results/compile_movie/latest`.
- `volume_export` is visualization-driven but source-selectable: it can export
  the active source component as either a published OME-Zarr image collection
  or in-store OME-TIFF artifacts.
- Offline movie rendering rebuilds captured `Points` and `Tracks` layers from
  the keyframe manifest so particle and track overlays can survive beyond the
  live napari session.
- `render_movie` now captures from a visible napari viewer by default. This
  avoids the empty-frame failures seen with hidden/offscreen capture while
  keeping both GPU-backed and CPU/software-rendered movie capture paths usable.
- When the ClearEx GUI is already running, `render_movie` opens that visible
  napari capture viewer in a dedicated subprocess so movie rendering does not
  create Qt/OpenGL objects from the GUI worker thread.
- MIP export writes TIFF outputs as OME-TIFF (`.tif`) with projection-aware physical pixel calibration (`PhysicalSizeX/Y`) derived from source `voxel_size_um_zyx`.
- `mip_export export_format=zarr` writes standalone OME-Zarr v3 image stores (`.ome.zarr`) with axis metadata and physical scale, so the outputs can be opened directly in OME-aware viewers such as napari.
- uSegment3D runs per `(t, p, selected channel)` volume task and publishes the latest result as `results/usegment3d/latest`.
  - GUI channel checkboxes now support selecting multiple channels in one run (`channel_indices`).
  - With GPU-aware `LocalCluster`, separate channel tasks can execute concurrently across GPUs.
  - `input_resolution_level` lets segmentation run on a selected prepared pyramid level.
  - `output_reference_space=level0` upsamples labels back to original resolution.
  - `save_native_labels=True` (when upsampling) also stores native-resolution labels as ClearEx-owned auxiliary artifacts.
- GPU awareness:
  - `gpu=True` requests GPU use for Cellpose/uSegment3D internals.
  - `require_gpu=True` fails fast if CUDA is unavailable.
  - For `LocalCluster` analysis runs with GPU-enabled uSegment3D, ClearEx
    now launches a GPU-pinned worker pool (one process per visible GPU by
    default) and advertises a `GPU=1` worker resource so segmentation tasks
    stay on GPU workers.
  - Local GPU mode can be controlled through `create_dask_client(..., gpu_enabled=True, gpu_device_ids=[...])`.
  - GPU execution still depends on the installed PyTorch/CUDA build supporting
    the device architecture (for example, modern PyTorch wheels may not support
    older Pascal GPUs like Tesla P100 `sm_60`).
  - CuPy must match your CUDA major version (`cupy-cuda12x` for CUDA 12
    environments, `cupy-cuda11x` for CUDA 11 environments). Worker startup
    should expose NVRTC/CUDA runtime libraries on `LD_LIBRARY_PATH`.
- Scalability notes:
  - Distributed execution is parallelized over `(t, p)` volumes through Dask.
  - Intra-volume scalability uses uSegment3D gradient-descent tiling (`aggregation_tile_*` parameters).
  - Visualization multiscale gate is configurable per volume layer:
    - `inherit`: follow global multiscale toggle and use existing pyramids when available.
    - `require`: fail if no pyramid exists for that layer.
    - `off`: force single-scale rendering for that layer.
  - Display pyramids are prepared explicitly through the `display_pyramid`
    operation and recorded under `clearex/results/display_pyramid/latest`.
  - Chunk-wise `map_overlap` stitching across labels is not yet enabled by default because label continuity/relabeling across chunk seams requires additional global reconciliation.
- GUI setup accepts Navigate `experiment.yml`/`experiment.yaml` files and
  managed experiment lists (`.clearex-experiment-list.json`).
- In the setup window you can:
  - load a single experiment,
  - recursively scan a folder for `experiment.yml` files,
  - drag and drop experiments, folders, or saved list files into the
    experiment pane,
  - save the current ordered experiment list for later reuse.
- Selecting an experiment in the setup list loads metadata automatically;
  double-clicking reloads that entry explicitly.
- Pressing `Next` batch-prepares canonical `data_store.ome.zarr` outputs for every
  listed experiment that does not already have a complete store, then opens
  analysis selection for the currently selected experiment.
- The setup dialog persists the last-used Zarr save config across sessions.
- `Rebuild Canonical Store` forces listed experiments to be rebuilt with the
  current chunking and pyramid settings; otherwise existing complete stores are
  reused as-is.
- The analysis window includes an `Analysis Scope` panel where you can:
  - pick which loaded `experiment.yml` drives the current analysis context, or
  - enable batch mode to run the same selected analysis sequence across every
    experiment in the loaded list.
- Analysis parameters now persist per dataset:
  - when you reopen a store, ClearEx restores the last saved GUI state for
    that dataset when available,
  - otherwise it can fall back to the latest completed provenance-backed run
    parameters,
  - `Restore Latest Run Parameters` resets the current dataset view back to the
    most recent completed run.
- This persistence path is part of the GUI contract:
  future analysis widgets and workflows should be wired into the same restore,
  save, and provenance-backed replay mechanism rather than introducing
  one-off state handling.
- The `Running Analysis` progress dialog includes a `Stop Analysis` button that
  requests a cooperative halt at the next progress checkpoint and records the
  run as `cancelled` in provenance.
- If GUI cannot launch (for example no display), ClearEx logs a warning and falls back to headless execution.

### Visualization Keyframes and Manifest
- Keyframe capture is enabled by default during visualization (`capture_keyframes=True`).
- In napari:
  - Press `K` to capture a keyframe.
  - Press `Shift-K` to remove the most recent keyframe.
- Default keyframe manifest path is `<analysis_store>/clearex/results/visualization/latest/keyframes.json` (override with `keyframe_manifest_path`).
- Each keyframe stores reproducible viewer state for movie reconstruction, including:
  - camera (angles, zoom, center, perspective),
  - dims (`current_step`, axis labels, order, and 2D/3D display mode),
  - layer order and selected/active layers,
  - per-layer display state (visibility, LUT/colormap, rendering mode, blending, opacity, contrast, and transforms when available),
  - reconstructable `Points` and `Tracks` overlay layers.
- The GUI exposes a popup table (`Layer/View Table...`) to define optional per-layer overrides and `Annotation` labels that are embedded in the keyframe manifest.
- The GUI also exposes a popup table (`Volume Layers...`) to configure overlay rows with:
  - component path,
  - layer type (`image` or `labels`),
  - channels, visibility, opacity, blending, colormap, rendering, and multiscale policy.
- Visualization parameters include `require_gpu_rendering` (enabled by default). Disable only when running intentionally without a GPU-backed OpenGL context.

### Movie Rendering and Compilation
- `render_movie` renders offline PNG frame sets from the captured keyframes.
- `compile_movie` encodes one selected frame set into MP4, ProRes MOV, or both.
- Recommended workflow:
  - capture keyframes in `visualization`,
  - render a preview pass from a coarser level (`resolution_levels=[1]` or `[2]`),
  - iterate on timing and codec settings with `compile_movie`,
  - re-render at level `0` and the final output size for publication delivery.
- Practical parameter ranges:
  - `default_transition_frames`: `12` to `96` (`48` is a good default),
  - `hold_frames`: `0` to `24`,
  - `orbit_degrees`: `10` to `90`,
  - `flythrough_distance_factor`: `0.02` to `0.20`,
  - `zoom_effect_factor`: `0.05` to `0.25`,
  - `mp4_crf`: `16` to `24`,
  - `render_size_xy`: preview `1280x720` or `1600x900`; final `1920x1080` to `3840x2160`.
- Default movie artifacts stay inside the canonical store:
  - `render_movie`: `clearex/results/render_movie/latest/render_manifest.json`
    and `clearex/results/render_movie/latest/level_<nn>_frames/frame_000000.png`
  - `compile_movie`: `clearex/results/compile_movie/latest/*.mp4` and
    `clearex/results/compile_movie/latest/*.mov`
  - Set `output_directory` only when you explicitly want an external export tree.

## Output Layout (Canonical Store)
- Public OME source image collection:
  - root `A/1/<field>/<level>` (`TCZYX`)
- Public OME image analysis collections:
  - `results/flatfield/latest/A/1/<field>/<level>`
  - `results/deconvolution/latest/A/1/<field>/<level>`
  - `results/shear_transform/latest/A/1/<field>/<level>`
  - `results/fusion/latest/A/1/<field>/<level>`
  - `results/usegment3d/latest/A/1/<field>/<level>`
- ClearEx metadata and runtime namespaces:
  - `clearex/metadata`
  - `clearex/provenance/runs/<run_id>`
  - `clearex/provenance/latest_outputs/<analysis>`
  - `clearex/gui_state`
  - `clearex/results/registration/latest`
  - `clearex/results/render_movie/latest`
  - `clearex/results/compile_movie/latest`
  - `clearex/runtime_cache/source/data`
  - `clearex/runtime_cache/source/data_pyramid/level_*`
  - `clearex/runtime_cache/results/<analysis>/latest/data`
  - `clearex/runtime_cache/results/<analysis>/latest/data_pyramid/level_*`
  - `clearex/results/<analysis>/latest`
- External latest-only projection exports:
  - `<output_directory>/mip_<projection>_... .tif`
  - `<output_directory>/mip_<projection>_... .ome.zarr`
- Default latest-only movie exports inside the store:
  - `clearex/results/visualization/latest/keyframes.json`
  - `clearex/results/render_movie/latest/render_manifest.json`
  - `clearex/results/render_movie/latest/level_<nn>_frames/frame_000000.png`
  - `clearex/results/compile_movie/latest/*.mp4`
  - `clearex/results/compile_movie/latest/*.mov`
- Migration-only legacy layouts:
  - root `data`
  - root `data_pyramid`
  - `results/<analysis>/latest/data`
