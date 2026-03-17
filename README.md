<h1 align="center">ClearEx</h1>
<h2 align="center">Scalable Analytics for Cleared and Expanded Tissue Imaging.</h2>

[![Tests](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml/badge.svg)](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/TheDeanLab/clearex/graph/badge.svg?token=ONldpMpLse)](https://codecov.io/gh/TheDeanLab/clearex)

ClearEx is an open source Python package for scalable analytics of cleared and expanded tissue imaging data.

## Current Functionality
- GUI-first entrypoint with headless fallback.
- Headless CLI for scripted runs.
- Input support for TIFF/OME-TIFF, Zarr/N5, HDF5 (`.h5/.hdf5/.hdf`), and NumPy (`.npy/.npz`).
- Navigate experiment ingestion from `experiment.yml` / `experiment.yaml`.
- Canonical analysis store layout with axis order `(t, p, c, z, y, x)`.
- Analysis operations available from the main entrypoint:
  - deconvolution (`results/deconvolution/latest/data`)
  - particle detection (`results/particle_detection/latest`)
  - uSegment3D segmentation (`results/usegment3d/latest/data`)
  - visualization metadata (`results/visualization/latest`) with napari launch
  - registration workflow hook (currently initialized from CLI/GUI, but latest-output persistence is not yet wired like other analyses)
- FAIR-style provenance records persisted in Zarr/N5 (`provenance/runs`) with append-only run history and hash chaining.

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
uv pip install -e ".[decon]"      # deconvolution stack (PyPetaKit5D + PSFmodels)
uv pip install -e ".[usegment3d]" # uSegment3D segmentation stack
uv pip install -e ".[dev]"        # tests/lint/dev tools
uv pip install -e ".[docs]"       # docs build stack
uv pip install -e ".[dev,docs,decon,usegment3d]"
```

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
               [--shear-transform] [-r] [-v] [--mip-export]
               [-f FILE] [--dask | --no-dask] [--chunks CHUNKS]
               [--gui | --no-gui] [--headless]
```

### Options
- `-f, --file`: Path to input image/store or Navigate `experiment.yml`.
- `--deconvolution`: Run deconvolution workflow.
- `--particle-detection`: Run particle detection workflow.
- `--usegment3d`: Run uSegment3D segmentation workflow.
- `--channel-indices`: uSegment3D channels to process (`0,1,2` or `all`).
- `--input-resolution-level`: uSegment3D input pyramid level (`0`, `1`, ...).
- `-r, --registration`: Run registration workflow hook.
- `-v, --visualization`: Run visualization workflow.
- `--dask / --no-dask`: Enable/disable Dask-backed reading.
- `--chunks`: Chunk spec for Dask reads, for example `256` or `1,256,256`.
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

Run headless uSegment3D on all channels:

```bash
clearex --headless \
  --file /path/to/experiment.yml \
  --usegment3d \
  --channel-indices all \
  --input-resolution-level 1
```

Run headless particle detection on an existing canonical Zarr store:

```bash
clearex --headless \
  --file /path/to/data_store.zarr \
  --particle-detection
```

Disable Dask lazy loading:

```bash
clearex --headless --no-dask --file /path/to/data_store.zarr --particle-detection
```

## Runtime Behavior Notes
- If `--file` points to Navigate `experiment.yml`, ClearEx resolves acquisition data and materializes a canonical store first.
- For non-Zarr/N5 acquisition data, materialization target is `data_store.zarr` beside `experiment.yml`.
- For Zarr/N5 acquisition data, ClearEx reuses the source store path in place.
- Deconvolution, particle detection, uSegment3D, and visualization operations run against canonical Zarr/N5 stores.
- Visualization supports multi-volume overlays (for example raw `data` + `results/usegment3d/latest/data`) with per-layer image/labels display controls.
- Visualization now probes napari OpenGL renderer info (`vendor`/`renderer`/`version`) and can fail fast when software rendering is detected or GPU rendering cannot be confirmed (`require_gpu_rendering=True`).
- MIP export writes TIFF outputs as OME-TIFF (`.tif`) with projection-aware physical pixel calibration (`PhysicalSizeX/Y`) derived from source `voxel_size_um_zyx`.
- uSegment3D runs per `(t, p, selected channel)` volume task and writes labels to `results/usegment3d/latest/data`.
  - GUI channel checkboxes now support selecting multiple channels in one run (`channel_indices`).
  - With GPU-aware `LocalCluster`, separate channel tasks can execute concurrently across GPUs.
  - `input_resolution_level` lets segmentation run on a selected pyramid level (for example `data_pyramid/level_1`).
  - `output_reference_space=level0` upsamples labels back to original resolution.
  - `save_native_labels=True` (when upsampling) also writes native-resolution labels to `results/usegment3d/latest/data_native`.
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
    - `auto_build`: generate a visualization pyramid cache for that layer when missing.
    - `off`: force single-scale rendering for that layer.
  - Auto-generated visualization pyramids are written under `results/visualization_cache/pyramids/...` and reused on subsequent runs.
  - Chunk-wise `map_overlap` stitching across labels is not yet enabled by default because label continuity/relabeling across chunk seams requires additional global reconciliation.
- GUI setup currently requires Navigate `experiment.yml`/`experiment.yaml`.
- If GUI cannot launch (for example no display), ClearEx logs a warning and falls back to headless execution.

### Visualization Keyframes and Manifest
- Keyframe capture is enabled by default during visualization (`capture_keyframes=True`).
- In napari:
  - Press `K` to capture a keyframe.
  - Press `Shift-K` to remove the most recent keyframe.
- Default keyframe manifest path is `<analysis_store>.visualization_keyframes.json` (override with `keyframe_manifest_path`).
- Each keyframe stores reproducible viewer state for movie reconstruction, including:
  - camera (angles, zoom, center, perspective),
  - dims (`current_step`, axis labels, order, and 2D/3D display mode),
  - layer order and selected/active layers,
  - per-layer display state (visibility, LUT/colormap, rendering mode, blending, opacity, contrast, and transforms when available).
- The GUI exposes a popup table (`Layer/View Table...`) to define optional per-layer overrides and `Annotation` labels that are embedded in the keyframe manifest.
- The GUI also exposes a popup table (`Volume Layers...`) to configure overlay rows with:
  - component path,
  - layer type (`image` or `labels`),
  - channels, visibility, opacity, blending, colormap, rendering, and multiscale policy.
- Visualization parameters include `require_gpu_rendering` (enabled by default). Disable only when running intentionally without a GPU-backed OpenGL context.

## Output Layout (Canonical Store)
- Base image data: `data`
- Multiscale pyramid levels: `data_pyramid/level_*`
- Latest analysis outputs:
  - `results/deconvolution/latest`
  - `results/particle_detection/latest`
  - `results/usegment3d/latest`
    - optional native-level labels: `results/usegment3d/latest/data_native`
  - `results/visualization/latest`
- Provenance:
  - run records: `provenance/runs/<run_id>`
  - latest output pointers: `provenance/latest_outputs/<analysis>`
