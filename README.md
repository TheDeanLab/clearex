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
  - visualization metadata (`results/visualization/latest`) with napari launch
  - registration workflow hook (currently initialized from CLI/GUI, but latest-output persistence is not yet wired like other analyses)
- FAIR-style provenance records persisted in Zarr/N5 (`provenance/runs`) with append-only run history and hash chaining.

## Installation

### Requirements
- Python `>=3.12`
- macOS/Linux/Windows supported by Python stack
- For GUI usage: display server + `PyQt6` (installed in base dependencies)

### Recommended: install with `uv`
Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Then install ClearEx from this repository root:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install ClearEx (editable install)
uv pip install -e .

# Optional extras
uv pip install -e ".[decon]"      # deconvolution stack (PyPetaKit5D + PSFmodels)
uv pip install -e ".[dev]"        # tests/lint/dev tools
uv pip install -e ".[docs]"       # docs build stack
uv pip install -e ".[dev,docs,decon]"
```

Alternative repo-aware install with lockfile:

```bash
uv sync
```

### Install with pip (alternative)
```bash
python -m venv .venv
source .venv/bin/activate
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
usage: clearex [-h] [--deconvolution] [--particle-detection] [-r] [-v]
               [-f FILE] [--dask | --no-dask] [--chunks CHUNKS]
               [--gui | --no-gui] [--headless]
```

### Options
- `-f, --file`: Path to input image/store or Navigate `experiment.yml`.
- `--deconvolution`: Run deconvolution workflow.
- `--particle-detection`: Run particle detection workflow.
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
  --deconvolution --particle-detection
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
- Deconvolution, particle detection, and visualization operations run against canonical Zarr/N5 stores.
- GUI setup currently requires Navigate `experiment.yml`/`experiment.yaml`.
- If GUI cannot launch (for example no display), ClearEx logs a warning and falls back to headless execution.

## Output Layout (Canonical Store)
- Base image data: `data`
- Multiscale pyramid levels: `data_pyramid/level_*`
- Latest analysis outputs:
  - `results/deconvolution/latest`
  - `results/particle_detection/latest`
  - `results/visualization/latest`
- Provenance:
  - run records: `provenance/runs/<run_id>`
  - latest output pointers: `provenance/latest_outputs/<analysis>`
