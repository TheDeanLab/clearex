<h1 align="center"> ClearEx
<h2 align="center"> Scalable Analytics for Cleared and Expanded Tissue Imaging.
</h2>
</h1>

[![Tests](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml/badge.svg)](https://github.com/TheDeanLab/navigate/actions/workflows/push_checks.yaml)
[![codecov](https://codecov.io/gh/TheDeanLab/clearex/graph/badge.svg?token=ONldpMpLse)](https://codecov.io/gh/TheDeanLab/clearex)

**ClearEx** is an open source Python package for scalable analytics of cleared and expanded tissue imaging data. It relies heavily on next-generation file formats and cloud-based chunk computing to accelerate image analysis workflows, enabling tissue-scale computer vision and machine learning.

## Installation with UV 
You can also manage dependencies using [uv](https://github.com/astral-sh/uv),
which provides faster installs and lockfile support. First, install `uv` via the
official script:

### Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Confirm that `uv` is installed
```bash
uv --version
```

### Install ClearEx

Then, in the root directory of the cloned ClearEx repository, run:

```bash
# Create a virtual environment in the project root.
uv venv

# Activate virtual environment
source .venv/bin/activate

# If you want to operate the package in a non-editable mode, run:
uv sync

# Install the ClearEx package in editable mode with all core dependencies.
uv pip install -e .

# Optional - Install development dependencies
uv pip install -e ".[dev]"

# Optional - Install the documentation's dependencies.
uv pip install -e ".[docs]"

# Or install everything at once:
uv pip install -e ".[dev,docs]"
```

## Installation in a High Performance Computing environment.
There are several considerations for installation in a high performance computing (HPC) environment. While these are not unique to ClearEx, we provide some guidelines here to help you get started. Please see the INSTALLATION_NOTES.md file for more details.