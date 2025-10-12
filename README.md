<h1 align="center"> ClearEx
<h2 align="center"> Scalable Analytics for Cleared and Expanded Tissue Imaging.
</h2>
</h1>

[![Tests](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml/badge.svg)](https://github.com/TheDeanLab/navigate/actions/workflows/push_checks.yaml)
[![codecov](https://codecov.io/gh/TheDeanLab/clearex/graph/badge.svg?token=ONldpMpLse)](https://codecov.io/gh/TheDeanLab/clearex)

**ClearEx** is an open source Python package for scalable analytics of cleared and expanded tissue imaging data. It relies heavily on next-generation file formats and cloud-based chunk computing to accelerate image analysis workflows, enabling tissue-scale computer vision and machine learning.

<details>

<summary>Installation with UV</summary>
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
uv sync

# Optional - Install development dependencies
uv sync --extra dev 

# Optional - Install all additional dependencies.
uv sync --all-extras
```

`uv` is compatible with existing `pip` or `conda` workflows, so you can continue
to use those tools if preferred.

</details>

<details>

<summary>Installation with pip</summary>

We recommend installing ClearEx in a dedicated Anaconda environment:

```bash
conda create -n clearex python=3.12
conda activate clearex

# Install core dependencies to circumvent BioHPC-specific issues.
conda install -c conda-forge pyarrow "numpy>=1.25" cython 
cd to/your/cloned/clearex/directory
pip install -e .
```

</details>

<details>

<summary>Installation with conda-forge</summary>

If you encounter compilation issues during installation, you can install ClearEx entirely from conda-forge:

```bash
cd to/your/cloned/clearex/directory
conda env create -f environment.yml
conda activate clearex
```

This installs all dependencies from conda-forge and the ClearEx package in editable mode. The environment is named `clearex` by default (specified in `environment.yml`).

</details>
