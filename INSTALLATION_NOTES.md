# ClearEx Installation Notes (BioHPC / HPC environment)

This document describes how to set up a working ClearEx environment on a high-performance computing (HPC) system, using `uv` as the Python package and environment manager. It is written for users who are **not** full-time computational developers but need a reliable way to install and run ClearEx.

The examples below use the UT Southwestern BioHPC layout (with `/project` and `/archive` filesystems), but the same principles apply on other clusters.

---

## 1. Overview

ClearEx uses:

- **Python 3.12**
- **uv** as the package manager and virtual environment tool
- A number of scientific Python libraries (Dask, Zarr, SimpleITK, etc.)
- **PyPetaKit5D**, which installs and uses the **MATLAB Runtime** (~5 GB download)

Because of this, there are a few important rules:

1. The **uv cache** and your **Python virtual environment** must live on the **same filesystem** (e.g., both under `/project/...`) for best performance.
2. The cluster’s **HTTP/HTTPS proxy** must be available so that PyPetaKit5D can download the MATLAB Runtime.
3. SOCKS proxies (`ALL_PROXY`) should **not** be used during installation; they can interfere with tools like `uv` and PyCharm on GPU nodes.

Once the environment is configured, most users only need to run a few commands to install ClearEx and its dependencies.

---

## 2. Shell configuration (`.bashrc`)

The following `.bashrc` snippet sets up the basics:

- Ensures the system `bashrc` is loaded.
- Defines a `global_scratch` location.
- Configures `uv` so its **install location** and **cache** live under `/project`.
- Adds helper functions for activating uv, Conda, etc.

You can copy and adapt this snippet to your own `.bashrc` (paths may need to be changed if your username or project directory differ).

```bash
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# -----------------------------------------------------------------------------
# uv configuration
# -----------------------------------------------------------------------------
# Install uv under /project so it does not consume home quota.
export UV_INSTALL_DIR="/project/path/to/uv"

# uv cache must be on the same filesystem as your Python envs (.venv)
export UV_CACHE_DIR="${UV_INSTALL_DIR}/.cache"

# Put uv's bin directory on PATH
export PATH="$UV_INSTALL_DIR/bin:$PATH"

# Helper function: activate uv's own environment (if needed)
activate_uv() {
    . "/project/path/to/uv/env"
}

# Remainder of your .bashrc...
```

After updating your `.bashrc`, run `source ~/.bashrc` to apply the changes.

## 3. PyCharm configuration
If you use PyCharm for development, or Jupyter notebooks in general, you may need to:
- Disable SOCKS proxies.
- Relocate PyCharm caches/configs from quota-limited home directories to `/project` or '/archive`.'

Here is an example launch script that you can adapt:

```bash
#!/bin/bash

# If there is a GPU, load GPU-related modules
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    module load gpu_prepare
    echo "GPU Modules Loaded"
fi

# SOCKS proxies interfere with PyCharm and Jupyter Notebooks on GPU nodes, so clear them
unset ALL_PROXY
unset all_proxy

# Move JetBrains cache/config/data to /archive or /project to avoid home quota issues
export XDG_CACHE_HOME=/archive/path/to/the/.jetbrains/cache
export XDG_CONFIG_HOME=/archive/path/to/the/.jetbrains/config
export XDG_DATA_HOME=/archive/path/to/the/.jetbrains/data

export IDEA_USE_DEFAULT_KEYRING="true"
export SECRET_SERVICE_DISABLE=1

# Launch PyCharm
/project/path/to/pycharm-2024.3.1/bin/pycharm
```
