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

## 2. Shell configuration (`.bashrc` and `bash_profile`)

The following shell configuration ensures that `uv` behaves correctly on the cluster and that Python interpreters, caches, and environments remain stable across nodes.

### Why this matters
On HPC systems:

- `~/.bashrc` is typically sourced only for interactive shells
- `~/.bash_profile` is sourced for **login shells**, GUI applications, and some scheduler contexts
- uv installs **Python runtimes separately from its cache**
- If the Python install directory is not explicitly set, uv may default to a node-local or temporary filesystem (e.g. /tmp), causing virtual environments to break when switching nodes

To avoid these issues, we explicitly configure **three uv locations** and ensure they are available in all shell contexts.

------------
Configuration snippet

You can copy and adapt the following snippet (paths may need adjustment depending on your username or project directory):

```bash
# .bashrc

# -----------------------------------------------------------------------------
# uv configuration
# -----------------------------------------------------------------------------
# Install uv under /project so it does not consume home quota.
export UV_INSTALL_DIR="/project/path/to/uv"

# uv cache (wheels, sdists, metadata). Must be on the same filesystem
# as Python environments (e.g. project-local .venv directories).
export UV_CACHE_DIR="${UV_INSTALL_DIR}/.cache"

# Location where uv installs Python runtimes (CPython, PyPy, etc.).
# This MUST be on a persistent, shared filesystem on the cluster.
export UV_PYTHON_INSTALL_DIR="${UV_INSTALL_DIR}/python"

# Make uv and uvx available on the command line
export PATH="$UV_INSTALL_DIR/bin:$PATH"

# Remainder of your .bashrc...
```

-------

### Notes and best practices

- `UV_INSTALL_DIR` - Controls where the uv executable and its internal metadata live.
- `UV_CACHE_DIR` - Controls where uv stores cached build artifacts (wheels, sdists, index metadata). This does not control where Python itself is installed.
- `UV_PYTHON_INSTALL_DIR` (**required on the cluster**) - Controls where uv installs Python runtimes. If unset, uv may fall back to $TMPDIR or other node-local storage, causing virtual environments to break when moving between nodes.
- `PATH` - Only needs to be modified once, so the shell can locate the `uv` executable.
The other variables are read internally by `uv` and do not affect command lookup.

---------

### Applying Changes
After updating your `.bashrc`, run `source ~/.bashrc` to apply the changes.

(You may also need to restart terminals or relaunch GUI applications such as PyCharm.)


----------

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
