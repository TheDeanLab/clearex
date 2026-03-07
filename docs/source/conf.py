"""Sphinx configuration for ClearEx documentation."""

from __future__ import annotations

# Standard Library Imports
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_ants_stub() -> None:
    """Install a lightweight ``ants`` stub for docs imports.

    Returns
    -------
    None
        This helper mutates ``sys.modules`` in place.
    """

    ants_module = types.ModuleType("ants")
    ants_core_module = types.ModuleType("ants.core")
    ants_image_module = types.ModuleType("ants.core.ants_image")
    ants_transform_module = types.ModuleType("ants.core.ants_transform")

    class ANTsImage:  # noqa: N801
        """Docs-only placeholder for ANTs image objects."""

    class ANTsTransform:  # noqa: N801
        """Docs-only placeholder for ANTs transform objects."""

    ants_image_module.ANTsImage = ANTsImage
    ants_transform_module.ANTsTransform = ANTsTransform
    ants_core_module.ants_image = ants_image_module
    ants_core_module.ants_transform = ants_transform_module

    ants_module.__version__ = "0.0.0"
    ants_module.ANTsImage = ANTsImage
    ants_module.ANTsTransform = ANTsTransform
    ants_module.core = ants_core_module

    sys.modules["ants"] = ants_module
    sys.modules["ants.core"] = ants_core_module
    sys.modules["ants.core.ants_image"] = ants_image_module
    sys.modules["ants.core.ants_transform"] = ants_transform_module


_install_ants_stub()


project = "ClearEx"
author = "The Dean Lab, UT Southwestern Medical Center"

try:
    release = package_version("clearex")
except PackageNotFoundError:
    release = "0.1.1"
version = ".".join(release.split(".")[:3])


extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "exclude-members": "ndarray,cKDTree,ks_2samp,wasserstein_distance",
    "show-inheritance": True,
    "undoc-members": False,
}

autodoc_mock_imports = [
    "PyPetaKit5D",
    "ants",
    "antspyx",
    "cv2",
    "napari",
    "neuroglancer",
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtWidgets",
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

myst_heading_anchors = 3
myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "napari": ("https://napari.org/stable/", None),
}

todo_include_todos = False

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release} Documentation"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
}
