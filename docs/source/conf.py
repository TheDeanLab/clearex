"""Sphinx configuration for ClearEx documentation."""

from __future__ import annotations

# Standard Library Imports
from importlib.metadata import PackageNotFoundError, version as package_version
import datetime
import os
import sys
import types


sys.path.insert(0, os.path.abspath("../../src"))


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


year = datetime.datetime.now().year
project = "ClearEx"
copyright = f"{year}, Dean Lab, UT Southwestern Medical Center"
author = "Dean Lab, UT Southwestern Medical Center"

try:
    release = package_version("clearex")
except PackageNotFoundError:
    release = "0.1.1"
version = ".".join(release.split(".")[:3])


extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx_toolbox.collapse",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_issues",
    "sphinx_design",
]

autosectionlabel_prefix_document = True

# The suffix of source filenames.
source_suffix = ".rst"

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
napoleon_use_ivar = True

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

todo_include_todos = False
issues_github_path = "TheDeanLab/clearex"

html_theme = "sphinx_rtd_theme"
html_logo = "_static/clearex_header_logo.png"
html_title = f"{project} {release} Documentation"
html_static_path = ["_static"]
html_show_sphinx = False
