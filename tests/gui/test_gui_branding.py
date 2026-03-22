#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

from __future__ import annotations

# Standard Library Imports
from pathlib import Path

# Local Imports
import clearex.gui.app as app_module


def test_resolve_gui_asset_paths_exist() -> None:
    splash_path = app_module._resolve_gui_asset_path(app_module._GUI_SPLASH_IMAGE)
    header_path = app_module._resolve_gui_asset_path(app_module._GUI_HEADER_IMAGE)

    assert splash_path.is_file()
    assert header_path.is_file()
    assert splash_path.exists()
    assert header_path.exists()
    assert splash_path.suffix.lower() == ".png"
    assert header_path.suffix.lower() == ".png"
    assert splash_path.stat().st_size > 0
    assert header_path.stat().st_size > 0
    assert splash_path.parent == header_path.parent


def test_resolve_gui_asset_path_returns_absolute_path() -> None:
    resolved = app_module._resolve_gui_asset_path("ClearEx_full_header.png")

    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
