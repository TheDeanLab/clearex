#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

"""Shared GUI layout spacing presets and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

Margins = Tuple[int, int, int, int]


@dataclass(frozen=True)
class LayoutSpacingSpec:
    """Describe margins and spacing values for one Qt layout.

    Parameters
    ----------
    margins : tuple[int, int, int, int]
        Layout contents margins as ``(left, top, right, bottom)``.
    spacing : int, optional
        Uniform item spacing applied via ``setSpacing`` when available.
    horizontal_spacing : int, optional
        Horizontal spacing for form/grid layouts.
    vertical_spacing : int, optional
        Vertical spacing for form/grid layouts.
    """

    margins: Margins
    spacing: Optional[int] = None
    horizontal_spacing: Optional[int] = None
    vertical_spacing: Optional[int] = None


_WINDOW_ROOT_SPEC = LayoutSpacingSpec(margins=(24, 24, 24, 24), spacing=18)
_POPUP_ROOT_SPEC = LayoutSpacingSpec(margins=(20, 20, 20, 20), spacing=14)
_STACK_SPEC = LayoutSpacingSpec(margins=(0, 0, 0, 0), spacing=12)
_ROW_SPEC = LayoutSpacingSpec(margins=(0, 0, 0, 0), spacing=12)
_COMPACT_ROW_SPEC = LayoutSpacingSpec(margins=(0, 0, 0, 0), spacing=10)
_FOOTER_ROW_SPEC = LayoutSpacingSpec(margins=(0, 8, 0, 0), spacing=12)
_GRID_SPEC = LayoutSpacingSpec(
    margins=(8, 8, 8, 8),
    horizontal_spacing=20,
    vertical_spacing=14,
)
_METADATA_GRID_SPEC = LayoutSpacingSpec(
    margins=(10, 10, 10, 10),
    horizontal_spacing=24,
    vertical_spacing=14,
)
_FORM_SPEC = LayoutSpacingSpec(
    margins=(0, 0, 0, 0),
    horizontal_spacing=14,
    vertical_spacing=12,
)
_HELP_STACK_SPEC = LayoutSpacingSpec(margins=(0, 0, 0, 0), spacing=8)


def _apply_layout_spec(layout: Any, spec: LayoutSpacingSpec) -> None:
    """Apply a layout spacing specification to a Qt layout object.

    Parameters
    ----------
    layout : Any
        Qt layout-like object that supports spacing and margin setters.
    spec : LayoutSpacingSpec
        Target margin and spacing values to apply.

    Returns
    -------
    None
        Layout geometry is updated in-place.
    """

    left, top, right, bottom = spec.margins
    if hasattr(layout, "setContentsMargins"):
        layout.setContentsMargins(left, top, right, bottom)
    if spec.spacing is not None and hasattr(layout, "setSpacing"):
        layout.setSpacing(int(spec.spacing))
    if spec.horizontal_spacing is not None and hasattr(layout, "setHorizontalSpacing"):
        layout.setHorizontalSpacing(int(spec.horizontal_spacing))
    if spec.vertical_spacing is not None and hasattr(layout, "setVerticalSpacing"):
        layout.setVerticalSpacing(int(spec.vertical_spacing))


def apply_window_root_spacing(layout: Any) -> None:
    """Apply top-level spacing preset for main windows/dialogs.

    Parameters
    ----------
    layout : Any
        Root layout for a primary window dialog.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _WINDOW_ROOT_SPEC)


def apply_popup_root_spacing(layout: Any) -> None:
    """Apply root spacing preset for configuration popup dialogs.

    Parameters
    ----------
    layout : Any
        Root layout for a popup dialog.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _POPUP_ROOT_SPEC)


def apply_stack_spacing(layout: Any) -> None:
    """Apply spacing preset for vertical/horizontal content stacks.

    Parameters
    ----------
    layout : Any
        Stack layout that groups related widgets.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _STACK_SPEC)


def apply_row_spacing(layout: Any) -> None:
    """Apply spacing preset for standard row layouts.

    Parameters
    ----------
    layout : Any
        Row layout containing horizontally arranged widgets.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _ROW_SPEC)


def apply_compact_row_spacing(layout: Any) -> None:
    """Apply compact row spacing preset for dense controls.

    Parameters
    ----------
    layout : Any
        Row layout containing compact controls.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _COMPACT_ROW_SPEC)


def apply_footer_row_spacing(layout: Any) -> None:
    """Apply spacing preset for footer/action button rows.

    Parameters
    ----------
    layout : Any
        Footer row layout containing status text and actions.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _FOOTER_ROW_SPEC)


def apply_dialog_grid_spacing(layout: Any) -> None:
    """Apply grid spacing preset for popup/configuration grids.

    Parameters
    ----------
    layout : Any
        Grid layout for form-like popup rows.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _GRID_SPEC)


def apply_metadata_grid_spacing(layout: Any) -> None:
    """Apply spacing preset for metadata key/value grid layouts.

    Parameters
    ----------
    layout : Any
        Grid layout used for metadata display sections.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _METADATA_GRID_SPEC)


def apply_form_spacing(layout: Any) -> None:
    """Apply spacing preset for ``QFormLayout`` controls.

    Parameters
    ----------
    layout : Any
        Form layout containing label/field rows.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _FORM_SPEC)


def apply_help_stack_spacing(layout: Any) -> None:
    """Apply compact spacing preset for inline help cards.

    Parameters
    ----------
    layout : Any
        Help-card stack layout.

    Returns
    -------
    None
        Layout spacing is updated in-place.
    """

    _apply_layout_spec(layout, _HELP_STACK_SPEC)
