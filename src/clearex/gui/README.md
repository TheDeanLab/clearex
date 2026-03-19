# GUI Agent Notes

This folder owns the PyQt6 UX in `app.py`.

## Window Structure

- Setup window (`ClearExSetupDialog`):
  - Manage an experiment list from:
    - `Load Experiment` for one `experiment.yml`,
    - `Create Experiment List` for recursive folder scans or saved-list reloads,
    - drag and drop of experiments, folders, or `.clearex-experiment-list.json` files
  - Add/remove experiment entries from the list and persist the list for reuse
  - Auto-load metadata when the current list selection changes
  - Configure Dask backend and Zarr save options
  - Display image metadata
  - On `Next`, batch-materialize missing canonical stores for every listed
    experiment, then continue with the currently selected experiment
- Analysis window (`AnalysisSelectionDialog`):
  - Left: operation selection, execution order, and `Configure` buttons
  - Right: operation parameter panels + parameter-help panel

## Analysis Dialog Invariants

- Operation parameters should **not** be always visible.
- Only one operation configuration panel is visible at a time.
- Clicking `Configure` on one operation hides the prior panel.
- Unselected operations should not be configurable.
- Per-operation `Input source` options depend on selected upstream operations and execution order.
- `Visualization` is treated as a terminal/view step; it should not be offered as an upstream image source for later operations.
- Visualization configuration currently exposes:
  - `position_index` for multiposition datasets
  - multiscale loading toggle
  - particle-overlay toggle
- Segmentation tab currently includes:
  - `Particle Detection`
  - `uSegment3D` (resolution-level selection, output-space/upscale controls, GPU toggle, view selection, Cellpose model settings, aggregation/tiling, and postprocess controls)

## Theme Requirements

- Keep styling consistent with the dark theme.
- Keep controls/popups/content cards in the dark theme; avoid introducing new light surfaces there.
- Exception: top branding header cards intentionally use a light tint (`#f0f2ef`) to match the header image background.
- Explicitly style setup-list context menus and any other new pop-up menus; do not fall back to platform-default light menus.
- Explicitly style dropdown list views:
  - `QComboBox QAbstractItemView`
  - selected background and selected text colors
- Keep parameter cards, help panels, and popups visually aligned with title/label palette.

## Branding Assets and Layout

- Branding assets live in `src/clearex/gui/`:
  - `ClearEx_full.png` (startup splash)
  - `ClearEx_full_header.png` (header banner)
- Use the shared branding helpers in `app.py`:
  - `_load_branding_pixmap(...)` (QImageReader-based load path for very large source files)
  - `_create_scaled_branding_label(...)`
  - `_show_startup_splash(...)`
- Setup dialog header is image-only (remove duplicate `"ClearEx"` / `"Experiment Setup"` text labels).
- Analysis dialog title/subtitle lives in the left column above the `Data store` row (not in the top header card).

## Spacing System

- Use `src/clearex/gui/spacing.py` as the single source of truth for layout spacing/margins.
- Do not add ad hoc `setSpacing`, `setHorizontalSpacing`, or `setVerticalSpacing` calls in dialog/window code.
- Apply helper presets by layout role:
  - root windows/popups: `apply_window_root_spacing` / `apply_popup_root_spacing`
  - stacked sections: `apply_stack_spacing`
  - normal rows: `apply_row_spacing`
  - compact control rows: `apply_compact_row_spacing`
  - action/status footers: `apply_footer_row_spacing`
  - config grids and metadata grids: `apply_dialog_grid_spacing` / `apply_metadata_grid_spacing`
  - forms and help cards: `apply_form_spacing` / `apply_help_stack_spacing`
- If a new layout role is needed, add a typed preset/helper in `spacing.py` instead of embedding numeric spacing in `app.py`.

## Parameter Help UX

- Verbose help text is shown in-panel, not as disruptive popups.
- New parameters should register:
  - tooltip text,
  - hover/focus event wiring,
  - help text mapping entry.

## Adding a New Operation Parameter

1. Add default/normalization in `src/clearex/workflow.py`.
2. Add widgets to the operation panel in `AnalysisSelectionDialog`.
3. Register parameter help text.
4. Hydrate from normalized config.
5. Collect values into `analysis_parameters` on Run.

## Validation

- `uv run ruff check src/clearex/gui/app.py`
- `uv run --with pytest --with requests python -m pytest -q tests/test_workflow.py`
- Manually verify combo/dropdown readability and experiment-list/menu styling in the GUI.
- Capture an updated setup-dialog screenshot when the front-panel layout changes materially.
