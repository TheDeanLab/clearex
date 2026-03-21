# GUI Agent Notes

This folder owns the PyQt6 UX in `app.py`.

## Window Structure

- Setup window (`ClearExSetupDialog`):
  - Select file / `experiment.yml`
  - Configure Dask backend and Zarr save options
  - Configure per-experiment `Spatial Calibration` (world `z/y/x` to Navigate
    stage `X/Y/Z/F` or `none`)
  - Display image metadata
  - Materialize canonical store when missing, persist resolved spatial
    calibration to all requested stores, and show progress dialog
- Analysis window (`AnalysisSelectionDialog`):
  - `Analysis Scope` manages the active experiment/store context for single or
    batch analysis
  - Left: operation selection, execution order, and `Configure` buttons
  - Right: operation parameter panels + parameter-help panel

## Persistence Contract

- Treat per-dataset GUI persistence as part of the analysis-dialog contract,
  not an optional enhancement.
- `AnalysisSelectionDialog` must continue to:
  - restore dataset-local saved GUI state when available,
  - fall back to the latest completed provenance-backed workflow state,
  - persist current widget state on experiment switch, close, and run,
  - keep `Restore Latest Run Parameters` working for the active dataset.
- The active analysis workflow must always carry the selected target store's
  spatial calibration so visualization and future registration use the same
  placement metadata.
- When adding a new analysis workflow, new operation widget, or new parameter:
  - add the default and normalization path in `src/clearex/workflow.py`,
  - hydrate the widget from restored `analysis_parameters`,
  - collect the widget back into `analysis_parameters`,
  - ensure the value survives dataset-local GUI-state persistence and
    provenance-backed restore,
  - add/update targeted regression tests for restore/save behavior.
- Do not introduce GUI-only parameters that bypass this persistence path unless
  there is a deliberate documented reason.

## Analysis Dialog Invariants

- Operation parameters should **not** be always visible.
- Only one operation configuration panel is visible at a time.
- Clicking `Configure` on one operation hides the prior panel.
- Unselected operations should not be configurable.
- Per-operation `Input source` options depend on selected upstream operations and execution order.
- `Visualization` is treated as a terminal/view step; it should not be offered as an upstream image source for later operations.
- Visualization placement must come from persisted store metadata
  (`spatial_calibration`), not transient GUI-only state.
- Visualization configuration currently exposes:
  - `position_index` for multiposition datasets
  - multiscale loading toggle
  - particle-overlay toggle

## Theme Requirements

- Keep styling consistent with the dark theme.
- Keep controls/popups/content cards in the dark theme; avoid introducing new light surfaces there.
- Exception: top branding header cards intentionally use a light tint (`#f0f2ef`) to match the header image background.
- When a rounded dark card contains a `QScrollArea`, `QStackedWidget`, or other
  nested viewport/page widget, explicitly theme the scroll area, its viewport,
  and the inner content/page widgets together so Qt does not leak default light
  corners behind rounded borders.
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
- Manually verify combo/dropdown readability in the GUI.
