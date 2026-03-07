# GUI Agent Notes

This folder owns the PyQt6 UX in `app.py`.

## Window Structure

- Setup window (`ClearExSetupDialog`):
  - Select file / `experiment.yml`
  - Configure Dask backend and Zarr save options
  - Display image metadata
  - Materialize canonical store when missing, with progress dialog
- Analysis window (`AnalysisSelectionDialog`):
  - Left: operation selection, execution order, and `Configure` buttons
  - Right: operation parameter panels + parameter-help panel

## Analysis Dialog Invariants

- Operation parameters should **not** be always visible.
- Only one operation configuration panel is visible at a time.
- Clicking `Configure` on one operation hides the prior panel.
- Unselected operations should not be configurable.
- Per-operation `Input source` options depend on selected upstream operations and execution order.

## Theme Requirements

- Keep styling consistent with the dark theme.
- Do not introduce white backgrounds or black text in controls/popups.
- Explicitly style dropdown list views:
  - `QComboBox QAbstractItemView`
  - selected background and selected text colors
- Keep parameter cards, help panels, and popups visually aligned with title/label palette.

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
