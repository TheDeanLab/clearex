# ClearEx Agent Strategy

This file summarizes the current engineering strategy for agent-driven changes in this repository.

## Scope

- Primary entrypoint: `src/clearex/main.py`
- Shared runtime config/parsing: `src/clearex/workflow.py`
- GUI layer: `src/clearex/gui/app.py`
- CLI parser: `src/clearex/io/cli.py`
- Tests for strategy-critical behavior:
  - `tests/test_workflow.py`
  - `tests/io/test_cli.py`

## Execution Model

- Default mode is GUI-first.
- Headless operation must remain first-class and easy to run.
- CLI controls:
  - `--gui` / `--no-gui`
  - `--headless` (overrides GUI launch)
  - `--file`, `--dask`, `--chunks`
  - `--deconvolution`, `--particle-detection`, `--registration`, `--visualization`
- GUI launch failures (missing display, missing Qt bindings, runtime issues) should gracefully fall back to headless mode with clear logs.

## `workflow.py` Contract

- `WorkflowConfig` is the shared schema used by both GUI and headless paths.
- `parse_chunks` is the single source of truth for validating chunk specs.
- `format_chunks` is the single source of truth for rendering chunk specs back into UI text.
- Keep parsing/formatting logic centralized here to avoid drift between CLI and GUI behavior.

## GUI Strategy

- GUI is implemented in `src/clearex/gui/app.py`.
- UI responsibilities:
  - data path selection (file/folder),
  - metadata loading via `ImageOpener`,
  - metadata display (path, shape, dtype, axes, channels, positions, image size, time points, pixel size, metadata keys),
  - workflow selection toggles.
- Metadata panel should stay operator-friendly and compact. Current design uses a 2-column key/value layout.
- Visual direction: clean modern dark theme.
- GUI layer should only collect/validate user intent and return a `WorkflowConfig`; core execution remains in `main.py`.

## Analysis Hooks

- Current selectable operations:
  - deconvolution,
  - particle detection,
  - registration,
  - visualization.
- Registration and visualization have runtime hooks.
- Deconvolution and particle detection are currently scaffolded as hooks with logging placeholders.

## Documentation and Typing Standards

For all new or changed public/internal functions in this workflow stack:

- Use explicit type hints for parameters and return values.
- Use numpydoc docstrings with, at minimum:
  - `Parameters`
  - `Returns`
  - `Raises` (or explicit note when exceptions are handled internally)
- Include `Notes` when behavior has fallback logic, side effects, or environment constraints.

## Quality Gate

Before finishing changes in this area, run:

- `ruff check src/clearex/main.py src/clearex/workflow.py src/clearex/gui/app.py src/clearex/io/cli.py`
- `pytest -q tests/test_workflow.py tests/io/test_cli.py`

If a change touches metadata parsing or UI rendering logic, add/adjust targeted tests accordingly.

## Dependency Guidance

- GUI implementation targets `PyQt6`.
- Avoid introducing `PyQt5` requirements for the GUI layer.
- If dependency resolution breaks on macOS arm64, verify lockfile resolution for Qt-related transitive dependencies before changing runtime code.

## Future Roadmap (Not Implemented Yet)

- Provenance capture: persist exactly which workflow operations were run and with what parameters.
- Replay/loading: ability to restore a prior workflow configuration automatically.
- Keep this future capability in mind when evolving `WorkflowConfig` and execution wiring.
