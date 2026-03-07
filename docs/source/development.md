# Development Notes

## Documentation Workflow

1. Edit docs under `docs/source/`.
2. Keep API docstrings in NumPy style so `numpydoc` renders cleanly.
3. Build and validate docs with warnings as errors:

```bash
uv run python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

## API Docs Generation

The API reference pages use Sphinx `autosummary` + `autodoc` and are generated from modules in `src/clearex`.

- Add new modules to `docs/source/api/index.rst`.
- Keep type hints and docstrings up to date to get high quality rendered docs.

## Useful Commands

```bash
uv run ruff check src tests
uv run --with pytest --with requests python -m pytest -q
```
