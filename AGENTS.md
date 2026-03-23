# ClearEx Repository Agent Guide

This file defines repository-wide guidance only. Package runtime strategy and
workflow behavior for ClearEx lives in `src/clearex/AGENTS.md`.

## Scope

- Applies to the entire repository.
- Use this file for repo layout, documentation hierarchy, validation
  expectations, and change coordination.
- Use `src/clearex/AGENTS.md` for package-wide runtime, workflow, and
  provenance policy.
- Use `src/clearex/<subsystem>/README.md` for subsystem-specific behavior and
  implementation notes.

## Documentation Hierarchy

- `/AGENTS.md`: repository-wide conventions only.
- `src/clearex/AGENTS.md`: authoritative package-level guide for the
  application/runtime surface.
- `src/clearex/<subsystem>/README.md`: authoritative subsystem guides.
- If instructions conflict, the more specific file wins.
- When behavior changes, update the most specific relevant documentation in the
  same change set.

## Naming Policy

- Use `AGENTS.md` for agent-facing guidance at repository or package
  boundaries.
- Use `README.md` for detailed subsystem guidance inside `src/clearex/*`.
- Do not introduce new `CODEX.md` files.
- If an older `CODEX.md` is encountered, migrate any unique guidance into the
  corresponding `AGENTS.md` or `README.md` and remove the redundant file.

## Repository Layout

- `src/clearex/`: application code and package-level runtime logic.
- `tests/`: automated tests; keep coverage aligned with behavior changes.
- Package and subsystem docs are co-located with the code they govern.

## Change Expectations

- Keep code, tests, and documentation in sync.
- Prefer targeted tests and targeted documentation updates over broad unrelated
  churn.
- Keep changes scoped to the area being modified unless a cross-cutting update
  is necessary.
- Avoid duplicating subsystem strategy in this root file; link or defer to the
  more specific package/subsystem docs instead.

## Canonical Store Policy

- Treat OME-Zarr v3 (``*.ome.zarr``) as the only canonical ClearEx store
  format in new code, tests, examples, and docs.
- Treat legacy ClearEx root-``data`` / root-``data_pyramid`` layouts in
  ``.zarr`` / ``.n5`` as migration-only inputs. Do not describe them as the
  preferred runtime contract and do not introduce new fixtures or examples that
  rely on them as canonical.
- Public microscopy-facing image data must use the OME-Zarr HCS contract.
  ClearEx-owned execution caches, provenance, GUI state, and non-image
  artifacts belong under the namespaced ``clearex/`` tree.
- Storage-contract changes must update the repository ``README.md``,
  ``src/clearex/AGENTS.md``, and the affected ``docs/source/runtime/*.rst`` /
  subsystem ``README.md`` files in the same change set.

## Validation

- Run linting and tests that match the files and behavior you changed.
- For changes under `src/clearex`, follow the validation steps in
  `src/clearex/AGENTS.md` and any touched subsystem `README.md`.
- If a change spans multiple subsystems, run the union of the relevant checks.

## Dependency Guidance

- Avoid unnecessary dependency additions or lockfile churn.
- When dependency changes are required, document the reason and any platform
  constraints in the same change set.
