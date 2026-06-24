# Navigate Stage Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve stage-scan physical extent for Navigate oblique acquisitions by teaching `shear_transform` to distinguish stage travel metadata from reconstructed affine spacing.

**Architecture:** Persist a small Navigate stage-geometry payload during `experiment.yml` parsing/materialization, resolve it through shared store metadata helpers, then let `shear_transform` switch to a geometry-aware affine path that adjusts the transform-space Z spacing while preserving downstream output metadata lineage. Keep the existing generic shear path unchanged for stores without the payload.

**Tech Stack:** Python, Zarr, NumPy, ANTsPy, pytest

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/io/test_experiment.py`
- Modify: `tests/io/test_ome_store_scale.py`
- Modify: `tests/shear/test_pipeline.py`

- [ ] Add a Navigate parsing/materialization test that expects stage-geometry metadata on the parsed experiment, namespaced store metadata, and canonical source attrs.
- [ ] Add a resolver test that expects the shared metadata helper to recover the Navigate stage-geometry payload through source-component/store metadata lookup.
- [ ] Add a shear regression test that seeds a canonical source with `z_step_um=5.0`, `xy=1.0`, Navigate stage-geometry metadata, `shear_yz=1.0`, and `auto_rotate_from_shear=True`, then expects the transformed output Y extent to match the stage span rather than the inflated generic extent.
- [ ] Run the targeted tests and confirm they fail for the expected missing-payload / wrong-geometry reasons.

### Task 2: Persist and resolve Navigate stage geometry

**Files:**
- Modify: `src/clearex/io/experiment.py`
- Modify: `src/clearex/io/ome_store.py`
- Test: `tests/io/test_experiment.py`
- Test: `tests/io/test_ome_store_scale.py`

- [ ] Extend `NavigateExperiment` metadata serialization with a compact stage-geometry payload that records the source scan axis, stage-aligned output axis, scan step, and geometry mode.
- [ ] During store initialization/materialization, write that payload onto both `clearex/runtime_cache/source/data` attrs and `clearex/metadata`.
- [ ] Add a shared resolver in `ome_store.py` that finds the payload from component attrs first, then store metadata, then legacy nested Navigate metadata fallbacks.
- [ ] Re-run the new IO-focused tests and confirm they pass.

### Task 3: Apply geometry-aware shear logic

**Files:**
- Modify: `src/clearex/shear/pipeline.py`
- Test: `tests/shear/test_pipeline.py`

- [ ] Thread the resolved Navigate stage-geometry payload into affine construction.
- [ ] In the specialized path, compute the transform-space Z spacing from the configured affine so one source scan step maps to one output stage step along the declared stage-aligned axis; fall back to generic behavior when the payload is absent or the transform provides no usable coupling.
- [ ] Use the corrected transform-space voxel size for bounds calculation and resampling, and persist the effective output `voxel_size_um_zyx` plus geometry provenance on the shear output attrs.
- [ ] Re-run the shear tests and confirm the new regression passes without breaking existing generic-path tests.

### Task 4: Document and validate

**Files:**
- Modify: `src/clearex/io/README.md`
- Modify: `src/clearex/shear/README.md`

- [ ] Document that Navigate stage geometry is preserved separately from reconstructed affine spacing, and that `shear_transform` consumes the payload when present.
- [ ] Run targeted lint/tests for the touched files, then run the subsystem validation commands required by the IO and shear guides.
