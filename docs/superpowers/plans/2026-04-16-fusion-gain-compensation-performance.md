# Fusion Gain Compensation Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gain_compensated_feather` practical on large shared-filesystem HPC fusion runs by adding bounded `1x/2x/4x` intensity-fit estimation, sparse full-resolution validation, small rerun-safe edge-fit metadata reuse, and granular prepass progress.

**Architecture:** Keep the public fusion image contract unchanged and keep final fused chunk rendering at full resolution. Add new fusion policy parameters in `workflow.py` and the GUI, implement scale-aware estimation plus validation inside `src/clearex/registration/pipeline.py`, and persist a small `intensity_fit_edges` metadata group under the fusion auxiliary tree so reruns on unchanged registration geometry can skip the expensive edge-fit prepass.

**Tech Stack:** Python 3.12, Zarr v3, NumPy, SciPy, Dask, PyQt6, pytest, and repository docs under `src/clearex/registration/README.md`.

---

## File Map

- Modify: `src/clearex/workflow.py`
  - Add normalized fusion parameters for intensity-estimation scale selection and validation budgets.
  - Extend fusion defaults so GUI, provenance, and headless config all share the same policy surface.
- Modify: `src/clearex/gui/app.py`
  - Add fusion UI hints, widgets, widget state hydration, enable/disable rules, and collection for the new gain-compensation policy parameters.
- Modify: `src/clearex/registration/pipeline.py`
  - Add bounded estimation-scale selection helpers, sparse validation helpers, metadata cache read/write helpers, and granular prepass progress.
  - Extend `_estimate_pairwise_overlap_intensity(...)` and `run_fusion_analysis(...)` to use the new policy and reuse path.
- Modify: `tests/test_workflow.py`
  - Cover new defaults, normalization, and invalid-value rejection for fusion intensity-estimation parameters.
- Modify: `tests/gui/test_gui_execution.py`
  - Verify the analysis dialog round-trips the new fusion parameters.
- Modify: `tests/registration/test_pipeline.py`
  - Add unit coverage for scale selection and validation.
  - Add fusion integration coverage for scale-aware estimation, rerun reuse, invalidation, and granular progress.
- Modify: `src/clearex/registration/README.md`
  - Document the new gain-compensation policy knobs, rerun reuse behavior, and the fact that coarse estimation affects only the intensity-fit prepass.

## Task 1: Add Fusion Policy Defaults and Workflow Normalization

**Files:**
- Modify: `tests/test_workflow.py:203-213`
- Modify: `tests/test_workflow.py:781-840`
- Modify: `src/clearex/workflow.py:674-686`
- Modify: `src/clearex/workflow.py:1983-2041`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing workflow tests**

Add one default-coverage assertion block to `TestWorkflowConfig.test_default_zarr_save_config(...)` and two new normalization tests immediately after `test_normalizes_registration_parameters(...)`.

```python
assert cfg.analysis_parameters["fusion"]["intensity_estimation_scale_mode"] == "auto"
assert cfg.analysis_parameters["fusion"]["max_intensity_fit_voxels"] == 8_000_000
assert cfg.analysis_parameters["fusion"]["full_res_validation_samples"] == 50_000
assert cfg.analysis_parameters["fusion"][
    "max_intensity_validation_rmse_fraction"
] == pytest.approx(0.05)
assert cfg.analysis_parameters["fusion"][
    "min_intensity_validation_correlation"
] == pytest.approx(0.95)


def test_normalizes_fusion_intensity_estimation_policy(self):
    cfg = WorkflowConfig(
        analysis_parameters={
            "fusion": {
                "blend_mode": "gain_compensated_feather",
                "intensity_estimation_scale_mode": "DOWNsample_4x",
                "max_intensity_fit_voxels": "9000000",
                "full_res_validation_samples": "60000",
                "max_intensity_validation_rmse_fraction": "0.08",
                "min_intensity_validation_correlation": "0.91",
            }
        }
    )

    fusion = cfg.analysis_parameters["fusion"]
    assert fusion["intensity_estimation_scale_mode"] == "downsample_4x"
    assert fusion["max_intensity_fit_voxels"] == 9_000_000
    assert fusion["full_res_validation_samples"] == 60_000
    assert fusion["max_intensity_validation_rmse_fraction"] == pytest.approx(0.08)
    assert fusion["min_intensity_validation_correlation"] == pytest.approx(0.91)


def test_rejects_invalid_fusion_intensity_estimation_policy(self):
    with pytest.raises(
        ValueError, match="fusion intensity_estimation_scale_mode must be one of"
    ):
        WorkflowConfig(
            analysis_parameters={
                "fusion": {"intensity_estimation_scale_mode": "downsample_8x"}
            }
        )
```

- [ ] **Step 2: Run the workflow tests to verify they fail**

Run: `uv run pytest -q tests/test_workflow.py -k "fusion_intensity_estimation_policy or default_zarr_save_config"`

Expected: FAIL because the fusion defaults and `_normalize_fusion_parameters(...)` do not yet define `intensity_estimation_scale_mode`, `max_intensity_fit_voxels`, `full_res_validation_samples`, `max_intensity_validation_rmse_fraction`, or `min_intensity_validation_correlation`.

- [ ] **Step 3: Add the new fusion defaults and normalization**

Extend the fusion defaults in `src/clearex/workflow.py` and normalize the new values inside `_normalize_fusion_parameters(...)`.

```python
"fusion": {
    "execution_order": 5,
    "input_source": "registration",
    "force_rerun": False,
    "chunk_basis": "3d",
    "detect_2d_per_slice": False,
    "use_map_overlap": True,
    "blend_overlap_zyx": [8, 32, 32],
    "memory_overhead_factor": 2.5,
    "blend_mode": "feather",
    "blend_exponent": 1.0,
    "gain_clip_range": [0.25, 4.0],
    "intensity_estimation_scale_mode": "auto",
    "max_intensity_fit_voxels": 8_000_000,
    "full_res_validation_samples": 50_000,
    "max_intensity_validation_rmse_fraction": 0.05,
    "min_intensity_validation_correlation": 0.95,
},
```

```python
    intensity_scale_mode = (
        str(normalized.get("intensity_estimation_scale_mode", "auto")).strip().lower()
        or "auto"
    )
    if intensity_scale_mode not in {"auto", "full_res", "downsample_2x", "downsample_4x"}:
        raise ValueError(
            "fusion intensity_estimation_scale_mode must be one of auto, "
            "full_res, downsample_2x, or downsample_4x."
        )
    normalized["intensity_estimation_scale_mode"] = intensity_scale_mode

    max_intensity_fit_voxels = int(normalized.get("max_intensity_fit_voxels", 8_000_000))
    if max_intensity_fit_voxels <= 0:
        raise ValueError("fusion max_intensity_fit_voxels must be greater than zero.")
    normalized["max_intensity_fit_voxels"] = max_intensity_fit_voxels

    full_res_validation_samples = int(
        normalized.get("full_res_validation_samples", 50_000)
    )
    if full_res_validation_samples <= 0:
        raise ValueError(
            "fusion full_res_validation_samples must be greater than zero."
        )
    normalized["full_res_validation_samples"] = full_res_validation_samples

    validation_rmse_fraction = float(
        normalized.get("max_intensity_validation_rmse_fraction", 0.05)
    )
    if validation_rmse_fraction <= 0.0:
        raise ValueError(
            "fusion max_intensity_validation_rmse_fraction must be greater than zero."
        )
    normalized["max_intensity_validation_rmse_fraction"] = validation_rmse_fraction

    validation_correlation = float(
        normalized.get("min_intensity_validation_correlation", 0.95)
    )
    if not 0.0 <= validation_correlation <= 1.0:
        raise ValueError(
            "fusion min_intensity_validation_correlation must be between 0 and 1."
        )
    normalized["min_intensity_validation_correlation"] = validation_correlation
```

- [ ] **Step 4: Run the workflow tests to verify they pass**

Run: `uv run pytest -q tests/test_workflow.py -k "fusion_intensity_estimation_policy or default_zarr_save_config"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_workflow.py src/clearex/workflow.py
git commit -m "feat: normalize fusion intensity estimation policy"
```

## Task 2: Add GUI Controls for Fusion Gain-Compensation Policy

**Files:**
- Modify: `tests/gui/test_gui_execution.py:740-798`
- Modify: `src/clearex/gui/app.py:8211-8230`
- Modify: `src/clearex/gui/app.py:11053-11139`
- Modify: `src/clearex/gui/app.py:14767-14795`
- Modify: `src/clearex/gui/app.py:15875-15922`
- Modify: `src/clearex/gui/app.py:17101-17128`
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Write the failing GUI regression test**

Add a new test immediately after `test_analysis_dialog_collects_registration_and_fusion_parameters(...)`.

```python
def test_analysis_dialog_collects_fusion_intensity_estimation_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module,
        "_save_last_used_dask_backend_config",
        lambda _config: None,
    )

    store_path = _create_gui_analysis_store(tmp_path)
    dialog = app_module.AnalysisSelectionDialog(
        initial=app_module.WorkflowConfig(file=str(store_path))
    )
    dialog._persist_analysis_gui_state_for_target = lambda _target: None
    dialog._operation_checkboxes["registration"].setChecked(True)
    dialog._operation_checkboxes["fusion"].setChecked(True)
    dialog._fusion_blend_mode_combo.setCurrentIndex(
        dialog._fusion_blend_mode_combo.findData("gain_compensated_feather")
    )
    dialog._fusion_intensity_estimation_scale_mode_combo.setCurrentIndex(
        dialog._fusion_intensity_estimation_scale_mode_combo.findData("downsample_4x")
    )
    dialog._fusion_max_intensity_fit_voxels_spin.setValue(9_500_000)
    dialog._fusion_full_res_validation_samples_spin.setValue(60_000)
    dialog._fusion_max_intensity_validation_rmse_fraction_spin.setValue(0.08)
    dialog._fusion_min_intensity_validation_correlation_spin.setValue(0.92)

    app_module.AnalysisSelectionDialog._set_fusion_parameter_enabled_state(dialog)
    dialog._on_run()

    fusion = dialog.result_config.analysis_parameters["fusion"]
    assert fusion["intensity_estimation_scale_mode"] == "downsample_4x"
    assert fusion["max_intensity_fit_voxels"] == 9_500_000
    assert fusion["full_res_validation_samples"] == 60_000
    assert fusion["max_intensity_validation_rmse_fraction"] == pytest.approx(0.08)
    assert fusion["min_intensity_validation_correlation"] == pytest.approx(0.92)
```

- [ ] **Step 2: Run the GUI regression test to verify it fails**

Run: `uv run pytest -q tests/gui/test_gui_execution.py -k "fusion_intensity_estimation_policy"`

Expected: FAIL because the dialog currently has no fusion widgets for the new intensity-estimation policy.

- [ ] **Step 3: Add the new fusion widgets, hints, hydration, and collection**

Add parameter hints in `src/clearex/gui/app.py`:

```python
"fusion_intensity_estimation_scale_mode": (
    "Resolution used only for the gain-compensation intensity-fit prepass. "
    "Final fused chunks still render at full resolution."
),
"fusion_max_intensity_fit_voxels": (
    "Target voxel budget for one overlap intensity fit when scale mode is auto. "
    "The runtime chooses the coarsest 1x/2x/4x estimation grid that stays "
    "within this budget."
),
"fusion_full_res_validation_samples": (
    "Number of sparse full-resolution overlap samples used to validate a "
    "coarse gain-compensation fit before accepting it."
),
"fusion_max_intensity_validation_rmse_fraction": (
    "Maximum normalized RMSE allowed when validating a coarse intensity fit "
    "against sparse full-resolution samples."
),
"fusion_min_intensity_validation_correlation": (
    "Minimum correlation required for a coarse intensity fit to pass "
    "full-resolution validation."
),
```

Extend `_build_fusion_parameter_rows(...)`:

```python
self._fusion_intensity_estimation_scale_mode_combo = QComboBox()
self._fusion_intensity_estimation_scale_mode_combo.addItem("Auto", "auto")
self._fusion_intensity_estimation_scale_mode_combo.addItem("Full resolution", "full_res")
self._fusion_intensity_estimation_scale_mode_combo.addItem("Downsample 2x", "downsample_2x")
self._fusion_intensity_estimation_scale_mode_combo.addItem("Downsample 4x", "downsample_4x")
blend_form.addRow(
    "intensity fit scale", self._fusion_intensity_estimation_scale_mode_combo
)

self._fusion_max_intensity_fit_voxels_spin = QSpinBox()
self._fusion_max_intensity_fit_voxels_spin.setRange(1, 1_000_000_000)
self._fusion_max_intensity_fit_voxels_spin.setValue(8_000_000)
blend_form.addRow(
    "max fit voxels", self._fusion_max_intensity_fit_voxels_spin
)

self._fusion_full_res_validation_samples_spin = QSpinBox()
self._fusion_full_res_validation_samples_spin.setRange(1, 10_000_000)
self._fusion_full_res_validation_samples_spin.setValue(50_000)
blend_form.addRow(
    "validation samples", self._fusion_full_res_validation_samples_spin
)

self._fusion_max_intensity_validation_rmse_fraction_spin = QDoubleSpinBox()
self._fusion_max_intensity_validation_rmse_fraction_spin.setDecimals(3)
self._fusion_max_intensity_validation_rmse_fraction_spin.setRange(0.001, 1.0)
self._fusion_max_intensity_validation_rmse_fraction_spin.setSingleStep(0.01)
self._fusion_max_intensity_validation_rmse_fraction_spin.setValue(0.05)
blend_form.addRow(
    "max validation nrmse",
    self._fusion_max_intensity_validation_rmse_fraction_spin,
)

self._fusion_min_intensity_validation_correlation_spin = QDoubleSpinBox()
self._fusion_min_intensity_validation_correlation_spin.setDecimals(3)
self._fusion_min_intensity_validation_correlation_spin.setRange(0.0, 1.0)
self._fusion_min_intensity_validation_correlation_spin.setSingleStep(0.01)
self._fusion_min_intensity_validation_correlation_spin.setValue(0.95)
blend_form.addRow(
    "min validation corr",
    self._fusion_min_intensity_validation_correlation_spin,
)
```

Extend `_set_fusion_parameter_enabled_state(...)`, hydration, and collection:

```python
gain_clip_widgets = (
    self._fusion_gain_clip_min_spin,
    self._fusion_gain_clip_max_spin,
    self._fusion_intensity_estimation_scale_mode_combo,
    self._fusion_max_intensity_fit_voxels_spin,
    self._fusion_full_res_validation_samples_spin,
    self._fusion_max_intensity_validation_rmse_fraction_spin,
    self._fusion_min_intensity_validation_correlation_spin,
)
for widget in gain_clip_widgets:
    widget.setEnabled(gain_clip_enabled)
```

```python
self._fusion_intensity_estimation_scale_mode_combo.setCurrentIndex(
    max(
        0,
        self._fusion_intensity_estimation_scale_mode_combo.findData(
            str(fusion_params.get("intensity_estimation_scale_mode", "auto")).strip().lower()
            or "auto"
        ),
    )
)
self._fusion_max_intensity_fit_voxels_spin.setValue(
    max(1, int(fusion_params.get("max_intensity_fit_voxels", 8_000_000)))
)
self._fusion_full_res_validation_samples_spin.setValue(
    max(1, int(fusion_params.get("full_res_validation_samples", 50_000)))
)
self._fusion_max_intensity_validation_rmse_fraction_spin.setValue(
    max(
        0.001,
        float(fusion_params.get("max_intensity_validation_rmse_fraction", 0.05)),
    )
)
self._fusion_min_intensity_validation_correlation_spin.setValue(
    min(
        1.0,
        max(
            0.0,
            float(fusion_params.get("min_intensity_validation_correlation", 0.95)),
        ),
    )
)
```

```python
"intensity_estimation_scale_mode": str(
    self._fusion_intensity_estimation_scale_mode_combo.currentData() or "auto"
).strip()
or "auto",
"max_intensity_fit_voxels": int(self._fusion_max_intensity_fit_voxels_spin.value()),
"full_res_validation_samples": int(
    self._fusion_full_res_validation_samples_spin.value()
),
"max_intensity_validation_rmse_fraction": float(
    self._fusion_max_intensity_validation_rmse_fraction_spin.value()
),
"min_intensity_validation_correlation": float(
    self._fusion_min_intensity_validation_correlation_spin.value()
),
```

- [ ] **Step 4: Run the GUI regression test to verify it passes**

Run: `uv run pytest -q tests/gui/test_gui_execution.py -k "fusion_intensity_estimation_policy"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/gui/test_gui_execution.py src/clearex/gui/app.py
git commit -m "feat: expose fusion intensity estimation policy in gui"
```

## Task 3: Add Pure Pipeline Helpers for Scale Selection and Validation

**Files:**
- Modify: `tests/registration/test_pipeline.py:1240-1320`
- Modify: `src/clearex/registration/pipeline.py:52-70`
- Modify: `src/clearex/registration/pipeline.py:907-951`
- Modify: `src/clearex/registration/pipeline.py:1079-1145`
- Test: `tests/registration/test_pipeline.py`

- [ ] **Step 1: Write the failing helper tests**

Add these tests near `TestBlendWeightProfiles`.

```python
def test_select_intensity_estimation_scale_prefers_coarsest_budget() -> None:
    assert registration_pipeline._select_intensity_estimation_scale(
        overlap_voxels=96_000_000,
        mode="auto",
        max_intensity_fit_voxels=8_000_000,
    ) == (4, 4, 4)
    assert registration_pipeline._select_intensity_estimation_scale(
        overlap_voxels=12_000_000,
        mode="auto",
        max_intensity_fit_voxels=8_000_000,
    ) == (2, 2, 2)
    assert registration_pipeline._select_intensity_estimation_scale(
        overlap_voxels=6_000_000,
        mode="auto",
        max_intensity_fit_voxels=8_000_000,
    ) == (1, 1, 1)


def test_validate_intensity_estimate_accepts_good_fit_and_rejects_bad_fit() -> None:
    moving = np.linspace(10.0, 100.0, 2000, dtype=np.float32).reshape(20, 10, 10)
    fixed = (moving * 0.5).astype(np.float32)

    good = registration_pipeline._validate_intensity_estimate(
        fixed_crop_zyx=fixed,
        moving_crop_zyx=moving,
        scale=0.5,
        offset=0.0,
        full_res_validation_samples=2000,
        max_validation_rmse_fraction=0.05,
        min_validation_correlation=0.95,
    )
    assert good["accepted"] is True
    assert good["sample_count"] == 2000

    bad = registration_pipeline._validate_intensity_estimate(
        fixed_crop_zyx=fixed,
        moving_crop_zyx=moving,
        scale=1.0,
        offset=0.0,
        full_res_validation_samples=2000,
        max_validation_rmse_fraction=0.05,
        min_validation_correlation=0.95,
    )
    assert bad["accepted"] is False
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "select_intensity_estimation_scale or validate_intensity_estimate"`

Expected: FAIL because `_select_intensity_estimation_scale(...)` and `_validate_intensity_estimate(...)` do not exist yet.

- [ ] **Step 3: Add the pure helper functions**

Add new constants near the top of `src/clearex/registration/pipeline.py`:

```python
_DEFAULT_INTENSITY_ESTIMATION_SCALE_MODE = "auto"
_DEFAULT_MAX_INTENSITY_FIT_VOXELS = 8_000_000
_DEFAULT_FULL_RES_VALIDATION_SAMPLES = 50_000
_DEFAULT_MAX_INTENSITY_VALIDATION_RMSE_FRACTION = 0.05
_DEFAULT_MIN_INTENSITY_VALIDATION_CORRELATION = 0.95
_FUSION_INTENSITY_CACHE_POLICY_VERSION = 1
```

Add the new helpers immediately after `_crop_from_overlap_bbox(...)` and `_estimate_linear_intensity_map(...)`:

```python
def _select_intensity_estimation_scale(
    *,
    overlap_voxels: int,
    mode: str,
    max_intensity_fit_voxels: int,
) -> tuple[int, int, int]:
    normalized = str(mode).strip().lower() or _DEFAULT_INTENSITY_ESTIMATION_SCALE_MODE
    if normalized == "full_res":
        return (1, 1, 1)
    if normalized == "downsample_2x":
        return (2, 2, 2)
    if normalized == "downsample_4x":
        return (4, 4, 4)

    for factor in (4, 2, 1):
        estimated = int(math.ceil(float(overlap_voxels) / float(factor**3)))
        if estimated <= int(max_intensity_fit_voxels):
            return (factor, factor, factor)
    return (4, 4, 4)
```

```python
def _downsample_crop_for_intensity_estimation(
    volume_zyx: np.ndarray,
    *,
    scale_zyx: tuple[int, int, int],
) -> np.ndarray:
    zoom = tuple(1.0 / max(1, int(axis_scale)) for axis_scale in scale_zyx)
    return ndimage.zoom(
        np.asarray(volume_zyx, dtype=np.float32, copy=False),
        zoom=zoom,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)
```

```python
def _validate_intensity_estimate(
    *,
    fixed_crop_zyx: np.ndarray,
    moving_crop_zyx: np.ndarray,
    scale: float,
    offset: float,
    full_res_validation_samples: int,
    max_validation_rmse_fraction: float,
    min_validation_correlation: float,
) -> dict[str, Any]:
    valid_mask = (
        np.isfinite(fixed_crop_zyx)
        & np.isfinite(moving_crop_zyx)
        & (fixed_crop_zyx > 0)
        & (moving_crop_zyx > 0)
    )
    sample_count = int(np.count_nonzero(valid_mask))
    if sample_count == 0:
        return {
            "accepted": False,
            "correlation": 0.0,
            "normalized_rmse": float("inf"),
            "sample_count": 0,
        }

    fixed = np.asarray(fixed_crop_zyx[valid_mask], dtype=np.float64)
    moving = np.asarray(moving_crop_zyx[valid_mask], dtype=np.float64)
    if fixed.size > int(full_res_validation_samples):
        sample_indices = np.linspace(
            0,
            int(fixed.size) - 1,
            int(full_res_validation_samples),
            dtype=np.int64,
        )
        fixed = fixed[sample_indices]
        moving = moving[sample_indices]

    corrected = (moving * float(scale)) + float(offset)
    fixed_range = max(1e-6, float(np.max(fixed) - np.min(fixed)))
    rmse = float(np.sqrt(np.mean((corrected - fixed) ** 2)))
    normalized_rmse = rmse / fixed_range
    if fixed.size < 2:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(corrected, fixed)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    return {
        "accepted": bool(
            correlation >= float(min_validation_correlation)
            and normalized_rmse <= float(max_validation_rmse_fraction)
        ),
        "correlation": correlation,
        "normalized_rmse": normalized_rmse,
        "sample_count": int(fixed.size),
    }
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "select_intensity_estimation_scale or validate_intensity_estimate"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/registration/test_pipeline.py src/clearex/registration/pipeline.py
git commit -m "feat: add fusion intensity estimation helper functions"
```

## Task 4: Integrate Scale-Aware Intensity Estimation and Granular Progress

**Files:**
- Modify: `tests/registration/test_pipeline.py:428-515`
- Modify: `tests/registration/test_pipeline.py:679-798`
- Modify: `src/clearex/registration/pipeline.py:1467-1538`
- Modify: `src/clearex/registration/pipeline.py:3429-3518`
- Test: `tests/registration/test_pipeline.py`

- [ ] **Step 1: Write the failing fusion integration tests**

Add two tests.

```python
def test_run_fusion_analysis_reports_intensity_fit_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )

    monkeypatch.setattr(
        registration_pipeline,
        "_estimate_pairwise_overlap_intensity",
        lambda **kwargs: {
            "fixed_position": int(kwargs["edge"].fixed_position),
            "moving_position": int(kwargs["edge"].moving_position),
            "success": True,
            "overlap_voxels": int(kwargs["edge"].overlap_voxels),
            "intensity_success": True,
            "intensity_scale": 0.5,
            "intensity_offset": 0.0,
            "intensity_samples": 500,
            "status": "coarse_plus_validate",
            "estimation_scale_zyx": [4, 4, 4],
            "validation_sample_count": 500,
            "validation_corr": 0.99,
            "validation_nrmse": 0.01,
        },
    )

    registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
    )

    progress_updates: list[tuple[int, str]] = []
    registration_pipeline.run_fusion_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "registration",
            "blend_mode": "gain_compensated_feather",
            "blend_overlap_zyx": [0, 0, 2],
            "intensity_estimation_scale_mode": "downsample_4x",
            "max_intensity_fit_voxels": 8_000_000,
            "full_res_validation_samples": 50_000,
            "max_intensity_validation_rmse_fraction": 0.05,
            "min_intensity_validation_correlation": 0.95,
        },
        client=None,
        progress_callback=lambda percent, message: progress_updates.append(
            (int(percent), str(message))
        ),
    )

    assert any("Intensity fits" in message for _, message in progress_updates)
    assert any(12 <= percent <= 20 for percent, _ in progress_updates)


def test_run_fusion_analysis_passes_scale_policy_to_edge_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )

    captured: dict[str, Any] = {}

    def _fake_intensity(**kwargs):
        captured.update(kwargs)
        return {
            "fixed_position": int(kwargs["edge"].fixed_position),
            "moving_position": int(kwargs["edge"].moving_position),
            "success": True,
            "overlap_voxels": int(kwargs["edge"].overlap_voxels),
            "intensity_success": True,
            "intensity_scale": 0.5,
            "intensity_offset": 0.0,
            "intensity_samples": 500,
            "status": "coarse_plus_validate",
            "estimation_scale_zyx": [2, 2, 2],
            "validation_sample_count": 500,
            "validation_corr": 0.99,
            "validation_nrmse": 0.01,
        }

    monkeypatch.setattr(
        registration_pipeline, "_estimate_pairwise_overlap_intensity", _fake_intensity
    )

    registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
    )

    registration_pipeline.run_fusion_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "registration",
            "blend_mode": "gain_compensated_feather",
            "blend_overlap_zyx": [0, 0, 2],
            "intensity_estimation_scale_mode": "downsample_2x",
            "max_intensity_fit_voxels": 7_500_000,
            "full_res_validation_samples": 40_000,
            "max_intensity_validation_rmse_fraction": 0.04,
            "min_intensity_validation_correlation": 0.97,
        },
        client=None,
    )

    assert captured["intensity_estimation_scale_mode"] == "downsample_2x"
    assert captured["max_intensity_fit_voxels"] == 7_500_000
    assert captured["full_res_validation_samples"] == 40_000
    assert captured["max_intensity_validation_rmse_fraction"] == pytest.approx(0.04)
    assert captured["min_intensity_validation_correlation"] == pytest.approx(0.97)
```

- [ ] **Step 2: Run the integration tests to verify they fail**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "reports_intensity_fit_progress or passes_scale_policy_to_edge_estimator"`

Expected: FAIL because `run_fusion_analysis(...)` does not yet pass the new policy fields into `_estimate_pairwise_overlap_intensity(...)` and does not emit granular prepass progress.

- [ ] **Step 3: Integrate scale-aware estimation and progress reporting**

Extend `_estimate_pairwise_overlap_intensity(...)` with the new parameters and fallback loop:

```python
def _estimate_pairwise_overlap_intensity(
    *,
    zarr_path: str,
    source_component: str,
    t_index: int,
    registration_channel: int,
    edge: _EdgeSpec,
    fixed_transform_xyz: np.ndarray,
    moving_transform_xyz: np.ndarray,
    voxel_size_um_zyx: Sequence[float],
    overlap_zyx: Sequence[int],
    gain_clip_range: Sequence[float] = _DEFAULT_GAIN_CLIP_RANGE,
    intensity_estimation_scale_mode: str = _DEFAULT_INTENSITY_ESTIMATION_SCALE_MODE,
    max_intensity_fit_voxels: int = _DEFAULT_MAX_INTENSITY_FIT_VOXELS,
    full_res_validation_samples: int = _DEFAULT_FULL_RES_VALIDATION_SAMPLES,
    max_intensity_validation_rmse_fraction: float = (
        _DEFAULT_MAX_INTENSITY_VALIDATION_RMSE_FRACTION
    ),
    min_intensity_validation_correlation: float = (
        _DEFAULT_MIN_INTENSITY_VALIDATION_CORRELATION
    ),
) -> dict[str, Any]:
    ...
    initial_scale = _select_intensity_estimation_scale(
        overlap_voxels=int(edge.overlap_voxels),
        mode=intensity_estimation_scale_mode,
        max_intensity_fit_voxels=max_intensity_fit_voxels,
    )
    scale_candidates = [initial_scale]
    if initial_scale == (4, 4, 4):
        scale_candidates.extend([(2, 2, 2), (1, 1, 1)])
    elif initial_scale == (2, 2, 2):
        scale_candidates.append((1, 1, 1))

    for scale_zyx in scale_candidates:
        fixed_est = (
            fixed_crop
            if scale_zyx == (1, 1, 1)
            else _downsample_crop_for_intensity_estimation(
                fixed_crop,
                scale_zyx=scale_zyx,
            )
        )
        moving_est = (
            moving_crop
            if scale_zyx == (1, 1, 1)
            else _downsample_crop_for_intensity_estimation(
                moving_crop,
                scale_zyx=scale_zyx,
            )
        )
        success, scale, offset, fit_sample_count = _estimate_linear_intensity_map(
            fixed_est, moving_est, gain_clip_range=gain_clip_range
        )
        if not success:
            continue
        validation = _validate_intensity_estimate(
            fixed_crop_zyx=fixed_crop,
            moving_crop_zyx=moving_crop,
            scale=scale,
            offset=offset,
            full_res_validation_samples=full_res_validation_samples,
            max_validation_rmse_fraction=max_intensity_validation_rmse_fraction,
            min_validation_correlation=min_intensity_validation_correlation,
        )
        if validation["accepted"]:
            return {
                "fixed_position": int(edge.fixed_position),
                "moving_position": int(edge.moving_position),
                "success": True,
                "overlap_voxels": int(edge.overlap_voxels),
                "intensity_success": True,
                "intensity_scale": float(scale),
                "intensity_offset": float(offset),
                "intensity_samples": int(fit_sample_count),
                "status": "coarse_plus_validate"
                if scale_zyx != (1, 1, 1)
                else "full_res",
                "estimation_scale_zyx": [int(v) for v in scale_zyx],
                "validation_sample_count": int(validation["sample_count"]),
                "validation_corr": float(validation["correlation"]),
                "validation_nrmse": float(validation["normalized_rmse"]),
            }
```

Pass the new policy into `run_fusion_analysis(...)` and emit prepass progress per completed edge:

```python
effective_intensity_estimation_scale_mode = str(
    parameters.get(
        "intensity_estimation_scale_mode",
        _DEFAULT_INTENSITY_ESTIMATION_SCALE_MODE,
    )
).strip().lower() or _DEFAULT_INTENSITY_ESTIMATION_SCALE_MODE
effective_max_intensity_fit_voxels = int(
    parameters.get("max_intensity_fit_voxels", _DEFAULT_MAX_INTENSITY_FIT_VOXELS)
)
effective_full_res_validation_samples = int(
    parameters.get(
        "full_res_validation_samples", _DEFAULT_FULL_RES_VALIDATION_SAMPLES
    )
)
effective_max_intensity_validation_rmse_fraction = float(
    parameters.get(
        "max_intensity_validation_rmse_fraction",
        _DEFAULT_MAX_INTENSITY_VALIDATION_RMSE_FRACTION,
    )
)
effective_min_intensity_validation_correlation = float(
    parameters.get(
        "min_intensity_validation_correlation",
        _DEFAULT_MIN_INTENSITY_VALIDATION_CORRELATION,
    )
)
```

```python
completed_edge_fits = 0
total_edge_fits = max(1, len(delayed_intensity_edges))
for future in as_completed(client.compute(delayed_intensity_edges)):
    edge_results.append(future.result())
    completed_edge_fits += 1
    latest = edge_results[-1]
    latest_scale = "x".join(str(int(v)) for v in latest["estimation_scale_zyx"])
    _emit(
        progress_callback,
        12 + int((completed_edge_fits / total_edge_fits) * 8),
        "Intensity fits "
        f"{completed_edge_fits}/{total_edge_fits} edges "
        f"(latest scale={latest_scale}, status={latest['status']})",
    )
```

- [ ] **Step 4: Run the integration tests to verify they pass**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "reports_intensity_fit_progress or passes_scale_policy_to_edge_estimator"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/registration/test_pipeline.py src/clearex/registration/pipeline.py
git commit -m "feat: add scale-aware fusion intensity estimation"
```

## Task 5: Persist and Reuse Small `intensity_fit_edges` Metadata on Reruns

**Files:**
- Modify: `tests/registration/test_pipeline.py:679-798`
- Modify: `src/clearex/registration/pipeline.py:2055-2078`
- Modify: `src/clearex/registration/pipeline.py:3429-3651`
- Test: `tests/registration/test_pipeline.py`

- [ ] **Step 1: Write the failing rerun reuse tests**

Add two tests immediately after `test_run_fusion_analysis_parallelizes_intensity_estimation_with_client(...)`.

```python
def test_run_fusion_analysis_reuses_matching_intensity_fit_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )

    call_count = {"estimate": 0}

    def _fake_intensity(**kwargs):
        call_count["estimate"] += 1
        return {
            "fixed_position": int(kwargs["edge"].fixed_position),
            "moving_position": int(kwargs["edge"].moving_position),
            "success": True,
            "overlap_voxels": int(kwargs["edge"].overlap_voxels),
            "intensity_success": True,
            "intensity_scale": 0.5,
            "intensity_offset": 0.0,
            "intensity_samples": 500,
            "status": "coarse_plus_validate",
            "estimation_scale_zyx": [4, 4, 4],
            "validation_sample_count": 500,
            "validation_corr": 0.99,
            "validation_nrmse": 0.01,
        }

    monkeypatch.setattr(
        registration_pipeline, "_estimate_pairwise_overlap_intensity", _fake_intensity
    )

    registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
    )

    params = {
        "input_source": "registration",
        "blend_mode": "gain_compensated_feather",
        "blend_overlap_zyx": [0, 0, 2],
        "intensity_estimation_scale_mode": "downsample_4x",
        "max_intensity_fit_voxels": 8_000_000,
        "full_res_validation_samples": 50_000,
        "max_intensity_validation_rmse_fraction": 0.05,
        "min_intensity_validation_correlation": 0.95,
    }
    registration_pipeline.run_fusion_analysis(
        zarr_path=store_path, parameters=params, client=None
    )
    registration_pipeline.run_fusion_analysis(
        zarr_path=store_path, parameters=params, client=None
    )

    assert call_count["estimate"] == 1


def test_run_fusion_analysis_invalidates_intensity_fit_cache_on_geometry_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = _create_registration_store(
        tmp_path,
        timepoints=1,
        positions=2,
        channels=1,
        shape_zyx=(4, 4, 6),
        include_pyramid=False,
    )

    call_count = {"estimate": 0}

    def _fake_intensity(**kwargs):
        call_count["estimate"] += 1
        return {
            "fixed_position": int(kwargs["edge"].fixed_position),
            "moving_position": int(kwargs["edge"].moving_position),
            "success": True,
            "overlap_voxels": int(kwargs["edge"].overlap_voxels),
            "intensity_success": True,
            "intensity_scale": 0.5,
            "intensity_offset": 0.0,
            "intensity_samples": 500,
            "status": "coarse_plus_validate",
            "estimation_scale_zyx": [2, 2, 2],
            "validation_sample_count": 500,
            "validation_corr": 0.99,
            "validation_nrmse": 0.01,
        }

    monkeypatch.setattr(
        registration_pipeline, "_estimate_pairwise_overlap_intensity", _fake_intensity
    )

    registration_pipeline.run_registration_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "registration_channel": 0,
            "registration_type": "translation",
            "input_resolution_level": 0,
            "anchor_mode": "central",
            "anchor_position": None,
            "pairwise_overlap_zyx": [0, 0, 2],
        },
        client=None,
    )

    params = {
        "input_source": "registration",
        "blend_mode": "gain_compensated_feather",
        "blend_overlap_zyx": [0, 0, 2],
        "intensity_estimation_scale_mode": "downsample_2x",
        "max_intensity_fit_voxels": 8_000_000,
        "full_res_validation_samples": 50_000,
        "max_intensity_validation_rmse_fraction": 0.05,
        "min_intensity_validation_correlation": 0.95,
    }
    registration_pipeline.run_fusion_analysis(
        zarr_path=store_path, parameters=params, client=None
    )

    root = zarr.open_group(str(store_path), mode="a")
    affines = root["clearex/results/registration/latest/affines_tpx44"]
    updated = np.asarray(affines[:], dtype=np.float64)
    updated[0, 1, 0, 3] += 1.0
    affines[:] = updated

    registration_pipeline.run_fusion_analysis(
        zarr_path=store_path, parameters=params, client=None
    )
    assert call_count["estimate"] == 2
```

- [ ] **Step 2: Run the rerun reuse tests to verify they fail**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "reuses_matching_intensity_fit_cache or invalidates_intensity_fit_cache_on_geometry_change"`

Expected: FAIL because fusion currently deletes the old latest group and recomputes all edge fits on every rerun.

- [ ] **Step 3: Add cache hashing, cache read/write, and rerun reuse**

Add `import hashlib` near the existing top-level imports in
`src/clearex/registration/pipeline.py`, then add cache helpers near
`_solve_intensity_corrections(...)`:

```python
def _fusion_intensity_geometry_sha256(
    *,
    affines_tpx44: np.ndarray,
    transformed_bboxes_tpx6: np.ndarray,
    edges_pe2: np.ndarray,
) -> str:
    payload = hashlib.sha256()
    payload.update(np.asarray(affines_tpx44, dtype=np.float64).tobytes())
    payload.update(np.asarray(transformed_bboxes_tpx6, dtype=np.float64).tobytes())
    payload.update(np.asarray(edges_pe2, dtype=np.int32).tobytes())
    return payload.hexdigest()
```

```python
def _load_cached_fusion_intensity_edges(
    root: zarr.Group,
    *,
    registration_component: str,
    source_component: str,
    blend_mode: str,
    gain_clip_range: tuple[float, float],
    intensity_estimation_scale_mode: str,
    max_intensity_fit_voxels: int,
    full_res_validation_samples: int,
    max_intensity_validation_rmse_fraction: float,
    min_intensity_validation_correlation: float,
    geometry_sha256: str,
) -> Optional[dict[str, np.ndarray]]:
    group_path = f"{analysis_auxiliary_root('fusion')}/intensity_fit_edges"
    if group_path not in root:
        return None
    group = root[group_path]
    attrs = dict(group.attrs)
    if attrs.get("cache_policy_version") != _FUSION_INTENSITY_CACHE_POLICY_VERSION:
        return None
    if attrs.get("geometry_sha256") != geometry_sha256:
        return None
    if attrs.get("registration_component") != str(registration_component):
        return None
    if attrs.get("source_component") != str(source_component):
        return None
    if attrs.get("blend_mode") != str(blend_mode):
        return None
    if list(attrs.get("gain_clip_range", [])) != [float(gain_clip_range[0]), float(gain_clip_range[1])]:
        return None
    if attrs.get("intensity_estimation_scale_mode") != str(intensity_estimation_scale_mode):
        return None
    if int(attrs.get("max_intensity_fit_voxels", -1)) != int(max_intensity_fit_voxels):
        return None
    if int(attrs.get("full_res_validation_samples", -1)) != int(full_res_validation_samples):
        return None
    if float(attrs.get("max_intensity_validation_rmse_fraction", -1.0)) != float(
        max_intensity_validation_rmse_fraction
    ):
        return None
    if float(attrs.get("min_intensity_validation_correlation", -1.0)) != float(
        min_intensity_validation_correlation
    ):
        return None
    return {
        "gains_te": np.asarray(group["gains_te"][:], dtype=np.float32),
        "offsets_te": np.asarray(group["offsets_te"][:], dtype=np.float32),
        "status_te": np.asarray(group["status_te"][:], dtype=np.uint8),
        "estimation_scale_te3": np.asarray(group["estimation_scale_te3"][:], dtype=np.uint8),
        "fit_sample_count_te": np.asarray(group["fit_sample_count_te"][:], dtype=np.int32),
        "validation_sample_count_te": np.asarray(group["validation_sample_count_te"][:], dtype=np.int32),
        "validation_corr_te": np.asarray(group["validation_corr_te"][:], dtype=np.float32),
        "validation_nrmse_te": np.asarray(group["validation_nrmse_te"][:], dtype=np.float32),
}
```

In `run_fusion_analysis(...)`, load any reusable cache before `_prepare_output_group(...)` deletes the old auxiliary group, then skip edge estimation when a matching cache exists:

```python
geometry_sha256 = _fusion_intensity_geometry_sha256(
    affines_tpx44=affines,
    transformed_bboxes_tpx6=transformed_bboxes,
    edges_pe2=edges_pe2,
)
reuse_root = zarr.open_group(str(zarr_path), mode="r")
cached_intensity_edges = _load_cached_fusion_intensity_edges(
    reuse_root,
    registration_component=str(registration_component),
    source_component=str(source_component),
    blend_mode=str(blend_mode),
    gain_clip_range=effective_gain_clip_range,
    intensity_estimation_scale_mode=effective_intensity_estimation_scale_mode,
    max_intensity_fit_voxels=effective_max_intensity_fit_voxels,
    full_res_validation_samples=effective_full_res_validation_samples,
    max_intensity_validation_rmse_fraction=effective_max_intensity_validation_rmse_fraction,
    min_intensity_validation_correlation=effective_min_intensity_validation_correlation,
    geometry_sha256=geometry_sha256,
)
```

Create the per-edge metadata arrays before the gain-compensation branch so both
fresh computation and cache reuse fill the same storage structures:

```python
edge_count = int(edges_pe2.shape[0])
edge_gains_te = np.ones((int(source_shape_tpczyx[0]), edge_count), dtype=np.float32)
edge_offsets_te = np.zeros((int(source_shape_tpczyx[0]), edge_count), dtype=np.float32)
edge_status_te = np.zeros((int(source_shape_tpczyx[0]), edge_count), dtype=np.uint8)
edge_scale_te3 = np.ones((int(source_shape_tpczyx[0]), edge_count, 3), dtype=np.uint8)
edge_fit_sample_count_te = np.zeros(
    (int(source_shape_tpczyx[0]), edge_count), dtype=np.int32
)
edge_validation_sample_count_te = np.zeros(
    (int(source_shape_tpczyx[0]), edge_count), dtype=np.int32
)
edge_validation_corr_te = np.zeros(
    (int(source_shape_tpczyx[0]), edge_count), dtype=np.float32
)
edge_validation_nrmse_te = np.full(
    (int(source_shape_tpczyx[0]), edge_count), np.nan, dtype=np.float32
)
```

When the cache matches, populate those arrays directly from the cached payload
and skip `_estimate_pairwise_overlap_intensity(...)`. When the cache does not
match, fill those arrays from fresh `edge_results` before calling
`_solve_intensity_corrections(...)`.

After `_prepare_output_group(...)`, write the cache metadata back into the new fusion auxiliary group:

```python
fit_group = fusion_group.create_group("intensity_fit_edges", overwrite=True)
fit_group.attrs.update(
    {
        "cache_policy_version": _FUSION_INTENSITY_CACHE_POLICY_VERSION,
        "registration_component": str(registration_component),
        "source_component": str(source_component),
        "blend_mode": str(blend_mode),
        "gain_clip_range": [float(v) for v in effective_gain_clip_range],
        "intensity_estimation_scale_mode": str(effective_intensity_estimation_scale_mode),
        "max_intensity_fit_voxels": int(effective_max_intensity_fit_voxels),
        "full_res_validation_samples": int(effective_full_res_validation_samples),
        "max_intensity_validation_rmse_fraction": float(
            effective_max_intensity_validation_rmse_fraction
        ),
        "min_intensity_validation_correlation": float(
            effective_min_intensity_validation_correlation
        ),
        "geometry_sha256": geometry_sha256,
    }
)
fit_group.create_array("gains_te", data=edge_gains_te, overwrite=True)
fit_group.create_array("offsets_te", data=edge_offsets_te, overwrite=True)
fit_group.create_array("status_te", data=edge_status_te, overwrite=True)
fit_group.create_array("estimation_scale_te3", data=edge_scale_te3, overwrite=True)
fit_group.create_array("fit_sample_count_te", data=edge_fit_sample_count_te, overwrite=True)
fit_group.create_array(
    "validation_sample_count_te",
    data=edge_validation_sample_count_te,
    overwrite=True,
)
fit_group.create_array("validation_corr_te", data=edge_validation_corr_te, overwrite=True)
fit_group.create_array("validation_nrmse_te", data=edge_validation_nrmse_te, overwrite=True)
```

- [ ] **Step 4: Run the rerun reuse tests to verify they pass**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "reuses_matching_intensity_fit_cache or invalidates_intensity_fit_cache_on_geometry_change"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/registration/test_pipeline.py src/clearex/registration/pipeline.py
git commit -m "feat: reuse cached fusion intensity edge fits"
```

## Task 6: Update Fusion Documentation and Run the Targeted Verification Sweep

**Files:**
- Modify: `src/clearex/registration/README.md:507-601`
- Test: `tests/test_workflow.py`, `tests/gui/test_gui_execution.py`, `tests/registration/test_pipeline.py`

- [ ] **Step 1: Write the README update**

Update the `Fusion Parameters` and guidance sections in `src/clearex/registration/README.md` so they document the new policy surface and reuse behavior.

```markdown
| `intensity_estimation_scale_mode` | `"auto"` | Resolution used only for the gain-compensation intensity-fit prepass. `auto` chooses among `1x`, `2x`, and `4x` estimation grids based on overlap voxel budget. Final fused chunks still render at full resolution. |
| `max_intensity_fit_voxels` | `8_000_000` | Overlap voxel budget used by `auto` scale selection for gain compensation. Larger values favor denser estimation and higher prepass cost. |
| `full_res_validation_samples` | `50_000` | Sparse full-resolution overlap samples used to validate a coarse gain estimate before it is accepted. |
| `max_intensity_validation_rmse_fraction` | `0.05` | Maximum normalized RMSE allowed during coarse-fit validation. |
| `min_intensity_validation_correlation` | `0.95` | Minimum correlation required during coarse-fit validation. |
```

Add this note under `gain_compensated_feather` guidance:

```markdown
- ClearEx persists small `intensity_fit_edges` metadata under
  `clearex/results/fusion/latest` so reruns on unchanged registration geometry
  can reuse accepted edge fits instead of recomputing the full gain-estimation
  prepass.
- Estimation downsampling affects only the gain-compensation prepass. Final
  fused image chunks are still rendered from full-resolution source tiles.
```

- [ ] **Step 2: Run the targeted regression suite**

Run: `uv run pytest -q tests/test_workflow.py tests/gui/test_gui_execution.py tests/registration/test_pipeline.py`

Expected: PASS

- [ ] **Step 3: Run lint on the touched Python files**

Run: `uv run ruff check src/clearex/workflow.py src/clearex/gui/app.py src/clearex/registration/pipeline.py tests/test_workflow.py tests/gui/test_gui_execution.py tests/registration/test_pipeline.py`

Expected: PASS

- [ ] **Step 4: Re-run the highest-signal fusion regression subset**

Run: `uv run pytest -q tests/registration/test_pipeline.py -k "gain_compensated_feather or intensity_fit or validation or cache"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/clearex/registration/README.md
git commit -m "docs: describe bounded gain-compensated fusion estimation"
```

## Self-Review

### Spec Coverage

- Bounded `1x/2x/4x` estimation scales: covered by Tasks 1, 3, and 6.
- Sparse full-resolution validation: covered by Tasks 1, 2, 3, and 6.
- Small rerun-safe `intensity_fit_edges` metadata reuse: covered by Task 5.
- Granular `12% -> 20%` prepass progress: covered by Task 4.
- No public fusion image-contract change: preserved across Tasks 3 and 5.
- GUI/operator access to the new policy: covered by Task 2.
- No spec gaps found.

### Placeholder Scan

- No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
- All code-changing steps include concrete code snippets.
- All verification steps use explicit commands and expected outcomes.

### Type Consistency

- Workflow parameter names are consistently:
  - `intensity_estimation_scale_mode`
  - `max_intensity_fit_voxels`
  - `full_res_validation_samples`
  - `max_intensity_validation_rmse_fraction`
  - `min_intensity_validation_correlation`
- Pipeline helper names are consistently:
  - `_select_intensity_estimation_scale`
  - `_downsample_crop_for_intensity_estimation`
  - `_validate_intensity_estimate`
  - `_fusion_intensity_geometry_sha256`
  - `_load_cached_fusion_intensity_edges`
