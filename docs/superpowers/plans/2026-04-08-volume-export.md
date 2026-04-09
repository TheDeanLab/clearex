# Volume Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visualization-family `volume_export` operation that exports one selected image-producing source as an in-store OME-Zarr result and optional TIFF artifacts, supporting either one explicit `(t, p, c)` preview volume or all available indices at a selected resolution level.

**Architecture:** Register `volume_export` as a first-class analysis operation in workflow normalization, GUI selection, CLI/main dispatch, and provenance. Implement the runtime in a dedicated `src/clearex/visualization/volume_export.py` module that writes canonical `TPCZYX` data into `clearex/runtime_cache/results/volume_export/latest/data`, reuses or builds display-pyramid helper levels on demand, emits optional TIFF artifacts, and relies on the existing main-owned `publish_analysis_collection_from_cache(...)` path to publish `results/volume_export/latest`.

**Tech Stack:** Python, Dask, Zarr, tifffile, Qt widgets in `src/clearex/gui/app.py`, existing ClearEx OME-Zarr publishing helpers in `src/clearex/io/ome_store.py`, pytest, ruff, basedpyright.

---

## File Map

- Modify: `src/clearex/workflow.py`
  - Register `volume_export` in canonical ordering, defaults, normalization, selection helpers, and `WorkflowConfig`.
- Modify: `src/clearex/io/cli.py`
  - Add `--volume-export` CLI flag.
- Modify: `src/clearex/main.py`
  - Import `run_volume_export_analysis`, propagate the CLI flag into `WorkflowConfig`, dispatch the new operation, record latest outputs, and publish the public OME image collection with `publish_analysis_collection_from_cache(..., analysis_name="volume_export")`.
- Modify: `src/clearex/gui/app.py`
  - Add operation labels/tabs/output component mappings, panel widgets, parameter collection, restore logic, and widget enablement rules.
- Create: `src/clearex/visualization/volume_export.py`
  - Add runtime dataclass, cache write helpers, resolution-level resolution/build helpers, TIFF writers, metadata/provenance writer, and the public `run_volume_export_analysis(...)` entrypoint.
- Modify: `src/clearex/visualization/__init__.py`
  - Expose `run_volume_export_analysis` and `VolumeExportSummary` lazily.
- Modify: `src/clearex/visualization/README.md`
  - Document the new visualization-family export workflow.
- Modify: `README.md`
  - Add the feature to the high-level workflow and storage documentation.
- Modify: `src/clearex/AGENTS.md`
  - Record the new runtime update and storage expectation.
- Modify: `docs/source/runtime/cli-and-execution.rst`
  - Document the CLI flag and output locations.
- Modify: `docs/source/runtime/architecture-overview.rst`
  - Add `volume_export` to the runtime map.
- Modify: `tests/test_workflow.py`
  - Cover defaults, normalization, selection flags, and execution ordering.
- Modify: `tests/gui/test_gui_execution.py`
  - Cover operation-tab registration, parameter collection, and widget enablement.
- Create: `tests/visualization/test_volume_export.py`
  - Cover current-selection export, all-indices export, on-demand pyramid generation, TIFF layouts, and metadata.
- Modify: `tests/test_main.py`
  - Cover CLI/workflow handoff and dispatch/publication for `volume_export`.

## Task 1: Register The Workflow Contract

**Files:**
- Modify: `tests/test_workflow.py`
- Modify: `src/clearex/workflow.py`

- [ ] **Step 1: Write the failing workflow tests**

Add these tests near the existing visualization and `mip_export` workflow assertions in `tests/test_workflow.py`:

```python
def test_normalize_analysis_operation_parameters_includes_volume_export_defaults():
    normalized = normalize_analysis_operation_parameters(None)

    assert normalized["volume_export"]["execution_order"] == 12
    assert normalized["volume_export"]["input_source"] == "data"
    assert normalized["volume_export"]["export_scope"] == "current_selection"
    assert normalized["volume_export"]["t_index"] == 0
    assert normalized["volume_export"]["p_index"] == 0
    assert normalized["volume_export"]["c_index"] == 0
    assert normalized["volume_export"]["resolution_level"] == 0
    assert normalized["volume_export"]["export_format"] == "ome-zarr"
    assert normalized["volume_export"]["tiff_file_layout"] == "single_file"
    assert normalized["mip_export"]["execution_order"] == 13


def test_normalizes_volume_export_parameters():
    normalized = normalize_analysis_operation_parameters(
        {
            "volume_export": {
                "input_source": "fusion",
                "export_scope": "all_indices",
                "t_index": "7",
                "p_index": "8",
                "c_index": "9",
                "resolution_level": "2",
                "export_format": "OME-TIFF",
                "tiff_file_layout": "per_volume_files",
            }
        }
    )

    params = normalized["volume_export"]
    assert params["input_source"] == "fusion"
    assert params["export_scope"] == "all_indices"
    assert params["t_index"] == 7
    assert params["p_index"] == 8
    assert params["c_index"] == 9
    assert params["resolution_level"] == 2
    assert params["export_format"] == "ome-tiff"
    assert params["tiff_file_layout"] == "per_volume_files"


def test_resolve_analysis_execution_sequence_includes_volume_export_before_mip_export():
    sequence = resolve_analysis_execution_sequence(
        flatfield=False,
        deconvolution=False,
        shear_transform=False,
        fusion=False,
        particle_detection=False,
        usegment3d=False,
        registration=False,
        display_pyramid=True,
        visualization=True,
        render_movie=True,
        compile_movie=True,
        volume_export=True,
        mip_export=True,
        analysis_parameters=None,
    )

    assert sequence == (
        "display_pyramid",
        "visualization",
        "render_movie",
        "compile_movie",
        "volume_export",
        "mip_export",
    )
```

- [ ] **Step 2: Run the workflow tests to verify they fail**

Run:

```bash
uv run pytest -q tests/test_workflow.py -k volume_export
```

Expected: FAIL with `KeyError: 'volume_export'` or `TypeError` from missing `volume_export` parameters in `resolve_analysis_execution_sequence(...)`.

- [ ] **Step 3: Implement the workflow contract in `src/clearex/workflow.py`**

Make the following focused changes:

```python
ANALYSIS_OPERATION_ORDER = (
    "flatfield",
    "deconvolution",
    "shear_transform",
    "registration",
    "fusion",
    "display_pyramid",
    "particle_detection",
    "usegment3d",
    "visualization",
    "render_movie",
    "compile_movie",
    "volume_export",
    "mip_export",
)

ANALYSIS_KNOWN_OUTPUT_COMPONENTS["volume_export"] = "clearex/results/volume_export/latest"

DEFAULT_ANALYSIS_OPERATION_PARAMETERS["volume_export"] = {
    "execution_order": 12,
    "input_source": "data",
    "force_rerun": False,
    "chunk_basis": "3d",
    "detect_2d_per_slice": False,
    "use_map_overlap": False,
    "overlap_zyx": [0, 0, 0],
    "memory_overhead_factor": 1.0,
    "export_scope": "current_selection",
    "t_index": 0,
    "p_index": 0,
    "c_index": 0,
    "resolution_level": 0,
    "export_format": "ome-zarr",
    "tiff_file_layout": "single_file",
}

DEFAULT_ANALYSIS_OPERATION_PARAMETERS["mip_export"]["execution_order"] = 13
```

Add a dedicated normalizer:

```python
def _normalize_volume_export_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_common_operation_parameters("volume_export", params)
    normalized["input_source"] = _normalize_operation_input_source_alias(
        "volume_export",
        normalized.get("input_source", "data"),
    )
    normalized["chunk_basis"] = "3d"
    normalized["detect_2d_per_slice"] = False
    normalized["use_map_overlap"] = False
    normalized["overlap_zyx"] = [0, 0, 0]

    export_scope = (
        str(normalized.get("export_scope", "current_selection")).strip().lower()
        or "current_selection"
    )
    if export_scope not in {"current_selection", "all_indices"}:
        raise ValueError(
            "volume_export export_scope must be 'current_selection' or 'all_indices'."
        )
    normalized["export_scope"] = export_scope
    normalized["t_index"] = max(0, int(normalized.get("t_index", 0)))
    normalized["p_index"] = max(0, int(normalized.get("p_index", 0)))
    normalized["c_index"] = max(0, int(normalized.get("c_index", 0)))
    normalized["resolution_level"] = max(
        0, int(normalized.get("resolution_level", 0))
    )

    export_format_raw = (
        str(normalized.get("export_format", "ome-zarr")).strip().lower()
        or "ome-zarr"
    )
    if export_format_raw in {"ome-zarr", "ome_zarr", "ome.zarr", "zarr"}:
        normalized["export_format"] = "ome-zarr"
    elif export_format_raw in {"ome-tiff", "ome_tiff", "ome.tiff", "tiff"}:
        normalized["export_format"] = "ome-tiff"
    else:
        raise ValueError(
            "volume_export export_format must be one of: ome-zarr, zarr, ome-tiff, tiff."
        )

    tiff_file_layout = (
        str(normalized.get("tiff_file_layout", "single_file")).strip().lower()
        or "single_file"
    )
    if tiff_file_layout not in {"single_file", "per_volume_files"}:
        raise ValueError(
            "volume_export tiff_file_layout must be 'single_file' or 'per_volume_files'."
        )
    normalized["tiff_file_layout"] = tiff_file_layout
    return normalized
```

Wire it into `normalize_analysis_operation_parameters(...)`, update `selected_analysis_operations(...)`, update `resolve_analysis_execution_sequence(...)`, add `volume_export: bool = False` to `WorkflowConfig`, and include that flag in `has_analysis_selection()`.

Do **not** add `volume_export` to `ANALYSIS_CHAINABLE_OUTPUT_COMPONENTS`; it is a known output, not a preferred scientific upstream alias.

- [ ] **Step 4: Run the workflow tests to verify they pass**

Run:

```bash
uv run pytest -q tests/test_workflow.py -k volume_export
```

Expected: PASS with the new defaults, normalization rules, and execution ordering.

- [ ] **Step 5: Commit the workflow contract**

Run:

```bash
git add tests/test_workflow.py src/clearex/workflow.py
git commit -m "feat: register volume export workflow contract"
```

## Task 2: Add GUI Registration And Parameter Collection

**Files:**
- Modify: `tests/gui/test_gui_execution.py`
- Modify: `src/clearex/gui/app.py`

- [ ] **Step 1: Write the failing GUI tests**

Add these tests to `tests/gui/test_gui_execution.py` next to the existing visualization and `mip_export` checks:

```python
def test_analysis_selection_dialog_includes_volume_export_in_visualization_tab() -> None:
    dialog_cls = app_module.AnalysisSelectionDialog
    assert dialog_cls._OPERATION_LABELS["volume_export"] == "Volume Export"
    visualization_tab = dict(dialog_cls._OPERATION_TABS)["Visualization"]
    assert "volume_export" in visualization_tab


def test_collect_volume_export_parameters() -> None:
    dialog = app_module.AnalysisSelectionDialog.__new__(
        app_module.AnalysisSelectionDialog
    )
    dialog._volume_export_defaults = {"memory_overhead_factor": 1.0}
    dialog._volume_export_scope_combo = _FakeCombo("current_selection")
    dialog._volume_export_t_spin = _FakeSpin(2)
    dialog._volume_export_p_spin = _FakeSpin(1)
    dialog._volume_export_c_spin = _FakeSpin(3)
    dialog._volume_export_resolution_level_spin = _FakeSpin(4)
    dialog._volume_export_format_combo = _FakeCombo("ome-tiff")
    dialog._volume_export_tiff_layout_combo = _FakeCombo("per_volume_files")

    params = app_module.AnalysisSelectionDialog._collect_volume_export_parameters(
        dialog
    )

    assert params["export_scope"] == "current_selection"
    assert params["t_index"] == 2
    assert params["p_index"] == 1
    assert params["c_index"] == 3
    assert params["resolution_level"] == 4
    assert params["export_format"] == "ome-tiff"
    assert params["tiff_file_layout"] == "per_volume_files"
```

- [ ] **Step 2: Run the GUI tests to verify they fail**

Run:

```bash
uv run pytest -q tests/gui/test_gui_execution.py -k volume_export
```

Expected: FAIL with missing label/key/panel collector errors such as `KeyError: 'volume_export'` or `AttributeError: _collect_volume_export_parameters`.

- [ ] **Step 3: Implement GUI registration and parameter collection in `src/clearex/gui/app.py`**

Update the dialog registration maps:

```python
_OPERATION_KEYS = (
    "flatfield",
    "deconvolution",
    "shear_transform",
    "registration",
    "fusion",
    "display_pyramid",
    "particle_detection",
    "usegment3d",
    "visualization",
    "render_movie",
    "compile_movie",
    "volume_export",
    "mip_export",
)

_OPERATION_LABELS["volume_export"] = "Volume Export"

_OPERATION_TABS = tuple(
    (
        tab_name,
        (
            "visualization",
            "render_movie",
            "compile_movie",
            "volume_export",
            "mip_export",
        )
        if tab_name == "Visualization"
        else operation_names,
    )
    for tab_name, operation_names in _OPERATION_TABS
)

_OPERATION_OUTPUT_COMPONENTS["volume_export"] = "clearex/results/volume_export/latest"
```

Add a dedicated builder and collector:

```python
def _build_volume_export_parameter_rows(self, form: QFormLayout) -> None:
    self._volume_export_scope_combo = QComboBox()
    self._volume_export_scope_combo.addItem("Current t/p/c selection", "current_selection")
    self._volume_export_scope_combo.addItem("All time/position/channel indices", "all_indices")
    form.addRow("Export scope", self._volume_export_scope_combo)

    self._volume_export_t_spin = QSpinBox()
    self._volume_export_p_spin = QSpinBox()
    self._volume_export_c_spin = QSpinBox()
    self._volume_export_resolution_level_spin = QSpinBox()
    self._volume_export_format_combo = QComboBox()
    self._volume_export_format_combo.addItem("OME-Zarr", "ome-zarr")
    self._volume_export_format_combo.addItem("OME-TIFF", "ome-tiff")
    self._volume_export_tiff_layout_combo = QComboBox()
    self._volume_export_tiff_layout_combo.addItem("Single file", "single_file")
    self._volume_export_tiff_layout_combo.addItem("Per-volume files", "per_volume_files")
```

```python
def _collect_volume_export_parameters(self) -> Dict[str, Any]:
    return {
        "chunk_basis": "3d",
        "detect_2d_per_slice": False,
        "use_map_overlap": False,
        "overlap_zyx": [0, 0, 0],
        "memory_overhead_factor": float(
            self._volume_export_defaults.get("memory_overhead_factor", 1.0)
        ),
        "export_scope": str(
            self._volume_export_scope_combo.currentData() or "current_selection"
        ).strip(),
        "t_index": int(self._volume_export_t_spin.value()),
        "p_index": int(self._volume_export_p_spin.value()),
        "c_index": int(self._volume_export_c_spin.value()),
        "resolution_level": int(self._volume_export_resolution_level_spin.value()),
        "export_format": str(
            self._volume_export_format_combo.currentData() or "ome-zarr"
        ).strip(),
        "tiff_file_layout": str(
            self._volume_export_tiff_layout_combo.currentData() or "single_file"
        ).strip(),
    }
```

Wire the builder into the operation panel `elif` chain, initialize `self._volume_export_defaults`, restore the widget state from normalized parameters in the dialog population block, and call the collector from `_collect_operation_parameters(...)`.

- [ ] **Step 4: Add and wire widget enablement**

Implement a dedicated enablement helper and call it from `_on_operation_selection_changed(...)`:

```python
def _set_volume_export_parameter_enabled_state(self) -> None:
    export_enabled = self._operation_checkboxes["volume_export"].isChecked()
    current_selection = (
        str(self._volume_export_scope_combo.currentData() or "current_selection")
        == "current_selection"
    )
    using_tiff = (
        str(self._volume_export_format_combo.currentData() or "ome-zarr")
        == "ome-tiff"
    )

    for widget in (
        self._volume_export_scope_combo,
        self._volume_export_resolution_level_spin,
        self._volume_export_format_combo,
    ):
        widget.setEnabled(export_enabled)

    for widget in (
        self._volume_export_t_spin,
        self._volume_export_p_spin,
        self._volume_export_c_spin,
    ):
        widget.setEnabled(export_enabled and current_selection)

    self._volume_export_tiff_layout_combo.setEnabled(export_enabled and using_tiff)
```

- [ ] **Step 5: Run the GUI tests to verify they pass**

Run:

```bash
uv run pytest -q tests/gui/test_gui_execution.py -k volume_export
```

Expected: PASS with the new visualization-tab registration and collected parameter values.

- [ ] **Step 6: Commit the GUI changes**

Run:

```bash
git add tests/gui/test_gui_execution.py src/clearex/gui/app.py
git commit -m "feat: add volume export gui controls"
```

## Task 3: Implement Current-Selection OME-Zarr Export

**Files:**
- Create: `tests/visualization/test_volume_export.py`
- Create: `src/clearex/visualization/volume_export.py`
- Modify: `src/clearex/visualization/__init__.py`

- [ ] **Step 1: Write the failing current-selection export test**

Create `tests/visualization/test_volume_export.py` with this first test:

```python
from pathlib import Path

import numpy as np
import zarr

from clearex.io.ome_store import publish_analysis_collection_from_cache
from clearex.visualization.volume_export import run_volume_export_analysis


def test_run_volume_export_analysis_writes_current_selection_cache_and_publishable_ome_zarr(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "volume_export_current.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(
        (2, 2, 2, 3, 4, 5)
    )
    root.create_array(
        name="data",
        data=data,
        chunks=(1, 1, 1, 3, 4, 5),
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [4.0, 2.0, 2.0]

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 1,
            "p_index": 0,
            "c_index": 1,
            "resolution_level": 0,
            "export_format": "ome-zarr",
        },
        client=None,
    )

    exported = zarr.open_group(str(store_path), mode="r")[
        "clearex/runtime_cache/results/volume_export/latest/data"
    ]
    assert summary.component == "clearex/results/volume_export/latest"
    assert exported.shape == (1, 1, 1, 3, 4, 5)
    np.testing.assert_array_equal(exported[0, 0, 0], data[1, 0, 1])

    publish_analysis_collection_from_cache(store_path, analysis_name="volume_export")
    published = zarr.open_group(str(store_path), mode="r")[
        "results/volume_export/latest/A/1/0/0"
    ]
    assert published.shape == (1, 1, 3, 4, 5)
```

- [ ] **Step 2: Run the new volume-export test to verify it fails**

Run:

```bash
uv run pytest -q tests/visualization/test_volume_export.py::test_run_volume_export_analysis_writes_current_selection_cache_and_publishable_ome_zarr
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `clearex.visualization.volume_export` does not exist yet.

- [ ] **Step 3: Create the runtime module and implement current-selection export**

Create `src/clearex/visualization/volume_export.py` with a focused dataclass and entrypoint:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import dask.array as da
import numpy as np
import zarr

from clearex.io.ome_store import (
    analysis_auxiliary_root,
    analysis_cache_data_component,
    ensure_group,
)
from clearex.io.provenance import register_latest_output_reference
from clearex.workflow import normalize_analysis_operation_parameters

ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class VolumeExportSummary:
    component: str
    data_component: str
    source_component: str
    resolved_resolution_component: str
    export_scope: str
    resolution_level: int
    generated_resolution_level: bool
    export_format: str
    tiff_file_layout: str
    artifact_paths: tuple[str, ...]
```

Implement `run_volume_export_analysis(...)` so it:

- normalizes parameters via `normalize_analysis_operation_parameters({"volume_export": ...})["volume_export"]`
- resolves the source array and requested selection
- writes a canonical singleton `TPCZYX` array into `analysis_cache_data_component("volume_export")`
- stores metadata in `analysis_auxiliary_root("volume_export")`
- registers the latest output reference

Use this write pattern for the current-selection path:

```python
target = root.require_group("clearex/runtime_cache/results/volume_export/latest")
if "data" in target:
    del target["data"]
array = target.create_array(
    "data",
    shape=(1, 1, 1, z_size, y_size, x_size),
    chunks=(1, 1, 1, z_chunk, y_chunk, x_chunk),
    dtype=source.dtype,
    overwrite=True,
    dimension_names=("t", "p", "c", "z", "y", "x"),
)
source_view = da.from_zarr(str(zarr_path), component=resolved_component)[
    t_index : t_index + 1,
    p_index : p_index + 1,
    c_index : c_index + 1,
    :,
    :,
    :,
]
da.store(source_view, array, lock=False, compute=True)
array.attrs.update(
    {
        "axes": ["t", "p", "c", "z", "y", "x"],
        "_ARRAY_DIMENSIONS": ["t", "p", "c", "z", "y", "x"],
        "source_component": str(source_component),
        "resolution_level": int(resolution_level),
    }
)
```

- [ ] **Step 4: Expose the new runtime entrypoint from `src/clearex/visualization/__init__.py`**

Update `__all__` and `__getattr__` to include:

```python
__all__ = [
    "CompileMovieSummary",
    "DisplayPyramidSummary",
    "RenderMovieSummary",
    "VisualizationSummary",
    "VolumeExportSummary",
    "run_compile_movie_analysis",
    "run_display_pyramid_analysis",
    "run_render_movie_analysis",
    "run_visualization_analysis",
    "run_volume_export_analysis",
]
```

and resolve `VolumeExportSummary` / `run_volume_export_analysis` from `.volume_export`.

- [ ] **Step 5: Run the current-selection test to verify it passes**

Run:

```bash
uv run pytest -q tests/visualization/test_volume_export.py::test_run_volume_export_analysis_writes_current_selection_cache_and_publishable_ome_zarr
```

Expected: PASS with one singleton runtime-cache export and a publishable `results/volume_export/latest/A/1/0/0` field.

- [ ] **Step 6: Commit the runtime scaffold**

Run:

```bash
git add tests/visualization/test_volume_export.py src/clearex/visualization/volume_export.py src/clearex/visualization/__init__.py
git commit -m "feat: add volume export runtime scaffold"
```

## Task 4: Add All-Indices Export And On-Demand Resolution Generation

**Files:**
- Modify: `tests/visualization/test_volume_export.py`
- Modify: `src/clearex/visualization/volume_export.py`

- [ ] **Step 1: Write the failing all-indices and missing-level tests**

Append these tests to `tests/visualization/test_volume_export.py`:

```python
def test_run_volume_export_analysis_writes_all_indices_cache(tmp_path: Path) -> None:
    store_path = tmp_path / "volume_export_all_indices.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 3 * 2 * 4 * 5 * 6, dtype=np.uint16).reshape(
        (2, 3, 2, 4, 5, 6)
    )
    root.create_array(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 5, 6),
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [3.0, 1.5, 1.5]

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 0,
            "export_format": "ome-zarr",
        },
        client=None,
    )

    exported = zarr.open_group(str(store_path), mode="r")[
        "clearex/runtime_cache/results/volume_export/latest/data"
    ]
    assert summary.generated_resolution_level is False
    assert exported.shape == data.shape
    np.testing.assert_array_equal(exported[:], data)


def test_run_volume_export_analysis_builds_missing_resolution_level(tmp_path: Path) -> None:
    store_path = tmp_path / "volume_export_build_level.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 2 * 1 * 4 * 8 * 8, dtype=np.uint16).reshape(
        (1, 2, 1, 4, 8, 8)
    )
    root.create_array(
        name="data",
        data=data,
        chunks=(1, 1, 1, 2, 4, 4),
        overwrite=True,
    )
    root["data"].attrs["voxel_size_um_zyx"] = [4.0, 1.0, 1.0]

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 1,
            "export_format": "ome-zarr",
        },
        client=None,
    )

    root_r = zarr.open_group(str(store_path), mode="r")
    assert summary.generated_resolution_level is True
    assert "data_pyramid/level_1" in root_r
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
uv run pytest -q tests/visualization/test_volume_export.py -k "all_indices or missing_resolution_level"
```

Expected: FAIL because the runtime only handles the singleton current-selection path and does not yet resolve or build a requested resolution level.

- [ ] **Step 3: Implement all-indices export and resolution-level resolution**

Add helpers to `src/clearex/visualization/volume_export.py` that:

- resolve existing source-adjacent levels from source attrs such as `display_pyramid_levels`, `pyramid_levels`, and legacy compatibility attrs
- call `run_display_pyramid_analysis(...)` when the requested `resolution_level` is missing
- return both the resolved component path and a `generated_resolution_level` boolean

Use this pattern:

```python
from clearex.visualization.pipeline import run_display_pyramid_analysis


def _resolve_or_build_resolution_component(
    *,
    zarr_path: Union[str, Path],
    source_component: str,
    resolution_level: int,
) -> tuple[str, bool]:
    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        available = _available_resolution_components(root, source_component)
    finally:
        _close_zarr_store(root)

    if resolution_level < len(available):
        return str(available[resolution_level]), False

    run_display_pyramid_analysis(
        zarr_path=zarr_path,
        parameters={"input_source": source_component, "force_rerun": False},
        progress_callback=None,
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        available = _available_resolution_components(root, source_component)
    finally:
        _close_zarr_store(root)
    if resolution_level >= len(available):
        raise ValueError(
            f"volume_export could not resolve resolution_level={resolution_level} for {source_component!r}."
        )
    return str(available[resolution_level]), True
```

For `all_indices`, export the entire resolved component into the runtime cache without slicing:

```python
source_view = da.from_zarr(str(zarr_path), component=resolved_component)
da.store(source_view, array, lock=False, compute=True)
```

- [ ] **Step 4: Run the all-indices tests to verify they pass**

Run:

```bash
uv run pytest -q tests/visualization/test_volume_export.py -k "all_indices or missing_resolution_level"
```

Expected: PASS with both full-shape export and automatic creation of `data_pyramid/level_1`.

- [ ] **Step 5: Commit the resolution-level support**

Run:

```bash
git add tests/visualization/test_volume_export.py src/clearex/visualization/volume_export.py
git commit -m "feat: add multiscale volume export support"
```

## Task 5: Add TIFF Artifact Layouts

**Files:**
- Modify: `tests/visualization/test_volume_export.py`
- Modify: `src/clearex/visualization/volume_export.py`

- [ ] **Step 1: Write the failing TIFF tests**

Add these tests to `tests/visualization/test_volume_export.py`:

```python
import tifffile


def test_run_volume_export_analysis_writes_current_selection_tiff(tmp_path: Path) -> None:
    store_path = tmp_path / "volume_export_tiff_current.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 1 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape((1, 1, 1, 3, 4, 5))
    root.create_array(name="data", data=data, chunks=(1, 1, 1, 3, 4, 5), overwrite=True)
    root["data"].attrs["voxel_size_um_zyx"] = [5.0, 2.0, 2.0]

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "current_selection",
            "t_index": 0,
            "p_index": 0,
            "c_index": 0,
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "single_file",
        },
        client=None,
    )

    [artifact_path] = summary.artifact_paths
    assert Path(artifact_path).exists()
    image = tifffile.imread(artifact_path)
    assert image.shape == (3, 4, 5)


def test_run_volume_export_analysis_writes_all_indices_single_file_tiff(tmp_path: Path) -> None:
    store_path = tmp_path / "volume_export_tiff_series.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(2 * 2 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 2, 1, 3, 4, 5))
    root.create_array(name="data", data=data, chunks=(1, 1, 1, 3, 4, 5), overwrite=True)
    root["data"].attrs["voxel_size_um_zyx"] = [6.0, 2.0, 2.0]

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "single_file",
        },
        client=None,
    )

    [artifact_path] = summary.artifact_paths
    with tifffile.TiffFile(artifact_path) as tif:
        assert len(tif.series) == 2
        assert tif.series[0].shape == (2, 1, 3, 4, 5)


def test_run_volume_export_analysis_writes_all_indices_per_volume_tiffs(tmp_path: Path) -> None:
    store_path = tmp_path / "volume_export_tiff_per_volume.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    data = np.arange(1 * 2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape((1, 2, 2, 3, 4, 5))
    root.create_array(name="data", data=data, chunks=(1, 1, 1, 3, 4, 5), overwrite=True)
    root["data"].attrs["voxel_size_um_zyx"] = [3.0, 1.0, 1.0]

    summary = run_volume_export_analysis(
        zarr_path=store_path,
        parameters={
            "input_source": "data",
            "export_scope": "all_indices",
            "resolution_level": 0,
            "export_format": "ome-tiff",
            "tiff_file_layout": "per_volume_files",
        },
        client=None,
    )

    assert len(summary.artifact_paths) == 4
    assert all(Path(path).exists() for path in summary.artifact_paths)
```

- [ ] **Step 2: Run the TIFF tests to verify they fail**

Run:

```bash
uv run pytest -q tests/visualization/test_volume_export.py -k tiff
```

Expected: FAIL because `artifact_paths` is empty and the runtime does not yet emit TIFF files.

- [ ] **Step 3: Implement TIFF writers in `src/clearex/visualization/volume_export.py`**

Add TIFF helpers that write into `clearex/results/volume_export/latest/files`:

```python
import tifffile


def _artifact_directory(store_path: Union[str, Path]) -> Path:
    return (
        Path(store_path).expanduser().resolve()
        / "clearex/results/volume_export/latest/files"
    )


def _write_current_selection_tiff(*, source_array: da.Array, output_path: Path) -> str:
    volume = np.asarray(source_array[0, 0, 0].compute())
    tifffile.imwrite(
        str(output_path),
        volume,
        ome=True,
        bigtiff=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    return str(output_path)
```

For all-position single-file TIFF, write one `TCZYX` series per position:

```python
with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
    for p_index in range(int(source_array.shape[1])):
        position_payload = np.asarray(source_array[:, p_index, :, :, :, :].compute())
        tif.write(
            position_payload,
            ome=True,
            metadata={"axes": "TCZYX"},
        )
```

For per-volume TIFFs, loop over `t/p/c` and write one `ZYX` file each. Update `artifact_paths` in `VolumeExportSummary` and persist those paths in `clearex/results/volume_export/latest`.

Use the resolved resolution-level voxel spacing when filling `PhysicalSizeZ/Y/X`; do **not** hard-code `1.0` in the final implementation.

- [ ] **Step 4: Run the TIFF tests to verify they pass**

Run:

```bash
uv run pytest -q tests/visualization/test_volume_export.py -k tiff
```

Expected: PASS with one current-selection OME-TIFF, one all-position multi-series BigTIFF, and one file per exported `(t, p, c)` volume when requested.

- [ ] **Step 5: Commit the TIFF export support**

Run:

```bash
git add tests/visualization/test_volume_export.py src/clearex/visualization/volume_export.py
git commit -m "feat: add volume export tiff outputs"
```

## Task 6: Wire CLI, Main Dispatch, And Publication

**Files:**
- Modify: `tests/test_main.py`
- Modify: `src/clearex/io/cli.py`
- Modify: `src/clearex/main.py`

- [ ] **Step 1: Write the failing dispatch and CLI-handoff tests**

Add these tests to `tests/test_main.py`:

```python
def test_build_workflow_config_propagates_volume_export_flag() -> None:
    parser = main_module.create_parser()
    args = parser.parse_args(["--file", "/tmp/data_store.zarr", "--volume-export"])

    workflow = main_module._build_workflow_config(args)

    assert workflow.volume_export is True


def test_run_workflow_dispatches_volume_export_and_publishes_collection(
    tmp_path: Path, monkeypatch
) -> None:
    store_path = tmp_path / "analysis_store_volume_export.zarr"
    root = main_module.zarr.open_group(str(store_path), mode="w")
    root.create_dataset(
        name=main_module.SOURCE_CACHE_COMPONENT,
        shape=(1, 1, 1, 2, 2, 2),
        chunks=(1, 1, 1, 2, 2, 2),
        dtype="uint16",
        overwrite=True,
    )

    workflow = WorkflowConfig(
        file=str(store_path),
        volume_export=True,
        analysis_parameters={
            "volume_export": {
                "input_source": "data",
                "export_scope": "current_selection",
                "t_index": 0,
                "p_index": 0,
                "c_index": 0,
                "resolution_level": 0,
                "export_format": "ome-zarr",
                "tiff_file_layout": "single_file",
            }
        },
    )

    published = {"analysis_name": None}
    called = {"value": False}

    def _fake_volume_export(*, zarr_path, parameters, client, progress_callback, run_id=None):
        del zarr_path, parameters, client, progress_callback, run_id
        called["value"] = True
        return SimpleNamespace(
            component=main_module.analysis_auxiliary_root("volume_export"),
            data_component="clearex/runtime_cache/results/volume_export/latest/data",
            source_component="data",
            resolved_resolution_component="data",
            export_scope="current_selection",
            resolution_level=0,
            generated_resolution_level=False,
            export_format="ome-zarr",
            tiff_file_layout="single_file",
            artifact_paths=(),
        )

    def _fake_publish(zarr_path, analysis_name):
        del zarr_path
        published["analysis_name"] = analysis_name

    monkeypatch.setattr(main_module, "run_volume_export_analysis", _fake_volume_export)
    monkeypatch.setattr(
        main_module, "publish_analysis_collection_from_cache", _fake_publish
    )
    monkeypatch.setattr(main_module, "is_navigate_experiment_file", lambda path: False)
    monkeypatch.setattr(main_module, "is_legacy_clearex_store", lambda path: False)

    main_module._run_workflow(
        workflow=workflow,
        logger=_test_logger("clearex.test.main.volume_export"),
    )

    assert called["value"] is True
    assert published["analysis_name"] == "volume_export"
```

- [ ] **Step 2: Run the dispatch tests to verify they fail**

Run:

```bash
uv run pytest -q tests/test_main.py -k volume_export
```

Expected: FAIL with parser errors for the unknown `--volume-export` flag or `AttributeError` because `run_volume_export_analysis` is not imported or dispatched from `main.py`.

- [ ] **Step 3: Implement CLI handoff and main dispatch**

Add the CLI flag in `src/clearex/io/cli.py`:

```python
input_args.add_argument(
    "--volume-export",
    required=False,
    default=False,
    action="store_true",
    help="Export full 3D volumes from a selected image-producing source",
)
```

Propagate the flag in `_build_workflow_config(...)` in `src/clearex/main.py`:

```python
return WorkflowConfig(
    visualization=args.visualization,
    render_movie=bool(getattr(args, "render_movie", False)),
    compile_movie=bool(getattr(args, "compile_movie", False)),
    volume_export=bool(getattr(args, "volume_export", False)),
    mip_export=args.mip_export,
    analysis_parameters=usegment3d_analysis_parameters,
)
```

Import and dispatch the runtime:

```python
from clearex.visualization.volume_export import run_volume_export_analysis
```

```python
if operation_name == "volume_export":
    volume_export_parameters = dict(operation_parameters)
    if provenance_store_path and is_zarr_store_path(provenance_store_path):

        def _volume_export_progress(percent: int, message: str) -> None:
            mapped = operation_start + int(
                (max(0, min(100, int(percent))) / 100)
                * max(1, operation_end - operation_start)
            )
            logger.info(f"[volume_export] {int(percent)}% - {message}")
            _emit_analysis_progress(mapped, f"volume_export: {message}")

        summary = run_volume_export_analysis(
            zarr_path=provenance_store_path,
            parameters=volume_export_parameters,
            client=analysis_client,
            progress_callback=_volume_export_progress,
            run_id=run_id,
        )
        publish_analysis_collection_from_cache(
            provenance_store_path,
            analysis_name="volume_export",
        )
        output_records["volume_export"] = {
            "component": summary.component,
            "data_component": summary.data_component,
            "source_component": summary.source_component,
            "resolved_resolution_component": summary.resolved_resolution_component,
            "export_scope": summary.export_scope,
            "resolution_level": summary.resolution_level,
            "generated_resolution_level": summary.generated_resolution_level,
            "export_format": summary.export_format,
            "tiff_file_layout": summary.tiff_file_layout,
            "artifact_paths": list(summary.artifact_paths),
            "storage_policy": "latest_only",
        }
```

Also update `_ANALYSIS_PROVENANCE_REQUIRED_COMPONENTS` so it includes:

```python
"volume_export": (
    analysis_cache_data_component("volume_export"),
    analysis_auxiliary_root("volume_export"),
),
```

- [ ] **Step 4: Run the main tests to verify they pass**

Run:

```bash
uv run pytest -q tests/test_main.py -k volume_export
```

Expected: PASS with both the parser handoff and dispatch/publication path working.

- [ ] **Step 5: Commit the main/CLI integration**

Run:

```bash
git add tests/test_main.py src/clearex/io/cli.py src/clearex/main.py
git commit -m "feat: wire volume export through cli and main"
```

## Task 7: Refresh Documentation And Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `src/clearex/AGENTS.md`
- Modify: `src/clearex/visualization/README.md`
- Modify: `docs/source/runtime/cli-and-execution.rst`
- Modify: `docs/source/runtime/architecture-overview.rst`

- [ ] **Step 1: Update the user-facing and agent-facing docs**

Update the docs with concrete wording that matches the implemented contract:

```md
- `volume_export` exports one selected image-producing source as a full 3D volume.
- `export_scope=current_selection` writes one explicit `(t, p, c)` preview export.
- `export_scope=all_indices` writes the full selected source across all `t/p/c`.
- `resolution_level` reuses existing helper levels when present and builds them on demand when missing.
- OME-Zarr publication lands at `results/volume_export/latest`.
- TIFF artifacts land at `clearex/results/volume_export/latest/files`.
```

Make these doc-specific edits:

- `README.md`
  - add `volume_export` to the feature list and storage/output descriptions
- `src/clearex/AGENTS.md`
  - add one runtime update bullet for `volume_export`
- `src/clearex/visualization/README.md`
  - describe the new operation, its scope modes, and TIFF layouts
- `docs/source/runtime/cli-and-execution.rst`
  - add `--volume-export` to the CLI list and output locations
- `docs/source/runtime/architecture-overview.rst`
  - include `volume_export` in the workflow map

- [ ] **Step 2: Run the targeted verification suite**

Run:

```bash
uv run pytest -q tests/test_workflow.py -k volume_export
uv run pytest -q tests/gui/test_gui_execution.py -k volume_export
uv run pytest -q tests/visualization/test_volume_export.py
uv run pytest -q tests/test_main.py -k volume_export
uv run ruff check src/clearex/workflow.py src/clearex/gui/app.py src/clearex/io/cli.py src/clearex/main.py src/clearex/visualization/__init__.py src/clearex/visualization/volume_export.py tests/test_workflow.py tests/gui/test_gui_execution.py tests/visualization/test_volume_export.py tests/test_main.py
uv run basedpyright src/clearex/workflow.py src/clearex/gui/app.py src/clearex/main.py src/clearex/visualization/volume_export.py
```

Expected:

- all four targeted pytest commands: PASS
- `ruff check`: `All checks passed!`
- `basedpyright`: `0 errors, 0 warnings`

- [ ] **Step 3: Run one broader regression check for neighboring workflows**

Run:

```bash
uv run pytest -q tests/visualization/test_pipeline.py tests/mip_export/test_pipeline.py
```

Expected: PASS, proving the new visualization-family export did not regress movie rendering, display pyramids, or MIP export.

- [ ] **Step 4: Commit the docs and final verification state**

Run:

```bash
git add README.md src/clearex/AGENTS.md src/clearex/visualization/README.md docs/source/runtime/cli-and-execution.rst docs/source/runtime/architecture-overview.rst
git commit -m "docs: document volume export workflow"
```
