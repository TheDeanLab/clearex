# Dask Backend Dialog Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the `Edit Dask Backend` popup footer visible at all times and add a hover/focus-driven `Parameter Help` card that explains backend settings in plain language.

**Architecture:** Refactor `DaskBackendConfigDialog` so only its settings body scrolls, while the help card and footer live in fixed bottom regions owned by the outer dialog layout. Reuse the analysis dialog's existing hover/focus parameter-help pattern locally inside the popup, with dialog-scoped help registration and event filtering instead of introducing a separate help system.

**Tech Stack:** PyQt6 widgets/layouts in `src/clearex/gui/app.py`, GUI regression tests in `tests/gui/test_gui_execution.py`, subsystem docs in `src/clearex/gui/README.md`.

---

## File Map

- Modify: `src/clearex/gui/app.py`
  - Refactor `DaskBackendConfigDialog._build_ui()` so the scroll area contains only the settings body.
  - Add a fixed `Parameter Help` frame below the scroll area.
  - Add dialog-local hover/focus help registration helpers and an `eventFilter(...)`.
  - Register plain-language help text for backend mode and all backend-specific fields.
- Modify: `tests/gui/test_gui_execution.py`
  - Extend the Dask backend dialog regression coverage for fixed footer placement and help-card visibility behavior.
- Modify: `src/clearex/gui/README.md`
  - Document the fixed footer and hover/focus-driven help behavior for the Dask backend popup.

## Task 1: Lock In the Popup Structure With a Failing Layout Test

**Files:**
- Modify: `tests/gui/test_gui_execution.py:946-976`
- Modify: `src/clearex/gui/app.py:3667-3764`
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Write the failing test**

Add a new test immediately after `test_dask_dialog_scrolls_body_on_short_screens(...)` that asserts the fixed footer lives outside the scrollable content and remains present when the body scrolls.

```python
def test_dask_dialog_keeps_footer_outside_scroll_area(monkeypatch) -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    monkeypatch.setattr(
        app_module, "_primary_screen_available_size", lambda: (800, 800)
    )

    dialog = app_module.DaskBackendConfigDialog(
        initial=app_module.DaskBackendConfig(),
        recommendation_shape_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_chunks_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_dtype_itemsize=2,
    )
    dialog.show()
    app.processEvents()

    scroll = dialog.findChild(app_module.QScrollArea, "popupDialogScroll")
    assert scroll is not None
    scroll_widget = scroll.widget()
    assert scroll_widget is not None

    assert dialog._defaults_button.parentWidget() is dialog
    assert dialog._cancel_button.parentWidget() is dialog
    assert dialog._apply_button.parentWidget() is dialog
    assert dialog._defaults_button.parentWidget() is not scroll_widget
    assert dialog._cancel_button.parentWidget() is not scroll_widget
    assert dialog._apply_button.parentWidget() is not scroll_widget

    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    app.processEvents()

    assert dialog._apply_button.isVisible()
    assert dialog._apply_button.geometry().height() >= 36

    dialog.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog_keeps_footer_outside_scroll_area" -q`

Expected: FAIL because the current dialog builds the footer inside `popupDialogContent`, so the buttons are parented under the scroll widget rather than the dialog root.

- [ ] **Step 3: Write minimal implementation**

Refactor `DaskBackendConfigDialog._build_ui()` so the outer dialog layout owns the scroll area and footer separately. Keep the existing footer row construction, but attach it to a fixed `footer_frame` below the scroll area instead of adding it to the scrolled `root` layout.

Use this shape inside `src/clearex/gui/app.py`:

```python
outer_root = QVBoxLayout(self)
outer_root.setContentsMargins(0, 0, 0, 0)
outer_root.setSpacing(0)

self._content_scroll = QScrollArea(self)
self._content_scroll.setObjectName("popupDialogScroll")
self._content_scroll.setWidgetResizable(True)
self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
self._content_scroll.setHorizontalScrollBarPolicy(
    Qt.ScrollBarPolicy.ScrollBarAlwaysOff
)
outer_root.addWidget(self._content_scroll, 1)

content_widget = QWidget()
content_widget.setObjectName("popupDialogContent")
self._content_scroll.setWidget(content_widget)

root = QVBoxLayout(content_widget)
apply_popup_root_spacing(root)

# existing overview, mode selector, mode-help label, and mode stack stay here

footer_frame = QFrame(self)
footer_frame.setObjectName("analysisFooterCard")
footer_frame.setSizePolicy(
    QSizePolicy.Policy.Preferred,
    QSizePolicy.Policy.Fixed,
)
footer_root = QVBoxLayout(footer_frame)
footer_root.setContentsMargins(0, 0, 0, 0)
footer_root.setSpacing(0)

footer = QHBoxLayout()
apply_footer_row_spacing(footer)
self._defaults_button = _configure_fixed_height_button(
    QPushButton("Reset Defaults")
)
self._cancel_button = _configure_fixed_height_button(QPushButton("Cancel"))
self._apply_button = _configure_fixed_height_button(QPushButton("Apply"))
self._apply_button.setObjectName("runButton")
footer.addWidget(self._defaults_button)
footer.addStretch(1)
footer.addWidget(self._cancel_button)
footer.addWidget(self._apply_button)
footer_root.addLayout(footer)
outer_root.addWidget(footer_frame, 0)
```

Also remove the old `root.addLayout(footer)` block from the scroll content so the footer is not duplicated.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog_keeps_footer_outside_scroll_area" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/gui/test_gui_execution.py src/clearex/gui/app.py
git commit -m "refactor: move dask backend dialog footer outside scroll area"
```

## Task 2: Add a Failing Help-Card Visibility Test

**Files:**
- Modify: `tests/gui/test_gui_execution.py`
- Modify: `src/clearex/gui/app.py`
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Write the failing test**

Add a test that verifies the help card starts hidden, appears on focus, and hides again on focus-out for a representative registered widget such as `Workers`.

```python
def test_dask_dialog_parameter_help_shows_on_focus_and_hides_on_blur() -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    dialog = app_module.DaskBackendConfigDialog(
        initial=app_module.DaskBackendConfig(),
        recommendation_shape_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_chunks_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_dtype_itemsize=2,
    )
    dialog.show()
    app.processEvents()

    assert dialog._parameter_help_card is not None
    assert dialog._parameter_help_label is not None
    assert dialog._parameter_help_card.isHidden()

    dialog._local_workers_input.setFocus()
    app.processEvents()

    assert dialog._parameter_help_card.isVisible()
    assert "how many separate Dask workers" in dialog._parameter_help_label.text()

    dialog._apply_button.setFocus()
    app.processEvents()

    assert dialog._parameter_help_card.isHidden()

    dialog.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog_parameter_help_shows_on_focus_and_hides_on_blur" -q`

Expected: FAIL because the dialog currently has no `_parameter_help_card`, no registered help map, and no event-filter-based help behavior.

- [ ] **Step 3: Write minimal implementation**

Add dialog-local help state in `DaskBackendConfigDialog.__init__` and `_build_ui()`:

```python
self._parameter_help_map: Dict[QObject, str] = {}
self._parameter_help_card: Optional[QFrame] = None
self._parameter_help_label: Optional[QLabel] = None
```

Insert a fixed help card between the scroll area and footer:

```python
self._parameter_help_card = QFrame(self)
self._parameter_help_card.setObjectName("helpCard")
help_layout = QVBoxLayout(self._parameter_help_card)
apply_help_stack_spacing(help_layout)
help_title = QLabel("Parameter Help")
help_title.setObjectName("helpTitle")
help_layout.addWidget(help_title)
self._parameter_help_label = QLabel("")
self._parameter_help_label.setObjectName("helpBody")
self._parameter_help_label.setWordWrap(True)
help_layout.addWidget(self._parameter_help_label)
self._parameter_help_card.hide()
outer_root.addWidget(self._parameter_help_card, 0)
```

Add dialog-local helpers:

```python
def _register_parameter_hint(self, widget: QWidget, message: str) -> None:
    widget.setToolTip(message)
    widget.installEventFilter(self)
    self._parameter_help_map[widget] = str(message)

def _set_parameter_help(self, text: str) -> None:
    if self._parameter_help_label is not None:
        self._parameter_help_label.setText(str(text))

def _show_parameter_help(self, text: str) -> None:
    self._set_parameter_help(text)
    if self._parameter_help_card is not None:
        self._parameter_help_card.show()

def _hide_parameter_help(self) -> None:
    if self._parameter_help_card is not None:
        self._parameter_help_card.hide()
```

Implement `eventFilter(...)` on the dialog using the same focus fallback pattern as the analysis dialog:

```python
def eventFilter(self, watched: QObject, event: Optional[QEvent]) -> bool:
    message = self._parameter_help_map.get(watched)
    if message and event is not None:
        event_type = event.type()
        if event_type in (QEvent.Type.Enter, QEvent.Type.FocusIn):
            self._show_parameter_help(message)
        elif event_type in (QEvent.Type.Leave, QEvent.Type.FocusOut):
            focus_widget = self.focusWidget()
            if focus_widget is not None and focus_widget in self._parameter_help_map:
                self._show_parameter_help(self._parameter_help_map[focus_widget])
            else:
                self._hide_parameter_help()
    return super().eventFilter(watched, event)
```

Register the `Workers` input and other fields in later tasks; for this task, register at least `self._local_workers_input` so the test can pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog_parameter_help_shows_on_focus_and_hides_on_blur" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/gui/test_gui_execution.py src/clearex/gui/app.py
git commit -m "feat: add contextual help card to dask backend dialog"
```

## Task 3: Register Plain-Language Help for Every Dask Backend Setting

**Files:**
- Modify: `src/clearex/gui/app.py`
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Write the failing test**

Add a test that checks representative fields across all three backend modes are registered with plain-language tooltip/help text.

```python
def test_dask_dialog_registers_plain_language_help_for_all_backend_modes() -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    dialog = app_module.DaskBackendConfigDialog(
        initial=app_module.DaskBackendConfig(),
        recommendation_shape_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_chunks_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_dtype_itemsize=2,
    )

    expected_widgets = [
        dialog._mode_combo,
        dialog._local_workers_input,
        dialog._local_threads_spin,
        dialog._local_memory_input,
        dialog._runner_scheduler_file_input,
        dialog._runner_wait_workers_spin,
        dialog._cluster_workers_spin,
        dialog._cluster_dashboard_input,
        dialog._cluster_allowed_failures_spin,
    ]

    for widget in expected_widgets:
        assert widget in dialog._parameter_help_map
        message = dialog._parameter_help_map[widget]
        assert isinstance(message, str)
        assert message.strip()
        assert widget.toolTip() == message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "registers_plain_language_help_for_all_backend_modes" -q`

Expected: FAIL because only the representative `Workers` field from Task 2 is registered so far.

- [ ] **Step 3: Write minimal implementation**

Add a local help-text mapping helper in `DaskBackendConfigDialog`, for example:

```python
def _dask_backend_parameter_help_texts(self) -> dict[str, str]:
    return {
        "mode": (
            "Choose how ClearEx gets Dask workers. LocalCluster starts workers "
            "on this machine, SLURMRunner attaches to an existing scheduler "
            "file, and SLURMCluster submits new scheduler and worker jobs."
        ),
        "local_workers": (
            "How many separate Dask workers ClearEx starts on this machine. "
            "Leave blank to let ClearEx choose automatically."
        ),
        "local_threads_per_worker": (
            "How many threads each local worker uses inside one process. "
            "Higher values increase shared-memory concurrency but can raise "
            "contention for CPU-heavy tasks."
        ),
        "local_memory_limit": (
            "Per-worker memory cap for the local cluster. Dask uses this limit "
            "to decide when to spill data or restart an overloaded worker."
        ),
        "runner_scheduler_file": (
            "Path to the Dask scheduler file created by a scheduler you started "
            "outside this dialog. ClearEx uses it to connect to that cluster."
        ),
        "runner_wait_for_workers": (
            "How many workers ClearEx waits for before it starts work. Use "
            "auto to accept the backend default."
        ),
        "cluster_dashboard_address": (
            "Address where the scheduler binds its dashboard service. This is "
            "the scheduler-side bind setting, not the localhost relay URL that "
            "the GUI opens in your browser."
        ),
        "cluster_allowed_failures": (
            "How many worker failures the scheduler tolerates before the run is "
            "aborted."
        ),
    }
```

Register all settings immediately after widget creation, for example:

```python
self._register_parameter_hint(
    self._mode_combo,
    help_texts["mode"],
)
self._register_parameter_hint(
    self._local_workers_input,
    help_texts["local_workers"],
)
self._register_parameter_hint(
    self._local_threads_spin,
    help_texts["local_threads_per_worker"],
)
```

Repeat this for:

- `self._mode_combo`
- `self._local_workers_input`
- `self._local_threads_spin`
- `self._local_memory_input`
- `self._local_directory_input`
- `self._local_directory_browse`
- `self._local_recommend_button`
- `self._runner_scheduler_file_input`
- `self._runner_scheduler_file_browse`
- `self._runner_wait_workers_spin`
- `self._cluster_workers_spin`
- `self._cluster_cores_spin`
- `self._cluster_processes_spin`
- `self._cluster_memory_input`
- `self._cluster_local_directory_input`
- `self._cluster_local_directory_browse`
- `self._cluster_interface_input`
- `self._cluster_walltime_input`
- `self._cluster_job_name_input`
- `self._cluster_queue_input`
- `self._cluster_death_timeout_input`
- `self._cluster_mail_user_input`
- `self._cluster_directives_input`
- `self._cluster_dashboard_input`
- `self._cluster_scheduler_interface_input`
- `self._cluster_idle_timeout_input`
- `self._cluster_allowed_failures_spin`

Keep the wording plain-language and action-oriented rather than Dask-internal.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "registers_plain_language_help_for_all_backend_modes" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/clearex/gui/app.py tests/gui/test_gui_execution.py
git commit -m "feat: document dask backend settings with contextual help"
```

## Task 4: Cover Hover Behavior and Regression-Proof the Help Card

**Files:**
- Modify: `tests/gui/test_gui_execution.py`
- Modify: `src/clearex/gui/app.py`
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Write the failing test**

Add a hover-driven test to prove the help card responds to mouse entry/leave, not just keyboard focus.

```python
def test_dask_dialog_parameter_help_shows_on_hover() -> None:
    if not app_module.HAS_PYQT6:
        return

    app = app_module.QApplication.instance()
    if app is None:
        app = app_module.QApplication([])

    dialog = app_module.DaskBackendConfigDialog(
        initial=app_module.DaskBackendConfig(),
        recommendation_shape_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_chunks_tpczyx=(1, 1, 1, 64, 64, 64),
        recommendation_dtype_itemsize=2,
    )
    dialog.show()
    app.processEvents()

    enter_event = app_module.QEvent(app_module.QEvent.Type.Enter)
    leave_event = app_module.QEvent(app_module.QEvent.Type.Leave)

    dialog.eventFilter(dialog._cluster_dashboard_input, enter_event)
    app.processEvents()

    assert dialog._parameter_help_card.isVisible()
    assert "scheduler binds its dashboard service" in dialog._parameter_help_label.text()

    dialog.eventFilter(dialog._cluster_dashboard_input, leave_event)
    app.processEvents()

    assert dialog._parameter_help_card.isHidden()

    dialog.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "parameter_help_shows_on_hover" -q`

Expected: FAIL if hover handling is incomplete or the card stays visible after `Leave`.

- [ ] **Step 3: Write minimal implementation**

Adjust the dialog event filter only if needed so hover and focus share the same behavior without flicker:

```python
if event_type in (QEvent.Type.Enter, QEvent.Type.FocusIn):
    self._show_parameter_help(message)
elif event_type in (QEvent.Type.Leave, QEvent.Type.FocusOut):
    focus_widget = self.focusWidget()
    if focus_widget is not None and focus_widget in self._parameter_help_map:
        self._show_parameter_help(self._parameter_help_map[focus_widget])
    else:
        self._hide_parameter_help()
```

Do not add custom timers or delayed transitions unless the new test proves they are necessary.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "parameter_help_shows_on_hover" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/gui/test_gui_execution.py src/clearex/gui/app.py
git commit -m "test: cover hover-driven help in dask backend dialog"
```

## Task 5: Update the Existing Short-Screen Dialog Test for the New Bottom Region

**Files:**
- Modify: `tests/gui/test_gui_execution.py:946-976`
- Modify: `src/clearex/gui/app.py`
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Write the failing test**

Extend `test_dask_dialog_scrolls_body_on_short_screens(...)` so it asserts the new help card exists and remains outside the scroll body even on short screens.

```python
assert dialog._parameter_help_card is not None
assert dialog._parameter_help_label is not None
assert dialog._parameter_help_card.parentWidget() is dialog
assert dialog._parameter_help_card.parentWidget() is not scroll.widget()
assert dialog._parameter_help_card.isHidden()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog_scrolls_body_on_short_screens" -q`

Expected: FAIL until the test is updated and the card exists at dialog scope.

- [ ] **Step 3: Write minimal implementation**

If needed, make small geometry/styling adjustments so the fixed help card and fixed footer both render correctly on short screens. Keep the existing `content_widget.setMinimumHeight(root.sizeHint().height())` logic for the scroll body so the body remains scrollable.

If the new fixed bottom region needs stable sizing, set the help card size policy explicitly:

```python
self._parameter_help_card.setSizePolicy(
    QSizePolicy.Policy.Preferred,
    QSizePolicy.Policy.Fixed,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog_scrolls_body_on_short_screens" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/gui/test_gui_execution.py src/clearex/gui/app.py
git commit -m "test: verify dask backend help card on short screens"
```

## Task 6: Document the New Popup Behavior

**Files:**
- Modify: `src/clearex/gui/README.md`
- Test: none

- [ ] **Step 1: Write the documentation change**

Update the `Setup window` or popup guidance section in `src/clearex/gui/README.md` to describe the Dask backend popup behavior in the same style as existing GUI notes.

Add text like:

```md
- `Edit Dask Backend` popup:
  - only the backend settings body scrolls
  - `Reset Defaults`, `Cancel`, and `Apply` stay visible in a fixed bottom footer
  - a fixed `Parameter Help` card appears below the scroll area when a backend
    control is hovered or focused
  - help text is written in plain language for non-expert users
```

- [ ] **Step 2: Review for consistency**

Read the surrounding section and confirm the new bullets do not contradict the existing `Parameter Help UX` guidance or popup spacing rules.

Expected: the README continues to describe a single in-panel help pattern rather than two competing systems.

- [ ] **Step 3: Commit**

```bash
git add src/clearex/gui/README.md
git commit -m "docs: describe dask backend dialog fixed footer and help card"
```

## Task 7: Run the Full Validation Set for the Change

**Files:**
- Modify: none
- Test: `tests/gui/test_gui_execution.py`

- [ ] **Step 1: Run targeted Dask dialog tests**

Run: `uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -k "dask_dialog" -q`

Expected: PASS with the Dask backend dialog tests green.

- [ ] **Step 2: Run the full GUI execution test file**

Run: `QT_QPA_PLATFORM=offscreen uv run --with pytest python -m pytest tests/gui/test_gui_execution.py -q`

Expected: PASS

- [ ] **Step 3: Run lint on modified files**

Run: `uv run ruff check src/clearex/gui/app.py tests/gui/test_gui_execution.py src/clearex/gui/README.md`

Expected: PASS

- [ ] **Step 4: Run formatting check on modified Python files**

Run: `uv run black --check src/clearex/gui/app.py tests/gui/test_gui_execution.py`

Expected: PASS

- [ ] **Step 5: Commit the final validation checkpoint**

```bash
git add src/clearex/gui/app.py tests/gui/test_gui_execution.py src/clearex/gui/README.md
git commit -m "chore: verify dask backend dialog layout refresh"
```

## Spec Coverage Check

- Fixed footer outside the scroll area: covered by Tasks 1 and 5.
- Fixed bottom `Parameter Help` panel below the scrolled settings: covered by Task 2.
- Help shown only on hover/focus and hidden otherwise: covered by Tasks 2 and 4.
- Plain-language explanations for backend settings: covered by Task 3.
- Preserve existing reset/apply behavior: covered by Tasks 1, 5, and 7.
- Documentation update: covered by Task 6.

No uncovered spec requirements remain.
