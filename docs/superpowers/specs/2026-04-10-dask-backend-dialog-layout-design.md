# 2026-04-10 Dask Backend Dialog Layout Design

## Status

Approved for implementation planning.

## Summary

Improve the `Edit Dask Backend` popup so the bottom actions remain visible at
all times and add a plain-language `Parameter Help` panel for non-expert users.

The dialog should be restructured into three fixed regions:

- a scrollable settings body
- a fixed `Parameter Help` card directly below the scrolled settings
- a fixed footer containing `Reset Defaults`, `Cancel`, and `Apply`

The help card is hidden by default and appears only while a registered setting
is hovered or focused.

## Goals

- Keep the Dask backend dialog action buttons visible without scrolling.
- Match the existing ClearEx fixed-footer popup behavior rather than creating a
  one-off layout.
- Add a bottom `Parameter Help` section that explains each backend setting in
  plain language.
- Show help only while the user is hovering or focusing a setting.
- Cover the backend mode selector and all mode-specific settings.
- Preserve existing validation, reset, and apply behavior.

## Non-goals

- Do not redesign the analysis dialog or other popups in this change.
- Do not replace existing tooltips; the help panel complements them.
- Do not make the help panel permanently visible.
- Do not change the underlying Dask backend configuration schema or runtime
  semantics.
- Do not add advanced expert-only diagnostics to the popup in the first
  version.

## Context

The current `DaskBackendConfigDialog` puts all controls, recommendation text,
and footer buttons inside one scrollable content widget in
`src/clearex/gui/app.py`. On smaller displays or when the `SLURMCluster` page
is active, the user has to scroll to reach `Cancel` or `Apply`.

The main analysis dialog already uses a better interaction pattern:

- scrollable parameter content
- fixed bottom help/status area
- fixed footer actions

That pattern is especially important here because Dask terminology is hard for
non-expert users. The popup already has a short mode description label, but it
does not provide field-by-field explanations in the place where the user needs
them.

The GUI subsystem documentation already requires verbose help text to be shown
in-panel and driven by hover/focus wiring. The Dask backend dialog should adopt
that same behavior instead of inventing a separate help mechanism.

## Chosen Approach

Restructure the popup into a fixed outer layout with a local hover/focus-driven
help card below the scroll area.

This is preferred over pinning only the footer because it solves both the
button-visibility problem and the plain-language guidance problem in one
coherent layout. It is also preferred over tooltip-only guidance because Dask
settings need longer explanations than a tooltip can comfortably carry.

## Layout Contract

### Outer structure

The dialog root should contain three sibling regions in this order:

1. scrollable settings area
2. fixed help card
3. fixed footer row

Only the settings area scrolls. The help card and footer must remain visible at
the bottom of the popup while the user scrolls through mode-specific settings.

### Scrollable settings area

The existing overview text, mode selector, mode help label, and stacked
mode-specific pages remain in the scroll area.

The scroll area keeps the existing popup styling:

- `popupDialogScroll`
- `popupDialogContent`
- dark themed surfaces

### Fixed help card

Add a `Parameter Help` card below the scroll area.

Requirements:

- hidden by default
- shown only when a registered widget is hovered or focused
- hidden again when no registered widget is active
- visually styled like the existing inline help card pattern used by the
  analysis dialog

Card contents:

- title: `Parameter Help`
- body: one plain-language explanation for the active setting

### Fixed footer

Keep the existing footer actions, but move them out of the scroll area:

- `Reset Defaults`
- `Cancel`
- `Apply`

The footer should keep `QHBoxLayout`-based alignment and existing button
styling, including the emphasized `Apply` button.

## Help Interaction Contract

### Triggering

The help card should react to both:

- pointer hover
- keyboard focus

This supports mouse and keyboard users consistently.

### Visibility rules

- initial state: hidden
- `Enter` or `FocusIn` on a registered widget: show card with that widget's
  message
- `Leave` or `FocusOut`: if another registered widget still has focus, show its
  message instead
- otherwise hide the help card

### Scope

Register help text for:

- backend mode selector
- all `LocalCluster` inputs
- all `SLURMRunner` inputs
- all `SLURMCluster` inputs
- the `Recommend Settings` button, because it materially changes the local
  worker fields

Buttons that only perform dialog actions do not need hover help in the first
version:

- `Reset Defaults`
- `Cancel`
- `Apply`

## Help Content Contract

Help text should explain practical effect, not internal implementation details.

Examples of desired tone:

- `Workers`: how many separate Dask workers ClearEx starts.
- `Threads per worker`: how much parallel work each worker tries to do inside a
  single process.
- `Memory limit`: the memory cap Dask applies per worker before spilling or
  restarting.
- `Scheduler file`: the file ClearEx uses to attach to a scheduler started
  outside this dialog.
- `Dashboard address`: where the scheduler binds its dashboard service, not the
  localhost relay URL the GUI opens for browsing.
- `Allowed failures`: how many worker failures the scheduler tolerates before
  aborting the run.

The text should be written for a user who understands their dataset and cluster
but may not understand Dask-specific jargon.

## Implementation Shape

### Dialog wiring

`DaskBackendConfigDialog` should gain:

- a local widget-to-help mapping
- a hidden help-card frame and label
- small helper methods to:
  - register parameter hints
  - show/hide the help card
  - update the help text

The dialog should use a local `eventFilter(...)` patterned after the analysis
dialog's parameter-help behavior, but scoped to this popup only.

### Reuse of existing patterns

Implementation should reuse existing GUI conventions where possible:

- `apply_footer_row_spacing(...)` for the action row
- `apply_help_stack_spacing(...)` for the help card internals
- existing popup stylesheet helpers
- existing fixed-height button helper

This should look like a natural extension of the current popup system rather
than a custom embedded mini-layout.

## Testing

Add targeted GUI regression tests for:

- footer buttons are outside the scrollable settings container
- help card is hidden when the dialog opens
- hovering or focusing a registered field shows the help card
- the shown help text matches the registered field
- leaving or unfocusing the active field hides the card when nothing else is
  active
- existing reset/apply behavior still works after the layout refactor

Representative fields are sufficient for event-behavior tests as long as the
registration coverage is validated for each backend mode.

## Documentation

Update `src/clearex/gui/README.md` in the same change set to document that the
Dask backend popup now uses:

- a fixed bottom footer
- a fixed hover/focus-driven `Parameter Help` card
- plain-language per-setting help for backend configuration

## Risks and Mitigations

- Risk: hover/focus behavior can flicker when moving between related widgets.
  Mitigation: prefer the same focus fallback logic already used by the analysis
  dialog's help panel.
- Risk: help text becomes too long and makes the bottom region unstable.
  Mitigation: keep text concise and size the card for wrapped multi-line text
  without forcing the footer to move.
- Risk: moving the footer outside the scroll area changes popup size behavior.
  Mitigation: keep the existing initial-geometry helper and add a test for the
  new structure rather than relying on visual inspection alone.
