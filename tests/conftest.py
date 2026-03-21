"""Pytest configuration for macOS-friendly GUI and napari tests."""

from __future__ import annotations

import os
import sys

import pytest

if sys.platform == "darwin":
    # Use Qt's offscreen platform during tests so transient dialogs/viewers do
    # not flash on the interactive desktop. Callers can still override this by
    # exporting QT_QPA_PLATFORM explicitly before running pytest.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_FAKE_GPU_RENDERER_INFO = {
    "vendor": "Test GPU Vendor",
    "renderer": "Test GPU Renderer",
    "version": "4.1",
    "software_renderer": False,
    "gpu_renderer": True,
    "gpu_vendor_hint": True,
}


@pytest.fixture(autouse=True)
def _stub_napari_opengl_probe_for_macos(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub vispy OpenGL probing for macOS unit tests.

    The direct napari-viewer tests in ``tests/visualization/test_pipeline.py``
    monkeypatch ``napari`` with lightweight fakes, but the runtime still calls
    ``_probe_napari_opengl_renderer``, which touches the real vispy OpenGL
    bindings. On macOS that can still open or probe the desktop renderer and
    occasionally segfault under pytest. Default those tests to a deterministic
    fake GPU renderer; tests that need different probe behavior can override the
    monkeypatch inside the test body.
    """
    if sys.platform != "darwin":
        return
    if "tests/visualization/test_pipeline.py" not in str(request.node.fspath):
        return

    import clearex.visualization.pipeline as visualization_pipeline

    monkeypatch.setattr(
        visualization_pipeline,
        "_probe_napari_opengl_renderer",
        lambda _viewer: dict(_FAKE_GPU_RENDERER_INFO),
    )
