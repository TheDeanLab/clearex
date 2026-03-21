"""Pytest configuration for macOS-friendly GUI and napari tests."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

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


def _zarr_shape_from_data(data: Any) -> Optional[Tuple[int, ...]]:
    """Return a concrete tuple shape for array-like input when available."""
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except Exception:
        return None


def _zarr_dtype_from_data(data: Any) -> Any:
    """Return a dtype-like object inferred from array-like input when available."""
    return getattr(data, "dtype", None)


@pytest.fixture(scope="session", autouse=True)
def _compat_zarr_v3_create_dataset_with_data() -> None:
    """Backfill v2-style ``create_dataset(data=...)`` semantics for tests.

    Notes
    -----
    Zarr v3 requires ``shape=...`` even when ``data=...`` is provided. A large
    portion of existing tests still use v2-style calls that omit ``shape``.
    This shim keeps test fixtures readable while preserving production behavior.
    """
    import zarr
    from _pytest.monkeypatch import MonkeyPatch

    original = zarr.core.group.Group.create_dataset
    monkeypatch = MonkeyPatch()

    def _compat_create_dataset(
        self: Any,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "shape" not in kwargs and "data" in kwargs:
            data = kwargs.get("data")
            inferred_shape = _zarr_shape_from_data(data)
            if inferred_shape is not None:
                kwargs["shape"] = inferred_shape
                if "dtype" not in kwargs:
                    inferred_dtype = _zarr_dtype_from_data(data)
                    if inferred_dtype is not None:
                        kwargs["dtype"] = inferred_dtype
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zarr.core.group.Group, "create_dataset", _compat_create_dataset)
    try:
        yield
    finally:
        monkeypatch.undo()


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
