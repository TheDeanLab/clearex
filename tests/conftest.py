"""Pytest configuration for macOS-friendly GUI and napari tests."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import zarr

try:
    from numcodecs import Blosc as _NumcodecsBlosc
except Exception:  # pragma: no cover - optional dependency import guard
    _NumcodecsBlosc = None

try:
    from zarr.codecs import BloscCodec
except Exception:  # pragma: no cover - optional dependency import guard
    BloscCodec = None

if sys.platform == "darwin":
    # Use Qt's offscreen platform during tests so transient dialogs/viewers do
    # not flash on the interactive desktop. Callers can still override this by
    # exporting QT_QPA_PLATFORM explicitly before running pytest.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config: pytest.Config) -> None:
    """Register repository-local pytest markers.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration object.

    Returns
    -------
    None
        Marker registration has side effects only.
    """
    config.addinivalue_line(
        "markers",
        "biohpc: local-only integration tests that require the BioHPC archive "
        "fixture bundle.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip BioHPC-only tests automatically on GitHub Actions.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration object.
    items : list[pytest.Item]
        Collected tests for the current run.

    Returns
    -------
    None
        Items are modified in place when CI exclusion applies.
    """
    del config
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() != "true":
        return
    skip_marker = pytest.mark.skip(
        reason="biohpc tests are local-only and are not run on GitHub Actions."
    )
    for item in items:
        if "biohpc" in item.keywords:
            item.add_marker(skip_marker)


def _normalize_legacy_zarr_compressor(value):
    """Translate legacy numcodecs compressors into Zarr 3 codecs when needed."""
    if _NumcodecsBlosc is not None and BloscCodec is not None:
        if isinstance(value, _NumcodecsBlosc):
            shuffle = getattr(value, "shuffle", None)
            if isinstance(shuffle, int):
                shuffle = {
                    0: "noshuffle",
                    1: "shuffle",
                    2: "bitshuffle",
                }.get(shuffle, shuffle)
            return BloscCodec(
                typesize=int(getattr(value, "typesize", 0) or 0) or None,
                cname=getattr(value, "cname", "zstd"),
                clevel=int(getattr(value, "clevel", 5)),
                shuffle=shuffle,
                blocksize=int(getattr(value, "blocksize", 0) or 0),
            )
    return value


_ORIGINAL_ZARR_GROUP_CREATE_DATASET = zarr.Group.create_dataset


def _compat_zarr_group_create_dataset(self: zarr.Group, name: str, **kwargs):
    """Restore legacy ``create_dataset(data=...)`` ergonomics for tests.

    Zarr 3 tightened ``create_dataset`` around the async API and now requires
    ``shape`` explicitly. The test suite still uses the older, widely used sync
    pattern that infers shape and dtype from ``data``. Mirror that behavior in
    tests so the suite can focus on ClearEx semantics rather than Zarr API
    churn.
    """
    normalized = dict(kwargs)
    data = normalized.get("data")
    if data is not None:
        array = data if isinstance(data, np.ndarray) else np.asarray(data)
        dtype = normalized.pop("dtype", None)
        if dtype is not None:
            array = np.asarray(array, dtype=dtype)
        normalized.pop("shape", None)
        normalized["data"] = array
    if "compressor" in normalized:
        normalized["compressor"] = _normalize_legacy_zarr_compressor(
            normalized["compressor"]
        )
    if "compressors" in normalized:
        compressors = normalized["compressors"]
        if isinstance(compressors, (tuple, list)):
            normalized["compressors"] = [
                _normalize_legacy_zarr_compressor(item) for item in compressors
            ]
        else:
            normalized["compressors"] = _normalize_legacy_zarr_compressor(compressors)
    return self.create_array(name, **normalized)


zarr.Group.create_dataset = _compat_zarr_group_create_dataset


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
