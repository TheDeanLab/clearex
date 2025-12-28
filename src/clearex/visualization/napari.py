#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

# Standard library imports
from typing import Tuple, List, Dict, Union, Optional, Callable, Any
import os
from pathlib import Path

# Third party imports
import napari
from qtpy.QtWidgets import QApplication
import numpy as np
from qtpy.QtCore import QTimer
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Local imports


def load_tracks(
    data_path: Union[str, os.PathLike],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load detections and tracks for napari visualization.

    This helper reads two CSV files written by the `crop_analysis` pipeline
    located under ``<data_path>/crop_analysis/``: ``detections.csv`` and
    ``tracks.csv``. The detections table must contain the columns ``t, z, y, x``
    and may include per-detection properties such as ``amp`` and ``int``. The
    tracks table is expected to be a comma-separated numeric table with either
    4 columns (2D: ``t, y, x``) or 5 columns (3D: ``t, z, y, x``).

    Parameters
    ----------
    data_path : str or os.PathLike
        Path to the dataset folder that contains the ``crop_analysis``
        subdirectory with the expected CSV files.

    Returns
    -------
    detection_data : numpy.ndarray
        Array of per-detection coordinates with shape (N, 4) and columns
        ``[t, z, y, x]``.
    detection_data_2D : numpy.ndarray
        Array of per-detection coordinates projected to 2D with shape (N, 3)
        and columns ``[t, y, x]``.
    tracks_data : numpy.ndarray
        Track table after removing masked (invalid) rows. Shape is (M, 4) or
        (M, 5) depending on whether tracks are 2D or 3D.
    tracks_data_2d : numpy.ndarray
        Track table converted to 2D by removing the z column (if present).
    properties : dict
        Dictionary of per-detection properties (e.g. ``{'amp': array, 'int': array}``).

    Raises
    ------
    FileNotFoundError
        If either ``detections.csv`` or ``tracks.csv`` are missing.
    KeyError
        If required columns are missing from ``detections.csv``.

    Notes
    -----
    The function will raise a :class:`KeyError` if the expected detection
    columns ``t, z, y, x`` are not present. The tracks loader is tolerant of
    single-row files (``np.loadtxt`` returns a 1-D array in that case) and will
    reshape as needed.
    """

    crop_dir = os.path.join(data_path, "crop_analysis")
    det_path = os.path.join(crop_dir, "detections.csv")
    tracks_path = os.path.join(crop_dir, "tracks.csv")

    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Detections file not found: {det_path}")
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")

    df = pd.read_csv(det_path)

    # Ensure required columns exist
    required = {"t", "z", "y", "x"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"Missing detection columns: {sorted(missing)}")

    # Coordinates (N, 4) → columns must be in the same order we wrote them
    detection_data = df[["t", "z", "y", "x"]].to_numpy()
    detection_data_2D = df[["t", "y", "x"]].to_numpy()

    # Optional per‑point features (amplitude, intensity). Missing features
    # will be returned as arrays of NaNs to preserve length.
    n = len(df)
    amp = df["amp"].to_numpy() if "amp" in df.columns else np.full(n, np.nan)
    inten = df["int"].to_numpy() if "int" in df.columns else np.full(n, np.nan)
    properties: Dict[str, np.ndarray] = {"amp": amp, "int": inten}

    # Load tracks robustly (handle single-row files)
    tracks_raw = np.loadtxt(tracks_path, delimiter=",")
    if tracks_raw.ndim == 1:
        # single track row; reshape to (1, ncols)
        tracks_data = tracks_raw.reshape(1, -1)
    else:
        tracks_data = tracks_raw

    # tracks_data.shape[1] decides if it is 2‑D (4 cols) or 3‑D (5 cols)
    ndisplay = 3 if tracks_data.shape[1] == 5 else 2
    print("Number of dimensions:", ndisplay)

    # Mask rows where x,y are both -0.5 (invalid placeholders)
    y_col = -2
    x_col = -1
    tol = 1e-6
    mask = ~(
        np.isclose(tracks_data[:, y_col], -0.5, atol=tol)
        & np.isclose(tracks_data[:, x_col], -0.5, atol=tol)
    )
    tracks_data = tracks_data[mask]

    # Convert to 2D tracks by removing z column (index 2) if present
    if tracks_data.shape[1] >= 5:
        tracks_data_2d = np.delete(tracks_data, 2, axis=1)
    else:
        # Already 2D-like
        tracks_data_2d = tracks_data

    return detection_data, detection_data_2D, tracks_data, tracks_data_2d, properties


def create_viewer(*, dimensions: int = 3, show: bool = True) -> napari.Viewer:
    """
    Create and return a new napari viewer instance.

    Parameters
    ----------
    dimensions : int, optional
        Number of spatial dimensions for the viewer (2 or 3), by default 3.
    show : bool, optional
        Whether to display the viewer window immediately, by default True.

    Returns
    -------
    napari.Viewer
        A new napari viewer instance.
    """
    viewer = napari.Viewer(ndisplay=dimensions, show=show)
    return viewer


def set_viewer_size_for_layer(
    viewer: napari.Viewer,
    layer_name: str,
    match_native_pixels: bool = True,
    scale: int = 1,
    extra_width: int = 200,
    extra_height: int = 50,
) -> Optional[Tuple[int, int]]:
    """Resize napari viewer and canvas to match a layer's lateral extent.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to resize.
    layer_name : str
        Name of an image layer already added to the viewer.
    match_native_pixels : bool, optional
        If True, make canvas at least as many pixels as the image XY dims.
        Default is True.
    scale : int, optional
        Integer scale multiplier to enlarge further. Default is 1.
    extra_width : int, optional
        Allowance for sidebars/window decoration in pixels. Default is 200.
    extra_height : int, optional
        Allowance for window decoration in pixels. Default is 50.

    Returns
    -------
    Optional[Tuple[int, int]]
        Target canvas dimensions (width, height) if successful, or None if
        sizing could not be applied to the inner canvas.

    Raises
    ------
    ValueError
        If the layer has fewer than 2 spatial dimensions.
    """
    layer = viewer.layers[layer_name]
    arr = layer.data
    if arr.ndim < 2:
        raise ValueError("Layer has < 2 spatial dims")

    img_h, img_w = arr.shape[-2], arr.shape[-1]
    try:
        canvas = viewer.window.qt_viewer.canvas.native  # QWidget
    except Exception:
        # fallback to top-level window — return None to indicate sizing
        # could not be applied to the inner canvas
        viewer.window.resize(
            int(img_w * scale) + extra_width, int(img_h * scale) + extra_height
        )
        QApplication.processEvents()
        return None

    # current canvas size
    cur_w, cur_h = canvas.width(), canvas.height()

    if match_native_pixels:
        target_w = max(cur_w, int(img_w * scale))
        target_h = max(cur_h, int(img_h * scale))
    else:
        target_w = cur_w * max(1, scale)
        target_h = cur_h * max(1, scale)

    # clamp to reasonable integers
    target_w, target_h = max(1, int(target_w)), max(1, int(target_h))

    # resize the canvas and then the window (give extra space for sidebars)
    canvas.resize(target_w, target_h)
    # Add extra to window size to account for dock widgets / decorations
    viewer.window.resize(target_w + extra_width, target_h + extra_height)
    QApplication.processEvents()
    return (target_w, target_h)


def update_viewer(
    viewer: napari.Viewer,
    angles: Tuple[float, float, float] | None = None,
    zoom: float | None = None,
    center: Tuple[float, float, float] | None = None,
    time: int | None = None,
) -> None:
    """
    Update the napari viewer's camera perspective and time step.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to update.
    angles : Tuple[float, float, float] | None
        Camera rotation angles (azimuth, elevation, roll) in degrees.
    zoom : float | None
        Camera zoom level.
    center : Tuple[float, float, float] | None
        Camera center point coordinates (z, y, x).
    time : int | None
        Time step index to set in the viewer dimensions.

    Returns
    -------
    None
    """
    if angles is not None:
        viewer.camera.angles = angles
    if zoom is not None:
        viewer.camera.zoom = zoom
    if center is not None:
        viewer.camera.center = center
    if time is not None:
        step = list(viewer.dims.current_step)
        step[0] = time
        viewer.dims.current_step = tuple(step)


def get_viewer_perspective(
    viewer: napari.Viewer,
) -> Tuple[
    Tuple[float, float, float], float, Tuple[float, float, float], List[int], int
]:
    """
    Get the current camera perspective and time step from a napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to query.

    Returns
    -------
    angle : Tuple[float, float, float]
        Camera rotation angles (azimuth, elevation, roll) in degrees.
    zoom : float
        Camera zoom level.
    center : Tuple[float, float, float]
        Camera center point coordinates (z, y, x).
    step : List[int]
        Current dimension step indices.
    time : int
        Time step index (first element of step).
    """
    angle = viewer.camera.angles
    zoom = viewer.camera.zoom
    center = viewer.camera.center
    step = list(viewer.dims.current_step)
    time = step[0]
    return angle, zoom, center, step, time


def spin_azimuth(viewer, n_frames=180, seconds=6.0, save_dir=None, prefix="az_spin"):
    """
    Sweep the camera azimuth by 360° around the current camera center.
    - n_frames: how many steps around the circle
    - seconds: total duration of the sweep
    - save_dir: folder path to save PNG frames (None = don't save)
    - prefix: filename prefix if saving frames
    Returns the QTimer so you can stop it if needed (timer.stop()).
    """
    # Lock camera center to the scene’s midpoint for a true “orbit”
    # (use your main 3D layer; replace 'viewer.layers[0]' if needed)
    main = viewer.layers[0]
    wmin, wmax = main.extent.world  # shape (3,)
    center = (wmin + wmax) / 2
    viewer.camera.center = tuple(center)

    # Keep current elevation & roll, vary azimuth
    elev, start_az, roll = viewer.camera.angles
    azimuths = np.linspace(start_az, start_az + 360.0, n_frames, endpoint=False)

    # Optional frame saving
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    idx = {"i": 0}
    interval_ms = max(1, int(1000 * seconds / n_frames))
    timer = QTimer()

    def step():
        i = idx["i"]
        if i >= len(azimuths):
            timer.stop()
            return
        viewer.camera.angles = (elev, float(azimuths[i]), roll)
        if save_dir is not None:
            out = save_dir / f"{prefix}_{i:04d}.png"
            viewer.screenshot(path=str(out), canvas_only=True, flash=False)
        idx["i"] += 1

    timer.timeout.connect(step)
    timer.start(interval_ms)
    return timer


def lut_strip_for_layer(layer, width=140, height=10, alpha=0.6) -> np.ndarray:
    gamma = float(getattr(layer, "gamma", 1.0))
    xs = np.linspace(0, 1, width, dtype=np.float32)
    xs_g = np.clip(xs, 0, 1) ** gamma

    rgba = _colormap_map(layer.colormap, xs_g)  # (W,4) float [0..1]
    strip = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)  # (W,4) u8
    strip = np.repeat(strip[None, :, :], height, axis=0)  # (H,W,4)

    # <- THIS LINE makes the LUT bar transparent
    strip[..., 3] = int(255 * float(alpha))
    return strip


def representative_text_rgba_for_layer(
    layer, u=0.85, alpha=0.85
) -> tuple[int, int, int, int]:
    gamma = float(getattr(layer, "gamma", 1.0))
    xg = float(np.clip(u, 0, 1) ** gamma)
    rgba = _colormap_map(layer.colormap, np.array([xg], dtype=np.float32))[
        0
    ]  # float [0..1]

    out = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    out[3] = int(255 * float(alpha))  # <- THIS LINE makes text transparent
    return tuple(out)


def _layer(viewer: napari.Viewer, name: str) -> Optional[napari.layers.Layer]:
    """Get a layer by name, trying a few tolerant name variants.

    Many places in this notebook add layers using `name=key.capitalize()` but
    other code expects the original dictionary key (e.g., 'ER'). This helper
    tries exact lookup first, then common capitalization variants, then a
    whitespace/case-insensitive normalized match across existing layer names.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    name : str
        The desired layer name.

    Returns
    -------
    Optional[napari.layers.Layer]
        The matched layer, or None if not found.
    """
    if name is None:
        return None

    # Fast exact lookup
    try:
        return viewer.layers[name]
    except Exception:
        pass

    # Try common variants
    variants = [
        name.capitalize(),
        name.title(),
        name.lower(),
        name.upper(),
    ]
    for v in variants:
        try:
            return viewer.layers[v]
        except Exception:
            continue

    # Normalized match: strip spaces and compare lowercase
    norm = str(name).replace(" ", "").lower()
    for lname in viewer.layers:
        if str(lname).replace(" ", "").lower() == norm:
            return viewer.layers[lname]

    # No match found
    return None


def _active_set(component: str, groups: dict, anchor_labels=()):
    """Return the set of labels that should be visible for this component.

    Parameters
    ----------
    component : str
        Component name.
    groups : dict
        Dictionary of component → list of labels.
    anchor_labels : tuple[str, ...], optional
        Labels to keep as anchors, by default ()

    Returns
    -------
    set
        Set of active layer names.
    """
    return set(anchor_labels) | set(groups.get(component, []))


def _targets_for_set(active: set, data: dict, anchor_labels=(), anchor_opacity=0.25):
    """
    Build per-layer target opacities:
      - active layers -> their configured opacity
      - inactive -> 0
      - anchor layers -> dimmed to anchor_opacity (min of both)

    Parameters
    ----------
    active : set
        Set of active layer names.
    data : dict
        Dictionary of layer configurations.
    anchor_labels : tuple[str, ...], optional
        Labels to keep as anchors, by default ()
    anchor_opacity : float, optional
        Opacity for anchor labels, by default 0.25

    Returns
    -------
    dict
        Dictionary of layer name → target opacity.
    """
    targets = {}

    # Iterate through the data dictionary
    # name is the name of the label. cfg is its configuration dict (with opacity, etc.)
    for name, cfg in data.items():

        # If the name is active, get the opacity from the config
        if name in active:
            targets[name] = float(cfg.get("opacity", 1.0))
            if name in anchor_labels:
                targets[name] = min(targets[name], float(anchor_opacity))
        else:
            targets[name] = 0.0
    return targets


def _apply_targets(
    viewer, targets: dict, hide_when_zero: bool = True, extra_whitelist=()
):
    """
    Apply opacity/visibility to layers named in `targets`. Also hide any
    viewer layers that are in `data` but not in targets (defensive), except
    for layers in extra_whitelist (text_overlay, scale bars, etc).
    """
    # apply targets for names we can look up
    for name, targ in targets.items():
        layer = _layer(viewer, name)
        if layer is None:
            # still record/log the missing name for debugging
            # (you can comment out the print to reduce verbosity)
            print(f"[DEBUG] _apply_targets: layer not found for '{name}'")
            continue
        layer.opacity = float(targ)
        if hide_when_zero:
            layer.visible = targ > 0
        else:
            layer.visible = True

    # Defensive sweep: hide any data layers that are not in targets
    # Build a normalized set of target names (stripped/lower)
    target_norm = {str(n).replace(" ", "").lower() for n in targets.keys()}

    # Common viewer layers to ignore/honor regardless (text overlays, gui helpers)
    whitelist = set(str(x) for x in extra_whitelist) | {
        "text_overlay",
    }

    for layer in list(viewer.layers):
        lname = str(layer.name)
        if lname in whitelist:
            continue
        # normalize and compare
        if lname.replace(" ", "").lower() not in target_norm:
            # This layer name isn't in the active targets — hide it
            try:
                layer.visible = False
                layer.opacity = 0.0
            except Exception:
                pass


def _fade_between(
    viewer,
    start_targets: dict,
    end_targets: dict,
    n_frames: int = 12,
    hide_when_zero: bool = True,
    on_frame=None,
):
    """
    Linear fade between two opacity dictionaries.
    `on_frame(i, alpha)` can be used to update camera/text/screenshot per frame.
    """
    names = sorted(set(start_targets.keys()) | set(end_targets.keys()))
    for i, a in enumerate(np.linspace(0, 1, max(n_frames, 1))):
        for name in names:
            layer = _layer(viewer, name)
            if layer is None:
                continue
            s = float(start_targets.get(name, layer.opacity))
            e = float(end_targets.get(name, layer.opacity))
            layer.opacity = (1 - a) * s + a * e
            if hide_when_zero:
                layer.visible = layer.opacity > 1e-6

        QApplication.processEvents()
        if on_frame is not None:
            on_frame(i, a)


def _screenshot(viewer, out_path: str):
    """Save a canvas-only PNG frame."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    QApplication.processEvents()
    try:
        viewer.screenshot(path=out_path, canvas_only=True, scale=10, flash=False)
    except TypeError:
        # older/newer signature differences
        viewer.screenshot(path=out_path, canvas_only=True, scale=10)


def _as_array(
    data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
) -> np.ndarray:
    """Convert layer data to a numpy array.

    Handles both single arrays and multiscale (list/tuple) data by
    returning the highest resolution level.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray or tuple of np.ndarray
        Layer data, either a single array or multiscale list/tuple.

    Returns
    -------
    np.ndarray
        The data as a numpy array (highest resolution if multiscale).
    """
    # layer.data can be ndarray or multiscale list; use highest-res
    if isinstance(data, (list, tuple)):
        return np.asarray(data[0])
    return np.asarray(data)


def representative_rgba_from_image_layer(
    layer, q=0.90, sample_max=200_000, fallback_x=0.85
):
    """
    Return an RGBA tuple (floats 0..1) that matches the layer's *display*
    mapping: contrast_limits -> gamma -> colormap.

    q: quantile of normalized-in-range values to sample (0..1). Use high (0.85–0.95)
       to get a bright, representative channel color.
    """
    if getattr(layer, "colormap", None) is None:
        return None

    vmin, vmax = map(float, layer.contrast_limits)
    gamma = float(getattr(layer, "gamma", 1.0))
    denom = (vmax - vmin) if vmax > vmin else 1.0

    arr = _as_array(layer.data).ravel()
    if arr.size == 0:
        x0 = fallback_x
    else:
        # subsample for speed
        if arr.size > sample_max:
            rng = np.random.default_rng(0)
            arr = arr[rng.integers(0, arr.size, size=sample_max)]

        arr = arr.astype(np.float32, copy=False)
        x = (arr - vmin) / (denom + 1e-12)
        x = x[np.isfinite(x)]
        x = x[(x >= 0) & (x <= 1)]

        x0 = float(np.quantile(x, q)) if x.size else fallback_x

    # napari gamma is applied in normalized space
    xg = float(np.clip(x0, 0, 1) ** gamma)

    rgba = np.asarray(layer.colormap.map(np.array([xg], dtype=np.float32)))
    rgba = rgba[0] if rgba.ndim == 2 else rgba
    rgba = rgba.astype(float)

    # Force alpha=1 for readable text
    if rgba.size == 3:
        rgba = np.concatenate([rgba, [1.0]])
    rgba[3] = 1.0

    return tuple(float(c) for c in rgba[:4])


def _ensure_rgba_u8(frame: Union[np.ndarray, Any]) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        # handle float screenshots (0..1) or odd ranges defensively
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)

    if arr.shape[-1] != 4:
        raise ValueError(f"Unexpected screenshot shape: {arr.shape}")

    return arr


def _colormap_map(cm, x: np.ndarray) -> np.ndarray:
    """Return (N,4) float RGBA in [0,1] from a napari colormap."""
    rgba = cm.map(x.astype(np.float32, copy=False))
    rgba = np.asarray(rgba, dtype=float)
    if rgba.ndim == 1:
        rgba = rgba[None, :]
    if rgba.shape[1] == 3:
        rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1))], axis=1)
    rgba[:, 3] = 1.0
    return rgba


def _load_font(size: int):
    # Try a common font; fall back to default
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_black_background_transparent(
    frame_rgba_u8: np.ndarray, thresh: int = 2
) -> np.ndarray:
    """
    Set alpha=0 where the RGB is near-black (<= thresh in all channels).
    This makes the “empty” canvas transparent for PowerPoint/figures.
    """
    fr = np.array(frame_rgba_u8, copy=True)
    rgb = fr[..., :3]
    mask = (rgb <= thresh).all(axis=-1)
    fr[mask, 3] = 0
    return fr


def annotate_frame_with_component_legend(
    frame_rgba_u8: np.ndarray,
    viewer,
    component: str,
    label_names: list[str],
    x=20,
    y=20,
    title_rgba=(255, 255, 255, 255),
    box_rgba=(0, 0, 0, 140),  # semi-transparent box
    draw_lut=True,
    transparent_background=False,
    bg_thresh=2,
        z_idx=None,
):
    frame_rgba_u8 = _ensure_rgba_u8(frame_rgba_u8)

    # OPTIONAL: make background transparent
    if transparent_background:
        frame_rgba_u8 = make_black_background_transparent(
            frame_rgba_u8, thresh=bg_thresh
        )

    base = Image.fromarray(frame_rgba_u8, mode="RGBA")

    # Draw everything on an overlay, then alpha-composite once
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fonts (use your existing _load_font if you have it)
    H = frame_rgba_u8.shape[0]
    title_font = _load_font(max(18, H // 40))
    label_font = _load_font(max(16, H // 55))
    line_gap = max(6, H // 200)

    title = component.replace("_", " ").title()
    labels = list(label_names)

    lut_w, lut_h = 140, 10
    lut_pad = 10

    # Measure box - title, optional z-index, then labels
    lines = [title]
    if z_idx is not None:
        # Label position in microns.
        # lines.append(f"Z-position: {z_idx * 0.065 / 4.2} µm")
        lines.append(f"Z-position: {z_idx * 0.15 / 4.2:.2f} µm")
    lines.extend(labels)

    max_w, total_h = 0, 0
    for i, line in enumerate(lines):
        # Title and z-index use title_font, labels use label_font
        # Check if we're past the title (and optional z-index line)
        z_line_count = 2 if z_idx is not None else 1
        font = title_font if i < z_line_count else label_font
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        extra = (lut_w + lut_pad) if (draw_lut and i > 0) else 0
        max_w = max(max_w, tw + extra)
        total_h += th + line_gap

    pad = 12
    draw.rectangle(
        [x - pad, y - pad, x + max_w + pad, y + total_h + pad], fill=box_rgba
    )

    # Title
    cy = y
    draw.text((x, cy), title, font=title_font, fill=title_rgba)
    cy += (draw.textbbox((0, 0), title, font=title_font)[3]) + line_gap

    # Z-index (if provided)
    if z_idx is not None:
        # z_text = (f"Z-position: {z_idx * 0.065 / 4.2} µm")
        z_text = (f"Z-position: {z_idx * 0.15 / 4.2:.2f} µm")

        draw.text((x, cy), z_text, font=title_font, fill=title_rgba)
        cy += (draw.textbbox((0, 0), z_text, font=title_font)[3]) + line_gap

    # Labels (each colored by its own LUT + gamma)
    for label in labels:
        if label not in viewer.layers:
            draw.text((x, cy), label, font=label_font, fill=(180, 180, 180, 255))
            cy += (draw.textbbox((0, 0), label, font=label_font)[3]) + line_gap
            continue

        layer = viewer.layers[label]
        txt_rgba = representative_text_rgba_for_layer(layer, u=0.85, alpha=0.80)
        draw.text((x, cy), label, font=label_font, fill=txt_rgba)

        if draw_lut:
            strip = lut_strip_for_layer(layer, width=lut_w, height=lut_h, alpha=0.55)
            strip_img = Image.fromarray(strip, mode="RGBA")

            tb = draw.textbbox((x, cy), label, font=label_font)
            sx = tb[2] + lut_pad
            sy = cy + max(0, (tb[3] - tb[1] - lut_h) // 2)
            overlay.alpha_composite(strip_img, dest=(sx, sy))

        cy += (draw.textbbox((0, 0), label, font=label_font)[3]) + line_gap

    # Composite overlay onto base
    base.alpha_composite(overlay)
    return np.asarray(base)


def save_frame(
    viewer, out_path: str, comp: str, groups: dict, transparent_background=True,
    z_idx=None
):
    frame = viewer.screenshot(canvas_only=True, flash=False, size=(4000, 4000))
    labels = groups.get(comp, [])

    frame2 = annotate_frame_with_component_legend(
        frame,
        viewer,
        comp,
        labels,
        draw_lut=True,
        transparent_background=transparent_background,
        bg_thresh=2,
        z_idx=z_idx,
    )

    Image.fromarray(frame2, mode="RGBA").save(out_path)


def make_component_movie_frames(
    viewer,
    data: dict,
    groups: dict,
    out_dir: str,
    components: list[str],
    frames_per_component: int = 90,  # 3 seconds at 30 fps
    transition_frames: int = 12,
    anchor_labels: tuple[str, ...] = ("Nuclei",),  # keep a dim anchor for context
    anchor_opacity: float = 0.20,
    hide_when_zero: bool = True,
    initial_angles: tuple[float, float, float] = None,
    initial_zoom: float = None,
    initial_center: tuple[float, float, float] = None,
):
    """
    Writes PNG frames to out_dir/frames/frame_00000.png, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Enforce the requested initial perspective so each movie run is consistent.
    # Use the user-provided initial values (one-shot). These are the values you
    # requested as the starting perspective for the sweep.
    # initial_angles = (np.float64(-7.077187481047181), np.float64(14.321447521197806), np.float64(152.7614333497747))
    # initial_zoom = float(np.float64(2.209623491786597))
    # initial_center = (np.float64(8.25), np.float64(171.9), np.float64(125.99999999999999))

    # Apply the initial perspective to the viewer (one-time)
    try:
        viewer.camera.angles = initial_angles
        viewer.camera.zoom = initial_zoom
        viewer.camera.center = tuple(initial_center)
    except Exception:
        # If the viewer doesn't accept angles/zoom/center assignment the usual
        # way, fall back to reading the current camera but continue.
        pass

    # Snapshot current camera as the “base" (use the applied initial perspective)
    base_angles = tuple(viewer.camera.angles)
    base_zoom = float(viewer.camera.zoom)
    base_center = tuple(viewer.camera.center)

    frame_idx = 0
    prev_targets = None

    for ci, comp in enumerate(components):

        # Determine which layers should be visible for this component
        active = _active_set(comp, groups, anchor_labels=anchor_labels)

        # Creates a dictionary with the target opacities for each layer
        targets = _targets_for_set(
            active, data, anchor_labels=anchor_labels, anchor_opacity=anchor_opacity
        )

        # Ensure all involved layers exist / match names
        missing = [name for name in active if _layer(viewer, name) is None]
        if missing:
            print(f"[WARNING] Missing layers for component '{comp}': {missing}")

        # Transition (fade) from previous component
        if prev_targets is None:

            # Sets visibility/opacities directly for the first component
            _apply_targets(
                viewer, targets, hide_when_zero=True, extra_whitelist=("Nuclei",)
            )

            # Sets the text overlay
            # _set_text_overlay(viewer, f"{comp}  \n  {', '.join(groups.get(comp, []))}", layer_name=groups.get(comp, [None])[0])
        else:
            # Transition: fade opacities while keeping the camera at the base
            # perspective (no complicated camera sweep). We still render every
            # transition frame so the fade is captured to disk.
            def on_fade_frame(i, a):
                # Keep camera fixed at the base perspective during fade
                viewer.camera.angles = base_angles
                viewer.camera.zoom = base_zoom
                viewer.camera.center = base_center

                # _set_text_overlay(viewer, overlay_text(comp, groups, show_labels=True), layer_name=groups.get(comp, [None])[0])
                out_path = os.path.join(frames_dir, f"frame_{frame_idx + i:05d}.png")
                # _screenshot(viewer, out_path)
                save_frame(viewer, out_path, comp, groups, transparent_background=False)

            _fade_between(
                viewer,
                start_targets=prev_targets,
                end_targets=targets,
                n_frames=transition_frames,
                hide_when_zero=hide_when_zero,
                on_frame=on_fade_frame,
            )
            frame_idx += transition_frames

        # Main segment: perform a single full 360° azimuth sweep around the
        # scene's vertical axis (azimuth). Keep elevation & roll fixed.
        _apply_targets(
            viewer, targets, hide_when_zero=True, extra_whitelist=("Nuclei",)
        )

        # Generate frame-wise azimuths for a full rotation
        azimuths = np.linspace(
            float(base_angles[1]),
            float(base_angles[1] + 360.0),
            frames_per_component,
            endpoint=True,
        )

        for t, az in enumerate(azimuths):
            viewer.camera.angles = (base_angles[0], float(az), base_angles[2])
            viewer.camera.zoom = base_zoom
            viewer.camera.center = base_center

            # _set_text_overlay(viewer, f"{comp}  |  {', '.join(groups.get(comp, []))}", layer_name=groups.get(comp, [None])[0])
            out_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
            save_frame(viewer, out_path, comp, groups, transparent_background=False)
            frame_idx += 1

        prev_targets = targets

    print(f"Done. Wrote {frame_idx} frames to: {frames_dir}")
    return frames_dir
