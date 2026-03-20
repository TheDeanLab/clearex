#!/usr/bin/env python3
"""Rigidly register per-position MIP tiles with ANTsPy and stitch channels.

This script is designed for ClearEx MIP-export TIFF tiles named like:

``mip_<projection>_p0000_t0000_c0000.tif``.

Registration is estimated from one channel (default ``c0000``) and then reused
for all channels to produce one blended multi-channel mosaic TIFF per timepoint.
Transforms are estimated against a fixed anchor position using only expected
left-right overlap regions to suppress spurious global matches.

uv run scripts/register_mip_tiles_antspy.py --input-dir /endosome/archive/bioinformatics/Danuser_lab/Dean/dean/2026-03-12-dvOPM-liver/CH00_000000.n5_mip_export/latest --projection xy --time-index 0 --registration-channel 0 --overlap-fraction 0.05 --aff-iterations 50x25x10x0 --aff-shrink-factors 8x4x2x1 --aff-smoothing-sigmas 3x2x1x0 --aff-random-sampling-rate 0.2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import ants
import numpy as np
import tifffile

LOGGER = logging.getLogger("register_mip_tiles_antspy")

MIP_TILE_PATTERN = re.compile(
    r"^mip_(?P<projection>[a-z]{2})_p(?P<position>\d{4})_t(?P<time>\d{4})_c(?P<channel>\d{4})\.tif$"
)


@dataclass(frozen=True)
class TileRecord:
    """One discovered MIP tile record."""

    projection: str
    position: int
    time_index: int
    channel: int
    path: Path


@dataclass(frozen=True)
class CanvasLayout:
    """Global canvas layout used for registration and stitching."""

    shape_yx: tuple[int, int]
    origin_y: int
    origin_x: int
    step_x: int


def _parse_schedule(raw: str, *, field_name: str) -> tuple[int, ...]:
    """Parse an integer schedule from ``a x b x c`` or ``a,b,c`` text."""
    parts = [part.strip() for part in re.split(r"[x,]", str(raw)) if part.strip()]
    if not parts:
        raise ValueError(f"{field_name} must contain at least one integer value.")
    schedule = tuple(int(part) for part in parts)
    if any(value < 0 for value in schedule):
        raise ValueError(f"{field_name} values must be non-negative integers.")
    return schedule


def _discover_tiles(input_dir: Path) -> list[TileRecord]:
    """Discover MIP tile TIFFs in ``input_dir``."""
    records: list[TileRecord] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = MIP_TILE_PATTERN.match(path.name)
        if match is None:
            continue
        records.append(
            TileRecord(
                projection=str(match.group("projection")).lower(),
                position=int(match.group("position")),
                time_index=int(match.group("time")),
                channel=int(match.group("channel")),
                path=path.resolve(),
            )
        )
    return records


def _group_tiles(
    records: Iterable[TileRecord],
) -> dict[str, dict[int, dict[int, dict[int, Path]]]]:
    """Group tiles as ``projection -> time -> channel -> position -> path``."""
    grouped: dict[str, dict[int, dict[int, dict[int, Path]]]] = {}
    for record in records:
        grouped.setdefault(record.projection, {}).setdefault(
            record.time_index, {}
        ).setdefault(record.channel, {})[record.position] = record.path
    return grouped


def _load_2d_tiff(path: Path) -> np.ndarray:
    """Load a 2D TIFF tile as ``float32``."""
    data = np.asarray(tifffile.imread(str(path)))
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D TIFF tile at {path}, found shape {tuple(data.shape)}."
        )
    return np.asarray(data, dtype=np.float32)


def _build_canvas_layout(
    *,
    tile_shape_yx: tuple[int, int],
    tile_count: int,
    overlap_fraction: float,
    margin_fraction: float,
) -> CanvasLayout:
    """Create the registration/stitching canvas geometry."""
    tile_h, tile_w = int(tile_shape_yx[0]), int(tile_shape_yx[1])
    step_x = int(round(float(tile_w) * (1.0 - float(overlap_fraction))))
    step_x = max(1, step_x)

    margin_x = max(128, int(round(tile_w * float(margin_fraction))))
    margin_y = max(64, int(round(tile_h * float(margin_fraction))))

    canvas_h = int(tile_h + 2 * margin_y)
    canvas_w = int(tile_w + max(0, tile_count - 1) * step_x + 2 * margin_x)
    return CanvasLayout(
        shape_yx=(canvas_h, canvas_w),
        origin_y=margin_y,
        origin_x=margin_x,
        step_x=step_x,
    )


def _tile_anchor(layout: CanvasLayout, tile_index: int) -> tuple[int, int]:
    """Return top-left canvas anchor for one tile index."""
    y = int(layout.origin_y)
    x = int(layout.origin_x + int(tile_index) * int(layout.step_x))
    return y, x


def _tile_bbox(
    *,
    layout: CanvasLayout,
    tile_index: int,
    tile_shape_yx: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return tile bounding box as ``(y0, y1, x0, x1)`` in canvas coordinates."""
    y0, x0 = _tile_anchor(layout, tile_index)
    h, w = int(tile_shape_yx[0]), int(tile_shape_yx[1])
    return y0, y0 + h, x0, x0 + w


def _place_tile_on_canvas(
    *,
    tile_yx: np.ndarray,
    layout: CanvasLayout,
    tile_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place one tile onto a zero canvas and return image + footprint mask."""
    canvas = np.zeros(layout.shape_yx, dtype=np.float32)
    mask = np.zeros(layout.shape_yx, dtype=np.float32)
    y0, x0 = _tile_anchor(layout, tile_index)
    h, w = int(tile_yx.shape[0]), int(tile_yx.shape[1])
    y1, x1 = y0 + h, x0 + w
    canvas[y0:y1, x0:x1] = tile_yx
    mask[y0:y1, x0:x1] = 1.0
    return canvas, mask


def _overlap_only_masks(
    *,
    layout: CanvasLayout,
    tile_shape_yx: tuple[int, int],
    fixed_tile_index: int,
    moving_tile_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed/moving masks constrained to their nominal overlap strip."""
    fy0, fy1, fx0, fx1 = _tile_bbox(
        layout=layout,
        tile_index=fixed_tile_index,
        tile_shape_yx=tile_shape_yx,
    )
    my0, my1, mx0, mx1 = _tile_bbox(
        layout=layout,
        tile_index=moving_tile_index,
        tile_shape_yx=tile_shape_yx,
    )

    oy0 = max(fy0, my0)
    oy1 = min(fy1, my1)
    ox0 = max(fx0, mx0)
    ox1 = min(fx1, mx1)
    if oy1 <= oy0 or ox1 <= ox0:
        raise ValueError(
            "No nominal overlap exists between fixed and moving tile. "
            "Choose an anchor adjacent to each moving tile or increase overlap."
        )

    fixed_mask = np.zeros(layout.shape_yx, dtype=np.float32)
    moving_mask = np.zeros(layout.shape_yx, dtype=np.float32)
    fixed_mask[oy0:oy1, ox0:ox1] = 1.0
    moving_mask[oy0:oy1, ox0:ox1] = 1.0
    return fixed_mask, moving_mask


def _overlap_slices(
    *,
    layout: CanvasLayout,
    tile_shape_yx: tuple[int, int],
    fixed_tile_index: int,
    moving_tile_index: int,
    pad_pixels: int = 0,
) -> tuple[slice, slice] | None:
    """Return canvas slices that tightly cover nominal overlap, optionally padded."""
    fy0, fy1, fx0, fx1 = _tile_bbox(
        layout=layout,
        tile_index=fixed_tile_index,
        tile_shape_yx=tile_shape_yx,
    )
    my0, my1, mx0, mx1 = _tile_bbox(
        layout=layout,
        tile_index=moving_tile_index,
        tile_shape_yx=tile_shape_yx,
    )
    oy0 = max(fy0, my0)
    oy1 = min(fy1, my1)
    ox0 = max(fx0, mx0)
    ox1 = min(fx1, mx1)
    if oy1 <= oy0 or ox1 <= ox0:
        return None

    pad = max(0, int(pad_pixels))
    oy0 = max(0, oy0 - pad)
    ox0 = max(0, ox0 - pad)
    oy1 = min(int(layout.shape_yx[0]), oy1 + pad)
    ox1 = min(int(layout.shape_yx[1]), ox1 + pad)
    return slice(int(oy0), int(oy1)), slice(int(ox0), int(ox1))


def _blend_profile_x(
    *,
    width: int,
    tile_index: int,
    tile_count: int,
    overlap_fraction: float,
    min_edge_weight: float,
) -> np.ndarray:
    """Build a 1D feather profile across X for one tile position."""
    width_int = int(width)
    profile = np.ones(width_int, dtype=np.float32)
    if tile_count <= 1 or overlap_fraction <= 0.0 or width_int <= 1:
        return profile

    overlap_px = int(round(float(width_int) * float(overlap_fraction)))
    overlap_px = max(1, min(overlap_px, max(1, width_int - 1)))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap_px, dtype=np.float32))
    floor = float(np.clip(float(min_edge_weight), 0.0, 1.0))

    if int(tile_index) > 0:
        profile[:overlap_px] = np.maximum(ramp, floor)
    if int(tile_index) < int(tile_count) - 1:
        profile[-overlap_px:] = np.maximum(ramp[::-1], floor)
    return profile


def _place_profile_on_canvas(
    *,
    layout: CanvasLayout,
    tile_index: int,
    tile_shape_yx: tuple[int, int],
    profile_x: np.ndarray,
) -> np.ndarray:
    """Place an X-profile into the tile footprint on a full canvas."""
    canvas = np.zeros(layout.shape_yx, dtype=np.float32)
    y0, x0 = _tile_anchor(layout, int(tile_index))
    h, w = int(tile_shape_yx[0]), int(tile_shape_yx[1])
    y1, x1 = y0 + h, x0 + w
    canvas[y0:y1, x0:x1] = np.asarray(profile_x, dtype=np.float32)[np.newaxis, :]
    return canvas


def _estimate_intensity_correction(
    *,
    fixed_values: np.ndarray,
    moving_values: np.ndarray,
    overlap_mask: np.ndarray,
    mode: str,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Estimate gain/offset to match moving intensities to fixed intensities."""
    mode_value = str(mode).strip().lower()
    if mode_value == "none":
        return 1.0, 0.0, 0

    valid = (
        np.asarray(overlap_mask, dtype=bool)
        & np.isfinite(fixed_values)
        & np.isfinite(moving_values)
        & (fixed_values > 0)
        & (moving_values > 0)
    )
    count = int(np.count_nonzero(valid))
    if count < 1024:
        return 1.0, 0.0, count

    fixed = np.asarray(fixed_values[valid], dtype=np.float32)
    moving = np.asarray(moving_values[valid], dtype=np.float32)
    if int(sample_size) > 0 and fixed.size > int(sample_size):
        idx = rng.choice(fixed.size, size=int(sample_size), replace=False)
        fixed = fixed[idx]
        moving = moving[idx]

    moving_lo, moving_hi = np.percentile(moving, [5.0, 95.0])
    fixed_lo, fixed_hi = np.percentile(fixed, [5.0, 95.0])

    gain = 1.0
    if mode_value in {"gain", "gain-offset"}:
        moving_span = float(moving_hi - moving_lo)
        fixed_span = float(fixed_hi - fixed_lo)
        if moving_span > 1e-6 and fixed_span > 0.0:
            gain = float(np.clip(fixed_span / moving_span, 0.1, 10.0))

    offset = 0.0
    if mode_value == "gain-offset":
        moving_median = float(np.median(moving))
        fixed_median = float(np.median(fixed))
        offset = float(fixed_median - gain * moving_median)
        span = max(1.0, float(fixed_hi - fixed_lo))
        offset = float(np.clip(offset, -5.0 * span, 5.0 * span))

    return gain, offset, int(fixed.size)


def _safe_blend(sum_image: np.ndarray, weight_image: np.ndarray) -> np.ndarray:
    """Compute weighted average image with zero-safe division."""
    blended = np.zeros_like(sum_image, dtype=np.float32)
    np.divide(
        sum_image,
        np.maximum(weight_image, 1e-6),
        out=blended,
        where=weight_image > 0,
    )
    return blended


def _compute_crop_slices(weight_image: np.ndarray) -> tuple[slice, slice]:
    """Compute tight crop around nonzero coverage."""
    ys, xs = np.nonzero(weight_image > 0)
    if ys.size == 0 or xs.size == 0:
        return slice(0, int(weight_image.shape[0])), slice(
            0, int(weight_image.shape[1])
        )
    return slice(int(ys.min()), int(ys.max()) + 1), slice(
        int(xs.min()), int(xs.max()) + 1
    )


def _cast_to_dtype(data: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """Cast blended float data to an output dtype with clipping for integers."""
    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        clipped = np.clip(np.rint(data), info.min, info.max)
        return clipped.astype(out_dtype, copy=False)
    return np.asarray(data, dtype=out_dtype)


def _register_positions(
    *,
    reg_channel_paths: Mapping[int, Path],
    overlap_fraction: float,
    margin_fraction: float,
    aff_iterations: tuple[int, ...],
    aff_shrink_factors: tuple[int, ...],
    aff_smoothing_sigmas: tuple[int, ...],
    aff_random_sampling_rate: float,
    anchor_position: int | None,
    verbose: bool,
) -> tuple[list[int], CanvasLayout, tuple[slice, slice], dict[int, Any]]:
    """Estimate rigid transforms from one channel against a fixed anchor."""
    positions = sorted(int(position) for position in reg_channel_paths)
    if not positions:
        raise ValueError("No registration-channel tiles were provided.")

    first_tile = _load_2d_tiff(reg_channel_paths[positions[0]])
    tile_shape = tuple(int(v) for v in first_tile.shape)
    layout = _build_canvas_layout(
        tile_shape_yx=tile_shape,
        tile_count=len(positions),
        overlap_fraction=overlap_fraction,
        margin_fraction=margin_fraction,
    )

    if anchor_position is None:
        anchor_position = int(positions[len(positions) // 2])
    if anchor_position not in reg_channel_paths:
        raise ValueError(
            f"Anchor position p{int(anchor_position):04d} is missing from registration channel."
        )

    position_to_index = {
        int(position): index for index, position in enumerate(positions)
    }
    anchor_index = int(position_to_index[int(anchor_position)])
    anchor_tile = _load_2d_tiff(reg_channel_paths[int(anchor_position)])
    anchor_canvas, anchor_mask_canvas = _place_tile_on_canvas(
        tile_yx=anchor_tile,
        layout=layout,
        tile_index=anchor_index,
    )

    fixed_img = ants.from_numpy(anchor_canvas)
    transforms_by_position: dict[int, Any] = {}
    coverage_weight = np.asarray(anchor_mask_canvas, dtype=np.float32)
    transforms_by_position[int(anchor_position)] = None
    LOGGER.info("Position p%04d fixed as anchor tile.", int(anchor_position))

    registered_count = 1
    for position in positions:
        if int(position) == int(anchor_position):
            continue
        tile_index = int(position_to_index[int(position)])
        tile = _load_2d_tiff(reg_channel_paths[position])
        moving_canvas, moving_mask_canvas = _place_tile_on_canvas(
            tile_yx=tile,
            layout=layout,
            tile_index=tile_index,
        )
        fixed_mask, moving_mask = _overlap_only_masks(
            layout=layout,
            tile_shape_yx=tile_shape,
            fixed_tile_index=anchor_index,
            moving_tile_index=tile_index,
        )

        moving_img = ants.from_numpy(moving_canvas)
        fixed_mask_img = ants.from_numpy(fixed_mask)
        moving_mask_img = ants.from_numpy(moving_mask)

        registration = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            fixed_mask=fixed_mask_img,
            moving_mask=moving_mask_img,
            type_of_transform="Rigid",
            initial_transform="Identity",
            aff_metric="mattes",
            aff_sampling=32,
            aff_random_sampling_rate=float(aff_random_sampling_rate),
            aff_iterations=tuple(int(v) for v in aff_iterations),
            aff_shrink_factors=tuple(int(v) for v in aff_shrink_factors),
            aff_smoothing_sigmas=tuple(int(v) for v in aff_smoothing_sigmas),
            mask_all_stages=True,
            verbose=bool(verbose),
        )
        transform = ants.read_transform(registration["fwdtransforms"][0])
        transforms_by_position[position] = transform

        warped_mask_img = transform.apply_to_image(
            image=ants.from_numpy(np.asarray(moving_mask_canvas, dtype=np.float32)),
            reference=fixed_img,
            interpolation="nearestNeighbor",
        )
        warped_mask = np.asarray(warped_mask_img.numpy(), dtype=np.float32)
        coverage_weight += warped_mask
        registered_count += 1
        LOGGER.info(
            "Registered position p%04d (%d/%d).",
            int(position),
            int(registered_count),
            int(len(positions)),
        )

    crop_slices = _compute_crop_slices(coverage_weight)
    return positions, layout, crop_slices, transforms_by_position


def _stitch_channels(
    *,
    channel_paths: Mapping[int, Mapping[int, Path]],
    positions: Sequence[int],
    layout: CanvasLayout,
    transforms_by_position: Mapping[int, Any],
    crop_slices: tuple[slice, slice],
    output_dtype: np.dtype[Any],
    overlap_fraction: float,
    intensity_normalization: str,
    norm_sample_size: int,
    blend_mode: str,
    feather_min_edge_weight: float,
) -> tuple[np.ndarray, list[int]]:
    """Apply position transforms to all channels and blend to a mosaic stack."""
    channels = sorted(int(channel) for channel in channel_paths)
    channel_mosaics: list[np.ndarray] = []
    reference_img = ants.from_numpy(np.zeros(layout.shape_yx, dtype=np.float32))
    blend_mode_value = str(blend_mode).strip().lower()
    normalization_mode = str(intensity_normalization).strip().lower()
    position_to_index = {
        int(position): int(idx) for idx, position in enumerate(positions)
    }
    anchor_candidates = [
        int(position)
        for position in positions
        if transforms_by_position.get(int(position)) is None
    ]
    if len(anchor_candidates) != 1:
        raise ValueError(
            "Expected exactly one fixed anchor position (transform is None). "
            f"Found {anchor_candidates}."
        )
    anchor_position = int(anchor_candidates[0])
    anchor_tile_index = int(position_to_index[anchor_position])
    rng = np.random.default_rng(0)

    for channel in channels:
        sum_image = np.zeros(layout.shape_yx, dtype=np.float32)
        weight_image = np.zeros(layout.shape_yx, dtype=np.float32)
        channel_corrections: dict[int, tuple[float, float, int]] = {}

        anchor_path = channel_paths[channel].get(anchor_position)
        if anchor_path is None:
            raise ValueError(
                f"Channel c{channel:04d} is missing anchor position p{anchor_position:04d}."
            )
        anchor_tile = _load_2d_tiff(anchor_path)
        anchor_shape = (int(anchor_tile.shape[0]), int(anchor_tile.shape[1]))
        anchor_canvas, anchor_mask_canvas = _place_tile_on_canvas(
            tile_yx=anchor_tile,
            layout=layout,
            tile_index=anchor_tile_index,
        )

        for tile_index, position in enumerate(positions):
            tile_path = channel_paths[channel].get(position)
            if tile_path is None:
                raise ValueError(
                    f"Channel c{channel:04d} is missing position p{position:04d}."
                )
            tile = _load_2d_tiff(tile_path)
            tile_shape = (int(tile.shape[0]), int(tile.shape[1]))
            moving_canvas, moving_mask_canvas = _place_tile_on_canvas(
                tile_yx=tile,
                layout=layout,
                tile_index=tile_index,
            )

            if blend_mode_value == "feather":
                profile_x = _blend_profile_x(
                    width=tile_shape[1],
                    tile_index=tile_index,
                    tile_count=len(positions),
                    overlap_fraction=overlap_fraction,
                    min_edge_weight=feather_min_edge_weight,
                )
                blend_canvas = _place_profile_on_canvas(
                    layout=layout,
                    tile_index=tile_index,
                    tile_shape_yx=tile_shape,
                    profile_x=profile_x,
                )
            else:
                blend_canvas = np.asarray(moving_mask_canvas, dtype=np.float32)

            transform = transforms_by_position.get(position)
            if transform is None:
                warped = np.asarray(moving_canvas, dtype=np.float32)
                warped_mask = np.asarray(moving_mask_canvas, dtype=np.float32)
                warped_weight = np.asarray(blend_canvas, dtype=np.float32) * warped_mask
            else:
                moving_img = ants.from_numpy(
                    np.asarray(moving_canvas, dtype=np.float32)
                )
                moving_mask_img = ants.from_numpy(
                    np.asarray(moving_mask_canvas, dtype=np.float32)
                )
                blend_img = ants.from_numpy(np.asarray(blend_canvas, dtype=np.float32))
                warped_img = transform.apply_to_image(
                    image=moving_img,
                    reference=reference_img,
                    interpolation="linear",
                )
                warped_mask_img = transform.apply_to_image(
                    image=moving_mask_img,
                    reference=reference_img,
                    interpolation="nearestNeighbor",
                )
                warped_blend_img = transform.apply_to_image(
                    image=blend_img,
                    reference=reference_img,
                    interpolation="linear",
                )
                warped = np.asarray(warped_img.numpy(), dtype=np.float32)
                warped_mask = np.asarray(warped_mask_img.numpy(), dtype=np.float32)
                warped_weight = (
                    np.asarray(warped_blend_img.numpy(), dtype=np.float32) * warped_mask
                )

            gain, offset, sample_count = 1.0, 0.0, 0
            if normalization_mode != "none" and int(position) != int(anchor_position):
                overlap_region = _overlap_slices(
                    layout=layout,
                    tile_shape_yx=anchor_shape,
                    fixed_tile_index=anchor_tile_index,
                    moving_tile_index=tile_index,
                    pad_pixels=max(32, int(round(float(tile_shape[1]) * 0.02))),
                )
                if overlap_region is not None:
                    ys, xs = overlap_region
                    overlap_mask = (
                        np.asarray(anchor_mask_canvas[ys, xs], dtype=np.float32) > 0.5
                    ) & (np.asarray(warped_mask[ys, xs], dtype=np.float32) > 0.5)
                    gain, offset, sample_count = _estimate_intensity_correction(
                        fixed_values=np.asarray(
                            anchor_canvas[ys, xs], dtype=np.float32
                        ),
                        moving_values=np.asarray(warped[ys, xs], dtype=np.float32),
                        overlap_mask=overlap_mask,
                        mode=normalization_mode,
                        sample_size=int(norm_sample_size),
                        rng=rng,
                    )
                    if sample_count >= 1024:
                        warped = warped * float(gain) + float(offset)
                        np.maximum(warped, 0.0, out=warped)
            channel_corrections[int(position)] = (
                float(gain),
                float(offset),
                int(sample_count),
            )

            sum_image += warped * warped_weight
            weight_image += warped_weight

        blended = _safe_blend(sum_image, weight_image)
        cropped = blended[crop_slices[0], crop_slices[1]]
        channel_mosaics.append(_cast_to_dtype(cropped, output_dtype))

        if normalization_mode != "none":
            for position in positions:
                if int(position) == int(anchor_position):
                    continue
                gain, offset, sample_count = channel_corrections.get(
                    int(position),
                    (1.0, 0.0, 0),
                )
                LOGGER.info(
                    "Channel c%04d p%04d intensity correction: gain=%.4f offset=%.2f samples=%d",
                    int(channel),
                    int(position),
                    float(gain),
                    float(offset),
                    int(sample_count),
                )
        LOGGER.info("Stitched channel c%04d.", channel)

    stacked = np.stack(channel_mosaics, axis=0)
    return stacked, channels


def _process_projection_time(
    *,
    input_dir: Path,
    projection: str,
    time_index: int,
    by_channel: Mapping[int, Mapping[int, Path]],
    registration_channel: int,
    overlap_fraction: float,
    margin_fraction: float,
    aff_iterations: tuple[int, ...],
    aff_shrink_factors: tuple[int, ...],
    aff_smoothing_sigmas: tuple[int, ...],
    aff_random_sampling_rate: float,
    anchor_position: int | None,
    intensity_normalization: str,
    norm_sample_size: int,
    blend_mode: str,
    feather_min_edge_weight: float,
    output_suffix: str,
    verbose: bool,
) -> Path:
    """Run registration+stitching for one projection/timepoint."""
    if registration_channel not in by_channel:
        raise ValueError(
            f"Projection '{projection}', time t{time_index:04d}: "
            f"registration channel c{registration_channel:04d} is missing."
        )

    reg_channel_paths = by_channel[registration_channel]
    sample_path = next(iter(reg_channel_paths.values()))
    sample_dtype = np.asarray(tifffile.imread(str(sample_path))).dtype

    positions, layout, crop_slices, transforms_by_position = _register_positions(
        reg_channel_paths=reg_channel_paths,
        overlap_fraction=overlap_fraction,
        margin_fraction=margin_fraction,
        aff_iterations=aff_iterations,
        aff_shrink_factors=aff_shrink_factors,
        aff_smoothing_sigmas=aff_smoothing_sigmas,
        aff_random_sampling_rate=aff_random_sampling_rate,
        anchor_position=anchor_position,
        verbose=verbose,
    )

    stitched_cyx, channels = _stitch_channels(
        channel_paths=by_channel,
        positions=positions,
        layout=layout,
        transforms_by_position=transforms_by_position,
        crop_slices=crop_slices,
        output_dtype=np.dtype(sample_dtype),
        overlap_fraction=overlap_fraction,
        intensity_normalization=intensity_normalization,
        norm_sample_size=norm_sample_size,
        blend_mode=blend_mode,
        feather_min_edge_weight=feather_min_edge_weight,
    )

    output_name = f"stitched_{projection}_t{time_index:04d}_{output_suffix}.tif"
    output_path = (input_dir / output_name).resolve()
    tifffile.imwrite(
        str(output_path),
        stitched_cyx,
        photometric="minisblack",
        metadata={"axes": "CYX", "channels": [int(c) for c in channels]},
        bigtiff=True,
    )
    LOGGER.info(
        "Saved stitched mosaic: %s (shape=%s, channels=%s).",
        output_path,
        tuple(int(v) for v in stitched_cyx.shape),
        [int(c) for c in channels],
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Register per-position ClearEx MIP TIFF tiles with rigid ANTsPy "
            "registration (from one channel), then stitch/blend all channels."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing mip_<projection>_p####_t####_c####.tif tiles.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="xy",
        choices=("xy", "xz", "yz", "all"),
        help="Projection to stitch. Use 'all' to process every discovered projection.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Optional time index to process. Default processes all discovered timepoints.",
    )
    parser.add_argument(
        "--registration-channel",
        type=int,
        default=0,
        help="Channel index used to estimate rigid transforms (default: 0).",
    )
    parser.add_argument(
        "--anchor-position",
        type=int,
        default=None,
        help="Position index held fixed during registration (default: median discovered position).",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.05,
        help="Initial left-right overlap fraction between neighboring positions (default: 0.05).",
    )
    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=0.20,
        help="Canvas margin fraction relative to tile size for rigid search headroom (default: 0.20).",
    )
    parser.add_argument(
        "--aff-iterations",
        type=str,
        default="2000x1000x500x250",
        help="Rigid affine-optimizer iteration schedule (x- or comma-separated).",
    )
    parser.add_argument(
        "--aff-shrink-factors",
        type=str,
        default="8x4x2x1",
        help="Pyramid shrink-factor schedule (x- or comma-separated).",
    )
    parser.add_argument(
        "--aff-smoothing-sigmas",
        type=str,
        default="3x2x1x0",
        help="Pyramid smoothing-sigma schedule (x- or comma-separated).",
    )
    parser.add_argument(
        "--aff-random-sampling-rate",
        type=float,
        default=0.25,
        help="ANTs rigid metric random sampling rate (default: 0.25).",
    )
    parser.add_argument(
        "--intensity-normalization",
        type=str,
        default="gain-offset",
        choices=("none", "gain", "gain-offset"),
        help=(
            "Per-channel overlap-based intensity harmonization mode before blending "
            "(default: gain-offset)."
        ),
    )
    parser.add_argument(
        "--norm-sample-size",
        type=int,
        default=250000,
        help="Max sampled overlap pixels for intensity correction fit (default: 250000).",
    )
    parser.add_argument(
        "--blend-mode",
        type=str,
        default="feather",
        choices=("average", "feather"),
        help="Tile blending mode in overlaps (default: feather).",
    )
    parser.add_argument(
        "--feather-min-edge-weight",
        type=float,
        default=0.05,
        help="Minimum feather weight at overlapped tile edges (default: 0.05).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="rigid_blended_multichannel",
        help="Suffix for stitched output TIFF filename.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ANTs logging.",
    )
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    """Execute stitching workflow from parsed CLI args."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    overlap_fraction = float(args.overlap_fraction)
    if overlap_fraction <= 0.0 or overlap_fraction >= 1.0:
        raise ValueError("--overlap-fraction must be strictly between 0 and 1.")

    margin_fraction = float(args.margin_fraction)
    if margin_fraction < 0.0:
        raise ValueError("--margin-fraction must be non-negative.")
    norm_sample_size = int(args.norm_sample_size)
    if norm_sample_size < 0:
        raise ValueError("--norm-sample-size must be >= 0.")
    feather_min_edge_weight = float(args.feather_min_edge_weight)
    if feather_min_edge_weight < 0.0 or feather_min_edge_weight > 1.0:
        raise ValueError("--feather-min-edge-weight must be in [0, 1].")

    aff_iterations = _parse_schedule(args.aff_iterations, field_name="--aff-iterations")
    aff_shrink_factors = _parse_schedule(
        args.aff_shrink_factors, field_name="--aff-shrink-factors"
    )
    aff_smoothing_sigmas = _parse_schedule(
        args.aff_smoothing_sigmas, field_name="--aff-smoothing-sigmas"
    )
    if not (
        len(aff_iterations) == len(aff_shrink_factors) == len(aff_smoothing_sigmas)
    ):
        raise ValueError(
            "Pyramid schedules must have matching lengths: "
            "--aff-iterations, --aff-shrink-factors, --aff-smoothing-sigmas."
        )

    records = _discover_tiles(input_dir)
    if not records:
        raise FileNotFoundError(
            f"No matching MIP TIFF tiles found in {input_dir} "
            "(expected mip_<projection>_p####_t####_c####.tif)."
        )

    grouped = _group_tiles(records)
    if args.projection == "all":
        projections = sorted(grouped)
    else:
        projection = str(args.projection).lower()
        if projection not in grouped:
            raise ValueError(
                f"Projection '{projection}' was not found in {input_dir}. "
                f"Available: {sorted(grouped)}"
            )
        projections = [projection]

    outputs: list[Path] = []
    for projection in projections:
        times = sorted(grouped[projection])
        if args.time_index is not None:
            time_value = int(args.time_index)
            if time_value not in grouped[projection]:
                raise ValueError(
                    f"Projection '{projection}' does not contain time t{time_value:04d}."
                )
            times = [time_value]

        for time_index in times:
            LOGGER.info(
                "Processing projection=%s, t%04d, registration_channel=c%04d",
                projection,
                time_index,
                int(args.registration_channel),
            )
            output = _process_projection_time(
                input_dir=input_dir,
                projection=projection,
                time_index=time_index,
                by_channel=grouped[projection][time_index],
                registration_channel=int(args.registration_channel),
                overlap_fraction=overlap_fraction,
                margin_fraction=margin_fraction,
                aff_iterations=aff_iterations,
                aff_shrink_factors=aff_shrink_factors,
                aff_smoothing_sigmas=aff_smoothing_sigmas,
                aff_random_sampling_rate=float(args.aff_random_sampling_rate),
                anchor_position=(
                    None if args.anchor_position is None else int(args.anchor_position)
                ),
                intensity_normalization=str(args.intensity_normalization)
                .strip()
                .lower(),
                norm_sample_size=norm_sample_size,
                blend_mode=str(args.blend_mode).strip().lower(),
                feather_min_edge_weight=feather_min_edge_weight,
                output_suffix=str(args.output_suffix).strip() or "stitched",
                verbose=bool(args.verbose),
            )
            outputs.append(output)
    return outputs


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    outputs = run(args)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
