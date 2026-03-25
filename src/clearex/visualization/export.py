#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

"""Movie-encoding helpers for rendered napari frame sequences."""

from __future__ import annotations

# Standard Library Imports
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional, Sequence, Union

# Third Party Imports
from PIL import Image

_FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.png$")


def _resolve_ffmpeg_runner() -> tuple[str, ...]:
    """Resolve the ffmpeg execution prefix for the current environment.

    Returns
    -------
    tuple[str, ...]
        Command prefix used to invoke ``ffmpeg`` directly or through the
        cluster module environment.
    """
    if shutil.which("ffmpeg") is not None:
        return ("ffmpeg",)
    return (
        "bash",
        "-lc",
        'module load ffmpeg/7.1 >/dev/null 2>&1 && ffmpeg "$@"',
        "ffmpeg",
    )


def _run_ffmpeg(command: Sequence[str]) -> None:
    """Run one ffmpeg command with direct-binary or module-based fallback.

    Parameters
    ----------
    command : sequence of str
        ``ffmpeg`` argument vector excluding the executable itself.

    Returns
    -------
    None
        Executes one subprocess for its side effects.

    Raises
    ------
    RuntimeError
        If ``ffmpeg`` exits with a non-zero status code.
    """
    prefix = _resolve_ffmpeg_runner()
    if len(prefix) == 1:
        completed = subprocess.run(
            list(prefix) + list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        completed = subprocess.run(
            list(prefix) + list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        stderr_text = str(completed.stderr).strip()
        raise RuntimeError(f"ffmpeg failed: {stderr_text or 'unknown error'}")


def verify_png_frame_directory(frames_directory: Union[str, Path]) -> int:
    """Verify that a frame directory contains contiguous readable PNG frames.

    Parameters
    ----------
    frames_directory : str or pathlib.Path
        Directory expected to contain ``frame_000000.png`` style PNG files.

    Returns
    -------
    int
        Number of validated frames.

    Raises
    ------
    ValueError
        If the directory is missing, empty, misnumbered, or contains invalid
        PNG files.
    """
    frames_path = Path(frames_directory).expanduser().resolve()
    if not frames_path.is_dir():
        raise ValueError(f"Frame directory '{frames_path}' does not exist.")
    frame_files = sorted(
        file_path
        for file_path in frames_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".png"
    )
    if not frame_files:
        raise ValueError(f"Frame directory '{frames_path}' contains no PNG frames.")
    expected_index = 0
    for file_path in frame_files:
        match = _FRAME_NAME_RE.match(file_path.name)
        if match is None:
            raise ValueError(
                f"Frame '{file_path.name}' does not match the expected "
                "frame_000000.png naming pattern."
            )
        frame_index = int(match.group(1))
        if frame_index != expected_index:
            raise ValueError(
                f"Frame numbering is not contiguous in '{frames_path}': "
                f"expected {expected_index:06d}, found {frame_index:06d}."
            )
        with Image.open(file_path) as image:
            image.verify()
        expected_index += 1
    return int(len(frame_files))


def _resize_filter(resize_xy: Optional[Sequence[int]]) -> Optional[str]:
    """Build an ffmpeg scale filter from an optional ``(x, y)`` size pair.

    Parameters
    ----------
    resize_xy : sequence of int, optional
        Optional output size in ``(x, y)`` order.

    Returns
    -------
    str, optional
        ``ffmpeg`` filter expression when resizing is requested; otherwise
        ``None``.

    Raises
    ------
    ValueError
        If ``resize_xy`` does not define exactly two integers.
    """
    if resize_xy is None:
        return None
    values = tuple(int(value) for value in resize_xy)
    if len(values) != 2:
        raise ValueError("resize_xy must contain exactly two integers.")
    width = max(1, int(values[0]))
    height = max(1, int(values[1]))
    return f"scale={width}:{height}:flags=lanczos"


def _resolve_output_pixel_format(
    pixel_format: Optional[str],
    *,
    default: str,
) -> str:
    """Resolve one optional ffmpeg pixel format with a safe default.

    Parameters
    ----------
    pixel_format : str, optional
        Requested ffmpeg pixel format.
    default : str
        Format used when ``pixel_format`` is missing or blank.

    Returns
    -------
    str
        Explicit ffmpeg pixel format value.
    """
    normalized_value = str(pixel_format or "").strip()
    if not normalized_value or normalized_value.casefold() in {"none", "null"}:
        return str(default).strip()
    return normalized_value


def compile_png_frames_to_mp4(
    *,
    frames_directory: Union[str, Path],
    output_path: Union[str, Path],
    fps: int,
    crf: int = 18,
    preset: str = "slow",
    pixel_format: Optional[str] = None,
    resize_xy: Optional[Sequence[int]] = None,
) -> str:
    """Compile PNG frames into an H.264 MP4 movie.

    Parameters
    ----------
    frames_directory : str or pathlib.Path
        Directory containing validated PNG frames.
    output_path : str or pathlib.Path
        Destination movie path.
    fps : int
        Output frame rate.
    crf : int, default=18
        x264 constant-rate-factor quality control.
    preset : str, default="slow"
        x264 speed/quality preset.
    pixel_format : str, optional
        Explicit output pixel format. Defaults to ``yuv420p``.
    resize_xy : sequence of int, optional
        Optional encode-time output size in ``(x, y)`` order.

    Returns
    -------
    str
        Resolved output movie path.

    Raises
    ------
    ValueError
        If the frame directory is invalid.
    RuntimeError
        If ``ffmpeg`` fails.
    """
    verify_png_frame_directory(frames_directory)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "-y",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        str(Path(frames_directory).expanduser().resolve() / "frame_%06d.png"),
    ]
    filter_graph = _resize_filter(resize_xy)
    if filter_graph:
        command.extend(["-vf", filter_graph])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(max(0, min(51, int(crf)))),
            "-preset",
            str(preset).strip() or "slow",
            "-pix_fmt",
            _resolve_output_pixel_format(pixel_format, default="yuv420p"),
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run_ffmpeg(command)
    return str(output)


def compile_png_frames_to_prores(
    *,
    frames_directory: Union[str, Path],
    output_path: Union[str, Path],
    fps: int,
    profile: int = 3,
    pixel_format: Optional[str] = None,
    resize_xy: Optional[Sequence[int]] = None,
) -> str:
    """Compile PNG frames into a ProRes MOV master.

    Parameters
    ----------
    frames_directory : str or pathlib.Path
        Directory containing validated PNG frames.
    output_path : str or pathlib.Path
        Destination movie path.
    fps : int
        Output frame rate.
    profile : int, default=3
        ProRes profile index passed to ``prores_ks``.
    pixel_format : str, optional
        Explicit output pixel format. Defaults to ``yuv422p10le``.
    resize_xy : sequence of int, optional
        Optional encode-time output size in ``(x, y)`` order.

    Returns
    -------
    str
        Resolved output movie path.

    Raises
    ------
    ValueError
        If the frame directory is invalid.
    RuntimeError
        If ``ffmpeg`` fails.
    """
    verify_png_frame_directory(frames_directory)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "-y",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        str(Path(frames_directory).expanduser().resolve() / "frame_%06d.png"),
    ]
    filter_graph = _resize_filter(resize_xy)
    if filter_graph:
        command.extend(["-vf", filter_graph])
    command.extend(
        [
            "-c:v",
            "prores_ks",
            "-profile:v",
            str(max(0, min(5, int(profile)))),
            "-pix_fmt",
            _resolve_output_pixel_format(pixel_format, default="yuv422p10le"),
            str(output),
        ]
    )
    _run_ffmpeg(command)
    return str(output)
