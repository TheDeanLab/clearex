import subprocess
import re
from pathlib import Path
import os
from PIL import Image


def renumber_frames(frames_dir: str, pattern: str = "frame_", prefix: str = "frame_"):
    """Renumber frames to be contiguous starting from 00000."""
    frames_path = Path(frames_dir)

    # Find all frames matching pattern
    frame_files = sorted(
        frames_path.glob(f"{pattern}*.png"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()),
    )

    print(f"Found {len(frame_files)} frames to renumber")

    # Rename to temporary names first (avoid collisions)
    temp_names = []
    for i, old_path in enumerate(frame_files):
        temp_name = frames_path / f"_temp_{i:05d}.png"
        old_path.rename(temp_name)
        temp_names.append(temp_name)

    # Rename from temp to final contiguous names
    for i, temp_path in enumerate(temp_names):
        new_name = frames_path / f"{prefix}{i:05d}.png"
        temp_path.rename(new_name)
        if i % 100 == 0:
            print(f"  Renamed {i}/{len(temp_names)} frames...")

    print(f"✅ Renumbered {len(frame_files)} frames")
    return len(frame_files)


def verify_frames(frames_dir):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

    print(f"Checking {len(frame_files)} frames...")
    for i, frame in enumerate(frame_files):
        try:
            img = Image.open(os.path.join(frames_dir, frame))
            img.verify()
        except Exception as e:
            print(f"Corrupted frame: {frame} - {e}")

    print("Frame check complete")


def export_frames_to_avi(
    frames_dir: str,
    output_path: str,
    fps: int = 15,
    quality: int = 2,
    pattern: str = "frame_%05d.png",
):
    """Export PNG frames to AVI using FFmpeg with MJPEG codec."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load module and run FFmpeg in same shell
    cmd = f"""
    module load ffmpeg/7.1 && \
    ffmpeg -y -framerate {fps} \
    -i {os.path.join(frames_dir, pattern)} \
    -c:v mjpeg -q:v {quality} \
    -pix_fmt yuvj420p {output_path}
    """

    print(f"Running command with module load:")
    print(cmd)

    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",  # Use bash to support module command
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}")
        raise RuntimeError("FFmpeg conversion failed")

    print(f"✅ Created {output_path}")
    return output_path


def export_frames_to_mp4(
    frames_dir: str,
    output_path: str,
    fps: int = 15,
    crf: int = 23,
    pattern: str = "frame_%05d.png",
):
    """Export PNG frames to MP4 using FFmpeg with H.264 codec."""

    renumber_frames(frames_dir)
    verify_frames(frames_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = f"""
    module load ffmpeg/7.1 && \
    ffmpeg -y -framerate {fps} \
    -i {os.path.join(frames_dir, pattern)} \
    -c:v libx264 -crf {crf} \
    -pix_fmt yuv420p -preset veryslow -tune stillimage \
    -profile:v high -level 4.1 -movflags +faststart \
    -r {fps} -g {fps} \
    {output_path}
    """

    print(f"Running command with module load:")
    print(cmd)

    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}")
        raise RuntimeError("FFmpeg conversion failed")

    print(f"✅ Created {output_path}")
    return output_path
