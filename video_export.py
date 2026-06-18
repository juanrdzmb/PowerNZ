from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _find_ffmpeg() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_is_available() -> bool:
    return _find_ffmpeg() is not None


def convert_to_mobile_mp4(
    input_path: str | Path,
    output_path: str | Path,
    max_dimension: int = 1920,
) -> None:
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg or imageio-ffmpeg is required to create a mobile-compatible MP4. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
    ]

    if max_dimension > 0:
        command.extend(
            [
                "-vf",
                (
                    f"scale='min({max_dimension},iw)':'min({max_dimension},ih)':"
                    "force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"
                ),
            ]
        )

    command.append(str(output_path))
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg conversion failed: {message}")


def make_mobile_compatible_in_place(video_path: str | Path, max_dimension: int = 1920) -> None:
    video_path = Path(video_path)
    temporary_path = video_path.with_name(f"{video_path.stem}.mobile.tmp{video_path.suffix}")

    try:
        convert_to_mobile_mp4(video_path, temporary_path, max_dimension=max_dimension)
        temporary_path.replace(video_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
