from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


@dataclass(frozen=True)
class ValidationRunPaths:
    run_id: str
    root: Path
    videos_dir: Path
    screenshots_dir: Path
    reports_dir: Path


def create_validation_run(label: str, base_dir: Path = Path("outputs/validation/runs")) -> ValidationRunPaths:
    safe_label = _safe_label(label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{safe_label}" if safe_label else timestamp
    root = base_dir / run_id
    videos_dir = root / "videos"
    screenshots_dir = root / "screenshots"
    reports_dir = root / "reports"

    for directory in (videos_dir, screenshots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ValidationRunPaths(
        run_id=run_id,
        root=root,
        videos_dir=videos_dir,
        screenshots_dir=screenshots_dir,
        reports_dir=reports_dir,
    )


def save_validation_screenshots(
    video_path: Path,
    screenshots_dir: Path,
    label: str,
    max_screenshots: int = 5,
) -> list[Path]:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return []

    if frame_count <= max_screenshots:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = sorted(
            {
                int(round(index * (frame_count - 1) / max(1, max_screenshots - 1)))
                for index in range(max_screenshots)
            }
        )

    saved_paths: list[Path] = []
    safe_label = _safe_label(label) or "frame"
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        screenshot_path = screenshots_dir / f"{safe_label}_frame_{frame_index:05d}.jpg"
        if cv2.imwrite(str(screenshot_path), frame):
            saved_paths.append(screenshot_path)

    capture.release()
    return saved_paths


def _safe_label(label: str) -> str:
    normalized = label.strip().lower().replace(" ", "_")
    safe = "".join(character for character in normalized if character.isalnum() or character in {"_", "-"})
    return safe.strip("_-")
