from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from io_video import (
    VideoMetadata,
    _apply_output_geometry,
    _resolve_output_geometry,
    measure_video_fps,
    read_video_metadata,
)


def _write_synthetic_video(path: Path, fps: float, frame_count: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open synthetic video writer for tests.")
    try:
        for index in range(frame_count):
            frame = np.full((64, 64, 3), index * 10 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_measure_video_fps_returns_value_for_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic_30fps.mp4"
    _write_synthetic_video(video_path, fps=30.0, frame_count=20)

    measured = measure_video_fps(video_path, max_samples=15)

    assert measured is not None
    assert 25.0 <= measured <= 35.0


def test_measure_video_fps_returns_none_for_short_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic_short.mp4"
    _write_synthetic_video(video_path, fps=30.0, frame_count=1)

    assert measure_video_fps(video_path, max_samples=10) is None


def test_measure_video_fps_returns_none_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.mp4"

    with pytest.raises(FileNotFoundError):
        measure_video_fps(missing_path)


def test_read_video_metadata_returns_fps_for_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic_meta.mp4"
    _write_synthetic_video(video_path, fps=24.0, frame_count=15)

    metadata = read_video_metadata(video_path)

    assert metadata.width == 64
    assert metadata.height == 64
    assert metadata.frame_count == 15
    assert 20.0 <= metadata.fps <= 30.0


def test_portrait_720_geometry_letterboxes_without_cropping() -> None:
    metadata = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=1,
        codec="mp4v",
    )
    geometry = _resolve_output_geometry(metadata, target_resolution=0, output_mode="portrait-720")
    frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)

    output = _apply_output_geometry(frame, geometry)

    assert geometry.metadata.width == 720
    assert geometry.metadata.height == 1280
    assert output.shape == (1280, 720, 3)
    assert output[geometry.pad_y + 10, 10].mean() == 120
    assert output[10, 10].mean() < 40
