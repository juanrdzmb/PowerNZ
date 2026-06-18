from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterator, Literal

import cv2
import numpy as np


Frame = np.ndarray
FrameProcessor = Callable[[Frame, int], Frame]
FrameAnalyzer = Callable[[Frame, int], None]
OutputMode = Literal["portrait-720", "source"]


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str


@dataclass(frozen=True)
class OutputGeometry:
    metadata: VideoMetadata
    scale: float
    resized_width: int
    resized_height: int
    pad_x: int = 0
    pad_y: int = 0

    @property
    def needs_resize(self) -> bool:
        return abs(self.scale - 1.0) > 1e-6 or self.letterboxed

    @property
    def letterboxed(self) -> bool:
        return self.pad_x != 0 or self.pad_y != 0


class VideoReader:
    def __init__(self, input_path: str | Path) -> None:
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_path}")

        self._capture = cv2.VideoCapture(str(self.input_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.input_path}")

        self.metadata = self._read_metadata()

    def _read_metadata(self) -> VideoMetadata:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            codec="mp4v",
        )

    def frames(self) -> Iterator[tuple[int, Frame]]:
        frame_index = 0

        while True:
            ok, frame = self._capture.read()
            if not ok:
                break

            yield frame_index, frame
            frame_index += 1

    def sample_frames_with_timestamps(
        self,
        max_samples: int,
    ) -> list[tuple[int, Frame, float]]:
        samples: list[tuple[int, Frame, float]] = []

        for frame_index, frame in self.frames():
            if frame_index >= max_samples:
                break
            timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            samples.append((frame_index, frame, timestamp_ms))

        return samples

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class VideoWriter:
    def __init__(self, output_path: str | Path, metadata: VideoMetadata) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*metadata.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height),
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Could not create output video: {self.output_path}")

    def write(self, frame: Frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _even_dimension(value: float) -> int:
    dimension = max(2, int(round(value)))
    return dimension if dimension % 2 == 0 else dimension - 1


def _resolve_output_geometry(
    original_metadata: VideoMetadata,
    target_resolution: int,
    output_mode: OutputMode = "source",
) -> OutputGeometry:
    """Return the output geometry used by both analysis and rendering.

    ``portrait-720`` places the full source frame inside a 720x1280 canvas without
    cropping. ``source`` keeps the original geometry unless ``target_resolution``
    asks for a longest-side resize.
    """
    original_width = original_metadata.width
    original_height = original_metadata.height

    if output_mode == "portrait-720":
        canvas_width = 720
        canvas_height = 1280
        scale = min(canvas_width / max(1, original_width), canvas_height / max(1, original_height))
        out_width = min(canvas_width, _even_dimension(original_width * scale))
        out_height = min(canvas_height, _even_dimension(original_height * scale))
        pad_x = (canvas_width - out_width) // 2
        pad_y = (canvas_height - out_height) // 2
        metadata = VideoMetadata(
            width=canvas_width,
            height=canvas_height,
            fps=original_metadata.fps,
            frame_count=original_metadata.frame_count,
            codec=original_metadata.codec,
        )
        return OutputGeometry(metadata, scale, out_width, out_height, pad_x, pad_y)

    if target_resolution > 0:
        larger = max(original_width, original_height)
        smaller = min(original_width, original_height)
        if larger > target_resolution:
            scale = target_resolution / larger
            min_smaller = target_resolution * 0.55
            if smaller * scale < min_smaller:
                scale = min_smaller / smaller
            out_width = _even_dimension(original_width * scale)
            out_height = _even_dimension(original_height * scale)
            metadata = VideoMetadata(
                width=out_width,
                height=out_height,
                fps=original_metadata.fps,
                frame_count=original_metadata.frame_count,
                codec=original_metadata.codec,
            )
            return OutputGeometry(metadata, scale, out_width, out_height)

    return OutputGeometry(original_metadata, 1.0, original_width, original_height)


def _apply_output_geometry(frame: Frame, geometry: OutputGeometry) -> Frame:
    if not geometry.needs_resize:
        return frame

    resized = cv2.resize(
        frame,
        (geometry.resized_width, geometry.resized_height),
        interpolation=cv2.INTER_AREA,
    )
    if not geometry.letterboxed:
        return resized

    canvas = np.full(
        (geometry.metadata.height, geometry.metadata.width, 3),
        (18, 18, 20),
        dtype=frame.dtype,
    )
    y1 = geometry.pad_y
    x1 = geometry.pad_x
    canvas[y1:y1 + geometry.resized_height, x1:x1 + geometry.resized_width] = resized
    return canvas


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    process_frame: FrameProcessor,
    max_frames: int = 0,
    target_resolution: int = 0,
    output_mode: OutputMode = "source",
) -> VideoMetadata:
    with VideoReader(input_path) as reader:
        geometry = _resolve_output_geometry(reader.metadata, target_resolution, output_mode)
        metadata = geometry.metadata

        with VideoWriter(output_path, metadata) as writer:
            for frame_index, frame in reader.frames():
                if max_frames > 0 and frame_index >= max_frames:
                    break

                processed_frame = process_frame(_apply_output_geometry(frame, geometry), frame_index)
                writer.write(processed_frame)

    return metadata


def process_video_two_pass(
    input_path: str | Path,
    output_path: str | Path,
    analyze_frame: FrameAnalyzer,
    render_frame: FrameProcessor,
    on_analysis_complete: Callable[[int], None] | None = None,
    max_frames: int = 0,
    target_resolution: int = 0,
    output_mode: OutputMode = "source",
) -> VideoMetadata:
    """Procesa el video dos veces para dibujar el overlay con el clip ya analizado.

    La primera pasada ejecuta inferencia sin renderizar. Despues ``on_analysis_complete``
    calcula totales como reps, series suavizadas y escala global. La segunda pasada vuelve
    a leer los frames y dibuja el video final. Ambas pasadas usan la misma geometria para
    que los resultados cacheados coincidan por ``frame_index``.
    """
    with VideoReader(input_path) as reader:
        geometry = _resolve_output_geometry(reader.metadata, target_resolution, output_mode)
        analyzed_frames = 0
        for frame_index, frame in reader.frames():
            if max_frames > 0 and frame_index >= max_frames:
                break
            analyze_frame(_apply_output_geometry(frame, geometry), frame_index)
            analyzed_frames += 1

    if on_analysis_complete is not None:
        on_analysis_complete(analyzed_frames)

    with VideoReader(input_path) as reader:
        geometry = _resolve_output_geometry(reader.metadata, target_resolution, output_mode)
        metadata = geometry.metadata
        with VideoWriter(output_path, metadata) as writer:
            for frame_index, frame in reader.frames():
                if max_frames > 0 and frame_index >= max_frames:
                    break
                writer.write(render_frame(_apply_output_geometry(frame, geometry), frame_index))

    return metadata


def read_video_metadata(input_path: str | Path) -> VideoMetadata:
    with VideoReader(input_path) as reader:
        return reader.metadata


_MIN_REASONABLE_FPS = 1.0
_MAX_REASONABLE_FPS = 240.0


def measure_video_fps(input_path: str | Path, max_samples: int = 30) -> float | None:
    with VideoReader(input_path) as reader:
        samples = reader.sample_frames_with_timestamps(max_samples)

    if len(samples) < 2:
        return None

    deltas_ms = [
        samples[index][2] - samples[index - 1][2]
        for index in range(1, len(samples))
    ]
    positive_deltas = [delta for delta in deltas_ms if delta > 0]
    if not positive_deltas:
        return None

    median_delta_ms = float(median(positive_deltas))
    if median_delta_ms <= 0:
        return None

    measured_fps = 1000.0 / median_delta_ms
    if not _MIN_REASONABLE_FPS <= measured_fps <= _MAX_REASONABLE_FPS:
        return None

    return measured_fps
