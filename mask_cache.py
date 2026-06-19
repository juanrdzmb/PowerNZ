"""Disk-backed, temporary cache for rendered subject masks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


class MaskFrameCache:
    """Keeps the expensive first-pass masks available to the render pass.

    PNG is lossless for the alpha mask and normally far smaller than retaining a
    full uncompressed 720p mask for every frame in RAM.
    """

    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="powernz_masks_")
        self.root = Path(self._temporary_directory.name)

    def put(self, frame_index: int, mask: np.ndarray | None) -> None:
        if mask is None:
            return
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            return
        (self.root / f"{frame_index:08d}.png").write_bytes(encoded.tobytes())

    def get(self, frame_index: int) -> np.ndarray | None:
        path = self.root / f"{frame_index:08d}.png"
        if not path.exists():
            return None
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> "MaskFrameCache":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
