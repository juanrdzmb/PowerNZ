from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration import SpatialCalibration
from pose import PoseResult
from track import OneEuroFilter


# Logical anchor points measured on the skeleton. For each we pick whichever side
# (left/right) is more visible per frame, so the velocity stays reliable as the
# athlete rotates. The wrist tracks the bar (where it is gripped).
ANCHOR_GROUPS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("shoulder", ("left_shoulder", "right_shoulder")),
    ("elbow", ("left_elbow", "right_elbow")),
    ("wrist", ("left_wrist", "right_wrist")),
    ("hip", ("left_hip", "right_hip")),
    ("knee", ("left_knee", "right_knee")),
)

_MIN_VISIBILITY = 0.4


def smooth_series(values: list[float], window: int = 7) -> list[float]:
    """Zero-phase (centered) moving-average smoothing for a velocity series.

    Because the whole series is known up front (two-pass analysis), we can smooth
    symmetrically — averaging each sample with its neighbours on *both* sides — which
    has no lag, unlike the causal OneEuroFilter used live. ``NaN`` gaps (frames where a
    joint was not visible) are preserved so the chart still breaks the line there, and
    only finite neighbours contribute to each average.
    """
    array = np.asarray(values, dtype=float)
    count = array.size
    if count == 0:
        return []

    radius = max(0, window // 2)
    if radius == 0:
        return array.tolist()

    finite = np.isfinite(array)
    smoothed = np.full(count, np.nan, dtype=float)
    for index in range(count):
        if not finite[index]:
            continue
        low = max(0, index - radius)
        high = min(count, index + radius + 1)
        neighbours = array[low:high][finite[low:high]]
        if neighbours.size:
            smoothed[index] = float(neighbours.mean())
    return smoothed.tolist()


@dataclass(frozen=True)
class AnchorVelocity:
    name: str
    x: float
    y: float
    velocity_mps: float
    confidence: float


class AnchorVelocityTracker:
    def __init__(self, fps: float, calibration: SpatialCalibration) -> None:
        self._fps = fps
        self._calibration = calibration
        self._previous_y_m: dict[str, float] = {}
        self._filters: dict[str, OneEuroFilter] = {}
        self.reset()

    def reset(self) -> None:
        self._previous_y_m = {}
        self._filters = {
            label: OneEuroFilter(frequency_hz=self._fps, min_cutoff=1.0, beta=0.02)
            for label, _ in ANCHOR_GROUPS
        }

    def update(self, pose: PoseResult) -> list[AnchorVelocity]:
        by_name = {kp.name: kp for kp in pose.keypoints}
        anchors: list[AnchorVelocity] = []

        for label, sides in ANCHOR_GROUPS:
            keypoint = self._most_visible(by_name, sides)
            if keypoint is None or keypoint.visibility < _MIN_VISIBILITY:
                continue

            y_m = -keypoint.y * self._calibration.meters_per_pixel
            previous_y_m = self._previous_y_m.get(label)
            raw_velocity = 0.0 if previous_y_m is None else (y_m - previous_y_m) * self._fps
            self._previous_y_m[label] = y_m

            filtered_velocity = self._filters[label].apply(raw_velocity)
            anchors.append(
                AnchorVelocity(
                    name=label,
                    x=keypoint.x,
                    y=keypoint.y,
                    velocity_mps=filtered_velocity,
                    confidence=keypoint.visibility,
                )
            )

        return anchors

    @staticmethod
    def _most_visible(by_name: dict, sides: tuple[str, str]):
        candidates = [by_name[name] for name in sides if name in by_name]
        if not candidates:
            return None
        return max(candidates, key=lambda kp: kp.visibility)
