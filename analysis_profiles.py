"""Runtime presets for the video-analysis pipeline.

The presets intentionally affect inference only.  Output geometry remains a user
choice, so the balanced profile can be fast without degrading the 720p overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AnalysisProfileName = Literal["balanced", "precision", "fast"]


@dataclass(frozen=True)
class AnalysisProfile:
    name: AnalysisProfileName
    inference_max_side: int
    segmentation_stride: int
    velocity_window_seconds: float


ANALYSIS_PROFILES: dict[AnalysisProfileName, AnalysisProfile] = {
    "balanced": AnalysisProfile(
        name="balanced",
        inference_max_side=960,
        segmentation_stride=2,
        velocity_window_seconds=4.5,
    ),
    "precision": AnalysisProfile(
        name="precision",
        inference_max_side=0,
        segmentation_stride=1,
        velocity_window_seconds=4.5,
    ),
    "fast": AnalysisProfile(
        name="fast",
        inference_max_side=640,
        segmentation_stride=3,
        velocity_window_seconds=4.5,
    ),
}


def get_analysis_profile(name: str) -> AnalysisProfile:
    try:
        return ANALYSIS_PROFILES[name]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unknown analysis profile {name!r}.") from exc
