from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detect_objects import Detection
from io_video import Frame
from track import Point2D


PLATE_COLOR_WEIGHTS_KG = {
    "gray": 5.0,
    "green": 10.0,
    "yellow": 15.0,
    "blue": 20.0,
    "red": 25.0,
}


@dataclass(frozen=True)
class LoadEstimate:
    total_kg: float
    side_weight_kg: float
    colors: tuple[str, ...]
    confidence: float


def estimate_load_from_detections(
    detections: list[Detection],
    frame: Frame,
    bar_point: Point2D | None = None,
    bar_weight_kg: float = 20.0,
) -> LoadEstimate | None:
    plate_weights: list[tuple[float, str, float, float]] = []
    for plate in _dedupe_plate_detections([d for d in detections if d.label == "plate"]):
        color, confidence = classify_plate_color(frame, plate)
        weight = PLATE_COLOR_WEIGHTS_KG.get(color)
        if weight is None:
            continue
        plate_weights.append((weight, color, confidence, plate.center[0]))

    if not plate_weights:
        return None

    if bar_point is not None:
        left = [item for item in plate_weights if item[3] < bar_point.x]
        right = [item for item in plate_weights if item[3] >= bar_point.x]
    else:
        left = []
        right = []

    if left and right:
        side_weight = max(sum(item[0] for item in left), sum(item[0] for item in right))
        total = bar_weight_kg + sum(item[0] for item in plate_weights)
    else:
        side_weight = sum(item[0] for item in plate_weights)
        total = bar_weight_kg + side_weight * 2.0

    colors = tuple(dict.fromkeys(item[1] for item in plate_weights))
    confidence = sum(item[2] for item in plate_weights) / len(plate_weights)
    return LoadEstimate(
        total_kg=round(total, 1),
        side_weight_kg=round(side_weight, 1),
        colors=colors,
        confidence=confidence,
    )


def classify_plate_color(frame: Frame, detection: Detection) -> tuple[str, float]:
    height, width = frame.shape[:2]
    x1 = max(0, int(round(detection.x1)))
    y1 = max(0, int(round(detection.y1)))
    x2 = min(width, int(round(detection.x2)))
    y2 = min(height, int(round(detection.y2)))
    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        "red": _red_mask(hsv),
        "blue": cv2.inRange(hsv, np.array([90, 45, 40]), np.array([132, 255, 255])),
        "yellow": cv2.inRange(hsv, np.array([18, 55, 70]), np.array([38, 255, 255])),
        "green": cv2.inRange(hsv, np.array([38, 40, 35]), np.array([88, 255, 240])),
        "gray": cv2.inRange(hsv, np.array([0, 0, 45]), np.array([179, 55, 230])),
    }
    scores = {
        color: float(np.count_nonzero(mask)) / max(1.0, float(mask.size))
        for color, mask in masks.items()
    }
    color, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.08:
        return "unknown", 0.0
    return color, min(1.0, score * 2.5)


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    low = cv2.inRange(hsv, np.array([0, 55, 35]), np.array([12, 255, 255]))
    high = cv2.inRange(hsv, np.array([168, 55, 35]), np.array([179, 255, 255]))
    orange_red = cv2.inRange(hsv, np.array([8, 60, 45]), np.array([18, 255, 255]))
    return cv2.bitwise_or(cv2.bitwise_or(low, high), orange_red)


def _dedupe_plate_detections(plates: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for plate in sorted(plates, key=lambda detection: detection.confidence, reverse=True):
        center = plate.center
        size = max(plate.width, plate.height)
        if any(
            np.hypot(center[0] - other.center[0], center[1] - other.center[1])
            <= max(size, max(other.width, other.height)) * 0.35
            for other in kept
        ):
            continue
        kept.append(plate)
    return kept
