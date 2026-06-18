from __future__ import annotations

import numpy as np

from detect_objects import Detection
from load_estimation import classify_plate_color, estimate_load_from_detections


def test_classify_plate_color_detects_red_plate() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 220)
    detection = Detection("plate", 0.9, 0.0, 0.0, 80.0, 80.0)

    color, confidence = classify_plate_color(frame, detection)

    assert color == "red"
    assert confidence > 0.8


def test_estimate_load_doubles_visible_side_weight() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 220)
    detection = Detection("plate", 0.9, 0.0, 0.0, 80.0, 80.0)

    estimate = estimate_load_from_detections([detection], frame)

    assert estimate is not None
    assert estimate.total_kg == 70.0
    assert estimate.side_weight_kg == 25.0
    assert estimate.colors == ("red",)
