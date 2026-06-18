from __future__ import annotations

import numpy as np
import pytest

import cv2

from detect_objects import (
    DEFAULT_TARGET_CLASSES,
    BeigePlateDetector,
    ColorPlateDetector,
    MultiColorPlateDetector,
)


def _black_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _circle_frame(
    circles: list[tuple[tuple[int, int], int, tuple[int, int, int]]],
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    frame = _black_frame(width=width, height=height)
    for center, radius, color in circles:
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    return frame


def test_color_plate_detector_runs_on_black_frame() -> None:
    detector = ColorPlateDetector()
    result = detector.detect(_black_frame())

    assert result == []


def test_default_yolo_target_classes_are_v1_bar_classes() -> None:
    assert "plate" in DEFAULT_TARGET_CLASSES
    assert "bar_hub" in DEFAULT_TARGET_CLASSES
    assert "bar_sleeve" not in DEFAULT_TARGET_CLASSES
    assert "bar_shaft" not in DEFAULT_TARGET_CLASSES


def test_color_plate_detector_runs_with_expected_diameter() -> None:
    detector = ColorPlateDetector(expected_diameter_pixels=120)
    result = detector.detect(_black_frame())

    assert result == []


def test_beige_plate_detector_runs_on_black_frame() -> None:
    detector = BeigePlateDetector()
    result = detector.detect(_black_frame())

    assert result == []


def test_multi_color_plate_detector_runs_on_black_frame() -> None:
    detector = MultiColorPlateDetector()
    result = detector.detect(_black_frame())

    assert result == []


def test_multi_color_plate_detector_runs_with_expected_diameter() -> None:
    detector = MultiColorPlateDetector(expected_diameter_pixels=120)
    result = detector.detect(_black_frame())

    assert result == []


def test_color_plate_detector_internal_methods_exist() -> None:
    detector = ColorPlateDetector()

    assert hasattr(detector, "_diameter_is_plausible")
    assert hasattr(detector, "_select_plate")
    assert hasattr(detector, "_stabilize_plate_center")
    assert hasattr(detector, "_score_candidate")
    assert hasattr(detector, "_reuse_last_detection")


def test_color_plate_diameter_plausibility_uses_tight_range() -> None:
    detector = ColorPlateDetector(expected_diameter_pixels=100)

    assert detector._diameter_is_plausible(80)
    assert detector._diameter_is_plausible(120)
    assert not detector._diameter_is_plausible(50)
    assert not detector._diameter_is_plausible(200)


def test_color_plate_detector_tags_red_plate_color() -> None:
    detector = ColorPlateDetector(expected_diameter_pixels=100)
    frame = _circle_frame([((320, 320), 50, (0, 0, 220))])

    detections = detector.detect(frame)
    plates = [detection for detection in detections if detection.label == "plate"]

    assert plates
    assert plates[0].color == "red"


def test_beige_plate_detector_does_not_select_white_silhouette() -> None:
    detector = BeigePlateDetector(expected_diameter_pixels=180)
    frame = _black_frame()
    cv2.ellipse(frame, (320, 280), (95, 170), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)

    detections = detector.detect(frame)

    assert [detection for detection in detections if detection.label == "plate"] == []


def test_beige_plate_detector_rejects_initial_edge_clipped_false_positive() -> None:
    detector = BeigePlateDetector(expected_diameter_pixels=120)
    frame = _black_frame(width=396, height=915)
    cv2.circle(frame, (35, 880), 65, (130, 155, 160), -1, cv2.LINE_AA)

    detections = detector.detect(frame)

    assert [detection for detection in detections if detection.label == "plate"] == []


def test_beige_plate_detector_rejects_initial_full_width_false_positive() -> None:
    detector = BeigePlateDetector(expected_diameter_pixels=None)
    frame = _black_frame(width=396, height=916)
    cv2.ellipse(frame, (198, 260), (210, 210), 0, 0, 360, (130, 155, 160), -1, cv2.LINE_AA)

    detections = detector.detect(frame)

    assert [detection for detection in detections if detection.label == "plate"] == []


def test_multi_color_detector_returns_red_and_beige_plates() -> None:
    detector = MultiColorPlateDetector(expected_diameter_pixels=100)
    frame = _circle_frame(
        [
            ((190, 320), 50, (120, 160, 190)),
            ((440, 320), 50, (0, 0, 220)),
        ]
    )

    detections = detector.detect(frame)
    plate_colors = sorted(detection.color for detection in detections if detection.label == "plate")

    assert plate_colors == ["beige", "red"]
