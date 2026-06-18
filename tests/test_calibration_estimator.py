from __future__ import annotations

from calibration import (
    CalibrationConfig,
    CalibrationEstimator,
    estimate_plate_diameter_pixels,
)
from detect_objects import Detection


def _plate(width: float, height: float, confidence: float = 0.9) -> Detection:
    return Detection(
        label="plate",
        confidence=confidence,
        x1=0.0,
        y1=0.0,
        x2=width,
        y2=height,
    )


def test_estimate_plate_diameter_pixels_returns_median_size() -> None:
    detections = [_plate(100.0, 100.0), _plate(110.0, 110.0), _plate(120.0, 120.0)]

    diameter = estimate_plate_diameter_pixels(detections)

    assert diameter == 110.0


def test_estimate_plate_diameter_pixels_ignores_non_plate_detections() -> None:
    detections = [
        _plate(100.0, 100.0),
        Detection(label="barbell", confidence=0.9, x1=0.0, y1=0.0, x2=200.0, y2=200.0),
    ]

    diameter = estimate_plate_diameter_pixels(detections)

    assert diameter == 100.0


def test_estimate_plate_diameter_pixels_ignores_low_confidence_detections() -> None:
    detections = [_plate(100.0, 100.0), _plate(120.0, 120.0, confidence=0.1)]

    diameter = estimate_plate_diameter_pixels(
        detections, config=CalibrationConfig(min_plate_confidence=0.5)
    )

    assert diameter == 100.0


def test_estimate_plate_diameter_pixels_returns_none_without_plates() -> None:
    assert estimate_plate_diameter_pixels([]) is None


def test_calibration_estimator_stabilizes_after_min_observations() -> None:
    estimator = CalibrationEstimator(min_observations=3)
    estimator.update([_plate(100.0, 100.0)])
    estimator.update([_plate(105.0, 105.0)])
    calibration = estimator.update([_plate(110.0, 110.0)])

    assert calibration is not None
    assert calibration.plate_diameter_pixels == 105.0


def test_calibration_estimator_returns_none_until_min_observations() -> None:
    estimator = CalibrationEstimator(min_observations=5)

    assert estimator.update([_plate(100.0, 100.0)]) is None
    assert estimator.update([_plate(105.0, 105.0)]) is None
    calibration = estimator.update([_plate(110.0, 110.0)])

    assert calibration is None


def test_calibration_estimator_keeps_last_calibration_without_observations() -> None:
    estimator = CalibrationEstimator(min_observations=3)
    estimator.update([_plate(100.0, 100.0)])
    estimator.update([_plate(105.0, 105.0)])
    estimator.update([_plate(110.0, 110.0)])

    calibration = estimator.update([])

    assert calibration is not None
    assert calibration.plate_diameter_pixels == 105.0
