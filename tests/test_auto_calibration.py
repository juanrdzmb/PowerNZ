from __future__ import annotations

import numpy as np

from bar_anchor import BarAnchorState
from detect_objects import Detection
from main import (
    _bar_path_horizontal_drift_cm,
    _nearest_plate_detection_to_anchor,
    _plate_diameter_observation_is_plausible,
    _processing_scale_from_video,
    _reject_outlier_observations,
    _warp_mask_with_optical_flow,
)
from track import Point2D


def test_reject_outlier_observations_keeps_consistent_values() -> None:
    observations = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 100.1, 100.3]

    filtered = _reject_outlier_observations(observations)

    assert filtered == observations


def test_reject_outlier_observations_drops_extreme_values() -> None:
    observations = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 100.1, 100.3, 500.0, 30.0]

    filtered = _reject_outlier_observations(observations)

    assert 500.0 not in filtered
    assert 30.0 not in filtered
    assert 100.0 in filtered


def test_reject_outlier_observations_returns_copy_when_input_too_small() -> None:
    observations = [100.0, 101.0, 99.0]

    filtered = _reject_outlier_observations(observations)

    assert filtered == observations
    assert filtered is not observations


def test_reject_outlier_observations_returns_copy_when_iqr_is_zero() -> None:
    observations = [100.0, 100.0, 100.0, 100.0]

    filtered = _reject_outlier_observations(observations)

    assert filtered == observations


def test_plate_diameter_observation_rejects_impossible_frame_size() -> None:
    frame = np.zeros((720, 310, 3), dtype=np.uint8)

    assert _plate_diameter_observation_is_plausible(120.0, frame)
    assert not _plate_diameter_observation_is_plausible(1107.6, frame)


def test_processing_scale_uses_first_decoded_frame(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, input_path):
            self.input_path = input_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

        def frames(self):
            yield 0, np.zeros((3150, 1362, 3), dtype=np.uint8)

    monkeypatch.setattr("main.VideoReader", FakeReader)

    assert _processing_scale_from_video("video.mp4", 720) == (720 * 0.55) / 1362


def test_nearest_plate_detection_to_anchor_uses_anchor_point() -> None:
    state = BarAnchorState(
        point=Point2D(205.0, 205.0),
        rect=None,
        confidence=0.9,
        missing_frames=0,
        locked=True,
        source="detection",
    )
    background = Detection("plate", 0.9, 0.0, 0.0, 80.0, 80.0)
    bar_plate = Detection("plate", 0.8, 180.0, 180.0, 230.0, 230.0)

    selected = _nearest_plate_detection_to_anchor(state, [background, bar_plate])

    assert selected is bar_plate


def test_bar_path_horizontal_drift_cm_uses_calibration_scale() -> None:
    drift = _bar_path_horizontal_drift_cm([(10.0, 100.0), (30.0, 80.0)], 0.0045)

    assert drift == 9.0


def test_warp_mask_with_optical_flow_keeps_static_mask() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 24:44] = 255

    warped = _warp_mask_with_optical_flow(frame, frame.copy(), mask)

    assert warped is not None
    assert np.mean(np.abs(warped.astype(float) - mask.astype(float))) < 1.0
