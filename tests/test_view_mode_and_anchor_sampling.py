from __future__ import annotations

import numpy as np

from bar_anchor import BarAnchorState, BarAnchorTracker, Point2D
from anchor_metrics import ANCHOR_GROUPS, AnchorVelocity
from detect_objects import Detection
from metrics import KinematicSample
from main import (
    _append_visible_motion_history,
    _anchor_point_is_usable,
    _filter_detections_near_bar,
    _plate_heuristic_enabled,
    _reset_visible_motion_history,
    _sample_point_from_single_anchor,
)
from pose import PoseKeypoint, PoseResult


def test_single_anchor_selects_one_visible_plate_when_two_exist() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection("plate", 0.8, 60.0, 220.0, 180.0, 340.0, color="beige"),
        Detection("plate", 0.8, 460.0, 220.0, 580.0, 340.0, color="red"),
    ]

    state = tracker.update(frame, detections)

    assert state.point is not None
    assert state.rect is not None
    assert state.source == "detection"


def test_anchor_sampling_rejects_pose_and_long_hold_sources() -> None:
    point = Point2D(100.0, 200.0)

    assert not _anchor_point_is_usable(point, 0.8, "pose_seed", 0)
    assert not _anchor_point_is_usable(point, 0.8, "wrist", 0)
    assert not _anchor_point_is_usable(point, 0.8, "hold", 20)
    assert _anchor_point_is_usable(point, 0.8, "detection", 0)


def test_single_anchor_sample_comes_from_hub_point_only_when_reliable() -> None:
    hub_point = Point2D(120.0, 260.0)
    reliable = BarAnchorState(
        point=hub_point,
        rect=None,
        confidence=0.75,
        missing_frames=0,
        locked=True,
        source="detection",
        hub_detected=True,
        hub_confidence=0.8,
        measurement_point=hub_point,
        measurement_confidence=0.8,
        measurable=True,
    )
    plate_only = BarAnchorState(
        point=Point2D(120.0, 260.0),
        rect=None,
        confidence=0.75,
        missing_frames=0,
        locked=True,
        source="detection",
        hub_detected=False,
        hub_confidence=0.0,
        measurement_point=None,
        measurement_confidence=0.0,
        measurable=False,
    )
    pose_seed = BarAnchorState(
        point=Point2D(120.0, 260.0),
        rect=None,
        confidence=0.55,
        missing_frames=0,
        locked=True,
        source="pose_seed",
    )

    assert _sample_point_from_single_anchor(reliable) == reliable.point
    assert _sample_point_from_single_anchor(plate_only) is None
    assert _sample_point_from_single_anchor(pose_seed) is None


def test_visible_motion_history_starts_once_and_accumulates_series() -> None:
    bar_path = [(10.0, 20.0)]
    history = {label: [] for label, _ in ANCHOR_GROUPS}
    history["bar"] = []
    frames: list[int] = []
    reposo_sample = KinematicSample(
        frame_index=2,
        time_seconds=0.07,
        position_m=0.0,
        velocity_mps=0.0,
        smoothed_velocity_mps=0.0,
        state="reposo",
        rep_index=0,
        rep_displacement_m=0.0,
    )
    movement_sample = KinematicSample(
        frame_index=3,
        time_seconds=0.10,
        position_m=0.0,
        velocity_mps=0.20,
        smoothed_velocity_mps=0.20,
        state="tirón",
        rep_index=1,
        rep_displacement_m=0.02,
    )
    second_rep_sample = KinematicSample(
        frame_index=30,
        time_seconds=1.0,
        position_m=0.0,
        velocity_mps=0.18,
        smoothed_velocity_mps=0.18,
        state="tirón",
        rep_index=2,
        rep_displacement_m=0.03,
    )

    analysis_started = False
    if analysis_started:
        _append_visible_motion_history(history, frames, [], reposo_sample, reposo_sample.frame_index)
    assert history["bar"] == []
    assert frames == []

    history["bar"].append(99.0)
    frames.append(1)
    _reset_visible_motion_history(bar_path, history, frames)
    _append_visible_motion_history(
        history,
        frames,
        [AnchorVelocity("hip", 100.0, 200.0, 0.12, 0.9)],
        movement_sample,
        movement_sample.frame_index,
    )
    _append_visible_motion_history(
        history,
        frames,
        [AnchorVelocity("hip", 100.0, 190.0, 0.15, 0.9)],
        second_rep_sample,
        second_rep_sample.frame_index,
    )

    assert bar_path == []
    assert history["bar"] == [0.20, 0.18]
    assert history["hip"] == [0.12, 0.15]
    assert frames == [3, 30]


def test_plate_heuristic_is_strict_by_default_when_trained_detector_exists() -> None:
    trained_detector = object()

    assert not _plate_heuristic_enabled(False, False, trained_detector)
    assert _plate_heuristic_enabled(False, True, trained_detector)
    assert not _plate_heuristic_enabled(True, True, trained_detector)
    assert _plate_heuristic_enabled(False, False, None)


def test_filter_keeps_hub_near_kept_plate_even_when_far_from_wrist_center() -> None:
    pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_wrist", 305.0, 620.0, 0.9),
            PoseKeypoint("right_wrist", 335.0, 620.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )
    detections = [
        Detection("plate", 0.9, 110.0, 540.0, 230.0, 700.0),
        Detection("bar_hub", 0.8, 55.0, 600.0, 85.0, 635.0),
        Detection("plate", 0.8, 610.0, 880.0, 700.0, 970.0),
    ]

    filtered = _filter_detections_near_bar(detections, pose, (1280, 720, 3))

    assert any(detection.label == "bar_hub" for detection in filtered)
    assert detections[2] not in filtered
