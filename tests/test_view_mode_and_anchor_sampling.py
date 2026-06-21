from __future__ import annotations

import numpy as np
import pytest

from bar_anchor import BarAnchorState, BarAnchorTracker, Point2D
from anchor_metrics import ANCHOR_GROUPS, AnchorVelocity
from detect_objects import Detection
from metrics import KinematicSample
from main import (
    _append_visible_motion_history,
    _anchor_point_is_usable,
    _bar_point_is_plausibly_held,
    _body_bar_proxy_from_pose,
    _compute_ipf_flags_with_pose_fallback,
    _filter_detections_near_bar,
    _manual_load_estimate,
    _plate_heuristic_enabled,
    _plate_detections_for_visual_anchor,
    _reset_visible_motion_history,
    _sample_point_from_single_anchor,
    _strict_ipf_gate,
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


def test_manual_load_is_hidden_by_default_and_exact_when_set() -> None:
    assert _manual_load_estimate(None) is None

    estimate = _manual_load_estimate(142.5)

    assert estimate is not None
    assert estimate.total_kg == 142.5
    assert estimate.side_weight_kg == 61.25
    assert estimate.colors == ("manual",)
    assert estimate.confidence == 1.0


def test_strict_ipf_gate_rejects_unknown_pose_by_default() -> None:
    assert _strict_ipf_gate(None, strict=True) is False
    assert _strict_ipf_gate(None, strict=False) is True
    assert _strict_ipf_gate(True, strict=True) is True
    assert _strict_ipf_gate(False, strict=False) is False


def test_ipf_flags_use_raw_pose_when_refined_pose_cannot_decide() -> None:
    refined = PoseResult(keypoints=[], backend="yolo", detected=False)
    raw = PoseResult(
        keypoints=[
            PoseKeypoint("left_shoulder", 100.0, 100.0, 0.9),
            PoseKeypoint("left_hip", 100.0, 180.0, 0.9),
            PoseKeypoint("left_knee", 100.0, 260.0, 0.9),
            PoseKeypoint("left_ankle", 100.0, 340.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )

    assert _compute_ipf_flags_with_pose_fallback("deadlift", refined, raw) == (None, True)


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


def test_bar_point_guard_rejects_ceiling_or_background_hub() -> None:
    pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_wrist", 355.0, 900.0, 0.9),
            PoseKeypoint("right_wrist", 385.0, 900.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )

    assert _bar_point_is_plausibly_held(Point2D(150.0, 940.0), pose, (1280, 720, 3))
    assert not _bar_point_is_plausibly_held(Point2D(510.0, 240.0), pose, (1280, 720, 3))


def test_body_proxy_uses_wrist_midpoint_when_hub_is_occluded() -> None:
    pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_wrist", 220.0, 640.0, 0.92),
            PoseKeypoint("right_wrist", 300.0, 660.0, 0.84),
        ],
        backend="mediapipe",
        detected=True,
    )

    proxy = _body_bar_proxy_from_pose(pose)

    assert proxy is not None
    point, confidence = proxy
    assert point == Point2D(260.0, 650.0)
    assert confidence == pytest.approx(0.84 * 0.76)


def test_visual_plate_anchor_prefers_outer_disc_over_torso_false_positive() -> None:
    pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_shoulder", 310.0, 420.0, 0.9),
            PoseKeypoint("right_shoulder", 410.0, 420.0, 0.9),
            PoseKeypoint("left_hip", 320.0, 650.0, 0.9),
            PoseKeypoint("right_hip", 400.0, 650.0, 0.9),
            PoseKeypoint("left_ankle", 320.0, 960.0, 0.9),
            PoseKeypoint("right_ankle", 400.0, 960.0, 0.9),
        ],
        backend="mediapipe",
        detected=True,
    )
    torso_false_positive = Detection("plate", 0.98, 320.0, 560.0, 405.0, 690.0)
    outer_disc = Detection("plate", 0.86, 12.0, 570.0, 145.0, 745.0)

    chosen = _plate_detections_for_visual_anchor(
        [torso_false_positive, outer_disc],
        pose,
        (1280, 720, 3),
    )

    assert chosen == [outer_disc]
