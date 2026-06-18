"""IPF-style rep validation: pose joint angles + FSM depth/lockout gating."""

from __future__ import annotations

import numpy as np
import pytest

from anchor_metrics import smooth_series
from biomech_angles import (
    compute_ipf_flags,
    elbow_angle_deg,
    knee_angle_deg,
)
from metrics import (
    EXERCISE_DISPLACEMENT_DEFAULTS,
    BiomechanicsConfig,
    BiomechanicsEngine,
    LiftStateMachine,
    get_exercise_profile,
)
from pose import PoseKeypoint, PoseResult


def _pose(points: dict[str, tuple[float, float]], visibility: float = 1.0) -> PoseResult:
    keypoints = [
        PoseKeypoint(name=name, x=x, y=y, visibility=visibility)
        for name, (x, y) in points.items()
    ]
    return PoseResult(keypoints=keypoints, backend="yolo", detected=True)


# --- joint angles -------------------------------------------------------------

def _kpmap(points: dict[str, tuple[float, float]]) -> dict[str, PoseKeypoint]:
    return {name: PoseKeypoint(name, x, y, 1.0) for name, (x, y) in points.items()}


def test_knee_angle_straight_vs_parallel():
    straight = {"left_hip": (0.0, 0.0), "left_knee": (0.0, 1.0), "left_ankle": (0.0, 2.0)}
    assert knee_angle_deg(_kpmap(straight)) == pytest.approx(180.0, abs=1.0)

    bent = {"left_hip": (0.0, 0.0), "left_knee": (0.0, 1.0), "left_ankle": (1.0, 1.0)}
    assert knee_angle_deg(_kpmap(bent)) == pytest.approx(90.0, abs=1.0)


def test_elbow_angle_extended():
    pts = {"left_shoulder": (0.0, 0.0), "left_elbow": (0.0, 1.0), "left_wrist": (0.0, 2.0)}
    assert elbow_angle_deg(_kpmap(pts)) == pytest.approx(180.0, abs=1.0)


def test_compute_ipf_flags_squat_depth():
    deep = _pose({"left_hip": (0.0, 0.0), "left_knee": (0.0, 1.0), "left_ankle": (1.0, 1.0)})
    depth_ok, _ = compute_ipf_flags("squat", deep)
    assert depth_ok is True  # ~90 deg knee = parallel

    shallow = _pose({"left_hip": (0.0, 0.0), "left_knee": (0.0, 1.0), "left_ankle": (0.0, 2.0)})
    depth_ok, lockout_ok = compute_ipf_flags("squat", shallow)
    assert depth_ok is False  # straight knee, not deep
    assert lockout_ok is True  # standing = locked out


def test_compute_ipf_flags_none_without_pose():
    empty = PoseResult(keypoints=[], backend="yolo", detected=False)
    assert compute_ipf_flags("deadlift", empty) == (None, None)


# --- FSM gating ---------------------------------------------------------------

def _deadlift_samples():
    # (frame, position_m, velocity_mps): rest -> pull -> still at top -> lower -> rest
    return [
        (0, 0.0, 0.0),
        (1, 0.05, 0.2),
        (2, 0.25, 0.2),
        (3, 0.25, 0.0),
        (4, 0.10, -0.2),
        (5, 0.02, 0.0),
    ]


def _deadlift_config():
    return BiomechanicsConfig(
        upward_velocity_threshold_mps=0.1,
        downward_velocity_threshold_mps=-0.1,
        lockout_velocity_threshold_mps=0.1,
        rest_velocity_threshold_mps=0.1,
        min_rep_displacement_m=0.2,
        min_rep_frames=4,
        lockout_hold_frames=1,
        rest_hold_frames=1,
    )


def test_deadlift_rep_counts_when_locked_out():
    machine = LiftStateMachine(_deadlift_config())
    for frame_index, position_m, velocity_mps in _deadlift_samples():
        machine.update(frame_index, position_m, velocity_mps, lockout_ok=True)
    assert len(machine.completed_reps) == 1


def test_deadlift_rep_rejected_without_lockout():
    machine = LiftStateMachine(_deadlift_config())
    for frame_index, position_m, velocity_mps in _deadlift_samples():
        # Bar reaches the top but the joints never extend -> no valid lockout.
        machine.update(frame_index, position_m, velocity_mps, lockout_ok=False)
    assert machine.completed_reps == []


def test_deadlift_rep_rejects_early_velocity_dip_before_mature_pull():
    config = _deadlift_config()
    machine = LiftStateMachine(config)
    samples = [
        (0, 0.0, 0.0),
        (1, 0.22, 0.2),
        (2, 0.22, 0.0),  # enough range but too early to be a real lockout
        (3, 0.10, -0.2),
        (4, 0.02, 0.0),
    ]

    for frame_index, position_m, velocity_mps in samples:
        machine.update(frame_index, position_m, velocity_mps, lockout_ok=True)

    assert machine.completed_reps == []


def _run_squat(positions: list[float], depth_ok: bool = True) -> BiomechanicsEngine:
    profile = get_exercise_profile("squat")
    min_d, max_d = EXERCISE_DISPLACEMENT_DEFAULTS["squat"]
    config = BiomechanicsConfig(min_rep_displacement_m=min_d, max_reasonable_rep_displacement_m=max_d)
    engine = BiomechanicsEngine(fps=30.0, config=config, profile=profile)
    for index, position in enumerate(positions):
        engine.update(frame_index=index, vertical_position_m=float(position), depth_ok=depth_ok)
    engine.finalize(len(positions))
    return engine


def _down_then_up(top: float, bottom: float, rest: int = 12, span: int = 22, hold: int = 16):
    return (
        [top] * rest
        + list(np.linspace(top, bottom, span))
        + list(np.linspace(bottom, top, span))
        + [top] * hold
    )


def test_squat_rep_counts_when_parallel_reached():
    engine = _run_squat(_down_then_up(top=1.0, bottom=0.45), depth_ok=True)
    assert len(engine.validated_reps) == 1


def test_squat_rep_rejected_without_depth():
    # Same bar travel, but pose says parallel is never reached -> not a valid rep.
    engine = _run_squat(_down_then_up(top=1.0, bottom=0.45), depth_ok=False)
    assert engine.validated_reps == []


# --- chart smoothing ----------------------------------------------------------

def test_smooth_series_reduces_jitter_and_keeps_length():
    noisy = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    smoothed = smooth_series(noisy, window=5)
    assert len(smoothed) == len(noisy)
    # The midpoint should be pulled toward the local mean (~0.5), not the raw extremes.
    assert 0.2 < smoothed[3] < 0.8


def test_smooth_series_preserves_nan_gaps():
    values = [0.5, float("nan"), 0.5, 0.5]
    smoothed = smooth_series(values, window=3)
    assert np.isnan(smoothed[1])
    assert all(np.isfinite(smoothed[i]) for i in (0, 2, 3))
