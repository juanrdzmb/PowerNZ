import pytest

from calibration import create_calibration_from_plate_diameter
from anchor_metrics import AnchorVelocityTracker
from metrics import BiomechanicsConfig, BiomechanicsEngine, CompletedRep, LiftStateMachine, VelocityEstimator
from pose import PoseKeypoint, PoseResult


def test_velocity_estimator_computes_smoothed_velocity() -> None:
    estimator = VelocityEstimator(fps=10.0, min_cutoff=1000.0, beta=0.0, deadband_mps=0.0)

    raw_velocity, smoothed_velocity = estimator.update(0.0, 0)
    assert raw_velocity == 0.0
    assert smoothed_velocity == 0.0

    raw_velocity, smoothed_velocity = estimator.update(0.1, 1)
    assert raw_velocity == pytest.approx(1.0)
    assert smoothed_velocity == pytest.approx(1.0, abs=0.05)


def test_velocity_estimator_converts_pixel_motion_with_plate_calibration() -> None:
    calibration = create_calibration_from_plate_diameter(plate_diameter_pixels=150.0)
    estimator = VelocityEstimator(
        fps=30.0,
        min_cutoff=1000.0,
        beta=0.0,
        deadband_mps=0.0,
        max_abs_velocity_mps=10.0,
    )

    estimator.update(-300.0 * calibration.meters_per_pixel, 0)
    raw_velocity, smoothed_velocity = estimator.update(-290.0 * calibration.meters_per_pixel, 1)

    assert raw_velocity == pytest.approx(0.9)
    assert smoothed_velocity == pytest.approx(0.9, abs=0.05)


def test_anchor_velocity_tracker_reset_drops_previous_pose_history() -> None:
    calibration = create_calibration_from_plate_diameter(plate_diameter_pixels=150.0)
    tracker = AnchorVelocityTracker(fps=30.0, calibration=calibration)

    first_pose = PoseResult(
        keypoints=[PoseKeypoint("left_hip", 100.0, 300.0, 0.9)],
        backend="yolo",
        detected=True,
    )
    moving_pose = PoseResult(
        keypoints=[PoseKeypoint("left_hip", 100.0, 270.0, 0.9)],
        backend="yolo",
        detected=True,
    )
    after_reset_pose = PoseResult(
        keypoints=[PoseKeypoint("left_hip", 100.0, 240.0, 0.9)],
        backend="yolo",
        detected=True,
    )

    tracker.update(first_pose)
    moving = tracker.update(moving_pose)
    assert next(anchor for anchor in moving if anchor.name == "hip").velocity_mps > 0.0

    tracker.reset()
    reset = tracker.update(after_reset_pose)

    assert next(anchor for anchor in reset if anchor.name == "hip").velocity_mps == pytest.approx(0.0)


def test_velocity_estimator_suppresses_stationary_jitter() -> None:
    estimator = VelocityEstimator(fps=60.0, min_cutoff=1000.0, beta=0.0, deadband_mps=0.03)

    samples = [0.0, 0.0002, 0.0001, 0.0003, 0.0002]
    outputs = [estimator.update(position, index) for index, position in enumerate(samples)]

    assert all(raw == 0.0 for raw, _ in outputs)
    assert all(smoothed == 0.0 for _, smoothed in outputs)


def test_lift_state_machine_completes_rep() -> None:
    config = BiomechanicsConfig(
        upward_velocity_threshold_mps=0.1,
        downward_velocity_threshold_mps=-0.1,
        lockout_velocity_threshold_mps=0.1,
        rest_velocity_threshold_mps=0.1,
        min_rep_displacement_m=0.2,
        min_rep_frames=4,
        lockout_hold_frames=1,
        rest_hold_frames=1,
    )
    machine = LiftStateMachine(config)

    samples = [
        (0, 0.0, 0.0),
        (1, 0.05, 0.2),
        (2, 0.25, 0.2),
        (3, 0.25, 0.0),
        (4, 0.10, -0.2),
        (5, 0.02, 0.0),
    ]
    for frame_index, position_m, velocity_mps in samples:
        machine.update(frame_index, position_m, velocity_mps)

    assert len(machine.completed_reps) == 1
    assert machine.completed_reps[0].displacement_m >= 0.2


def test_biomechanics_engine_rejects_unreasonable_rep_displacement() -> None:
    config = BiomechanicsConfig(max_reasonable_rep_displacement_m=0.5)
    engine = BiomechanicsEngine(fps=30.0, config=config)
    engine.completed_reps.append(CompletedRep(1, 0, 10, 20, 0.8, 1.0))

    validations = engine.validate_reps()

    assert validations[0].accepted is False


def test_reconstructed_engine_suppresses_tiny_false_descent_before_lockout() -> None:
    config = BiomechanicsConfig(
        upward_velocity_threshold_mps=0.1,
        downward_velocity_threshold_mps=-0.1,
        min_rep_displacement_m=0.2,
        min_rep_frames=4,
    )
    engine = BiomechanicsEngine(fps=30.0, config=config)
    engine.update_reconstructed(0, 0.00, 0.00)
    engine.update_reconstructed(1, 0.05, 0.20)
    engine.update_reconstructed(2, 0.16, 0.28)
    wobble = engine.update_reconstructed(3, 0.155, -0.12)

    assert wobble.smoothed_velocity_mps == 0.0
    assert wobble.raw_velocity_mps == -0.12
    assert wobble.state in {"inicio", "tirón"}


def test_reconstructed_engine_holds_zero_velocity_during_lockout_jitter() -> None:
    config = BiomechanicsConfig(
        upward_velocity_threshold_mps=0.1,
        downward_velocity_threshold_mps=-0.1,
        lockout_velocity_threshold_mps=0.2,
        min_rep_displacement_m=0.2,
        min_rep_frames=2,
        lockout_hold_frames=1,
    )
    engine = BiomechanicsEngine(fps=30.0, config=config)
    engine.update_reconstructed(0, 0.00, 0.00)
    engine.update_reconstructed(1, 0.08, 0.25)
    engine.update_reconstructed(2, 0.25, 0.30)
    lockout = engine.update_reconstructed(3, 0.25, 0.00)
    jitter = engine.update_reconstructed(4, 0.247, -0.13)

    assert lockout.state == "bloqueo"
    assert jitter.state == "bloqueo"
    assert jitter.smoothed_velocity_mps == 0.0
    assert jitter.raw_velocity_mps == -0.13
