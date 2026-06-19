import numpy as np
import pytest

from bar_anchor import BarAnchorTracker, Point2D
from detect_objects import Detection, TrackedDetection
from pose import PoseKeypoint, PoseResult


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _pose_with_wrists(left=(200.0, 300.0, 0.9), right=(280.0, 300.0, 0.9)) -> PoseResult:
    return PoseResult(
        keypoints=[
            PoseKeypoint(name="left_wrist", x=left[0], y=left[1], visibility=left[2]),
            PoseKeypoint(name="right_wrist", x=right[0], y=right[1], visibility=right[2]),
        ],
        backend="yolo",
        detected=True,
    )


def test_bar_anchor_initializes_from_plate_without_wrists() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])

    assert state.locked is True
    assert state.point is not None
    assert state.rect is not None
    assert state.source == "detection"
    assert state.point.x == 140.0
    assert state.point.y == 260.0
    assert state.rect.width == pytest.approx(state.rect.height)
    assert 65.0 <= state.rect.width <= 90.0
    assert state.hub_rect is None
    assert state.measurable is False


def test_bar_anchor_uses_plate_rect_when_barbell_marks_hub() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.72, 100.0, 160.0, 300.0, 360.0),
            Detection("barbell", 0.70, 185.0, 245.0, 215.0, 275.0),
        ],
    )

    assert state.point is not None
    assert state.rect is not None
    assert state.point.x == 200.0
    assert state.point.y == 260.0
    assert 165.0 <= state.rect.width <= 205.0
    assert 165.0 <= state.rect.height <= 205.0
    assert state.hub_rect is not None
    assert state.hub_rect.width < state.rect.width


def test_bar_anchor_pairs_tracked_hub_with_plate_rect() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(
        _frame(),
        [
            TrackedDetection("plate", 0.82, 100.0, 160.0, 300.0, 360.0, track_id=11),
            TrackedDetection("bar_hub", 0.90, 184.0, 244.0, 216.0, 276.0, track_id=21),
        ],
    )

    assert state.point is not None
    assert state.rect is not None
    assert state.hub_rect is not None
    assert state.hub_detected is True
    assert state.measurable is True
    assert state.measurement_point == state.point
    assert state.track_id == 11
    assert state.point.x == pytest.approx(200.0)
    assert state.rect.center.x == pytest.approx(200.0)
    assert state.hub_rect.width < state.rect.width
    assert state.hub_confidence == pytest.approx(0.90)
    assert state.plate_confidence == pytest.approx(0.82)


def test_bar_anchor_prefers_hub_pair_over_large_plate_only_candidate() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.96, 40.0, 40.0, 300.0, 300.0),
            Detection("plate", 0.76, 360.0, 220.0, 460.0, 320.0),
            Detection("bar_hub", 0.82, 398.0, 258.0, 422.0, 282.0),
        ],
    )

    assert state.point is not None
    assert state.hub_detected is True
    assert state.measurable is True
    assert state.point.x == pytest.approx(410.0)
    assert state.point.y == pytest.approx(270.0)


def test_bar_anchor_accepts_trained_hub_near_plate_edge() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.93, 320.0, 180.0, 560.0, 420.0),
            Detection("bar_hub", 0.65, 538.0, 286.0, 568.0, 314.0),
        ],
    )

    assert state.point is not None
    assert state.hub_rect is not None
    assert state.hub_detected is True
    assert state.measurable is True
    assert state.point.x == pytest.approx(553.0)
    assert state.point.y == pytest.approx(300.0)
    assert state.display_rect is not None
    assert state.display_rect.x1 == pytest.approx(320.0)


def test_bar_anchor_accepts_trained_hub_next_to_small_plate_box() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.91, 155.0, 250.0, 245.0, 340.0),
            Detection("bar_hub", 0.79, 84.0, 286.0, 108.0, 310.0),
        ],
    )

    assert state.point is not None
    assert state.hub_rect is not None
    assert state.hub_detected is True
    assert state.measurable is True
    assert state.point.x == pytest.approx(96.0)
    assert state.point.y == pytest.approx(298.0)
    assert state.display_rect is not None
    assert state.display_rect.center.x == pytest.approx(200.0)


def test_bar_anchor_rejects_edge_hub_farther_from_pose_hint_than_plate() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.set_pose_hint(Point2D(350.0, 300.0))

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.91, 155.0, 250.0, 245.0, 340.0),
            Detection("bar_hub", 0.79, 84.0, 286.0, 108.0, 310.0),
        ],
    )

    assert state.point is not None
    assert state.hub_detected is False
    assert state.measurable is False
    assert state.point.x == pytest.approx(200.0)


def test_bar_anchor_keeps_plate_visual_but_stops_measurement_when_hub_drops() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    first = tracker.update(
        _frame(),
        [
            Detection("plate", 0.93, 320.0, 180.0, 560.0, 420.0),
            Detection("bar_hub", 0.65, 538.0, 286.0, 568.0, 314.0),
        ],
    )

    state = tracker.update(
        _frame(),
        [Detection("plate", 0.92, 320.0, 120.0, 560.0, 360.0)],
    )

    assert first.point is not None
    assert state.point is not None
    assert state.source == "detection"
    assert state.measurable is False
    assert state.measurement_point is None
    assert state.hub_rect is None
    assert state.display_rect is not None
    assert state.display_rect.center.y == pytest.approx(240.0)


def test_bar_anchor_uses_raw_model_hub_without_visual_lag() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(
        _frame(),
        [
            Detection("plate", 0.93, 320.0, 180.0, 560.0, 420.0),
            Detection("bar_hub", 0.65, 538.0, 286.0, 568.0, 314.0),
        ],
    )

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.93, 320.0, 150.0, 560.0, 390.0),
            Detection("bar_hub", 0.65, 538.0, 256.0, 568.0, 284.0),
        ],
    )

    assert state.point is not None
    assert state.source == "detection"
    assert state.point.x == pytest.approx(553.0)
    assert state.point.y == pytest.approx(270.0)


def test_bar_anchor_ignores_hub_outside_plate_region() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.82, 100.0, 160.0, 300.0, 360.0),
            Detection("bar_hub", 0.95, 360.0, 420.0, 390.0, 450.0),
        ],
    )

    assert state.point is not None
    assert state.hub_detected is False
    assert state.measurable is False
    assert state.measurement_point is None
    assert state.point.x == pytest.approx(200.0)
    assert state.point.y == pytest.approx(260.0)


def test_bar_anchor_does_not_switch_discs_when_locked_disc_remains_nearby() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(_frame(), [Detection("plate", 0.8, 80.0, 180.0, 220.0, 320.0)])

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.72, 86.0, 184.0, 226.0, 324.0),
            Detection("plate", 0.72, 310.0, 184.0, 450.0, 324.0),
            Detection("bar_hub", 0.95, 358.0, 232.0, 382.0, 256.0),
        ],
    )

    assert state.point is not None
    assert state.point.x == pytest.approx(156.0, abs=8.0)


def test_bar_anchor_rejects_large_jump_and_holds_last_position() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])

    state = tracker.update(_frame(), [Detection("plate", 0.9, 520.0, 40.0, 600.0, 120.0)])

    assert state.locked is True
    assert state.source in {"hold", "optical_flow", "prediction"}
    assert state.point is not None
    assert state.point.x == 140.0
    assert state.point.y == 260.0


def test_bar_anchor_reacquires_vertical_hub_on_same_plate_column() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    first = tracker.update(
        _frame(),
        [
            Detection("plate", 0.86, 120.0, 220.0, 280.0, 380.0),
            Detection("bar_hub", 0.92, 188.0, 288.0, 212.0, 312.0),
        ],
    )

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.88, 124.0, 100.0, 284.0, 260.0),
            Detection("bar_hub", 0.94, 192.0, 168.0, 216.0, 192.0),
        ],
    )

    assert first.point is not None
    assert state.point is not None
    assert state.display_rect is not None
    assert state.source == "detection"
    assert state.hub_detected is True
    assert state.point.y < first.point.y - 30.0
    assert state.display_rect.center.y == pytest.approx(180.0)


def test_bar_anchor_rejects_lateral_hub_switch_during_short_lock() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(
        _frame(),
        [
            Detection("plate", 0.86, 120.0, 220.0, 280.0, 380.0),
            Detection("bar_hub", 0.92, 188.0, 288.0, 212.0, 312.0),
        ],
    )

    state = tracker.update(
        _frame(),
        [
            Detection("plate", 0.92, 360.0, 100.0, 520.0, 260.0),
            Detection("bar_hub", 0.97, 428.0, 168.0, 452.0, 192.0),
        ],
    )

    assert state.point is not None
    assert state.source in {"hold", "optical_flow", "prediction"}
    assert state.point.x < 260.0


def test_bar_anchor_does_not_switch_from_locked_hub_to_far_plate_only_detection() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    first = tracker.update(
        _frame(),
        [
            Detection("plate", 0.86, 120.0, 220.0, 280.0, 380.0),
            Detection("bar_hub", 0.92, 188.0, 288.0, 212.0, 312.0),
        ],
    )

    state = tracker.update(
        _frame(),
        [Detection("plate", 0.98, 430.0, 40.0, 590.0, 200.0)],
    )

    assert first.point is not None
    assert state.point is not None
    assert state.source in {"hold", "optical_flow", "prediction"}
    assert state.point.x == pytest.approx(first.point.x, abs=6.0)
    assert state.point.y == pytest.approx(first.point.y, abs=6.0)


def test_bar_anchor_survives_short_detection_loss() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])

    state = tracker.update(_frame(), [])

    assert state.locked is True
    assert state.point is not None
    assert state.missing_frames == 1
    assert state.source in {"hold", "optical_flow", "prediction"}
    assert state.measurable is False
    assert state.measurement_point is None


def test_bar_anchor_lost_hub_does_not_keep_fake_hub_rect() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(
        _frame(),
        [
            Detection("plate", 0.86, 120.0, 220.0, 280.0, 380.0),
            Detection("bar_hub", 0.92, 188.0, 288.0, 212.0, 312.0),
        ],
    )

    state = tracker.update(_frame(), [])

    assert state.locked is True
    assert state.hub_rect is None
    assert state.hub_confidence == 0.0
    assert state.measurable is False


def test_bar_anchor_predicts_short_occlusion_from_recent_motion() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    last_detection_state = None
    for center_y in [260.0, 248.0, 236.0, 224.0]:
        last_detection_state = tracker.update(
            _frame(),
            [Detection("plate", 0.9, 100.0, center_y - 40.0, 180.0, center_y + 40.0)],
        )

    assert last_detection_state is not None
    assert last_detection_state.point is not None

    predicted = tracker.update(_frame(), [])

    assert predicted.locked is True
    assert predicted.source == "prediction"
    assert predicted.point is not None
    assert predicted.point.y < last_detection_state.point.y


def test_bar_anchor_loses_after_long_detection_gap() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    state = tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])

    for _ in range(930):
        state = tracker.update(_frame(), [])

    assert state.locked is False
    assert state.point is None
    assert state.source == "lost"


def test_bar_anchor_reacquires_after_long_gap() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])

    for _ in range(930):
        tracker.update(_frame(), [])

    state = tracker.update(_frame(), [Detection("plate", 0.9, 500.0, 210.0, 590.0, 300.0)])

    assert state.locked is True
    assert state.source == "detection"
    assert state.point is not None


def test_seed_from_pose_initializes_locked_anchor() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    seeded = tracker.seed_from_pose(_frame(), _pose_with_wrists())

    assert seeded is True
    state = tracker.state
    assert state.locked is True
    assert state.source == "pose_seed"
    assert state.point is not None
    assert state.rect is not None
    assert state.point.x == pytest.approx(240.0)
    assert state.point.y == pytest.approx(300.0)
    assert 0.15 < state.confidence <= 0.55


def test_seed_from_pose_ignored_when_already_locked() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])
    original_point = tracker.state.point

    seeded = tracker.seed_from_pose(_frame(), _pose_with_wrists())

    assert seeded is False
    assert tracker.state.source == "detection"
    assert tracker.state.point == original_point


def test_seed_from_pose_rejects_low_visibility_wrists() -> None:
    tracker = BarAnchorTracker(fps=30.0)

    seeded = tracker.seed_from_pose(
        _frame(),
        _pose_with_wrists(left=(200.0, 300.0, 0.2), right=(280.0, 300.0, 0.9)),
    )

    assert seeded is False
    assert tracker.state.locked is False
    assert tracker.state.source == "lost"


def test_seed_from_pose_works_after_long_detection_gap() -> None:
    tracker = BarAnchorTracker(fps=30.0)
    tracker.update(_frame(), [Detection("plate", 0.8, 100.0, 220.0, 180.0, 300.0)])

    for _ in range(930):
        tracker.update(_frame(), [])

    assert tracker.state.source == "lost"
    assert tracker.state.locked is False

    seeded = tracker.seed_from_pose(_frame(), _pose_with_wrists())

    assert seeded is True
    assert tracker.state.source == "pose_seed"
    assert tracker.state.locked is True
