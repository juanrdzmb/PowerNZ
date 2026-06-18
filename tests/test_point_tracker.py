from __future__ import annotations

import pytest

from track import Point2D, PointTracker


def test_point_tracker_first_sample_is_returned_as_filtered() -> None:
    tracker = PointTracker(frequency_hz=30.0, min_cutoff=1000.0, beta=0.0, deadband_pixels=0.0)

    tracked = tracker.update(Point2D(10.0, 20.0), frame_index=0)

    assert tracked.is_valid is True
    assert tracked.missing_frames == 0
    assert tracked.filtered is not None
    assert tracked.filtered.x == 10.0
    assert tracked.filtered.y == 20.0


def test_point_tracker_survives_short_detection_loss() -> None:
    tracker = PointTracker(frequency_hz=30.0, max_missing_frames=3, min_cutoff=1000.0, beta=0.0)

    tracker.update(Point2D(10.0, 20.0), frame_index=0)
    tracked = tracker.update(None, frame_index=1)

    assert tracked.is_valid is True
    assert tracked.missing_frames == 1
    assert tracked.filtered is not None


def test_point_tracker_loses_track_after_too_many_missing_frames() -> None:
    tracker = PointTracker(frequency_hz=30.0, max_missing_frames=1, min_cutoff=1000.0, beta=0.0)

    tracker.update(Point2D(10.0, 20.0), frame_index=0)
    still_valid = tracker.update(None, frame_index=1)
    lost = tracker.update(None, frame_index=2)

    assert still_valid.is_valid is True
    assert lost.is_valid is False
    assert lost.filtered is None


def test_point_tracker_resumes_after_valid_sample() -> None:
    tracker = PointTracker(frequency_hz=30.0, max_missing_frames=1, min_cutoff=1000.0, beta=0.0)

    tracker.update(Point2D(10.0, 20.0), frame_index=0)
    tracker.update(None, frame_index=1)
    tracker.update(None, frame_index=2)
    tracked = tracker.update(Point2D(30.0, 40.0), frame_index=3)

    assert tracked.is_valid is True
    assert tracked.missing_frames == 0
    assert tracked.filtered is not None
    assert tracked.filtered.x == pytest.approx(30.0, abs=0.5)
    assert tracked.filtered.y == pytest.approx(40.0, abs=0.5)


def test_point_tracker_respects_deadband() -> None:
    tracker = PointTracker(
        frequency_hz=30.0,
        min_cutoff=1000.0,
        beta=0.0,
        deadband_pixels=5.0,
    )

    tracker.update(Point2D(10.0, 20.0), frame_index=0)
    tracked = tracker.update(Point2D(10.5, 20.5), frame_index=1)

    assert tracked.is_valid is True
    assert tracked.filtered is not None
    assert tracked.filtered.x == 10.0
    assert tracked.filtered.y == 20.0
