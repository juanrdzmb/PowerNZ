from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pose import _person_center, _select_pose_person_index


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, index: int) -> "_FakeTensor":
        return _FakeTensor(self._array[index])

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _make_yolo_result(
    boxes_xyxy: list[tuple[float, float, float, float]],
) -> SimpleNamespace:
    xyxy_array = _FakeTensor(np.asarray(boxes_xyxy, dtype=np.float32))
    boxes = SimpleNamespace(xyxy=xyxy_array)
    return SimpleNamespace(boxes=boxes)


def test_select_pose_person_index_picks_largest_when_no_previous() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (100.0, 100.0, 180.0, 220.0),
        ]
    )

    chosen = _select_pose_person_index(result, previous_center=None, max_jump_pixels=250.0)

    assert chosen == 1


def test_select_pose_person_index_prefers_person_near_previous_center() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (300.0, 300.0, 360.0, 420.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(20.0, 30.0),
        max_jump_pixels=250.0,
    )

    assert chosen == 0


def test_select_pose_person_index_falls_back_to_largest_on_large_jump() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (300.0, 300.0, 360.0, 420.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(500.0, 500.0),
        max_jump_pixels=50.0,
    )

    assert chosen == 1


def test_select_pose_person_index_handles_single_box() -> None:
    result = _make_yolo_result([(10.0, 10.0, 30.0, 50.0)])

    chosen = _select_pose_person_index(
        result,
        previous_center=(1000.0, 1000.0),
        max_jump_pixels=50.0,
    )

    assert chosen == 0


def test_person_center_returns_midpoint() -> None:
    result = _make_yolo_result(
        [
            (10.0, 20.0, 30.0, 60.0),
            (100.0, 200.0, 300.0, 600.0),
        ]
    )

    assert _person_center(result, 0) == (20.0, 40.0)
    assert _person_center(result, 1) == (200.0, 400.0)


def test_select_pose_person_index_prefers_bar_center() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (300.0, 300.0, 360.0, 420.0),
            (50.0, 400.0, 90.0, 440.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(600.0, 600.0),
        max_jump_pixels=250.0,
        preferred_center=(70.0, 420.0),
    )

    assert chosen == 2


def test_select_pose_person_index_bar_owner_overrides_continuity() -> None:
    # The person whose box encloses the bar hub is the athlete, even if another
    # person is closer to the previously tracked center (e.g. a bystander).
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (300.0, 300.0, 360.0, 420.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(20.0, 30.0),
        max_jump_pixels=250.0,
        preferred_center=(330.0, 360.0),
    )

    assert chosen == 1


def test_select_pose_person_index_continuity_when_no_bar_owner() -> None:
    # With no bar hub provided, continuity still wins.
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (300.0, 300.0, 360.0, 420.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(20.0, 30.0),
        max_jump_pixels=250.0,
    )

    assert chosen == 0


def test_select_pose_person_index_prefers_locked_box() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (200.0, 200.0, 280.0, 320.0),
            (500.0, 500.0, 540.0, 540.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(400.0, 400.0),
        max_jump_pixels=250.0,
        locked_box=(210.0, 210.0, 270.0, 310.0),
        lock_max_jump_pixels=200.0,
    )

    assert chosen == 1


def test_select_pose_person_locked_box_ignored_if_far_away() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (500.0, 500.0, 540.0, 540.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(510.0, 510.0),
        max_jump_pixels=250.0,
        locked_box=(200.0, 200.0, 280.0, 280.0),
        lock_max_jump_pixels=80.0,
    )

    assert chosen == 1


def test_select_pose_person_lock_rejected_when_size_mismatch() -> None:
    result = _make_yolo_result(
        [
            (10.0, 10.0, 30.0, 50.0),
            (200.0, 200.0, 300.0, 320.0),
        ]
    )

    chosen = _select_pose_person_index(
        result,
        previous_center=(20.0, 30.0),
        max_jump_pixels=250.0,
        locked_box=(210.0, 210.0, 230.0, 230.0),
        lock_max_jump_pixels=200.0,
    )

    assert chosen == 0
