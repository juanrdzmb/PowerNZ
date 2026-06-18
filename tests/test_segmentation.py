import numpy as np

import segmentation
from pose import PoseKeypoint, PoseResult
from segmentation import PoseHullSegmenter, _person_seg_model_candidates, create_segmenter


def test_pose_hull_segmenter_creates_mask_from_visible_keypoints() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_shoulder", 120.0, 60.0, 0.9),
            PoseKeypoint("right_shoulder", 180.0, 60.0, 0.9),
            PoseKeypoint("left_hip", 125.0, 130.0, 0.9),
            PoseKeypoint("right_hip", 175.0, 130.0, 0.9),
            PoseKeypoint("left_knee", 130.0, 190.0, 0.9),
            PoseKeypoint("right_knee", 170.0, 190.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )

    result = PoseHullSegmenter().segment(frame, pose)

    assert result.mask is not None
    assert result.mask.shape == frame.shape[:2]
    assert np.max(result.mask) > 0
    assert result.backend == "pose-hull"


def test_create_segmenter_none_returns_empty_mask() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    pose = PoseResult(keypoints=[], backend="yolo", detected=False)

    result = create_segmenter("none").segment(frame, pose)

    assert result.mask is None
    assert result.backend == "none"


def test_pose_hull_segmenter_keeps_previous_mask_when_pose_jumps_to_background() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    segmenter = PoseHullSegmenter()
    foreground_pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_shoulder", 90.0, 60.0, 0.9),
            PoseKeypoint("right_shoulder", 130.0, 60.0, 0.9),
            PoseKeypoint("left_hip", 95.0, 120.0, 0.9),
            PoseKeypoint("right_hip", 125.0, 120.0, 0.9),
            PoseKeypoint("left_knee", 100.0, 180.0, 0.9),
            PoseKeypoint("right_knee", 120.0, 180.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )
    background_pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_shoulder", 250.0, 40.0, 0.9),
            PoseKeypoint("right_shoulder", 285.0, 40.0, 0.9),
            PoseKeypoint("left_hip", 252.0, 90.0, 0.9),
            PoseKeypoint("right_hip", 282.0, 90.0, 0.9),
            PoseKeypoint("left_knee", 255.0, 140.0, 0.9),
            PoseKeypoint("right_knee", 280.0, 140.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )

    first = segmenter.segment(frame, foreground_pose)
    jumped = segmenter.segment(frame, background_pose)

    assert first.mask is not None
    assert jumped.mask is not None
    assert np.array_equal(jumped.mask, first.mask)


def test_auto_segmentation_prefers_powerai_athlete_model_when_present(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "powerai_athlete_seg.pt").touch()
    monkeypatch.setattr(segmentation, "__file__", str(tmp_path / "segmentation.py"))

    candidates = _person_seg_model_candidates(None)

    assert candidates[0].endswith("powerai_athlete_seg.pt")
