import numpy as np

from pose import PoseKeypoint, PoseResult
from segmentation import SegmentationResult, select_subject_mask


def test_mask_selection_prefers_candidate_that_contains_pose() -> None:
    pose = PoseResult(
        keypoints=[PoseKeypoint("left_hip", 20, 20, 1.0), PoseKeypoint("right_hip", 25, 20, 1.0)],
        backend="yolo",
        detected=True,
    )
    wrong = np.zeros((50, 50), dtype=np.uint8)
    wrong[35:45, 35:45] = 255
    right = np.zeros((50, 50), dtype=np.uint8)
    right[10:30, 10:30] = 255

    selected = select_subject_mask(
        [SegmentationResult(wrong, "yolo", 0.85), SegmentationResult(right, "mediapipe", 0.80)],
        pose,
    )
    assert selected.backend == "mediapipe"
