from __future__ import annotations

import numpy as np

from pose import PoseKeypoint, PoseResult, refine_pose_with_mask


def test_refine_pose_with_mask_keeps_landmark_inside_silhouette() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    pose = PoseResult(
        keypoints=[PoseKeypoint("left_hip", 40.0, 40.0, 0.9)],
        backend="yolo",
        detected=True,
    )

    refined = refine_pose_with_mask(pose, mask)

    keypoint = refined.keypoints[0]
    assert keypoint.x == 40.0
    assert keypoint.y == 40.0
    assert keypoint.visibility == 0.9


def test_refine_pose_with_mask_snaps_nearby_landmark_to_silhouette() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    pose = PoseResult(
        keypoints=[PoseKeypoint("left_knee", 62.0, 40.0, 0.8)],
        backend="yolo",
        detected=True,
    )

    refined = refine_pose_with_mask(pose, mask, search_radius_pixels=5)

    keypoint = refined.keypoints[0]
    assert keypoint.x == 59.0
    assert keypoint.y == 40.0
    assert keypoint.visibility == 0.8 * 0.85


def test_refine_pose_with_mask_downweights_far_background_landmark() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    pose = PoseResult(
        keypoints=[PoseKeypoint("left_ankle", 5.0, 5.0, 0.8)],
        backend="yolo",
        detected=True,
    )

    refined = refine_pose_with_mask(pose, mask, search_radius_pixels=4)

    keypoint = refined.keypoints[0]
    assert keypoint.x == 5.0
    assert keypoint.y == 5.0
    assert keypoint.visibility == 0.8 * 0.25
