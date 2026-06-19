import numpy as np

from pose import PoseKeypoint, PoseResult, fuse_pose_results


def _pose(backend: str, points: list[PoseKeypoint], box=None) -> PoseResult:
    return PoseResult(
        keypoints=points,
        backend=backend,  # type: ignore[arg-type]
        detected=True,
        segmentation_mask=np.full((100, 100), 255, dtype=np.uint8),
        person_box=box,
        confidence=0.9,
    )


def test_fusion_keeps_mediapipe_landmarks_when_it_matches_yolo_athlete() -> None:
    yolo = _pose("yolo", [PoseKeypoint("left_wrist", 40, 50, 0.6)], (10, 10, 90, 90))
    mediapipe = _pose(
        "mediapipe",
        [
            PoseKeypoint("left_wrist", 41, 50, 0.9),
            PoseKeypoint("right_wrist", 55, 50, 0.9),
            PoseKeypoint("left_shoulder", 40, 30, 0.9),
            PoseKeypoint("right_shoulder", 55, 30, 0.9),
            PoseKeypoint("left_hip", 42, 70, 0.9),
            PoseKeypoint("right_hip", 54, 70, 0.9),
        ],
    )

    fused = fuse_pose_results(yolo, mediapipe)
    assert fused.backend == "auto"
    assert fused.source == "hybrid"
    assert next(point for point in fused.keypoints if point.name == "left_wrist").visibility == 0.9


def test_fusion_rejects_mediapipe_pose_from_other_person() -> None:
    yolo = _pose("yolo", [PoseKeypoint("left_wrist", 40, 50, 0.9)], (10, 10, 45, 90))
    mediapipe = _pose(
        "mediapipe",
        [PoseKeypoint(f"point_{index}", 80, 80, 0.9) for index in range(6)],
    )

    assert fuse_pose_results(yolo, mediapipe) is yolo
