import numpy as np

from inference import InferenceTransform
from io_video import VideoMetadata, resolve_output_geometry
from pose import PoseKeypoint, PoseResult


def test_transform_uses_active_content_not_portrait_letterbox() -> None:
    geometry = resolve_output_geometry(
        VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=10, codec="mp4v"),
        output_mode="portrait-720",
    )
    transform = InferenceTransform.from_output_geometry(geometry, inference_max_side=960)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    prepared = transform.prepare(frame)

    assert prepared.shape[:2] == (geometry.resized_height, geometry.resized_width)
    assert prepared.shape[:2] != frame.shape[:2]


def test_transform_round_trips_pose_and_mask_to_output_canvas() -> None:
    geometry = resolve_output_geometry(
        VideoMetadata(width=720, height=1280, fps=30.0, frame_count=10, codec="mp4v"),
        output_mode="portrait-720",
    )
    transform = InferenceTransform.from_output_geometry(geometry, inference_max_side=640)
    pose = PoseResult(
        keypoints=[PoseKeypoint("left_wrist", 100.0, 200.0, 1.0)],
        backend="yolo",
        detected=True,
        segmentation_mask=np.full((transform.inference_height, transform.inference_width), 255, dtype=np.uint8),
    )
    output = transform.pose_to_output(pose)

    assert output.keypoints[0].x == 200.0
    assert output.keypoints[0].y == 400.0
    assert output.segmentation_mask is not None
    assert output.segmentation_mask.shape == (1280, 720)
