"""Coordinate-safe resizing for model inference.

The renderer always receives the final output canvas.  Models can work on only
the active video rectangle at a smaller size, then their results are mapped back
to that canvas without moving overlays, masks, or tracking coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detect_objects import Detection, TrackedDetection
from io_video import Frame, OutputGeometry
from pose import PoseKeypoint, PoseResult


@dataclass(frozen=True)
class InferenceTransform:
    output_width: int
    output_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    inference_width: int
    inference_height: int

    @classmethod
    def from_output_geometry(
        cls,
        geometry: OutputGeometry,
        inference_max_side: int,
    ) -> "InferenceTransform":
        crop_width = geometry.resized_width
        crop_height = geometry.resized_height
        max_side = max(crop_width, crop_height)
        scale = 1.0
        if inference_max_side > 0 and max_side > inference_max_side:
            scale = inference_max_side / max_side
        inference_width = max(2, int(round(crop_width * scale)))
        inference_height = max(2, int(round(crop_height * scale)))
        return cls(
            output_width=geometry.metadata.width,
            output_height=geometry.metadata.height,
            crop_x=geometry.pad_x,
            crop_y=geometry.pad_y,
            crop_width=crop_width,
            crop_height=crop_height,
            inference_width=inference_width,
            inference_height=inference_height,
        )

    @property
    def scale_x(self) -> float:
        return self.crop_width / max(1, self.inference_width)

    @property
    def scale_y(self) -> float:
        return self.crop_height / max(1, self.inference_height)

    def prepare(self, frame: Frame) -> Frame:
        content = frame[
            self.crop_y : self.crop_y + self.crop_height,
            self.crop_x : self.crop_x + self.crop_width,
        ]
        if content.shape[1] == self.inference_width and content.shape[0] == self.inference_height:
            return content
        return cv2.resize(
            content,
            (self.inference_width, self.inference_height),
            interpolation=cv2.INTER_AREA,
        )

    def point_to_output(self, x: float, y: float) -> tuple[float, float]:
        return self.crop_x + x * self.scale_x, self.crop_y + y * self.scale_y

    def point_to_inference(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.crop_x) / self.scale_x, (y - self.crop_y) / self.scale_y

    def pose_to_output(self, pose: PoseResult) -> PoseResult:
        keypoints = []
        for keypoint in pose.keypoints:
            x, y = self.point_to_output(keypoint.x, keypoint.y)
            keypoints.append(
                PoseKeypoint(
                    name=keypoint.name,
                    x=x,
                    y=y,
                    visibility=keypoint.visibility,
                )
            )
        person_box = pose.person_box
        if person_box is not None:
            x1, y1 = self.point_to_output(person_box[0], person_box[1])
            x2, y2 = self.point_to_output(person_box[2], person_box[3])
            person_box = (x1, y1, x2, y2)
        return PoseResult(
            keypoints=keypoints,
            backend=pose.backend,
            detected=pose.detected,
            segmentation_mask=self.mask_to_output(pose.segmentation_mask),
            source=pose.source,
            confidence=pose.confidence,
            person_box=person_box,
        )

    def detections_to_output(self, detections: list[Detection]) -> list[Detection]:
        mapped: list[Detection] = []
        for detection in detections:
            x1, y1 = self.point_to_output(detection.x1, detection.y1)
            x2, y2 = self.point_to_output(detection.x2, detection.y2)
            common = dict(
                label=detection.label,
                confidence=detection.confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                color=detection.color,
            )
            if isinstance(detection, TrackedDetection):
                mapped.append(TrackedDetection(**common, track_id=detection.track_id))
            else:
                mapped.append(Detection(**common))
        return mapped

    def mask_to_output(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.ndim == 3:
            # MediaPipe Tasks can return an HxWx1 confidence mask whereas
            # OpenCV's BGR conversion only accepts 3/4-channel images.
            mask = mask[:, :, 0] if mask.shape[2] == 1 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            mask,
            (self.crop_width, self.crop_height),
            interpolation=cv2.INTER_LINEAR,
        )
        result = np.zeros((self.output_height, self.output_width), dtype=resized.dtype)
        result[
            self.crop_y : self.crop_y + self.crop_height,
            self.crop_x : self.crop_x + self.crop_width,
        ] = resized
        return result
