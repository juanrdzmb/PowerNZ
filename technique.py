from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot
from typing import Literal

from calibration import SpatialCalibration
from detect_objects import Detection
from pose import PoseKeypoint, PoseResult

LiftStateName = Literal["reposo", "inicio", "tirón", "bloqueo", "bajada"]


@dataclass(frozen=True)
class TechniqueAssessment:
    status: str
    cues: list[str]
    torso_angle_degrees: float | None
    bar_to_midfoot_m: float | None
    view: str
    quality_score: float
    quality_label: str


class TechniqueMonitor:
    def __init__(self, stable_frames: int = 10, exercise: str = "deadlift") -> None:
        self._stable_frames = stable_frames
        self._exercise = exercise
        self._last_signature: tuple[str, tuple[str, ...]] | None = None
        self._candidate: TechniqueAssessment | None = None
        self._candidate_frames = 0
        self._stable: TechniqueAssessment | None = None
        self._active_rep_index: int | None = None
        self._active_start_torso_angle: float | None = None
        self._active_bar_x_values: list[float] = []

    def update(
        self,
        pose: PoseResult,
        detections: list[Detection],
        calibration: SpatialCalibration,
        bar_velocity_mps: float = 0.0,
        lift_state: LiftStateName = "reposo",
        rep_index: int = 0,
    ) -> TechniqueAssessment:
        if lift_state == "reposo":
            self._reset_active_rep()
            return _preparation_assessment(pose)

        assessment = assess_technique(
            exercise=self._exercise,
            pose=pose,
            detections=detections,
            calibration=calibration,
            bar_velocity_mps=bar_velocity_mps,
            lift_state=lift_state,
            rep_index=rep_index,
            active_start_torso_angle=self._active_start_torso_angle,
            active_bar_x_values=self._active_bar_x_values,
        )

        self._update_active_rep_state(
            pose=pose,
            detections=detections,
            rep_index=rep_index,
        )
        signature = (assessment.status, tuple(assessment.cues))

        if signature == self._last_signature:
            self._candidate_frames += 1
        else:
            self._candidate = assessment
            self._candidate_frames = 1
            self._last_signature = signature

        if self._stable is None or self._candidate_frames >= self._stable_frames:
            self._stable = self._candidate

        return self._stable or assessment

    def _update_active_rep_state(
        self,
        pose: PoseResult,
        detections: list[Detection],
        rep_index: int,
    ) -> None:
        keypoints = {
            keypoint.name: keypoint
            for keypoint in pose.keypoints
            if keypoint.visibility >= 0.35
        }
        torso_angle = _torso_angle_degrees(keypoints)
        barbell = next((detection for detection in detections if detection.label == "barbell"), None)

        if self._active_rep_index != rep_index:
            self._active_rep_index = rep_index
            self._active_start_torso_angle = torso_angle
            self._active_bar_x_values = []

        if barbell is not None:
            center_x, _ = barbell.center
            self._active_bar_x_values.append(center_x)

        if len(self._active_bar_x_values) > 180:
            self._active_bar_x_values = self._active_bar_x_values[-180:]

    def _reset_active_rep(self) -> None:
        self._active_rep_index = None
        self._active_start_torso_angle = None
        self._active_bar_x_values = []


def assess_deadlift_technique(
    pose: PoseResult,
    detections: list[Detection],
    calibration: SpatialCalibration,
    bar_velocity_mps: float = 0.0,
    lift_state: LiftStateName = "reposo",
    rep_index: int = 0,
    active_start_torso_angle: float | None = None,
    active_bar_x_values: list[float] | None = None,
) -> TechniqueAssessment:
    keypoints = {
        keypoint.name: keypoint
        for keypoint in pose.keypoints
        if keypoint.visibility >= 0.35
    }
    cues: list[str] = []
    view = _estimate_view(keypoints)
    score = 10.0
    active_bar_x_values = active_bar_x_values or []

    if view == "frontal":
        cues.append("vista frontal: tecnica poco fiable")
        score -= 2.0
    elif view == "diagonal":
        cues.append("vista diagonal: tecnica menos fiable")
        score -= 1.0

    torso_angle = _torso_angle_degrees(keypoints)
    if (
        torso_angle is not None
        and active_start_torso_angle is not None
        and lift_state in {"tirón", "bloqueo", "bajada"}
    ):
        torso_drift = abs(torso_angle - active_start_torso_angle)
        if torso_drift > 18.0:
            cues.append("torso cambia demasiado")
            score -= 1.5

    bar_to_midfoot_m = _bar_to_midfoot_distance_m(keypoints, detections, calibration)
    if bar_to_midfoot_m is not None and bar_to_midfoot_m > 0.18:
        cues.append("barra lejos del pie")
        score -= 1.5

    bar_path_drift_m = _bar_path_drift_m(active_bar_x_values, calibration)
    if bar_path_drift_m is not None and bar_path_drift_m > 0.08:
        cues.append("trayectoria de barra poco vertical")
        score -= 2.0

    if abs(bar_velocity_mps) > 2.5:
        cues.append("velocidad inestable")
        score -= 1.0

    score = max(1.0, min(10.0, score))
    quality_label = _quality_label(score)

    if not cues:
        return TechniqueAssessment(
            status="OK",
            cues=["trayectoria estable"],
            torso_angle_degrees=torso_angle,
            bar_to_midfoot_m=bar_to_midfoot_m,
            view=view,
            quality_score=score,
            quality_label=quality_label,
        )

    return TechniqueAssessment(
        status="Revisar",
        cues=cues,
        torso_angle_degrees=torso_angle,
        bar_to_midfoot_m=bar_to_midfoot_m,
        view=view,
        quality_score=score,
        quality_label=quality_label,
        )


def assess_technique(
    exercise: str,
    pose: PoseResult,
    detections: list[Detection],
    calibration: SpatialCalibration,
    bar_velocity_mps: float = 0.0,
    lift_state: LiftStateName = "reposo",
    rep_index: int = 0,
    active_start_torso_angle: float | None = None,
    active_bar_x_values: list[float] | None = None,
) -> TechniqueAssessment:
    """Dispatch technique scoring per exercise. Deadlift keeps its original logic;
    squat/bench use a vertical-bar-path + stability assessment."""
    if exercise == "deadlift":
        return assess_deadlift_technique(
            pose=pose,
            detections=detections,
            calibration=calibration,
            bar_velocity_mps=bar_velocity_mps,
            lift_state=lift_state,
            rep_index=rep_index,
            active_start_torso_angle=active_start_torso_angle,
            active_bar_x_values=active_bar_x_values,
        )
    return _assess_vertical_path_technique(
        exercise=exercise,
        pose=pose,
        calibration=calibration,
        bar_velocity_mps=bar_velocity_mps,
        lift_state=lift_state,
        active_start_torso_angle=active_start_torso_angle,
        active_bar_x_values=active_bar_x_values,
    )


def _assess_vertical_path_technique(
    exercise: str,
    pose: PoseResult,
    calibration: SpatialCalibration,
    bar_velocity_mps: float,
    lift_state: LiftStateName,
    active_start_torso_angle: float | None,
    active_bar_x_values: list[float] | None,
) -> TechniqueAssessment:
    """Squat/bench: reward a vertical bar path and stable velocity. Squat also checks
    torso drift. Richer per-exercise cues (depth, touch point) are a TODO that needs the
    trained model for reliable keypoints."""
    keypoints = {
        keypoint.name: keypoint
        for keypoint in pose.keypoints
        if keypoint.visibility >= 0.35
    }
    cues: list[str] = []
    view = _estimate_view(keypoints)
    score = 10.0
    active_bar_x_values = active_bar_x_values or []
    # Bench is filmed from the side too, but a frontal view is still less reliable.
    if view == "frontal":
        cues.append("vista frontal: tecnica poco fiable")
        score -= 2.0
    elif view == "diagonal":
        cues.append("vista diagonal: tecnica menos fiable")
        score -= 1.0

    drift_threshold = 0.06 if exercise == "bench" else 0.10
    bar_path_drift_m = _bar_path_drift_m(active_bar_x_values, calibration)
    if bar_path_drift_m is not None and bar_path_drift_m > drift_threshold:
        cues.append("trayectoria de barra poco vertical")
        score -= 2.0

    torso_angle = _torso_angle_degrees(keypoints)
    if (
        exercise == "squat"
        and torso_angle is not None
        and active_start_torso_angle is not None
        and lift_state in {"tirón", "bloqueo", "bajada"}
    ):
        torso_drift = abs(torso_angle - active_start_torso_angle)
        if torso_drift > 22.0:
            cues.append("torso cambia demasiado")
            score -= 1.5

    if abs(bar_velocity_mps) > 2.5:
        cues.append("velocidad inestable")
        score -= 1.0

    score = max(1.0, min(10.0, score))
    quality_label = _quality_label(score)
    if not cues:
        return TechniqueAssessment(
            status="OK",
            cues=["trayectoria estable"],
            torso_angle_degrees=torso_angle,
            bar_to_midfoot_m=None,
            view=view,
            quality_score=score,
            quality_label=quality_label,
        )
    return TechniqueAssessment(
        status="Revisar",
        cues=cues,
        torso_angle_degrees=torso_angle,
        bar_to_midfoot_m=None,
        view=view,
        quality_score=score,
        quality_label=quality_label,
    )


def _preparation_assessment(pose: PoseResult) -> TechniqueAssessment:
    keypoints = {
        keypoint.name: keypoint
        for keypoint in pose.keypoints
        if keypoint.visibility >= 0.35
    }
    view = _estimate_view(keypoints)
    return TechniqueAssessment(
        status="Preparacion",
        cues=["sin puntuar hasta iniciar tiron"],
        torso_angle_degrees=_torso_angle_degrees(keypoints),
        bar_to_midfoot_m=None,
        view=view,
        quality_score=0.0,
        quality_label="pendiente",
    )


def _bar_path_drift_m(
    bar_x_values: list[float],
    calibration: SpatialCalibration,
) -> float | None:
    if len(bar_x_values) < 8:
        return None

    return (max(bar_x_values) - min(bar_x_values)) * calibration.meters_per_pixel


def _quality_label(score: float) -> str:
    if score >= 8.5:
        return "excelente"

    if score >= 7.0:
        return "buena"

    if score >= 5.5:
        return "mejorable"

    return "revisar"


def _estimate_view(keypoints: dict[str, PoseKeypoint]) -> str:
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")
    left_ankle = keypoints.get("left_ankle")
    right_ankle = keypoints.get("right_ankle")

    widths = []
    for first, second in ((left_shoulder, right_shoulder), (left_hip, right_hip), (left_ankle, right_ankle)):
        if first is not None and second is not None:
            widths.append(abs(first.x - second.x))

    if not widths:
        return "desconocida"

    average_width = sum(widths) / len(widths)
    body_height = _visible_body_height(keypoints)
    if body_height is None or body_height <= 0:
        if average_width < 45:
            return "lateral"

        if average_width < 130:
            return "diagonal"

        return "frontal"

    width_ratio = average_width / body_height
    if width_ratio < 0.16:
        return "lateral"

    if width_ratio < 0.34:
        return "diagonal"

    return "frontal"


def _visible_body_height(keypoints: dict[str, PoseKeypoint]) -> float | None:
    y_values = [
        keypoint.y
        for name, keypoint in keypoints.items()
        if name
        in {
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        }
    ]
    if len(y_values) < 2:
        return None

    return max(y_values) - min(y_values)


def _torso_angle_degrees(keypoints: dict[str, PoseKeypoint]) -> float | None:
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")

    shoulder = _midpoint(left_shoulder, right_shoulder)
    hip = _midpoint(left_hip, right_hip)
    if shoulder is None or hip is None:
        return None

    dx = shoulder.x - hip.x
    dy = shoulder.y - hip.y
    length = hypot(dx, dy)
    if length <= 0:
        return None

    horizontal_dot = abs(dx) / length
    return float(degrees(acos(max(-1.0, min(1.0, horizontal_dot)))))


def _bar_to_midfoot_distance_m(
    keypoints: dict[str, PoseKeypoint],
    detections: list[Detection],
    calibration: SpatialCalibration,
) -> float | None:
    barbell = next((detection for detection in detections if detection.label == "barbell"), None)
    if barbell is None:
        return None

    left_ankle = keypoints.get("left_ankle")
    right_ankle = keypoints.get("right_ankle")
    left_foot = keypoints.get("left_foot_index")
    right_foot = keypoints.get("right_foot_index")
    ankle = _midpoint(left_ankle, right_ankle)
    toe = _midpoint(left_foot, right_foot)
    if ankle is None:
        return None

    midfoot_x = ankle.x if toe is None else (ankle.x + toe.x) / 2.0
    bar_x, _ = barbell.center
    return abs(bar_x - midfoot_x) * calibration.meters_per_pixel


def _midpoint(
    first: PoseKeypoint | None,
    second: PoseKeypoint | None,
) -> PoseKeypoint | None:
    if first is None and second is None:
        return None

    if first is None:
        return second

    if second is None:
        return first

    return PoseKeypoint(
        name=f"{first.name}_{second.name}_midpoint",
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        visibility=(first.visibility + second.visibility) / 2.0,
    )
