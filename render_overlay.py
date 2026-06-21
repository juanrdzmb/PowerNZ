from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from anchor_metrics import AnchorVelocity
from bar_anchor import BarAnchorState
from detect_objects import Detection
from io_video import Frame
from load_estimation import LoadEstimate
from track import Point2D
from metrics import KinematicSample
from pose import PoseKeypoint, PoseResult
from reporting import RepReport
from technique import TechniqueAssessment

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - OpenCV font fallback is tested implicitly
    Image = None
    ImageDraw = None
    ImageFont = None


BarPathPoint = tuple[float, float] | None


POSE_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

BIOMECHANICAL_EDGES = [
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]

BIOMECHANICAL_KEYPOINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


# --- Professional font (Hershey-Duplex: elegant, highly legible) ------------
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
UI_FONT_PLAIN = cv2.FONT_HERSHEY_DUPLEX
TTF_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)


# --- Cohesive UI palette (BGR) for a clean, professional overlay ------------
PANEL_BG = (24, 21, 25)          # near-black glass, slightly cool
PANEL_BORDER = (64, 58, 66)      # subtle hairline
ACCENT = (224, 198, 96)          # refined aqua/cyan (BGR) — the single brand accent
ACCENT_SOFT = (150, 132, 64)
TEXT = (245, 245, 243)
TEXT_DIM = (166, 160, 156)       # muted neutral gray for captions
POS_COLOR = (120, 235, 140)      # upward velocity (green)
NEG_COLOR = (95, 120, 240)       # downward velocity (red)
NEUTRAL_COLOR = (208, 208, 208)
PLATE_BOX_COLOR = (224, 198, 96)  # aqua box around the plate (matches accent)
HUB_COLOR = (70, 200, 250)        # amber hub marker
# Trajectory shares the hub's amber so the bar path reads as the hub's trail, and stands
# out from the cyan plate box (same-colour lines used to camouflage the path).
TRAJECTORY_COLOR = (80, 205, 255)   # bright amber core
TRAJECTORY_GLOW = (18, 85, 140)     # dark amber halo for contrast on any background
SILHOUETTE_TINT = (238, 215, 80)    # teal-blue athlete fill, distinct from the white skeleton
SILHOUETTE_OUTLINE = (236, 228, 158)


# OpenCV's Hershey fonts have no glyphs for accented characters, so Spanish text
# like "tirón" renders as "tir??n". Transliterate to ASCII before drawing.
_ASCII_MAP = str.maketrans(
    {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "ü": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ñ": "N", "Ü": "U",
        "¿": "?", "¡": "!", "·": "-", "—": "-", "–": "-",
    }
)


def _ascii(text: str) -> str:
    return text.translate(_ASCII_MAP)


@lru_cache(maxsize=32)
def _load_ttf_font(pixel_size: int):
    if ImageFont is None:
        return None
    for font_path in TTF_FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), pixel_size)
        except OSError:
            continue
    return None


def _draw_ttf_text(
    frame: Frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    shadow: bool,
) -> bool:
    if Image is None or ImageDraw is None:
        return False

    font_size = max(10, int(round(scale * 34)))
    font = _load_ttf_font(font_size)
    if font is None:
        return False

    scratch = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    pad = max(2, int(font_size * 0.18))
    shadow_offset = 1 if shadow else 0
    overlay_w = text_w + pad * 2 + shadow_offset
    overlay_h = text_h + pad * 2 + shadow_offset
    x1 = int(origin[0])
    y1 = int(origin[1] - text_h - pad)
    x2 = x1 + overlay_w
    y2 = y1 + overlay_h
    if x2 <= 0 or y2 <= 0 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
        return True

    overlay = Image.new("RGBA", (overlay_w, overlay_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_origin = (pad - bbox[0], pad - bbox[1])
    if shadow:
        draw.text(
            (text_origin[0] + shadow_offset, text_origin[1] + shadow_offset),
            text,
            font=font,
            fill=(0, 0, 0, 210),
        )
    draw.text(text_origin, text, font=font, fill=(color[2], color[1], color[0], 255))

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(frame.shape[1], x2)
    crop_y2 = min(frame.shape[0], y2)
    overlay_crop = overlay.crop((crop_x1 - x1, crop_y1 - y1, crop_x2 - x1, crop_y2 - y1))
    overlay_array = np.asarray(overlay_crop, dtype=np.float32)
    alpha = overlay_array[:, :, 3:4] / 255.0
    if not np.any(alpha > 0):
        return True
    rgb = overlay_array[:, :, :3][:, :, ::-1]
    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
    blended = roi * (1.0 - alpha) + rgb * alpha
    frame[crop_y1:crop_y2, crop_x1:crop_x2] = blended.clip(0, 255).astype(np.uint8)
    return True


# Velocity chart series for the comparative overlay. Ordered so the bar
# (the lift's headline metric) is drawn last and stays on top. Elbow is
# omitted: in a deadlift the arms stay straight, so it just duplicates the
# wrist/bar curve and clutters the chart.
ANCHOR_CHART_SERIES = (
    ("knee", "Knee", (80, 190, 255)),
    ("hip", "Hip", (96, 220, 170)),
    ("shoulder", "Shoulder", (245, 175, 95)),
    ("bar", "Bar", TRAJECTORY_COLOR),
)


ANCHOR_SOURCE_COLORS = {
    "detection": (210, 200, 70),
    "prediction": (185, 170, 95),
    "pose_seed": (110, 220, 180),
    "wrist": (80, 230, 160),
    "optical_flow": (210, 175, 60),
    "template": (210, 175, 60),
    "hold": (150, 150, 150),
    "lost": (90, 90, 90),
}


def _anchor_color(source: str) -> tuple[int, int, int]:
    return ANCHOR_SOURCE_COLORS.get(source, PLATE_BOX_COLOR)


def color_for_velocity(velocity_mps: float) -> tuple[int, int, int]:
    if velocity_mps > 0.05:
        return POS_COLOR
    if velocity_mps < -0.05:
        return NEG_COLOR
    return NEUTRAL_COLOR


@dataclass(frozen=True)
class OverlayConfig:
    keypoint_visibility_threshold: float = 0.35
    glow_radius: int = 20
    glow_strength: float = 0.07
    panel_alpha: float = 0.72
    background_dim_alpha: float = 0.5
    silhouette_alpha: float = 0.34
    pose_smoothing_alpha: float = 0.08
    pose_deadband_pixels: float = 10.0
    pose_max_jump_ratio: float = 0.18
    path_max_jump_pixels: float = 85.0
    biomechanical_skeleton_only: bool = True
    velocity_chart_mode: str = "bar"
    body_velocity_display: str = "compact"
    # The tracked rect is sized for the metric scale (~0.72 of the plate); enlarge it
    # only for display so the corner brackets sit on the disc edge.
    plate_box_display_scale: float = 1.36
    plate_box_style: str = "corners"
    velocity_window_seconds: float = 4.5
    visual_hold_frames: int = 8


class OverlayRenderer:
    def __init__(self, config: OverlayConfig = OverlayConfig()) -> None:
        self._config = config
        self._smoothed_keypoints: dict[str, PoseKeypoint] = {}
        self._pose_missing_frames = 0
        self._max_pose_hold_frames = 60

    @staticmethod
    def _ui_scale(frame: Frame) -> float:
        return float(max(0.42, min(1.0, frame.shape[1] / 720.0)))

    @staticmethod
    def _is_compact(frame: Frame) -> bool:
        return frame.shape[1] < 520

    @staticmethod
    def _rects_overlap_x(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0])

    @staticmethod
    def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    def _telemetry_rect(self, frame: Frame) -> tuple[int, int, int, int]:
        """Fixed top-left telemetry panel rect, shared by drawing and layout so other
        panels can avoid overlapping it (important on 9:16 / short frames)."""
        scale = self._ui_scale(frame)
        compact = self._is_compact(frame)
        margin = max(8, int(24 * scale))
        panel_width = frame.shape[1] - margin * 2 if compact else min(560, frame.shape[1] - margin * 2)
        panel_height = max(96, int((136 if compact else 246) * scale))
        return margin, margin, margin + panel_width, margin + panel_height

    @staticmethod
    def _telemetry_stats(
        sample: KinematicSample,
        completed_reps: int,
        bar_drift_cm: float | None,
        load_estimate: LoadEstimate | None,
        rep_text: str | None = None,
    ) -> list[tuple[str, str]]:
        drift_val = "-" if bar_drift_cm is None else f"{bar_drift_cm:.1f} cm"
        stats = [
            ("REP", rep_text or str(completed_reps)),
            ("ROM", f"{sample.rep_displacement_m:.2f} m"),
            ("DRIFT", drift_val),
        ]
        if load_estimate is not None:
            stats.append(("CARGA", f"{load_estimate.total_kg:.0f} kg"))
        return stats

    @staticmethod
    def _rounded_panel(
        frame: Frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        alpha: float = 0.72,
        radius: int = 14,
        border: tuple[int, int, int] | None = PANEL_BORDER,
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        """Translucent rounded panel with an optional thin accent bar on top."""
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        radius = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), PANEL_BG, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), PANEL_BG, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(overlay, (cx, cy), radius, PANEL_BG, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

        if border is not None:
            cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), border, 1, cv2.LINE_AA)
            cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), border, 1, cv2.LINE_AA)
            cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), border, 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), border, 1, cv2.LINE_AA)
        if accent is not None:
            cv2.line(frame, (x1 + radius, y1 + 1), (x1 + radius + int((x2 - x1) * 0.30), y1 + 1), accent, 2, cv2.LINE_AA)

    @staticmethod
    def _text(
        frame: Frame,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int] = TEXT,
        thickness: int = 1,
        shadow: bool = True,
    ) -> None:
        text = _ascii(text)
        if _draw_ttf_text(frame, text, origin, scale, color, thickness, shadow):
            return
        if shadow:
            cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1), UI_FONT,
                        scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, origin, UI_FONT, scale, color, thickness, cv2.LINE_AA)

    def _label_pill(
        self,
        frame: Frame,
        text: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
        prefer_above: bool = True,
    ) -> None:
        scale = self._ui_scale(frame)
        font_scale = max(0.34, 0.48 * scale)
        thickness = max(1, int(2 * scale))
        pad_x = max(5, int(7 * scale))
        pad_y = max(3, int(5 * scale))
        text_size, _ = cv2.getTextSize(text, UI_FONT, font_scale, thickness)
        x1 = anchor[0]
        y2 = anchor[1] - max(4, int(5 * scale)) if prefer_above else anchor[1] + text_size[1] + pad_y * 2
        y1 = y2 - text_size[1] - pad_y * 2
        x2 = x1 + text_size[0] + pad_x * 2

        if y1 < 2:
            y1 = anchor[1] + max(4, int(5 * scale))
            y2 = y1 + text_size[1] + pad_y * 2
        if x2 > frame.shape[1] - 2:
            shift = x2 - (frame.shape[1] - 2)
            x1 -= shift
            x2 -= shift
        x1 = max(2, x1)
        x2 = min(frame.shape[1] - 2, x2)
        if y2 > frame.shape[0] - 2:
            shift = y2 - (frame.shape[0] - 2)
            y1 -= shift
            y2 -= shift

        self._rounded_panel(
            frame,
            x1,
            y1,
            x2,
            y2,
            alpha=0.82,
            radius=max(4, int(7 * scale)),
            border=color,
        )
        self._text(frame, text, (x1 + pad_x, y2 - pad_y - 1), font_scale, color, thickness)

    def render(
        self,
        frame: Frame,
        pose: PoseResult | None = None,
        detections: list[Detection] | None = None,
        sample: KinematicSample | None = None,
        completed_reps: int = 0,
        total_reps: int | None = None,
        technique: TechniqueAssessment | None = None,
        bar_path: list[BarPathPoint] | None = None,
        velocity_history: list[float] | None = None,
        anchor_velocity_history: dict[str, list[float]] | None = None,
        velocity_frame_history: list[int] | None = None,
        chart_max_abs: float | None = None,
        video_fps: float = 30.0,
        anchor_velocities: list[AnchorVelocity] | None = None,
        rep_reports: list[RepReport] | None = None,
        bar_anchor: BarAnchorState | None = None,
        subject_mask: np.ndarray | None = None,
        load_estimate: LoadEstimate | None = None,
        bar_drift_cm: float | None = None,
        debug_anchor: bool = False,
    ) -> Frame:
        output = frame.copy()
        pose = pose or PoseResult(keypoints=[], backend="mediapipe", detected=False)
        detections = detections or []
        pose = self._smooth_pose(pose, output)

        self._dim_background(output)
        self._draw_subject_silhouette(output, subject_mask)
        self._draw_silhouette_outline(output, subject_mask)
        compact = self._is_compact(output)
        telemetry_rect = self._telemetry_rect(output)
        self._draw_subject_glow(output, pose, detections)
        self._draw_pose(output, pose)
        if debug_anchor:
            self._draw_detections(output, detections)
        if bar_anchor is not None:
            self._draw_bar_anchor(output, bar_anchor, detections, debug_anchor)
        self._draw_bar_path(output, bar_path or [])
        if self._config.body_velocity_display == "compact" and not compact:
            self._draw_anchor_velocities(output, anchor_velocities or [])
        history = anchor_velocity_history
        if history is None and velocity_history:
            history = {"bar": velocity_history}
        chart_top = self._draw_multi_velocity_chart(
            output,
            history or {},
            rep_reports or [],
            velocity_frame_history or [],
            max_abs_override=chart_max_abs,
            video_fps=video_fps,
        )
        self._draw_rep_table(output, rep_reports or [], bottom_limit=chart_top, avoid_rect=telemetry_rect)
        # The floating bar-velocity badge was dropped: it sat on top of the plate/hub boxes.
        # The same value lives in the telemetry hero and the bottom chart.
        self._draw_telemetry_panel(output, sample, completed_reps, technique, bar_drift_cm, load_estimate, total_reps)
        return output

    def _dim_background(self, frame: Frame) -> None:
        frame[:] = (frame.astype(np.float32) * (1.0 - self._config.background_dim_alpha)).astype(np.uint8)

    def _draw_subject_silhouette(self, frame: Frame, mask: np.ndarray | None) -> None:
        if mask is None:
            return

        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        alpha = (mask.astype(np.float32) / 255.0) * self._config.silhouette_alpha
        if not np.any(alpha > 0.01):
            return

        # A cool, translucent tint keeps the real athlete visible and clearly
        # separates their trained segmentation from the dimmed gym background.
        silhouette = np.full_like(frame, SILHOUETTE_TINT, dtype=np.uint8)
        alpha_3 = alpha[:, :, None]
        blended = frame.astype(np.float32) * (1.0 - alpha_3) + silhouette.astype(np.float32) * alpha_3
        frame[:] = blended.clip(0, 255).astype(np.uint8)

    def _draw_silhouette_outline(self, frame: Frame, mask: np.ndarray | None) -> None:
        """Dibuja un contorno claro alrededor del atleta."""
        if mask is None:
            return
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        binary = (mask > 110).astype(np.uint8) * 255
        if not np.any(binary):
            return
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        scale = self._ui_scale(frame)
        contours = [c for c in contours if cv2.contourArea(c) > 400 * scale * scale]
        if not contours:
            return

        glow = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(glow, contours, -1, 255, max(3, int(7 * scale)), cv2.LINE_AA)
        radius = self._ensure_odd(int(13 * scale))
        glow = cv2.GaussianBlur(glow, (radius, radius), 0)
        tint = np.zeros_like(frame, dtype=np.float32)
        for channel in range(3):
            tint[:, :, channel] = glow.astype(np.float32) / 255.0 * ACCENT[channel] * 0.45
        np.clip(frame.astype(np.float32) + tint, 0, 255, out=tint)
        frame[:] = tint.astype(np.uint8)

        cv2.drawContours(frame, contours, -1, SILHOUETTE_OUTLINE, max(1, int(2 * scale)), cv2.LINE_AA)

    def _smooth_pose(self, pose: PoseResult, frame: Frame) -> PoseResult:
        if not pose.detected:
            self._pose_missing_frames += 1
            if self._smoothed_keypoints and self._pose_missing_frames <= self._max_pose_hold_frames:
                return PoseResult(
                    keypoints=list(self._smoothed_keypoints.values()),
                    backend=pose.backend,
                    detected=True,
                )
            return pose

        alpha = self._config.pose_smoothing_alpha
        smoothed: list[PoseKeypoint] = []
        self._pose_missing_frames = 0
        max_jump = max(frame.shape[:2]) * self._config.pose_max_jump_ratio

        for keypoint in pose.keypoints:
            previous = self._smoothed_keypoints.get(keypoint.name)
            if previous is None:
                new_keypoint = keypoint
            else:
                delta = float(np.hypot(keypoint.x - previous.x, keypoint.y - previous.y))
                if delta < self._config.pose_deadband_pixels:
                    new_keypoint = PoseKeypoint(
                        name=keypoint.name,
                        x=previous.x,
                        y=previous.y,
                        visibility=max(keypoint.visibility, previous.visibility * 0.96),
                    )
                elif delta > max_jump and previous.visibility >= self._config.keypoint_visibility_threshold:
                    new_keypoint = PoseKeypoint(
                        name=keypoint.name,
                        x=previous.x,
                        y=previous.y,
                        visibility=previous.visibility * 0.92,
                    )
                else:
                    new_keypoint = PoseKeypoint(
                        name=keypoint.name,
                        x=alpha * keypoint.x + (1.0 - alpha) * previous.x,
                        y=alpha * keypoint.y + (1.0 - alpha) * previous.y,
                        visibility=max(keypoint.visibility, previous.visibility * 0.94),
                    )

            self._smoothed_keypoints[keypoint.name] = new_keypoint
            smoothed.append(new_keypoint)

        return PoseResult(keypoints=smoothed, backend=pose.backend, detected=pose.detected)

    def _draw_subject_glow(
        self,
        frame: Frame,
        pose: PoseResult,
        detections: list[Detection],
    ) -> None:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        visible = self._visible_keypoints(pose)

        edges = BIOMECHANICAL_EDGES if self._config.biomechanical_skeleton_only else POSE_EDGES

        for start_name, end_name in edges:
            start = visible.get(start_name)
            end = visible.get(end_name)
            if start is not None and end is not None:
                cv2.line(mask, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 255, max(4, int(10 * self._ui_scale(frame))), cv2.LINE_AA)

        for keypoint in visible.values():
            if self._config.biomechanical_skeleton_only and keypoint.name not in BIOMECHANICAL_KEYPOINTS:
                continue

            cv2.circle(mask, (int(keypoint.x), int(keypoint.y)), max(5, int(12 * self._ui_scale(frame))), 255, -1, cv2.LINE_AA)

        for detection in detections:
            center_x, center_y = detection.center
            if detection.label == "plate":
                radius = int(max(14.0, min(detection.width, detection.height) * 0.10))
                cv2.circle(mask, (int(center_x), int(center_y)), max(4, int(radius * self._ui_scale(frame))), 255, -1, cv2.LINE_AA)
            elif detection.label == "barbell":
                cv2.circle(mask, (int(center_x), int(center_y)), max(5, int(18 * self._ui_scale(frame))), 255, -1, cv2.LINE_AA)

        if not np.any(mask):
            return

        kernel = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        radius = self._ensure_odd(int(self._config.glow_radius * self._ui_scale(frame)))
        blurred = cv2.GaussianBlur(dilated, (radius, radius), 0)
        glow_f = blurred.astype(np.float32) / 255.0 * self._config.glow_strength
        tinted = frame.astype(np.float32)
        for channel in range(3):
            tinted[:, :, channel] += glow_f * ACCENT[channel]
        np.clip(tinted, 0, 255, out=tinted)
        frame[:] = tinted.astype(np.uint8)

    def _draw_pose(self, frame: Frame, pose: PoseResult) -> None:
        visible = self._visible_keypoints(pose)
        scale = self._ui_scale(frame)
        edges = BIOMECHANICAL_EDGES if self._config.biomechanical_skeleton_only else POSE_EDGES
        core = max(2, int(3 * scale))

        for start_name, end_name in edges:
            start = visible.get(start_name)
            end = visible.get(end_name)
            if start is None or end is None:
                continue
            p1 = (int(start.x), int(start.y))
            p2 = (int(end.x), int(end.y))
            cv2.line(frame, p1, p2, (0, 0, 0), core + 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (240, 240, 235), core, cv2.LINE_AA)

        for keypoint in visible.values():
            if self._config.biomechanical_skeleton_only and keypoint.name not in BIOMECHANICAL_KEYPOINTS:
                continue
            center = (int(keypoint.x), int(keypoint.y))
            cv2.circle(frame, center, max(4, int(6 * scale)), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, center, max(3, int(5 * scale)), ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, center, max(4, int(6 * scale)), (245, 245, 240), 1, cv2.LINE_AA)

    def _draw_detections(self, frame: Frame, detections: list[Detection]) -> None:
        plate = next((detection for detection in detections if detection.label == "plate"), None)
        barbell = next((detection for detection in detections if detection.label == "barbell"), None)

        if plate is not None:
            center_x, center_y = plate.center
            cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 190, 230), -1, cv2.LINE_AA)

        if plate is not None and barbell is not None:
            center_x, center_y = barbell.center
            radius = max(plate.width, plate.height) / 2.0
            cv2.line(
                frame,
                (int(center_x - radius * 1.18), int(center_y)),
                (int(center_x + radius * 1.18), int(center_y)),
                (0, 205, 205),
                2,
                cv2.LINE_AA,
            )

        if barbell is not None:
            center_x, center_y = barbell.center
            cv2.drawMarker(
                frame,
                (int(center_x), int(center_y)),
                (235, 235, 235),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

    def _draw_bar_anchor(
        self,
        frame: Frame,
        anchor: BarAnchorState | None,
        detections: list[Detection] | None,
        debug_anchor: bool,
    ) -> None:
        if anchor is None or anchor.point is None or anchor.rect is None:
            return
        if anchor.source != "detection" and anchor.missing_frames > self._config.visual_hold_frames:
            return

        point = anchor.point
        rect = anchor.rect
        scale = self._ui_scale(frame)
        fade = 1.0
        if anchor.source != "detection":
            fade = max(0.20, 1.0 - anchor.missing_frames / (self._config.visual_hold_frames + 1))
        plate_color = self._faded_color(PLATE_BOX_COLOR, fade)
        hub_color = self._faded_color(HUB_COLOR, fade)

        display_rect = getattr(anchor, "display_rect", None)
        if display_rect is not None:
            plate_box = (display_rect.x1, display_rect.y1, display_rect.x2, display_rect.y2)
        else:
            # Prefer the trained model's raw plate box when it's confident and close
            # to the tracked anchor. The tracking rect is re-fit by color (a
            # workaround for the old, noisy model) which shrinks and de-centers the
            # box; the trained detector gives a tight, accurate box we can draw 1:1.
            plate_box = self._plate_detection_box_near(detections or [], point)
        if plate_box is not None:
            x1 = max(0, int(plate_box[0]))
            y1 = max(0, int(plate_box[1]))
            x2 = min(frame.shape[1] - 1, int(plate_box[2]))
            y2 = min(frame.shape[0] - 1, int(plate_box[3]))
        else:
            # Enlarge the tracking rect for display only so the brackets hug the plate edge.
            box_scale = self._config.plate_box_display_scale
            center_x = (rect.x1 + rect.x2) / 2.0
            center_y = (rect.y1 + rect.y2) / 2.0
            half_w = rect.width * 0.5 * box_scale
            half_h = rect.height * 0.5 * box_scale
            x1 = int(max(0, center_x - half_w))
            y1 = int(max(0, center_y - half_h))
            x2 = int(min(frame.shape[1] - 1, center_x + half_w))
            y2 = int(min(frame.shape[0] - 1, center_y + half_h))

        if self._config.plate_box_style == "corners":
            self._draw_corner_box(frame, x1, y1, x2, y2, plate_color, scale)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), plate_color, max(1, int(2.4 * scale)), cv2.LINE_AA)
        self._label_pill(frame, "Plate", (x1, y1), plate_color)

        if debug_anchor:
            cv2.line(frame, (int(point.x), 0), (int(point.x), frame.shape[0] - 1), ACCENT_SOFT, 1, cv2.LINE_AA)

        # bar_hub: a small rectangle around the sleeve/hub centre (the model's 2nd class).
        hub_rect = anchor.hub_rect
        if not getattr(anchor, "measurable", False) or hub_rect is None:
            if debug_anchor:
                self._draw_anchor_debug_label(frame, anchor, x1, y1)
            return
        hub_x1 = max(0, int(hub_rect.x1))
        hub_y1 = max(0, int(hub_rect.y1))
        hub_x2 = min(frame.shape[1] - 1, int(hub_rect.x2))
        hub_y2 = min(frame.shape[0] - 1, int(hub_rect.y2))
        cv2.rectangle(frame, (hub_x1, hub_y1), (hub_x2, hub_y2), hub_color, max(1, int(2 * scale)), cv2.LINE_AA)
        # Label the hub to the right of its box so it never sits on top of the "Plate" pill.
        self._label_pill(frame, "Bar", (hub_x2 + max(4, int(6 * scale)), hub_y1), hub_color, prefer_above=False)
        cv2.circle(frame, (int(point.x), int(point.y)), max(3, int(4 * scale)), (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(point.x), int(point.y)), max(6, int(9 * scale)), hub_color, max(1, int(2 * scale)), cv2.LINE_AA)

        if not debug_anchor:
            return

        self._draw_anchor_debug_label(frame, anchor, x1, y1)

    @staticmethod
    def _faded_color(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
        return tuple(int(channel * alpha + 38 * (1.0 - alpha)) for channel in color)

    def _draw_anchor_debug_label(self, frame: Frame, anchor: BarAnchorState, x1: int, y1: int) -> None:
        label = (
            f"{anchor.source} {anchor.confidence:.2f} "
            f"hub:{anchor.hub_confidence:.2f} meas:{int(getattr(anchor, 'measurable', False))} "
            f"miss:{anchor.missing_frames}"
        )
        text_size, _ = cv2.getTextSize(label, UI_FONT, 0.48, 1)
        label_x = x1
        label_y = max(18, y1 - 8)
        cv2.rectangle(
            frame,
            (label_x, label_y - text_size[1] - 6),
            (label_x + text_size[0] + 8, label_y + 4),
            (8, 10, 14),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (label_x + 4, label_y),
            UI_FONT,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _plate_detection_box_near(
        detections: list[Detection],
        point: Point2D,
        *,
        min_confidence: float = 0.45,
        max_distance_ratio: float = 0.6,
    ) -> tuple[float, float, float, float] | None:
        """Return the raw (x1, y1, x2, y2) of the plate detection closest to
        the tracked anchor point, or None when no confident plate is nearby.

        Used to draw the plate box straight from the trained detector (tight,
        accurate) instead of the color-re-fit tracking rect, which shrinks and
        de-centers the box. ``max_distance_ratio`` is relative to the plate's
        own size so the match tolerates the anchor drifting within the disc.
        """
        candidates = [
            det for det in detections
            if det.label.lower() == "plate" and det.confidence >= min_confidence
        ]
        if not candidates:
            return None
        best: Detection | None = None
        best_dist = float("inf")
        for det in candidates:
            cx, cy = det.center
            dist = float(np.hypot(cx - point.x, cy - point.y))
            tol = max(det.width, det.height) * max_distance_ratio
            if dist > tol:
                continue
            if dist < best_dist:
                best_dist = dist
                best = det
        if best is None:
            return None
        return (best.x1, best.y1, best.x2, best.y2)

    @staticmethod
    def _draw_corner_box(
        frame: Frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        """Targeting-style corner brackets (cleaner than a full rectangle)."""
        length = max(12, int(min(x2 - x1, y2 - y1) * 0.22))
        thickness = max(2, int(2.4 * scale))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)

    def _draw_bar_path(self, frame: Frame, bar_path: list[BarPathPoint]) -> None:
        visible_points = [point for point in bar_path if point is not None]
        if len(visible_points) < 2:
            return

        raw_segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for point in bar_path[-150:]:
            if point is None:
                if current:
                    raw_segments.append(current)
                    current = []
                continue
            current.append(point)
        if current:
            raw_segments.append(current)

        scale = self._ui_scale(frame)
        core_w = max(1, int(2 * scale))
        glow_w = max(2, int(4 * scale))
        max_jump = self._config.path_max_jump_pixels

        def _smooth(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            if len(points) < 3:
                return points
            ema_alpha = 0.55
            smoothed = [points[0]]
            for i in range(1, len(points)):
                sx = ema_alpha * points[i][0] + (1.0 - ema_alpha) * smoothed[i - 1][0]
                sy = ema_alpha * points[i][1] + (1.0 - ema_alpha) * smoothed[i - 1][1]
                smoothed.append((sx, sy))
            return smoothed

        def _seg_ok(pts: list[tuple[int, int]], i: int) -> bool:
            horizontal = abs(pts[i][0] - pts[i - 1][0])
            vertical = abs(pts[i][1] - pts[i - 1][1])
            total = float(np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
            # Reject big jumps and mostly side-to-side moves. A bar can drift a
            # little, but a long horizontal trace is almost always a false hub
            # detection in the rack/background rather than a real lift path.
            horizontal_limit = max(12.0 * scale, vertical * 1.15 + 5.0 * scale)
            return total <= max_jump and horizontal <= horizontal_limit

        last_pt: tuple[int, int] | None = None
        for segment in raw_segments:
            if len(segment) < 2:
                if segment:
                    last_pt = (int(segment[-1][0]), int(segment[-1][1]))
                continue
            points = _smooth(segment)
            pts = [(int(x), int(y)) for x, y in points]
            n = len(pts)
            last_pt = pts[-1]

            for index in range(1, n):
                if _seg_ok(pts, index):
                    cv2.line(frame, pts[index - 1], pts[index], TRAJECTORY_GLOW, glow_w, cv2.LINE_AA)
            for index in range(1, n):
                if not _seg_ok(pts, index):
                    continue
                age = index / max(1, n - 1)
                fade = 0.35 + 0.65 * age
                color = tuple(
                    int(TRAJECTORY_GLOW[k] + (TRAJECTORY_COLOR[k] - TRAJECTORY_GLOW[k]) * fade)
                    for k in range(3)
                )
                cv2.line(frame, pts[index - 1], pts[index], color, core_w, cv2.LINE_AA)

        if last_pt is not None:
            cv2.circle(frame, last_pt, max(3, int(4 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, last_pt, max(4, int(6 * scale)), TRAJECTORY_COLOR, max(1, int(1.5 * scale)), cv2.LINE_AA)

    def _draw_bar_velocity_badge(
        self,
        frame: Frame,
        detections: list[Detection],
        sample: KinematicSample | None,
        hub_point: object = None,
        avoid_rects: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        if sample is None:
            return

        barbell = next((detection for detection in detections if detection.label == "barbell"), None)
        if barbell is None and hub_point is None:
            return

        if hub_point is not None:
            center_x, center_y = hub_point.x, hub_point.y
        else:
            center_x, center_y = barbell.center

        scale = self._ui_scale(frame)
        text = f"{sample.smoothed_velocity_mps:+.2f} m/s"
        font_scale = max(0.46, 0.98 * scale)
        thickness = max(2, int(3 * scale))
        pad = max(5, int(10 * scale))
        text_size, _ = cv2.getTextSize(text, UI_FONT, font_scale, thickness)
        origin_x = int(center_x + 18 * scale)
        origin_y = int(center_y - 8 * scale)
        x1 = origin_x - pad
        y1 = origin_y - text_size[1] - pad
        x2 = origin_x + text_size[0] + pad
        y2 = origin_y + pad

        if x2 > frame.shape[1] - 4:
            shift = x2 - (frame.shape[1] - 4)
            x1 -= shift
            x2 -= shift
            origin_x -= shift
        if x1 < 4:
            shift = 4 - x1
            x1 += shift
            x2 += shift
            origin_x += shift
        if y1 < 4:
            shift = 4 - y1
            y1 += shift
            y2 += shift
            origin_y += shift

        # Keep the floating badge clear of the panels. The same value is in the
        # telemetry hero and the chart, so if there's no clean spot we just hide it.
        rects = [rect for rect in (avoid_rects or []) if rect is not None]
        gap = max(6, int(8 * scale))
        height = y2 - y1

        def _hits(top: int) -> bool:
            return any(self._rects_intersect((x1, top, x2, top + height), rect) for rect in rects)

        if _hits(y1):
            candidates = []
            for rect in rects:
                candidates.append(rect[3] + gap)            # just below the panel
                candidates.append(rect[1] - gap - height)   # just above the panel
            placed = next(
                (top for top in candidates if top >= 4 and top + height <= frame.shape[0] - 4 and not _hits(top)),
                None,
            )
            if placed is None:
                return
            origin_y += placed - y1
            y1, y2 = placed, placed + height

        vel_color = color_for_velocity(sample.smoothed_velocity_mps)
        self._rounded_panel(frame, x1, y1, x2, y2, alpha=0.84, radius=max(4, int(9 * scale)), border=vel_color)
        self._text(frame, text, (origin_x, origin_y), font_scale, vel_color, thickness)

    # Velocidades sobre el cuerpo: solo articulaciones grandes para no ensuciar la barra.
    # Muneca y codo suelen taparse con la barra y los brazos, por eso van fuera del cuerpo.
    _ON_BODY_ANCHORS = {"shoulder", "hip", "knee"}

    def _draw_anchor_velocities(self, frame: Frame, anchors: list[AnchorVelocity]) -> None:
        """Dibuja etiquetas compactas de velocidad en hombro, cadera y rodilla."""
        scale = self._ui_scale(frame)
        font_scale = max(0.4, 0.58 * scale)
        thickness = max(1, int(2 * scale))
        for anchor in anchors:
            if anchor.name not in self._ON_BODY_ANCHORS:
                continue
            color = color_for_velocity(anchor.velocity_mps)
            cx, cy = int(anchor.x), int(anchor.y)
            cv2.circle(frame, (cx, cy), max(3, int(4 * scale)), color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(4, int(6 * scale)), (245, 245, 240), 1, cv2.LINE_AA)

            text = f"{anchor.velocity_mps:+.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, UI_FONT, font_scale, thickness)
            ox = cx + int(12 * scale)
            oy = cy - int(6 * scale)
            if ox + text_w + 6 > frame.shape[1]:
                ox = cx - int(12 * scale) - text_w
            self._rounded_panel(
                frame, ox - 5, oy - text_h - 5, ox + text_w + 5, oy + 5,
                alpha=0.66, radius=max(3, int(5 * scale)), border=None,
            )
            self._text(frame, text, (ox, oy), font_scale, color, thickness)

    def _draw_velocity_chart(self, frame: Frame, velocity_history: list[float]) -> int:
        return self._draw_multi_velocity_chart(frame, {"bar": velocity_history}, [], [])

    def _velocity_chart_series(
        self,
        anchor_velocity_history: dict[str, list[float]],
    ) -> list[tuple[str, str, tuple[int, int, int], list[float]]]:
        series: list[tuple[str, str, tuple[int, int, int], list[float]]] = []
        chart_keys = {"bar"} if self._config.velocity_chart_mode == "bar" else None
        for key, label, color in ANCHOR_CHART_SERIES:
            if chart_keys is not None and key not in chart_keys:
                continue
            values = anchor_velocity_history.get(key, [])
            finite_count = sum(1 for value in values if np.isfinite(value))
            if finite_count >= 2:
                series.append((key, label, color, values))
        return series

    def _draw_multi_velocity_chart(
        self,
        frame: Frame,
        anchor_velocity_history: dict[str, list[float]],
        rep_reports: list[RepReport],
        frame_history: list[int],
        max_abs_override: float | None = None,
        video_fps: float = 30.0,
    ) -> int:
        """Draw the multi-anchor velocity chart and return its top y so the rep
        table can stack above it without overlapping."""
        series = self._velocity_chart_series(anchor_velocity_history)

        if not series:
            return frame.shape[0]

        scale = self._ui_scale(frame)
        compact = self._is_compact(frame)
        margin = max(8, int(24 * scale))
        chart_width = frame.shape[1] - margin * 2
        chart_height = max(66, int((104 if compact else 178) * scale))
        x1 = margin
        y2 = frame.shape[0] - margin
        y1 = y2 - chart_height
        x2 = x1 + chart_width
        plot_x1 = x1 + max(12, int(36 * scale))
        plot_x2 = x2 - max(10, int(14 * scale))
        plot_y1 = y1 + max(34, int((42 if compact else 48) * scale))
        plot_y2 = y2 - max(10, int(16 * scale))
        plot_h = max(1, plot_y2 - plot_y1)

        self._rounded_panel(frame, x1, y1, x2, y2, alpha=0.66, radius=max(6, int(12 * scale)), accent=ACCENT)

        newest_frame = frame_history[-1] if frame_history else max(len(values) for _, _, _, values in series) - 1
        window_start_frame = newest_frame - int(round(self._config.velocity_window_seconds * max(1.0, video_fps)))
        if frame_history:
            start_index = next(
                (index for index, frame_index in enumerate(frame_history) if frame_index >= window_start_frame),
                max(0, len(frame_history) - 1),
            )
        else:
            start_index = max(0, max(len(values) for _, _, _, values in series) - int(round(self._config.velocity_window_seconds * max(1.0, video_fps))))
        window_size = max(1, max(len(values) for _, _, _, values in series) - start_index)
        all_values = [
            value
            for _, _, _, values in series
            for value in values[start_index:]
            if np.isfinite(value)
        ]
        if len(all_values) < 2:
            return y1

        # Two-pass analysis can pass a stable global scale so the axis never jumps between
        # frames; live single-pass falls back to the rolling-window maximum.
        if max_abs_override is not None and max_abs_override > 0:
            max_abs = max(0.75, max_abs_override)
        else:
            max_abs = max(0.75, max(abs(value) for value in all_values))
        zero_y = int(plot_y1 + plot_h / 2)
        self._draw_rep_bands(frame, rep_reports, frame_history[start_index:], plot_x1, plot_y1, plot_x2, plot_y2)
        cv2.line(frame, (plot_x1, zero_y), (plot_x2, zero_y), PANEL_BORDER, 1, cv2.LINE_AA)

        if not compact:
            # Ticks follow the real scale, so +max_abs/-max_abs always align with the curve.
            top_y = int(plot_y1 + plot_h * 0.10)
            bottom_y = int(plot_y2 - plot_h * 0.10)
            for frac, prefix in ((0.0, "+"), (0.5, " "), (1.0, "-")):
                tick_y = int(plot_y1 + plot_h * (0.10 + frac * 0.80))
                value_label = "0" if frac == 0.5 else f"{prefix}{max_abs:.2f}"
                cv2.line(frame, (plot_x1 - 8, tick_y), (plot_x1 - 2, tick_y), ACCENT_SOFT, 1, cv2.LINE_AA)
                self._text(frame, value_label, (x1 + 9, tick_y + 5), 0.42, TEXT_DIM, 1, shadow=False)

        self._text(
            frame,
            "VELOCITY",
            (x1 + 12, y1 + int(24 * scale)),
            max(0.42, 0.78 * scale),
            ACCENT,
            max(1, int(2 * scale)),
        )
        self._draw_velocity_legend(frame, series, x1, y1, x2)

        half_plot = plot_h * 0.40  # leaves headroom so peaks don't touch the panel edge
        for key, _, color, values in series:
            visible_values = values[start_index:]
            # Missing measurements are intentionally visible as gaps.  Inventing a
            # curve through an occluded hub makes the graph look delayed or false.
            interpolated = self._interpolate_gaps(visible_values, max_gap=0)
            points: list[tuple[int, int] | None] = []
            for index, value in enumerate(interpolated):
                x = int(plot_x1 + index * (plot_x2 - plot_x1) / max(1, len(interpolated) - 1))
                if np.isfinite(value):
                    y = int(zero_y - (value / max_abs) * half_plot)
                    points.append((x, y))
                else:
                    points.append(None)

            # Anti-aliased polyline through the surviving points.
            # The bar curve is the headline metric: draw it thicker so it
            # stays readable when the hip/shoulder/knee curves cross it.
            line_thickness = max(1, int(2 * scale)) if key == "bar" else 1
            poly = [pt for pt in points if pt is not None]
            if len(poly) >= 2:
                cv2.polylines(frame, [np.array(poly, dtype=np.int32)], False, color, line_thickness, cv2.LINE_AA)

            last_point = next((point for point in reversed(points) if point is not None), None)
            if last_point is not None:
                cv2.circle(frame, last_point, max(2, int(3 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, last_point, max(3, int(4 * scale)), color, max(1, int(1.5 * scale)), cv2.LINE_AA)

        bar_values = anchor_velocity_history.get("bar", [])
        current_bar = next((value for value in reversed(bar_values) if np.isfinite(value)), None)
        if current_bar is not None:
            big_text = f"{current_bar:+.2f}"
            big_scale = max(0.48, 0.98 * scale)
            big_thickness = max(1, int(3 * scale))
            big_size, _ = cv2.getTextSize(big_text, UI_FONT, big_scale, big_thickness)
            unit_w = 0 if compact else int(42 * scale)
            self._text(
                frame,
                big_text,
                (x2 - big_size[0] - unit_w - 12, y1 + int(28 * scale)),
                big_scale,
                color_for_velocity(current_bar),
                big_thickness,
            )
            if not compact:
                self._text(frame, "m/s", (x2 - unit_w - 4, y1 + int(27 * scale)), 0.54, TEXT_DIM, 2)

        return y1

    @staticmethod
    def _interpolate_gaps(values: list[float], max_gap: int = 6) -> list[float]:
        """Linearly interpolate short NaN runs so the velocity curve stays
        continuous when the tracker briefly loses the bar (hands crossing the
        plate, etc.). Longer gaps are kept as NaN (broken line)."""
        result = list(values)
        n = len(result)
        i = 0
        while i < n:
            if np.isfinite(result[i]):
                i += 1
                continue
            start = i
            while i < n and not np.isfinite(result[i]):
                i += 1
            end = i  # first finite value after the gap (or n)
            gap_len = end - start
            if gap_len > max_gap or start == 0 or end >= n:
                continue  # leave as NaN
            left = result[start - 1]
            right = result[end]
            for k in range(start, end):
                frac = (k - (start - 1)) / (end - (start - 1))
                result[k] = left + (right - left) * frac
        return result

    def _draw_velocity_legend(
        self,
        frame: Frame,
        series: list[tuple[str, str, tuple[int, int, int], list[float]]],
        x1: int,
        y1: int,
        x2: int,
    ) -> None:
        scale = self._ui_scale(frame)
        font_scale = max(0.34, 0.48 * scale)
        thickness = 1
        x = x1 + max(96, int(118 * scale))
        y = y1 + int(24 * scale)
        for _, label, color, values in series:
            # Show the live value next to each label so the user can read which
            # joint is moving fastest right now (e.g. "Bar +0.52", "Hip +0.38").
            last_value = next((v for v in reversed(values) if np.isfinite(v)), None)
            value_str = f"{last_value:+.2f}" if last_value is not None else "  -- "
            legend_text = f"{label} {value_str}"
            text_size, _ = cv2.getTextSize(legend_text, UI_FONT, font_scale, thickness)
            item_w = text_size[0] + max(18, int(26 * scale))
            if x + item_w > x2 - max(60, int(90 * scale)):
                break
            cv2.line(frame, (x, y - int(5 * scale)), (x + int(12 * scale), y - int(5 * scale)), color, max(1, int(2 * scale)), cv2.LINE_AA)
            self._text(frame, legend_text, (x + int(16 * scale), y), font_scale, TEXT_DIM, thickness, shadow=False)
            x += item_w + max(8, int(12 * scale))

    @staticmethod
    def _draw_rep_bands(
        frame: Frame,
        rep_reports: list[RepReport],
        frame_history: list[int],
        plot_x1: int,
        plot_y1: int,
        plot_x2: int,
        plot_y2: int,
    ) -> None:
        if not rep_reports or len(frame_history) < 2:
            return

        first_frame = frame_history[0]
        last_frame = frame_history[-1]
        if last_frame <= first_frame:
            return

        overlay = frame.copy()
        for index, rep in enumerate(rep_reports):
            start = max(rep.start_frame, first_frame)
            end = min(rep.end_frame, last_frame)
            if end <= start:
                continue
            x_start = int(plot_x1 + (start - first_frame) / (last_frame - first_frame) * (plot_x2 - plot_x1))
            x_end = int(plot_x1 + (end - first_frame) / (last_frame - first_frame) * (plot_x2 - plot_x1))
            color = (42, 40, 48) if index % 2 == 0 else (34, 38, 42)
            cv2.rectangle(overlay, (x_start, plot_y1), (max(x_start + 1, x_end), plot_y2), color, -1)
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, dst=frame)

    def _rep_table_geometry(
        self,
        frame: Frame,
        rep_reports: list[RepReport],
        bottom_limit: int | None,
        avoid_rect: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int, list[RepReport]] | None:
        """Compute the rep-table rect so it never overlaps the velocity chart (below)
        nor the telemetry panel (above). Returns None when there is no room."""
        if not rep_reports:
            return None
        scale = self._ui_scale(frame)
        table_width = min(int(640 * scale) + 70, frame.shape[1] - max(24, int(40 * scale)))
        row_height = max(26, int(32 * scale))
        header_h = max(54, int(66 * scale))
        tail = max(8, int(12 * scale))
        x2 = frame.shape[1] - max(12, int(20 * scale))
        x1 = x2 - table_width
        gap = max(8, int(12 * scale))
        bottom = (bottom_limit - gap) if bottom_limit is not None else (frame.shape[0] - max(12, int(20 * scale)))

        # Reserve a safe top so the table never grows into the telemetry panel.
        top_limit = max(8, int(20 * scale))
        if avoid_rect is not None and self._rects_overlap_x((x1, 0, x2, 0), avoid_rect):
            top_limit = avoid_rect[3] + gap

        available = bottom - top_limit
        max_rows_fit = (available - header_h - tail) // row_height if row_height > 0 else 0
        if max_rows_fit < 1:
            return None  # not enough vertical room; skip the table rather than overlap
        rows = rep_reports[-min(6, int(max_rows_fit)):]
        table_height = header_h + row_height * len(rows) + tail
        return x1, bottom - table_height, x2, bottom, rows

    def _draw_rep_table(
        self,
        frame: Frame,
        rep_reports: list[RepReport],
        bottom_limit: int | None = None,
        avoid_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        geometry = self._rep_table_geometry(frame, rep_reports, bottom_limit, avoid_rect)
        if geometry is None:
            return
        x1, y1, x2, y2, rows = geometry
        scale = self._ui_scale(frame)
        fastest = max(rep_reports, key=lambda rep: rep.mean_concentric_velocity_mps)
        table_width = x2 - x1
        row_height = max(26, int(32 * scale))

        self._rounded_panel(frame, x1, y1, x2, y2, alpha=0.7, radius=max(6, int(12 * scale)), accent=ACCENT)

        pad = max(10, int(14 * scale))
        self._text(frame, f"FASTEST  R{fastest.rep_index}", (x1 + pad, y1 + int(24 * scale)), max(0.46, 0.68 * scale), ACCENT, max(1, int(2 * scale)))

        columns = [
            ("Rep", x1 + pad),
            ("Con(s)", x1 + pad + int(table_width * 0.17)),
            ("Vel(m/s)", x1 + pad + int(table_width * 0.33)),
            ("Peak(m/s)", x1 + pad + int(table_width * 0.52)),
            ("Ecc(s)", x1 + pad + int(table_width * 0.72)),
            ("Loss", x1 + pad + int(table_width * 0.88)),
        ]
        header_y = y1 + int(50 * scale)
        for label, column_x in columns:
            self._text(frame, label, (column_x, header_y), max(0.34, 0.46 * scale), TEXT_DIM, 1, shadow=False)
        cv2.line(frame, (x1 + pad, header_y + int(8 * scale)), (x2 - pad, header_y + int(8 * scale)), PANEL_BORDER, 1, cv2.LINE_AA)

        y = header_y + row_height
        for rep in rows:
            is_best = rep.rep_index == fastest.rep_index
            color = POS_COLOR if is_best else TEXT
            thickness = max(1, int(2 * scale)) if is_best else 1
            cells = [
                f"R{rep.rep_index}",
                f"{rep.concentric_seconds:.2f}",
                f"{rep.mean_concentric_velocity_mps:.2f}",
                f"{rep.peak_velocity_mps:.2f}",
                f"{rep.eccentric_seconds:.2f}",
                f"{rep.velocity_loss_from_best_percent:.0f}%",
            ]
            for (_, column_x), value in zip(columns, cells):
                cell_color = NEG_COLOR if value.endswith("%") and rep.velocity_loss_warning else color
                self._text(frame, value, (column_x, y), max(0.34, 0.50 * scale), cell_color, thickness)
            if is_best:
                cv2.circle(frame, (x1 + int(pad * 0.55), y - int(5 * scale)), max(2, int(3 * scale)), ACCENT, -1, cv2.LINE_AA)
            y += row_height

    def _draw_telemetry_panel(
        self,
        frame: Frame,
        sample: KinematicSample | None,
        completed_reps: int,
        technique: TechniqueAssessment | None,
        bar_drift_cm: float | None = None,
        load_estimate: LoadEstimate | None = None,
        total_reps: int | None = None,
    ) -> None:
        scale = self._ui_scale(frame)
        compact = self._is_compact(frame)
        x1, y1, x2, y2 = self._telemetry_rect(frame)
        panel_width = x2 - x1
        # With a pre-analysis (two-pass) the total rep count is known up front, so show
        # "done/total" (e.g. 0/5 counting up). Live single-pass keeps the legacy
        # "current attempt / completed-so-far" reading.
        if total_reps is not None:
            rep_numerator = completed_reps
            rep_denominator = total_reps
        else:
            rep_numerator = sample.rep_index if sample is not None else 0
            rep_denominator = completed_reps
        rep_text = f"{rep_numerator}/{rep_denominator}"

        self._rounded_panel(frame, x1, y1, x2, y2, alpha=self._config.panel_alpha, radius=max(6, int(13 * scale)), accent=ACCENT)

        pad = max(10, int(18 * scale))
        text_x = x1 + pad

        # --- Brand wordmark + state pill ---
        title_scale = max(0.5, 0.84 * scale)
        title_th = max(1, int(2 * scale))
        title_y = y1 + max(22, int(31 * scale))
        self._text(frame, "PowerNZ", (text_x, title_y), title_scale, ACCENT, title_th)
        if sample is not None:
            self._draw_state_pill(frame, sample.state, self._state_color(sample.state), x2 - pad, title_y, scale)
        div_y = title_y + max(8, int(13 * scale))
        cv2.line(frame, (text_x, div_y), (x2 - pad, div_y), PANEL_BORDER, 1, cv2.LINE_AA)

        if sample is None:
            self._text(frame, "Esperando seguimiento", (text_x, div_y + max(24, int(34 * scale))),
                       max(0.42, 0.6 * scale), TEXT_DIM, max(1, int(2 * scale)))
            if total_reps == 0:
                self._text(frame, "Sin repeticion valida", (text_x, div_y + max(46, int(64 * scale))),
                           max(0.42, 0.6 * scale), TEXT_DIM, max(1, int(2 * scale)))
            else:
                waiting_reps = rep_text if total_reps is not None else f"{completed_reps}"
                self._text(frame, f"Reps  {waiting_reps}", (text_x, div_y + max(46, int(64 * scale))),
                           max(0.42, 0.6 * scale), TEXT, max(1, int(2 * scale)))
            return

        vel_color = color_for_velocity(sample.smoothed_velocity_mps)

        if compact:
            # Tight stacked layout for very narrow frames.
            y = div_y + max(22, int(30 * scale))
            self._text(frame, f"{sample.smoothed_velocity_mps:+.2f} m/s", (text_x, y),
                       max(0.5, 0.86 * scale), vel_color, max(1, int(2 * scale)))
            y += max(22, int(30 * scale))
            self._text(frame, f"R{rep_text}   ROM {sample.rep_displacement_m:.2f} m",
                       (text_x, y), max(0.4, 0.54 * scale), TEXT, max(1, int(2 * scale)))
            y += max(20, int(26 * scale))
            self._text(frame, f"Calidad  {self._quality_text(technique)}", (text_x, y),
                       max(0.36, 0.48 * scale), TEXT_DIM, 1, shadow=False)
            return

        # --- Hero: bar velocity ---
        cap_y = div_y + max(20, int(27 * scale))
        self._text(frame, "VELOCIDAD DE BARRA", (text_x, cap_y), max(0.34, 0.42 * scale), TEXT_DIM, 1, shadow=False)
        hero = f"{sample.smoothed_velocity_mps:+.2f}"
        hero_scale = max(0.8, 1.5 * scale)
        hero_th = max(2, int(3 * scale))
        hero_y = cap_y + max(28, int(42 * scale))
        self._text(frame, hero, (text_x, hero_y), hero_scale, vel_color, hero_th)
        hsize, _ = cv2.getTextSize(_ascii(hero), UI_FONT, hero_scale, hero_th)
        self._text(frame, "m/s", (text_x + hsize[0] + int(10 * scale), hero_y),
                   max(0.4, 0.56 * scale), TEXT_DIM, max(1, int(2 * scale)))

        # --- Secondary stat blocks ---
        stat_y = hero_y + max(28, int(40 * scale))
        stats = self._telemetry_stats(sample, completed_reps, bar_drift_cm, load_estimate, rep_text)
        col_w = (panel_width - pad * 2) / max(1, len(stats))
        for index, (caption, value) in enumerate(stats):
            self._draw_stat(frame, caption, value, int(text_x + col_w * index), stat_y, scale)

        # --- Quality + view line ---
        q_y = stat_y + max(36, int(50 * scale))
        self._text(frame, "CALIDAD", (text_x, q_y), max(0.34, 0.42 * scale), TEXT_DIM, 1, shadow=False)
        self._text(frame, self._quality_text(technique), (text_x + int(98 * scale), q_y),
                   max(0.4, 0.54 * scale), self._quality_color(technique), max(1, int(2 * scale)))
        view = "-" if technique is None else technique.view.upper()
        vsize, _ = cv2.getTextSize(_ascii(view), UI_FONT, max(0.34, 0.46 * scale), 1)
        self._text(frame, view, (x2 - pad - vsize[0], q_y), max(0.34, 0.46 * scale), TEXT_DIM, 1, shadow=False)

    def _draw_state_pill(
        self,
        frame: Frame,
        state: str,
        color: tuple[int, int, int],
        x_right: int,
        baseline_y: int,
        scale: float,
    ) -> None:
        text = _ascii(state).upper()
        font_scale = max(0.32, 0.42 * scale)
        thickness = max(1, int(1 * scale))
        tsize, _ = cv2.getTextSize(text, UI_FONT, font_scale, thickness)
        dot_r = max(2, int(3 * scale))
        pad_x = max(6, int(9 * scale))
        pad_y = max(3, int(5 * scale))
        chip_w = dot_r * 2 + int(6 * scale) + tsize[0] + pad_x * 2
        chip_h = tsize[1] + pad_y * 2
        cx2 = x_right
        cx1 = x_right - chip_w
        cy2 = baseline_y + pad_y
        cy1 = cy2 - chip_h
        self._rounded_panel(frame, cx1, cy1, cx2, cy2, alpha=0.55, radius=max(4, chip_h // 2), border=None)
        dot_x = cx1 + pad_x + dot_r
        dot_cy = (cy1 + cy2) // 2
        cv2.circle(frame, (dot_x, dot_cy), dot_r, color, -1, cv2.LINE_AA)
        self._text(frame, text, (dot_x + dot_r + int(5 * scale), cy2 - pad_y - 1), font_scale, color, thickness, shadow=False)

    def _draw_stat(
        self,
        frame: Frame,
        caption: str,
        value: str,
        x: int,
        y: int,
        scale: float,
        value_color: tuple[int, int, int] = TEXT,
    ) -> None:
        self._text(frame, caption, (x, y), max(0.32, 0.40 * scale), TEXT_DIM, 1, shadow=False)
        self._text(frame, value, (x, y + max(20, int(26 * scale))), max(0.44, 0.62 * scale),
                   value_color, max(1, int(2 * scale)))

    @staticmethod
    def _quality_color(technique: TechniqueAssessment | None) -> tuple[int, int, int]:
        if technique is None or technique.quality_score <= 0:
            return TEXT_DIM
        if technique.quality_score >= 7.0:
            return POS_COLOR
        if technique.quality_score >= 5.5:
            return ACCENT
        return NEG_COLOR

    @staticmethod
    def _state_color(state: str) -> tuple[int, int, int]:
        if state in {"inicio", "tirón"}:
            return POS_COLOR
        if state == "bajada":
            return NEG_COLOR
        if state == "bloqueo":
            return ACCENT
        return NEUTRAL_COLOR

    @staticmethod
    def _quality_text(technique: TechniqueAssessment | None) -> str:
        if technique is None:
            return "pending"

        if technique.quality_score <= 0:
            return "pending"

        return f"{technique.quality_score:.1f}/10 ({technique.quality_label})"

    def _visible_keypoints(self, pose: PoseResult) -> dict[str, PoseKeypoint]:
        return {
            keypoint.name: keypoint
            for keypoint in pose.keypoints
            if keypoint.visibility >= self._config.keypoint_visibility_threshold
        }

    @staticmethod
    def _ensure_odd(value: int) -> int:
        if value < 3:
            return 3

        return value if value % 2 == 1 else value + 1
