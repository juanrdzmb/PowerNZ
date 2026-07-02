from __future__ import annotations

import numpy as np

from bar_anchor import AnchorRect, BarAnchorState
from load_estimation import LoadEstimate
from metrics import KinematicSample
from pose import PoseKeypoint, PoseResult
from render_overlay import OverlayConfig, OverlayRenderer
from track import Point2D


def test_overlay_renderer_handles_narrow_frames_without_shape_changes() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((720, 310, 3), dtype=np.uint8)
    sample = KinematicSample(
        frame_index=10,
        time_seconds=0.33,
        position_m=0.0,
        velocity_mps=0.0,
        smoothed_velocity_mps=0.0,
        state="reposo",
        rep_index=0,
        rep_displacement_m=0.0,
    )

    output = renderer.render(
        frame=frame,
        sample=sample,
        completed_reps=0,
        velocity_history=[0.0, 0.0, 0.0],
    )

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype


def test_overlay_telemetry_stats_include_load_without_replacing_drift() -> None:
    sample = KinematicSample(
        frame_index=10,
        time_seconds=0.33,
        position_m=0.0,
        velocity_mps=0.0,
        smoothed_velocity_mps=0.0,
        state="tirón",
        rep_index=2,
        rep_displacement_m=0.42,
    )
    load = LoadEstimate(total_kg=120.0, side_weight_kg=50.0, colors=("red",), confidence=0.8)

    stats = OverlayRenderer._telemetry_stats(
        sample=sample,
        completed_reps=1,
        bar_drift_cm=3.4,
        load_estimate=load,
    )

    assert stats == [
        ("REP", "1"),
        ("ROM", "0.42 m"),
        ("DRIFT", "3.4 cm"),
        ("CARGA", "120 kg"),
    ]


def test_overlay_telemetry_stats_omit_load_by_default() -> None:
    sample = KinematicSample(
        frame_index=10,
        time_seconds=0.33,
        position_m=0.0,
        velocity_mps=0.0,
        smoothed_velocity_mps=0.0,
        state="tirón",
        rep_index=2,
        rep_displacement_m=0.42,
    )

    stats = OverlayRenderer._telemetry_stats(
        sample=sample,
        completed_reps=1,
        bar_drift_cm=3.4,
        load_estimate=None,
    )

    assert stats == [
        ("REP", "1"),
        ("ROM", "0.42 m"),
        ("DRIFT", "3.4 cm"),
    ]


def test_overlay_telemetry_stats_explain_body_proxy_instead_of_drift() -> None:
    sample = KinematicSample(
        frame_index=10,
        time_seconds=0.33,
        position_m=0.0,
        velocity_mps=0.0,
        smoothed_velocity_mps=0.0,
        state="tirón",
        rep_index=2,
        rep_displacement_m=0.42,
        tracking_source="body_proxy",
    )

    stats = OverlayRenderer._telemetry_stats(
        sample=sample,
        completed_reps=1,
        bar_drift_cm=None,
        load_estimate=None,
    )

    assert stats == [
        ("REP", "1"),
        ("ROM", "0.42 m"),
        ("FUENTE", "MUÑECAS*"),
    ]


def test_overlay_bar_path_breaks_on_missing_segments() -> None:
    renderer = OverlayRenderer(OverlayConfig(background_dim_alpha=0.0, glow_strength=0.0))
    frame = np.zeros((260, 260, 3), dtype=np.uint8)

    renderer._draw_bar_path(
        frame,
        [(20.0, 30.0), (30.0, 40.0), None, (210.0, 40.0), (220.0, 50.0)],
    )

    assert frame[35, 25].sum() > 0
    assert frame[45, 215].sum() > 0
    assert frame[40, 120].sum() == 0


def test_overlay_bar_path_rejects_mostly_horizontal_false_trace() -> None:
    renderer = OverlayRenderer(OverlayConfig(background_dim_alpha=0.0, glow_strength=0.0))
    frame = np.zeros((260, 260, 3), dtype=np.uint8)

    renderer._draw_bar_path(frame, [(20.0, 100.0), (80.0, 103.0), (140.0, 106.0)])

    # The end marker remains, but the spurious cross-frame horizontal line is hidden.
    assert frame[102, 50].sum() == 0


def test_bar_path_reference_uses_robust_start_axis_and_requires_travel() -> None:
    reference = OverlayRenderer._path_reference_geometry(
        [(100.0, 220.0), (102.0, 200.0), (99.0, 180.0), (101.0, 150.0)]
    )

    assert reference == (100.5, 150.0, 220.0, 220.0)
    assert OverlayRenderer._path_reference_geometry(
        [(100.0, 200.0), (130.0, 202.0), (160.0, 204.0)]
    ) is None


def test_overlay_silhouette_alpha_is_attenuated() -> None:
    renderer = OverlayRenderer(OverlayConfig(silhouette_alpha=0.50, background_dim_alpha=0.0, glow_strength=0.0))
    frame = np.zeros((180, 180, 3), dtype=np.uint8)
    mask = np.full((180, 180), 255, dtype=np.uint8)

    output = renderer.render(frame=frame, subject_mask=mask)

    pixel = output[140, 140]
    assert 40 <= int(pixel[0]) <= 55
    assert max(int(channel) for channel in pixel) - min(int(channel) for channel in pixel) <= 5


def test_overlay_pose_smoothing_rejects_large_keypoint_jump() -> None:
    renderer = OverlayRenderer(OverlayConfig(glow_strength=0.0, background_dim_alpha=0.0))
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    first_pose = PoseResult(
        keypoints=[PoseKeypoint("left_shoulder", 100.0, 100.0, 0.9)],
        backend="yolo",
        detected=True,
    )
    jumped_pose = PoseResult(
        keypoints=[PoseKeypoint("left_shoulder", 380.0, 380.0, 0.9)],
        backend="yolo",
        detected=True,
    )

    renderer.render(frame=frame, pose=first_pose)
    smoothed = renderer._smooth_pose(jumped_pose, frame)

    shoulder = smoothed.keypoints[0]
    assert shoulder.x == 100.0
    assert shoulder.y == 100.0


def test_overlay_draws_clean_plate_corner_box_by_default_without_fake_hub() -> None:
    renderer = OverlayRenderer(OverlayConfig(background_dim_alpha=0.0, glow_strength=0.0))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    anchor = BarAnchorState(
        point=Point2D(160.0, 120.0),
        rect=AnchorRect(100.0, 70.0, 220.0, 190.0),
        confidence=0.8,
        missing_frames=0,
        locked=True,
        source="detection",
        display_rect=AnchorRect(100.0, 70.0, 220.0, 190.0),
        hub_detected=False,
        measurable=False,
    )

    output = frame.copy()
    renderer._draw_bar_anchor(output, anchor, [], False)

    assert output[70, 110].sum() > 0
    assert output[85, 100].sum() > 0
    assert output[70, 160].sum() == 0
    assert output[120, 160].sum() == 0


def test_overlay_label_pills_avoid_reserved_hud_area() -> None:
    renderer = OverlayRenderer(OverlayConfig(background_dim_alpha=0.0, glow_strength=0.0))
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    reserved = (20, 20, 380, 130)

    rect = renderer._label_pill(
        frame,
        "Bar",
        (180, 120),
        (220, 220, 220),
        avoid_rects=[reserved],
    )

    assert not renderer._rects_intersect(rect, reserved)


def test_overlay_velocity_chart_defaults_to_bar_only() -> None:
    renderer = OverlayRenderer()
    history = {
        "bar": [0.1, 0.2, 0.3],
        "hip": [0.2, 0.3, 0.4],
        "knee": [0.3, 0.4, 0.5],
    }

    series = renderer._velocity_chart_series(history)

    assert [key for key, *_ in series] == ["bar"]


def test_overlay_velocity_chart_can_show_multi_series() -> None:
    renderer = OverlayRenderer(OverlayConfig(velocity_chart_mode="multi"))
    history = {
        "bar": [0.1, 0.2, 0.3],
        "hip": [0.2, 0.3, 0.4],
        "knee": [0.3, 0.4, 0.5],
    }

    series = renderer._velocity_chart_series(history)

    assert set(key for key, *_ in series) >= {"bar", "hip", "knee"}


def test_velocity_chart_does_not_connect_across_missing_measurements() -> None:
    segments = OverlayRenderer._split_chart_points(
        [(10, 20), (20, 18), None, (40, 16), (50, 14)]
    )

    assert segments == [[(10, 20), (20, 18)], [(40, 16), (50, 14)]]


def test_overlay_states_no_valid_rep_instead_of_zero_over_zero() -> None:
    renderer = OverlayRenderer()
    drawn: list[str] = []
    renderer._text = lambda _frame, text, *_args, **_kwargs: drawn.append(text)  # type: ignore[method-assign]
    renderer._rounded_panel = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    renderer._draw_telemetry_panel(
        frame=np.zeros((720, 720, 3), dtype=np.uint8),
        sample=None,
        completed_reps=0,
        technique=None,
        total_reps=0,
    )

    assert "Sin repeticion valida" in drawn
    assert not any(text == "Reps  0/0" for text in drawn)


def test_overlay_labels_body_proxy_velocity_as_estimated() -> None:
    renderer = OverlayRenderer()
    drawn: list[str] = []
    renderer._text = lambda _frame, text, *_args, **_kwargs: drawn.append(text)  # type: ignore[method-assign]
    renderer._rounded_panel = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    sample = KinematicSample(
        frame_index=10,
        time_seconds=0.33,
        position_m=0.2,
        velocity_mps=0.25,
        smoothed_velocity_mps=0.25,
        state="tirón",
        rep_index=1,
        rep_displacement_m=0.2,
        tracking_source="body_proxy",
    )

    renderer._draw_telemetry_panel(
        frame=np.zeros((720, 720, 3), dtype=np.uint8),
        sample=sample,
        completed_reps=0,
        technique=None,
        total_reps=1,
    )

    assert "VELOCIDAD CORPORAL*" in drawn
    assert "* MUÑECAS" in drawn


def test_overlay_labels_plate_center_velocity_as_bar_from_disc() -> None:
    renderer = OverlayRenderer()
    drawn: list[str] = []
    renderer._text = lambda _frame, text, *_args, **_kwargs: drawn.append(text)  # type: ignore[method-assign]
    renderer._rounded_panel = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    renderer._draw_multi_velocity_chart(
        frame=np.zeros((720, 720, 3), dtype=np.uint8),
        anchor_velocity_history={"bar": [0.0, 0.20, 0.10]},
        rep_reports=[],
        frame_history=[0, 1, 2],
        metric_source="plate_center",
    )

    assert "VELOCIDAD BARRA · CENTRO DE DISCO" in drawn
    assert "VELOCIDAD CORPORAL*" not in drawn


def test_signal_label_discloses_measurement_source_and_confidence() -> None:
    fused = KinematicSample(
        1, 0.03, 0.1, 0.2, 0.2, "tirón", 1, 0.1,
        hub_confidence=0.80,
        plate_confidence=0.70,
        tracking_source="bar_hub",
    )
    plate = KinematicSample(
        1, 0.03, 0.1, 0.2, 0.2, "tirón", 1, 0.1,
        plate_confidence=0.76,
        tracking_source="plate_center",
    )

    assert OverlayRenderer._signal_label(fused, None) == "- · HUB+DISCO 77%"
    assert OverlayRenderer._signal_label(plate, None) == "- · DISCO 76%"


def test_lateral_overlay_draws_torso_axis_but_frontal_does_not() -> None:
    pose = PoseResult(
        keypoints=[
            PoseKeypoint("left_shoulder", 100.0, 80.0, 0.9),
            PoseKeypoint("right_shoulder", 104.0, 82.0, 0.9),
            PoseKeypoint("left_hip", 130.0, 180.0, 0.9),
            PoseKeypoint("right_hip", 134.0, 182.0, 0.9),
        ],
        backend="yolo",
        detected=True,
    )
    renderer = OverlayRenderer(OverlayConfig(background_dim_alpha=0.0, glow_strength=0.0))
    lateral = np.zeros((260, 260, 3), dtype=np.uint8)
    frontal = lateral.copy()

    renderer._draw_torso_guide(lateral, pose, None, "lateral")
    renderer._draw_torso_guide(frontal, pose, None, "frontal")

    assert lateral.sum() > 0
    assert frontal.sum() == 0
