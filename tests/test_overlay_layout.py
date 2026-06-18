"""Regression tests: overlay panels must not overlap on any aspect ratio."""

from __future__ import annotations

import numpy as np

from metrics import KinematicSample
from render_overlay import OverlayRenderer, _ascii
from reporting import RepReport


def _rep(index: int) -> RepReport:
    return RepReport(
        rep_index=index,
        start_frame=index * 30,
        lockout_frame=index * 30 + 15,
        end_frame=index * 30 + 30,
        duration_seconds=1.0,
        concentric_seconds=0.5,
        eccentric_seconds=0.5,
        rom_m=0.4,
        mean_concentric_velocity_mps=0.8,
        peak_velocity_mps=1.2,
        velocity_loss_from_best_percent=5.0,
        velocity_loss_from_previous_percent=2.0,
        velocity_loss_warning=False,
    )


def _intersects(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


FRAME_SIZES = [
    (720, 1280),   # landscape 1280x720
    (1280, 720),   # vertical 9:16 (downscaled preview)
    (1920, 1080),  # vertical full-res
    (405, 720),    # short landscape -> classic overlap case
    (720, 310),    # narrow/compact
]


def test_rep_table_never_overlaps_telemetry():
    renderer = OverlayRenderer()
    reps = [_rep(i) for i in range(1, 9)]
    for height, width in FRAME_SIZES:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        tele = renderer._telemetry_rect(frame)
        chart_top = int(height * 0.80)  # emulate the bottom velocity chart
        for bottom_limit in (None, chart_top):
            geom = renderer._rep_table_geometry(frame, reps, bottom_limit, tele)
            if geom is None:
                continue  # skipped because no room -> acceptable, no overlap
            x1, y1, x2, y2, _rows = geom
            assert not _intersects((x1, y1, x2, y2), tele), (
                f"rep table {(x1, y1, x2, y2)} overlaps telemetry {tele} at {width}x{height}"
            )
            assert y1 >= 0 and y2 <= height


def test_rep_table_shows_on_tall_frame():
    renderer = OverlayRenderer()
    reps = [_rep(i) for i in range(1, 5)]
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    tele = renderer._telemetry_rect(frame)
    geom = renderer._rep_table_geometry(frame, reps, int(1280 * 0.80), tele)
    assert geom is not None  # plenty of room on a tall 9:16 frame


def test_full_render_runs_on_vertical_frame_with_reps():
    renderer = OverlayRenderer()
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    out = renderer.render(
        frame=frame,
        rep_reports=[_rep(i) for i in range(1, 7)],
        velocity_history=[0.1, 0.2, 0.3, 0.2],
    )
    assert out.shape == frame.shape


def test_ascii_transliteration_removes_accents():
    # OpenCV's Hershey font has no accented glyphs (would render "tir??n").
    assert _ascii("tirón") == "tiron"
    assert _ascii("preparación · técnica") == "preparacion - tecnica"


def test_render_with_accented_state_does_not_crash():
    renderer = OverlayRenderer()
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    sample = KinematicSample(
        frame_index=5, time_seconds=0.2, position_m=0.2, velocity_mps=0.5,
        smoothed_velocity_mps=0.5, state="tirón", rep_index=1, rep_displacement_m=0.2,
    )
    out = renderer.render(frame=frame, sample=sample, completed_reps=1)
    assert out.shape == frame.shape
