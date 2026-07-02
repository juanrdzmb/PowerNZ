import numpy as np
import pytest

from kinematics import BarMeasurement, reconstruct_bar_kinematics
from track import Point2D


def _measurement(frame: int, y: float | None) -> BarMeasurement:
    return BarMeasurement(
        frame_index=frame,
        time_seconds=frame / 30.0,
        point=None if y is None else Point2D(100.0, y),
        meters_per_pixel=0.001,
        confidence=0.95,
        measurable=y is not None,
    )


def test_reconstructed_velocity_tracks_constant_motion_without_phase_delay() -> None:
    # Pixel y decreases by 10 px/frame: vertical velocity is +0.30 m/s.
    result = reconstruct_bar_kinematics([_measurement(index, 400.0 - index * 10.0) for index in range(20)], fps=30.0)
    middle = result[10]

    assert middle.valid
    assert middle.velocity_mps == pytest.approx(0.30, abs=0.02)


def test_reconstructed_velocity_keeps_occlusion_as_graph_gap() -> None:
    result = reconstruct_bar_kinematics(
        [_measurement(index, None if index in {8, 9} else 400.0 - index * 5.0) for index in range(20)],
        fps=30.0,
    )

    assert result[8].observed is False
    assert result[8].valid is False
    assert result[10].valid is True


def test_reconstruction_rejects_single_frame_false_direction_spike() -> None:
    measurements = [_measurement(index, 400.0 - index * 4.0) for index in range(24)]
    measurements[11] = _measurement(11, 400.0 - 11 * 4.0 + 28.0)

    result = reconstruct_bar_kinematics(measurements, fps=30.0)

    assert result[11].observed is False
    reliable = [sample.velocity_mps for sample in result[7:16] if sample.valid]
    assert reliable
    assert min(reliable) >= -0.02
