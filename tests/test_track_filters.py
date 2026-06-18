from __future__ import annotations

import pytest

from track import LowPassFilter, OneEuroFilter


def test_low_pass_filter_initializes_with_first_value() -> None:
    filter_ = LowPassFilter()

    first = filter_.apply(1.0, alpha=0.5)
    second = filter_.apply(2.0, alpha=0.5)

    assert first == 1.0
    assert 1.0 < second < 2.0


def test_low_pass_filter_blends_with_previous() -> None:
    filter_ = LowPassFilter()
    filter_.apply(0.0, alpha=1.0)

    blended = filter_.apply(10.0, alpha=0.5)

    assert blended == 5.0


def test_one_euro_filter_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(frequency_hz=0.0)


def test_one_euro_filter_smooths_constant_input() -> None:
    filter_ = OneEuroFilter(frequency_hz=30.0, min_cutoff=1.0, beta=0.0)

    outputs = [filter_.apply(1.0) for _ in range(5)]

    for value in outputs[1:]:
        assert abs(value - 1.0) < 1e-6


def test_one_euro_filter_tracks_ramp_input() -> None:
    filter_ = OneEuroFilter(frequency_hz=30.0, min_cutoff=10.0, beta=0.0, derivative_cutoff=10.0)

    outputs = [filter_.apply(index * 0.1) for index in range(20)]

    assert outputs[-1] == pytest.approx(1.9, abs=0.2)
