from analysis_profiles import get_analysis_profile


def test_balanced_profile_preserves_720_overlay_budget() -> None:
    profile = get_analysis_profile("balanced")
    assert profile.inference_max_side == 960
    assert profile.segmentation_stride == 2
    assert profile.velocity_window_seconds == 4.5


def test_fast_profile_is_lighter_than_balanced() -> None:
    assert get_analysis_profile("fast").inference_max_side < get_analysis_profile("balanced").inference_max_side
