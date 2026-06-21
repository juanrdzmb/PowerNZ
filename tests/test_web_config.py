from pathlib import Path

from web.config import WebConfig


def test_web_defaults_preserve_quality_pipeline() -> None:
    config = WebConfig(root=Path("."), data_dir=Path("./web-data"))

    assert config.analysis_profile == "balanced"
    assert config.pose_backend == "auto"
    assert config.segmentation_backend == "auto"
    assert config.normalize_max_dimension == 1920
