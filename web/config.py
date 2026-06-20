from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WebConfig:
    """Small, deliberately conservative runtime configuration for the beta."""

    root: Path
    data_dir: Path
    max_upload_bytes: int = 250 * 1024 * 1024
    max_duration_seconds: float = 60.0
    max_width: int = 1920
    max_height: int = 1920
    job_ttl_seconds: int = 24 * 60 * 60
    feedback_ttl_seconds: int = 30 * 24 * 60 * 60
    submissions_per_hour: int = 2
    secure_cookies: bool = False
    analysis_profile: str = "balanced"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "powernz_beta.sqlite3"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @classmethod
    def from_environment(cls) -> "WebConfig":
        data_dir = Path(os.environ.get("POWERNZ_WEB_DATA_DIR", ROOT / "web_data"))
        return cls(
            root=ROOT,
            data_dir=data_dir,
            max_upload_bytes=int(os.environ.get("POWERNZ_MAX_UPLOAD_BYTES", 250 * 1024 * 1024)),
            max_duration_seconds=float(os.environ.get("POWERNZ_MAX_DURATION_SECONDS", 60)),
            secure_cookies=os.environ.get("POWERNZ_SECURE_COOKIES", "0") == "1",
            analysis_profile=os.environ.get("POWERNZ_WEB_PROFILE", "balanced"),
        )
