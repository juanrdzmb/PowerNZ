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
    # Beta speed tuning for a CPU-only VM. Pose stays on "auto": it keeps the YOLO
    # athlete tracker and only adds MediaPipe when it loads, so it never crashes if
    # MediaPipe is unavailable in the container. The big saving is the silhouette:
    # "pose-hull" reuses the pose with no extra model instead of the YOLO segmenter.
    pose_backend: str = "auto"
    segmentation_backend: str = "pose-hull"
    # Your trained athlete-segmentation weights. Only loaded when the segmentation
    # backend actually uses a model (auto / yolo-seg); ignored by the fast pose-hull
    # default. This makes the "quality" silhouette use your model, not a generic one.
    segmentation_model: str = "models/powerai_athlete_seg.pt"
    normalize_max_dimension: int = 1280
    privacy_controller: str = "PowerNZ Beta"
    privacy_contact: str = "la persona que te compartió este enlace"
    privacy_notice_version: str = "2026-06-21"

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
            pose_backend=os.environ.get("POWERNZ_WEB_POSE_BACKEND", "auto"),
            segmentation_backend=os.environ.get("POWERNZ_WEB_SEGMENTATION", "pose-hull"),
            segmentation_model=os.environ.get("POWERNZ_WEB_SEGMENTATION_MODEL", "models/powerai_athlete_seg.pt"),
            normalize_max_dimension=int(os.environ.get("POWERNZ_WEB_NORMALIZE_MAX", 1280)),
            privacy_controller=os.environ.get("POWERNZ_PRIVACY_CONTROLLER", "PowerNZ Beta"),
            privacy_contact=os.environ.get(
                "POWERNZ_PRIVACY_CONTACT", "la persona que te compartió este enlace"
            ),
            privacy_notice_version=os.environ.get("POWERNZ_PRIVACY_NOTICE_VERSION", "2026-06-21"),
        )
