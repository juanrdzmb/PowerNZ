from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from web.app import create_app
from web.config import WebConfig


ROOT = Path(__file__).resolve().parents[1]


def _client(tmp_path: Path) -> TestClient:
    config = WebConfig(root=ROOT, data_dir=tmp_path / "web_data")
    return TestClient(create_app(config, start_worker=False))


def test_upload_creates_private_job_visible_only_to_its_browser(tmp_path: Path) -> None:
    with _client(tmp_path) as owner:
        response = owner.post(
            "/jobs",
            files={"video": ("levantamiento.mp4", b"small-video", "video/mp4")},
            data={"exercise": "deadlift", "load_kg": "180", "privacy_accepted": "yes"},
            follow_redirects=False,
        )

        assert response.status_code == 303
        location = response.headers["location"]
        assert location.startswith("/job/")
        assert owner.get(location).status_code == 200
        assert owner.get(location.replace("/job/", "/api/jobs/")).json()["status"] == "queued"

        with _client(tmp_path) as stranger:
            assert stranger.get(location).status_code == 404


def test_upload_requires_explicit_privacy_consent(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/jobs",
            files={"video": ("levantamiento.mp4", b"small-video", "video/mp4")},
            data={"exercise": "squat"},
        )

    assert response.status_code == 422
