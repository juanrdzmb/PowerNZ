from __future__ import annotations

from pathlib import Path

from web.store import JobStore, csrf_is_valid, make_csrf_token


def _create_job(store: JobStore, tmp_path: Path):
    work_dir = tmp_path / "job"
    work_dir.joinpath("upload").mkdir(parents=True)
    return store.create_job(
        exercise="deadlift",
        load_kg=180.0,
        source_filename="video.mp4",
        work_dir=work_dir,
        ttl_seconds=3600,
    )


def test_job_secret_is_required_and_queue_claims_one_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "beta.sqlite3")
    store.initialize()
    job, secret = _create_job(store, tmp_path)

    assert store.get_authorized(job.id, "incorrecto") is None
    assert store.get_authorized(job.id, secret) is not None

    claimed = store.claim_next()

    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == "processing"
    assert store.claim_next() is None


def test_expired_job_returns_its_folder_for_removal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "beta.sqlite3")
    store.initialize()
    job, _ = _create_job(store, tmp_path)

    folders = store.purge_expired_jobs(now=job.expires_at + 1)

    assert folders == [job.work_dir]


def test_csrf_token_is_bound_to_the_job_and_its_cookie_secret() -> None:
    token = make_csrf_token("secret", "job-a")

    assert csrf_is_valid("secret", "job-a", token)
    assert not csrf_is_valid("other", "job-a", token)
    assert not csrf_is_valid("secret", "job-b", token)
