from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = frozenset({"completed", "failed", "deleted"})


@dataclass(frozen=True)
class Job:
    id: str
    status: str
    progress: int
    stage: str
    exercise: str
    load_kg: float | None
    source_filename: str
    work_dir: Path
    source_path: Path
    normalized_path: Path
    result_path: Path
    report_json_path: Path
    report_csv_path: Path
    created_at: int
    expires_at: int
    error_message: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class JobStore:
    """SQLite-backed queue metadata. Video files remain in per-job folders."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    secret_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    exercise TEXT NOT NULL,
                    load_kg REAL,
                    source_filename TEXT NOT NULL,
                    work_dir TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    normalized_path TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    report_json_path TEXT NOT NULL,
                    report_csv_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS jobs_expiry_idx ON jobs(expires_at);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS feedback_expiry_idx ON feedback(expires_at);
                """
            )

    def create_job(
        self,
        *,
        exercise: str,
        load_kg: float | None,
        source_filename: str,
        work_dir: Path,
        ttl_seconds: int,
    ) -> tuple[Job, str]:
        job_id = uuid.uuid4().hex
        secret = secrets.token_urlsafe(32)
        now = int(time.time())
        source_path = work_dir / "upload" / source_filename
        normalized_path = work_dir / "work" / "normalizado.mp4"
        result_path = work_dir / "result" / "powernz_analizado.mp4"
        report_json_path = work_dir / "result" / "informe.json"
        report_csv_path = work_dir / "result" / "repeticiones.csv"
        values = {
            "id": job_id,
            "secret_digest": self._secret_digest(secret),
            "status": "queued",
            "progress": 0,
            "stage": "En cola",
            "exercise": exercise,
            "load_kg": load_kg,
            "source_filename": source_filename,
            "work_dir": str(work_dir),
            "source_path": str(source_path),
            "normalized_path": str(normalized_path),
            "result_path": str(result_path),
            "report_json_path": str(report_json_path),
            "report_csv_path": str(report_csv_path),
            "created_at": now,
            "expires_at": now + ttl_seconds,
            "error_message": None,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, secret_digest, status, progress, stage, exercise, load_kg, source_filename,
                    work_dir, source_path, normalized_path, result_path, report_json_path, report_csv_path,
                    created_at, expires_at, error_message
                ) VALUES (
                    :id, :secret_digest, :status, :progress, :stage, :exercise, :load_kg, :source_filename,
                    :work_dir, :source_path, :normalized_path, :result_path, :report_json_path, :report_csv_path,
                    :created_at, :expires_at, :error_message
                )
                """,
                values,
            )
        return self._row_to_job(values), secret

    def get_authorized(self, job_id: str, secret: str | None) -> Job | None:
        if not secret:
            return None
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or not secrets.compare_digest(str(row["secret_digest"]), self._secret_digest(secret)):
            return None
        return self._row_to_job(row)

    def claim_next(self) -> Job | None:
        """Atomically take one queued job. There is intentionally only one worker."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE jobs SET status = 'processing', progress = 2, stage = 'Preparando el vídeo' WHERE id = ?",
                (row["id"],),
            )
            claimed = dict(row)
            claimed.update(status="processing", progress=2, stage="Preparando el vídeo")
            return self._row_to_job(claimed)

    def update_progress(self, job_id: str, progress: int, stage: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET progress = ?, stage = ? WHERE id = ? AND status = 'processing'",
                (max(0, min(100, progress)), stage, job_id),
            )

    def complete(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', progress = 100, stage = 'Tu análisis está listo', error_message = NULL
                WHERE id = ?
                """,
                (job_id,),
            )

    def fail(self, job_id: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', stage = 'No se pudo completar el análisis', error_message = ?
                WHERE id = ?
                """,
                (message, job_id),
            )

    def delete_if_not_processing(self, job_id: str) -> Path | None:
        with self._connect() as connection:
            row = connection.execute("SELECT work_dir, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] == "processing":
                return None
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return Path(str(row["work_dir"]))

    def purge_expired_jobs(self, now: int | None = None) -> list[Path]:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT work_dir FROM jobs WHERE expires_at <= ? AND status != 'processing'", (now,)
            ).fetchall()
            connection.execute(
                "DELETE FROM jobs WHERE expires_at <= ? AND status != 'processing'", (now,)
            )
        return [Path(str(row["work_dir"])) for row in rows]

    def add_feedback(
        self,
        *,
        job: Job,
        category: str,
        comment: str,
        metadata_json: str,
        ttl_seconds: int,
    ) -> None:
        now = int(time.time())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback (job_id, category, comment, metadata_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job.id, category, comment, metadata_json, now, now + ttl_seconds),
            )

    def purge_expired_feedback(self, now: int | None = None) -> None:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("DELETE FROM feedback WHERE expires_at <= ?", (now,))

    @staticmethod
    def _secret_digest(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @staticmethod
    def _row_to_job(row: sqlite3.Row | dict[str, Any]) -> Job:
        return Job(
            id=str(row["id"]),
            status=str(row["status"]),
            progress=int(row["progress"]),
            stage=str(row["stage"]),
            exercise=str(row["exercise"]),
            load_kg=float(row["load_kg"]) if row["load_kg"] is not None else None,
            source_filename=str(row["source_filename"]),
            work_dir=Path(str(row["work_dir"])),
            source_path=Path(str(row["source_path"])),
            normalized_path=Path(str(row["normalized_path"])),
            result_path=Path(str(row["result_path"])),
            report_json_path=Path(str(row["report_json_path"])),
            report_csv_path=Path(str(row["report_csv_path"])),
            created_at=int(row["created_at"]),
            expires_at=int(row["expires_at"]),
            error_message=str(row["error_message"]) if row["error_message"] else None,
        )


def make_csrf_token(secret: str, job_id: str) -> str:
    return hmac.new(secret.encode("utf-8"), job_id.encode("utf-8"), hashlib.sha256).hexdigest()


def csrf_is_valid(secret: str, job_id: str, token: str | None) -> bool:
    return bool(token) and secrets.compare_digest(make_csrf_token(secret, job_id), token)
