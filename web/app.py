from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.config import WebConfig
from web.store import Job, JobStore, csrf_is_valid, make_csrf_token
from web.worker import ALLOWED_EXTENSIONS, JobWorker


EXERCISES = {
    "deadlift": "Peso muerto",
    "squat": "Sentadilla",
    "bench": "Press banca",
}
COOKIE_PREFIX = "pnz_job_"
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
FEEDBACK_CATEGORIES = {
    "conteo": "El conteo de repeticiones no cuadra",
    "deteccion": "La barra o el atleta se detectaron mal",
    "video": "El vídeo final tiene un problema visual",
    "otro": "Otro problema",
}


class SubmissionLimiter:
    """Memory-only, best-effort abuse protection for this very small beta."""

    def __init__(self, per_hour: int) -> None:
        self.per_hour = per_hour
        self._submissions: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, ip_address: str) -> bool:
        now = time.monotonic()
        attempts = self._submissions[ip_address]
        while attempts and now - attempts[0] >= 3600:
            attempts.popleft()
        if len(attempts) >= self.per_hour:
            return False
        attempts.append(now)
        return True


def create_app(config: WebConfig | None = None, *, start_worker: bool = True) -> FastAPI:
    config = config or WebConfig.from_environment()
    store = JobStore(config.database_path)
    limiter = SubmissionLimiter(config.submissions_per_hour)
    templates = Jinja2Templates(directory=str(config.root / "web" / "templates"))
    worker = JobWorker(store, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config.jobs_dir.mkdir(parents=True, exist_ok=True)
        store.initialize()
        for work_dir in store.purge_expired_jobs():
            shutil.rmtree(work_dir, ignore_errors=True)
        store.purge_expired_feedback()
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="PowerNZ Beta", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.config = config
    app.state.store = store
    app.state.worker = worker
    app.mount("/static", StaticFiles(directory=str(config.root / "web" / "static")), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response: Response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers[
            "Content-Security-Policy"
        ] = "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        return response

    @app.get("/")
    async def home(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "exercises": EXERCISES,
                "max_upload_mb": round(config.max_upload_bytes / 1024 / 1024),
                "max_duration_seconds": int(config.max_duration_seconds),
                "recent_jobs": _recent_owned_jobs(request, store),
                "privacy": _privacy_context(config),
                "job_status_labels": {
                    "queued": "En cola",
                    "processing": "Procesando",
                    "completed": "Listo para descargar",
                    "failed": "Necesita otro intento",
                },
            },
        )

    @app.post("/jobs")
    async def create_job(
        request: Request,
        video: UploadFile = File(...),
        exercise: str = Form("deadlift"),
        load_kg: str = Form(""),
        privacy_accepted: str | None = Form(None),
    ) -> Response:
        if privacy_accepted != "yes":
            raise HTTPException(status_code=422, detail="Debes aceptar el aviso de privacidad para continuar.")
        if exercise not in EXERCISES:
            raise HTTPException(status_code=422, detail="Elige un ejercicio válido.")
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            raise HTTPException(status_code=429, detail="Esta beta permite dos análisis por hora desde la misma conexión.")
        safe_name = _safe_upload_name(video.filename)
        if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=422, detail="Por ahora aceptamos vídeos MP4 o MOV.")
        parsed_load = _parse_load(load_kg)

        work_dir = config.jobs_dir / _new_work_dir_name()
        work_dir.joinpath("upload").mkdir(parents=True, exist_ok=False)
        try:
            source_path = work_dir / "upload" / safe_name
            uploaded_size = await _save_upload(video, source_path, config.max_upload_bytes)
            if uploaded_size == 0:
                raise HTTPException(status_code=422, detail="El archivo está vacío.")
            job, secret = store.create_job(
                exercise=exercise,
                load_kg=parsed_load,
                source_filename=safe_name,
                work_dir=work_dir,
                ttl_seconds=config.job_ttl_seconds,
                privacy_notice_version=config.privacy_notice_version,
            )
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        finally:
            await video.close()

        response = RedirectResponse(url=f"/job/{job.id}", status_code=303)
        response.set_cookie(
            key=_cookie_name(job.id),
            value=secret,
            max_age=config.job_ttl_seconds,
            httponly=True,
            secure=config.secure_cookies,
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/job/{job_id}")
    async def job_page(job_id: str, request: Request, feedback: str | None = None) -> Response:
        job, secret = _authorized_job(request, store, job_id)
        return templates.TemplateResponse(
            request,
            "job.html",
            {
                "job": job,
                "exercise_label": EXERCISES[job.exercise],
                "expires_text": _format_expiry(job.expires_at),
                "csrf_token": make_csrf_token(secret, job.id),
                "feedback_categories": FEEDBACK_CATEGORIES,
                "feedback_sent": feedback == "ok",
                "privacy": _privacy_context(config),
            },
        )

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str, request: Request) -> JSONResponse:
        job, _ = _authorized_job(request, store, job_id)
        return JSONResponse(_job_public_data(job))

    @app.get("/job/{job_id}/download")
    async def download_job(job_id: str, request: Request) -> FileResponse:
        job, _ = _authorized_job(request, store, job_id)
        if job.status != "completed" or not job.result_path.exists():
            raise HTTPException(status_code=409, detail="Tu vídeo todavía no está listo.")
        return FileResponse(
            job.result_path,
            media_type="video/mp4",
            filename=f"powernz_{job.exercise}_{job.id[:8]}.mp4",
        )

    @app.post("/job/{job_id}/delete")
    async def delete_job(job_id: str, request: Request, csrf_token: str = Form(...)) -> Response:
        job, secret = _authorized_job(request, store, job_id)
        _require_csrf(secret, job.id, csrf_token)
        work_dir = store.delete_if_not_processing(job.id)
        if work_dir is None:
            raise HTTPException(status_code=409, detail="No se puede borrar un análisis mientras está procesándose.")
        shutil.rmtree(work_dir, ignore_errors=True)
        response = RedirectResponse(url="/", status_code=303)
        response.delete_cookie(_cookie_name(job.id), path="/")
        return response

    @app.post("/job/{job_id}/feedback")
    async def send_feedback(
        job_id: str,
        request: Request,
        csrf_token: str = Form(...),
        category: str = Form(...),
        comment: str = Form(...),
    ) -> Response:
        job, secret = _authorized_job(request, store, job_id)
        _require_csrf(secret, job.id, csrf_token)
        if category not in FEEDBACK_CATEGORIES:
            raise HTTPException(status_code=422, detail="Elige una categoría válida.")
        cleaned_comment = " ".join(comment.split())
        if not 8 <= len(cleaned_comment) <= 1000:
            raise HTTPException(status_code=422, detail="Describe el problema en entre 8 y 1000 caracteres.")
        metadata = {
            "exercise": job.exercise,
            "load_kg": job.load_kg,
            "status": job.status,
            "progress": job.progress,
        }
        store.add_feedback(
            job=job,
            category=category,
            comment=cleaned_comment,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
            ttl_seconds=config.feedback_ttl_seconds,
        )
        return RedirectResponse(url=f"/job/{job.id}?feedback=ok", status_code=303)

    @app.get("/privacy")
    async def privacy_notice(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "privacy.html",
            {"privacy": _privacy_context(config)},
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "powernz-beta"})

    return app


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> int:
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                output.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="El vídeo supera el límite de esta beta.")
            output.write(chunk)
    return total


def _authorized_job(request: Request, store: JobStore, job_id: str) -> tuple[Job, str]:
    secret = request.cookies.get(_cookie_name(job_id))
    job = store.get_authorized(job_id, secret)
    if job is None:
        raise HTTPException(status_code=404, detail="No encontramos este análisis en este navegador.")
    return job, secret  # type: ignore[return-value]


def _recent_owned_jobs(request: Request, store: JobStore) -> list[Job]:
    """Recover jobs from the browser's private cookies after a tab was closed.

    Each cookie includes an unguessable secret and is verified by the database;
    an arbitrary job ID in a browser cookie can never expose another person's job.
    """
    jobs: list[Job] = []
    for cookie_name, secret in request.cookies.items():
        if not cookie_name.startswith(COOKIE_PREFIX):
            continue
        job_id = cookie_name.removeprefix(COOKIE_PREFIX)
        if not JOB_ID_RE.fullmatch(job_id):
            continue
        job = store.get_authorized(job_id, secret)
        if job is not None:
            jobs.append(job)
    return sorted(jobs, key=lambda job: job.created_at, reverse=True)[:3]


def _require_csrf(secret: str, job_id: str, csrf_token: str) -> None:
    if not csrf_is_valid(secret, job_id, csrf_token):
        raise HTTPException(status_code=403, detail="No hemos podido verificar esta acción.")


def _safe_upload_name(filename: str | None) -> str:
    name = Path(filename or "video.mp4").name
    name = SAFE_FILENAME_RE.sub("-", name).strip(".-")
    return name[:100] or "video.mp4"


def _new_work_dir_name() -> str:
    return f"job_{int(time.time())}_{os_random_suffix()}"


def os_random_suffix() -> str:
    import secrets

    return secrets.token_hex(8)


def _parse_load(value: str) -> float | None:
    candidate = value.strip().replace(",", ".")
    if not candidate:
        return None
    try:
        parsed = float(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="El peso debe ser un número válido.") from exc
    if not 0 < parsed <= 1000:
        raise HTTPException(status_code=422, detail="El peso debe estar entre 0 y 1000 kg.")
    return parsed


def _cookie_name(job_id: str) -> str:
    return f"{COOKIE_PREFIX}{job_id}"


def _job_public_data(job: Job) -> dict[str, object]:
    return {
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "error_message": job.error_message,
        "terminal": job.is_terminal,
    }


def _format_expiry(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%d/%m/%Y a las %H:%M")


def _privacy_context(config: WebConfig) -> dict[str, str]:
    return {
        "controller": config.privacy_controller,
        "contact": config.privacy_contact,
        "notice_version": config.privacy_notice_version,
        "updated": "21 de junio de 2026",
    }


app = create_app()
