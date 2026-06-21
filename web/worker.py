from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from io_video import read_video_metadata
from video_export import convert_to_mobile_mp4
from web.config import WebConfig
from web.store import Job, JobStore


PROGRESS_RE = re.compile(r"^PROGRESS\s+(\w+)\s+(\d+)\s+(\d+)")
ALLOWED_EXTENSIONS = frozenset({".mp4", ".mov"})


class JobWorker:
    """Runs one analysis at a time so a small VM remains predictable."""

    def __init__(self, store: JobStore, config: WebConfig) -> None:
        self.store = store
        self.config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_purge = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="powernz-web-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._purge_if_due()
            job = self.store.claim_next()
            if job is None:
                self._stop.wait(0.75)
                continue
            self._process(job)

    def _purge_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_purge < 300:
            return
        for work_dir in self.store.purge_expired_jobs():
            shutil.rmtree(work_dir, ignore_errors=True)
        self.store.purge_expired_feedback()
        self._last_purge = now

    def _process(self, job: Job) -> None:
        try:
            self._validate_source(job)
            self.store.update_progress(job.id, 5, "Preparando un formato compatible")
            job.normalized_path.parent.mkdir(parents=True, exist_ok=True)
            convert_to_mobile_mp4(
                job.source_path,
                job.normalized_path,
                max_dimension=self.config.normalize_max_dimension,
            )
            self._validate_normalized(job)
            self._ensure_models_are_ready()
            self._run_analysis(job)
            if not job.result_path.exists() or job.result_path.stat().st_size == 0:
                raise RuntimeError("El análisis no generó un vídeo descargable.")
            self.store.complete(job.id)
        except ValueError as exc:
            self.store.fail(job.id, str(exc))
        except Exception:  # noqa: BLE001 - technical details stay out of the public page
            self.store.fail(
                job.id,
                "No pudimos analizar este vídeo. Prueba con un clip corto en MP4 o MOV, grabado con buena luz.",
            )

    def _validate_source(self, job: Job) -> None:
        if job.source_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Por ahora solo aceptamos vídeos MP4 o MOV.")
        if not job.source_path.exists() or job.source_path.stat().st_size == 0:
            raise ValueError("El archivo subido está vacío o no se pudo guardar.")
        if job.source_path.stat().st_size > self.config.max_upload_bytes:
            raise ValueError("El vídeo supera el límite de 250 MB de esta beta.")
        # A short 4K clip is acceptable if it fits the upload limit: it is
        # converted to our 1080p-compatible working format before analysis.
        # Keeping the size/duration checks here still protects the small beta
        # VM from huge uploads.
        self._validate_metadata(job.source_path, enforce_dimensions=False)

    def _validate_normalized(self, job: Job) -> None:
        self._validate_metadata(job.normalized_path)

    def _validate_metadata(self, video_path: Path, *, enforce_dimensions: bool = True) -> None:
        try:
            metadata = read_video_metadata(video_path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("No hemos podido leer el vídeo. Prueba a exportarlo como MP4.") from exc
        duration = metadata.frame_count / max(1.0, metadata.fps)
        if metadata.width <= 0 or metadata.height <= 0 or metadata.frame_count <= 1:
            raise ValueError("El archivo no parece contener un vídeo válido.")
        if duration > self.config.max_duration_seconds:
            raise ValueError("Esta beta acepta vídeos de hasta 60 segundos.")
        if enforce_dimensions and (
            metadata.width > self.config.max_width or metadata.height > self.config.max_height
        ):
            raise ValueError("El vídeo supera 1080p. Expórtalo a 1080p o menos e inténtalo de nuevo.")

    def _ensure_models_are_ready(self) -> None:
        result = subprocess.run(
            [sys.executable, "model_downloader.py", "--check"],
            cwd=self.config.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Los modelos de PowerNZ todavía no están preparados.")

    def _run_analysis(self, job: Job) -> None:
        job.result_path.parent.mkdir(parents=True, exist_ok=True)
        job_temp = job.work_dir / "tmp"
        job_temp.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "main.py",
            "--input",
            str(job.normalized_path),
            "--output",
            str(job.result_path),
            "--exercise",
            job.exercise,
            "--profile",
            self.config.analysis_profile,
            "--pose-backend",
            self.config.pose_backend,
            "--segmentation-backend",
            self.config.segmentation_backend,
            "--calibration-mode",
            "auto",
            "--output-format",
            "portrait-720",
            "--report-json",
            str(job.report_json_path),
            "--report-csv",
            str(job.report_csv_path),
            "--log-level",
            "WARNING",
        ]
        # Use the trained athlete weights when the chosen backend loads a model.
        seg_model = self.config.root / self.config.segmentation_model
        if seg_model.exists():
            command.extend(["--segmentation-model", str(seg_model)])
        if job.load_kg is not None:
            command.extend(["--load-kg", str(job.load_kg)])

        environment = os.environ.copy()
        environment.update({"TMPDIR": str(job_temp), "TEMP": str(job_temp), "TMP": str(job_temp)})
        # Inside a container the BLAS/OpenMP backends often misread the available
        # cores and under- or over-subscribe threads. Pin them to the VM's vCPUs so
        # the per-frame model inference actually uses the machine we are paying for.
        threads = str(max(1, os.cpu_count() or 1))
        for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment.setdefault(thread_var, threads)
        self.store.update_progress(job.id, 8, "Analizando técnica y trayectoria")
        process = subprocess.Popen(
            command,
            cwd=self.config.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self._record_progress(job.id, line)
        if process.wait() != 0:
            raise RuntimeError("PowerNZ no pudo terminar el análisis.")

    def _record_progress(self, job_id: str, line: str) -> None:
        match = PROGRESS_RE.match(line.strip())
        if not match:
            return
        phase, current, total = match.groups()
        fraction = int(current) / max(1, int(total))
        ranges = {
            "analyzing": (8, 72, "Analizando técnica y trayectoria"),
            "rendering": (72, 94, "Dibujando tu vídeo final"),
            "exporting": (94, 99, "Preparando la descarga"),
        }
        start, end, label = ranges.get(phase, (8, 99, "Procesando el vídeo"))
        self.store.update_progress(job_id, round(start + (end - start) * fraction), label)
