"""Configura Label Studio para etiquetar PowerAI sin tocar la interfaz.

Crea (o reutiliza) un proyecto por ejercicio, le engancha el almacenamiento
local de su carpeta `frames\\` y sincroniza todas las imagenes de una vez.

Requisito: el servidor de Label Studio tiene que estar arrancado (usa
`lanzar_label_studio.ps1`). La autenticacion se hace por sesion, igual que el
navegador, porque las versiones nuevas de Label Studio bloquean los tokens
legacy por defecto.

Es idempotente: si lo vuelves a ejecutar despues de extraer mas frames,
re-sincroniza el almacenamiento y solo anade las imagenes nuevas.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

# Clases del detector. El orden define el indice YOLO al exportar:
#   0 -> plate, 1 -> bar_hub  (debe coincidir con dataset_bar.yaml del kit).
LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="plate" background="#FF6B00"/>
    <Label value="bar_hub" background="#1F6FEB"/>
  </RectangleLabels>
</View>
""".strip()

# (titulo del proyecto, nombre de la carpeta del ejercicio)
EXERCISES = [
    ("PowerAI - Peso muerto", "Peso muerto"),
    ("PowerAI - Sentadilla", "Sentadilla"),
    ("PowerAI - Press Banca", "Press Banca"),
]

IMAGE_REGEX = r".*\.(jpe?g|png)$"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("LS_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--email", default=os.environ.get("LABEL_STUDIO_USERNAME", "juanda@powerai.local"))
    parser.add_argument("--password", default=os.environ.get("LABEL_STUDIO_PASSWORD", "powerai-local-2026"))
    parser.add_argument(
        "--entrenamiento-dir",
        type=Path,
        default=Path.home() / "Documents" / "Entrenamiento",
        help="Carpeta que contiene una subcarpeta por ejercicio con frames\\ dentro.",
    )
    return parser


def login(base_url: str, email: str, password: str) -> requests.Session:
    s = requests.Session()
    s.get(f"{base_url}/user/login/", timeout=20)
    csrf = s.cookies.get("csrftoken")
    r = s.post(
        f"{base_url}/user/login/",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{base_url}/user/login/"},
        timeout=20,
        allow_redirects=False,
    )
    # Login correcto redirige (302). Un 200 normalmente significa credenciales malas.
    if r.status_code not in (301, 302):
        raise SystemExit(
            f"No pude iniciar sesion en Label Studio (HTTP {r.status_code}). "
            "Revisa usuario/contrasena o que el servidor este arrancado."
        )
    s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": base_url})
    return s


def get_projects(s: requests.Session, base_url: str) -> dict[str, int]:
    r = s.get(f"{base_url}/api/projects?page_size=1000", timeout=20)
    r.raise_for_status()
    return {p["title"]: p["id"] for p in r.json()["results"]}


def ensure_project(s: requests.Session, base_url: str, title: str, existing: dict[str, int]) -> int:
    if title in existing:
        return existing[title]
    r = s.post(
        f"{base_url}/api/projects",
        json={"title": title, "label_config": LABEL_CONFIG},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["id"]


def ensure_storage(s: requests.Session, base_url: str, project_id: int, frames_dir: Path) -> int:
    r = s.get(f"{base_url}/api/storages/localfiles?project={project_id}", timeout=20)
    r.raise_for_status()
    target = str(frames_dir)
    for storage in r.json():
        if Path(storage["path"]) == frames_dir:
            return storage["id"]
    r = s.post(
        f"{base_url}/api/storages/localfiles",
        json={
            "project": project_id,
            "title": "frames",
            "path": target,
            "regex_filter": IMAGE_REGEX,
            "use_blob_urls": True,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["id"]


def sync_storage(s: requests.Session, base_url: str, storage_id: int) -> None:
    r = s.post(f"{base_url}/api/storages/localfiles/{storage_id}/sync", json={}, timeout=600)
    r.raise_for_status()


def task_count(s: requests.Session, base_url: str, project_id: int) -> int:
    r = s.get(f"{base_url}/api/projects/{project_id}", timeout=20)
    r.raise_for_status()
    return r.json().get("task_number", 0)


def main() -> None:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    s = login(base_url, args.email, args.password)
    existing = get_projects(s, base_url)

    print(f"Label Studio: {base_url}\n")
    rows = []
    for title, folder in EXERCISES:
        frames_dir = (args.entrenamiento_dir / folder / "frames").resolve()
        if not frames_dir.is_dir():
            print(f"[saltado] {title}: no existe {frames_dir}")
            continue
        pid = ensure_project(s, base_url, title, existing)
        sid = ensure_storage(s, base_url, pid, frames_dir)
        sync_storage(s, base_url, sid)
        n = task_count(s, base_url, pid)
        rows.append((title, n, pid))
        print(f"[ok] {title}: {n} imagenes (proyecto {pid})")

    if not rows:
        print("\nNo se configuro ningun proyecto. Genera frames primero con los scripts extraer_frames_*.ps1")
        sys.exit(1)

    print("\nListo. Abre http://localhost:8080 y entra a cada proyecto para etiquetar.")


if __name__ == "__main__":
    main()
