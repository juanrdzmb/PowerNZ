from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "model_manifest.json"
MODEL_SPECS = {
    "detect": {
        "filename": "powerai_bar_detector.pt",
        "name": "Detector de barra y discos",
        "description": "Modelo YOLO detect con clases plate y bar_hub.",
        "required": True,
    },
    "segment": {
        "filename": "powerai_athlete_seg.pt",
        "name": "Segmentacion del atleta",
        "description": "Modelo YOLO segment con clase athlete.",
        "required": True,
    },
    "pose": {
        "filename": "powerai_bar_pose.pt",
        "name": "Keypoints de la barra",
        "description": "Modelo YOLO pose opcional para centros de discos y eje de barra.",
        "required": False,
    },
    "obb": {
        "filename": "powerai_bar_obb.pt",
        "name": "Eje orientado de la barra",
        "description": "Modelo YOLO OBB opcional para el angulo del eje de barra.",
        "required": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_errors(task: str, model_task: str, names: dict[int, str] | list[str]) -> list[str]:
    normalized = {str(value).strip().lower() for value in (names.values() if isinstance(names, dict) else names)}
    errors: list[str] = []
    if task != model_task:
        errors.append(f"tarea esperada={task}, obtenida={model_task}")
    if task == "detect" and not ("plate" in normalized and "bar_hub" in normalized):
        errors.append("faltan las clases exactas plate y bar_hub")
    if task == "segment" and "athlete" not in normalized:
        errors.append("falta la clase athlete")
    if task in {"pose", "obb"} and not ({"barbell", "bar", "plate"} & normalized):
        errors.append("falta una clase barbell/bar/plate")
    return errors


def update_manifest(manifest_path: Path, task: str, digest: str) -> dict[str, Any]:
    spec = MODEL_SPECS[task]
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = f"models/{spec['filename']}"
    repo_id = "dzmbo/PowerNZ-Models"
    existing = next((item for item in data.get("models", []) if item.get("path") == relative_path), None)
    payload = {
        "name": spec["name"],
        "path": relative_path,
        "url": f"https://huggingface.co/{repo_id}/resolve/main/{spec['filename']}",
        "sha256": digest,
        "required": spec["required"],
        "description": spec["description"],
    }
    if existing is None:
        data.setdefault("models", []).append(payload)
    else:
        existing.update(payload)
    data["version"] = "models-v2"
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida e instala un modelo entrenado en Google Cloud.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Instala las dependencias del proyecto antes de integrar el modelo.") from exc
    if not args.source.exists() or args.source.stat().st_size == 0:
        raise SystemExit(f"Modelo inexistente o vacio: {args.source}")

    model = YOLO(str(args.source))
    errors = contract_errors(args.task, str(model.task), model.names)
    if errors:
        raise SystemExit("Modelo rechazado: " + "; ".join(errors))

    spec = MODEL_SPECS[args.task]
    destination = ROOT / "models" / str(spec["filename"])
    backup_dir = ROOT / "models" / "backups"
    if destination.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(destination, backup_dir / f"{destination.stem}-{timestamp}{destination.suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, destination)
    digest = sha256(destination)
    update_manifest(args.manifest, args.task, digest)

    print(f"Modelo instalado: {destination}")
    print(f"Tarea: {model.task}; clases: {model.names}")
    print(f"SHA256: {digest}")
    if args.task in {"pose", "obb"}:
        print("AVISO: el peso queda preparado, pero requiere integrar su salida en el pipeline antes de usarlo.")
    print(f"Para publicarlo: python upload_models_to_huggingface.py --only {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
