from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml


TASK_DEFAULTS = {
    "detect": ("yolo26s.pt", "powerai_bar_detector.pt"),
    "segment": ("yolo26s-seg.pt", "powerai_athlete_seg.pt"),
    "pose": ("yolo26s-pose.pt", "powerai_bar_pose.pt"),
    "obb": ("yolo26s-obb.pt", "powerai_bar_obb.pt"),
}


def parse_batch(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena un modelo PowerNZ desde un ZIP YOLO/Roboflow.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-model", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=parse_batch, default=-1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser


def safe_extract_zip(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"ZIP inseguro: {member.filename}") from exc
        archive.extractall(destination)


def find_dataset_yaml(root: Path) -> Path:
    candidates = [
        path
        for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
        for path in root.rglob(name)
        if "__MACOSX" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError("No se encontro data.yaml dentro del ZIP exportado.")
    return min(candidates, key=lambda path: (len(path.parts), len(str(path))))


def _resolve_split_path(value: str, yaml_dir: Path, declared_root: Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or value.startswith(("http://", "https://", "gs://")):
        return value

    attempts = [yaml_dir / candidate, declared_root / candidate]
    trimmed = value.replace("\\", "/")
    while trimmed.startswith("../"):
        trimmed = trimmed[3:]
    attempts.append(yaml_dir / trimmed)
    for attempt in attempts:
        if attempt.exists():
            return str(attempt.resolve())
    return str((declared_root / candidate).resolve())


def normalize_dataset_yaml(source: Path, destination: Path) -> dict[str, Any]:
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    yaml_dir = source.parent.resolve()
    declared = Path(str(data.get("path", ".")))
    declared_root = declared if declared.is_absolute() else (yaml_dir / declared).resolve()

    for split in ("train", "val", "test"):
        value = data.get(split)
        if isinstance(value, str):
            data[split] = _resolve_split_path(value, yaml_dir, declared_root)
        elif isinstance(value, list):
            data[split] = [
                _resolve_split_path(item, yaml_dir, declared_root) if isinstance(item, str) else item
                for item in value
            ]
    data.pop("path", None)
    if not data.get("train") or not data.get("val"):
        raise ValueError("El data.yaml debe incluir train y val.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data


def model_contract_errors(task: str, model_task: str, names: dict[int, str] | list[str]) -> list[str]:
    normalized = {str(value).strip().lower() for value in (names.values() if isinstance(names, dict) else names)}
    errors: list[str] = []
    if model_task != task:
        errors.append(f"tarea esperada={task}, obtenida={model_task}")
    if task == "detect" and not ("plate" in normalized and "bar_hub" in normalized):
        errors.append("el detector necesita exactamente las clases plate y bar_hub")
    if task == "segment" and "athlete" not in normalized:
        errors.append("el segmentador necesita la clase athlete")
    if task in {"pose", "obb"} and not ({"barbell", "bar", "plate"} & normalized):
        errors.append(f"el modelo {task} necesita una clase barbell/bar/plate")
    return errors


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Falta una dependencia de entrenamiento: {exc}") from exc

    if not args.dataset.exists():
        raise SystemExit(f"No existe el dataset: {args.dataset}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA no esta disponible. El entrenamiento se cancela para no gastar una VM sin GPU.")

    base_default, output_default = TASK_DEFAULTS[args.task]
    base_model = args.base_model or base_default
    output_model = args.output_model or output_default
    output_dir = args.output_dir.resolve()
    dataset_root = output_dir / "dataset"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_extract_zip(args.dataset, dataset_root)
    source_yaml = find_dataset_yaml(dataset_root)
    normalized_yaml = output_dir / "data.normalized.yaml"
    dataset_config = normalize_dataset_yaml(source_yaml, normalized_yaml)
    dataset_errors = model_contract_errors(args.task, args.task, dataset_config.get("names", {}))
    if dataset_errors:
        raise ValueError("Dataset incompatible: " + "; ".join(dataset_errors))

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"Dataset: {source_yaml}", flush=True)
    print(f"Tarea: {args.task}; base: {base_model}; epochs: {args.epochs}; imgsz: {args.imgsz}", flush=True)

    model = YOLO(base_model)
    train_kwargs: dict[str, Any] = {
        "data": str(normalized_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(runs_dir),
        "name": "powernz",
        "exist_ok": True,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "cache": "disk",
        "plots": True,
        "close_mosaic": 10,
        "degrees": 6.0,
        "translate": 0.10,
        "scale": 0.45,
        "hsv_h": 0.015,
        "hsv_s": 0.45,
        "hsv_v": 0.30,
    }
    if args.task == "pose":
        train_kwargs["fliplr"] = 0.0
        train_kwargs["mosaic"] = 0.35
        train_kwargs["mixup"] = 0.0
    else:
        train_kwargs["fliplr"] = 0.5
        train_kwargs["mosaic"] = 0.6
        train_kwargs["mixup"] = 0.05

    model.train(**train_kwargs)
    best_pt = runs_dir / "powernz" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"El entrenamiento termino sin best.pt: {best_pt}")

    trained = YOLO(str(best_pt))
    errors = model_contract_errors(args.task, str(trained.task), trained.names)
    if errors:
        raise RuntimeError("Modelo incompatible: " + "; ".join(errors))
    metrics = trained.val(data=str(normalized_yaml), device=args.device, plots=True)

    final_model = output_dir / output_model
    shutil.copy2(best_pt, final_model)
    summary = {
        "task": args.task,
        "base_model": base_model,
        "output_model": output_model,
        "sha256": sha256(final_model),
        "classes": trained.names,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dataset_names": dataset_config.get("names", {}),
        "metrics": _json_value(getattr(metrics, "results_dict", {})),
        "python": sys.version,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"MODELO_LISTO={final_model}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
