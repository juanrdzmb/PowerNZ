from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "dzmbo/PowerNZ-Models"
DEFAULT_SOURCE_DIR = Path(r"C:\Users\Juanda\Documents\PowerNZ\models")
MODEL_FILES = (
    "PowerNZ_bar_detector.pt",
    "PowerNZ_athlete_seg.pt",
    "pose_landmarker_lite.task",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sube los modelos entrenados de PowerNZ a Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Repositorio destino, por ejemplo usuario/PowerNZ-Models.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Carpeta local donde estan los pesos.")
    parser.add_argument("--private", action="store_true", help="Crea el repo como privado. Para usuarios finales recomiendo publico.")
    parser.add_argument("--revision", default="main", help="Rama/revision destino.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError as exc:
        raise SystemExit("Instala huggingface_hub con: python -m pip install huggingface-hub") from exc

    missing = [name for name in MODEL_FILES if not (args.source_dir / name).exists()]
    if missing:
        raise SystemExit(f"Faltan modelos en {args.source_dir}: {', '.join(missing)}")

    api = HfApi()
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - Hugging Face envuelve el 403 segun version
        message = str(exc)
        if "403" in message or "Forbidden" in message:
            raise SystemExit(
                "Hugging Face no me deja crear ese repo. Revisa que el repo-id use tu usuario real "
                "(por ejemplo dzmbo/PowerNZ-Models) y que hayas iniciado sesion con un token Write."
            ) from exc
        raise

    for name in MODEL_FILES:
        local_path = args.source_dir / name
        print(f"Subiendo {name}...")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            commit_message=f"Upload {name}",
        )

    print(f"Listo: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
