from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "dzmbo/PowerNZ-Models"
DEFAULT_MANIFEST = ROOT / "models" / "model_manifest.json"


def load_upload_assets(
    manifest_path: Path,
    *,
    include_optional: bool = False,
    only: str | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = []
    for item in data.get("models", []):
        path = ROOT / str(item["path"])
        if only and only not in {path.name, str(item.get("name", ""))}:
            continue
        if not include_optional and not bool(item.get("required", True)) and not only:
            continue
        assets.append({**item, "local_path": path})
    return assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Sube los modelos del manifest de PowerNZ a Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Repositorio destino usuario/PowerNZ-Models.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private", action="store_true", help="Crea el repo como privado.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include-optional", action="store_true", help="Incluye bar pose/OBB opcionales.")
    parser.add_argument("--only", default=None, help="Sube solo este archivo o nombre de modelo.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError as exc:
        raise SystemExit("Instala huggingface_hub con: python -m pip install huggingface-hub") from exc

    assets = load_upload_assets(args.manifest, include_optional=args.include_optional, only=args.only)
    if not assets:
        raise SystemExit("No hay modelos que coincidan con la seleccion.")
    missing = [item["local_path"].name for item in assets if not item["local_path"].exists()]
    if missing:
        raise SystemExit(f"Faltan modelos locales: {', '.join(missing)}")

    api = HfApi()
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - la API envuelve los errores HTTP
        message = str(exc)
        if "403" in message or "Forbidden" in message:
            raise SystemExit(
                "Hugging Face rechazo el acceso. Revisa el repo-id y entra con un token Write usando `hf auth login`."
            ) from exc
        raise
    for item in assets:
        local_path: Path = item["local_path"]
        print(f"Subiendo {local_path.name}...")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            commit_message=f"Update {local_path.name}",
        )

    upload_file(
        path_or_fileobj=str(args.manifest),
        path_in_repo="model_manifest.json",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Update model manifest",
    )
    print(f"Listo: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
