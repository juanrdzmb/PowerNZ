from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "models" / "model_manifest.json"


@dataclass(frozen=True)
class ModelAsset:
    name: str
    path: Path
    url: str
    sha256: str
    required: bool
    description: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ModelAsset":
        return cls(
            name=str(data["name"]),
            path=ROOT / str(data["path"]),
            url=str(data["url"]),
            sha256=str(data.get("sha256", "")).lower(),
            required=bool(data.get("required", True)),
            description=str(data.get("description", "")),
        )


def _load_manifest(path: Path) -> list[ModelAsset]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ModelAsset.from_json(item) for item in data.get("models", [])]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid(asset: ModelAsset) -> bool:
    if not asset.path.exists():
        return False
    if not asset.sha256:
        return True
    return _sha256(asset.path) == asset.sha256


def _download_http(asset: ModelAsset, temporary_path: Path) -> None:
    request = urllib.request.Request(asset.url, headers={"User-Agent": "PowerNZ-model-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded * 100 / total
                print(f"\r  {percent:5.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
        print()


def _download_with_gh(asset: ModelAsset) -> bool:
    marker = "github.com/"
    release_marker = "/releases/download/"
    if marker not in asset.url or release_marker not in asset.url:
        return False
    if shutil.which("gh") is None:
        return False

    _, rest = asset.url.split(marker, 1)
    repo, release_and_file = rest.split(release_marker, 1)
    tag, _filename = release_and_file.split("/", 1)
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            asset.path.name,
            "--dir",
            str(asset.path.parent),
            "--clobber",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return False
    return asset.path.exists()


def _download_asset(asset: ModelAsset, force: bool) -> None:
    if _is_valid(asset) and not force:
        print(f"OK: {asset.path.relative_to(ROOT)}")
        return

    asset.path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = asset.path.with_suffix(asset.path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    print(f"Descargando {asset.name}...")
    try:
        _download_http(asset, temporary_path)
        temporary_path.replace(asset.path)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        print(f"  Descarga directa no disponible: {exc}")
        print("  Intento con GitHub CLI si el repo/release es privado...")
        if not _download_with_gh(asset):
            raise RuntimeError(
                "No pude descargar el modelo. Si la release es privada, inicia sesion con "
                "`gh auth login`, o usa una URL publica en models/model_manifest.json."
            ) from exc

    if not _is_valid(asset):
        if asset.path.exists():
            asset.path.unlink()
        raise RuntimeError(f"El hash de {asset.name} no coincide. Descarga cancelada.")
    print(f"Listo: {asset.path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga los modelos entrenados de PowerNZ.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="Vuelve a descargar aunque el archivo exista.")
    parser.add_argument("--check", action="store_true", help="Solo comprueba si los modelos estan presentes.")
    parser.add_argument("--include-optional", action="store_true", help="Incluye modelos opcionales.")
    parser.add_argument("--only", default=None, help="Descarga solo el archivo indicado, por ejemplo powerai_bar_detector.pt.")
    args = parser.parse_args()

    assets = _load_manifest(args.manifest)
    if args.only:
        assets = [asset for asset in assets if asset.path.name == args.only or asset.name == args.only]
    if not args.include_optional:
        assets = [asset for asset in assets if asset.required]

    if not assets:
        print("No hay modelos que procesar.")
        return 0

    missing_or_invalid = [asset for asset in assets if not _is_valid(asset)]
    if args.check:
        for asset in assets:
            status = "OK" if _is_valid(asset) else "FALTA"
            print(f"{status}: {asset.path.relative_to(ROOT)}")
        return 1 if missing_or_invalid else 0

    for asset in assets:
        _download_asset(asset, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
