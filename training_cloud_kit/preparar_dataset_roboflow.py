"""Convierte uno o mas exports de Roboflow (YOLOv8/v11) al layout que espera el kit de Kaggle.

Roboflow exporta `train/valid/test` con `images/` + `labels/` y un `data.yaml`,
y suele ordenar las clases alfabeticamente (p.ej. `Bar_hub`=0, `Plate`=1), que es
lo CONTRARIO de lo que entrena `kaggle_train_powerai.py` (espera `plate`=0,
`bar_hub=1` y descarta cualquier clase con indice >= 2).

Este script:
  - aplana todos los splits en frames + labels (o masks en modo seg);
  - remapea los indices de clase POR NOMBRE a los indices correctos;
  - cuando recibe varios ZIPs los combina en un solo dataset con prefijo
    por ZIP para evitar colisiones de nombre;
  - genera un .zip listo para subir a Kaggle.

Uso (un ZIP):
    python training_cloud_kit/preparar_dataset_roboflow.py --roboflow-export <zip>

Uso (combinar varios ZIPs):
    python training_cloud_kit/preparar_dataset_roboflow.py --roboflow-export <zip1> <zip2> ...

Para segmentacion del atleta (mascaras YOLOv8/v11-seg):
    python training_cloud_kit/preparar_dataset_roboflow.py --roboflow-export <zips...> --task seg
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Reutiliza los empaquetadores del kit (mismo directorio).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_powerai_cloud_dataset import (  # noqa: E402
    IMAGE_EXTENSIONS,
    _ensure_dataset_dirs,
    _write_dataset_files,
    _zip_dataset,
)

# Sinonimos de Roboflow -> clase canonica del kit. Todo en minusculas.
DETECT_SYNONYMS = {
    "plate": "plate",
    "plates": "plate",
    "weight_plate": "plate",
    "weight plate": "plate",
    "bumper": "plate",
    "bumper_plate": "plate",
    "disc": "plate",
    "disco": "plate",
    "bar_hub": "bar_hub",
    "barhub": "bar_hub",
    "bar-hub": "bar_hub",
    "hub": "bar_hub",
    "barbell": "bar_hub",
    "bar_sleeve": "bar_hub",
    "sleeve": "bar_hub",
    "collar": "bar_hub",
}
SEG_SYNONYMS = {
    "athlete": "athlete",
    "atleta": "athlete",
    "lifter": "athlete",
    "person": "athlete",
    "subject": "athlete",
    "background_person": "background_person",
    "background": "background_person",
    "other_person": "background_person",
}

TASK_TARGETS = {
    "detect": {"plate": 0, "bar_hub": 1},
    "seg": {"athlete": 0, "background_person": 1},
}
TASK_SYNONYMS = {"detect": DETECT_SYNONYMS, "seg": SEG_SYNONYMS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roboflow-export",
        type=Path,
        nargs="+",
        required=True,
        help="Uno o mas ZIPs (o carpetas) del export de Roboflow (formato YOLOv8/v11).",
    )
    parser.add_argument(
        "--task",
        choices=("detect", "seg"),
        default="detect",
        help="'detect' = barra/discos (cajas). 'seg' = mascara del atleta (poligonos).",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Nombre del dataset. Por defecto powerai_bar_v1 (detect) o powerai_athlete_v1 (seg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida. Por defecto training_cloud_kit/work/<dataset-name>.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task = args.task
    dataset_name = args.dataset_name or ("powerai_bar_v1" if task == "detect" else "powerai_athlete_v1")
    kit_root = Path(__file__).resolve().parent
    work_root = (args.output_dir or (kit_root / "work" / dataset_name)).resolve()
    sources: list[Path] = args.roboflow_export
    multi = len(sources) > 1

    # First pass: reset output only once (for the first ZIP).
    # Subsequent ZIPs are merged with a prefix to avoid name collisions.
    all_names: list[str] = []
    all_remap: dict[int, int] = {}
    combined_stats: dict[str, int] = {"frames": 0, "labels": 0, "orphan_images": 0, "dropped_lines": 0}

    for idx, source in enumerate(sources):
        is_first = idx == 0
        with _materialized_export(source) as export_root:
            names = _read_class_names(export_root)
            remap = _build_class_remap(names, task)
            splits = _find_split_dirs(export_root)
            if not splits:
                raise SystemExit(
                    f"No encontre pares images/labels en {export_root}. "
                    "Asegurate de exportar en formato YOLOv8/v11 desde Roboflow."
                )

            if is_first:
                _reset_output(work_root)
            _ensure_dataset_dirs(work_root)

            # Always use a prefix when combining multiple ZIPs; single ZIP gets no prefix.
            prefix = _sanitize_tag(source.name) if multi else None
            stats = _flatten_splits(
                splits=splits,
                work_root=work_root,
                remap=remap,
                segmentation=(task == "seg"),
                prefix=prefix,
            )

        all_names.extend(names)
        all_remap.update(remap)
        for key in combined_stats:
            combined_stats[key] += stats.get(key, 0)

        print(f"\n--- ZIP {idx + 1}/{len(sources)}: {source.name} ---")
        print(f"  frames: {stats['frames']}  labels: {stats['labels']}")

    exercise_tag = "bar" if task == "detect" else "athlete"
    total_frames = sum(
        1 for p in (work_root / "frames").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    combined_stats["total_frames"] = total_frames
    _write_dataset_files(work_root=work_root, exercise=exercise_tag, frames=total_frames, videos=[])
    zip_path = work_root.parent / f"{dataset_name}_corrected.zip"
    _zip_dataset(
        root=work_root,
        zip_path=zip_path,
        include_labels=(task == "detect"),
        include_masks=(task == "seg"),
    )

    _print_report(all_names, all_remap, combined_stats, work_root, zip_path)


# --------------------------------------------------------------------------- #
# Entrada (zip o carpeta)
# --------------------------------------------------------------------------- #
class _materialized_export:
    """Context manager: devuelve una carpeta del export, extrayendo el zip si hace falta."""

    def __init__(self, source: Path) -> None:
        self._source = source.resolve()
        self._tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if not self._source.exists():
            raise SystemExit(f"No existe el export: {self._source}")
        if self._source.is_dir():
            return self._source
        if self._source.suffix.lower() != ".zip":
            raise SystemExit(f"Esperaba un .zip o una carpeta, recibi: {self._source}")
        self._tmp = tempfile.TemporaryDirectory(prefix="roboflow_export_")
        with zipfile.ZipFile(self._source) as archive:
            archive.extractall(self._tmp.name)
        return Path(self._tmp.name)

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


# --------------------------------------------------------------------------- #
# Clases / remapeo
# --------------------------------------------------------------------------- #
def _read_class_names(export_root: Path) -> list[str]:
    """Lee `names` del data.yaml de Roboflow (lista o dict). Sin dependencia de PyYAML."""
    yaml_files = sorted(export_root.rglob("data.yaml")) + sorted(export_root.rglob("data.yml"))
    if not yaml_files:
        raise SystemExit(
            f"No encontre data.yaml en {export_root}. Roboflow lo incluye en el export YOLOv8."
        )
    text = yaml_files[0].read_text(encoding="utf-8")

    # Formato lista en una linea: names: ['Plate', 'Bar_hub']  o  names: [plate, bar_hub]
    inline = re.search(r"^names:\s*\[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL)
    if inline:
        raw = inline.group(1)
        items = [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]
        if items:
            return items

    # Formato bloque:
    #   names:
    #     0: plate
    #     1: bar_hub
    # o lista con guiones:
    #   names:
    #     - plate
    #     - bar_hub
    lines = text.splitlines()
    names: list[str] = []
    in_block = False
    for line in lines:
        if re.match(r"^names:\s*$", line):
            in_block = True
            continue
        if in_block:
            indexed = re.match(r"^\s+(\d+)\s*:\s*(.+?)\s*$", line)
            dashed = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if indexed:
                names.append(indexed.group(2).strip().strip("'\""))
            elif dashed:
                names.append(dashed.group(1).strip().strip("'\""))
            elif line.strip() and not line.startswith((" ", "\t")):
                break
    if not names:
        raise SystemExit(f"No pude leer la lista 'names' de {yaml_files[0]}.")
    return names


def _build_class_remap(names: list[str], task: str) -> dict[int, int]:
    """Mapa indice_origen -> indice_destino, emparejando por nombre canonico."""
    synonyms = TASK_SYNONYMS[task]
    targets = TASK_TARGETS[task]
    remap: dict[int, int] = {}
    unknown: list[str] = []
    for src_index, raw_name in enumerate(names):
        canonical = synonyms.get(raw_name.strip().lower())
        if canonical is None:
            unknown.append(raw_name)
            continue
        remap[src_index] = targets[canonical]
    if not remap:
        raise SystemExit(
            f"Ninguna clase del export {names} coincide con {sorted(targets)} para task={task}. "
            "Revisa que etiquetaste las clases correctas."
        )
    if unknown:
        print(f"[aviso] clases ignoradas (no reconocidas): {unknown}")
    return remap


# --------------------------------------------------------------------------- #
# Splits / aplanado
# --------------------------------------------------------------------------- #
def _find_split_dirs(export_root: Path) -> list[tuple[Path, Path]]:
    """Devuelve pares (images_dir, labels_dir) de train/valid/test (o flat)."""
    pairs: list[tuple[Path, Path]] = []
    for labels_dir in sorted(export_root.rglob("labels")):
        if not labels_dir.is_dir():
            continue
        images_dir = labels_dir.parent / "images"
        if images_dir.is_dir():
            pairs.append((images_dir, labels_dir))
    return pairs


def _sanitize_tag(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "src"


def _flatten_splits(
    *,
    splits: list[tuple[Path, Path]],
    work_root: Path,
    remap: dict[int, int],
    segmentation: bool,
    prefix: str | None = None,
) -> dict[str, int]:
    frames_dir = work_root / "frames"
    labels_out_dir = work_root / ("masks" if segmentation else "labels")
    frames = 0
    labels_written = 0
    orphan_images = 0
    dropped_lines = 0
    seen_stems: set[str] = set()

    for images_dir, labels_dir in splits:
        for image_path in sorted(images_dir.iterdir()):
            if not (image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            stem = image_path.stem
            base = f"{prefix}__{stem}" if prefix else stem
            dest_stem = base
            if dest_stem in seen_stems:
                dest_stem = f"{base}__{images_dir.parent.name}"
            seen_stems.add(dest_stem)

            shutil.copy2(image_path, frames_dir / f"{dest_stem}{image_path.suffix.lower()}")
            frames += 1

            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                orphan_images += 1
                continue
            remapped, dropped = _remap_label_file(label_path, remap, segmentation=segmentation)
            dropped_lines += dropped
            (labels_out_dir / f"{dest_stem}.txt").write_text(
                "\n".join(remapped) + ("\n" if remapped else ""),
                encoding="utf-8",
            )
            if remapped:
                labels_written += 1

    return {
        "frames": frames,
        "labels": labels_written,
        "orphan_images": orphan_images,
        "dropped_lines": dropped_lines,
    }


def _remap_label_file(
    label_path: Path,
    remap: dict[int, int],
    *,
    segmentation: bool,
) -> tuple[list[str], int]:
    out: list[str] = []
    dropped = 0
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        min_parts = 7 if segmentation else 5
        if len(parts) < min_parts:
            dropped += 1
            continue
        try:
            src_cls = int(float(parts[0]))
        except ValueError:
            dropped += 1
            continue
        if src_cls not in remap:
            dropped += 1  # clase fuera del set objetivo (p.ej. una 3a clase)
            continue
        parts[0] = str(remap[src_cls])
        out.append(" ".join(parts))
    return out, dropped


# --------------------------------------------------------------------------- #
# Salida / reporte
# --------------------------------------------------------------------------- #
def _reset_output(work_root: Path) -> None:
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)


def _print_report(
    names: list[str],
    remap: dict[int, int],
    stats: dict[str, int],
    work_root: Path,
    zip_path: Path,
) -> None:
    print("\n== Roboflow -> Kaggle ==")
    print("Clases del export:")
    for src_index, name in enumerate(names):
        dest = remap.get(src_index)
        arrow = f"-> {dest}" if dest is not None else "-> (ignorada)"
        print(f"  {src_index}: {name} {arrow}")
    print(f"\nframes:            {stats['frames']}")
    print(f"labels con cajas:  {stats['labels']}")
    if stats["orphan_images"]:
        print(f"imagenes sin label: {stats['orphan_images']} (frames negativos, ok)")
    if stats["dropped_lines"]:
        print(f"lineas descartadas: {stats['dropped_lines']} (clases fuera del set)")
    print(f"\nCarpeta:  {work_root}")
    print(f"ZIP listo para Kaggle (MODE='train'):\n  {zip_path}")


if __name__ == "__main__":
    main()
