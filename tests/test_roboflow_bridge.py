"""Tests for the Roboflow -> Kaggle dataset bridge (training_cloud_kit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

KIT_ROOT = Path(__file__).resolve().parents[1] / "training_cloud_kit"
sys.path.insert(0, str(KIT_ROOT))

import preparar_dataset_roboflow as bridge  # noqa: E402


def _make_roboflow_export(root: Path, names_line: str) -> None:
    """Create a minimal Roboflow YOLOv8 export with train/valid splits."""
    (root / "data.yaml").write_text(
        "\n".join(
            [
                "train: ../train/images",
                "val: ../valid/images",
                "nc: 2",
                names_line,
                "",
            ]
        ),
        encoding="utf-8",
    )
    for split, stem, label in (
        ("train", "peso_001", "0 0.50 0.50 0.10 0.10\n1 0.40 0.40 0.20 0.20\n"),
        ("valid", "sent_002", "1 0.30 0.30 0.25 0.25\n"),
    ):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # dummy jpg bytes
        (labels / f"{stem}.txt").write_text(label, encoding="utf-8")


def test_read_class_names_inline_list(tmp_path: Path) -> None:
    _make_roboflow_export(tmp_path, "names: ['Bar_hub', 'Plate']")
    assert bridge._read_class_names(tmp_path) == ["Bar_hub", "Plate"]


def test_read_class_names_block_dash(tmp_path: Path) -> None:
    _make_roboflow_export(tmp_path, "names:\n  - Bar_hub\n  - Plate")
    assert bridge._read_class_names(tmp_path) == ["Bar_hub", "Plate"]


def test_remap_swaps_alphabetical_roboflow_order() -> None:
    # Roboflow sorts alphabetically: Bar_hub=0, Plate=1 -> must become plate=0, bar_hub=1
    remap = bridge._build_class_remap(["Bar_hub", "Plate"], "detect")
    assert remap == {0: 1, 1: 0}


def test_remap_keeps_correct_order() -> None:
    remap = bridge._build_class_remap(["plate", "bar_hub"], "detect")
    assert remap == {0: 0, 1: 1}


def test_flatten_remaps_label_indices(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    _make_roboflow_export(export, "names: ['Bar_hub', 'Plate']")

    work_root = tmp_path / "powerai_bar_v1"
    from prepare_powerai_cloud_dataset import _ensure_dataset_dirs

    _ensure_dataset_dirs(work_root)
    names = bridge._read_class_names(export)
    remap = bridge._build_class_remap(names, "detect")
    splits = bridge._find_split_dirs(export)
    stats = bridge._flatten_splits(
        splits=splits, work_root=work_root, remap=remap, segmentation=False
    )

    assert stats["frames"] == 2
    assert stats["labels"] == 2
    # peso_001: '0 ...'(Bar_hub)->'1 ...', '1 ...'(Plate)->'0 ...'
    peso = (work_root / "labels" / "peso_001.txt").read_text(encoding="utf-8").splitlines()
    classes = sorted(line.split()[0] for line in peso)
    assert classes == ["0", "1"]
    # the box that was Plate (class 1, w=0.20) must now be class 0
    plate_line = next(line for line in peso if line.split()[3] == "0.200000" or line.split()[3] == "0.20")
    assert plate_line.split()[0] == "0"


def test_unknown_classes_raise(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        bridge._build_class_remap(["dog", "cat"], "detect")
