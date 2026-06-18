from __future__ import annotations

import numpy as np
import cv2

from datasets.autolabel_world import _box_area_ratio, _to_yolo_line
from datasets.yolo_dataset import materialize_yolo_detection_dataset


def test_materialize_yolo_detection_dataset_repairs_and_splits_labels(tmp_path) -> None:
    source = tmp_path / "source"
    frames = source / "frames"
    labels = source / "labels"
    frames.mkdir(parents=True)
    labels.mkdir(parents=True)

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    for index in range(4):
        image_path = frames / f"frame_{index:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        labels.joinpath(f"frame_{index:03d}.txt").write_text(
            "0 0.500000 1.100000 0.400000 0.400000\n"
            "3 0.500000 0.500000 0.200000 0.200000\n",
            encoding="utf-8",
        )

    prepared = materialize_yolo_detection_dataset(
        source_root=source,
        output_root=tmp_path / "prepared",
        class_count=2,
        val_ratio=0.25,
        seed=7,
        repair_labels=True,
    )

    assert prepared.train_count == 3
    assert prepared.val_count == 1

    output_labels = list((prepared.output_root / "labels" / "train").glob("*.txt"))
    assert output_labels
    first_line = output_labels[0].read_text(encoding="utf-8").strip()
    parts = first_line.split()
    assert parts[0] == "0"
    assert all(0.0 <= float(value) <= 1.0 for value in parts[1:])


def test_autolabel_world_clamps_boxes_to_yolo_format() -> None:
    line = _to_yolo_line(1, -10.0, 5.0, 110.0, 55.0, 100, 100)

    assert line == "1 0.500000 0.300000 1.000000 0.500000"
    assert _box_area_ratio(0.0, 0.0, 100.0, 80.0, 100, 100) == 0.8
