from __future__ import annotations

import json
import zipfile
from pathlib import Path

from google_cloud.install_trained_model import contract_errors, update_manifest
from google_cloud.remote_train import find_dataset_yaml, normalize_dataset_yaml, parse_batch, safe_extract_zip
from upload_models_to_huggingface import load_upload_assets


def test_remote_trainer_finds_and_normalizes_roboflow_yaml(tmp_path: Path) -> None:
    root = tmp_path / "export"
    (root / "train" / "images").mkdir(parents=True)
    (root / "valid" / "images").mkdir(parents=True)
    source = root / "data.yaml"
    source.write_text(
        "train: train/images\nval: valid/images\nnames: [plate, bar_hub]\nnc: 2\n",
        encoding="utf-8",
    )

    found = find_dataset_yaml(tmp_path)
    data = normalize_dataset_yaml(found, tmp_path / "normalized.yaml")

    assert Path(data["train"]).is_absolute()
    assert Path(data["train"]).exists()
    assert Path(data["val"]).exists()
    assert data["names"] == ["plate", "bar_hub"]


def test_safe_extract_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    try:
        safe_extract_zip(archive, tmp_path / "out")
    except ValueError as exc:
        assert "inseguro" in str(exc)
    else:
        raise AssertionError("unsafe zip was accepted")


def test_model_contracts_cover_runtime_classes() -> None:
    assert contract_errors("detect", "detect", {0: "plate", 1: "bar_hub"}) == []
    assert contract_errors("segment", "segment", {0: "athlete", 1: "background_person"}) == []
    assert contract_errors("detect", "detect", {0: "loose_plate"})
    assert contract_errors("pose", "detect", {0: "barbell"})


def test_installer_updates_manifest_and_huggingface_asset_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "models-v1",
                "models": [
                    {
                        "name": "Detector de barra y discos",
                        "path": "models/powerai_bar_detector.pt",
                        "url": "https://example.com/powerai_bar_detector.pt",
                        "sha256": "old",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    updated = update_manifest(manifest, "detect", "abc123")

    assert updated["version"] == "models-v2"
    assert updated["models"][0]["sha256"] == "abc123"


def test_upload_asset_filter_uses_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {"name": "required", "path": "models/a.pt", "required": True},
                    {"name": "optional", "path": "models/b.pt", "required": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    required = load_upload_assets(manifest)
    selected = load_upload_assets(manifest, only="b.pt")

    assert [item["local_path"].name for item in required] == ["a.pt"]
    assert [item["local_path"].name for item in selected] == ["b.pt"]


def test_batch_parser_preserves_auto_int_and_fraction() -> None:
    assert parse_batch("-1") == -1
    assert parse_batch("6") == 6
    assert parse_batch("0.60") == 0.6


def test_cloud_job_reuses_existing_custom_models() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "google_cloud" / "New-PowerNZTrainingJob.ps1").read_text(encoding="utf-8")
    startup = (root / "google_cloud" / "startup_train.sh").read_text(encoding="utf-8")

    assert 'models\\powerai_bar_detector.pt' in launcher
    assert 'models\\powerai_athlete_seg.pt' in launcher
    assert '"base_model_uri=$baseModelUri"' in launcher
    assert 'gcloud storage cp "$BASE_MODEL_URI" "$WORK_DIR/base_model.pt"' in startup
