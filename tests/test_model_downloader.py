from pathlib import Path

from model_downloader import ModelAsset, _is_valid, _sha256


def test_model_asset_validation_uses_sha256(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"powernz")
    asset = ModelAsset(
        name="modelo de prueba",
        path=model_path,
        url="https://example.com/model.pt",
        sha256=_sha256(model_path),
        required=True,
    )

    assert _is_valid(asset) is True

    model_path.write_bytes(b"cambio")

    assert _is_valid(asset) is False
