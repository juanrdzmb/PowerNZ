from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

try:
    from google_cloud.remote_train import (
        find_dataset_yaml,
        model_contract_errors,
        normalize_dataset_yaml,
        safe_extract_zip,
    )
except ModuleNotFoundError:
    from remote_train import (  # type: ignore[no-redef]
        find_dataset_yaml,
        model_contract_errors,
        normalize_dataset_yaml,
        safe_extract_zip,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba un ZIP YOLO antes de gastar GPU.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", choices=["detect", "segment", "pose", "obb"], required=True)
    args = parser.parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"No existe: {args.dataset}")

    with tempfile.TemporaryDirectory(prefix="powernz-preflight-") as directory:
        root = Path(directory)
        safe_extract_zip(args.dataset, root / "dataset")
        source = find_dataset_yaml(root / "dataset")
        data = normalize_dataset_yaml(source, root / "data.normalized.yaml")
        errors = model_contract_errors(args.task, args.task, data.get("names", {}))
        if errors:
            raise SystemExit("Dataset rechazado: " + "; ".join(errors))
        print(f"Dataset correcto: {source.name}")
        print(f"Clases: {data.get('names')}")
        print(f"Train: {data.get('train')}")
        print(f"Val: {data.get('val')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
