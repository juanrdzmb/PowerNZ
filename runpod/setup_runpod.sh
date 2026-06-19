#!/usr/bin/env bash
set -euo pipefail

cd "${POWERNZ_DIR:-/workspace/PowerNZ}"

echo "[PowerNZ] Actualizando pip e instalando dependencias..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "[PowerNZ] Instalando herramientas utiles para RunPod..."
python -m pip install "ultralytics>=8.3.0" huggingface_hub roboflow

echo "[PowerNZ] Comprobando GPU..."
python - <<'PY'
import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

mkdir -p /workspace/datasets /workspace/outputs /workspace/PowerNZ/runpod_outputs

echo
echo "[PowerNZ] Setup listo."
echo "Dataset recomendado: /workspace/datasets/<tu_dataset>.zip"
echo "Entrenar: bash runpod/train_runpod.sh --dataset-zip /workspace/datasets/<tu_dataset>.zip"
