#!/usr/bin/env bash
set -euo pipefail

cd "${POWERNZ_DIR:-/workspace/PowerNZ}"

DATASET_ZIP=""
DATASET_ROOT=""
MODE="train"
WORK_DIR="/workspace/PowerNZ_runpod_training"
OUTPUT_ZIP="/workspace/outputs/PowerNZ_trained_models_runpod.zip"
DEVICE="0"
DETECTOR_EPOCHS="140"
DETECTOR_IMGSZ="960"
DETECTOR_BATCH="12"
DETECTOR_BASE="yolo11s.pt"
ATHLETE_EPOCHS="120"
ATHLETE_IMGSZ="768"
ATHLETE_BATCH="6"
ATHLETE_BASE="yolo11s-seg.pt"
SKIP_ATHLETE_SEG="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-zip) DATASET_ZIP="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --output-zip) OUTPUT_ZIP="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --detector-epochs) DETECTOR_EPOCHS="$2"; shift 2 ;;
    --detector-imgsz) DETECTOR_IMGSZ="$2"; shift 2 ;;
    --detector-batch) DETECTOR_BATCH="$2"; shift 2 ;;
    --detector-base-model) DETECTOR_BASE="$2"; shift 2 ;;
    --athlete-epochs) ATHLETE_EPOCHS="$2"; shift 2 ;;
    --athlete-imgsz) ATHLETE_IMGSZ="$2"; shift 2 ;;
    --athlete-batch) ATHLETE_BATCH="$2"; shift 2 ;;
    --athlete-base-model) ATHLETE_BASE="$2"; shift 2 ;;
    --skip-athlete-seg) SKIP_ATHLETE_SEG="true"; shift ;;
    -h|--help)
      cat <<'EOF'
Uso:
  bash runpod/train_runpod.sh --dataset-zip /workspace/datasets/PowerNZ_bar_v1_corrected.zip

Opciones utiles:
  --dataset-root /ruta/con/frames_labels   Usa carpeta en vez de ZIP.
  --mode train                             train | clean | autolabel-clean-train.
  --detector-base-model yolo11s.pt         yolo11n.pt rapido, yolo11s.pt equilibrado, yolo11m.pt mas lento.
  --detector-epochs 140
  --detector-imgsz 960
  --detector-batch 12                      Baja a 8/6/4 si falta VRAM.
  --skip-athlete-seg                       Entrena solo plate/bar_hub aunque haya masks/.
EOF
      exit 0
      ;;
    *) echo "Opcion desconocida: $1"; exit 2 ;;
  esac
done

if [[ -z "$DATASET_ZIP" && -z "$DATASET_ROOT" ]]; then
  echo "Error: pasa --dataset-zip o --dataset-root"
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_ZIP")" "$WORK_DIR"

CMD=(python training_cloud_kit/kaggle_train_powerai.py
  --mode "$MODE"
  --work-dir "$WORK_DIR"
  --output-zip "$OUTPUT_ZIP"
  --device "$DEVICE"
  --detector-base-model "$DETECTOR_BASE"
  --detector-epochs "$DETECTOR_EPOCHS"
  --detector-imgsz "$DETECTOR_IMGSZ"
  --detector-batch "$DETECTOR_BATCH"
  --athlete-base-model "$ATHLETE_BASE"
  --athlete-epochs "$ATHLETE_EPOCHS"
  --athlete-imgsz "$ATHLETE_IMGSZ"
  --athlete-batch "$ATHLETE_BATCH")

if [[ -n "$DATASET_ZIP" ]]; then
  CMD+=(--dataset-zip "$DATASET_ZIP")
fi
if [[ -n "$DATASET_ROOT" ]]; then
  CMD+=(--dataset-root "$DATASET_ROOT")
fi
if [[ "$SKIP_ATHLETE_SEG" == "true" ]]; then
  CMD+=(--skip-athlete-seg)
fi

echo "[PowerNZ] Ejecutando entrenamiento:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

MODELS_DIR="$WORK_DIR/models"
LOCAL_MODELS_DIR="/workspace/PowerNZ/models"
mkdir -p "$LOCAL_MODELS_DIR"

if [[ -f "$MODELS_DIR/PowerNZ_bar_detector.pt" ]]; then
  cp -f "$MODELS_DIR/PowerNZ_bar_detector.pt" "$LOCAL_MODELS_DIR/powerai_bar_detector.pt"
  cp -f "$MODELS_DIR/PowerNZ_bar_detector.pt" "$LOCAL_MODELS_DIR/PowerNZ_bar_detector.pt"
  echo "[PowerNZ] Detector copiado a models/powerai_bar_detector.pt"
fi

if [[ -f "$MODELS_DIR/PowerNZ_athlete_seg.pt" ]]; then
  cp -f "$MODELS_DIR/PowerNZ_athlete_seg.pt" "$LOCAL_MODELS_DIR/powerai_athlete_seg.pt"
  cp -f "$MODELS_DIR/PowerNZ_athlete_seg.pt" "$LOCAL_MODELS_DIR/PowerNZ_athlete_seg.pt"
  echo "[PowerNZ] Segmentacion copiada a models/powerai_athlete_seg.pt"
fi

echo
echo "[PowerNZ] Entrenamiento terminado."
echo "ZIP final: $OUTPUT_ZIP"
echo "Resumen: $WORK_DIR/training_summary.json"
echo "Modelos locales: $LOCAL_MODELS_DIR"
