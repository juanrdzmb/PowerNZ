#!/usr/bin/env bash
set -Eeuo pipefail

WORK_DIR="/opt/powernz-training"
LOG_FILE="$WORK_DIR/startup.log"
mkdir -p "$WORK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

metadata() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

retry() {
  local attempts=0
  local max_attempts=8
  until "$@"; do
    attempts=$((attempts + 1))
    if [[ $attempts -ge $max_attempts ]]; then
      echo "[PowerNZ] Comando fallido tras $attempts intentos: $*"
      return 1
    fi
    sleep $((attempts * 5))
  done
}

JOB_ID="$(metadata job_id)"
DATASET_URI="$(metadata dataset_uri)"
TRAINER_URI="$(metadata trainer_uri)"
OUTPUT_URI="$(metadata output_uri)"
TASK="$(metadata task)"
BASE_MODEL="$(metadata base_model)"
BASE_MODEL_URI="$(metadata base_model_uri)"
OUTPUT_MODEL="$(metadata output_model)"
EPOCHS="$(metadata epochs)"
IMGSZ="$(metadata imgsz)"
BATCH="$(metadata batch)"
MAX_HOURS="$(metadata max_hours)"
RESULT_DIR="$WORK_DIR/result"

# Corte de seguridad independiente del proceso de Python. Aunque pip o una descarga
# se queden colgados, la VM se apagara como maximo 30 minutos despues del limite.
shutdown -h "+$((MAX_HOURS * 60 + 30))" "PowerNZ safety shutdown" || true

finish() {
  code=$?
  set +e
  echo "[PowerNZ] Finalizando job $JOB_ID con codigo $code"
  gcloud storage cp "$LOG_FILE" "$OUTPUT_URI/startup.log"
  if [[ $code -eq 0 ]]; then
    printf 'ok\n' >/tmp/_SUCCESS
    gcloud storage cp /tmp/_SUCCESS "$OUTPUT_URI/_SUCCESS"
  else
    printf 'failed code=%s\n' "$code" >/tmp/_FAILED
    gcloud storage cp /tmp/_FAILED "$OUTPUT_URI/_FAILED"
  fi
  sync
  shutdown -h now
}
trap finish EXIT

echo "[PowerNZ] Job: $JOB_ID"
echo "[PowerNZ] Dataset: $DATASET_URI"
echo "[PowerNZ] Salida: $OUTPUT_URI"
nvidia-smi

retry gcloud storage cp "$DATASET_URI" "$WORK_DIR/dataset.zip"
retry gcloud storage cp "$TRAINER_URI" "$WORK_DIR/remote_train.py"
if [[ "$BASE_MODEL_URI" != "none" ]]; then
  echo "[PowerNZ] Descargando tu modelo actual para continuar su entrenamiento."
  retry gcloud storage cp "$BASE_MODEL_URI" "$WORK_DIR/base_model.pt"
  BASE_MODEL="$WORK_DIR/base_model.pt"
fi

python3 -m venv --system-site-packages "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "ultralytics>=8.4.0" "pyyaml>=6.0"

mkdir -p "$RESULT_DIR"
timeout --signal=TERM "${MAX_HOURS}h" python "$WORK_DIR/remote_train.py" \
  --dataset "$WORK_DIR/dataset.zip" \
  --task "$TASK" \
  --base-model "$BASE_MODEL" \
  --output-model "$OUTPUT_MODEL" \
  --output-dir "$RESULT_DIR" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device 0

retry gcloud storage cp "$RESULT_DIR/$OUTPUT_MODEL" "$OUTPUT_URI/"
retry gcloud storage cp "$RESULT_DIR/training_summary.json" "$OUTPUT_URI/"
if [[ -d "$RESULT_DIR/runs/powernz" ]]; then
  tar -C "$RESULT_DIR/runs" -czf "$WORK_DIR/training_artifacts.tar.gz" powernz
  gcloud storage cp "$WORK_DIR/training_artifacts.tar.gz" "$OUTPUT_URI/"
fi

echo "[PowerNZ] Entrenamiento y subida completados."
