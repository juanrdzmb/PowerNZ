$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exerciseRoot = Join-Path $HOME "Documents\Entrenamiento\Press Banca"
New-Item -ItemType Directory -Force -Path $exerciseRoot | Out-Null

python (Join-Path $PSScriptRoot "prepare_powerai_cloud_dataset.py") `
  --stage frames `
  --exercise bench `
  --videos-dir $exerciseRoot `
  --work-root $exerciseRoot `
  --any-video-name `
  --max-frames-per-video 150 `
  --stride 15 `
  --max-dimension 1280 `
  --jpeg-quality 92 `
  --overwrite

Write-Host ""
Write-Host "Listo. Frames en: $exerciseRoot\frames"
Write-Host "ZIP para subir si lo necesitas: $exerciseRoot\powerai_bench_v1_frames.zip"
