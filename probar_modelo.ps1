# Primeras pruebas del modelo de barra entrenado.
#
# Cuando dejes models\powerai_bar_detector.pt en su sitio, ejecuta este script:
#   .\probar_modelo.ps1
# Analiza un video de cada ejercicio (peso muerto, sentadilla, press) usando el
# modelo entrenado y guarda video + capturas + reportes para revisar.
#
# Parametros:
#   -MaxFrames 0   -> procesa el video completo (por defecto 250, mas rapido)

param(
    [int]$MaxFrames = 250,
    [double]$PlateDiameterPx = 120
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$model = Join-Path $repoRoot "models\powerai_bar_detector.pt"
$videosBase = Join-Path $HOME "Documents\Entrenamiento"
$outDir = Join-Path $repoRoot "outputs\primeras_pruebas"

if (-not (Test-Path $model)) {
    Write-Host "No encuentro el modelo en: $model" -ForegroundColor Yellow
    Write-Host "Entrena el modelo (ver models\README.md) y deja el .pt ahi antes de probar."
    exit 1
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Carpeta de videos -> ejercicio de la app
$tests = @(
    @{ Folder = "Peso muerto"; Exercise = "deadlift" },
    @{ Folder = "Sentadilla";  Exercise = "squat" },
    @{ Folder = "Press Banca"; Exercise = "bench" }
)

$exts = @("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v")

foreach ($t in $tests) {
    $folder = Join-Path $videosBase $t.Folder
    if (-not (Test-Path $folder)) {
        Write-Host "[salto] no existe $folder" -ForegroundColor DarkGray
        continue
    }
    $video = Get-ChildItem -Path $folder -Include $exts -File -Recurse |
             Sort-Object Length | Select-Object -First 1
    if (-not $video) {
        Write-Host "[salto] sin videos en $folder" -ForegroundColor DarkGray
        continue
    }

    $out = Join-Path $outDir ("{0}.mp4" -f $t.Exercise)
    Write-Host ""
    Write-Host ("=== {0}  ({1}) ===" -f $t.Exercise, $video.Name) -ForegroundColor Cyan

    $cmd = @(
        "main.py",
        "--input", $video.FullName,
        "--output", $out,
        "--pose-backend", "yolo",
        "--exercise", $t.Exercise,
        "--plate-diameter-px", $PlateDiameterPx,
        "--disable-plate-heuristic",
        "--validation-run-label", ("primera_{0}" -f $t.Exercise),
        "--save-validation-screenshots",
        "--no-mobile-conversion"
    )
    if ($MaxFrames -gt 0) { $cmd += @("--max-frames", $MaxFrames) }

    & python @cmd
}

Write-Host ""
Write-Host "Listo. Revisa los resultados en:" -ForegroundColor Green
Write-Host "  $outDir"
Write-Host "  $repoRoot\outputs\validation\runs\  (capturas y reportes JSON/CSV)"
