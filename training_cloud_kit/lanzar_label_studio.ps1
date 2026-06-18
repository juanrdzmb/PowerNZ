$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Arranca Label Studio en local para etiquetar los frames de PowerAI.
#
# - Sirve las imagenes desde C:\Users\Juanda\Documents\Entrenamiento (local,
#   nada sube a internet).
# - La primera vez crea el entorno virtual e instala Label Studio.
# - En paralelo crea/sincroniza los 3 proyectos (peso muerto, sentadilla,
#   press banca) y abre el navegador.
#
# Para parar: Ctrl+C en esta ventana.
# ---------------------------------------------------------------------------

$repoRoot      = Split-Path -Parent $PSScriptRoot
$venv          = Join-Path $repoRoot ".venv_labelstudio"
$python        = Join-Path $venv "Scripts\python.exe"
$labelStudio   = Join-Path $venv "Scripts\label-studio.exe"
$dataDir       = Join-Path $PSScriptRoot "label_studio_data"
$configScript  = Join-Path $PSScriptRoot "configurar_label_studio.py"
$documentRoot  = Join-Path $HOME "Documents\Entrenamiento"
$port          = 8080

# --- 1. Entorno virtual + instalacion (solo la primera vez) ----------------
if (-not (Test-Path $python)) {
    Write-Host "Creando entorno virtual de Label Studio..."
    python -m venv $venv
}
if (-not (Test-Path $labelStudio)) {
    Write-Host "Instalando Label Studio (tarda unos minutos la primera vez)..."
    & $python -m pip install --upgrade pip --quiet
    & $python -m pip install label-studio requests
}

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

# --- 2. Variables de entorno -----------------------------------------------
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = $documentRoot
$env:LABEL_STUDIO_BASE_DATA_DIR               = $dataDir
$env:LABEL_STUDIO_USERNAME                    = "juanda@powerai.local"
$env:LABEL_STUDIO_PASSWORD                    = "powerai-local-2026"
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = "true"

Write-Host ""
Write-Host "Document root (imagenes locales): $documentRoot"
Write-Host "Datos de Label Studio:           $dataDir"
Write-Host ""

# --- 3. Job en paralelo: espera, configura proyectos y abre el navegador ----
$configJob = Start-Job -ScriptBlock {
    param($python, $configScript, $port)
    $base = "http://localhost:$port"
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "$base/health" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { break }
        } catch { Start-Sleep -Seconds 3 }
    }
    & $python $configScript --base-url $base 2>&1
    Start-Process $base
} -ArgumentList $python, $configScript, $port

# --- 4. Servidor en primer plano (Ctrl+C para parar) -----------------------
Write-Host "Abriendo Label Studio en http://localhost:$port ..."
Write-Host "(la primera vez tarda ~20s en estar listo; el navegador se abre solo)"
Write-Host ""
try {
    & $labelStudio start --port $port --no-browser
}
finally {
    # Mostrar lo que hizo la configuracion automatica al cerrar.
    Receive-Job $configJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
    Remove-Job $configJob -Force -ErrorAction SilentlyContinue | Out-Null
}
