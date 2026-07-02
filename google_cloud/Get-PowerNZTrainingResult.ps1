[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$JobId,
    [ValidateSet("detect", "segment", "pose", "obb")][string]$Task = "detect",
    [string]$Zone = "europe-west4-a",
    [string]$InstanceName = "",
    [string]$BucketName = "",
    [string]$OutputModel = "",
    [switch]$Wait,
    [switch]$KeepVm
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "No encuentro gcloud. Instala Google Cloud CLI y abre una PowerShell nueva."
}
$defaultModels = @{
    detect = "powerai_bar_detector.pt"
    segment = "powerai_athlete_seg.pt"
    pose = "powerai_bar_pose.pt"
    obb = "powerai_bar_obb.pt"
}
if (-not $OutputModel) { $OutputModel = $defaultModels[$Task] }
if (-not $BucketName) {
    $baseBucket = ($ProjectId.ToLower() -replace '[^a-z0-9-]', '-') + "-powernz-training"
    $BucketName = $baseBucket.Substring(0, [Math]::Min(63, $baseBucket.Length)).Trim('-')
}
$outputUri = "gs://$BucketName/jobs/$JobId/output"
$successUri = "$outputUri/_SUCCESS"
$failedUri = "$outputUri/_FAILED"

function Test-GcsObject([string]$Uri) {
    & gcloud storage ls $Uri --project $ProjectId *> $null
    return $LASTEXITCODE -eq 0
}

if ($Wait) {
    Write-Host "Esperando al entrenamiento $JobId. Puedes cerrar esta ventana; la VM se apaga sola." -ForegroundColor Cyan
    while (-not (Test-GcsObject $successUri) -and -not (Test-GcsObject $failedUri)) {
        $status = "desconocido"
        if ($InstanceName) {
            $status = (& gcloud compute instances describe $InstanceName --project $ProjectId --zone $Zone --format="value(status)" 2>$null | Out-String).Trim()
            if (-not $status) { $status = "no encontrada" }
        }
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - VM: $status; entrenamiento en curso..."
        Start-Sleep -Seconds 30
    }
}

$destination = Join-Path $Root "outputs\google_cloud\$JobId"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
if (Test-GcsObject $failedUri) {
    & gcloud storage cp --recursive "$outputUri/*" $destination --project $ProjectId
    Write-Host "El entrenamiento fallo. Log descargado en: $destination\startup.log" -ForegroundColor Red
    if (-not $KeepVm -and $InstanceName) {
        & gcloud compute instances delete $InstanceName --project $ProjectId --zone $Zone --quiet
    }
    exit 1
}
if (-not (Test-GcsObject $successUri)) {
    Write-Host "Todavia no hay resultado. Ejecuta otra vez este comando con -Wait." -ForegroundColor Yellow
    exit 2
}

& gcloud storage cp --recursive "$outputUri/*" $destination --project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "No pude descargar los resultados desde $outputUri" }
$modelPath = Join-Path $destination $OutputModel
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "El job termino, pero falta $OutputModel en $destination"
}

& python (Join-Path $PSScriptRoot "install_trained_model.py") --source $modelPath --task $Task
if ($LASTEXITCODE -ne 0) { throw "El modelo descargado no paso la validacion." }

if (-not $KeepVm -and $InstanceName) {
    & gcloud compute instances delete $InstanceName --project $ProjectId --zone $Zone --quiet
}

Write-Host "Todo listo. Modelo instalado en models\$OutputModel" -ForegroundColor Green
Write-Host "Informe del entrenamiento: $destination\training_summary.json"
