[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$DatasetZip,
    [ValidateSet("detect", "segment", "pose", "obb")][string]$Task = "detect",
    [string]$Zone = "europe-west4-a",
    [string]$BucketName = "",
    [string]$BaseModel = "",
    [string]$OutputModel = "",
    [int]$Epochs = 0,
    [int]$ImageSize = 0,
    [double]$Batch = -1,
    [int]$MaxHours = 8,
    [switch]$Spot,
    [switch]$NoWait,
    [switch]$KeepVm
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DatasetZip = (Resolve-Path -LiteralPath $DatasetZip).Path
if (-not (Test-Path -LiteralPath $DatasetZip -PathType Leaf)) {
    throw "No existe el ZIP del dataset: $DatasetZip"
}
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "No encuentro gcloud. Instala Google Cloud CLI y abre una PowerShell nueva."
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "No encuentro Python. Instala primero las dependencias de PowerNZ."
}

$defaults = @{
    detect  = @{ base = "yolo26s.pt"; output = "powerai_bar_detector.pt"; epochs = 140; imgsz = 960 }
    segment = @{ base = "yolo26s-seg.pt"; output = "powerai_athlete_seg.pt"; epochs = 140; imgsz = 768 }
    pose    = @{ base = "yolo26s-pose.pt"; output = "powerai_bar_pose.pt"; epochs = 160; imgsz = 960 }
    obb     = @{ base = "yolo26s-obb.pt"; output = "powerai_bar_obb.pt"; epochs = 140; imgsz = 960 }
}
$selected = $defaults[$Task]
$baseModelLocalPath = ""
if (-not $BaseModel) {
    $currentModels = @{
        detect  = Join-Path $Root "models\powerai_bar_detector.pt"
        segment = Join-Path $Root "models\powerai_athlete_seg.pt"
    }
    if ($currentModels.ContainsKey($Task) -and (Test-Path -LiteralPath $currentModels[$Task] -PathType Leaf)) {
        $baseModelLocalPath = (Resolve-Path -LiteralPath $currentModels[$Task]).Path
        $BaseModel = $baseModelLocalPath
    } else {
        $BaseModel = $selected.base
    }
} elseif (Test-Path -LiteralPath $BaseModel -PathType Leaf) {
    $baseModelLocalPath = (Resolve-Path -LiteralPath $BaseModel).Path
    $BaseModel = $baseModelLocalPath
}
if (-not $OutputModel) { $OutputModel = $selected.output }
if ($Epochs -le 0) { $Epochs = $selected.epochs }
if ($ImageSize -le 0) { $ImageSize = $selected.imgsz }
if (-not $BucketName) {
    $baseBucket = ($ProjectId.ToLower() -replace '[^a-z0-9-]', '-') + "-powernz-training"
    $BucketName = $baseBucket.Substring(0, [Math]::Min(63, $baseBucket.Length)).Trim('-')
}

Write-Host "Comprobando el ZIP antes de crear recursos de pago..." -ForegroundColor Cyan
& python (Join-Path $PSScriptRoot "preflight_dataset.py") --dataset $DatasetZip --task $Task
if ($LASTEXITCODE -ne 0) { throw "El dataset no paso la comprobacion previa; no se ha creado ninguna VM." }

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [switch]$Capture)
    if ($Capture) {
        $result = & gcloud @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Fallo: gcloud $($Arguments -join ' ')" }
        return (($result | Out-String).Trim())
    }
    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Fallo: gcloud $($Arguments -join ' ')" }
}

$account = Invoke-Gcloud -Capture -Arguments @("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
if (-not $account) { throw "No hay una cuenta activa. Ejecuta primero: gcloud init" }
Write-Host "Cuenta Google: $account" -ForegroundColor Cyan
Write-Host "Proyecto: $ProjectId" -ForegroundColor Cyan

Invoke-Gcloud -Arguments @("config", "set", "project", $ProjectId)
Invoke-Gcloud -Arguments @("services", "enable", "compute.googleapis.com", "storage.googleapis.com", "iam.googleapis.com", "--project", $ProjectId)

$bucketUri = "gs://$BucketName"
& gcloud storage buckets describe $bucketUri --project $ProjectId *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creando bucket $bucketUri..." -ForegroundColor Cyan
    Invoke-Gcloud -Arguments @("storage", "buckets", "create", $bucketUri, "--project", $ProjectId, "--location", "europe-west4", "--uniform-bucket-level-access")
}

$serviceAccountName = "powernz-trainer"
$serviceAccount = "$serviceAccountName@$ProjectId.iam.gserviceaccount.com"
& gcloud iam service-accounts describe $serviceAccount --project $ProjectId *> $null
if ($LASTEXITCODE -ne 0) {
    Invoke-Gcloud -Arguments @("iam", "service-accounts", "create", $serviceAccountName, "--project", $ProjectId, "--display-name", "PowerNZ trainer")
}
Invoke-Gcloud -Arguments @(
    "projects", "add-iam-policy-binding", $ProjectId,
    "--member", "serviceAccount:$serviceAccount",
    "--role", "roles/storage.objectAdmin",
    "--condition=None",
    "--quiet"
)

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$jobId = "powernz-$Task-$timestamp"
$instanceName = $jobId.ToLower()
$jobUri = "$bucketUri/jobs/$jobId"
$datasetUri = "$jobUri/input/dataset.zip"
$trainerUri = "$jobUri/input/remote_train.py"
$outputUri = "$jobUri/output"
$batchText = $Batch.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$baseModelUri = "none"
$baseModelMetadata = $BaseModel

Write-Host "Subiendo dataset y entrenador..." -ForegroundColor Cyan
Invoke-Gcloud -Arguments @("storage", "cp", $DatasetZip, $datasetUri)
Invoke-Gcloud -Arguments @("storage", "cp", (Join-Path $PSScriptRoot "remote_train.py"), $trainerUri)
if ($baseModelLocalPath) {
    $baseModelUri = "$jobUri/input/base_model.pt"
    $baseModelMetadata = "base_model.pt"
    Write-Host "Afinando desde tu modelo actual: $baseModelLocalPath" -ForegroundColor Green
    Invoke-Gcloud -Arguments @("storage", "cp", $baseModelLocalPath, $baseModelUri)
} else {
    Write-Host "Entrenando desde el modelo base: $BaseModel" -ForegroundColor Yellow
}

$metadata = @(
    "job_id=$jobId",
    "dataset_uri=$datasetUri",
    "trainer_uri=$trainerUri",
    "output_uri=$outputUri",
    "task=$Task",
    "base_model=$baseModelMetadata",
    "base_model_uri=$baseModelUri",
    "output_model=$OutputModel",
    "epochs=$Epochs",
    "imgsz=$ImageSize",
    "batch=$batchText",
    "max_hours=$MaxHours"
) -join ','

$createArgs = @(
    "compute", "instances", "create", $instanceName,
    "--project", $ProjectId,
    "--zone", $Zone,
    "--machine-type", "n1-standard-8",
    "--accelerator", "type=nvidia-tesla-t4,count=1",
    "--image-family", "pytorch-2-9-cu129-ubuntu-2204-nvidia-580",
    "--image-project", "deeplearning-platform-release",
    "--boot-disk-size", "100GB",
    "--boot-disk-type", "pd-balanced",
    "--maintenance-policy", "TERMINATE",
    "--no-restart-on-failure",
    "--service-account", $serviceAccount,
    "--scopes", "https://www.googleapis.com/auth/cloud-platform",
    "--metadata", $metadata,
    "--metadata-from-file", "startup-script=$(Join-Path $PSScriptRoot 'startup_train.sh')"
)
if ($Spot) {
    $createArgs += @("--provisioning-model", "SPOT", "--instance-termination-action", "STOP")
}

Write-Host "Creando VM GPU $instanceName..." -ForegroundColor Green
Invoke-Gcloud -Arguments $createArgs
Write-Host "Job creado: $jobId" -ForegroundColor Green
Write-Host "Resultados: $outputUri"

$resultCommand = ".\google_cloud\Get-PowerNZTrainingResult.ps1 -ProjectId `"$ProjectId`" -JobId `"$jobId`" -Task $Task -Zone `"$Zone`" -InstanceName `"$instanceName`" -BucketName `"$BucketName`" -OutputModel `"$OutputModel`" -Wait"
if ($KeepVm) { $resultCommand += " -KeepVm" }
if ($NoWait) {
    Write-Host "La VM seguira trabajando sola. Para recoger el resultado ejecuta:" -ForegroundColor Yellow
    Write-Host $resultCommand -ForegroundColor White
    exit 0
}

& (Join-Path $PSScriptRoot "Get-PowerNZTrainingResult.ps1") `
    -ProjectId $ProjectId -JobId $jobId -Task $Task -Zone $Zone `
    -InstanceName $instanceName -BucketName $BucketName -OutputModel $OutputModel `
    -Wait -KeepVm:$KeepVm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
