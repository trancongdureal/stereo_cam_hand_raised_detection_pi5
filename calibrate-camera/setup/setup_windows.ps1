Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "   Stereo Vision - Windows Setup Script    " -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

Write-Host "[1/4] Checking Python..."
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[2/4] Creating virtual environment in .venv..."
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $ProjectDir ".venv"

if (-not (Test-Path $VenvDir)) {
    Set-Location $ProjectDir
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

Write-Host "[3/4] Activating virtual environment & installing dependencies..."
Set-Location $ProjectDir
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
} else {
    Write-Host "Could not find activate script." -ForegroundColor Red
    exit 1
}

Write-Host "[4/4] Creating necessary directories..."
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "calibration_images\left") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "calibration_images\right") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "calibration_result") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "debug") | Out-Null

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "To begin, activate the environment:"
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host "Then run the environment check:"
Write-Host "    python environment_check.py"
Write-Host "===========================================" -ForegroundColor Cyan
