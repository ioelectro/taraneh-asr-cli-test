$ErrorActionPreference = "Stop"

$Python = "py"
$Venv = ".venv"
$ModelDir = "models/Taraneh-Q0.6B"

Write-Host "== Taraneh-Q0.6B Persian ASR setup (Windows) ==" -ForegroundColor Cyan

# Verify Python 3.12 is available.
& $Python -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install 64-bit Python 3.12 and enable the Python launcher."
}

& $Python -3.12 -m venv $Venv
$VenvPython = Join-Path $Venv "Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
& $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing Python requirements..." -ForegroundColor Yellow
& $VenvPython -m pip install -r requirements.txt

Write-Host "Downloading Taraneh-Q0.6B model..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
& (Join-Path $Venv "Scripts\hf.exe") download areffarhadi/Taraneh-Q0.6B --local-dir $ModelDir

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run your script with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python speech_recognition.py"
Write-Host "Check INPUT_DEVICE in your script before running."
