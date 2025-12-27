Param(
  [switch]$NoVenv,
  [string]$Python = "python",
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "==> doe_nbi_hpo_project setup (Windows PowerShell)"
Write-Host "==> Python: $Python"

& $Python --version

if (-not $NoVenv) {
  if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating venv at $VenvDir"
    & $Python -m venv $VenvDir
  } else {
    Write-Host "==> venv already exists at $VenvDir"
  }

  Write-Host "==> Activating venv"
  & "$VenvDir\Scripts\Activate.ps1"
} else {
  Write-Host "==> -NoVenv selected. Installing into current Python environment."
}

Write-Host "==> Upgrading pip"
& $Python -m pip install --upgrade pip

Write-Host "==> Installing requirements.txt"
& $Python -m pip install -r requirements.txt

Write-Host "==> Smoke test imports"
& $Python -c "import numpy, pandas, sklearn, xgboost; print('OK: numpy/pandas/sklearn/xgboost imported')"

Write-Host ""
Write-Host "✅ Setup complete."
if (-not $NoVenv) {
  Write-Host "To activate later: $VenvDir\Scripts\Activate.ps1"
}
