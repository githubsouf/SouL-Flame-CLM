# ============================================================
#  SouL Flame - one-time environment setup (PowerShell)
#  run:  .\setup.ps1
#  if it refuses:  Set-ExecutionPolicy -Scope Process Bypass
# ============================================================
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "`n=== SouL Flame setup ===" -ForegroundColor Cyan
Write-Host "folder: $PWD`n"

# --- 1. find python ---
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { $null }

if (-not $py) {
    Write-Host "[FAIL] python not found." -ForegroundColor Red
    Write-Host "       get it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'"
    exit 1
}
Write-Host "[1/5] python found:"
& $py --version

# --- 2. venv ---
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "[2/5] .venv already exists, keeping it"
} else {
    Write-Host "[2/5] creating .venv ..."
    & $py -m venv .venv
}
$vpy = ".\.venv\Scripts\python.exe"

# --- 3. pip ---
Write-Host "[3/5] upgrading pip ..."
& $vpy -m pip install --upgrade pip --quiet

# --- 4. packages ---
Write-Host "[4/5] installing torch, tiktoken, numpy, pytest ..."
Write-Host "      (torch is ~200 MB, first run takes a few minutes)"
& $vpy -m pip install -r requirements.txt

# --- 5. check ---
Write-Host "`n[5/5] checking the environment ...`n"
& $vpy check_env.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " done. in every new terminal, run:"
Write-Host "     .\.venv\Scripts\Activate.ps1"
Write-Host " then:  pytest -q"
Write-Host "============================================================" -ForegroundColor Cyan
