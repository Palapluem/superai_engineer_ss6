$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_paddle_fast_path.ps1 before exam day."
}

& $Python "$PSScriptRoot\audit_fast_path_preflight.py" --decode-images
if ($LASTEXITCODE -ne 0) {
    throw "Render preflight failed."
}

& $Python "$PSScriptRoot\build_fast_ocr_work_queue.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fast OCR work queue validation failed."
}

Write-Host "OCR exam preflight passed."
