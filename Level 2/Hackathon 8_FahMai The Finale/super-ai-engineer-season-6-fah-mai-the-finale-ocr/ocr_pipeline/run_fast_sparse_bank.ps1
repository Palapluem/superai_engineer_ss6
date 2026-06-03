$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_paddle_fast_path.ps1 before exam day."
}

$env:FLAGS_use_mkldnn = "1"
& $Python "$PSScriptRoot\run_fast_sparse_bank_crop_ocr.py"
if ($LASTEXITCODE -ne 0) {
    throw "Sparse direct-bank crop OCR failed."
}

& $Python "$PSScriptRoot\get_fast_sparse_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Sparse direct-bank checkpoint status contains failures."
}
