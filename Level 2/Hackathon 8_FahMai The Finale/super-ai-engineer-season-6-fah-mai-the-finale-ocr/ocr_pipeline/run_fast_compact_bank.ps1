$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_paddle_fast_path.ps1 before exam day."
}

$env:FLAGS_use_mkldnn = "1"
& $Python "$PSScriptRoot\run_fast_compact_bank_crop_ocr.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compact bank crop OCR failed."
}

& $Python "$PSScriptRoot\repair_compact_statement_chains.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compact bank visible-chain repair failed."
}

& $Python "$PSScriptRoot\get_fast_compact_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compact bank checkpoint status contains failures."
}

& $Python "$PSScriptRoot\build_fast_partial_submission.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission build failed."
}

& $Python "$PSScriptRoot\validate_submission.py" "$PSScriptRoot\..\ocr_outputs\submission_OCR_fast_partial.csv"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission validation failed."
}
