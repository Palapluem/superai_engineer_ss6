$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_paddle_fast_path.ps1 before exam day."
}

$env:FLAGS_use_mkldnn = "1"
& $Python "$PSScriptRoot\run_fast_dense_bank_crop_ocr.py"
if ($LASTEXITCODE -ne 0) {
    throw "Dense bank crop OCR failed."
}

& $Python "$PSScriptRoot\repair_dense_page_boundary_gaps.py"
if ($LASTEXITCODE -ne 0) {
    throw "Dense page-boundary repair failed."
}

& $Python "$PSScriptRoot\get_fast_dense_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Dense bank checkpoint status contains failures."
}

& $Python "$PSScriptRoot\build_fast_partial_submission.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission build failed."
}

& $Python "$PSScriptRoot\validate_submission.py" "$PSScriptRoot\..\ocr_outputs\submission_OCR_fast_partial.csv"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission validation failed."
}
