$ErrorActionPreference = "Stop"

$GpuPython = "C:\fahmai_paddle_gpu\Scripts\python.exe"
$CpuPython = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $GpuPython)) {
    throw "Missing $GpuPython. Use the CPU fallback or install the dedicated GPU environment."
}
if (-not (Test-Path -LiteralPath $CpuPython)) {
    throw "Missing $CpuPython. Run setup_paddle_fast_path.ps1 before exam day."
}

& $GpuPython "$PSScriptRoot\check_paddle_gpu_ready.py"
if ($LASTEXITCODE -ne 0) {
    throw "Paddle GPU readiness check failed."
}

& $GpuPython "$PSScriptRoot\run_fast_dense_bank_crop_ocr.py" --device "gpu:0"
if ($LASTEXITCODE -ne 0) {
    throw "Dense bank GPU crop OCR failed."
}

& $CpuPython "$PSScriptRoot\repair_dense_page_boundary_gaps.py"
if ($LASTEXITCODE -ne 0) {
    throw "Dense page-boundary repair failed."
}

& $CpuPython "$PSScriptRoot\get_fast_dense_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Dense bank checkpoint status contains failures."
}

& $GpuPython "$PSScriptRoot\run_fast_compact_bank_crop_ocr.py" --device "gpu:0"
if ($LASTEXITCODE -ne 0) {
    throw "Compact bank GPU crop OCR failed."
}

& $CpuPython "$PSScriptRoot\repair_compact_statement_chains.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compact bank visible-chain repair failed."
}

& $CpuPython "$PSScriptRoot\get_fast_compact_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compact bank checkpoint status contains failures."
}

& $GpuPython "$PSScriptRoot\run_fast_bbl_bank_crop_ocr.py" --device "gpu:0"
if ($LASTEXITCODE -ne 0) {
    throw "BBL bank GPU crop OCR failed."
}

& $CpuPython "$PSScriptRoot\get_fast_bbl_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "BBL bank checkpoint status contains failures."
}

& $GpuPython "$PSScriptRoot\run_fast_sparse_bank_crop_ocr.py" --device "gpu:0"
if ($LASTEXITCODE -ne 0) {
    throw "Sparse direct-bank GPU crop OCR failed."
}

& $CpuPython "$PSScriptRoot\get_fast_sparse_bank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Sparse direct-bank checkpoint status contains failures."
}

& $CpuPython "$PSScriptRoot\build_fast_partial_submission.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission build failed."
}

& $CpuPython "$PSScriptRoot\validate_submission.py" "$PSScriptRoot\..\ocr_outputs\submission_OCR_fast_partial.csv"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission validation failed."
}
