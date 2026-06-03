$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_paddle_fast_path.ps1 before exam day."
}
$GpuPython = "C:\fahmai_paddle_gpu\Scripts\python.exe"
$StatementScript = "$PSScriptRoot\run_fast_statement_bank.ps1"
if (Test-Path -LiteralPath $GpuPython) {
    & $GpuPython "$PSScriptRoot\check_paddle_gpu_ready.py"
    if ($LASTEXITCODE -eq 0) {
        $StatementScript = "$PSScriptRoot\run_fast_statement_bank_gpu.ps1"
    } else {
        Write-Warning "GPU environment is not ready. Falling back to CPU statement OCR."
    }
}
Write-Host "Statement runner: $StatementScript"

function Start-HiddenLane {
    param(
        [string]$Name,
        [string]$Script
    )
    $stdout = Join-Path $PSScriptRoot "..\ocr_outputs\$Name.stdout.log"
    $stderr = Join-Path $PSScriptRoot "..\ocr_outputs\$Name.stderr.log"
    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$Script`"" `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    # Cache the handle while the process is alive so ExitCode remains available
    # even when a short lane completes before the foreground bank lane.
    $null = $process.Handle
    return $process
}

& powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\run_exam_preflight.ps1"
if ($LASTEXITCODE -ne 0) {
    throw "Exam preflight failed."
}

$forms = Start-HiddenLane -Name "exam_fast_fixed_nonbank" -Script "$PSScriptRoot\run_fast_fixed_nonbank.ps1"
$documents = Start-HiddenLane -Name "exam_fast_general_documents" -Script "$PSScriptRoot\run_fast_general_documents.ps1"

& powershell -NoProfile -ExecutionPolicy Bypass -File $StatementScript
if ($LASTEXITCODE -ne 0) {
    throw "Statement bank lane failed."
}

$forms.WaitForExit()
$documents.WaitForExit()
$forms.Refresh()
$documents.Refresh()
if ($forms.ExitCode -ne 0) {
    throw "Fixed-layout non-bank lane failed. Check ..\ocr_outputs\exam_fast_fixed_nonbank.stderr.log"
}
if ($documents.ExitCode -ne 0) {
    throw "General-document lane failed. Check ..\ocr_outputs\exam_fast_general_documents.stderr.log"
}

& $Python "$PSScriptRoot\get_fast_fixed_nonbank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fixed-layout non-bank status contains failures."
}

& python "$PSScriptRoot\get_fast_general_document_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "General-document status contains failures."
}

& $Python "$PSScriptRoot\build_fast_partial_submission.py"
if ($LASTEXITCODE -ne 0) {
    throw "Final checkpoint submission build failed."
}

& $Python "$PSScriptRoot\apply_verified_image_annotations.py"
if ($LASTEXITCODE -ne 0) {
    throw "Verified image annotation application failed."
}

& $Python "$PSScriptRoot\validate_submission.py" "$PSScriptRoot\..\ocr_outputs\submission_OCR_image_reviewed.csv"
if ($LASTEXITCODE -ne 0) {
    throw "Final submission validation failed."
}
