$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}
$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$OutputRoot = Join-Path $Root "ocr_outputs"
$CheckpointDirs = @(
    (Join-Path $OutputRoot "fast_dense_bank"),
    (Join-Path $OutputRoot "fast_compact_bank"),
    (Join-Path $OutputRoot "fast_bbl_bank"),
    (Join-Path $OutputRoot "fast_sparse_bank"),
    (Join-Path $OutputRoot "fast_fixed_nonbank"),
    (Join-Path $OutputRoot "fast_general_documents")
)
$FastPartial = Join-Path $OutputRoot "submission_OCR_fast_partial.csv"
$FastPartialAudit = Join-Path $OutputRoot "submission_OCR_fast_partial.audit.json"
$FinalSubmission = Join-Path $OutputRoot "submission_OCR_image_reviewed.csv"
$Annotations = Join-Path $OutputRoot "audits\image_grounded\manual_ground_truth_annotations.csv"

& $Python "$PSScriptRoot\build_fast_partial_submission.py" `
    --data-root "$Root" `
    --checkpoint-dirs $CheckpointDirs `
    --output "$FastPartial" `
    --audit-output "$FastPartialAudit"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission build failed."
}

& $Python "$PSScriptRoot\validate_submission.py" "$FastPartial" --root "$Root"
if ($LASTEXITCODE -ne 0) {
    throw "Fast partial submission validation failed."
}

& $Python "$PSScriptRoot\apply_verified_image_annotations.py" `
    --data-root "$Root" `
    --base-submission "$FastPartial" `
    --annotations "$Annotations" `
    --output "$FinalSubmission" `
    --skip-render-path-check
if ($LASTEXITCODE -ne 0) {
    throw "Verified public-render annotations could not be applied."
}

& $Python "$PSScriptRoot\validate_submission.py" "$FinalSubmission" --root "$Root"
if ($LASTEXITCODE -ne 0) {
    throw "Image-reviewed submission validation failed."
}

Write-Host "Final submission written to ..\ocr_outputs\submission_OCR_image_reviewed.csv"
