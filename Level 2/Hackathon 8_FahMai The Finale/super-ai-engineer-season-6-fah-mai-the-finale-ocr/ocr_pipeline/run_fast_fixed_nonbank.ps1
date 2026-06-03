$ErrorActionPreference = "Stop"

$Python = "C:\fahmai_paddle\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_paddle_fast_path.ps1 before exam day."
}

$env:FLAGS_use_mkldnn = "1"
& $Python "$PSScriptRoot\run_fast_fixed_nonbank_ocr.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fixed-layout non-bank crop OCR failed."
}

& $Python "$PSScriptRoot\repair_invoice_checkpoint_values.py"
if ($LASTEXITCODE -ne 0) {
    throw "Invoice checkpoint format repair failed."
}

& $Python "$PSScriptRoot\repair_receipt_payment_visual_templates.py"
if ($LASTEXITCODE -ne 0) {
    throw "Receipt payment-method visual-template repair failed."
}

& $Python "$PSScriptRoot\get_fast_fixed_nonbank_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fixed-layout non-bank checkpoint status contains failures."
}
