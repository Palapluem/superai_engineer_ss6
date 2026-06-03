$ErrorActionPreference = "Stop"

& python "$PSScriptRoot\run_fast_general_document_parser.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fast general-document parser failed."
}

& python "$PSScriptRoot\get_fast_general_document_status.py"
if ($LASTEXITCODE -ne 0) {
    throw "Fast general-document status contains failures."
}
