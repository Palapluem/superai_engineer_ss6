$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataLink = "C:\fahmai_ocr_data"
$Venv = "C:\fahmai_paddle"

if (Test-Path -LiteralPath $DataLink) {
    $Item = Get-Item -LiteralPath $DataLink -Force
    if ($Item.LinkType -ne "Junction") {
        throw "$DataLink exists but is not a directory junction."
    }
    $CurrentTarget = [string]($Item.Target | Select-Object -First 1)
    if ($CurrentTarget -ne $RepoRoot) {
        throw "$DataLink points to '$CurrentTarget', expected '$RepoRoot'."
    }
} else {
    New-Item -ItemType Junction -Path $DataLink -Target $RepoRoot | Out-Null
}

if (-not (Test-Path -LiteralPath "$Venv\Scripts\python.exe")) {
    py -3 -m venv $Venv
}

& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install "paddlepaddle==3.3.1" "paddleocr==3.6.0"
& "$Venv\Scripts\python.exe" "$PSScriptRoot\audit_fast_path_preflight.py"
