$ErrorActionPreference = "Stop"

$GpuEnv = "C:\fahmai_paddle_gpu"
$GpuPython = Join-Path $GpuEnv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $GpuPython)) {
    & py -3.12 -m venv $GpuEnv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create $GpuEnv"
    }
}

& $GpuPython -m pip install `
    "paddlepaddle-gpu==3.3.1" `
    "paddleocr==3.6.0" `
    -i "https://www.paddlepaddle.org.cn/packages/stable/cu126/" `
    --extra-index-url "https://pypi.org/simple"
if ($LASTEXITCODE -ne 0) {
    throw "Paddle GPU package installation failed."
}

# The cu126 Paddle 3.3.1 wheel reports that it was compiled with cuDNN 9.9.
# Align the runtime package explicitly to avoid the cuDNN compatibility warning.
& $GpuPython -m pip install --no-deps --upgrade "nvidia-cudnn-cu12==9.9.0.52"
if ($LASTEXITCODE -ne 0) {
    throw "cuDNN runtime alignment failed."
}

& $GpuPython "$PSScriptRoot\check_paddle_gpu_ready.py"
if ($LASTEXITCODE -ne 0) {
    throw "Paddle GPU readiness check failed."
}

Write-Host "Paddle GPU fast path is ready."
