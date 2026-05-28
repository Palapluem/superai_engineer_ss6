param(
  [int]$Port = 8000
)

Set-Location $PSScriptRoot

python -m pip install -r .\requirements_ml.txt
python -m uvicorn model2_fastapi_app:app --host 0.0.0.0 --port $Port --reload
