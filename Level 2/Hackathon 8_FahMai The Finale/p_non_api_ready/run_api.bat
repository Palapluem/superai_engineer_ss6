@echo off
cd /d %~dp0
set OCR_WORKERS=1
set OCR_QUEUE_MAXSIZE=128
set OCR_JOB_TIMEOUT_SECONDS=900
if "%OCR_DEVICE%"=="" set OCR_DEVICE=gpu:0
if "%OCR_TEMPLATE_PATH%"=="" set OCR_TEMPLATE_PATH=%~dp0submission_template_OCR.csv
if "%FAHMAI_OCR_PIPELINE_DIR%"=="" set FAHMAI_OCR_PIPELINE_DIR=%~dp0ocr_pipeline_runtime
set PYTHON_EXE=python
if exist C:\fahmai_paddle_gpu\Scripts\python.exe set PYTHON_EXE=C:\fahmai_paddle_gpu\Scripts\python.exe
if not exist "%PYTHON_EXE%" if exist C:\fahmai_paddle\Scripts\python.exe set PYTHON_EXE=C:\fahmai_paddle\Scripts\python.exe
"%PYTHON_EXE%" -m uvicorn api:app --host 0.0.0.0 --port 8000
