@echo off
cd /d %~dp0
python test_ocr_endpoint_once.py %*
