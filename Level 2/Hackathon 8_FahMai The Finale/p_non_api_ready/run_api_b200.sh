#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OCR_WORKERS="${OCR_WORKERS:-1}"
export OCR_QUEUE_MAXSIZE="${OCR_QUEUE_MAXSIZE:-128}"
export OCR_JOB_TIMEOUT_SECONDS="${OCR_JOB_TIMEOUT_SECONDS:-900}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OCR_DEVICE="${OCR_DEVICE:-gpu:0}"
export OCR_TEMPLATE_PATH="${OCR_TEMPLATE_PATH:-$PWD/submission_template_OCR.csv}"
export FAHMAI_OCR_PIPELINE_DIR="${FAHMAI_OCR_PIPELINE_DIR:-$PWD/ocr_pipeline_runtime}"

exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8009}"
