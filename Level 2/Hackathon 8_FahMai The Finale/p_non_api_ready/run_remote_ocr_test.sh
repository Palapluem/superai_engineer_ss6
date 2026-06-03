#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python test_ocr_endpoint_once.py "$@"
