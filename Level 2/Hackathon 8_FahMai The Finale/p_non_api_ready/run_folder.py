from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_service import run_ocr_on_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bank statement OCR on a local folder.")
    parser.add_argument("--input-dir", required=True, help="Folder containing *_header.png and *_transactions_pN.png files.")
    parser.add_argument("--output-dir", default="", help="Folder for CSV outputs. Defaults to p_non_api_ready/runs/run_xxx.")
    args = parser.parse_args()

    result = run_ocr_on_folder(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps({"meta": result["meta"], "output_dir": result["output_dir"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

