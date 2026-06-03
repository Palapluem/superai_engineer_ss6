#!/usr/bin/env python3
"""FahMai OCR API adapter.

The FastAPI service writes each request into a temporary folder and calls this
module through the legacy `main()` contract. This adapter reuses the production
FahMai bank-statement crop OCR scripts and emits both the old CSV debug files
and `answers.json`, whose values match the Kaggle `pred_json` object format.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OCR_ROOT = Path(
    os.environ.get(
        "FAHMAI_OCR_ROOT",
        PROJECT_DIR / "super-ai-engineer-season-6-fah-mai-the-finale-ocr",
    )
)
LOCAL_PIPELINE_DIR = SCRIPT_DIR / "ocr_pipeline_runtime"
LOCAL_TEMPLATE_PATH = SCRIPT_DIR / "submission_template_OCR.csv"
OCR_PIPELINE_DIR = Path(
    os.environ.get(
        "FAHMAI_OCR_PIPELINE_DIR",
        LOCAL_PIPELINE_DIR if LOCAL_PIPELINE_DIR.exists() else OCR_ROOT / "ocr_pipeline",
    )
)
OCR_DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", OCR_ROOT))
OCR_TEMPLATE_PATH = Path(
    os.environ.get(
        "OCR_TEMPLATE_PATH",
        LOCAL_TEMPLATE_PATH if LOCAL_TEMPLATE_PATH.exists() else OCR_DATA_ROOT / "submission_template_OCR.csv",
    )
)

INPUT_DIR = Path(os.environ.get("OCR_INPUT_DIR", SCRIPT_DIR / "input"))
OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR", SCRIPT_DIR / "runs" / "api"))
CROP_DIR = OUTPUT_DIR / "debug_crops"
BY_BANK_DIR = OUTPUT_DIR / "by_bank"
BY_ARTIFACT_DIR = OUTPUT_DIR / "by_artifact"
TRANSACTIONS_CSV = OUTPUT_DIR / "transactions_canonical.csv"
HEADERS_CSV = OUTPUT_DIR / "headers.csv"
OCR_DEBUG_CSV = OUTPUT_DIR / "ocr_debug.csv"
ARTIFACT_SCHEMA_CSV = OUTPUT_DIR / "artifact_schema_summary.csv"
META_JSON = OUTPUT_DIR / "run_meta.json"
ANSWERS_JSON = OUTPUT_DIR / "answers.json"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RENDER_NAME = re.compile(r"(.+?)_(header|transactions_p(\d+))$")
DEFAULT_DEVICE = os.environ.get("OCR_DEVICE", "gpu:0")
DEFAULT_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "64"))
DEFAULT_CPU_THREADS = int(os.environ.get("OCR_CPU_THREADS", "10"))
DEFAULT_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.75"))
DISABLE_MKLDNN = os.environ.get("OCR_DISABLE_MKLDNN", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

_TEXT_RECOGNITION_MODEL: Any | None = None
_PIPELINE_IMPORTS: dict[str, Any] | None = None


def _ensure_pipeline_imports() -> dict[str, Any]:
    global _PIPELINE_IMPORTS
    if _PIPELINE_IMPORTS is not None:
        return _PIPELINE_IMPORTS
    if not OCR_PIPELINE_DIR.exists():
        raise RuntimeError(f"OCR pipeline folder not found: {OCR_PIPELINE_DIR}")
    sys.path.insert(0, str(OCR_PIPELINE_DIR))

    import build_fast_partial_submission as formatter
    import run_fast_bbl_bank_crop_ocr as bbl_bank
    import run_fast_compact_bank_crop_ocr as compact_bank
    import run_fast_dense_bank_crop_ocr as dense_bank
    import run_fast_sparse_bank_crop_ocr as sparse_bank

    _PIPELINE_IMPORTS = {
        "formatter": formatter,
        "dense": dense_bank,
        "compact": compact_bank,
        "bbl": bbl_bank,
        "sparse": sparse_bank,
    }
    return _PIPELINE_IMPORTS


def _recognition_model() -> Any:
    global _TEXT_RECOGNITION_MODEL
    if _TEXT_RECOGNITION_MODEL is not None:
        return _TEXT_RECOGNITION_MODEL
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    _TEXT_RECOGNITION_MODEL = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        device=DEFAULT_DEVICE,
        enable_mkldnn=not DISABLE_MKLDNN,
        cpu_threads=DEFAULT_CPU_THREADS,
    )
    return _TEXT_RECOGNITION_MODEL


def _natural_key(value: str) -> list[Any]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value)]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(2_147_483_647)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_bank_schemas(path: Path = OCR_TEMPLATE_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"OCR template file not found: {path}. "
            "Copy submission_template_OCR.csv next to p_non_api_ready or set OCR_TEMPLATE_PATH."
        )
    schemas: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(path):
        artifact_id = row.get("artifact_id", "")
        if artifact_id.startswith("BS-"):
            schemas[artifact_id] = json.loads(row.get("pred_json", "{}"))
    if not schemas:
        raise RuntimeError(f"OCR template contains no bank-statement schemas: {path}")
    return schemas


def _input_render_groups(input_dir: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.iterdir(), key=lambda item: _natural_key(item.name)):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = RENDER_NAME.fullmatch(path.stem)
        if match:
            artifact_id, page_kind, page_number = match.groups()
            item = groups.setdefault(artifact_id, {"header": None, "transactions": {}})
            if page_kind == "header":
                item["header"] = path
            else:
                item["transactions"][int(page_number)] = path
            continue

        artifact_id = path.stem
        item = groups.setdefault(artifact_id, {"header": None, "transactions": {}})
        item["transactions"][len(item["transactions"]) + 1] = path

    return {
        artifact_id: item
        for artifact_id, item in groups.items()
        if item.get("transactions")
    }


def _dark_pixels(path: Path, y: int, x_start: int = 60, x_stop: int = 1180) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        return sum(grayscale.getpixel((x, y)) < 100 for x in range(x_start, x_stop))


def _first_transaction_path(renders: dict[str, Any]) -> Path:
    transactions = renders.get("transactions", {})
    if not transactions:
        raise RuntimeError("No transaction page was provided.")
    return transactions[min(transactions)]


def _detect_layout(artifact_id: str, renders: dict[str, Any]) -> str:
    first_page = _first_transaction_path(renders)
    with Image.open(first_page) as image:
        size = image.size

    if size == (2968, 2850):
        return "dense"
    if artifact_id.startswith(("BS-SCB-OPER-", "BS-BBL-OPER-")) and len(renders["transactions"]) == 1:
        return "sparse"
    if size == (1240, 1234):
        dark_top = _dark_pixels(first_page, 43)
        dark_middle = _dark_pixels(first_page, 135)
        if dark_middle > 800 and dark_middle >= dark_top:
            return "bbl"
        if dark_top > 800:
            return "compact"
    raise RuntimeError(f"Unsupported bank-statement layout for {artifact_id}: first_page_size={size}")


def _extract_record(
    artifact_id: str,
    renders: dict[str, Any],
    schema: dict[str, str],
) -> dict[str, Any]:
    modules = _ensure_pipeline_imports()
    model = _recognition_model()
    layout = _detect_layout(artifact_id, renders)
    if layout == "dense":
        return modules["dense"].extract_artifact(
            model=model,
            header_model=model,
            artifact_id=artifact_id,
            renders=renders,
            schema=schema,
            batch_size=DEFAULT_BATCH_SIZE,
            min_confidence=DEFAULT_MIN_CONFIDENCE,
        )
    if layout == "compact":
        return modules["compact"].extract_artifact(
            model=model,
            header_model=model,
            artifact_id=artifact_id,
            renders=renders,
            schema=schema,
            batch_size=DEFAULT_BATCH_SIZE,
            min_confidence=DEFAULT_MIN_CONFIDENCE,
        )
    if layout == "bbl":
        return modules["bbl"].extract_artifact(
            model=model,
            artifact_id=artifact_id,
            renders=renders,
            schema=schema,
            batch_size=DEFAULT_BATCH_SIZE,
        )
    if layout == "sparse":
        return modules["sparse"].extract_artifact(
            model=model,
            artifact_id=artifact_id,
            renders=renders,
            schema=schema,
            batch_size=DEFAULT_BATCH_SIZE,
        )
    raise RuntimeError(f"Unsupported OCR layout: {layout}")


def _prediction_from_record(
    artifact_id: str,
    schema: dict[str, str],
    record: dict[str, Any],
) -> dict[str, str]:
    modules = _ensure_pipeline_imports()
    prediction = dict(schema)
    if schema and not record.get("errors"):
        modules["formatter"].fill_checkpoint(prediction, record)
    if artifact_id.startswith("BS-"):
        modules["formatter"].normalize_bank_business_dates(prediction)
    return {key: "" if value is None else str(value) for key, value in prediction.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _transaction_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        artifact_id = str(record.get("artifact_id", ""))
        for row in record.get("rows", []):
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "bank": record.get("layout", record.get("engine", "")),
                    "family": record.get("layout", ""),
                    "account_id": row.get("account_id", record.get("account_id", "")),
                    "account_number": record.get("account_number", ""),
                    "account_role": "SAVINGS",
                    "currency": "THB",
                    "page": row.get("physical_page", row.get("level", "")),
                    "row_index": row.get("row_index", ""),
                    "business_event_date": row.get("business_event_date", ""),
                    "effective_time": "",
                    "transaction_type": row.get("transaction_type", ""),
                    "direction": row.get("direction", ""),
                    "amount_thb": row.get("amount_thb", ""),
                    "balance_after_thb": row.get("balance_after_thb", ""),
                    "channel": row.get("channel", ""),
                    "description": row.get("description", ""),
                    "date_raw": row.get("date_raw", ""),
                    "amount_raw": row.get("amount_raw", ""),
                    "balance_raw": row.get(
                        "balance_raw",
                        row.get("visual_balance_raw", ""),
                    ),
                    "description_raw": row.get("description_raw", ""),
                    "parse_status": "ok" if not row.get("fallback_reasons") else "fallback",
                }
            )
    return rows


def _header_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": record.get("artifact_id", ""),
            "bank": record.get("layout", ""),
            "family": record.get("layout", ""),
            "account_id": record.get("account_id", ""),
            "account_number": record.get("account_number", ""),
            "account_number_raw": record.get("account_number_raw", ""),
            "account_number_score": record.get("account_number_score", ""),
            "account_role": "SAVINGS",
            "currency": "THB",
        }
        for record in records
    ]


def _schema_rows(answers: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"artifact_id": artifact_id, "field": field, "value": value}
        for artifact_id, prediction in answers.items()
        for field, value in prediction.items()
    ]


def _debug_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        errors = record.get("errors", [])
        rows.append(
            {
                "artifact_id": record.get("artifact_id", ""),
                "engine": record.get("engine", "paddle_crop_recognition"),
                "layout": record.get("layout", ""),
                "elapsed_seconds": record.get("elapsed_seconds", ""),
                "rows": len(record.get("rows", [])),
                "fallback_rows": record.get("fallback_rows", 0),
                "errors": json.dumps(errors, ensure_ascii=False),
            }
        )
    return rows


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    BY_BANK_DIR.mkdir(parents=True, exist_ok=True)
    BY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    schemas = _load_bank_schemas()
    groups = _input_render_groups(Path(INPUT_DIR))
    records: list[dict[str, Any]] = []
    answers: dict[str, dict[str, str]] = {}

    for artifact_id, renders in sorted(groups.items(), key=lambda item: _natural_key(item[0])):
        schema = schemas.get(artifact_id)
        if not schema:
            record = {
                "artifact_id": artifact_id,
                "engine": "paddle_crop_recognition",
                "rows": [],
                "errors": [f"schema_not_found: {artifact_id}"],
            }
            prediction: dict[str, str] = {}
        else:
            try:
                record = _extract_record(artifact_id, renders, schema)
            except Exception as exc:
                record = {
                    "artifact_id": artifact_id,
                    "engine": "paddle_crop_recognition",
                    "rows": [],
                    "errors": [str(exc)],
                }
            prediction = _prediction_from_record(artifact_id, schema, record)
        records.append(record)
        answers[artifact_id] = prediction

        artifact_output = BY_ARTIFACT_DIR / f"{artifact_id}.json"
        artifact_output.write_text(
            json.dumps({"record": record, "answer": prediction}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _write_csv(
        TRANSACTIONS_CSV,
        _transaction_rows(records),
        [
            "artifact_id",
            "bank",
            "family",
            "account_id",
            "account_number",
            "account_role",
            "currency",
            "page",
            "row_index",
            "business_event_date",
            "effective_time",
            "transaction_type",
            "direction",
            "amount_thb",
            "balance_after_thb",
            "channel",
            "description",
            "date_raw",
            "amount_raw",
            "balance_raw",
            "description_raw",
            "parse_status",
        ],
    )
    _write_csv(
        HEADERS_CSV,
        _header_rows(records),
        [
            "artifact_id",
            "bank",
            "family",
            "account_id",
            "account_number",
            "account_number_raw",
            "account_number_score",
            "account_role",
            "currency",
        ],
    )
    _write_csv(ARTIFACT_SCHEMA_CSV, _schema_rows(answers), ["artifact_id", "field", "value"])
    _write_csv(
        OCR_DEBUG_CSV,
        _debug_rows(records),
        ["artifact_id", "engine", "layout", "elapsed_seconds", "rows", "fallback_rows", "errors"],
    )

    ANSWERS_JSON.write_text(
        json.dumps({"answers": answers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    META_JSON.write_text(
        json.dumps(
            {
                "input_dir": str(Path(INPUT_DIR)),
                "output_dir": str(Path(OUTPUT_DIR)),
                "template_path": str(OCR_TEMPLATE_PATH),
                "device": DEFAULT_DEVICE,
                "artifact_count": len(records),
                "artifacts": [record.get("artifact_id", "") for record in records],
                "errors": {
                    str(record.get("artifact_id", "")): record.get("errors", [])
                    for record in records
                    if record.get("errors")
                },
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
