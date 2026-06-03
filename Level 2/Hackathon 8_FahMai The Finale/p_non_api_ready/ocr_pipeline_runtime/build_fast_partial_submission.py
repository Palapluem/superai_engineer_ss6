#!/usr/bin/env python3
"""Build a schema-valid partial submission from render-only fast checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
SHORT_ROOT = Path(r"C:\fahmai_ocr_data")
DEFAULT_DATA_ROOT = SHORT_ROOT if SHORT_ROOT.exists() else DEFAULT_ROOT
TX_FIELDS = [
    "business_event_date",
    "transaction_type",
    "amount_thb",
    "balance_after_thb",
    "description",
    "account_id",
]
CHECKPOINT_VERSIONS = {
    "compact_scb_operating": 5,
    "compact_bbl_operating": 1,
    "sparse_scb_direct": 2,
    "sparse_bbl_direct": 2,
    "fixed_receipt": 1,
    "fixed_vendor_invoice": 1,
    "fixed_warranty_form": 1,
    "fast_general_document": 1,
}
TIME_SUFFIX = re.compile(r"[ T]\d{1,2}:\d{2}(?::\d{2})?\s*$")


def load_template(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fill_dense_bank(prediction: dict[str, str], checkpoint: dict) -> None:
    account_id = checkpoint["account_id"]
    layout = checkpoint.get("layout")
    is_scb = layout in {"compact_scb_operating", "sparse_scb_direct"}
    is_bbl = layout in {"compact_bbl_operating", "sparse_bbl_direct"}
    header = {
        "L0_account_id": account_id,
        "L0_bank": "BBL" if is_bbl else "SCB" if is_scb else "KBANK",
        "L0_account_number": checkpoint.get("account_number", ""),
        "L0_account_role": (
            "SAVINGS"
            if is_scb or is_bbl
            else "SAVING"
            if account_id.startswith("KBANK-")
            else "SAVINGS"
        ),
        "L0_currency": "THB",
    }
    for key, value in header.items():
        if key in prediction:
            prediction[key] = value
    for row in checkpoint["rows"]:
        for field in TX_FIELDS:
            key = f"L{row['level']}_{row['slot_id']}_{field}"
            if key in prediction:
                prediction[key] = row.get(field, "")
    for key, value in checkpoint.get("generic_values", {}).items():
        if key in prediction:
            prediction[key] = value


def is_current_checkpoint(checkpoint: dict) -> bool:
    expected = CHECKPOINT_VERSIONS.get(checkpoint.get("layout"))
    return expected is None or checkpoint.get("checkpoint_version") == expected


def fill_checkpoint(prediction: dict[str, str], checkpoint: dict) -> None:
    if "prediction" in checkpoint:
        for key, value in checkpoint.get("prediction", {}).items():
            if key in prediction:
                prediction[key] = value
        return
    fill_dense_bank(prediction, checkpoint)


def normalize_bank_business_dates(prediction: dict[str, str]) -> int:
    normalized = 0
    for key, value in prediction.items():
        if not key.endswith("_business_event_date") or not value:
            continue
        date_only = TIME_SUFFIX.sub("", str(value)).strip()
        if date_only != value:
            prediction[key] = date_only
            normalized += 1
    return normalized


def build(args: argparse.Namespace) -> int:
    data_root = args.data_root.absolute()
    rows = load_template(data_root / "submission_template_OCR.csv")
    checkpoints = {}
    for checkpoint_dir in args.checkpoint_dirs:
        checkpoints.update(
            {
                path.stem: json.loads(path.read_text(encoding="utf-8"))
                for path in checkpoint_dir.absolute().glob("*.json")
            }
        )
    output_rows: list[dict[str, str]] = []
    filled_by_prefix: Counter[str] = Counter()
    total_by_prefix: Counter[str] = Counter()
    checkpoint_rows = fallback_rows = used_checkpoints = normalized_bank_dates = 0
    for template_row in rows:
        artifact_id = template_row["artifact_id"]
        prediction = json.loads(template_row["pred_json"])
        checkpoint = checkpoints.get(artifact_id)
        if checkpoint and not checkpoint.get("errors") and is_current_checkpoint(checkpoint):
            fill_checkpoint(prediction, checkpoint)
            used_checkpoints += 1
            checkpoint_rows += len(checkpoint.get("rows", []))
            fallback_rows += int(checkpoint.get("fallback_rows", checkpoint.get("fallback_count", 0)))
        if artifact_id.startswith("BS-"):
            normalized_bank_dates += normalize_bank_business_dates(prediction)
        prefix = artifact_id.split("-", 1)[0]
        total_by_prefix[prefix] += len(prediction)
        filled_by_prefix[prefix] += sum(value != "" for value in prediction.values())
        output_rows.append(
            {
                "artifact_id": artifact_id,
                "pred_json": json.dumps(prediction, ensure_ascii=False),
            }
        )

    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact_id", "pred_json"])
        writer.writeheader()
        writer.writerows(output_rows)
    audit = {
        "output": str(output),
        "artifacts": len(output_rows),
        "checkpoints_found": len(checkpoints),
        "checkpoints_used": used_checkpoints,
        "checkpoint_rows": checkpoint_rows,
        "fallback_rows": fallback_rows,
        "normalized_bank_business_event_dates": normalized_bank_dates,
        "filled_fields_by_prefix": dict(sorted(filled_by_prefix.items())),
        "total_fields_by_prefix": dict(sorted(total_by_prefix.items())),
    }
    args.audit_output.absolute().write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint-dirs",
        type=Path,
        nargs="+",
        default=[
            DEFAULT_ROOT / "ocr_outputs" / "fast_dense_bank",
            DEFAULT_ROOT / "ocr_outputs" / "fast_compact_bank",
            DEFAULT_ROOT / "ocr_outputs" / "fast_bbl_bank",
            DEFAULT_ROOT / "ocr_outputs" / "fast_sparse_bank",
            DEFAULT_ROOT / "ocr_outputs" / "fast_fixed_nonbank",
            DEFAULT_ROOT / "ocr_outputs" / "fast_general_documents",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "submission_OCR_fast_partial.csv",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "submission_OCR_fast_partial.audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
