#!/usr/bin/env python3
"""Repair compact SCB checkpoint gaps from render-only OCR chains."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_fast_compact_bank_crop_ocr import (
    CHECKPOINT_VERSION,
    decimal_value,
    repair_visible_balance_chain,
)
from run_fast_dense_bank_crop_ocr import DEFAULT_ROOT, atomic_write_json


def normalize_existing_dates(rows: list[dict]) -> int:
    import re

    repaired = 0
    for row in rows:
        if row.get("business_event_date"):
            continue
        match = re.search(r"\b(\d{2})(\d{2})-(\d{2})\b", row.get("date_raw", ""))
        if not match:
            continue
        row["business_event_date"] = "-".join(match.groups())
        row.setdefault("fallback_actions", []).append("date_crop_compact_normalized")
        row["fallback_reasons"] = [
            reason for reason in row.get("fallback_reasons", []) if reason != "invalid_date_crop"
        ]
        repaired += 1
    return repaired


def run(args: argparse.Namespace) -> int:
    repaired_fields: Counter[str] = Counter()
    remaining_reasons: Counter[str] = Counter()
    artifacts = 0
    for path in sorted(args.checkpoint_dir.absolute().glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("layout") != "compact_scb_operating"
            or record.get("checkpoint_version") != CHECKPOINT_VERSION
            or record.get("errors")
        ):
            continue
        artifacts += 1
        rows = record.get("rows", [])
        before = [
            (
                row.get("business_event_date", ""),
                row.get("amount_thb", ""),
                row.get("balance_after_thb", ""),
            )
            for row in rows
        ]
        normalize_existing_dates(rows)
        repair_visible_balance_chain(rows, decimal_value(record.get("opening_balance", "")))
        for previous, row in zip(before, rows):
            current = (
                row.get("business_event_date", ""),
                row.get("amount_thb", ""),
                row.get("balance_after_thb", ""),
            )
            for field, old, new in zip(
                ["business_event_date", "amount_thb", "balance_after_thb"],
                previous,
                current,
            ):
                if not old and new:
                    repaired_fields[field] += 1
        record["fallback_rows"] = sum(bool(row.get("fallback_reasons")) for row in rows)
        remaining_reasons.update(
            reason
            for row in rows
            for reason in row.get("fallback_reasons", [])
        )
        atomic_write_json(path, record)
    audit = {
        "checkpoint_dir": str(args.checkpoint_dir.absolute()),
        "boundary": "Existing public-render OCR checkpoint values only.",
        "artifacts": artifacts,
        "repaired_fields": dict(repaired_fields.most_common()),
        "remaining_reasons": dict(remaining_reasons.most_common()),
    }
    output = args.audit_output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"wrote={output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_compact_bank",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "audits" / "compact_statement_chain_repair.audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
