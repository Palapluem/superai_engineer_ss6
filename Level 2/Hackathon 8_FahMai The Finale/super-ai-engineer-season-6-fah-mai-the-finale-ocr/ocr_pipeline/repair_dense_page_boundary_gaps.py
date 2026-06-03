#!/usr/bin/env python3
"""Repair dense-statement page-boundary gaps from public-render OCR values.

Some dense pages end one visual row before the template level ends. The next
page still exposes a readable balance, so derive its amount from the adjacent
visible balances while preserving the absent padding cells as intentional
blanks. This reads and updates local OCR checkpoints only.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "ocr_outputs"


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def decimal_value(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "").replace(" ", ""))
    except (InvalidOperation, ValueError):
        return None


def format_amount(value: Decimal) -> str:
    return f"{value:,.2f}"


def append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def remove_reason(row: dict[str, Any], reason: str) -> None:
    row["fallback_reasons"] = [
        current for current in row.get("fallback_reasons", []) if current != reason
    ]


def field_path(row: dict[str, Any], field: str) -> str:
    return f"L{row['level']}_{row['slot_id']}_{field}"


def repair_checkpoint(record: dict[str, Any]) -> Counter[str]:
    rows = record.get("rows", [])
    expected_blank_fields = set(record.get("expected_blank_fields", []))
    repaired: Counter[str] = Counter()

    for index, row in enumerate(rows):
        if row.get("amount_thb") or not row.get("balance_after_thb"):
            continue
        previous_index = index - 1
        while previous_index >= 0 and not rows[previous_index].get("balance_after_thb"):
            previous_index -= 1
        if previous_index < 0 or previous_index == index - 1:
            continue
        previous = rows[previous_index]
        padding = rows[previous_index + 1 : index]
        if int(row.get("physical_page", 0)) <= int(previous.get("physical_page", 0)):
            continue
        if any(item.get("amount_thb") or item.get("balance_after_thb") for item in padding):
            continue
        previous_balance = decimal_value(previous.get("balance_after_thb"))
        current_balance = decimal_value(row.get("balance_after_thb"))
        if previous_balance is None or current_balance is None:
            continue
        amount = current_balance - previous_balance
        if amount <= 0 or amount >= Decimal("1000000"):
            continue

        row["amount_thb"] = format_amount(amount)
        append_once(row.setdefault("fallback_actions", []), "amount_from_visible_page_boundary_delta")
        remove_reason(row, "missing_balance_delta")
        repaired["amount_thb"] += 1

        for blank_row in padding:
            expected_blank_fields.add(field_path(blank_row, "amount_thb"))
            expected_blank_fields.add(field_path(blank_row, "balance_after_thb"))
            append_once(
                blank_row.setdefault("fallback_actions", []),
                "visual_page_boundary_padding_expected_blank",
            )
            blank_row["fallback_reasons"] = []
            repaired["expected_blank_fields"] += 2

    record["expected_blank_fields"] = sorted(expected_blank_fields)
    record["fallback_rows"] = sum(bool(row.get("fallback_reasons")) for row in rows)
    return repaired


def run(args: argparse.Namespace) -> int:
    checkpoint_dir = args.checkpoint_dir.absolute()
    totals: Counter[str] = Counter()
    artifacts_changed = 0
    for path in sorted(checkpoint_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        repaired = repair_checkpoint(record)
        if not repaired:
            continue
        atomic_write_json(path, record)
        artifacts_changed += 1
        totals.update(repaired)
        print(f"repaired={record.get('artifact_id', path.stem)} values={dict(repaired)}")
    print(f"artifacts_changed={artifacts_changed}")
    print(f"repaired_fields={dict(totals)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fast_dense_bank",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
