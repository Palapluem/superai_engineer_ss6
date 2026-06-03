#!/usr/bin/env python3
"""Repair deterministic invoice crop OCR formatting artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_fast_dense_bank_crop_ocr import DEFAULT_ROOT, atomic_write_json
from run_fast_fixed_nonbank_ocr import parse_money, parse_payment_id


def run(args: argparse.Namespace) -> int:
    checkpoint_dir = args.checkpoint_dir.absolute()
    repaired: Counter[str] = Counter()
    unresolved: Counter[str] = Counter()
    for path in sorted(checkpoint_dir.glob("VI-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        prediction = record.get("prediction", {})
        raw = record.get("raw_crops", {})
        if not prediction.get("paid_amount_thb"):
            value = parse_money(raw.get("paid_amount", {}).get("text", ""))
            if value:
                prediction["paid_amount_thb"] = value
                repaired["paid_amount_thb"] += 1
        if not prediction.get("payment_id"):
            public_month = Path(record["source_render_path"]).parent.name.replace("-", "")
            value = parse_payment_id(
                [
                    raw.get("payment_id", {}).get("text", ""),
                    raw.get("payment_id_retry", {}).get("text", ""),
                ],
                public_month,
            )
            if value:
                prediction["payment_id"] = value
                repaired["payment_id"] += 1
        record["fallback_fields"] = [
            key for key, value in prediction.items() if value == ""
        ]
        record["fallback_count"] = len(record["fallback_fields"])
        record.setdefault("fallback_actions", []).append("invoice_checkpoint_format_repair")
        unresolved.update(record["fallback_fields"])
        atomic_write_json(path, record)
    audit = {
        "checkpoint_dir": str(checkpoint_dir),
        "boundary": "Existing render-only invoice crop checkpoints only.",
        "repaired_fields": dict(repaired.most_common()),
        "unresolved_fields": dict(unresolved.most_common()),
    }
    output = args.audit_output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    print(f"wrote={output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_fixed_nonbank",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "audits" / "invoice_checkpoint_format_repair.audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
