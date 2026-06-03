#!/usr/bin/env python3
"""Report checkpoint progress for the compact BBL operating-statement lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_fast_bbl_bank_crop_ocr import CHECKPOINT_VERSION, bbl_render_groups
from run_fast_compact_bank_crop_ocr import flatten_slots
from run_fast_dense_bank_crop_ocr import DEFAULT_DATA_ROOT, DEFAULT_ROOT, load_bank_schemas


def status(args: argparse.Namespace) -> int:
    data_root = args.data_root.absolute()
    schemas = load_bank_schemas(data_root / "submission_template_OCR.csv")
    groups = bbl_render_groups(data_root / "fahmai_renders_with_json")
    expected_ids = sorted(set(schemas) & set(groups))
    expected_rows = sum(len(flatten_slots(schemas[artifact_id])) for artifact_id in expected_ids)
    completed = failed = outdated = rows = fallback_rows = 0
    elapsed_seconds = 0.0
    for artifact_id in expected_ids:
        path = args.checkpoint_dir.absolute() / f"{artifact_id}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("checkpoint_version") != CHECKPOINT_VERSION:
            outdated += 1
            continue
        if record.get("errors"):
            failed += 1
            continue
        completed += 1
        rows += len(record.get("rows", []))
        fallback_rows += int(record.get("fallback_rows", 0))
        elapsed_seconds += float(record.get("elapsed_seconds", 0.0))
    rate = rows / elapsed_seconds if elapsed_seconds else 0.0
    remaining_rows = max(expected_rows - rows, 0)
    report = {
        "artifacts_completed": completed,
        "artifacts_expected": len(expected_ids),
        "artifacts_failed": failed,
        "artifacts_outdated": outdated,
        "rows_completed": rows,
        "rows_expected": expected_rows,
        "progress_percent": round(rows / expected_rows * 100, 2) if expected_rows else 0.0,
        "fallback_rows": fallback_rows,
        "elapsed_checkpoint_minutes": round(elapsed_seconds / 60, 2),
        "observed_rows_per_second": round(rate, 3),
        "estimated_remaining_minutes": round(remaining_rows / rate / 60, 2) if rate else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_bbl_bank",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(status(parse_args()))
