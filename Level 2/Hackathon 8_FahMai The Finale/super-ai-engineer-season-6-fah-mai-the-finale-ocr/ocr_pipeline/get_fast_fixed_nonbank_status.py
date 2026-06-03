#!/usr/bin/env python3
"""Report fixed-layout non-bank checkpoint coverage and fallbacks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_fast_dense_bank_crop_ocr import DEFAULT_DATA_ROOT, DEFAULT_ROOT
from run_fast_fixed_nonbank_ocr import CHECKPOINT_VERSION, load_template, render_paths


def status(args: argparse.Namespace) -> int:
    schemas = load_template(args.data_root.absolute() / "submission_template_OCR.csv")
    paths = render_paths(args.data_root.absolute() / "fahmai_renders_with_json")
    expected = sorted(set(schemas) & set(paths))
    completed = failed = outdated = 0
    fallback_fields: Counter[str] = Counter()
    unobservable_fields: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    for artifact_id in expected:
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
        by_type[record["artifact_type"]] += 1
        fallback_fields.update(record.get("fallback_fields", []))
        unobservable_fields.update(record.get("unobservable_fields", []))
    report = {
        "artifacts_completed": completed,
        "artifacts_expected": len(expected),
        "artifacts_failed": failed,
        "artifacts_outdated": outdated,
        "progress_percent": round(completed / len(expected) * 100, 2) if expected else 0.0,
        "completed_by_type": dict(sorted(by_type.items())),
        "fallback_fields": dict(fallback_fields.most_common()),
        "unobservable_fields": dict(unobservable_fields.most_common()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed or outdated else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_fixed_nonbank",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(status(parse_args()))
