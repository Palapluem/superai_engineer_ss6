#!/usr/bin/env python3
"""Report fast general-document parser coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_fast_dense_bank_crop_ocr import DEFAULT_DATA_ROOT, DEFAULT_ROOT
from run_fast_general_document_parser import CHECKPOINT_VERSION, load_template, render_paths


def status(args: argparse.Namespace) -> int:
    schemas = load_template(args.data_root.absolute() / "submission_template_OCR.csv")
    paths = render_paths(args.data_root.absolute() / "fahmai_renders_with_json")
    expected = sorted(set(schemas) & set(paths))
    completed = failed = outdated = 0
    fallback_fields: Counter[str] = Counter()
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
        fallback_fields.update(record.get("fallback_fields", []))
    print(
        json.dumps(
            {
                "artifacts_completed": completed,
                "artifacts_expected": len(expected),
                "artifacts_failed": failed,
                "artifacts_outdated": outdated,
                "progress_percent": round(completed / len(expected) * 100, 2) if expected else 0.0,
                "fallback_fields": dict(fallback_fields.most_common()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed or outdated else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_general_documents",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(status(parse_args()))
