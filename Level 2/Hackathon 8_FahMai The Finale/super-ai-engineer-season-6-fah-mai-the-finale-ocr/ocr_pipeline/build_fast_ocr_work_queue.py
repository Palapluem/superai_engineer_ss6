#!/usr/bin/env python3
"""Build a render-only OCR work queue for the exam fast path.

The queue is derived from the public submission template and render filenames.
It intentionally avoids provenance JSON and enterprise tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
SHORT_ROOT = Path(r"C:\fahmai_ocr_data")
DEFAULT_DATA_ROOT = SHORT_ROOT if SHORT_ROOT.exists() else DEFAULT_ROOT
EXPECTED_ARTIFACTS = 3750
EXPECTED_RENDERS = 6128


def classify(artifact_id: str) -> tuple[str, str]:
    if artifact_id.startswith("BS-"):
        return "bank_statement", "fixed_layout_crop_ocr"
    if artifact_id.startswith("RC-"):
        return "receipt", "fixed_layout_crop_ocr"
    if artifact_id.startswith("VI-"):
        return "vendor_invoice", "fixed_layout_crop_ocr"
    if artifact_id.startswith("WC-"):
        return "warranty_form", "fixed_layout_crop_ocr"
    if artifact_id.startswith("BN-"):
        return "e7_banner", "parallel_api"
    if artifact_id.startswith("T3-"):
        return "t3_doc", "parallel_api"
    return "t2_doc", "parallel_api"


def bank_family(artifact_id: str) -> str | None:
    if artifact_id.startswith("BS-BBL-"):
        return "bbl"
    if artifact_id.startswith("BS-KBANK-"):
        return "kbank"
    if artifact_id.startswith("BS-"):
        return "branch_statement"
    return None


def artifact_id_from_render(artifact_type: str, path: Path) -> str:
    stem = path.stem
    if artifact_type == "bank_statement":
        stem = re.sub(r"_(?:header|transactions_p\d+)$", "", stem)
    return stem


def load_schema_counts(template: Path) -> dict[str, int]:
    csv.field_size_limit(2_147_483_647)
    counts: dict[str, int] = {}
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            artifact_id = row["artifact_id"]
            pred_json = json.loads(row["pred_json"])
            counts[artifact_id] = len(pred_json)
    return counts


def collect_renders(bundle: Path) -> dict[str, list[Path]]:
    by_artifact: dict[str, list[Path]] = defaultdict(list)
    renders = bundle / "renders"
    for artifact_type_dir in sorted(path for path in renders.iterdir() if path.is_dir()):
        artifact_type = artifact_type_dir.name
        for path in sorted(item for item in artifact_type_dir.rglob("*") if item.is_file()):
            if path.suffix.lower() not in {".png", ".pdf"}:
                continue
            artifact_id = artifact_id_from_render(artifact_type, path)
            by_artifact[artifact_id].append(path)
    return dict(by_artifact)


def build_queue(args: argparse.Namespace) -> int:
    data_root = args.data_root.absolute()
    bundle = data_root / "fahmai_renders_with_json"
    schema_counts = load_schema_counts(data_root / "submission_template_OCR.csv")
    renders = collect_renders(bundle)

    missing_renders = sorted(set(schema_counts) - set(renders))
    extra_renders = sorted(set(renders) - set(schema_counts))
    queue: list[dict[str, Any]] = []
    for artifact_id in sorted(schema_counts):
        artifact_type, lane = classify(artifact_id)
        paths = renders.get(artifact_id, [])
        queue.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "lane": lane,
                "requested_fields": schema_counts[artifact_id],
                "renders": [str(path.relative_to(data_root)) for path in paths],
            }
        )

    lane_artifacts = Counter(item["lane"] for item in queue)
    lane_fields = Counter()
    lane_renders = Counter()
    type_artifacts = Counter()
    type_fields = Counter()
    type_renders = Counter()
    bank_family_artifacts = Counter()
    bank_family_renders = Counter()
    for item in queue:
        lane_fields[item["lane"]] += item["requested_fields"]
        lane_renders[item["lane"]] += len(item["renders"])
        type_artifacts[item["artifact_type"]] += 1
        type_fields[item["artifact_type"]] += item["requested_fields"]
        type_renders[item["artifact_type"]] += len(item["renders"])
        family = bank_family(item["artifact_id"])
        if family:
            bank_family_artifacts[family] += 1
            bank_family_renders[family] += len(item["renders"])

    total_renders = sum(len(item["renders"]) for item in queue)
    passed = (
        len(queue) == EXPECTED_ARTIFACTS
        and total_renders == EXPECTED_RENDERS
        and not missing_renders
        and not extra_renders
    )
    summary = {
        "passed": passed,
        "data_root": str(data_root),
        "artifacts": len(queue),
        "renders": total_renders,
        "requested_fields": sum(schema_counts.values()),
        "lanes": {
            lane: {
                "artifacts": lane_artifacts[lane],
                "renders": lane_renders[lane],
                "requested_fields": lane_fields[lane],
            }
            for lane in sorted(lane_artifacts)
        },
        "artifact_types": {
            artifact_type: {
                "artifacts": type_artifacts[artifact_type],
                "renders": type_renders[artifact_type],
                "requested_fields": type_fields[artifact_type],
            }
            for artifact_type in sorted(type_artifacts)
        },
        "bank_statement_families": {
            family: {
                "artifacts": bank_family_artifacts[family],
                "renders": bank_family_renders[family],
            }
            for family in sorted(bank_family_artifacts)
        },
        "missing_render_artifacts": missing_renders[: args.max_examples],
        "extra_render_artifacts": extra_renders[: args.max_examples],
    }

    args.queue_output.parent.mkdir(parents=True, exist_ok=True)
    with args.queue_output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in queue:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote={args.queue_output}")
    print(f"wrote={args.summary_output}")
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_path_work_queue.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_path_work_queue_summary.json",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build_queue(parse_args()))
