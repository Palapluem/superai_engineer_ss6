#!/usr/bin/env python3
"""Validate submission shape and JSON syntax against the canonical template."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
csv.field_size_limit(2**31 - 1)
TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    template_rows = read_rows(root / "submission_template_OCR.csv")
    submission_rows = read_rows(args.submission.resolve())
    expected_ids = [row["artifact_id"] for row in template_rows]
    actual_ids = [row["artifact_id"] for row in submission_rows]

    problems: list[str] = []
    if actual_ids != expected_ids:
        problems.append("artifact_id sequence differs from submission_template_OCR.csv")
    empty = 0
    invalid = 0
    schema_mismatch = 0
    missing_keys = 0
    extra_keys = 0
    bs_business_event_dates_with_time = 0
    total_fields = 0
    template_by_id = {
        row["artifact_id"]: json.loads(row["pred_json"])
        for row in template_rows
    }
    for row in submission_rows:
        value = row.get("pred_json", "")
        if not value:
            empty += 1
            continue
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                invalid += 1
            else:
                total_fields += len(parsed)
                expected = template_by_id.get(row["artifact_id"])
                if expected is not None and set(parsed) != set(expected):
                    schema_mismatch += 1
                    missing_keys += len(set(expected) - set(parsed))
                    extra_keys += len(set(parsed) - set(expected))
                if row["artifact_id"].startswith("BS-"):
                    bs_business_event_dates_with_time += sum(
                        bool(TIME_PATTERN.search(str(field_value)))
                        for key, field_value in parsed.items()
                        if key.endswith("_business_event_date") and field_value
                    )
        except json.JSONDecodeError:
            invalid += 1
    if empty:
        problems.append(f"empty pred_json rows={empty}")
    if invalid:
        problems.append(f"invalid JSON rows={invalid}")
    if schema_mismatch:
        problems.append(
            f"schema mismatch rows={schema_mismatch} missing_keys={missing_keys} extra_keys={extra_keys}"
        )
    if bs_business_event_dates_with_time:
        problems.append(
            "BS business_event_date values must not contain time "
            f"values={bs_business_event_dates_with_time}"
        )

    print(f"rows={len(submission_rows)} expected_rows={len(expected_ids)}")
    print(
        f"empty={empty} invalid_json={invalid} schema_mismatch={schema_mismatch} "
        f"missing_keys={missing_keys} extra_keys={extra_keys} total_fields={total_fields} "
        f"bs_business_event_dates_with_time={bs_business_event_dates_with_time}"
    )
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print("validation=PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(validate(parse_args()))
