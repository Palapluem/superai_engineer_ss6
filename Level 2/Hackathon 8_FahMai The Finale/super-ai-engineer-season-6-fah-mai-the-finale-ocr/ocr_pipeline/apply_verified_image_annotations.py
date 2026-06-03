#!/usr/bin/env python3
"""Apply only human-verified, image-grounded OCR corrections to a candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


csv.field_size_limit(2_147_483_647)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
SHORT_ROOT = Path(r"C:\fahmai_ocr_data")
DEFAULT_DATA_ROOT = SHORT_ROOT if SHORT_ROOT.exists() else DEFAULT_ROOT
VERIFIED_STATUS = "verified_from_image"
REQUIRED_COLUMNS = {
    "artifact_id",
    "field_path",
    "verified_value",
    "verification_status",
    "reviewer",
    "source_render_path",
    "notes",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_submission(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    rows = read_csv(path)
    return (
        [row["artifact_id"] for row in rows],
        {row["artifact_id"]: json.loads(row["pred_json"]) for row in rows},
    )


def validate_render_path(data_root: Path, relative_path: str) -> Path:
    render_root = (data_root / "fahmai_renders_with_json" / "renders").resolve()
    candidate = (data_root / relative_path).resolve()
    try:
        candidate.relative_to(render_root)
    except ValueError as exc:
        raise ValueError(f"source_render_path is outside public renders: {relative_path}") from exc
    if not candidate.is_file():
        raise ValueError(f"source_render_path does not exist: {relative_path}")
    return candidate


def apply(args: argparse.Namespace) -> int:
    data_root = args.data_root.absolute()
    template_order, template = load_submission(data_root / "submission_template_OCR.csv")
    base_order, candidate = load_submission(args.base_submission.absolute())
    if base_order != template_order:
        raise SystemExit("Base submission artifact order differs from public template.")
    annotations = read_csv(args.annotations.absolute())
    if annotations and not REQUIRED_COLUMNS.issubset(annotations[0]):
        missing = sorted(REQUIRED_COLUMNS - set(annotations[0]))
        raise SystemExit(f"Annotation CSV is missing columns: {missing}")

    applied: dict[tuple[str, str], dict[str, str]] = {}
    ignored_status = 0
    errors: list[str] = []
    for index, row in enumerate(annotations, start=2):
        if row.get("verification_status", "").strip() != VERIFIED_STATUS:
            ignored_status += 1
            continue
        artifact_id = row.get("artifact_id", "").strip()
        field_path = row.get("field_path", "").strip()
        reviewer = row.get("reviewer", "").strip()
        source_render_path = row.get("source_render_path", "").strip()
        if artifact_id not in template:
            errors.append(f"line {index}: unknown artifact_id={artifact_id!r}")
            continue
        if field_path not in template[artifact_id]:
            errors.append(f"line {index}: unknown field_path={field_path!r} artifact_id={artifact_id!r}")
            continue
        if not reviewer:
            errors.append(f"line {index}: reviewer is required")
            continue
        if not args.skip_render_path_check:
            try:
                validate_render_path(data_root, source_render_path)
            except ValueError as exc:
                errors.append(f"line {index}: {exc}")
                continue
        key = (artifact_id, field_path)
        if key in applied and applied[key]["verified_value"] != row.get("verified_value", ""):
            errors.append(f"line {index}: conflicting verified values for {artifact_id}.{field_path}")
            continue
        applied[key] = row

    if errors:
        raise SystemExit("Annotation validation failed:\n- " + "\n- ".join(errors[:50]))

    changed = unchanged = intentional_blank = 0
    by_prefix: dict[str, int] = {}
    for (artifact_id, field_path), row in applied.items():
        value = row.get("verified_value", "")
        previous = str(candidate[artifact_id].get(field_path, ""))
        candidate[artifact_id][field_path] = value
        if value == "":
            intentional_blank += 1
        if previous == value:
            unchanged += 1
        else:
            changed += 1
        prefix = artifact_id.split("-", 1)[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    output_rows = [
        {
            "artifact_id": artifact_id,
            "pred_json": json.dumps(candidate[artifact_id], ensure_ascii=False),
        }
        for artifact_id in template_order
    ]
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact_id", "pred_json"])
        writer.writeheader()
        writer.writerows(output_rows)

    audit: dict[str, Any] = {
        "base_submission": str(args.base_submission.absolute()),
        "annotations": str(args.annotations.absolute()),
        "output": str(output),
        "safety": {
            "required_status": VERIFIED_STATUS,
            "public_render_path_required": not args.skip_render_path_check,
            "peer_submission_read": False,
            "grader_only_provenance_read": False,
            "enterprise_tables_read": False,
        },
        "annotation_rows": len(annotations),
        "verified_rows_applied": len(applied),
        "ignored_unverified_status_rows": ignored_status,
        "changed_fields": changed,
        "unchanged_fields": unchanged,
        "intentional_blank_fields": intentional_blank,
        "applied_by_prefix": dict(sorted(by_prefix.items())),
    }
    audit_output = output.with_suffix(".audit.json")
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"wrote={output}")
    print(f"wrote={audit_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "submission_OCR_fast_partial.csv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "audits" / "image_grounded" / "manual_ground_truth_annotations.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "submission_OCR_image_reviewed.csv",
    )
    parser.add_argument(
        "--skip-render-path-check",
        action="store_true",
        help="Trust already-recorded verified annotations when reproducing from packaged checkpoints without public renders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(apply(parse_args()))
