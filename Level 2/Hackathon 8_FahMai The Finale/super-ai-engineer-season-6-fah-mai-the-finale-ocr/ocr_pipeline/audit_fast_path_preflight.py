#!/usr/bin/env python3
"""Verify that the public OCR renders are readable before an exam run.

The audit intentionally reads rendered files only. It does not inspect
render_provenance.jsonl, per_artifact JSON files, or enterprise tables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
SHORT_BUNDLE = Path(r"C:\fahmai_ocr_data\fahmai_renders_with_json")
DEFAULT_BUNDLE = SHORT_BUNDLE if SHORT_BUNDLE.exists() else DEFAULT_ROOT / "fahmai_renders_with_json"
EXPECTED = {
    "bank_statement": {"extension": ".png", "count": 2714},
    "e7_banner": {"extension": ".png", "count": 4},
    "receipt": {"extension": ".png", "count": 563},
    "t2_doc": {"extension": ".pdf", "count": 81},
    "t3_doc": {"extension": ".png", "count": 11},
    "vendor_invoice": {"extension": ".png", "count": 792},
    "warranty_form": {"extension": ".png", "count": 1963},
}


def probe_file(path: Path, decode_images: bool) -> tuple[bool, str | None]:
    try:
        if path.suffix.lower() == ".pdf":
            with path.open("rb") as handle:
                return handle.read(4) == b"%PDF", None
        if decode_images:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            return True, None
        with path.open("rb") as handle:
            return bool(handle.read(1)), None
    except Exception as exc:  # Keep scanning to report every broken render.
        return False, f"{type(exc).__name__}: {exc}"


def audit(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    # Preserve the junction instead of expanding it to the long repo path.
    bundle = args.bundle.absolute()
    renders = bundle / "renders"
    summary: dict[str, Any] = {}
    failures: list[dict[str, str]] = []

    for artifact_type, config in EXPECTED.items():
        extension = config["extension"]
        paths = sorted((renders / artifact_type).rglob(f"*{extension}"))
        readable = 0
        for path in paths:
            ok, error = probe_file(path, args.decode_images)
            if ok:
                readable += 1
            elif len(failures) < args.max_failures:
                failures.append({"path": str(path), "error": error or "unknown error"})
        summary[artifact_type] = {
            "extension": extension,
            "expected": config["count"],
            "found": len(paths),
            "readable": readable,
            "missing_or_extra": len(paths) - config["count"],
            "unreadable": len(paths) - readable,
        }

    expected_total = sum(config["count"] for config in EXPECTED.values())
    found_total = sum(item["found"] for item in summary.values())
    readable_total = sum(item["readable"] for item in summary.values())
    passed = expected_total == found_total == readable_total
    report = {
        "passed": passed,
        "bundle": str(bundle),
        "mode": "decode_images" if args.decode_images else "open_bytes",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "expected_total": expected_total,
        "found_total": found_total,
        "readable_total": readable_total,
        "summary": summary,
        "failure_examples": failures,
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote={output}")
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_path_preflight.json",
    )
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--max-failures", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(audit(parse_args()))
