#!/usr/bin/env python3
"""Preflight and smoke checks for the FahMai OCR API.

Typical use after starting the server:

    python check_api_ready.py --url http://127.0.0.1:8000

Optional real OCR smoke test with local render files:

    python check_api_ready.py --sample-artifact BS-BBL-OPER-2567-01 --send

The default run does not call OCR. It only checks health, OpenAPI routes,
base64 validation behavior, logs, and recent persisted run metadata.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_API_URL = os.environ.get("OCR_API_URL", "http://127.0.0.1:8000")
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_OCR_ROOT = PROJECT_DIR / "super-ai-engineer-season-6-fah-mai-the-finale-ocr"
DEFAULT_RENDER_BANK_DIR = DEFAULT_OCR_ROOT / "fahmai_renders_with_json" / "renders" / "bank_statement"
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
}


@dataclass
class HttpResult:
    status: int
    payload: Any
    raw: str


def section(title: str) -> None:
    print(f"\n== {title} ==")


def as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> HttpResult:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return HttpResult(response.status, json.loads(raw) if raw else {}, raw)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return HttpResult(error.code, parsed, raw)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def mime_type(path: Path) -> str:
    return MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def encoded_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    decoded = base64.b64decode(encoded, validate=True)
    if decoded != content:
        fail(f"Base64 roundtrip mismatch: {path}")
    entry = {
        "filename": path.name,
        "content": encoded,
        "mime_type": mime_type(path),
    }
    report = {
        "path": str(path),
        "bytes": len(content),
        "base64_chars": len(encoded),
        "mime_type": entry["mime_type"],
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return entry, report


def discover_sample_files(render_dir: Path, artifact_id: str, max_transactions: int) -> tuple[Path | None, list[Path]]:
    header = next(render_dir.rglob(f"{artifact_id}_header.png"), None) if render_dir.exists() else None
    transactions = (
        sorted(
            render_dir.rglob(f"{artifact_id}_transactions_p*.png"),
            key=lambda path: natural_path_key(path.name),
        )
        if render_dir.exists()
        else []
    )
    if max_transactions > 0:
        transactions = transactions[:max_transactions]
    return header, transactions


def natural_path_key(value: str) -> list[Any]:
    import re

    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value)]


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    header = Path(args.header).resolve() if args.header else None
    transactions = [Path(path).resolve() for path in args.transaction]

    if args.sample_artifact and (header is None or not transactions):
        sample_header, sample_transactions = discover_sample_files(
            Path(args.render_dir),
            args.sample_artifact,
            args.max_transactions,
        )
        header = header or sample_header
        transactions = transactions or sample_transactions

    if header is None and not transactions:
        return None, []
    if header is None:
        fail("Provide --header or --sample-artifact with a matching header render.")
    if not header.exists():
        fail(f"Header file not found: {header}")
    if not transactions:
        fail("Provide at least one --transaction or use --sample-artifact.")
    for path in transactions:
        if not path.exists():
            fail(f"Transaction file not found: {path}")

    header_entry, header_report = encoded_file(header)
    transaction_entries: list[dict[str, Any]] = []
    reports = [{"role": "header", **header_report}]
    for index, path in enumerate(transactions, start=1):
        entry, report = encoded_file(path)
        transaction_entries.append(entry)
        reports.append({"role": f"transaction_{index}", **report})

    request_id = args.id or args.sample_artifact or header.stem.replace("_header", "")
    payload = {
        "id": request_id,
        "header": header_entry,
        "transaction": transaction_entries,
        "persist": args.persist,
    }
    return payload, reports


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def tail_lines(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def summarize_logs(runs_dir: Path, lines: int) -> dict[str, list[str]]:
    return {
        "stderr": tail_lines(runs_dir / "api_server.stderr.log", lines),
        "stdout": tail_lines(runs_dir / "api_server.stdout.log", lines),
    }


def read_csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.DictReader(handle)))


def summarize_run(path: Path) -> dict[str, Any]:
    meta_path = path / "run_meta.json"
    answers_path = path / "answers.json"
    debug_path = path / "ocr_debug.csv"
    meta: dict[str, Any] = {}
    answers: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {"_error": "invalid run_meta.json"}
    if answers_path.exists():
        try:
            answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))
            answers = answers_payload.get("answers", answers_payload)
        except json.JSONDecodeError:
            answers = {"_error": "invalid answers.json"}
    return {
        "run_dir": str(path),
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "artifact_count": meta.get("artifact_count", len(answers) if isinstance(answers, dict) else 0),
        "artifacts": meta.get("artifacts", list(answers) if isinstance(answers, dict) else []),
        "elapsed_seconds": meta.get("elapsed_seconds", ""),
        "errors": meta.get("errors", {}),
        "debug_rows": read_csv_count(debug_path),
    }


def summarize_recent_runs(runs_dir: Path, limit: int) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    run_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and ((path / "run_meta.json").exists() or (path / "answers.json").exists())
    ]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [summarize_run(path) for path in run_dirs[:limit]]


def check_health(base_url: str, timeout: int) -> HttpResult:
    result = http_json("GET", f"{base_url}/health", timeout=timeout)
    if result.status != 200:
        fail(f"/health returned HTTP {result.status}: {result.raw}")
    return result


def check_routes(base_url: str, timeout: int) -> list[str]:
    result = http_json("GET", f"{base_url}/openapi.json", timeout=timeout)
    if result.status != 200:
        fail(f"/openapi.json returned HTTP {result.status}: {result.raw}")
    paths = sorted(result.payload.get("paths", {}).keys())
    if "/ocr" not in paths:
        fail("OpenAPI does not expose /ocr.")
    extra_ocr_paths = [path for path in paths if path.startswith("/ocr/")]
    if extra_ocr_paths:
        fail(f"OpenAPI exposes unexpected OCR subpaths: {extra_ocr_paths}. Expected only /ocr.")
    return paths


def check_invalid_base64(base_url: str, timeout: int) -> HttpResult:
    payload = {
        "id": "base64_probe",
        "header": "not-valid-base64",
        "transaction": ["not-valid-base64"],
        "persist": False,
    }
    result = http_json("POST", f"{base_url}/ocr", payload=payload, timeout=timeout)
    if result.status != 422:
        fail(f"Invalid-base64 probe returned HTTP {result.status}; expected 422. Body={result.raw}")
    return result


def post_ocr(base_url: str, payload: dict[str, Any], timeout: int) -> HttpResult:
    result = http_json("POST", f"{base_url}/ocr", payload=payload, timeout=timeout)
    if result.status != 200:
        fail(f"OCR smoke request returned HTTP {result.status}: {result.raw}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FahMai OCR API readiness, base64 payloads, and logs.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API base URL. Default: %(default)s")
    parser.add_argument("--id", default="", help="Request id/artifact id for a real OCR smoke request.")
    parser.add_argument("--header", default="", help="Header image/PDF path for real OCR smoke request.")
    parser.add_argument(
        "--transaction",
        action="append",
        default=[],
        help="Transaction image/PDF path. Repeat for multiple pages.",
    )
    parser.add_argument(
        "--sample-artifact",
        default="",
        help="Auto-load local render files for an artifact id, e.g. BS-BBL-OPER-2567-01.",
    )
    parser.add_argument("--render-dir", default=str(DEFAULT_RENDER_BANK_DIR), help="Bank-statement render folder.")
    parser.add_argument("--max-transactions", type=int, default=1, help="Max auto-discovered transaction pages.")
    parser.add_argument("--send", action="store_true", help="POST the real base64 payload to /ocr.")
    parser.add_argument("--persist", action="store_true", help="Persist OCR outputs when --send is used.")
    parser.add_argument(
        "--skip-invalid-base64",
        action="store_true",
        help="Skip the no-OCR invalid-base64 422 probe.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for light checks.")
    parser.add_argument("--ocr-timeout", type=int, default=900, help="HTTP timeout for --send.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Runs/log folder.")
    parser.add_argument("--log-lines", type=int, default=40, help="Server log lines to show.")
    parser.add_argument("--recent-runs", type=int, default=5, help="Recent persisted run summaries to show.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON report instead of human sections.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.url.rstrip("/")
    runs_dir = Path(args.runs_dir)

    payload, base64_reports = build_payload(args)
    report: dict[str, Any] = {
        "base_url": base_url,
        "health": None,
        "routes": [],
        "base64_files": base64_reports,
        "invalid_base64_probe": None,
        "ocr_response": None,
        "logs": {},
        "recent_runs": [],
    }

    report["health"] = check_health(base_url, args.timeout).payload
    report["routes"] = check_routes(base_url, args.timeout)
    if not args.skip_invalid_base64:
        probe = check_invalid_base64(base_url, args.timeout)
        report["invalid_base64_probe"] = {"status": probe.status, "payload": probe.payload}
    if args.send:
        if payload is None:
            fail("--send requires --header/--transaction or --sample-artifact.")
        response = post_ocr(base_url, payload, args.ocr_timeout)
        answer = response.payload.get("answer", {})
        report["ocr_response"] = {
            "status": response.status,
            "id": response.payload.get("id"),
            "answer_fields": len(answer) if isinstance(answer, dict) else 0,
            "answer_preview": dict(list(answer.items())[:10]) if isinstance(answer, dict) else answer,
        }

    report["logs"] = summarize_logs(runs_dir, args.log_lines)
    report["recent_runs"] = summarize_recent_runs(runs_dir, args.recent_runs)

    if args.json:
        print_json(report)
        return 0

    section("Health")
    print_json(report["health"])
    section("Routes")
    print_json(report["routes"])

    section("Base64")
    if report["base64_files"]:
        print_json(report["base64_files"])
    else:
        print("No real files supplied. Use --sample-artifact or --header/--transaction for file roundtrip checks.")

    section("Invalid Base64 Probe")
    print_json(report["invalid_base64_probe"] or {"skipped": True})

    if report["ocr_response"] is not None:
        section("OCR Smoke Response")
        print_json(report["ocr_response"])

    section("Recent Runs")
    print_json(report["recent_runs"])

    section("Server Logs")
    logs = report["logs"]
    for name in ["stderr", "stdout"]:
        print(f"-- {name} --")
        lines = logs.get(name, [])
        print("\n".join(lines) if lines else "(no log lines)")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
