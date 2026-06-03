#!/usr/bin/env python3
"""One-shot OCR endpoint smoke test.

Default command:

    python test_ocr_endpoint_once.py

It sends one local FahMai bank-statement render pair as base64 to the remote
`/ocr` endpoint and writes a compact response audit under `runs/`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENDPOINT = "http://swarm-manager.modelharbor.com:57444/ocr"
DEFAULT_OCR_ROOT = PROJECT_DIR / "super-ai-engineer-season-6-fah-mai-the-finale-ocr"
DEFAULT_RENDER_BANK_DIR = DEFAULT_OCR_ROOT / "fahmai_renders_with_json" / "renders" / "bank_statement"
DEFAULT_SAMPLE_ARTIFACT = "BS-BBL-OPER-2567-11"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
}


def natural_key(value: str) -> list[Any]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value)]


def mime_type(path: Path) -> str:
    return MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def encode_file(path: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    decoded = base64.b64decode(encoded, validate=True)
    if decoded != content:
        raise RuntimeError(f"Base64 roundtrip mismatch: {path}")
    return (
        encoded,
        {
            "filename": path.name,
            "content": encoded,
            "mime_type": mime_type(path),
        },
        {
            "path": str(path),
            "bytes": len(content),
            "base64_chars": len(encoded),
            "mime_type": mime_type(path),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    )


def discover_artifact_files(render_dir: Path, artifact_id: str, max_transactions: int) -> tuple[Path, list[Path]]:
    if not render_dir.exists():
        raise FileNotFoundError(f"Render folder not found: {render_dir}")
    header = next(render_dir.rglob(f"{artifact_id}_header.png"), None)
    if header is None:
        raise FileNotFoundError(f"Header render not found for {artifact_id} under {render_dir}")
    transactions = sorted(
        render_dir.rglob(f"{artifact_id}_transactions_p*.png"),
        key=lambda path: natural_key(path.name),
    )
    if max_transactions > 0:
        transactions = transactions[:max_transactions]
    if not transactions:
        raise FileNotFoundError(f"Transaction renders not found for {artifact_id} under {render_dir}")
    return header, transactions


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.header:
        header = Path(args.header).resolve()
        transactions = [Path(path).resolve() for path in args.transaction]
        artifact_id = args.id or header.stem.replace("_header", "")
        if not transactions:
            raise RuntimeError("--transaction is required when --header is used.")
    else:
        artifact_id = args.id or args.sample_artifact
        header, transactions = discover_artifact_files(
            Path(args.render_dir),
            artifact_id,
            args.max_transactions,
        )

    if not header.exists():
        raise FileNotFoundError(f"Header file not found: {header}")
    for path in transactions:
        if not path.exists():
            raise FileNotFoundError(f"Transaction file not found: {path}")

    header_base64, header_entry, header_audit = encode_file(header)
    transaction_entries: list[dict[str, str]] = []
    transaction_base64: list[str] = []
    file_audit = [{"role": "header", **header_audit}]
    for index, transaction_path in enumerate(transactions, start=1):
        encoded, entry, audit = encode_file(transaction_path)
        transaction_entries.append(entry)
        transaction_base64.append(encoded)
        file_audit.append({"role": f"transaction_{index}", **audit})

    if args.payload_style == "object":
        payload = {
            "id": artifact_id,
            "header": header_entry,
            "transaction": transaction_entries,
            "persist": args.persist,
        }
    else:
        payload = {
            "id": artifact_id,
            "header": header_base64,
            "transaction": transaction_base64,
        }
        if args.persist:
            payload["persist"] = True
    return payload, file_audit


def post_json(endpoint: str, payload: dict[str, Any], timeout: int) -> tuple[int, Any, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw else {}, raw
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return error.code, parsed, raw
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        message = str(reason)
        return 0, {"error": "connection_failed", "message": message}, message


def response_summary(
    status: int,
    response: Any,
    elapsed_seconds: float,
    min_answer_fields: int,
    min_nonempty_fields: int,
) -> dict[str, Any]:
    answer = response.get("answer", {}) if isinstance(response, dict) else {}
    answer_fields = len(answer) if isinstance(answer, dict) else 0
    nonempty_fields = (
        sum(str(value).strip() != "" for value in answer.values())
        if isinstance(answer, dict)
        else 0
    )
    connection_error = (
        isinstance(response, dict)
        and response.get("error") == "connection_failed"
    )
    passed = (
        status == 200
        and isinstance(answer, dict)
        and answer_fields >= min_answer_fields
        and nonempty_fields >= min_nonempty_fields
    )
    return {
        "http_status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "connection_error": connection_error,
        "error_message": response.get("message", "") if isinstance(response, dict) else "",
        "id": response.get("id") if isinstance(response, dict) else "",
        "answer_type": type(answer).__name__,
        "answer_fields": answer_fields,
        "nonempty_fields": nonempty_fields,
        "min_answer_fields": min_answer_fields,
        "min_nonempty_fields": min_nonempty_fields,
        "passed": passed,
        "answer_preview": dict(list(answer.items())[:12]) if isinstance(answer, dict) else answer,
    }


def write_audit(audit: dict[str, Any], output: Path | None) -> Path:
    if output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_RUNS_DIR / f"remote_ocr_smoke_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one FahMai OCR smoke request to an /ocr endpoint.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Full OCR endpoint URL.")
    parser.add_argument("--id", default="", help="Request id / artifact id.")
    parser.add_argument("--sample-artifact", default=DEFAULT_SAMPLE_ARTIFACT, help="Local sample artifact id.")
    parser.add_argument("--render-dir", default=str(DEFAULT_RENDER_BANK_DIR), help="Bank-statement render folder.")
    parser.add_argument("--max-transactions", type=int, default=1, help="Number of transaction pages to send.")
    parser.add_argument("--header", default="", help="Manual header image/PDF path.")
    parser.add_argument("--transaction", action="append", default=[], help="Manual transaction image/PDF path.")
    parser.add_argument("--timeout", type=int, default=900, help="POST timeout seconds.")
    parser.add_argument("--persist", action="store_true", help="Ask API to persist outputs.")
    parser.add_argument(
        "--payload-style",
        choices=["eval", "object"],
        default="eval",
        help="eval sends header as a base64 string and transaction as base64 strings, matching the PDF spec.",
    )
    parser.add_argument(
        "--min-answer-fields",
        type=int,
        default=1,
        help="Fail unless response.answer has at least this many fields. Use 0 to allow empty answers.",
    )
    parser.add_argument(
        "--min-nonempty-fields",
        type=int,
        default=1,
        help="Fail unless response.answer has at least this many non-empty values. Use 0 for schema-only checks.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Audit JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload, files = build_payload(args)
    endpoint = args.endpoint.rstrip("/")

    print("== OCR Endpoint Smoke Test ==")
    print(f"endpoint: {endpoint}")
    print(f"id: {payload['id']}")
    print("files:")
    for file_info in files:
        print(
            f"  - {file_info['role']}: {file_info['bytes']} bytes, "
            f"{file_info['base64_chars']} base64 chars, sha256={file_info['sha256'][:12]}..."
        )

    started = time.perf_counter()
    status, response, raw = post_json(endpoint, payload, args.timeout)
    elapsed = time.perf_counter() - started
    summary = response_summary(
        status,
        response,
        elapsed,
        args.min_answer_fields,
        args.min_nonempty_fields,
    )
    audit = {
        "endpoint": endpoint,
        "request_id": payload["id"],
        "files": files,
        "summary": summary,
        "response": response,
        "raw_response": raw[:20000],
    }
    output = write_audit(audit, args.output)

    print("\n== Result ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\naudit_json: {output}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
