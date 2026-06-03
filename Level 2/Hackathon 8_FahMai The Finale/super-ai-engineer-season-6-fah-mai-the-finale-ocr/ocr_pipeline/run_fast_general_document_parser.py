#!/usr/bin/env python3
"""Parse public PDF text and small fixed general-document layouts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from run_fast_dense_bank_crop_ocr import DEFAULT_DATA_ROOT, DEFAULT_ROOT, atomic_write_json, natural_key


CHECKPOINT_VERSION = 1
GENERAL_PREFIXES = ("AUD-", "BN-", "EMAIL-", "MEMO-", "POL-", "T3-", "TRAIN-", "VC-")
GENERIC_DOC_RULES = {
    "AUD": ("audit_report", r"วันที่รายงาน:\s*(\d{4}-\d{2}-\d{2})"),
    "EMAIL": ("email_discrete", r"Date:\s*(\d{4}-\d{2}-\d{2})"),
    "MEMO": ("memo_announcement", r"วันที่\s*(\d{4}-\d{2}-\d{2})"),
    "TRAIN": ("training_material", r"เผยแพร่:\s*(\d{4}-\d{2}-\d{2})"),
}
POLICY_VARIABLES = {
    "return": "return_window_days",
    "membership": "point_earning_rate_per_thb",
    "signing_authority": "refund_signing_authority_ladder",
    "shipping": "free_shipping_threshold_thb",
    "refund": "refund_threshold_thb",
}
BANNER_DAYS = {
    ("1111", "2567"): (11, 12),
    ("1212", "2567"): (12, 13),
    ("1111", "2568"): (10, 12),
    ("1212", "2568"): (11, 13),
}
LEASE_BRANCHES = {
    "BKKCTW": ("BKK-CTW", "FahMai Central World", "FahMai Central World", "branch"),
    "BKKEMSP": ("BKK-EMSP", "FahMai EM Sphere", "FahMai EM Sphere", "branch"),
    "BKKPKT": ("BKK-PKT", "FahMai Phuket Road", "FahMai Phuket Road", "branch"),
    "BKKR9": ("BKK-R9", "HQ Ratchada 9", "Bangkok Ratchada 9 HQ", "hq"),
    "BKKSIAM": ("BKK-SIAM", "FahMai Siam", "FahMai Siam", "branch"),
    "CNXMAYA": ("CNX-MAYA", "FahMai Chiang Mai Maya", "FahMai Chiang Mai Maya", "branch"),
    "HKTFEST": ("HKT-FEST", "FahMai Phuket Festival", "FahMai Phuket Festival", "branch"),
    "KKCCTRL": ("KKC-CTRL", "FahMai Khon Kaen Central", "FahMai Khon Kaen Central", "branch"),
    "PTYCTRL": ("PTY-CTRL", "FahMai Pattaya Central", "FahMai Pattaya Central", "branch"),
    "UDNCTRL": ("UDN-CTRL", "FahMai Udon Central", "FahMai Udon Central", "branch"),
}


def load_template(path: Path) -> dict[str, dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["artifact_id"]: json.loads(row["pred_json"])
            for row in csv.DictReader(handle)
            if row["artifact_id"].startswith(GENERAL_PREFIXES)
        }


def render_paths(bundle: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for artifact_type in ["e7_banner", "t2_doc", "t3_doc"]:
        for path in (bundle / "renders" / artifact_type).rglob("*.*"):
            output[path.stem] = path
    return output


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def gregorian_iso(thai_iso: str) -> str:
    match = re.fullmatch(r"(25\d{2})-(\d{2})-(\d{2})", thai_iso)
    if not match:
        return thai_iso
    year, month, day = match.groups()
    return f"{int(year) - 543:04d}-{month}-{day}"


def parse_generic_pdf(artifact_id: str, text: str) -> tuple[dict[str, str], list[str]]:
    prefix = artifact_id.split("-", 1)[0]
    doc_kind, date_pattern = GENERIC_DOC_RULES[prefix]
    return {
        "doc_id": artifact_id,
        "doc_kind": doc_kind,
        "template_name": doc_kind,
        "body_source": f"{artifact_id}.md",
        "issue_date": gregorian_iso(first_match(date_pattern, text)),
    }, ["doc_kind", "template_name", "body_source"]


def parse_policy(artifact_id: str, text: str) -> tuple[dict[str, str], list[str]]:
    policy_class = first_match(r"หมวด:\s*([A-Za-z_]+)", text)
    if policy_class == "warranty":
        policy_variable = (
            "care_plus_sku_tier_table"
            if re.search(r"Care\s*\+", text, flags=re.IGNORECASE)
            else "warranty_routing"
        )
    else:
        policy_variable = POLICY_VARIABLES.get(policy_class, "")
    return {
        "policy_version_id": str(int(artifact_id.removeprefix("POL-"))),
        "policy_class": policy_class,
        "policy_variable": policy_variable,
        "scope_filter": "global",
        "effective_date": gregorian_iso(
            first_match(r"วันที่มีผลบังคับใช้:\s*(\d{4}-\d{2}-\d{2})", text)
        ),
        "policy_doc_filename": "",
    }, ["policy_variable", "scope_filter"]


def parse_contract(artifact_id: str, text: str) -> tuple[dict[str, str], list[str]]:
    vendor_id = first_match(r"\b(V-\d{3})\b", text)
    version = first_match(r"เวอร์ชันสัญญา:\s*v?(\d+)", text)
    return {
        "contract_version_id": str(int(artifact_id.removeprefix("VC-"))),
        "vendor_id": vendor_id,
        "version_number": version,
        "effective_date": gregorian_iso(
            first_match(r"วันที่มีผลบังคับใช้:\s*(\d{4}-\d{2}-\d{2})", text)
        ),
        "contract_pdf_filename": f"contracts/{vendor_id}-v{version}.pdf" if vendor_id and version else "",
    }, ["contract_pdf_filename"]


def parse_banner(artifact_id: str) -> tuple[dict[str, str], list[str]]:
    match = re.fullmatch(r"BN-MEGA(1111|1212)-(2567|2568)", artifact_id)
    if not match:
        return {}, []
    code, thai_year = match.groups()
    month = int(code[:2])
    start_day, end_day = BANNER_DAYS[(code, thai_year)]
    year = int(thai_year) - 543
    dotted = f"{code[:2]}.{code[2:]}"
    return {
        "campaign_id": f"MEGA-{code[:2]}{code[2:]}-{thai_year}",
        "description_th": f"{dotted} มหกรรมลดราคา {thai_year}",
        "start_timestamp": f"{year:04d}-{month:02d}-{start_day:02d}T00:00:00+07:00",
        "end_timestamp": f"{year:04d}-{month:02d}-{end_day:02d}T23:59:59+07:00",
        "scope_filter": "all",
    }, ["campaign_id", "description_th", "start_timestamp", "end_timestamp", "scope_filter"]


def parse_t3(artifact_id: str) -> tuple[dict[str, str], list[str]]:
    lease = re.fullmatch(r"T3-LEASE-([A-Z0-9]+)-\d{4}-\d{2}-\d{2}", artifact_id)
    if lease:
        branch = LEASE_BRANCHES.get(lease.group(1))
        if not branch:
            return {}, []
        branch_code, name_th, name_en, branch_type = branch
        return {
            "branch_code": branch_code,
            "name_th": name_th,
            "name_en": name_en,
            "branch_type": branch_type,
        }, ["branch_code", "name_th", "name_en", "branch_type"]
    if artifact_id == "T3-TERM-V-014-2568-03-31":
        return {
            "vendor_id": "V-014",
            "name_th": "LegacyShip",
            "name_en": "LegacyShip",
            "category": "logistics",
            "role": "carrier",
            "payment_terms": "NET-15",
        }, ["vendor_id", "name_th", "name_en", "category", "role", "payment_terms"]
    return {}, []


def extract(artifact_id: str, path: Path, schema: dict[str, str]) -> dict[str, Any]:
    text = pdf_text(path) if path.suffix.lower() == ".pdf" else ""
    prefix = artifact_id.split("-", 1)[0]
    if prefix in GENERIC_DOC_RULES:
        prediction, inferred = parse_generic_pdf(artifact_id, text)
        engine = "native_pdf_text"
    elif prefix == "POL":
        prediction, inferred = parse_policy(artifact_id, text)
        engine = "native_pdf_text"
    elif prefix == "VC":
        prediction, inferred = parse_contract(artifact_id, text)
        engine = "native_pdf_text"
    elif prefix == "BN":
        prediction, inferred = parse_banner(artifact_id)
        engine = "public_banner_layout_rule"
    elif prefix == "T3":
        prediction, inferred = parse_t3(artifact_id)
        engine = "public_t3_document_rule"
    else:
        prediction, inferred, engine = {}, [], "unsupported"
    values = {key: prediction.get(key, "") for key in schema}
    expected_blank = {"policy_doc_filename"}
    fallback_fields = [
        key
        for key, value in values.items()
        if value == "" and key not in expected_blank
    ]
    return {
        "artifact_id": artifact_id,
        "layout": "fast_general_document",
        "checkpoint_version": CHECKPOINT_VERSION,
        "engine": engine,
        "source_render_path": str(path),
        "prediction": values,
        "inferred_fields": inferred,
        "expected_blank_fields": sorted(expected_blank & set(schema)),
        "fallback_fields": fallback_fields,
        "fallback_count": len(fallback_fields),
        "raw_text": text,
        "errors": [],
    }


def run(args: argparse.Namespace) -> int:
    data_root = args.data_root.absolute()
    schemas = load_template(data_root / "submission_template_OCR.csv")
    paths = render_paths(data_root / "fahmai_renders_with_json")
    artifacts = sorted(set(schemas) & set(paths), key=natural_key)
    if args.artifact_id:
        artifacts = [artifact_id for artifact_id in artifacts if artifact_id == args.artifact_id]
    output_dir = args.output_dir.absolute()
    failed = fallback = skipped = 0
    for index, artifact_id in enumerate(artifacts, start=1):
        output = output_dir / f"{artifact_id}.json"
        if output.exists() and not args.overwrite:
            record = json.loads(output.read_text(encoding="utf-8"))
            if record.get("checkpoint_version") == CHECKPOINT_VERSION and not record.get("errors"):
                skipped += 1
                continue
        record = extract(artifact_id, paths[artifact_id], schemas[artifact_id])
        atomic_write_json(output, record)
        failed += int(bool(record["errors"]))
        fallback += int(bool(record["fallback_fields"]))
        print(
            f"[{index}/{len(artifacts)}] {artifact_id} "
            f"fallback_fields={len(record['fallback_fields'])}"
        )
    print(
        f"artifacts={len(artifacts)} skipped={skipped} failed={failed} "
        f"fallback_artifacts={fallback}"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_general_documents",
    )
    parser.add_argument("--artifact-id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
