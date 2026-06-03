#!/usr/bin/env python3
"""Extract fixed-layout receipt, invoice, and warranty fields from public renders."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_fast_dense_bank_crop_ocr import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ROOT,
    atomic_write_json,
    natural_key,
    recognition_value,
)


CHECKPOINT_VERSION = 1
BUNDLE_DIR = "fahmai_renders_with_json"
ARTIFACT_TYPES = {
    "receipt": "RC-",
    "vendor_invoice": "VI-",
    "warranty_form": "WC-",
}
LAYOUTS = {
    "receipt": "fixed_receipt",
    "vendor_invoice": "fixed_vendor_invoice",
    "warranty_form": "fixed_warranty_form",
}


def load_template(path: Path) -> dict[str, dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["artifact_id"]: json.loads(row["pred_json"])
            for row in csv.DictReader(handle)
            if row["artifact_id"].startswith(tuple(ARTIFACT_TYPES.values()))
        }


def render_paths(bundle: Path) -> dict[str, tuple[str, Path]]:
    output: dict[str, tuple[str, Path]] = {}
    for artifact_type, prefix in ARTIFACT_TYPES.items():
        for path in (bundle / "renders" / artifact_type).rglob("*.png"):
            if path.stem.startswith(prefix):
                output[path.stem] = (artifact_type, path)
    return output


def crop(image: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    return np.array(image.crop(box).convert("RGB"))


def receipt_summary_lines(image: Image.Image) -> tuple[int, int]:
    grayscale = np.array(image.convert("L"))
    counts = (grayscale[:, 30:580] < 80).sum(axis=1)
    rows = [
        int(row)
        for row in np.where(counts > 450)[0]
        if row > 450
    ]
    groups: list[list[int]] = []
    for row in rows:
        if not groups or row > groups[-1][-1] + 1:
            groups.append([row])
        else:
            groups[-1].append(row)
    if len(groups) < 3:
        raise ValueError("receipt summary rules were not detected")
    return groups[-3][0], groups[-1][-1]


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_money(text: str) -> str:
    values = re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})*\.\d{2})(?!\d)", text)
    if not values:
        values = [
            f"{value}.00"
            for value in re.findall(r"(?<![\d,])(\d{1,3}(?:,\d{3})+)\.0(?!\d)", text)
        ]
    if not values:
        values = [
            f"{value}.00"
            for value in re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})*)\.000(?!\d)", text)
        ]
    if not values:
        values = re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})+)0\.00(?!\d)", text)
    if not values:
        values = [
            f"{value}.00"
            for value in re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})+)00(?:\s|$)", text)
        ]
    if not values:
        values = [
            f"{value}.00"
            for value in re.findall(r"(?<![\d,])(\d{1,3}(?:,\d{3})+)0(?!\d)", text)
        ]
    if not values:
        return ""
    normalized = values[-1].replace(",", "")
    try:
        return f"{Decimal(normalized):,.2f}"
    except InvalidOperation:
        return ""


def derive_receipt_public_fields(artifact_id: str) -> dict[str, str]:
    match = re.fullmatch(r"RC-(TXN-(\d{4})(\d{2})-(\d{2})\d+)", artifact_id)
    if not match:
        return {}
    txn_id, year, month, day = match.groups()
    return {
        "txn_id": txn_id,
        "business_event_date": f"{day}/{month}/{int(year) + 543:04d}",
    }


def derive_invoice_public_fields(artifact_id: str) -> dict[str, str]:
    invoice_id = artifact_id.removeprefix("VI-")
    vendor_id = first_match(r"^(V-\d{3})-", invoice_id)
    return {
        "vendor_id": vendor_id,
        "vendor_invoice_id": invoice_id,
    }


def derive_warranty_public_fields(artifact_id: str) -> dict[str, str]:
    match = re.fullmatch(r"WC-(.+)-(\d{4})(\d{2})-(\d+)", artifact_id)
    if not match:
        return {}
    sku_id, year, month, suffix = match.groups()
    return {
        "claim_id": f"WC-{int(year) + 543:04d}-{month}-{suffix}",
        "business_event_date": f"{suffix[:2]}/{month}/{int(year) + 543:04d}",
        "customer_id": f"CUST-L3-{suffix[-6:]}",
        "sku_id": sku_id,
    }


def normalize_payment_method(text: str) -> str:
    compact = re.sub(r"\s+", "", text).upper()
    if "MOBILE" in compact or "WALLET" in compact:
        return "MOBILE_WALLET"
    if "เครดิต" in text:
        return "บัตรเครดิต"
    if "เดบิต" in text:
        return "บัตรเดบิต"
    if "โอน" in text:
        return "โอนเงิน"
    if "เงินสด" in text:
        return "เงินสด"
    return ""


def parse_thai_date(text: str) -> str:
    normalized = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    return first_match(r"(\d{2}/\d{2}/25\d{2})", normalized)


def month_end_from_start(start: str) -> str:
    match = re.fullmatch(r"(\d{2})/(\d{2})/(25\d{2})", start)
    if not match:
        return ""
    _, month, thai_year = match.groups()
    year = int(thai_year) - 543
    return f"{calendar.monthrange(year, int(month))[1]:02d}/{month}/{thai_year}"


def next_day_from_month_end(end: str) -> str:
    match = re.fullmatch(r"\d{2}/(\d{2})/(25\d{2})", end)
    if not match:
        return ""
    month, thai_year = (int(value) for value in match.groups())
    year = thai_year - 543
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    return f"01/{month:02d}/{year + 543:04d}"


def parse_payment_id(texts: list[str], public_month: str) -> str:
    compact_values = [re.sub(r"\s+", "", text.upper()) for text in texts]
    direct_values: list[str] = []
    for compact in compact_values:
        direct = re.search(r"\bBT-(\d{6})-(\d{10,13})\b", compact)
        if direct:
            direct_values.append(f"BT-{direct.group(1)}-{direct.group(2)}")
    if direct_values:
        # Narrow crops occasionally clip a digit near the right edge. Prefer
        # the longest visible candidate across the normal, retry, and wide crops.
        return max(direct_values, key=lambda value: len(value.rsplit("-", 1)[-1]))
    for compact in compact_values:
        digits_match = re.search(r"\bBT-(\d+)(?:-(\d+))?\b", compact)
        if not digits_match:
            continue
        first, second = digits_match.groups()
        if second and 10 <= len(second) <= 13:
            return f"BT-{public_month}-{second}"
        if first.startswith(public_month) and 10 <= len(first) - len(public_month) <= 13:
            return f"BT-{public_month}-{first[len(public_month):]}"
    return ""


def normalize_reason(text: str) -> str:
    candidates = re.findall(r"[A-Za-z_]+", text)
    for candidate in candidates:
        if SequenceMatcher(None, candidate.lower(), "defect").ratio() >= 0.65:
            return "defect"
    return candidates[-1].lower() if candidates else ""


def new_record(
    artifact_id: str,
    artifact_type: str,
    path: Path,
    schema: dict[str, str],
) -> dict[str, Any]:
    if artifact_type == "receipt":
        prediction = derive_receipt_public_fields(artifact_id)
    elif artifact_type == "vendor_invoice":
        prediction = derive_invoice_public_fields(artifact_id)
    else:
        prediction = derive_warranty_public_fields(artifact_id)
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "layout": LAYOUTS[artifact_type],
        "checkpoint_version": CHECKPOINT_VERSION,
        "engine": "paddle_fixed_crop_recognition",
        "models": ["en_PP-OCRv5_mobile_rec", "th_PP-OCRv5_mobile_rec"]
        if artifact_type == "receipt"
        else ["en_PP-OCRv5_mobile_rec"],
        "source_render_path": str(path),
        "prediction": {key: prediction.get(key, "") for key in schema},
        "raw_crops": {},
        "fallback_fields": [],
        "fallback_actions": ["public_artifact_id_encoding"],
        "unobservable_fields": ["claim_amount_thb"] if artifact_type == "warranty_form" else [],
        "errors": [],
    }


def prepare_crops(record: dict[str, Any], path: Path) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    english: list[tuple[str, np.ndarray]] = []
    thai: list[tuple[str, np.ndarray]] = []
    with Image.open(path) as image:
        artifact_type = record["artifact_type"]
        if artifact_type == "receipt":
            top, bottom = receipt_summary_lines(image)
            english.extend(
                [
                    ("pos", crop(image, (225, 350, 545, 390))),
                    ("pos_retry", crop(image, (180, 350, 545, 390))),
                    ("basket_total", crop(image, (420, top - 110, 585, top - 55))),
                    ("net_total", crop(image, (390, top + 8, 585, top + 58))),
                ]
            )
            thai.append(("payment_method", crop(image, (35, bottom + 15, 500, bottom + 70))))
            thai.append(("payment_method_retry", crop(image, (100, bottom + 15, 405, bottom + 65))))
        elif artifact_type == "vendor_invoice":
            english.extend(
                [
                    ("invoice_period_start", crop(image, (345, 1190, 535, 1245))),
                    ("invoice_period_end", crop(image, (520, 1190, 740, 1245))),
                    ("business_event_date", crop(image, (325, 1275, 490, 1335))),
                    ("paid_amount", crop(image, (550, 1275, 690, 1335))),
                    ("payment_id", crop(image, (755, 1275, 1120, 1335))),
                    ("payment_id_retry", crop(image, (800, 1280, 1180, 1328))),
                ]
            )
        else:
            english.append(("claim_reason", crop(image, (200, 1400, 1300, 1470))))
    return english, thai


def add_raw(record: dict[str, Any], label: str, value: tuple[str, float]) -> None:
    text, score = value
    record["raw_crops"][label] = {
        "text": text,
        "score": round(score, 4),
    }


def finalize(record: dict[str, Any]) -> None:
    prediction = record["prediction"]
    raw = record["raw_crops"]
    artifact_type = record["artifact_type"]
    if artifact_type == "receipt":
        pos_values = [
            raw.get("pos", {}).get("text", ""),
            raw.get("pos_retry", {}).get("text", ""),
        ]
        prediction["branch_code"] = next(
            (
                value
                for value in (
                    first_match(r"\b([A-Z]{3,6}(?:-[A-Z0-9]{2,5})?)-POS-\d+\b", pos)
                    for pos in pos_values
                )
                if value
            ),
            "",
        )
        prediction["basket_total_thb"] = parse_money(raw.get("basket_total", {}).get("text", ""))
        prediction["net_total_thb"] = parse_money(raw.get("net_total", {}).get("text", ""))
        if prediction["basket_total_thb"] and prediction["net_total_thb"]:
            basket = Decimal(prediction["basket_total_thb"].replace(",", ""))
            net = Decimal(prediction["net_total_thb"].replace(",", ""))
            prediction["discount_total_thb"] = f"{basket - net:,.2f}"
            record["fallback_actions"].append("discount_from_visible_basket_minus_net")
        prediction["payment_method"] = normalize_payment_method(
            " ".join(
                [
                    raw.get("payment_method", {}).get("text", ""),
                    raw.get("payment_method_retry", {}).get("text", ""),
                ]
            )
        )
    elif artifact_type == "vendor_invoice":
        prediction["invoice_period_start"] = parse_thai_date(
            raw.get("invoice_period_start", {}).get("text", "")
        )
        prediction["invoice_period_end"] = parse_thai_date(
            raw.get("invoice_period_end", {}).get("text", "")
        )
        if not prediction["invoice_period_end"]:
            prediction["invoice_period_end"] = month_end_from_start(prediction["invoice_period_start"])
            record["fallback_actions"].append("period_end_from_visible_start_calendar")
        prediction["business_event_date"] = parse_thai_date(
            raw.get("business_event_date", {}).get("text", "")
        )
        if not prediction["business_event_date"]:
            prediction["business_event_date"] = next_day_from_month_end(prediction["invoice_period_end"])
            record["fallback_actions"].append("event_date_from_visible_period_calendar")
        prediction["paid_amount_thb"] = parse_money(raw.get("paid_amount", {}).get("text", ""))
        public_month = Path(record["source_render_path"]).parent.name.replace("-", "")
        prediction["payment_id"] = parse_payment_id(
            [
                raw.get("payment_id", {}).get("text", ""),
                raw.get("payment_id_retry", {}).get("text", ""),
            ],
            public_month,
        )
    else:
        prediction["claim_reason"] = normalize_reason(raw.get("claim_reason", {}).get("text", ""))

    record["fallback_fields"] = [
        key
        for key, value in prediction.items()
        if value == "" and key not in record["unobservable_fields"]
    ]
    record["fallback_count"] = len(record["fallback_fields"])


def predict_tasks(model: Any, tasks: list[tuple[str, str, np.ndarray]], batch_size: int) -> list[tuple[str, str, tuple[str, float]]]:
    if not tasks:
        return []
    values = [
        recognition_value(result)
        for result in model.predict(
            input=[task[2] for task in tasks],
            batch_size=batch_size,
        )
    ]
    return [
        (artifact_id, label, value)
        for (artifact_id, label, _), value in zip(tasks, values)
    ]


def process_chunk(
    english_model: Any,
    thai_model: Any,
    chunk: list[tuple[str, str, Path, dict[str, str]]],
    output_dir: Path,
    batch_size: int,
) -> tuple[int, int]:
    records: dict[str, dict[str, Any]] = {}
    english_tasks: list[tuple[str, str, np.ndarray]] = []
    thai_tasks: list[tuple[str, str, np.ndarray]] = []
    for artifact_id, artifact_type, path, schema in chunk:
        record = new_record(artifact_id, artifact_type, path, schema)
        records[artifact_id] = record
        try:
            english, thai = prepare_crops(record, path)
            english_tasks.extend((artifact_id, label, image) for label, image in english)
            thai_tasks.extend((artifact_id, label, image) for label, image in thai)
        except Exception as error:  # noqa: BLE001
            record["errors"].append(str(error))

    for artifact_id, label, value in predict_tasks(english_model, english_tasks, batch_size):
        add_raw(records[artifact_id], label, value)
    for artifact_id, label, value in predict_tasks(thai_model, thai_tasks, batch_size):
        add_raw(records[artifact_id], label, value)

    failed = fallback = 0
    for artifact_id, record in records.items():
        finalize(record)
        fallback += int(record["fallback_count"] > 0)
        failed += int(bool(record["errors"]))
        atomic_write_json(output_dir / f"{artifact_id}.json", record)
    return failed, fallback


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    data_root = args.data_root.absolute()
    schemas = load_template(data_root / "submission_template_OCR.csv")
    paths = render_paths(data_root / BUNDLE_DIR)
    artifacts = [
        (artifact_id, artifact_type, path, schemas[artifact_id])
        for artifact_id, (artifact_type, path) in paths.items()
        if artifact_id in schemas
        and (not args.artifact_type or artifact_type == args.artifact_type)
        and (not args.artifact_id or artifact_id == args.artifact_id)
    ]
    artifacts.sort(key=lambda item: natural_key(item[0]))
    if args.limit_artifacts is not None:
        artifacts = artifacts[: args.limit_artifacts]
    output_dir = args.output_dir.absolute()
    pending: list[tuple[str, str, Path, dict[str, str]]] = []
    skipped = 0
    for artifact in artifacts:
        output = output_dir / f"{artifact[0]}.json"
        if output.exists() and not args.overwrite:
            record = json.loads(output.read_text(encoding="utf-8"))
            if record.get("checkpoint_version") == CHECKPOINT_VERSION and not record.get("errors"):
                skipped += 1
                continue
        pending.append(artifact)
    if not artifacts:
        raise SystemExit("No matching fixed-layout non-bank artifacts.")

    english_model = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        device=args.device,
        enable_mkldnn=not args.disable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    thai_model = TextRecognition(
        model_name="th_PP-OCRv5_mobile_rec",
        device=args.device,
        enable_mkldnn=not args.disable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    failed = fallback = 0
    started = time.perf_counter()
    for index in range(0, len(pending), args.chunk_size):
        chunk = pending[index : index + args.chunk_size]
        chunk_failed, chunk_fallback = process_chunk(
            english_model,
            thai_model,
            chunk,
            output_dir,
            args.batch_size,
        )
        failed += chunk_failed
        fallback += chunk_fallback
        print(
            f"processed={min(index + len(chunk), len(pending))}/{len(pending)} "
            f"failed={failed} fallback_artifacts={fallback}"
        )
    elapsed = time.perf_counter() - started
    print(
        f"completed={len(pending) - failed} skipped={skipped} failed={failed} "
        f"fallback_artifacts={fallback} elapsed_seconds={elapsed:.3f}"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_fixed_nonbank",
    )
    parser.add_argument("--artifact-type", choices=sorted(ARTIFACT_TYPES))
    parser.add_argument("--artifact-id")
    parser.add_argument("--limit-artifacts", type=int)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
