#!/usr/bin/env python3
"""Extract one-row sparse SCB and BBL statements with crop recognition."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_fast_compact_bank_crop_ocr import parse_compact_amount
from run_fast_dense_bank_crop_ocr import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ROOT,
    account_id_from_artifact_id,
    atomic_write_json,
    clean_description,
    format_amount,
    load_bank_schemas,
    natural_key,
    recognition_value,
    slot_ids_by_level,
)


CHECKPOINT_VERSION = 2
SCB_HEADER_ACCOUNT_BOXES = [
    (590, 260, 790, 305),
    (580, 250, 830, 310),
    (595, 262, 760, 300),
]
SCB_ROW_TOP = 126
SCB_ROW_HEIGHT = 44
SCB_DATE_BOX = (82, 231)
SCB_BALANCE_BOX = (791, 937)
SCB_DESCRIPTION_BOX = (937, 1159)
BBL_ROW_TOP = 230
BBL_ROW_HEIGHT = 42
BBL_DATE_BOX = (68, 158)
BBL_BALANCE_BOX = (923, 1093)
BBL_DESCRIPTION_BOX = (158, 438)
BBL_HEADER_ACCOUNT_BOXES = [
    (300, 80, 450, 112),
    (285, 75, 470, 120),
    (290, 70, 460, 120),
]


def sparse_render_groups(bundle: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    bank = bundle / "renders" / "bank_statement"
    for path in sorted(bank.rglob("*.png"), key=lambda item: natural_key(str(item))):
        match = re.fullmatch(r"(.+?)_(header|transactions_p(\d+))", path.stem)
        if not match:
            continue
        artifact_id, page_kind, page_number = match.groups()
        if not artifact_id.startswith(("BS-SCB-OPER-", "BS-BBL-OPER-")):
            continue
        item = groups.setdefault(artifact_id, {"header": None, "transactions": {}})
        if page_kind == "header":
            item["header"] = path
        else:
            item["transactions"][int(page_number)] = path
    return {
        artifact_id: item
        for artifact_id, item in groups.items()
        if item["header"] is not None and set(item["transactions"]) == {1}
    }


def crop_row(
    image: Image.Image,
    column: tuple[int, int],
    row_top: int,
    row_height: int,
    visual_row: int,
    scale: int = 1,
) -> np.ndarray:
    top = row_top + visual_row * row_height
    crop = image.crop((column[0], top, column[1], top + row_height)).convert("RGB")
    if scale != 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    return np.array(crop)


def recognize_account_number(
    model: Any,
    path: Path,
    boxes: list[tuple[int, int, int, int]],
    masked: bool,
) -> tuple[str, str, float]:
    with Image.open(path) as image:
        crops = [np.array(image.crop(box).convert("RGB")) for box in boxes]
    best_text = ""
    best_score = 0.0
    for result in model.predict(input=crops, batch_size=len(crops)):
        text, score = recognition_value(result)
        if score > best_score:
            best_text, best_score = text, score
        if masked:
            match = re.search(r"\b(\d{3})-(\d)-[Xx]{3,6}-(\d)\b", text)
            if match:
                return f"{match.group(1)}-{match.group(2)}-XXXXX-{match.group(3)}", text, round(score, 4)
        else:
            match = re.search(r"\b\d{3}-\d-\d{5}-\d\b", text)
            if match:
                return match.group(0), text, round(score, 4)
    return "", best_text, round(best_score, 4)


def only_slot(schema: dict[str, str]) -> tuple[int, str] | None:
    slots = [
        (level, slot_id)
        for level, level_slots in sorted(slot_ids_by_level(schema).items())
        for slot_id in level_slots
    ]
    return slots[0] if len(slots) == 1 else None


def parse_sparse_amount(text: str) -> Decimal | None:
    value = parse_compact_amount(text)
    if value is not None:
        return value
    normalized = re.sub(r"[,\s]", "", text)
    if re.fullmatch(r"-?\d+\.\d{2}", normalized):
        return Decimal(normalized)
    return None


def extract_artifact(
    model: Any,
    artifact_id: str,
    renders: dict[str, Any],
    schema: dict[str, str],
    batch_size: int,
) -> dict[str, Any]:
    slot = only_slot(schema)
    if slot is None:
        return {"artifact_id": artifact_id, "errors": ["expected exactly one transaction slot"], "rows": []}
    level, slot_id = slot
    is_scb = artifact_id.startswith("BS-SCB-")
    row_top = SCB_ROW_TOP if is_scb else BBL_ROW_TOP
    row_height = SCB_ROW_HEIGHT if is_scb else BBL_ROW_HEIGHT
    date_box = SCB_DATE_BOX if is_scb else BBL_DATE_BOX
    balance_box = SCB_BALANCE_BOX if is_scb else BBL_BALANCE_BOX
    description_box = SCB_DESCRIPTION_BOX if is_scb else BBL_DESCRIPTION_BOX
    transaction_page = renders["transactions"][1]
    with Image.open(transaction_page) as image:
        crops = [
            crop_row(image, balance_box, row_top, row_height, 0, scale=2 if is_scb else 1),
            crop_row(image, date_box, row_top, row_height, 1),
            crop_row(image, balance_box, row_top, row_height, 1),
            crop_row(image, description_box, row_top, row_height, 1),
        ]

    started = time.perf_counter()
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    opening_raw, opening_score = predictions[0]
    date_raw, date_score = predictions[1]
    balance_raw, balance_score = predictions[2]
    description_raw, description_score = predictions[3]
    opening_balance = parse_sparse_amount(opening_raw)
    balance = parse_sparse_amount(balance_raw)
    amount: Decimal | None = (
        balance - opening_balance
        if balance is not None and opening_balance is not None
        else None
    )
    date_match = re.search(r"\b\d{2}-\d{2}-\d{2}\b", date_raw)
    account_id = account_id_from_artifact_id(artifact_id)
    if is_scb:
        account_number, account_number_raw, account_number_score = recognize_account_number(
            model,
            renders["header"],
            SCB_HEADER_ACCOUNT_BOXES,
            masked=True,
        )
    else:
        account_number, account_number_raw, account_number_score = recognize_account_number(
            model,
            transaction_page,
            BBL_HEADER_ACCOUNT_BOXES,
            masked=False,
        )

    positive = amount is not None and amount >= 0
    description = clean_description(description_raw)
    if is_scb:
        transaction_type = "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19" if positive else "\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19"
        normalized_description = f"K PLUS {description}".strip()
        layout = "sparse_scb_direct"
    else:
        transaction_type = "Deposit" if positive else "Withdrawal"
        normalized_description = f"{'\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19' if positive else '\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19'} K PLUS"
        layout = "sparse_bbl_direct"
    fallback_reasons: list[str] = []
    if opening_balance is None:
        fallback_reasons.append("invalid_opening_balance")
    if date_match is None:
        fallback_reasons.append("invalid_date")
    if balance is None:
        fallback_reasons.append("invalid_balance")
    if amount is None:
        fallback_reasons.append("missing_balance_delta")
    row = {
        "slot_id": slot_id,
        "level": level,
        "physical_page": 1,
        "visual_row": 1,
        "row_index": 0,
        "account_id": account_id,
        "business_event_date": date_match.group(0) if date_match else "",
        "transaction_type": transaction_type,
        "amount_thb": format_amount(abs(amount)) if amount is not None else "",
        "balance_after_thb": format_amount(balance) if balance is not None else "",
        "description": normalized_description,
        "opening_balance_raw": opening_raw,
        "opening_balance_score": round(opening_score, 4),
        "date_raw": date_raw,
        "date_score": round(date_score, 4),
        "balance_raw": balance_raw,
        "balance_score": round(balance_score, 4),
        "description_raw": description_raw,
        "description_score": round(description_score, 4),
        "fallback_reasons": fallback_reasons,
        "fallback_actions": ["amount_from_balance_delta"] if amount is not None else [],
    }
    return {
        "artifact_id": artifact_id,
        "layout": layout,
        "checkpoint_version": CHECKPOINT_VERSION,
        "engine": "paddle_crop_recognition",
        "model": "en_PP-OCRv5_mobile_rec",
        "account_id": account_id,
        "account_number": account_number,
        "account_number_raw": account_number_raw,
        "account_number_score": account_number_score,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "opening_balance": format_amount(opening_balance) if opening_balance is not None else "",
        "level_render_paths": {str(level): str(transaction_page)},
        "rows": [row],
        "fallback_rows": int(bool(fallback_reasons)),
        "errors": [],
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    data_root = args.data_root.absolute()
    schemas = load_bank_schemas(data_root / "submission_template_OCR.csv")
    groups = sparse_render_groups(data_root / "fahmai_renders_with_json")
    artifact_ids = sorted(set(schemas) & set(groups), key=natural_key)
    if args.artifact_id:
        artifact_ids = [artifact_id for artifact_id in artifact_ids if artifact_id == args.artifact_id]
    if args.limit_artifacts is not None:
        artifact_ids = artifact_ids[: args.limit_artifacts]
    if not artifact_ids:
        raise SystemExit("No matching sparse direct-bank statement artifacts.")
    model = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        device=args.device,
        enable_mkldnn=not args.disable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    output_dir = args.output_dir.absolute()
    completed = failed = skipped = rows = fallback_rows = 0
    started = time.perf_counter()
    for index, artifact_id in enumerate(artifact_ids, start=1):
        output = output_dir / f"{artifact_id}.json"
        if output.exists() and not args.overwrite:
            record = json.loads(output.read_text(encoding="utf-8"))
            if not record.get("errors") and record.get("checkpoint_version") == CHECKPOINT_VERSION:
                skipped += 1
                rows += len(record.get("rows", []))
                fallback_rows += int(record.get("fallback_rows", 0))
                print(f"[{index}/{len(artifact_ids)}] skip {artifact_id}")
                continue
        record = extract_artifact(model, artifact_id, groups[artifact_id], schemas[artifact_id], args.batch_size)
        atomic_write_json(output, record)
        rows += len(record.get("rows", []))
        fallback_rows += int(record.get("fallback_rows", 0))
        if record.get("errors"):
            failed += 1
            print(f"[{index}/{len(artifact_ids)}] ERROR {artifact_id}: {record['errors']}")
        else:
            completed += 1
            print(
                f"[{index}/{len(artifact_ids)}] ok {artifact_id} rows={len(record['rows'])} "
                f"fallback={record['fallback_rows']} elapsed={record['elapsed_seconds']}s"
            )
    print(
        f"completed={completed} skipped={skipped} failed={failed} rows={rows} "
        f"fallback_rows={fallback_rows} elapsed_seconds={round(time.perf_counter() - started, 3)}"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_sparse_bank",
    )
    parser.add_argument("--artifact-id")
    parser.add_argument("--limit-artifacts", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
