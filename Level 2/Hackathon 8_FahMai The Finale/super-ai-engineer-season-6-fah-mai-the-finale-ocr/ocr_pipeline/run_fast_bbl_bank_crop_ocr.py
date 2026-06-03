#!/usr/bin/env python3
"""Extract compact BBL operating statements with batched crop recognition."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_fast_compact_bank_crop_ocr import flatten_slots, parse_compact_amount
from run_fast_dense_bank_crop_ocr import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ROOT,
    account_id_from_artifact_id,
    atomic_write_json,
    format_amount,
    generic_fields_by_level,
    load_bank_schemas,
    natural_key,
    recognition_value,
)


CHECKPOINT_VERSION = 1
ROW_TOP = 230
ROW_HEIGHT = 42
DATE_BOX = (68, 158)
AMOUNT_BOX = (753, 922)
BALANCE_BOX = (923, 1093)
BALANCE_CONFIRM_BOX = (930, 1090)
HEADER_ACCOUNT_BOXES = [
    (300, 80, 450, 112),
    (285, 75, 470, 120),
    (290, 70, 460, 120),
]


@dataclass(frozen=True)
class RowSpec:
    level: int
    slot_id: str
    physical_page: int
    visual_row: int
    row_index: int


@dataclass(frozen=True)
class GenericRowSpec:
    level: int
    physical_page: int
    visual_row: int
    row_index: int
    fields: dict[str, str]


def bbl_render_groups(bundle: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    bank = bundle / "renders" / "bank_statement"
    for path in sorted(bank.rglob("*.png"), key=lambda item: natural_key(str(item))):
        match = re.fullmatch(r"(.+?)_(header|transactions_p(\d+))", path.stem)
        if not match:
            continue
        artifact_id, page_kind, page_number = match.groups()
        if not artifact_id.startswith("BS-OPER-"):
            continue
        item = groups.setdefault(artifact_id, {"header": None, "transactions": {}})
        if page_kind == "header":
            item["header"] = path
        else:
            item["transactions"][int(page_number)] = path

    output: dict[str, dict[str, Any]] = {}
    for artifact_id, item in groups.items():
        pages = item["transactions"]
        if not pages:
            continue
        with Image.open(pages[min(pages)]) as image:
            grayscale = image.convert("L")
            dark_table_pixels = sum(
                grayscale.getpixel((x, 135)) < 100
                for x in range(60, 1180)
            )
            if image.size == (1240, 1234) and dark_table_pixels > 800:
                output[artifact_id] = item
    return output


def crop_row(image: Image.Image, column: tuple[int, int], visual_row: int) -> np.ndarray:
    top = ROW_TOP + visual_row * ROW_HEIGHT
    return np.array(image.crop((column[0], top, column[1], top + ROW_HEIGHT)).convert("RGB"))


def row_specs(transaction_pages: dict[int, Path], schema: dict[str, str]) -> tuple[list[RowSpec], list[str]]:
    slots = flatten_slots(schema)
    specs: list[RowSpec] = []
    cursor = 0
    for physical_page in sorted(transaction_pages):
        first_visual_row = 1 if physical_page == 1 else 0
        capacity = 15 if physical_page == 1 else 16
        for row_index in range(min(capacity, len(slots) - cursor)):
            level, slot_id = slots[cursor]
            specs.append(
                RowSpec(
                    level=level,
                    slot_id=slot_id,
                    physical_page=physical_page,
                    visual_row=first_visual_row + row_index,
                    row_index=row_index,
                )
            )
            cursor += 1
    problems = []
    if cursor != len(slots):
        problems.append(f"visual capacity mapped rows={cursor} schema rows={len(slots)}")
    return specs, problems


def generic_row_specs(
    transaction_pages: dict[int, Path],
    schema: dict[str, str],
    occupied: list[RowSpec],
) -> tuple[list[GenericRowSpec], list[str]]:
    generic_fields, _ = generic_fields_by_level(schema)
    if not generic_fields:
        return [], []
    specs: list[GenericRowSpec] = []
    problems: list[str] = []
    for level, rows in sorted(generic_fields.items()):
        if level not in transaction_pages:
            problems.append(f"generic level={level} has no matching transaction page")
            continue
        for row_index, fields in sorted(rows.items()):
            specs.append(
                GenericRowSpec(
                    level=level,
                    physical_page=level,
                    visual_row=row_index,
                    row_index=row_index,
                    fields=fields,
                )
            )
    return specs, problems


def recover_account_number(model: Any, first_page: Path) -> tuple[str, str, float]:
    with Image.open(first_page) as image:
        crops = [np.array(image.crop(box).convert("RGB")) for box in HEADER_ACCOUNT_BOXES]
    best_text = ""
    best_score = 0.0
    for result in model.predict(input=crops, batch_size=len(crops)):
        text, score = recognition_value(result)
        if score > best_score:
            best_text, best_score = text, score
        match = re.search(r"\b\d{3}-\d-\d{5}-\d\b", text)
        if match:
            return match.group(0), text, round(score, 4)
    return "", best_text, round(best_score, 4)


def extract_artifact(
    model: Any,
    artifact_id: str,
    renders: dict[str, Any],
    schema: dict[str, str],
    batch_size: int,
) -> dict[str, Any]:
    specs, problems = row_specs(renders["transactions"], schema)
    generic_specs, generic_problems = generic_row_specs(renders["transactions"], schema, specs)
    problems.extend(generic_problems)
    if problems:
        return {"artifact_id": artifact_id, "errors": problems, "rows": []}

    crops: list[np.ndarray] = []
    with Image.open(renders["transactions"][1]) as image:
        crops.append(crop_row(image, BALANCE_BOX, 0))
    for spec in specs:
        with Image.open(renders["transactions"][spec.physical_page]) as image:
            crops.append(crop_row(image, DATE_BOX, spec.visual_row))
            crops.append(crop_row(image, AMOUNT_BOX, spec.visual_row))
            crops.append(crop_row(image, BALANCE_BOX, spec.visual_row))
            crops.append(crop_row(image, BALANCE_CONFIRM_BOX, spec.visual_row))
    for spec in generic_specs:
        with Image.open(renders["transactions"][spec.physical_page]) as image:
            crops.append(crop_row(image, DATE_BOX, spec.visual_row))
            crops.append(crop_row(image, AMOUNT_BOX, spec.visual_row))
            crops.append(crop_row(image, BALANCE_BOX, spec.visual_row))
            crops.append(crop_row(image, BALANCE_CONFIRM_BOX, spec.visual_row))

    started = time.perf_counter()
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    account_id = account_id_from_artifact_id(artifact_id)
    account_number, account_number_raw, account_number_score = recover_account_number(
        model,
        renders["transactions"][1],
    )
    opening_raw, opening_score = predictions[0]
    opening_balance = parse_compact_amount(opening_raw)
    previous_balance = opening_balance
    rows: list[dict[str, Any]] = []
    prediction_index = 1
    for spec in specs:
        date_raw, date_score = predictions[prediction_index]
        amount_raw, amount_score = predictions[prediction_index + 1]
        visual_balance_raw, visual_balance_score = predictions[prediction_index + 2]
        visual_balance_confirm_raw, visual_balance_confirm_score = predictions[prediction_index + 3]
        prediction_index += 4

        date_match = re.search(r"\b\d{2}-\d{2}-\d{2}\b", date_raw)
        amount = parse_compact_amount(amount_raw)
        candidates = [
            candidate
            for text in [visual_balance_raw, visual_balance_confirm_raw]
            if (candidate := parse_compact_amount(text)) is not None
        ]
        candidate_counts = Counter(candidates)
        consensus_balance = candidate_counts.most_common(1)[0][0] if candidate_counts else None
        calculated_balance = (
            previous_balance + amount
            if previous_balance is not None and amount is not None
            else None
        )
        inferred_amount = (
            consensus_balance - previous_balance
            if consensus_balance is not None and previous_balance is not None
            else None
        )
        fallback_actions: list[str] = []
        if (
            inferred_amount is not None
            and inferred_amount > 0
            and (
                amount is None
                or (
                    calculated_balance not in candidate_counts
                    and candidate_counts[consensus_balance] >= 2
                )
            )
        ):
            amount = inferred_amount
            fallback_actions.append("amount_from_balance_delta")
        balance = (
            previous_balance + amount
            if previous_balance is not None and amount is not None
            else None
        )
        fallback_reasons: list[str] = []
        if date_match is None:
            fallback_reasons.append("invalid_date")
        if amount is None:
            fallback_reasons.append("invalid_amount")
        if balance is None:
            fallback_reasons.append("missing_balance")
        if candidates and balance is not None and balance not in candidate_counts:
            fallback_reasons.append("arithmetic_balance_mismatch")
        rows.append(
            {
                "slot_id": spec.slot_id,
                "level": spec.level,
                "physical_page": spec.physical_page,
                "visual_row": spec.visual_row,
                "row_index": spec.row_index,
                "account_id": account_id,
                "business_event_date": date_match.group(0) if date_match else "",
                "transaction_type": "Deposit",
                "amount_thb": format_amount(abs(amount)) if amount is not None else "",
                "balance_after_thb": format_amount(balance) if balance is not None else "",
                "description": "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19 K PLUS",
                "date_raw": date_raw,
                "date_score": round(date_score, 4),
                "amount_raw": amount_raw,
                "amount_score": round(amount_score, 4),
                "visual_balance_raw": visual_balance_raw,
                "visual_balance_score": round(visual_balance_score, 4),
                "visual_balance_confirm_raw": visual_balance_confirm_raw,
                "visual_balance_confirm_score": round(visual_balance_confirm_score, 4),
                "fallback_reasons": fallback_reasons,
                "fallback_actions": fallback_actions,
            }
        )
        previous_balance = balance

    generic_values: dict[str, str] = {}
    generic_rows: list[dict[str, Any]] = []
    expected_blank_fields: list[str] = []
    for spec in generic_specs:
        date_raw, date_score = predictions[prediction_index]
        amount_raw, amount_score = predictions[prediction_index + 1]
        visual_balance_raw, visual_balance_score = predictions[prediction_index + 2]
        visual_balance_confirm_raw, visual_balance_confirm_score = predictions[prediction_index + 3]
        prediction_index += 4

        date_match = re.search(r"\b\d{2}-\d{2}-\d{2}\b", date_raw)
        amount = parse_compact_amount(amount_raw)
        candidates = [
            candidate
            for text in [visual_balance_raw, visual_balance_confirm_raw]
            if (candidate := parse_compact_amount(text)) is not None
        ]
        candidate_counts = Counter(candidates)
        consensus_balance = candidate_counts.most_common(1)[0][0] if candidate_counts else None
        closing_row = (
            amount is None
            and consensus_balance is not None
            and consensus_balance == previous_balance
        )
        inferred_amount = (
            consensus_balance - previous_balance
            if consensus_balance is not None and previous_balance is not None
            else None
        )
        fallback_actions: list[str] = []
        if (
            inferred_amount is not None
            and inferred_amount > 0
            and (
                amount is None
                or candidate_counts.get(previous_balance + amount, 0) == 0
            )
        ):
            amount = inferred_amount
            fallback_actions.append("generic_amount_from_balance_delta")
        balance = (
            previous_balance + amount
            if previous_balance is not None and amount is not None
            else consensus_balance
        )
        fallback_reasons: list[str] = []
        if date_match is None:
            fallback_reasons.append("generic_invalid_date")
        if amount is None and not closing_row:
            fallback_reasons.append("generic_invalid_amount")
        if balance is None:
            fallback_reasons.append("generic_missing_balance")
        date = date_match.group(0) if date_match else ""
        field_values = {
            "account_id": account_id,
            "date": date,
            "particulars": "\u0e22\u0e2d\u0e14\u0e22\u0e01\u0e44\u0e1b" if closing_row else "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19 K PLUS",
            "chq. no.": "",
            "withdrawal": "",
            "deposit": format_amount(abs(amount)) if amount is not None else "",
            "balance": format_amount(balance) if balance is not None else "",
            "via": "" if closing_row else "K PLUS",
        }
        for field, key in spec.fields.items():
            normalized = field.lower()
            generic_values[key] = field_values.get(normalized, "")
            if normalized in {"chq. no.", "withdrawal"} or (
                closing_row and normalized in {"deposit", "via"}
            ):
                expected_blank_fields.append(key)
        generic_rows.append(
            {
                "level": spec.level,
                "physical_page": spec.physical_page,
                "visual_row": spec.visual_row,
                "row_index": spec.row_index,
                "date_raw": date_raw,
                "date_score": round(date_score, 4),
                "amount_raw": amount_raw,
                "amount_score": round(amount_score, 4),
                "visual_balance_raw": visual_balance_raw,
                "visual_balance_score": round(visual_balance_score, 4),
                "visual_balance_confirm_raw": visual_balance_confirm_raw,
                "visual_balance_confirm_score": round(visual_balance_confirm_score, 4),
                "fallback_reasons": fallback_reasons,
                "fallback_actions": fallback_actions,
            }
        )
        previous_balance = balance

    return {
        "artifact_id": artifact_id,
        "layout": "compact_bbl_operating",
        "checkpoint_version": CHECKPOINT_VERSION,
        "engine": "paddle_crop_recognition",
        "model": "en_PP-OCRv5_mobile_rec",
        "account_id": account_id,
        "account_number": account_number,
        "account_number_raw": account_number_raw,
        "account_number_score": account_number_score,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "opening_balance": format_amount(opening_balance) if opening_balance is not None else "",
        "opening_balance_raw": opening_raw,
        "opening_balance_score": round(opening_score, 4),
        "rows": rows,
        "generic_values": generic_values,
        "generic_rows": generic_rows,
        "expected_blank_fields": sorted(expected_blank_fields),
        "fallback_rows": sum(bool(row["fallback_reasons"]) for row in [*rows, *generic_rows]),
        "errors": [] if opening_balance is not None else ["invalid opening balance"],
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    data_root = args.data_root.absolute()
    schemas = load_bank_schemas(data_root / "submission_template_OCR.csv")
    groups = bbl_render_groups(data_root / "fahmai_renders_with_json")
    artifact_ids = sorted(set(schemas) & set(groups), key=natural_key)
    if args.artifact_id:
        artifact_ids = [artifact_id for artifact_id in artifact_ids if artifact_id == args.artifact_id]
    if args.limit_artifacts is not None:
        artifact_ids = artifact_ids[: args.limit_artifacts]
    if not artifact_ids:
        raise SystemExit("No matching compact BBL operating-statement artifacts.")

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
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_bbl_bank",
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
