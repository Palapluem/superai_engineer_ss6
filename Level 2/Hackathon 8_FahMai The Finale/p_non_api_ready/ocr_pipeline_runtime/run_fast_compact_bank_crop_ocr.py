#!/usr/bin/env python3
"""Extract compact SCB operating statements with batched crop recognition."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_fast_dense_bank_crop_ocr import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ROOT,
    account_id_from_artifact_id,
    atomic_write_json,
    clean_description,
    date_from_description,
    format_amount,
    load_bank_schemas,
    natural_key,
    parse_decimal,
    recognition_value,
    slot_ids_by_level,
)


ROW_TOP = 126
ROW_HEIGHT = 44
CHECKPOINT_VERSION = 5
DATE_BOX = (82, 231)
OPENING_BALANCE_BOX = (795, 940)
AMOUNT_BOX = (652, 791)
BALANCE_BOX = (791, 937)
BALANCE_CONFIRM_BOX = (785, 945)
DESCRIPTION_BOX = (937, 1159)
AMOUNT_RETRY_VARIANTS = [
    (AMOUNT_BOX, 2),
    ((645, 800), 1),
    ((660, 785), 1),
    ((660, 785), 2),
]
BALANCE_RETRY_VARIANTS = [
    ((795, 940), 1),
    ((791, 940), 2),
    ((785, 945), 1),
]
WIDE_AMOUNT_RETRY_VARIANTS = [
    ((610, 800), 1),
    ((625, 800), 1),
    ((640, 800), 1),
]
HEADER_ACCOUNT_BOXES = [
    (590, 260, 790, 305),
    (580, 250, 830, 310),
    (595, 262, 760, 300),
]


@dataclass(frozen=True)
class RowSpec:
    kind: str
    level: int
    slot_id: str
    physical_page: int
    visual_row: int
    row_index: int


def compact_render_groups(bundle: Path) -> dict[str, dict[str, Any]]:
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
                grayscale.getpixel((x, 43)) < 100
                for x in range(60, 1180)
            )
            if image.size == (1240, 1234) and dark_table_pixels > 800:
                output[artifact_id] = item
    return output


def flatten_slots(schema: dict[str, str]) -> list[tuple[int, str]]:
    levels = slot_ids_by_level(schema)
    return [
        (level, slot_id)
        for level in sorted(levels)
        for slot_id in levels[level]
    ]


def crop_row(
    image: Image.Image,
    column: tuple[int, int],
    visual_row: int,
    scale: int = 1,
) -> np.ndarray:
    top = ROW_TOP + visual_row * ROW_HEIGHT
    crop = image.crop((column[0], top, column[1], top + ROW_HEIGHT)).convert("RGB")
    if scale != 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    return np.array(crop)


def row_specs(transaction_pages: dict[int, Path], schema: dict[str, str]) -> tuple[list[RowSpec], list[str]]:
    slots = flatten_slots(schema)
    if not slots:
        return [], ["schema has no transaction slots"]
    specs: list[RowSpec] = []
    cursor = 0
    for physical_page in sorted(transaction_pages):
        first_visual_row = 1 if physical_page == 1 else 0
        capacity = 15 if physical_page == 1 else 16
        for row_index in range(min(capacity, len(slots) - cursor)):
            level, slot_id = slots[cursor]
            specs.append(
                RowSpec(
                    kind="transaction",
                    level=level,
                    slot_id=slot_id,
                    physical_page=physical_page,
                    visual_row=first_visual_row + row_index,
                    row_index=row_index,
                )
            )
            cursor += 1
    problems: list[str] = []
    if cursor != len(slots):
        problems.append(f"visual capacity mapped rows={cursor} schema rows={len(slots)}")
    return specs, problems


def recover_account_number(model: Any, header: Path) -> tuple[str, str, float]:
    with Image.open(header) as image:
        crops = [np.array(image.crop(box).convert("RGB")) for box in HEADER_ACCOUNT_BOXES]
    best_text = ""
    best_score = 0.0
    for result in model.predict(input=crops, batch_size=len(crops)):
        text, score = recognition_value(result)
        if score > best_score:
            best_text, best_score = text, score
        match = re.search(r"\b(\d{3})-(\d)-[Xx]{3,6}-(\d)\b", text)
        if match:
            return f"{match.group(1)}-{match.group(2)}-XXXXX-{match.group(3)}", text, round(score, 4)
    return "", best_text, round(best_score, 4)


def opening_date_from_artifact_id(artifact_id: str) -> str:
    match = re.search(r"-(256[78])-(\d{2})$", artifact_id)
    if not match:
        return ""
    buddhist_year, month = match.groups()
    year = int(buddhist_year) - 543
    return f"01-{month}-{year % 100:02d}"


def parse_compact_amount(text: str) -> Decimal | None:
    compact = text.replace(" ", "")
    match = re.search(r"(\d[\d,]*)\.+(\d{1,3})", compact)
    if not match:
        return None
    integer, fraction = match.groups()
    if "," in integer and not re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer):
        trimmed_integer = integer[:-1] if integer.endswith("0") else ""
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", trimmed_integer):
            return None
        integer = trimmed_integer
    try:
        return Decimal(f"{integer.replace(',', '')}.{fraction[:2].ljust(2, '0')}")
    except InvalidOperation:
        return None


def retry_invalid_amounts(
    model: Any,
    rows: list[dict[str, Any]],
    specs: list[RowSpec],
    transaction_pages: dict[int, Path],
    batch_size: int,
) -> None:
    invalid = [
        (row, spec)
        for row, spec in zip(rows, specs)
        if row["amount"] is None
    ]
    if not invalid:
        return
    crops: list[np.ndarray] = []
    for _, spec in invalid:
        with Image.open(transaction_pages[spec.physical_page]) as image:
            crops.extend(
                crop_row(image, column, spec.visual_row, scale=scale)
                for column, scale in AMOUNT_RETRY_VARIANTS
            )
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    for index, (row, _) in enumerate(invalid):
        retries = predictions[
            index * len(AMOUNT_RETRY_VARIANTS) : (index + 1) * len(AMOUNT_RETRY_VARIANTS)
        ]
        row["amount_retry_raw"] = [text for text, _ in retries]
        for text, score in retries:
            amount = parse_compact_amount(text)
            if amount is not None:
                row["amount"] = amount
                row["amount_raw"] = text
                row["amount_score"] = score
                row["fallback_actions"].append("amount_crop_retry")
                break


def retry_invalid_balances(
    model: Any,
    rows: list[dict[str, Any]],
    specs: list[RowSpec],
    transaction_pages: dict[int, Path],
    batch_size: int,
) -> None:
    invalid = [
        (row, spec)
        for row, spec in zip(rows, specs)
        if row["visual_balance"] != row["visual_balance_confirm"]
    ]
    if not invalid:
        return
    crops: list[np.ndarray] = []
    for _, spec in invalid:
        with Image.open(transaction_pages[spec.physical_page]) as image:
            crops.extend(
                crop_row(image, column, spec.visual_row, scale=scale)
                for column, scale in BALANCE_RETRY_VARIANTS
            )
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    for index, (row, _) in enumerate(invalid):
        retries = predictions[
            index * len(BALANCE_RETRY_VARIANTS) : (index + 1) * len(BALANCE_RETRY_VARIANTS)
        ]
        row["visual_balance_retry_raw"] = [text for text, _ in retries]
        for text, score in retries:
            balance = parse_compact_amount(text)
            if balance is not None:
                row["visual_balance"] = balance
                row["visual_balance_raw"] = text
                row["visual_balance_score"] = score
                row["fallback_actions"].append("balance_crop_retry")
                break


def recover_missing_dates(
    model: Any,
    rows: list[dict[str, Any]],
    transaction_pages: dict[int, Path],
    batch_size: int,
) -> None:
    unresolved = [row for row in rows if not row["business_event_date"]]
    if not unresolved:
        return
    crops: list[np.ndarray] = []
    for row in unresolved:
        with Image.open(transaction_pages[row["physical_page"]]) as image:
            crops.append(crop_row(image, DATE_BOX, row["visual_row"]))
    for row, result in zip(unresolved, model.predict(input=crops, batch_size=batch_size)):
        text, score = recognition_value(result)
        row["date_raw"] = text
        row["date_score"] = round(score, 4)
        match = re.search(r"\b(\d{2}-\d{2}-\d{2})\b", text)
        if not match:
            compact_match = re.search(r"\b(\d{2})(\d{2})-(\d{2})\b", text)
            if compact_match:
                row["business_event_date"] = "-".join(compact_match.groups())
                row["fallback_actions"].append("date_crop_compact_normalized")
                continue
        if match:
            row["business_event_date"] = match.group(1)
            row["fallback_actions"].append("date_crop_ocr")
        else:
            row["fallback_reasons"].append("invalid_date_crop")


def decimal_value(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "")) if value else None
    except InvalidOperation:
        return None


def visible_balance_consensus(row: dict[str, Any]) -> Decimal | None:
    texts = [
        row.get("visual_balance_raw", ""),
        row.get("visual_balance_confirm_raw", ""),
        *row.get("visual_balance_retry_raw", []),
    ]
    candidates = [
        candidate
        for text in texts
        if (candidate := parse_compact_amount(text)) is not None
    ]
    counts = Counter(candidates)
    if not counts:
        return None
    value, count = counts.most_common(1)[0]
    return value if count >= 2 else None


def visible_balance_candidates(row: dict[str, Any]) -> Counter[Decimal]:
    texts = [
        row.get("visual_balance_raw", ""),
        row.get("visual_balance_confirm_raw", ""),
        *row.get("visual_balance_retry_raw", []),
    ]
    return Counter(
        candidate
        for text in texts
        if (candidate := parse_compact_amount(text)) is not None
    )


def retry_arithmetic_mismatch_amounts(
    model: Any,
    rows: list[dict[str, Any]],
    specs: list[RowSpec],
    transaction_pages: dict[int, Path],
    batch_size: int,
) -> None:
    pending = [
        (row, spec)
        for row, spec in zip(rows, specs)
        if "arithmetic_balance_mismatch" in row["fallback_reasons"]
    ]
    if not pending:
        return
    crops: list[np.ndarray] = []
    for _, spec in pending:
        with Image.open(transaction_pages[spec.physical_page]) as image:
            crops.extend(
                crop_row(image, column, spec.visual_row, scale=scale)
                for column, scale in WIDE_AMOUNT_RETRY_VARIANTS
            )
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    for index, (row, _) in enumerate(pending):
        retries = predictions[
            index * len(WIDE_AMOUNT_RETRY_VARIANTS) : (index + 1) * len(WIDE_AMOUNT_RETRY_VARIANTS)
        ]
        row["wide_amount_retry_raw"] = [text for text, _ in retries]
        row["wide_amount_retry_score"] = [round(score, 4) for _, score in retries]


def is_left_clipped_balance(expected: Decimal, candidates: Counter[Decimal]) -> bool:
    expected_digits = f"{expected:.2f}".split(".", 1)[0]
    return any(
        candidate != expected
        and expected_digits.endswith(f"{candidate:.2f}".split(".", 1)[0])
        for candidate in candidates
    )


def reconcile_visible_amount_chain(rows: list[dict[str, Any]], opening_balance: Decimal | None) -> None:
    previous_balance = opening_balance
    transaction_rows = [row for row in rows if row.get("transaction_type")]
    for row in transaction_rows:
        amount = decimal_value(row.get("amount_thb", ""))
        balance = decimal_value(row.get("balance_after_thb", ""))
        raw_amount = parse_compact_amount(row.get("amount_raw", ""))
        wide_amounts = [
            candidate
            for text in row.get("wide_amount_retry_raw", [])
            if (candidate := parse_compact_amount(text)) is not None
        ]
        amount_counts = Counter(
            candidate
            for candidate in [amount, raw_amount, *wide_amounts]
            if candidate is not None and candidate > 0
        )
        balance_counts = visible_balance_candidates(row)
        matches = [
            candidate
            for candidate in amount_counts
            if previous_balance is not None
            and previous_balance + candidate in balance_counts
        ]
        selected_amount: Decimal | None = None
        selected_balance: Decimal | None = None
        if matches and previous_balance is not None:
            selected_amount = max(
                matches,
                key=lambda candidate: (
                    balance_counts[previous_balance + candidate],
                    amount_counts[candidate],
                    candidate == amount,
                ),
            )
            selected_balance = previous_balance + selected_amount
            retry_support = Counter(wide_amounts)[selected_amount]
            balance_support = balance_counts[selected_balance]
            is_existing_chain = selected_amount == amount
            if not is_existing_chain and retry_support < 2 and balance_support < 2:
                selected_amount = None
                selected_balance = None

        if selected_amount is not None and selected_balance is not None:
            if amount != selected_amount:
                row["amount_thb"] = format_amount(selected_amount)
                if selected_amount in wide_amounts:
                    row["fallback_actions"].append("amount_from_wide_crop_balance_match")
                else:
                    row["fallback_actions"].append("amount_from_raw_crop_visible_balance_chain")
            if balance != selected_balance:
                row["balance_after_thb"] = format_amount(selected_balance)
                row["fallback_actions"].append("balance_reconciled_from_visible_chain")
            row["fallback_reasons"] = [
                reason
                for reason in row["fallback_reasons"]
                if reason not in {"invalid_amount", "missing_balance", "arithmetic_balance_mismatch"}
            ]
            balance = selected_balance
        elif previous_balance is not None and amount is not None:
            calculated_balance = previous_balance + amount
            if balance != calculated_balance:
                row["balance_after_thb"] = format_amount(calculated_balance)
                row["fallback_actions"].append("balance_reconciled_from_amount_chain")
            balance = calculated_balance
            wide_counts = Counter(wide_amounts)
            if (
                "arithmetic_balance_mismatch" in row["fallback_reasons"]
                and raw_amount == amount
                and wide_counts[amount] >= 2
            ):
                reason = (
                    "visual_balance_left_clipped"
                    if is_left_clipped_balance(calculated_balance, balance_counts)
                    else "visual_balance_crop_unreliable_amount_consensus"
                )
                row["fallback_reasons"] = [
                    reason if item == "arithmetic_balance_mismatch" else item
                    for item in row["fallback_reasons"]
                ]
                row["fallback_actions"].append("accepted_wide_amount_consensus")
        previous_balance = balance


def repair_visible_balance_chain(rows: list[dict[str, Any]], opening_balance: Decimal | None) -> None:
    transaction_rows = [row for row in rows if row.get("transaction_type")]
    for _ in range(3):
        previous_balance = opening_balance
        for row in transaction_rows:
            amount = decimal_value(row.get("amount_thb", ""))
            balance = decimal_value(row.get("balance_after_thb", ""))
            consensus = visible_balance_consensus(row)
            calculated = (
                previous_balance + amount
                if previous_balance is not None and amount is not None
                else None
            )
            if balance is None and calculated is not None:
                balance = calculated
                row["balance_after_thb"] = format_amount(balance)
                row["fallback_actions"].append("balance_from_visible_amount_chain")
            if balance is None and consensus is not None:
                balance = consensus
                row["balance_after_thb"] = format_amount(balance)
                row["fallback_actions"].append("balance_reanchored_from_visible_crop")
            if (
                amount is None
                and previous_balance is not None
                and balance is not None
                and 0 < balance - previous_balance < Decimal("1000000")
            ):
                amount = balance - previous_balance
                row["amount_thb"] = format_amount(amount)
                row["fallback_actions"].append("amount_from_visible_balance_chain")
            previous_balance = balance

        next_balance: Decimal | None = None
        next_amount: Decimal | None = None
        for row in reversed(transaction_rows):
            amount = decimal_value(row.get("amount_thb", ""))
            balance = decimal_value(row.get("balance_after_thb", ""))
            if balance is None and next_balance is not None and next_amount is not None:
                balance = next_balance - next_amount
                row["balance_after_thb"] = format_amount(balance)
                row["fallback_actions"].append("balance_from_next_visible_chain")
            next_balance = balance
            next_amount = amount

    for row in transaction_rows:
        if row.get("amount_thb"):
            row["fallback_reasons"] = [
                reason for reason in row["fallback_reasons"] if reason != "invalid_amount"
            ]
        if row.get("balance_after_thb"):
            row["fallback_reasons"] = [
                reason for reason in row["fallback_reasons"] if reason != "missing_balance"
            ]


def extract_artifact(
    model: Any,
    header_model: Any,
    artifact_id: str,
    renders: dict[str, Any],
    schema: dict[str, str],
    batch_size: int,
    min_confidence: float,
) -> dict[str, Any]:
    specs, problems = row_specs(renders["transactions"], schema)
    if problems:
        return {"artifact_id": artifact_id, "errors": problems, "rows": []}
    crops: list[np.ndarray] = []
    with Image.open(renders["transactions"][1]) as image:
        crops.append(crop_row(image, OPENING_BALANCE_BOX, 0, scale=2))
    for spec in specs:
        if spec.kind == "opening_balance":
            continue
        with Image.open(renders["transactions"][spec.physical_page]) as image:
            crops.append(crop_row(image, AMOUNT_BOX, spec.visual_row))
            crops.append(crop_row(image, BALANCE_BOX, spec.visual_row))
            crops.append(crop_row(image, BALANCE_CONFIRM_BOX, spec.visual_row))
            crops.append(crop_row(image, DESCRIPTION_BOX, spec.visual_row))

    started = time.perf_counter()
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    account_id = account_id_from_artifact_id(artifact_id)
    account_number, account_number_raw, account_number_score = recover_account_number(
        header_model,
        renders["header"],
    )
    rows: list[dict[str, Any]] = []
    opening_raw, opening_score = predictions[0]
    opening_balance = parse_decimal(opening_raw)
    transaction_specs = [spec for spec in specs if spec.kind == "transaction"]
    if specs and specs[0].kind == "opening_balance":
        opening_spec = specs[0]
        rows.append(
            {
                "slot_id": opening_spec.slot_id,
                "level": opening_spec.level,
                "physical_page": 1,
                "visual_row": 0,
                "row_index": 0,
                "account_id": account_id,
                "business_event_date": opening_date_from_artifact_id(artifact_id),
                "transaction_type": "",
                "amount_thb": format_amount(opening_balance) if opening_balance is not None else "",
                "balance_after_thb": "\u0e22\u0e2d\u0e14\u0e22\u0e01\u0e21\u0e32",
                "description": "",
                "opening_balance_raw": opening_raw,
                "opening_balance_score": round(opening_score, 4),
                "fallback_reasons": [] if opening_balance is not None else ["invalid_opening_balance"],
                "fallback_actions": [],
            }
        )
    prediction_index = 1
    transaction_rows: list[dict[str, Any]] = []
    for spec in transaction_specs:
        amount_raw, amount_score = predictions[prediction_index]
        visual_balance_raw, visual_balance_score = predictions[prediction_index + 1]
        visual_balance_confirm_raw, visual_balance_confirm_score = predictions[prediction_index + 2]
        description_raw, description_score = predictions[prediction_index + 3]
        prediction_index += 4
        description = re.sub(r"\s+", " ", description_raw).strip()
        transaction_rows.append(
            {
                "slot_id": spec.slot_id,
                "level": spec.level,
                "physical_page": spec.physical_page,
                "visual_row": spec.visual_row,
                "row_index": spec.row_index,
                "account_id": account_id,
                "business_event_date": date_from_description(description),
                "transaction_type": "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19",
                "amount_thb": "",
                "balance_after_thb": "",
                "description": description,
                "amount": parse_compact_amount(amount_raw),
                "amount_raw": amount_raw,
                "amount_score": amount_score,
                "visual_balance": parse_compact_amount(visual_balance_raw),
                "visual_balance_raw": visual_balance_raw,
                "visual_balance_score": visual_balance_score,
                "visual_balance_confirm": parse_compact_amount(visual_balance_confirm_raw),
                "visual_balance_confirm_raw": visual_balance_confirm_raw,
                "visual_balance_confirm_score": visual_balance_confirm_score,
                "description_raw": description_raw,
                "description_score": round(description_score, 4),
                "fallback_reasons": [],
                "fallback_actions": [],
            }
        )
    rows.extend(transaction_rows)
    retry_invalid_amounts(model, transaction_rows, transaction_specs, renders["transactions"], batch_size)
    retry_invalid_balances(model, transaction_rows, transaction_specs, renders["transactions"], batch_size)
    previous_balance = opening_balance
    for row in transaction_rows:
        amount = row.pop("amount")
        visual_balance = row.pop("visual_balance")
        visual_balance_confirm = row.pop("visual_balance_confirm")
        visual_candidates = [
            candidate
            for text in [
                row["visual_balance_raw"],
                row["visual_balance_confirm_raw"],
                *row.get("visual_balance_retry_raw", []),
            ]
            if (candidate := parse_compact_amount(text)) is not None
        ]
        candidate_counts = Counter(visual_candidates)
        consensus_balance = candidate_counts.most_common(1)[0][0] if candidate_counts else None
        consensus_count = candidate_counts[consensus_balance] if consensus_balance is not None else 0
        calculated_balance = (
            previous_balance + amount
            if amount is not None and previous_balance is not None
            else None
        )
        inferred_amount = (
            consensus_balance - previous_balance
            if consensus_balance is not None and previous_balance is not None
            else None
        )
        if (
            inferred_amount is not None
            and inferred_amount > 0
            and (
                amount is None
                or (
                    calculated_balance not in candidate_counts
                    and consensus_count >= 2
                )
            )
        ):
            amount = inferred_amount
            row["fallback_actions"].append("amount_from_balance_delta")
        balance = previous_balance + amount if amount is not None and previous_balance is not None else None
        if amount is None:
            row["fallback_reasons"].append("invalid_amount")
        if balance is None:
            row["fallback_reasons"].append("missing_balance")
        if row["amount_score"] < min_confidence:
            row["fallback_reasons"].append("low_amount_confidence")
        if row["description_score"] < min_confidence:
            row["fallback_reasons"].append("low_description_confidence")
        if visual_candidates and balance is not None and balance not in candidate_counts:
            row["fallback_reasons"].append("arithmetic_balance_mismatch")
        row["amount_score"] = round(row["amount_score"], 4)
        row["visual_balance_score"] = round(row["visual_balance_score"], 4)
        row["visual_balance_confirm_score"] = round(row["visual_balance_confirm_score"], 4)
        row["amount_thb"] = format_amount(abs(amount)) if amount is not None else ""
        row["balance_after_thb"] = format_amount(balance) if balance is not None else ""
        previous_balance = balance
    retry_arithmetic_mismatch_amounts(
        model,
        transaction_rows,
        transaction_specs,
        renders["transactions"],
        batch_size,
    )
    reconcile_visible_amount_chain(rows, opening_balance)
    recover_missing_dates(model, rows, renders["transactions"], batch_size)
    repair_visible_balance_chain(rows, opening_balance)
    for row in transaction_rows:
        row["description"] = f"K PLUS {row['description']}".strip()
    return {
        "artifact_id": artifact_id,
        "layout": "compact_scb_operating",
        "checkpoint_version": CHECKPOINT_VERSION,
        "engine": "paddle_crop_recognition",
        "model": "en_PP-OCRv5_mobile_rec",
        "account_id": account_id,
        "account_number": account_number,
        "account_number_raw": account_number_raw,
        "account_number_score": account_number_score,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "opening_balance": format_amount(opening_balance) if opening_balance is not None else "",
        "rows": rows,
        "fallback_rows": sum(bool(row["fallback_reasons"]) for row in rows),
        "errors": [],
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    data_root = args.data_root.absolute()
    schemas = load_bank_schemas(data_root / "submission_template_OCR.csv")
    groups = compact_render_groups(data_root / "fahmai_renders_with_json")
    artifact_ids = sorted(set(schemas) & set(groups), key=natural_key)
    if args.artifact_id:
        artifact_ids = [artifact_id for artifact_id in artifact_ids if artifact_id == args.artifact_id]
    if args.limit_artifacts is not None:
        artifact_ids = artifact_ids[: args.limit_artifacts]
    if not artifact_ids:
        raise SystemExit("No matching compact operating statement artifacts.")

    model = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        device=args.device,
        enable_mkldnn=not args.disable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    header_model = TextRecognition(
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
        record = extract_artifact(
            model=model,
            header_model=header_model,
            artifact_id=artifact_id,
            renders=groups[artifact_id],
            schema=schemas[artifact_id],
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
        )
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
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_compact_bank",
    )
    parser.add_argument("--artifact-id")
    parser.add_argument("--limit-artifacts", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
