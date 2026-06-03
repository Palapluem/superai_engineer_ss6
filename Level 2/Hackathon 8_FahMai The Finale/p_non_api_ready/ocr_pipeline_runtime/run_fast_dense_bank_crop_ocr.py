#!/usr/bin/env python3
"""Extract dense KBank statement rows with batched crop recognition.

This production candidate reads public render files and the public submission
template only. It does not read sidecars, provenance JSON, or enterprise data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
SHORT_ROOT = Path(r"C:\fahmai_ocr_data")
DEFAULT_DATA_ROOT = SHORT_ROOT if SHORT_ROOT.exists() else DEFAULT_ROOT
TX_FIELDS = [
    "business_event_date",
    "transaction_type",
    "amount_thb",
    "balance_after_thb",
    "description",
    "account_id",
]
ROW_TOP = 128
ROW_HEIGHT = 59
BALANCE_BOX = (1344, 1654)
DESCRIPTION_BOX = (2010, 2670)
DATE_BOX = (324, 464)
AMOUNT_BOX = (1004, 1344)
HEADER_ACCOUNT_NUMBER_BOX = (1965, 422, 2300, 462)


@dataclass(frozen=True)
class CropTask:
    kind: str
    level: int
    physical_page: int
    slot_id: str
    row_index: int
    image: np.ndarray


@dataclass(frozen=True)
class GenericCropTask:
    kind: str
    level: int
    physical_page: int
    row_index: int
    image: np.ndarray


def natural_key(value: str) -> list[Any]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value)]


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def account_id_from_artifact_id(artifact_id: str) -> str:
    match = re.fullmatch(r"BS-(.+)-256[78]-\d{2}", artifact_id)
    return match.group(1) if match else ""


def load_bank_schemas(template: Path) -> dict[str, dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    schemas: dict[str, dict[str, str]] = {}
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            artifact_id = row["artifact_id"]
            if artifact_id.startswith("BS-"):
                schemas[artifact_id] = json.loads(row["pred_json"])
    return schemas


def dense_render_groups(bundle: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    bank = bundle / "renders" / "bank_statement"
    for path in sorted(bank.rglob("*.png"), key=lambda item: natural_key(str(item))):
        match = re.fullmatch(r"(.+?)_(header|transactions_p(\d+))", path.stem)
        if not match:
            continue
        artifact_id, page_kind, page_number = match.groups()
        item = groups.setdefault(artifact_id, {"header": None, "transactions": {}})
        if page_kind == "header":
            item["header"] = path
        else:
            item["transactions"][int(page_number)] = path

    dense: dict[str, dict[str, Any]] = {}
    for artifact_id, item in groups.items():
        pages = item["transactions"]
        if not pages:
            continue
        first_page = pages[min(pages)]
        with Image.open(first_page) as image:
            if image.size == (2968, 2850):
                dense[artifact_id] = item
    return dense


def slot_ids_by_level(schema: dict[str, str]) -> dict[int, list[str]]:
    levels: dict[int, list[str]] = {}
    for key in schema:
        match = re.fullmatch(r"L(\d+)_(BT-[^_]+)_business_event_date", key)
        if match:
            levels.setdefault(int(match.group(1)), []).append(match.group(2))
    return levels


def generic_fields_by_level(
    schema: dict[str, str],
) -> tuple[dict[int, dict[int, dict[str, str]]], dict[str, str]]:
    levels: dict[int, dict[int, dict[str, str]]] = {}
    header_fields: dict[str, str] = {}
    for key in schema:
        match = re.fullmatch(r"L(\d+)_(?:transactions_)?L(\d+)_(.+)", key)
        if match:
            level, row_index, field = match.groups()
            levels.setdefault(int(level), {}).setdefault(int(row_index), {})[field] = key
            continue
        match = re.fullmatch(r"L(\d+)_account_id", key)
        if match and int(match.group(1)) > 0:
            header_fields[key] = "account_id"
    return levels, header_fields


def crop_row(image: Image.Image, column: tuple[int, int], visual_row: int) -> np.ndarray:
    top = ROW_TOP + visual_row * ROW_HEIGHT
    return np.array(image.crop((column[0], top, column[1], top + ROW_HEIGHT)).convert("RGB"))


def build_crop_tasks(
    transaction_pages: dict[int, Path],
    slots: dict[int, list[str]],
    schema_levels: list[int] | None = None,
) -> tuple[list[CropTask], dict[int, Path], dict[int, Path], list[Path], list[str]]:
    tasks: list[CropTask] = []
    problems: list[str] = []
    levels = sorted(slots)
    render_levels = sorted(schema_levels if schema_levels is not None else levels)
    # The public template level numbers follow lexicographic output-path order:
    # p1, p10, ..., p2. Preserve that ordering instead of numeric page order.
    pages = sorted(transaction_pages.values(), key=lambda path: path.name)
    if len(pages) < len(render_levels):
        problems.append(
            f"page count differs renders={len(pages)} schema_levels={len(render_levels)}"
        )
        return tasks, {}, {}, [], problems
    # Some public artifacts include rendered pages beyond the schema prefix.
    # Keep them auditable but do not create rows that the submission omits.
    ignored_pages = pages[len(render_levels) :]
    pages = pages[: len(render_levels)]
    all_level_pages = dict(zip(render_levels, pages))
    level_pages = {level: all_level_pages[level] for level in levels}

    for level in levels:
        path = level_pages[level]
        page_match = re.search(r"_transactions_p(\d+)\.png$", path.name)
        if not page_match:
            problems.append(f"cannot parse physical page number path={path}")
            continue
        physical_page = int(page_match.group(1))
        with Image.open(path) as image:
            if image.size != (2968, 2850):
                problems.append(f"unexpected dense page size={image.size} path={path}")
                continue
            # Dense first pages include a visible opening balance row before the
            # first transaction. Later pages begin directly with transactions.
            if level == 1:
                tasks.append(
                    CropTask(
                        kind="opening_balance",
                        level=level,
                        physical_page=physical_page,
                        slot_id="",
                        row_index=-1,
                        image=crop_row(image, BALANCE_BOX, 0),
                    )
                )
            for row_index, slot_id in enumerate(slots[level]):
                visual_row = row_index + (1 if level == 1 else 0)
                tasks.append(
                    CropTask(
                        kind="balance",
                        level=level,
                        physical_page=physical_page,
                        slot_id=slot_id,
                        row_index=row_index,
                        image=crop_row(image, BALANCE_BOX, visual_row),
                    )
                )
                tasks.append(
                    CropTask(
                        kind="description",
                        level=level,
                        physical_page=physical_page,
                        slot_id=slot_id,
                        row_index=row_index,
                        image=crop_row(image, DESCRIPTION_BOX, visual_row),
                    )
                )
    return tasks, level_pages, all_level_pages, ignored_pages, problems


def build_generic_crop_tasks(
    level_pages: dict[int, Path],
    generic_fields: dict[int, dict[int, dict[str, str]]],
) -> tuple[list[GenericCropTask], list[str]]:
    tasks: list[GenericCropTask] = []
    problems: list[str] = []
    for level, rows in sorted(generic_fields.items()):
        path = level_pages[level]
        page_match = re.search(r"_transactions_p(\d+)\.png$", path.name)
        if not page_match:
            problems.append(f"cannot parse physical page number path={path}")
            continue
        physical_page = int(page_match.group(1))
        with Image.open(path) as image:
            if image.size != (2968, 2850):
                problems.append(f"unexpected dense page size={image.size} path={path}")
                continue
            for row_index in sorted(rows):
                for kind, column in [
                    ("date", DATE_BOX),
                    ("balance", BALANCE_BOX),
                    ("details", DESCRIPTION_BOX),
                ]:
                    tasks.append(
                        GenericCropTask(
                            kind=kind,
                            level=level,
                            physical_page=physical_page,
                            row_index=row_index,
                            image=crop_row(image, column, row_index),
                        )
                    )
    return tasks, problems


def recognition_value(result: Any) -> tuple[str, float]:
    payload = result.json if hasattr(result, "json") else result
    payload = payload.get("res", payload)
    return str(payload.get("rec_text", "")).strip(), float(payload.get("rec_score", 0.0))


def parse_decimal(text: str) -> Decimal | None:
    compact = text.replace(" ", "")
    match = re.search(r"-?(\d[\d,]*\.\d{2})", compact)
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def format_amount(value: Decimal) -> str:
    return f"{value:,.2f}"


def clean_description(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bPayrol\b", "Payroll", text)
    text = re.sub(r"(RFD-\d{6}-\d+)o\b", r"\1", text)
    text = re.sub(r"\bCUST-L3-(\d{1,5})\b", lambda match: f"CUST-L3-{match.group(1).zfill(6)}", text)
    return re.sub(r"_+$", "", text)


def transaction_type_from_visible_description(account_id: str, description: str) -> str:
    if account_id.startswith("KBANK-"):
        return (
            "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19"
            if "CUST-L3-B2B-" in description
            else "\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19"
        )
    return "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19"


def date_from_description(text: str) -> str:
    iso = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso:
        year, month, day = iso.groups()
        return f"{day}-{month}-{year[2:]}"
    compact = re.search(r"\bRFD-(20\d{2})(\d{2})-(\d{2})", text)
    if compact:
        year, month, day = compact.groups()
        return f"{day}-{month}-{year[2:]}"
    return ""


def propagate_dates(rows: list[dict[str, Any]]) -> None:
    index = 0
    while index < len(rows):
        if rows[index]["business_event_date"]:
            index += 1
            continue
        start = index
        while index < len(rows) and not rows[index]["business_event_date"]:
            index += 1
        end = index
        previous = rows[start - 1]["business_event_date"] if start else ""
        following = rows[end]["business_event_date"] if end < len(rows) else ""
        if previous and previous == following:
            for row in rows[start:end]:
                row["business_event_date"] = previous
                row["fallback_actions"].append("date_propagated_from_neighbors")


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
        path = transaction_pages[row["level"]]
        with Image.open(path) as image:
            visual_row = row["row_index"] + (1 if row["level"] == 1 else 0)
            crops.append(crop_row(image, DATE_BOX, visual_row))
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    for row, (text, score) in zip(unresolved, predictions):
        row["date_raw"] = text
        row["date_score"] = round(score, 4)
        match = re.search(r"\b(\d{2}-\d{2}-\d{2})\b", text)
        if match:
            row["business_event_date"] = match.group(1)
            row["fallback_actions"].append("date_crop_ocr")
        else:
            row["fallback_reasons"].append("invalid_date_crop")


def recover_suspicious_amounts(
    model: Any,
    rows: list[dict[str, Any]],
    transaction_pages: dict[int, Path],
    batch_size: int,
) -> None:
    payroll_rows = [row for row in rows if "Payroll" in row["description"] and row["amount_thb"]]
    if not payroll_rows:
        return
    from collections import Counter

    common_amount, frequency = Counter(row["amount_thb"] for row in payroll_rows).most_common(1)[0]
    if frequency < 10:
        return
    suspicious = [row for row in payroll_rows if row["amount_thb"] != common_amount]
    if not suspicious:
        return
    crops: list[np.ndarray] = []
    for row in suspicious:
        path = transaction_pages[row["level"]]
        with Image.open(path) as image:
            visual_row = row["row_index"] + (1 if row["level"] == 1 else 0)
            crops.append(crop_row(image, AMOUNT_BOX, visual_row))
    predictions = [
        recognition_value(result)
        for result in model.predict(input=crops, batch_size=batch_size)
    ]
    for row, (text, score) in zip(suspicious, predictions):
        amount = parse_decimal(text)
        row["amount_raw_fallback"] = text
        row["amount_score_fallback"] = round(score, 4)
        if amount is not None:
            row["amount_thb"] = format_amount(abs(amount))
            row["fallback_actions"].append("amount_crop_ocr")
        else:
            row["fallback_reasons"].append("invalid_amount_crop")


def recover_account_number(model: Any, header: Path) -> tuple[str, str, float]:
    boxes = [
        (1965, 420, 2350, 465),
        HEADER_ACCOUNT_NUMBER_BOX,
        (1980, 425, 2350, 460),
        (1955, 415, 2400, 468),
        (1970, 425, 2500, 465),
    ]
    with Image.open(header) as image:
        crops = [np.array(image.crop(box).convert("RGB")) for box in boxes]
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


def recognize_generic_rows(
    model: Any,
    tasks: list[GenericCropTask],
    generic_fields: dict[int, dict[int, dict[str, str]]],
    batch_size: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    predictions = [
        recognition_value(result)
        for result in model.predict(input=[task.image for task in tasks], batch_size=batch_size)
    ]
    if len(predictions) != len(tasks):
        raise ValueError(f"generic prediction count={len(predictions)} tasks={len(tasks)}")

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for task, (text, score) in zip(tasks, predictions):
        item = rows.setdefault(
            (task.level, task.row_index),
            {
                "level": task.level,
                "physical_page": task.physical_page,
                "row_index": task.row_index,
                "fields": generic_fields[task.level][task.row_index],
                "fallback_reasons": [],
            },
        )
        item[f"{task.kind}_raw"] = text
        item[f"{task.kind}_score"] = round(score, 4)
        if score < min_confidence:
            item["fallback_reasons"].append(f"low_{task.kind}_confidence")

    output: list[dict[str, Any]] = []
    for item in rows.values():
        date_match = re.search(r"\b(\d{2}-\d{2}-\d{2})\b", item.get("date_raw", ""))
        item["date"] = date_match.group(1) if date_match else ""
        item["balance"] = parse_decimal(item.get("balance_raw", ""))
        item["details"] = clean_description(item.get("details_raw", ""))
        # Rendered closing rows may expose a balance but do not have a date.
        item["is_visible_transaction"] = bool(item["date"])
        output.append(item)
    return output


def recalculate_amounts_with_generic_rows(
    rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    opening_balance: Decimal | None,
) -> None:
    previous_balance = opening_balance
    timeline = sorted(
        [*rows, *generic_rows],
        key=lambda row: (row["physical_page"], row["row_index"]),
    )
    for row in timeline:
        if "balance" in row:
            balance = row["balance"]
        else:
            balance = parse_decimal(row.get("balance_raw", ""))
        if not row.get("is_visible_transaction", True):
            continue
        delta = (
            balance - previous_balance
            if balance is not None and previous_balance is not None
            else None
        )
        if "slot_id" in row:
            row["amount_thb"] = format_amount(abs(delta)) if delta is not None else ""
        else:
            row["delta"] = delta
        if balance is not None:
            previous_balance = balance
        else:
            previous_balance = None


def generic_values(
    account_id: str,
    header_fields: dict[str, str],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    values = {key: account_id for key in header_fields}
    for row in rows:
        fields = row["fields"]
        visible = row.get("is_visible_transaction", False)
        delta = row.get("delta")
        label = ""
        if delta is not None:
            label = "\u0e23\u0e31\u0e1a\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19" if delta > 0 else "\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19"
        amount = format_amount(abs(delta)) if delta is not None else ""
        balance = row.get("balance")
        has_long_details_field = "details" in fields or "reference" in fields
        field_values = {
            "account_id": account_id,
            "date": row.get("date", ""),
            "time": "09:00",
            "date_time": f"{row.get('date', '')} 09:00".strip(),
            "transaction_type": label,
            "type": label,
            "description": label if has_long_details_field else row.get("details", ""),
            "amount": amount,
            "withdrawal": amount if delta is not None and delta < 0 else "",
            "debit": amount if delta is not None and delta < 0 else "",
            "deposit": amount if delta is not None and delta > 0 else "",
            "credit": amount if delta is not None and delta > 0 else "",
            "currency": "THB",
            "balance": format_amount(balance) if balance is not None else "",
            "channel": "K PLUS",
            "details": row.get("details", ""),
            "reference": row.get("details", ""),
        }
        for field, key in fields.items():
            values[key] = field_values.get(field, "") if visible else ""
    return values


def extract_artifact(
    model: Any,
    header_model: Any,
    artifact_id: str,
    renders: dict[str, Any],
    schema: dict[str, str],
    batch_size: int,
    min_confidence: float,
) -> dict[str, Any]:
    slots = slot_ids_by_level(schema)
    generic_fields, generic_header_fields = generic_fields_by_level(schema)
    schema_levels = sorted(set(slots) | set(generic_fields))
    tasks, level_pages, all_level_pages, ignored_pages, problems = build_crop_tasks(
        renders["transactions"],
        slots,
        schema_levels,
    )
    if problems:
        return {"artifact_id": artifact_id, "errors": problems, "rows": []}
    generic_tasks, problems = build_generic_crop_tasks(all_level_pages, generic_fields)
    if problems:
        return {"artifact_id": artifact_id, "errors": problems, "rows": []}

    started = time.perf_counter()
    predictions = [
        recognition_value(result)
        for result in model.predict(input=[task.image for task in tasks], batch_size=batch_size)
    ]
    if len(predictions) != len(tasks):
        return {
            "artifact_id": artifact_id,
            "errors": [f"prediction count={len(predictions)} tasks={len(tasks)}"],
            "rows": [],
        }

    account_id = account_id_from_artifact_id(artifact_id)
    account_number, account_number_raw, account_number_score = recover_account_number(
        header_model,
        renders["header"],
    )
    opening_balance: Decimal | None = None
    row_parts: dict[str, dict[str, Any]] = {}
    for task, (text, score) in zip(tasks, predictions):
        if task.kind == "opening_balance":
            opening_balance = parse_decimal(text)
            continue
        item = row_parts.setdefault(
            task.slot_id,
            {
                "slot_id": task.slot_id,
                "level": task.level,
                "physical_page": task.physical_page,
                "row_index": task.row_index,
                "account_id": account_id,
                "fallback_reasons": [],
                "fallback_actions": [],
            },
        )
        item[f"{task.kind}_raw"] = text
        item[f"{task.kind}_score"] = round(score, 4)
        if score < min_confidence:
            item["fallback_reasons"].append(f"low_{task.kind}_confidence")

    rows: list[dict[str, Any]] = []
    previous_balance = opening_balance
    for item in sorted(
        row_parts.values(),
        key=lambda row: (row["physical_page"], row["row_index"]),
    ):
        balance = parse_decimal(item.get("balance_raw", ""))
        description = clean_description(item.get("description_raw", ""))
        date = date_from_description(description)
        if balance is None:
            item["fallback_reasons"].append("invalid_balance")
        if previous_balance is None or balance is None:
            amount = None
            item["fallback_reasons"].append("missing_balance_delta")
        else:
            amount = balance - previous_balance
        if balance is not None:
            previous_balance = balance
        else:
            previous_balance = None

        item.update(
            {
                "business_event_date": date,
                "transaction_type": transaction_type_from_visible_description(
                    account_id,
                    description,
                ),
                "amount_thb": format_amount(abs(amount)) if amount is not None else "",
                "balance_after_thb": format_amount(balance) if balance is not None else "",
                "description": description,
            }
        )
        rows.append(item)

    try:
        raw_generic_rows = recognize_generic_rows(
            model,
            generic_tasks,
            generic_fields,
            batch_size,
            min_confidence,
        )
    except ValueError as error:
        return {"artifact_id": artifact_id, "errors": [str(error)], "rows": []}
    recalculate_amounts_with_generic_rows(rows, raw_generic_rows, opening_balance)
    recover_suspicious_amounts(model, rows, level_pages, batch_size)
    propagate_dates(rows)
    recover_missing_dates(model, rows, level_pages, batch_size)
    for row in rows:
        if row["business_event_date"]:
            row["business_event_date"] += " 09:00"
        row["description"] = f"K PLUS {row['description']}".strip()
    return {
        "artifact_id": artifact_id,
        "engine": "paddle_crop_recognition",
        "model": "en_PP-OCRv5_mobile_rec",
        "account_id": account_id,
        "account_number": account_number,
        "account_number_raw": account_number_raw,
        "account_number_score": account_number_score,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "opening_balance": format_amount(opening_balance) if opening_balance is not None else "",
        "level_render_paths": {
            str(level): str(path) for level, path in sorted(level_pages.items())
        },
        "ignored_render_paths": [str(path) for path in ignored_pages],
        "rows": rows,
        "generic_values": generic_values(
            account_id,
            generic_header_fields,
            raw_generic_rows,
        ),
        "generic_rows": len(raw_generic_rows),
        "generic_visible_rows": sum(
            row.get("is_visible_transaction", False) for row in raw_generic_rows
        ),
        "fallback_rows": sum(bool(row["fallback_reasons"]) for row in rows),
        "errors": [],
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    data_root = args.data_root.absolute()
    schemas = load_bank_schemas(data_root / "submission_template_OCR.csv")
    groups = dense_render_groups(data_root / "fahmai_renders_with_json")
    artifact_ids = sorted(set(schemas) & set(groups), key=natural_key)
    if args.artifact_id:
        selected_ids = set(args.artifact_id)
        artifact_ids = [artifact_id for artifact_id in artifact_ids if artifact_id in selected_ids]
    if args.limit_artifacts is not None:
        artifact_ids = artifact_ids[: args.limit_artifacts]
    if not artifact_ids:
        raise SystemExit("No matching dense statement artifacts.")

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
    completed = failed = skipped = fallback_rows = total_rows = 0
    started = time.perf_counter()
    for index, artifact_id in enumerate(artifact_ids, start=1):
        output = output_dir / f"{artifact_id}.json"
        if output.exists() and not args.overwrite:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if not existing.get("errors"):
                skipped += 1
                total_rows += len(existing.get("rows", []))
                fallback_rows += int(existing.get("fallback_rows", 0))
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
        total_rows += len(record["rows"])
        fallback_rows += int(record.get("fallback_rows", 0))
        if record["errors"]:
            failed += 1
            print(f"[{index}/{len(artifact_ids)}] ERROR {artifact_id}: {record['errors']}")
        else:
            completed += 1
            print(
                f"[{index}/{len(artifact_ids)}] ok {artifact_id} "
                f"rows={len(record['rows'])} fallback={record['fallback_rows']} "
                f"elapsed={record['elapsed_seconds']}s"
            )

    elapsed = round(time.perf_counter() - started, 3)
    print(
        f"completed={completed} skipped={skipped} failed={failed} "
        f"rows={total_rows} fallback_rows={fallback_rows} elapsed_seconds={elapsed}"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_dense_bank",
    )
    parser.add_argument("--artifact-id", action="append")
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
