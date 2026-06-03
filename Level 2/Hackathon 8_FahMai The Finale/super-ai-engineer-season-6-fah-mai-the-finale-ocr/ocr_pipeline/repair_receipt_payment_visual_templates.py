#!/usr/bin/env python3
"""Repair blank receipt payment methods with public-render visual prototypes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_fast_dense_bank_crop_ocr import DEFAULT_ROOT, atomic_write_json
from run_fast_fixed_nonbank_ocr import receipt_summary_lines


ALLOWED_METHODS = {"เงินสด", "โอนเงิน", "บัตรเครดิต", "บัตรเดบิต", "MOBILE_WALLET"}


def feature(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        _, bottom = receipt_summary_lines(image)
        pixels = image.crop((130, bottom + 10, 330, bottom + 65)).convert("L")
    return (np.array(pixels.resize((220, 50))) < 180).astype(np.float32)


def prototype(features: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(features), axis=0)


def predict(prototypes: dict[str, np.ndarray], value: np.ndarray) -> tuple[str, float, float]:
    distances = sorted(
        (float(np.mean((value - template) ** 2)), label)
        for label, template in prototypes.items()
    )
    best_distance, best_label = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else best_distance
    return best_label, best_distance, second_distance - best_distance


def run(args: argparse.Namespace) -> int:
    checkpoint_dir = args.checkpoint_dir.absolute()
    records: list[tuple[Path, dict[str, Any]]] = []
    by_label: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(checkpoint_dir.glob("RC-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("layout") != "fixed_receipt" or record.get("errors"):
            continue
        records.append((path, record))
        label = record.get("prediction", {}).get("payment_method", "")
        if label in ALLOWED_METHODS:
            by_label[label].append(feature(Path(record["source_render_path"])))
    if set(by_label) != ALLOWED_METHODS:
        raise SystemExit(f"Expected visual examples for {sorted(ALLOWED_METHODS)}, found {sorted(by_label)}")

    prototypes = {label: prototype(features) for label, features in by_label.items()}
    confusion: Counter[str] = Counter()
    evaluated = correct = repaired = 0
    repaired_labels: Counter[str] = Counter()
    margins: list[float] = []
    for path, record in records:
        image_feature = feature(Path(record["source_render_path"]))
        current = record["prediction"].get("payment_method", "")
        predicted, distance, margin = predict(prototypes, image_feature)
        if current in ALLOWED_METHODS:
            evaluated += 1
            correct += int(predicted == current)
            confusion[f"{current} -> {predicted}"] += 1
            continue
        record["prediction"]["payment_method"] = predicted
        record["fallback_fields"] = [
            field for field in record.get("fallback_fields", []) if field != "payment_method"
        ]
        record["fallback_count"] = len(record["fallback_fields"])
        record.setdefault("fallback_actions", []).append("payment_method_from_public_visual_template")
        record["visual_template_payment_method"] = {
            "label": predicted,
            "distance": round(distance, 6),
            "margin": round(margin, 6),
        }
        atomic_write_json(path, record)
        repaired += 1
        repaired_labels[predicted] += 1
        margins.append(margin)

    accuracy = correct / evaluated if evaluated else 0.0
    audit = {
        "checkpoint_dir": str(checkpoint_dir),
        "boundary": "Public receipt renders and existing render-only crop checkpoints only.",
        "prototype_examples": {key: len(value) for key, value in sorted(by_label.items())},
        "evaluated_labeled_receipts": evaluated,
        "prototype_agreement_percent": round(accuracy * 100, 2),
        "repaired_blank_receipts": repaired,
        "repaired_labels": dict(repaired_labels.most_common()),
        "minimum_repair_margin": round(min(margins), 6) if margins else None,
        "confusion": dict(confusion),
    }
    output = args.audit_output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    print(f"wrote={output}")
    return 0 if accuracy >= args.minimum_agreement else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "fast_fixed_nonbank",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_ROOT / "ocr_outputs" / "audits" / "receipt_payment_visual_templates.audit.json",
    )
    parser.add_argument("--minimum-agreement", type=float, default=0.99)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
