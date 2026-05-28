from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on feature-window CSV rows using a trained joblib bundle."
    )
    parser.add_argument("--model", required=True, help="Path to models/<run>/best_model.joblib.")
    parser.add_argument("--input", required=True, help="Input CSV with feature columns.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    parser.add_argument("--json-output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def risk_level(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def fall_probability(class_names: list[str], proba_row: np.ndarray | None) -> float | None:
    if proba_row is None:
        return None
    lower_names = [name.lower() for name in class_names]
    if "1" in class_names:
        return float(proba_row[class_names.index("1")])
    fall_indexes = [i for i, name in enumerate(lower_names) if "fall" in name or "ล้ม" in name]
    if fall_indexes:
        return float(np.sum(proba_row[fall_indexes]))
    return float(np.max(proba_row))


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    input_path = resolve_path(args.input)

    bundle: dict[str, Any] = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_cols: list[str] = list(bundle["feature_columns"])
    class_names: list[str] = list(bundle.get("class_names", label_encoder.classes_))

    df = pd.read_csv(input_path)
    model_input = df.copy()
    missing_cols = [col for col in feature_cols if col not in model_input.columns]
    for col in missing_cols:
        model_input[col] = np.nan
    model_input = model_input[feature_cols]

    pred_encoded = model.predict(model_input)
    pred_labels = label_encoder.inverse_transform(pred_encoded.astype(int))

    proba = model.predict_proba(model_input) if hasattr(model, "predict_proba") else None
    confidence = np.max(proba, axis=1) if proba is not None else np.full(len(df), np.nan)
    fall_scores = [
        fall_probability(class_names, proba[i] if proba is not None else None)
        for i in range(len(df))
    ]

    output = pd.DataFrame(
        {
            "predicted_label": pred_labels,
            "confidence": confidence,
            "fall_risk_score": fall_scores,
            "risk_level": [risk_level(score) for score in fall_scores],
            "model_name": bundle.get("model_name", model_path.stem),
            "target_col": bundle.get("target_col", ""),
        }
    )

    meta_cols = [
        col
        for col in [
            "window_start_ts",
            "window_end_ts",
            "session_id",
            "class_en",
            "category",
            "class",
            "label",
            "source",
        ]
        if col in df.columns
    ]
    output = pd.concat([df[meta_cols].reset_index(drop=True), output], axis=1)

    if proba is not None:
        for i, class_name in enumerate(class_names):
            output[f"proba_{class_name}"] = proba[:, i]

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved CSV predictions: {output_path}")

    if args.json_output:
        json_path = resolve_path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(output.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON predictions: {json_path}")

    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
