from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model2_risk_common import (
    RiskTargetConfig,
    apply_conservative_risk_guard,
    compute_component_scores,
    compute_feature_risk_score,
    high_risk_alert_from_outputs,
    risk_level_from_score,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Model 2 mobility risk score from feature-window CSV rows."
    )
    parser.add_argument("--model", required=True, help="Path to model2_risk_bundle.joblib")
    parser.add_argument("--input", required=True, help="Input CSV with window feature columns")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    parser.add_argument("--json-output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    input_path = resolve_path(args.input)

    bundle: dict[str, Any] = joblib.load(model_path)
    feature_cols: list[str] = list(bundle["feature_columns"])
    risk_regressor = bundle["risk_regressor"]
    high_risk_classifier = bundle.get("high_risk_classifier")
    normalizer = bundle["normalizer"]

    df = pd.read_csv(input_path)
    model_input = df.copy()
    missing_cols = [col for col in feature_cols if col not in model_input.columns]
    for col in missing_cols:
        model_input[col] = np.nan
    model_input = model_input[feature_cols]

    raw_predicted_score = np.clip(risk_regressor.predict(model_input), 0.0, 1.0)
    if high_risk_classifier is not None and hasattr(high_risk_classifier, "predict_proba"):
        high_risk_probability = high_risk_classifier.predict_proba(model_input)[:, 1]
    else:
        high_risk_probability = raw_predicted_score

    component_scores = compute_component_scores(df, normalizer)
    feature_rule_score = compute_feature_risk_score(component_scores)
    predicted_score, guard_info = apply_conservative_risk_guard(
        raw_predicted_score, component_scores, df, RiskTargetConfig()
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
    output = df[meta_cols].reset_index(drop=True).copy()
    output["raw_model2_risk_score"] = raw_predicted_score
    output["model2_risk_score"] = predicted_score.to_numpy()
    output["model2_high_risk_probability"] = high_risk_probability
    output["model2_risk_level"] = [
        risk_level_from_score(float(score)) for score in predicted_score
    ]
    output["model2_alert"] = [
        high_risk_alert_from_outputs(float(score), float(probability))
        for score, probability in zip(predicted_score, high_risk_probability)
    ]
    output["rule_feature_risk_score"] = feature_rule_score.to_numpy()
    for col in component_scores.columns:
        output[f"component_{col}"] = component_scores[col].to_numpy()
    for col in guard_info.columns:
        output[col] = guard_info[col].to_numpy()
    output["model2_regressor"] = bundle.get("best_regressor_name", "")
    output["model2_classifier"] = bundle.get("best_classifier_name", "")

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
