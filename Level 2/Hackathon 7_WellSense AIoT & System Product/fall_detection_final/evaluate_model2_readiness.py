from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model2_risk_common import (
    RiskTargetConfig,
    apply_conservative_risk_guard,
    compute_component_scores,
)

EXPECTED_CLASS_COUNTS = {
    "slow_collapse_fall": {
        "category": "fall",
        "thai_name": "ล้มแบบค่อยๆทรุด",
        "expected_count": 189,
    },
    "gradual_fall": {
        "category": "fall",
        "thai_name": "ค่อยๆล้ม",
        "expected_count": 25,
    },
    "sideways_fall": {
        "category": "fall",
        "thai_name": "ล้มข้าง",
        "expected_count": 806,
    },
    "backward_fall": {
        "category": "fall",
        "thai_name": "ล้มไปด้านหลัง",
        "expected_count": 805,
    },
    "normal_walk": {
        "category": "activity",
        "thai_name": "เดินปกติ",
        "expected_count": 29,
    },
    "limping_walk": {
        "category": "activity",
        "thai_name": "เดินกระเพก",
        "expected_count": 80,
    },
    "corrected_walking": {
        "category": "activity",
        "thai_name": "คนแก่เดิน",
        "expected_count": 59,
    },
    "stand_sit_alternating": {
        "category": "activity",
        "thai_name": "ลุกยืนสลับนั่ง",
        "expected_count": 56,
    },
    "elderly_pick_up_object": {
        "category": "activity",
        "thai_name": "คนแก่จับของระหว่างทาง",
        "expected_count": 63,
    },
    "standing": {
        "category": "static_activity",
        "thai_name": "ยืน",
        "expected_count": 9,
    },
    "lying_down": {
        "category": "static_activity",
        "thai_name": "นอน",
        "expected_count": 17,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Model 2 readiness report: class counts, feature completeness, inference time."
    )
    parser.add_argument("--data", default="windows_all.csv")
    parser.add_argument(
        "--model",
        default="models/model2_risk_assessment/model2_risk_bundle.joblib",
    )
    parser.add_argument("--output-dir", default="reports/model2_risk_assessment")
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = ["risk_regressor", "feature_columns"]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"Model bundle is missing key(s): {', '.join(missing)}")
    return bundle


def check_class_counts(df: pd.DataFrame) -> pd.DataFrame:
    actual_counts = df["class_en"].value_counts(dropna=False).to_dict()
    rows: list[dict[str, Any]] = []
    for class_en, info in EXPECTED_CLASS_COUNTS.items():
        actual = int(actual_counts.get(class_en, 0))
        expected = int(info["expected_count"])
        rows.append(
            {
                "category": info["category"],
                "class_en": class_en,
                "thai_name": info["thai_name"],
                "expected_count_from_image": expected,
                "actual_count_in_windows_all": actual,
                "difference": actual - expected,
                "match": actual == expected,
            }
        )

    known_classes = set(EXPECTED_CLASS_COUNTS)
    for class_en, actual in actual_counts.items():
        if class_en not in known_classes:
            rows.append(
                {
                    "category": "unknown",
                    "class_en": class_en,
                    "thai_name": "",
                    "expected_count_from_image": 0,
                    "actual_count_in_windows_all": int(actual),
                    "difference": int(actual),
                    "match": False,
                }
            )
    return pd.DataFrame(rows)


def check_feature_completeness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        present = feature in df.columns
        if present:
            is_numeric = pd.api.types.is_numeric_dtype(df[feature])
            null_count = int(df[feature].isna().sum())
            finite_count = int(np.isfinite(pd.to_numeric(df[feature], errors="coerce")).sum())
        else:
            is_numeric = False
            null_count = len(df)
            finite_count = 0

        rows.append(
            {
                "feature": feature,
                "present": present,
                "numeric": bool(is_numeric),
                "null_count": null_count,
                "finite_count": finite_count,
                "complete": bool(present and is_numeric and null_count == 0),
            }
        )
    return pd.DataFrame(rows)


def prepare_model_input(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    model_input = df.copy()
    for feature in feature_cols:
        if feature not in model_input.columns:
            model_input[feature] = np.nan
    return model_input[feature_cols]


def run_once(
    bundle: dict[str, Any],
    X: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    risk_regressor = bundle["risk_regressor"]
    high_risk_classifier = bundle.get("high_risk_classifier")
    risk_score = np.clip(risk_regressor.predict(X), 0.0, 1.0)
    high_risk_probability = None
    if high_risk_classifier is not None and hasattr(high_risk_classifier, "predict_proba"):
        high_risk_probability = high_risk_classifier.predict_proba(X)[:, 1]
    if raw_df is not None and "normalizer" in bundle:
        component_scores = compute_component_scores(raw_df, bundle["normalizer"])
        risk_score, _ = apply_conservative_risk_guard(
            risk_score, component_scores, raw_df, RiskTargetConfig()
        )
        risk_score = risk_score.to_numpy()
    return risk_score, high_risk_probability


def benchmark_inference(
    bundle: dict[str, Any],
    model_input: pd.DataFrame,
    raw_df: pd.DataFrame,
    repeats: int,
    batch_sizes: list[int] | None = None,
) -> pd.DataFrame:
    batch_sizes = batch_sizes or [1, 10, 50, 100, len(model_input)]
    batch_sizes = sorted(set(min(size, len(model_input)) for size in batch_sizes if size > 0))

    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        X_batch = model_input.iloc[:batch_size].copy()
        raw_batch = raw_df.iloc[:batch_size].copy()
        run_once(bundle, X_batch, raw_batch)

        timings_ms: list[float] = []
        local_repeats = repeats if batch_size <= 100 else max(10, repeats // 5)
        for _ in range(local_repeats):
            start = time.perf_counter()
            run_once(bundle, X_batch, raw_batch)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

        rows.append(
            {
                "batch_size": batch_size,
                "repeats": local_repeats,
                "mean_batch_ms": statistics.mean(timings_ms),
                "median_batch_ms": statistics.median(timings_ms),
                "p95_batch_ms": float(np.percentile(timings_ms, 95)),
                "mean_per_window_ms": statistics.mean(timings_ms) / batch_size,
                "median_per_window_ms": statistics.median(timings_ms) / batch_size,
                "windows_per_second_mean": batch_size / (statistics.mean(timings_ms) / 1000.0),
            }
        )
    return pd.DataFrame(rows)


def estimate_window_duration(df: pd.DataFrame) -> dict[str, float | None]:
    if "window_start_ts" not in df.columns or "window_end_ts" not in df.columns:
        return {"median_window_seconds": None, "median_stride_seconds": None}

    start = pd.to_datetime(df["window_start_ts"], errors="coerce", utc=True)
    end = pd.to_datetime(df["window_end_ts"], errors="coerce", utc=True)
    duration = (end - start).dt.total_seconds().dropna()

    stride = start.sort_values().diff().dt.total_seconds()
    stride = stride[(stride > 0) & (stride < 60)].dropna()

    return {
        "median_window_seconds": float(duration.median()) if not duration.empty else None,
        "median_stride_seconds": float(stride.median()) if not stride.empty else None,
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    bundle = load_bundle(model_path)
    feature_cols = list(bundle["feature_columns"])
    model_input = prepare_model_input(df, feature_cols)

    class_counts = check_class_counts(df)
    feature_completeness = check_feature_completeness(df, feature_cols)
    benchmark = benchmark_inference(bundle, model_input, df, repeats=args.repeats)
    window_timing = estimate_window_duration(df)

    class_counts.to_csv(
        output_dir / "model2_class_count_check.csv", index=False, encoding="utf-8-sig"
    )
    feature_completeness.to_csv(
        output_dir / "model2_feature_completeness.csv", index=False, encoding="utf-8-sig"
    )
    benchmark.to_csv(
        output_dir / "model2_inference_benchmark.csv", index=False, encoding="utf-8-sig"
    )

    missing_features = feature_completeness[~feature_completeness["present"]]["feature"].tolist()
    non_numeric_features = feature_completeness[
        feature_completeness["present"] & ~feature_completeness["numeric"]
    ]["feature"].tolist()
    features_with_null = feature_completeness[feature_completeness["null_count"] > 0][
        ["feature", "null_count"]
    ].to_dict(orient="records")

    summary = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "rows": int(len(df)),
        "feature_count_expected_by_model": int(len(feature_cols)),
        "feature_count_present": int(feature_completeness["present"].sum()),
        "feature_count_numeric": int(feature_completeness["numeric"].sum()),
        "class_count_all_match_image": bool(class_counts["match"].all()),
        "missing_features": missing_features,
        "non_numeric_features": non_numeric_features,
        "features_with_null": features_with_null,
        "window_timing": window_timing,
        "benchmark": benchmark.to_dict(orient="records"),
    }
    write_json(output_dir / "model2_readiness_summary.json", summary)

    class_match_text = "PASS" if class_counts["match"].all() else "CHECK"
    feature_pass_text = "PASS" if not missing_features and not non_numeric_features else "CHECK"
    null_note = (
        "No null values in model features."
        if not features_with_null
        else "Some features contain null values; the model pipeline imputes them with median values."
    )

    best_single = benchmark[benchmark["batch_size"] == 1].iloc[0]
    best_batch = benchmark.iloc[-1]
    lines = [
        "# Model 2 Readiness Report",
        "",
        "This report checks class counts, feature completeness, and inference time for Model 2 risk assessment.",
        "",
        "## Executive Summary",
        "",
        f"- Dataset rows: {len(df)}",
        f"- Model features required: {len(feature_cols)}",
        f"- Feature check: {feature_pass_text}",
        f"- Class count check against `image.png`: {class_match_text}",
        f"- Single-window inference mean: {best_single['mean_per_window_ms']:.3f} ms/window",
        f"- Full-batch inference mean: {best_batch['mean_per_window_ms']:.3f} ms/window",
        "",
        "## Interpretation For Seniors",
        "",
        "- Model 2 is not the binary fall/no-fall baseline.",
        "- It estimates a mobility risk score from movement features such as jerk, omega, theta, GSI, FCRI, and optional PPG-derived features.",
        "- The current output is a proxy risk score for prototype testing because the dataset does not include clinical future-fall labels.",
        "- The score can be used for dashboard risk level and heatmap accumulation with `(x, y)` location.",
        "",
        "## Class Count Check",
        "",
        class_counts.to_markdown(index=False),
        "",
        "## Feature Completeness",
        "",
        f"- Required features present: {int(feature_completeness['present'].sum())}/{len(feature_cols)}",
        f"- Required features numeric: {int(feature_completeness['numeric'].sum())}/{len(feature_cols)}",
        f"- Missing features: {missing_features or 'none'}",
        f"- Non-numeric features: {non_numeric_features or 'none'}",
        f"- Null note: {null_note}",
        "",
        "Features with null values:",
        "",
        (
            pd.DataFrame(features_with_null).to_markdown(index=False)
            if features_with_null
            else "none"
        ),
        "",
        "## Inference Time",
        "",
        "Measured on the current Windows machine with the saved `model2_risk_bundle.joblib`.",
        "",
        benchmark.to_markdown(index=False),
        "",
        "## Window Timing From Dataset",
        "",
        f"- Median feature window length: {window_timing['median_window_seconds']} seconds",
        f"- Median stride between windows: {window_timing['median_stride_seconds']} seconds",
        "",
        "Practical reading:",
        "",
        f"- One prediction call takes about {best_single['mean_per_window_ms'] / 1000.0:.4f} seconds per single window on this machine.",
        f"- Since each feature window represents about {window_timing['median_window_seconds']} seconds of sensor data, inference time is much smaller than the sensing window.",
        "",
        "## Output Files",
        "",
        "- `model2_class_count_check.csv`",
        "- `model2_feature_completeness.csv`",
        "- `model2_inference_benchmark.csv`",
        "- `model2_readiness_summary.json`",
    ]

    (output_dir / "model2_readiness_report.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )

    print(f"Class count check: {class_match_text}")
    print(f"Feature check: {feature_pass_text}")
    print(f"Single-window mean: {best_single['mean_per_window_ms']:.3f} ms/window")
    print(f"Full-batch mean: {best_batch['mean_per_window_ms']:.3f} ms/window")
    print(f"Report: {output_dir / 'model2_readiness_report.md'}")


if __name__ == "__main__":
    main()
