from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model2_live_features import extract_features_from_samples, latest_raw_values
from model2_risk_common import (
    RiskTargetConfig,
    apply_conservative_risk_guard,
    compute_component_scores,
    compute_feature_risk_score,
    high_risk_alert_from_outputs,
    risk_level_from_score,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "models" / "model2_risk_assessment" / "model2_risk_bundle.joblib"


ROOM_DEFAULTS = {
    "Bedroom": {"x": 252, "y": 214},
    "Bathroom": {"x": 206, "y": 316},
    "Kitchen": {"x": 430, "y": 340},
    "Living Room": {"x": 524, "y": 232},
    "Hallway": {"x": 286, "y": 292},
    "Balcony": {"x": 570, "y": 258},
}


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


class Model2RiskService:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.bundle: dict[str, Any] = joblib.load(self.model_path)
        self.feature_cols: list[str] = list(self.bundle["feature_columns"])
        self.risk_regressor = self.bundle["risk_regressor"]
        self.high_risk_classifier = self.bundle.get("high_risk_classifier")
        self.normalizer = self.bundle["normalizer"]
        self.config = RiskTargetConfig()

    def predict_feature_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            raise ValueError("At least one feature row is required.")

        df = pd.DataFrame(rows)
        model_input = df.copy()
        for col in self.feature_cols:
            if col not in model_input.columns:
                model_input[col] = np.nan
        model_input = model_input[self.feature_cols]

        raw_score = np.clip(self.risk_regressor.predict(model_input), 0.0, 1.0)
        if self.high_risk_classifier is not None and hasattr(self.high_risk_classifier, "predict_proba"):
            high_prob = self.high_risk_classifier.predict_proba(model_input)[:, 1]
        else:
            high_prob = raw_score

        component_scores = compute_component_scores(df, self.normalizer)
        feature_rule_score = compute_feature_risk_score(component_scores)
        guarded_score, guard_info = apply_conservative_risk_guard(
            raw_score, component_scores, df, self.config
        )

        output = pd.DataFrame(index=df.index)
        for col in [
            "window_start_ts",
            "window_end_ts",
            "session_id",
            "class_en",
            "category",
            "source",
            "room",
            "x",
            "y",
        ]:
            if col in df.columns:
                output[col] = df[col]

        output["raw_model2_risk_score"] = raw_score
        output["model2_risk_score"] = guarded_score.to_numpy()
        output["model2_high_risk_probability"] = high_prob
        output["model2_risk_percent"] = np.round(guarded_score.to_numpy() * 100.0, 1)
        output["model2_risk_level"] = [
            risk_level_from_score(float(score)) for score in guarded_score
        ]
        output["model2_alert"] = [
            high_risk_alert_from_outputs(float(score), float(prob), self.config)
            for score, prob in zip(guarded_score, high_prob)
        ]
        output["rule_feature_risk_score"] = feature_rule_score.to_numpy()

        for col in component_scores.columns:
            output[f"component_{col}"] = component_scores[col].to_numpy()
        for col in guard_info.columns:
            output[col] = guard_info[col].to_numpy()

        return output.replace({np.nan: None}).to_dict(orient="records")

    def predict_raw_samples(
        self,
        samples: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = meta or {}
        session_id = str(meta.get("session_id", "live_window"))
        feature_row = extract_features_from_samples(samples, session_id=session_id)
        feature_row.update(meta)
        result = self.predict_feature_rows([feature_row])[0]
        result["dashboard_reading"] = self.to_dashboard_reading(
            result=result,
            raw_values=latest_raw_values(samples),
            meta=meta,
        )
        return result

    def to_dashboard_reading(
        self,
        result: dict[str, Any],
        raw_values: dict[str, float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = meta or {}
        raw_values = raw_values or {}
        risk = float(result.get("model2_risk_score") or 0.0)
        risk_percent = int(round(clamp(risk * 100.0, 0, 100)))
        room = str(meta.get("room") or result.get("room") or "Hallway")
        if room not in ROOM_DEFAULTS:
            room = "Hallway"
        point = ROOM_DEFAULTS[room]
        x = float(meta.get("x", result.get("x", point["x"])))
        y = float(meta.get("y", result.get("y", point["y"])))

        rotation = float(result.get("component_rotation_balance") or 0.0)
        posture = float(result.get("component_posture_transition") or 0.0)
        gait = float(result.get("component_gait_motion") or 0.0)

        return {
            "timestamp": str(meta.get("timestamp") or datetime.now().isoformat(timespec="milliseconds")),
            "room": room,
            "x": clamp(x, 54, 688),
            "y": clamp(y, 58, 426),
            "gait_speed": round(clamp(1.15 - risk * 0.72, 0.34, 1.18), 3),
            "sway": round(clamp(1.4 + posture * 7.0 + gait * 2.0, 1.2, 8.4), 3),
            "cadence": round(clamp(110 - risk * 42, 62, 114), 1),
            "turning_velocity": round(clamp(102 - rotation * 76, 24, 104), 1),
            "instability_score": risk_percent,
            "fall_risk": risk_percent,
            "near_fall": bool(result.get("model2_alert", False)),
            "ax": round(float(raw_values.get("ax", 0.0)), 3),
            "ay": round(float(raw_values.get("ay", 0.0)), 3),
            "az": round(float(raw_values.get("az", 0.0)), 3),
            "gx": round(float(raw_values.get("gx", 0.0)), 3),
            "gy": round(float(raw_values.get("gy", 0.0)), 3),
            "gz": round(float(raw_values.get("gz", 0.0)), 3),
        }

    def config_payload(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "feature_count": len(self.feature_cols),
            "feature_columns": self.feature_cols,
            "thresholds": {
                "low": "0.00 - 0.35",
                "medium": "> 0.35 - 0.70",
                "high": "> 0.70",
                "alert": "score > 0.70 and probability >= 0.70",
            },
            "guard": {
                "quiet_motion_cap": self.config.quiet_motion_cap,
                "single_signal_cap": self.config.single_signal_cap,
                "min_high_components_for_high": self.config.min_high_components_for_high,
            },
            "bundle_note": self.bundle.get("note", ""),
        }
