from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


META_COLUMNS = {
    "window_start_ts",
    "window_end_ts",
    "session_id",
    "class_th",
    "label",
    "source",
    "class_en",
    "category",
    "class",
}


RISK_THRESHOLDS = {
    "low": 0.33,
    "medium": 0.66,
}


CLASS_RISK_PRIOR = {
    "standing": 0.05,
    "lying_down": 0.05,
    "normal_walk": 0.18,
    "corrected_walking": 0.20,
    "elderly_pick_up_object": 0.38,
    "limping_walk": 0.48,
    "stand_sit_alternating": 0.52,
    "gradual_fall": 0.86,
    "slow_collapse_fall": 0.82,
    "sideways_fall": 0.92,
    "backward_fall": 0.92,
}


CATEGORY_RISK_PRIOR = {
    "static_activity": 0.05,
    "activity": 0.32,
    "fall": 0.86,
}


FEATURE_GROUPS = {
    "gait_motion": {
        "weight": 0.22,
        "features": {
            "svm_std": 1.0,
            "svm_dev_mean": 0.8,
            "jerk_mean": 1.0,
            "jerk_std": 0.8,
            "jerk_max": 1.0,
            "jerk_energy": 0.7,
            "jerk_sparsity": 0.5,
        },
    },
    "rotation_balance": {
        "weight": 0.20,
        "features": {
            "omega_mean": 1.0,
            "omega_std": 0.8,
            "omega_max": 1.0,
            "angular_impulse": 0.8,
            "high_rot_n": 0.8,
        },
    },
    "posture_transition": {
        "weight": 0.20,
        "features": {
            "theta_std": 0.7,
            "theta_range": 1.0,
            "KII_mean": 0.8,
            "KII_std": 0.6,
            "KII_max": 0.8,
        },
    },
    "impact_event": {
        "weight": 0.25,
        "aggregation": "max",
        "features": {
            "GSI": 1.0,
            "fcri": 1.0,
            "free_fall_n": 0.8,
            "impact_n": 0.8,
        },
    },
    "physio_stress": {
        "weight": 0.13,
        "features": {
            "hr_delta": 0.6,
            "hr_spike": 0.8,
            "hr_accel": 0.6,
            "osi": 0.8,
            "css_max": 0.6,
            "css_mean": 0.6,
            "spo2_min": -0.7,
        },
    },
}


@dataclass(frozen=True)
class RiskTargetConfig:
    feature_weight: float = 0.60
    activity_prior_weight: float = 0.40
    activity_prior_floor_weight: float = 0.95
    low_threshold: float = RISK_THRESHOLDS["low"]
    medium_threshold: float = RISK_THRESHOLDS["medium"]


def select_feature_columns(df: pd.DataFrame, target_cols: set[str] | None = None) -> list[str]:
    ignored = set(META_COLUMNS)
    if target_cols:
        ignored.update(target_cols)

    feature_cols: list[str] = []
    for col in df.columns:
        if col in ignored:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            feature_cols.append(col)
    return feature_cols


def fit_robust_normalizer(
    df: pd.DataFrame, feature_cols: list[str], low_q: float = 0.05, high_q: float = 0.95
) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            params[col] = {"low": 0.0, "high": 1.0}
            continue
        low = float(finite.quantile(low_q))
        high = float(finite.quantile(high_q))
        if not np.isfinite(low):
            low = float(finite.min())
        if not np.isfinite(high):
            high = float(finite.max())
        if high <= low:
            high = low + 1.0
        params[col] = {"low": low, "high": high}
    return params


def normalize_feature(values: pd.Series, low: float, high: float, reverse: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    normalized = (numeric - low) / (high - low)
    normalized = normalized.clip(0.0, 1.0).fillna(0.0)
    if reverse:
        normalized = 1.0 - normalized
    return normalized


def compute_component_scores(
    df: pd.DataFrame, normalizer: dict[str, dict[str, float]]
) -> pd.DataFrame:
    components: dict[str, pd.Series] = {}
    for group_name, group_info in FEATURE_GROUPS.items():
        weighted_parts: list[pd.Series] = []
        weights: list[float] = []
        for feature_name, raw_weight in group_info["features"].items():
            if feature_name not in df.columns or feature_name not in normalizer:
                continue
            reverse = raw_weight < 0
            feature_weight = abs(float(raw_weight))
            params = normalizer[feature_name]
            weighted_parts.append(
                normalize_feature(
                    df[feature_name],
                    low=float(params["low"]),
                    high=float(params["high"]),
                    reverse=reverse,
                )
                * feature_weight
            )
            weights.append(feature_weight)

        if not weighted_parts or sum(weights) == 0:
            components[group_name] = pd.Series(np.zeros(len(df)), index=df.index)
        elif group_info.get("aggregation") == "max":
            stacked = pd.concat(
                [part / max(weight, 1e-12) for part, weight in zip(weighted_parts, weights)],
                axis=1,
            )
            components[group_name] = stacked.max(axis=1).clip(0.0, 1.0)
        else:
            components[group_name] = sum(weighted_parts) / sum(weights)
    return pd.DataFrame(components, index=df.index)


def compute_feature_risk_score(component_scores: pd.DataFrame) -> pd.Series:
    score = pd.Series(np.zeros(len(component_scores)), index=component_scores.index)
    total_weight = 0.0
    for group_name, group_info in FEATURE_GROUPS.items():
        if group_name not in component_scores.columns:
            continue
        weight = float(group_info["weight"])
        score += component_scores[group_name] * weight
        total_weight += weight
    if total_weight <= 0:
        return score.clip(0.0, 1.0)
    return (score / total_weight).clip(0.0, 1.0)


def compute_activity_prior(df: pd.DataFrame) -> pd.Series:
    prior = pd.Series(np.full(len(df), 0.30), index=df.index, dtype=float)

    if "category" in df.columns:
        category_prior = df["category"].astype(str).map(CATEGORY_RISK_PRIOR)
        prior = prior.where(category_prior.isna(), category_prior)

    if "class_en" in df.columns:
        class_prior = df["class_en"].astype(str).map(CLASS_RISK_PRIOR)
        prior = prior.where(class_prior.isna(), class_prior)

    if "class" in df.columns:
        class_prior = df["class"].astype(str).map(CLASS_RISK_PRIOR)
        prior = prior.where(class_prior.isna(), class_prior)

    return prior.clip(0.0, 1.0)


def risk_level_from_score(score: float) -> str:
    if score < RISK_THRESHOLDS["low"]:
        return "low"
    if score < RISK_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def build_proxy_risk_targets(
    df: pd.DataFrame,
    feature_cols: list[str],
    normalizer: dict[str, dict[str, float]] | None = None,
    config: RiskTargetConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    config = config or RiskTargetConfig()
    if normalizer is None:
        normalizer = fit_robust_normalizer(df, feature_cols)

    component_scores = compute_component_scores(df, normalizer)
    feature_score = compute_feature_risk_score(component_scores)
    activity_prior = compute_activity_prior(df)

    weighted_score = (
        config.feature_weight * feature_score
        + config.activity_prior_weight * activity_prior
    ).clip(0.0, 1.0)
    prior_floor = (config.activity_prior_floor_weight * activity_prior).clip(0.0, 1.0)
    risk_score = pd.concat([weighted_score, prior_floor], axis=1).max(axis=1).clip(0.0, 1.0)

    enriched = df.copy()
    for col in component_scores.columns:
        enriched[f"component_{col}"] = component_scores[col]
    enriched["feature_risk_score"] = feature_score
    enriched["activity_prior_score"] = activity_prior
    enriched["weighted_risk_score"] = weighted_score
    enriched["activity_prior_floor"] = prior_floor
    enriched["model2_risk_target"] = risk_score
    enriched["model2_risk_level"] = [
        risk_level_from_score(float(value)) for value in risk_score
    ]
    enriched["model2_high_risk_target"] = (risk_score >= config.medium_threshold).astype(int)
    return enriched, normalizer


def model2_config_payload(
    normalizer: dict[str, dict[str, float]], config: RiskTargetConfig | None = None
) -> dict[str, Any]:
    config = config or RiskTargetConfig()
    return {
        "note": (
            "Model 2 uses domain-informed proxy targets, not clinical ground-truth "
            "future fall outcomes. Use it as mobility risk assessment for prototype testing."
        ),
        "risk_target_formula": {
            "model2_risk_target": (
                "max("
                f"{config.feature_weight} * feature_risk_score + "
                f"{config.activity_prior_weight} * activity_prior_score, "
                f"{config.activity_prior_floor_weight} * activity_prior_score"
                ")"
            ),
            "level_thresholds": {
                "low": f"< {config.low_threshold}",
                "medium": f"{config.low_threshold} - < {config.medium_threshold}",
                "high": f">= {config.medium_threshold}",
            },
        },
        "class_risk_prior": CLASS_RISK_PRIOR,
        "category_risk_prior": CATEGORY_RISK_PRIOR,
        "feature_groups": FEATURE_GROUPS,
        "normalizer": normalizer,
    }
