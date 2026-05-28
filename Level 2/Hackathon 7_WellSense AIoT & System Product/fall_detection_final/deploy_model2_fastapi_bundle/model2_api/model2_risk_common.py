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
    "low": 0.35,
    "high": 0.70,
    "alert": 0.70,
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
    high_threshold: float = RISK_THRESHOLDS["high"]
    alert_threshold: float = RISK_THRESHOLDS["alert"]
    quiet_motion_cap: float = RISK_THRESHOLDS["low"]
    minor_motion_cap: float = 0.55
    single_signal_cap: float = 0.55
    high_component_threshold: float = 0.65
    impact_component_threshold: float = 0.45
    severe_motion_threshold: float = 0.85
    min_high_components_for_high: int = 2


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
    if score <= RISK_THRESHOLDS["low"]:
        return "low"
    if score <= RISK_THRESHOLDS["high"]:
        return "medium"
    return "high"


def apply_conservative_risk_guard(
    risk_score: pd.Series | np.ndarray | list[float],
    component_scores: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    config: RiskTargetConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cap high-risk outputs unless the window has enough motion evidence.

    A single sensitive channel can spike when the board is nudged by hand. For a
    true high-risk state we require either impact evidence or more than one
    elevated component. This keeps small board movements in low/medium instead
    of jumping directly to high/fall alert.
    """
    config = config or RiskTargetConfig()
    score = pd.Series(risk_score, index=component_scores.index, dtype=float).clip(0.0, 1.0)

    def component(name: str) -> pd.Series:
        if name in component_scores.columns:
            return pd.to_numeric(component_scores[name], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        return pd.Series(np.zeros(len(component_scores)), index=component_scores.index)

    gait = component("gait_motion")
    rotation = component("rotation_balance")
    posture = component("posture_transition")
    impact = component("impact_event")

    def feature(name: str) -> pd.Series:
        if feature_df is not None and name in feature_df.columns:
            return pd.to_numeric(feature_df[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(component_scores)), index=component_scores.index)

    free_fall_n = feature("free_fall_n")
    impact_n = feature("impact_n")
    high_rot_n = feature("high_rot_n")
    theta_range = feature("theta_range")
    omega_max = feature("omega_max")
    svm_max = feature("svm_max")
    svm_min = feature("svm_min")
    jerk_max = feature("jerk_max")

    motion_max = pd.concat([gait, rotation, posture], axis=1).max(axis=1)
    high_component_count = (
        (gait >= config.high_component_threshold).astype(int)
        + (rotation >= config.high_component_threshold).astype(int)
        + (posture >= config.high_component_threshold).astype(int)
        + (impact >= config.impact_component_threshold).astype(int)
    )
    component_quiet_motion = (
        (gait <= 0.35)
        & (rotation <= 0.35)
        & (posture <= 0.35)
        & (impact <= 0.20)
    )
    physically_quiet_motion = (
        (svm_max <= 1.20)
        & (svm_min >= 0.80)
        & (jerk_max <= 3.0)
        & (omega_max <= 25.0)
        & (theta_range <= 10.0)
        & (impact_n <= 0)
        & (free_fall_n <= 0)
    )
    minor_board_motion = (
        (svm_max <= 1.30)
        & (svm_min >= 0.70)
        & (jerk_max <= 6.0)
        & (omega_max <= 60.0)
        & (theta_range <= 20.0)
        & (impact_n <= 0)
        & (free_fall_n <= 0)
    )
    quiet_motion = component_quiet_motion | physically_quiet_motion
    explicit_impact = (impact_n > 0) | (free_fall_n > 0) | (svm_max >= 1.80)
    sustained_fall_motion = (
        ((high_rot_n >= 2) | (omega_max >= 120.0))
        & (theta_range >= 45.0)
    )
    has_enough_high_evidence = (
        (high_component_count >= config.min_high_components_for_high)
        | explicit_impact
        | sustained_fall_motion
        | (
            (motion_max >= config.severe_motion_threshold)
            & (high_component_count >= config.min_high_components_for_high - 1)
        )
    )

    guarded = score.copy()
    reason = pd.Series("none", index=component_scores.index, dtype=object)

    quiet_mask = quiet_motion & (guarded > config.quiet_motion_cap)
    guarded.loc[quiet_mask] = config.quiet_motion_cap
    reason.loc[quiet_mask] = "quiet_motion_cap"

    minor_motion_mask = (
        minor_board_motion
        & (~quiet_motion)
        & (guarded > config.minor_motion_cap)
    )
    guarded.loc[minor_motion_mask] = config.minor_motion_cap
    reason.loc[minor_motion_mask] = "minor_motion_cap"

    weak_high_mask = (guarded > config.high_threshold) & (~has_enough_high_evidence)
    guarded.loc[weak_high_mask] = config.single_signal_cap
    reason.loc[weak_high_mask & ~quiet_mask & ~minor_motion_mask] = "single_signal_cap"

    guard_info = pd.DataFrame(
        {
            "guard_reason": reason,
            "guard_high_component_count": high_component_count,
            "guard_motion_max": motion_max,
            "guard_impact_component": impact,
            "guard_physically_quiet_motion": physically_quiet_motion,
            "guard_minor_board_motion": minor_board_motion,
            "guard_explicit_impact": explicit_impact,
            "guard_sustained_fall_motion": sustained_fall_motion,
            "guard_has_high_evidence": has_enough_high_evidence,
        },
        index=component_scores.index,
    )
    return guarded.clip(0.0, 1.0), guard_info


def high_risk_alert_from_outputs(
    risk_score: float,
    high_risk_probability: float | None = None,
    config: RiskTargetConfig | None = None,
) -> bool:
    config = config or RiskTargetConfig()
    if risk_score <= config.alert_threshold:
        return False
    if high_risk_probability is None or not np.isfinite(high_risk_probability):
        return True
    return high_risk_probability >= config.alert_threshold


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
    raw_risk_score = pd.concat([weighted_score, prior_floor], axis=1).max(axis=1).clip(0.0, 1.0)
    risk_score, guard_info = apply_conservative_risk_guard(
        raw_risk_score, component_scores, df, config=config
    )

    enriched = df.copy()
    for col in component_scores.columns:
        enriched[f"component_{col}"] = component_scores[col]
    for col in guard_info.columns:
        enriched[col] = guard_info[col]
    enriched["feature_risk_score"] = feature_score
    enriched["activity_prior_score"] = activity_prior
    enriched["weighted_risk_score"] = weighted_score
    enriched["activity_prior_floor"] = prior_floor
    enriched["raw_model2_risk_target"] = raw_risk_score
    enriched["model2_risk_target"] = risk_score
    enriched["model2_risk_level"] = [
        risk_level_from_score(float(value)) for value in risk_score
    ]
    enriched["model2_high_risk_target"] = (risk_score > config.high_threshold).astype(int)
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
                "low": f"0.00 - {config.low_threshold}",
                "medium": f"> {config.low_threshold} - {config.high_threshold}",
                "high": f"> {config.high_threshold}",
                "alert": f"> {config.alert_threshold} and high-risk probability >= {config.alert_threshold}",
            },
            "conservative_guard": {
                "quiet_motion_cap": config.quiet_motion_cap,
                "minor_motion_cap": config.minor_motion_cap,
                "single_signal_cap": config.single_signal_cap,
                "high_component_threshold": config.high_component_threshold,
                "impact_component_threshold": config.impact_component_threshold,
                "severe_motion_threshold": config.severe_motion_threshold,
                "min_high_components_for_high": config.min_high_components_for_high,
            },
        },
        "class_risk_prior": CLASS_RISK_PRIOR,
        "category_risk_prior": CATEGORY_RISK_PRIOR,
        "feature_groups": FEATURE_GROUPS,
        "normalizer": normalizer,
    }
