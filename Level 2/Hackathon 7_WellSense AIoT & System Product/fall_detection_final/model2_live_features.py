from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from build_features_from_raw import extract_window_features


RAW_NUMERIC_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "press", "mic", "p2p"]


def _safe_datetime(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def normalize_raw_samples(samples: list[dict[str, Any]]) -> pd.DataFrame:
    if not samples:
        raise ValueError("At least one raw IMU sample is required.")

    df = pd.DataFrame(samples).copy()
    rename_map = {
        "pc_ts": "pc_timestamp_iso",
        "timestamp": "pc_timestamp_iso",
        "time": "pc_timestamp_iso",
    }
    for source, target in rename_map.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]

    missing = [col for col in ["ax", "ay", "az"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required accelerometer columns: {missing}")

    for col in RAW_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["gx", "gy", "gz", "press", "mic", "p2p"]:
        if col not in df.columns:
            df[col] = 0.0

    df[["ax", "ay", "az", "gx", "gy", "gz"]] = df[
        ["ax", "ay", "az", "gx", "gy", "gz"]
    ].fillna(0.0)

    if "pc_timestamp_iso" not in df.columns:
        now = datetime.now().isoformat(timespec="milliseconds")
        df["pc_timestamp_iso"] = now

    return df.reset_index(drop=True)


def infer_dt_seconds(df: pd.DataFrame, default_dt: float = 0.05) -> float:
    if "ms" in df.columns:
        ms = pd.to_numeric(df["ms"], errors="coerce").dropna().sort_values()
        diffs = ms.diff().dropna()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            return float(np.clip(diffs.median() / 1000.0, 0.01, 0.20))

    if "pc_timestamp_iso" in df.columns:
        ts = pd.to_datetime(df["pc_timestamp_iso"], errors="coerce").dropna().sort_values()
        diffs = ts.diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            return float(np.clip(diffs.median(), 0.01, 0.20))

    return default_dt


def extract_features_from_samples(
    samples: list[dict[str, Any]],
    session_id: str = "live_window",
) -> dict[str, Any]:
    df = normalize_raw_samples(samples)
    dt = infer_dt_seconds(df)
    features = extract_window_features(df, dt=dt, session_id=session_id)
    features["sample_count"] = int(len(df))
    features["dt_seconds"] = float(dt)
    return features


def latest_raw_values(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0}
    latest = samples[-1]
    return {
        key: float(pd.to_numeric(latest.get(key, 0.0), errors="coerce") or 0.0)
        for key in ["ax", "ay", "az", "gx", "gy", "gz"]
    }
