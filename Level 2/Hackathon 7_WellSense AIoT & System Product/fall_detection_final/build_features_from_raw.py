# build_features_from_raw.py
# --------------------------
# Read raw rows from merged_dataset_full.csv, compute sliding-window features
# required by Model 2 (svm, jerk, KII, omega, theta, GSI, fcri, ...).
#
# OUTPUT: windows_from_merged.csv
#   -> drop-in replacement for windows_all.csv with train_model2_risk_assessment.py
#
# Usage (PowerShell):
#   cd "<project>/fall_detection_final"
#   python .\build_features_from_raw.py
#   python .\build_features_from_raw.py --input merged_dataset_full.csv --output windows_from_merged.csv --window-sec 2.0 --step-sec 0.5
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
G = 9.80665          # m/s² per g  (Nano 33 BLE Sense ส่งมาเป็น g อยู่แล้ว)
DEG2RAD = np.pi / 180.0

# column name mapping: ชื่อใน merged_dataset_full.csv → ชื่อใน code
COL_MAP = {
    "ax": "ax",   # accel X (unit: g)
    "ay": "ay",
    "az": "az",
    "gx": "gx",   # gyro X (unit: deg/s)
    "gy": "gy",
    "gz": "gz",
    "press": "press",     # pressure (kPa)
    "mic":   "mic",       # microphone mean abs
    "p2p":   "p2p",       # mic peak-to-peak
    "class_en": "class_en",
    "category": "category",
}

# ─────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────

def svm(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Signal Vector Magnitude (g)"""
    return np.sqrt(ax**2 + ay**2 + az**2)


def jerk_magnitude(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, dt: float) -> np.ndarray:
    """Jerk = derivative of acceleration (g/s)"""
    dax = np.diff(ax, prepend=ax[0]) / dt
    day = np.diff(ay, prepend=ay[0]) / dt
    daz = np.diff(az, prepend=az[0]) / dt
    return np.sqrt(dax**2 + day**2 + daz**2)


def omega_magnitude(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Angular velocity magnitude (deg/s)"""
    return np.sqrt(gx**2 + gy**2 + gz**2)


def kii(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """
    KII = Kinematic Instability Index
    = SVM deviation from 1 g (ค่าเบี่ยงเบนของ SVM จาก 1g ซึ่งเป็น gravity baseline)
    """
    return np.abs(svm(ax, ay, az) - 1.0)


def theta_angle(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """
    Body tilt angle (degrees) = atan2(sqrt(ax²+ay²), |az|)
    ประมาณมุมเอียงของลำตัวจาก vertical
    """
    horizontal = np.sqrt(ax**2 + ay**2)
    return np.degrees(np.arctan2(horizontal, np.abs(az) + 1e-9))


def gsi(jerk: np.ndarray) -> float:
    """
    GSI (Global Shock Index) = RMS ของ jerk ทั้ง window
    ค่าสูง = มีแรงกระแทกรุนแรง
    """
    return float(np.sqrt(np.mean(jerk**2)))


def fcri(svm_arr: np.ndarray, jerk_arr: np.ndarray) -> float:
    """
    FCRI (Fall-Critical Risk Index) = max(svm) * max(jerk) / 10
    Composite impact severity index
    """
    return float(np.max(svm_arr) * np.max(jerk_arr) / 10.0)


def count_free_fall(svm_arr: np.ndarray, threshold: float = 0.5) -> int:
    """จำนวน samples ที่ SVM < threshold (free fall zone)"""
    return int(np.sum(svm_arr < threshold))


def count_impact(svm_arr: np.ndarray, threshold: float = 2.0) -> int:
    """จำนวน samples ที่ SVM > threshold (impact zone)"""
    return int(np.sum(svm_arr > threshold))


def count_high_rotation(omega_arr: np.ndarray, threshold: float = 100.0) -> int:
    """จำนวน samples ที่ angular velocity สูง (deg/s)"""
    return int(np.sum(omega_arr > threshold))


def angular_impulse(omega_arr: np.ndarray, dt: float) -> float:
    """Angular impulse = integral of omega * dt"""
    return float(np.sum(omega_arr) * dt)


def press_delta(press_arr: np.ndarray) -> float:
    """Pressure change over window (kPa)"""
    if len(press_arr) < 2:
        return 0.0
    return float(press_arr[-1] - press_arr[0])


def press_slope(press_arr: np.ndarray, dt: float) -> float:
    """Pressure slope (kPa/s)"""
    n = len(press_arr)
    if n < 2:
        return 0.0
    x = np.arange(n) * dt
    return float(np.polyfit(x, press_arr, 1)[0])


# ─────────────────────────────────────────────
# Window feature extractor
# ─────────────────────────────────────────────

def extract_window_features(chunk: pd.DataFrame, dt: float, session_id: str) -> dict:
    """
    รับ DataFrame chunk (1 window) แล้ว return dict ของ features
    """
    ax = chunk["ax"].to_numpy(dtype=float)
    ay = chunk["ay"].to_numpy(dtype=float)
    az = chunk["az"].to_numpy(dtype=float)
    gx = chunk["gx"].to_numpy(dtype=float) if "gx" in chunk.columns else np.zeros(len(ax))
    gy = chunk["gy"].to_numpy(dtype=float) if "gy" in chunk.columns else np.zeros(len(ax))
    gz = chunk["gz"].to_numpy(dtype=float) if "gz" in chunk.columns else np.zeros(len(ax))

    svm_arr    = svm(ax, ay, az)
    jerk_arr   = jerk_magnitude(ax, ay, az, dt)
    omega_arr  = omega_magnitude(gx, gy, gz)
    kii_arr    = kii(ax, ay, az)
    theta_arr  = theta_angle(ax, ay, az)

    # ── press / mic (optional) ──────────────────
    press_arr = chunk["press"].to_numpy(dtype=float) if "press" in chunk.columns else np.zeros(len(ax))
    mic_arr   = chunk["mic"].to_numpy(dtype=float)   if "mic"   in chunk.columns else np.zeros(len(ax))
    p2p_arr   = chunk["p2p"].to_numpy(dtype=float)   if "p2p"   in chunk.columns else np.zeros(len(ax))

    # ── timestamps ──────────────────────────────
    ts_start = str(chunk["pc_timestamp_iso"].iloc[0])  if "pc_timestamp_iso" in chunk.columns else ""
    ts_end   = str(chunk["pc_timestamp_iso"].iloc[-1]) if "pc_timestamp_iso" in chunk.columns else ""

    # ── meta ─────────────────────────────────────
    class_en  = str(chunk["class_en"].iloc[0])  if "class_en"  in chunk.columns else "unknown"
    class_th  = str(chunk["class_th"].iloc[0])  if "class_th"  in chunk.columns else ""
    category  = str(chunk["category"].iloc[0])  if "category"  in chunk.columns else "unknown"
    label_col = "label" if "label" in chunk.columns else None
    label_val = int(chunk[label_col].iloc[0]) if label_col and not pd.isna(chunk[label_col].iloc[0]) else (1 if category == "fall" else 0)
    source    = str(chunk["src_file"].iloc[0]) if "src_file" in chunk.columns else "merged"

    feat = {
        # ── meta ──────────────────────────────────────────────────────────
        "window_start_ts": ts_start,
        "window_end_ts":   ts_end,
        "session_id":      session_id,
        "class_th":        class_th,
        "label":           label_val,
        "source":          "original",   # raw data = original
        "class_en":        class_en,
        "category":        category,
        # ── SVM ───────────────────────────────────────────────────────────
        "svm_mean":    float(np.mean(svm_arr)),
        "svm_std":     float(np.std(svm_arr)),
        "svm_max":     float(np.max(svm_arr)),
        "svm_min":     float(np.min(svm_arr)),
        "svm_dev_mean": float(np.mean(np.abs(svm_arr - 1.0))),
        # ── Jerk ──────────────────────────────────────────────────────────
        "jerk_mean":   float(np.mean(jerk_arr)),
        "jerk_std":    float(np.std(jerk_arr)),
        "jerk_max":    float(np.max(jerk_arr)),
        "jerk_min":    float(np.min(jerk_arr)),
        "jerk_energy": float(np.sum(jerk_arr**2) * dt),
        "jerk_sparsity": float(np.mean(jerk_arr < np.percentile(jerk_arr, 25))),
        # ── KII ───────────────────────────────────────────────────────────
        "KII_mean":    float(np.mean(kii_arr)),
        "KII_std":     float(np.std(kii_arr)),
        "KII_max":     float(np.max(kii_arr)),
        "KII_min":     float(np.min(kii_arr)),
        # ── Omega (angular velocity) ───────────────────────────────────────
        "omega_mean":  float(np.mean(omega_arr)),
        "omega_std":   float(np.std(omega_arr)),
        "omega_max":   float(np.max(omega_arr)),
        "omega_min":   float(np.min(omega_arr)),
        # ── Theta (tilt angle) ────────────────────────────────────────────
        "theta_mean":  float(np.mean(theta_arr)),
        "theta_std":   float(np.std(theta_arr)),
        "theta_max":   float(np.max(theta_arr)),
        "theta_min":   float(np.min(theta_arr)),
        "theta_range": float(np.max(theta_arr) - np.min(theta_arr)),
        # ── Event counts ──────────────────────────────────────────────────
        "free_fall_n": count_free_fall(svm_arr),
        "impact_n":    count_impact(svm_arr),
        "high_rot_n":  count_high_rotation(omega_arr),
        # ── Impact severity ───────────────────────────────────────────────
        "GSI":  gsi(jerk_arr),
        "fcri": fcri(svm_arr, jerk_arr),
        # ── Angular impulse ───────────────────────────────────────────────
        "angular_impulse": angular_impulse(omega_arr, dt),
        # ── Pressure ──────────────────────────────────────────────────────
        "press_delta": press_delta(press_arr),
        "press_slope": press_slope(press_arr, dt),
        # ── Microphone ────────────────────────────────────────────────────
        "mic_p2p_max":  float(np.max(p2p_arr)) if len(p2p_arr) > 0 else 0.0,
        "mic_p2p_mean": float(np.mean(p2p_arr)) if len(p2p_arr) > 0 else 0.0,
        # ── PPG (ถ้าไม่มีข้อมูล ใส่ 0 ไปก่อน Model จะใช้ imputer เติมค่า) ──
        "hr_mean":  0.0,
        "hr_max":   0.0,
        "hr_delta": 0.0,
        "hr_spike": 0.0,
        "spo2_min": 0.0,
        "spo2_mean":0.0,
        "rmssd":    0.0,
        "sdnn":     0.0,
        "hr_accel": 0.0,
        "osi":      0.0,
        "css_max":  0.0,
        "css_mean": 0.0,
    }
    return feat


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build windowed features from raw merged_dataset_full.csv")
    parser.add_argument("--input",      default="merged_dataset_full.csv",  help="Input raw CSV")
    parser.add_argument("--output",     default="windows_from_merged.csv",  help="Output windowed CSV")
    parser.add_argument("--window-sec", type=float, default=2.0,  help="Window duration in seconds")
    parser.add_argument("--step-sec",   type=float, default=0.5,  help="Sliding window step in seconds")
    parser.add_argument("--sample-hz",  type=float, default=20.0, help="Expected sample rate (Hz). Used only if timestamp parse fails.")
    parser.add_argument("--min-samples", type=int, default=10,    help="Minimum samples per window (skip smaller windows)")
    parser.add_argument("--original-only", action="store_true",   help="Use only rows where source=original (no augmented data)")
    return parser.parse_args()


def infer_sample_rate(df: pd.DataFrame, fallback_hz: float = 20.0) -> float:
    """คำนวณ sample rate จริงจาก timestamp"""
    if "ms" in df.columns:
        ms_vals = pd.to_numeric(df["ms"], errors="coerce").dropna()
        if len(ms_vals) > 10:
            diffs = ms_vals.diff().dropna()
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                median_dt_ms = float(diffs.median())
                if 1 < median_dt_ms < 500:
                    hz = 1000.0 / median_dt_ms
                    print(f"  -> inferred sample rate: {hz:.1f} Hz (dt={median_dt_ms:.1f} ms)")
                    return hz
    print(f"  -> using fallback sample rate: {fallback_hz:.1f} Hz")
    return fallback_hz


def sliding_windows_for_group(
    group: pd.DataFrame,
    window_samples: int,
    step_samples: int,
    dt: float,
    session_id: str,
    min_samples: int,
) -> list[dict]:
    """Sliding window ภายใน 1 group (class session)"""
    rows: list[dict] = []
    n = len(group)
    start = 0
    while start + min_samples <= n:
        end = min(start + window_samples, n)
        chunk = group.iloc[start:end]
        if len(chunk) >= min_samples:
            feat = extract_window_features(chunk, dt, session_id)
            rows.append(feat)
        start += step_samples
    return rows


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        # ถ้า path ไม่ absolute ให้ resolve จาก CWD
        input_path = Path.cwd() / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {input_path}")

    print(f"\n{'='*60}")
    print(f"  build_features_from_raw.py")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Window : {args.window_sec:.1f} sec | Step: {args.step_sec:.1f} sec")
    print(f"{'='*60}\n")

    # ── 1. Load raw data ─────────────────────────────────────────
    print("[ 1/5 ] Loading raw CSV...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  -> {len(df):,} rows x {len(df.columns)} columns loaded")

    # กรอง original only ถ้า flag ตั้ง
    if args.original_only and "src_file" in df.columns:
        before = len(df)
        df = df[~df["src_file"].astype(str).str.contains("aug", na=False)].copy()
        print(f"  -> original-only filter: {before:,} -> {len(df):,} rows")

    # ── 2. Infer sample rate ─────────────────────────────────────
    print("\n[ 2/5 ] Inferring sample rate...")
    sample_hz      = infer_sample_rate(df, fallback_hz=args.sample_hz)
    dt             = 1.0 / sample_hz
    window_samples = max(1, int(args.window_sec * sample_hz))
    step_samples   = max(1, int(args.step_sec   * sample_hz))
    print(f"  -> window={window_samples} samples | step={step_samples} samples")

    # ── 3. Resolve column names ──────────────────────────────────
    print("\n[ 3/5 ] Resolving column names...")

    # รองรับทั้ง 'ax' และ 'acc_x_g'
    accel_x = "ax" if "ax" in df.columns else ("acc_x_g" if "acc_x_g" in df.columns else None)
    accel_y = "ay" if "ay" in df.columns else ("acc_y_g" if "acc_y_g" in df.columns else None)
    accel_z = "az" if "az" in df.columns else ("acc_z_g" if "acc_z_g" in df.columns else None)
    gyro_x  = "gx" if "gx" in df.columns else ("gyro_x_dps" if "gyro_x_dps" in df.columns else None)
    gyro_y  = "gy" if "gy" in df.columns else ("gyro_y_dps" if "gyro_y_dps" in df.columns else None)
    gyro_z  = "gz" if "gz" in df.columns else ("gyro_z_dps" if "gyro_z_dps" in df.columns else None)

    if accel_x is None:
        raise ValueError("ไม่พบ column acceleration (ax หรือ acc_x_g) ใน CSV")

    # rename ให้เป็น standard names
    rename_map = {}
    for src, dst in [(accel_x,"ax"),(accel_y,"ay"),(accel_z,"az"),
                     (gyro_x,"gx"),(gyro_y,"gy"),(gyro_z,"gz")]:
        if src and src != dst and src in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(columns=rename_map)
    print(f"  -> renamed: {rename_map}")

    # class_en fallback -> class_en
    if "class_en" not in df.columns and "class" in df.columns:
        df["class_en"] = df["class"]
        print("  -> using 'class' column as class_en")

    # category fallback
    if "category" not in df.columns and "class_en" in df.columns:
        fall_classes = {"slow_collapse_fall", "gradual_fall", "sideways_fall", "backward_fall"}
        df["category"] = df["class_en"].apply(
            lambda c: "fall" if str(c) in fall_classes else "activity"
        )
        print("  -> derived 'category' from class_en")

    # class_th fallback
    if "class_th" not in df.columns:
        df["class_th"] = df.get("class_en", "")

    required = ["ax", "ay", "az"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # convert to numeric
    for col in ["ax", "ay", "az", "gx", "gy", "gz", "press", "mic", "p2p", "ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ax", "ay", "az"]).reset_index(drop=True)
    print(f"  -> {len(df):,} rows after dropping NaN in accel columns")

    # ── 4. Group by class + file, sliding window ─────────────────
    print("\n[ 4/5 ] Extracting windowed features...")

    # สร้าง session_id from class_en + src_file (ถ้ามี)
    if "src_file" in df.columns:
        df["_session"] = df["class_en"].astype(str) + "||" + df["src_file"].astype(str)
    else:
        df["_session"] = df["class_en"].astype(str)

    all_rows: list[dict] = []
    groups = df.groupby("_session", sort=False)
    total_groups = len(groups)

    for i, (session_key, group) in enumerate(groups, 1):
        group = group.reset_index(drop=True)
        session_id = session_key.replace("||", "_")

        window_rows = sliding_windows_for_group(
            group,
            window_samples=window_samples,
            step_samples=step_samples,
            dt=dt,
            session_id=session_id,
            min_samples=args.min_samples,
        )
        all_rows.extend(window_rows)

        if i % 10 == 0 or i == total_groups:
            print(f"  [{i:4d}/{total_groups}] {session_key[:60]:<60} -> {len(window_rows):4d} windows")

    print(f"  -> Total windows extracted: {len(all_rows):,}")

    # ── 5. Save output ────────────────────────────────────────────
    print(f"\n[ 5/5 ] Saving to {output_path} ...")
    out_df = pd.DataFrame(all_rows)

    # แสดง class distribution
    if "class_en" in out_df.columns:
        print("\n  Class distribution in output:")
        counts = out_df["class_en"].value_counts()
        for cls, cnt in counts.items():
            print(f"    {cls:<35} {cnt:5d} windows")

    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Saved: {output_path}")
    print(f"       Rows : {len(out_df):,}")
    print(f"       Cols : {len(out_df.columns)}")
    print(f"\n{'='*60}")
    print("  [OK] Done! Use this file with Model 2:")
    print("")
    print(f"  python .\\train_model2_risk_assessment.py ^")
    print(f"    --data .\\{output_path.name} ^")
    print("    --run-name model2_from_merged")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
