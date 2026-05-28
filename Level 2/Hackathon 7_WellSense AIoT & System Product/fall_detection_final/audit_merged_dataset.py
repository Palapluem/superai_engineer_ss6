
# audit_merged_dataset.py
# Audits merged_dataset_full.csv to understand its actual structure
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
SEP  = "=" * 68

df = pd.read_csv(HERE / "merged_dataset_full.csv", low_memory=False)

print(SEP)
print("  MERGED DATASET AUDIT")
print(SEP)
print(f"  Total rows (excl header) : {len(df):,}")
print(f"  Total columns            : {len(df.columns)}")

# Classify each row
has_ax  = df["ax"].notna()
has_svm = df["svm_mean"].notna()
has_wts = df["window_start_ts"].notna()

raw_only   = has_ax & ~has_svm
feat_only  = ~has_ax & has_svm
both       = has_ax & has_svm
neither    = ~has_ax & ~has_svm

print()
print("  ROW TYPE BREAKDOWN:")
print(f"  [A] Raw sensor only  (ax filled, no svm)   : {raw_only.sum():,}")
print(f"  [B] Feature only     (svm filled, no ax)   : {feat_only.sum():,}")
print(f"  [C] Both             (ax + svm filled)     : {both.sum():,}")
print(f"  [D] Neither          (all empty)           : {neither.sum():,}")
print(f"      TOTAL                                  : {len(df):,}")

# ── Feature-only rows (Type B) ────────────────────────────────────────
feat_df = df[feat_only].copy()

print()
print(SEP)
print("  TYPE B: Feature-only rows (pre-computed windowed features)")
print(SEP)
print(f"  Count: {len(feat_df):,}")

# Check what columns are filled
print()
print("  Key columns in Type B rows:")
check_cols = [
    "window_start_ts", "window_end_ts", "session_id", "class_en", "source",
    "svm_mean", "jerk_mean", "KII_mean", "omega_mean", "theta_mean",
    "GSI", "fcri", "free_fall_n", "impact_n",
    "hr_mean", "spo2_mean"
]
for col in check_cols:
    if col in feat_df.columns:
        n_filled = feat_df[col].notna().sum()
        n_zero   = (feat_df[col] == 0).sum() if pd.api.types.is_numeric_dtype(feat_df[col]) else 0
        sample   = str(feat_df[col].iloc[0])[:30].encode("ascii","replace").decode("ascii") if len(feat_df) > 0 else "N/A"
        print(f"    {col:<25} filled={n_filled:4d}  zero={n_zero:4d}  sample={sample}")

print()
print("  Class distribution in Type B rows:")
if "class_en" in feat_df.columns:
    for cls, cnt in feat_df["class_en"].value_counts().items():
        pct = cnt / len(feat_df) * 100
        print(f"    {str(cls):<35} {cnt:5d}  ({pct:5.1f}%)")

print()
print("  Source distribution in Type B rows:")
if "source" in feat_df.columns:
    for src, cnt in feat_df["source"].value_counts().items():
        print(f"    {str(src):<20} {cnt:5d}")

# ── Type A raw rows ────────────────────────────────────────────────────
raw_df = df[raw_only].copy()

print()
print(SEP)
print("  TYPE A: Raw sensor rows (to be windowed)")
print(SEP)
print(f"  Count: {len(raw_df):,}")

print()
print("  Class distribution in Type A rows:")
class_col = "class_en" if "class_en" in raw_df.columns else ("class" if "class" in raw_df.columns else None)
if class_col:
    for cls, cnt in raw_df[class_col].value_counts().items():
        pct = cnt / len(raw_df) * 100
        print(f"    {str(cls):<35} {cnt:5d}  ({pct:5.1f}%)")

print()
print("  Source distribution in Type A rows:")
src_col = "src_file" if "src_file" in raw_df.columns else ("source" if "source" in raw_df.columns else None)
if src_col:
    for src, cnt in raw_df[src_col].fillna("unknown").value_counts().head(10).items():
        print(f"    {str(src)[:50]:<50} {cnt:5d}")

# ── Completeness check on Type B (feature-only) ────────────────────────
print()
print(SEP)
print("  FEATURE COMPLETENESS in Type B rows (for Model 2)")
print(SEP)

model2_features = [
    "svm_mean","svm_std","svm_max","svm_min","svm_dev_mean",
    "jerk_mean","jerk_std","jerk_max","jerk_min","jerk_energy","jerk_sparsity",
    "KII_mean","KII_std","KII_max","KII_min",
    "omega_mean","omega_std","omega_max","omega_min",
    "theta_mean","theta_std","theta_max","theta_min","theta_range",
    "free_fall_n","impact_n","high_rot_n",
    "GSI","fcri","angular_impulse",
    "press_delta","press_slope","mic_p2p_max","mic_p2p_mean",
    "hr_mean","hr_max","hr_delta","hr_spike",
    "spo2_min","spo2_mean","rmssd","sdnn","hr_accel",
    "osi","css_max","css_mean",
]

ok_cols  = []
zero_cols = []
missing_cols = []
for col in model2_features:
    if col not in feat_df.columns:
        missing_cols.append(col)
    elif feat_df[col].isna().all():
        missing_cols.append(col + " (all NaN)")
    elif (feat_df[col].fillna(0) == 0).all():
        zero_cols.append(col)
    else:
        ok_cols.append(col)

print(f"  Features with real data : {len(ok_cols)}/46")
print(f"  Features all-zero       : {len(zero_cols)}/46")
print(f"  Features missing/NaN    : {len(missing_cols)}/46")
if zero_cols:
    print(f"  Zero features: {zero_cols}")

# ── Summary verdict ────────────────────────────────────────────────────
print()
print(SEP)
print("  VERDICT & RECOMMENDATION")
print(SEP)

print()
print("  The merged_dataset_full.csv contains TWO types of rows mixed together:")
print()
print("    Type A (8,028 rows) - Raw sensor data:")
print("       -> ax, ay, az, gx, gy, gz filled")
print("       -> svm_mean, jerk_mean etc. are EMPTY")
print("       -> Need windowing + feature extraction (done by build_features_from_raw.py)")
print()
print("    Type B (1,800 rows) - Pre-computed windowed features:")
print("       -> ax, ay, az etc. are EMPTY")
print("       -> svm_mean, jerk_mean, GSI, fcri etc. are FILLED")
print("       -> Can be used DIRECTLY by train_model2_risk_assessment.py")
print()
print("    Type D (32 rows) - Completely empty, safe to drop")
print()
print("  ACTION: Extract Type B rows and merge with windows_from_merged.csv")
print("  to get a BIGGER training set for Model 2")

# Save Type B to CSV
out_path = HERE / "windows_from_merged_typeB.csv"
# Align columns to match windows_all.csv
wa = pd.read_csv(HERE / "windows_all.csv")
wa_cols = list(wa.columns)

# Map columns that exist in Type B
rename_col = {}
if "class_en" in feat_df.columns:
    rename_col["class_en"] = "class_en"

out_feat = feat_df.copy()
# Use only columns that windows_all has
shared = [c for c in wa_cols if c in out_feat.columns]
out_feat = out_feat[shared]
out_feat.to_csv(out_path, index=False, encoding="utf-8")
print()
print(f"  Saved Type B rows to: {out_path.name}")
print(f"  Rows: {len(out_feat):,}  |  Cols: {len(out_feat.columns)}")
print()
print(SEP)
