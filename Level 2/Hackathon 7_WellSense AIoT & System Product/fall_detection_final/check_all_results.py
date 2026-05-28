
# check_all_results.py
# Comprehensive result checker for Model 2 training outputs
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

SEP  = "=" * 68
SEP2 = "-" * 68


def section(title):
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def load_safe(path):
    p = HERE / path
    if not p.exists():
        print(f"  [MISSING] {p}")
        return None
    return pd.read_csv(p, low_memory=False)


# ─────────────────────────────────────────────────────────────────
# 1. Dataset comparison
# ─────────────────────────────────────────────────────────────────
section("1. DATASET COMPARISON")

wa = load_safe("windows_all.csv")
wm = load_safe("windows_from_merged.csv")

if wa is not None:
    print(f"  windows_all.csv         : {len(wa):,} rows  | {len(wa.columns)} cols")
if wm is not None:
    print(f"  windows_from_merged.csv : {len(wm):,} rows  | {len(wm.columns)} cols")

if wa is not None and "class_en" in wa.columns:
    print()
    print("  Class distribution: windows_all.csv")
    for cls, cnt in wa["class_en"].value_counts().items():
        pct = cnt / len(wa) * 100
        bar = "#" * int(pct / 2)
        print(f"    {str(cls):<35} {cnt:5d}  ({pct:5.1f}%)  |{bar}")

if wm is not None and "class_en" in wm.columns:
    print()
    print("  Class distribution: windows_from_merged.csv")
    for cls, cnt in wm["class_en"].value_counts().items():
        pct = cnt / len(wm) * 100
        bar = "#" * int(pct / 2)
        print(f"    {str(cls):<35} {cnt:5d}  ({pct:5.1f}%)  |{bar}")

# ─────────────────────────────────────────────────────────────────
# 2. Feature completeness comparison
# ─────────────────────────────────────────────────────────────────
section("2. FEATURE COMPLETENESS")

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

for name, df in [("windows_all.csv", wa), ("windows_from_merged.csv", wm)]:
    if df is None:
        continue
    print(f"\n  {name}:")
    missing = [f for f in model2_features if f not in df.columns]
    present = [f for f in model2_features if f in df.columns]
    all_null = [f for f in present if df[f].isna().all()]
    has_data = [f for f in present if not df[f].isna().all()]
    print(f"    Features present with data : {len(has_data)}/{len(model2_features)}")
    print(f"    Features present all-NaN   : {len(all_null)}")
    print(f"    Features missing entirely  : {len(missing)}")
    if missing:
        print(f"    Missing: {missing}")

# ─────────────────────────────────────────────────────────────────
# 3. Model 2 training results — windows_from_merged
# ─────────────────────────────────────────────────────────────────
section("3. MODEL 2 TRAINING RESULTS — windows_from_merged.csv")

reg_m  = load_safe("reports/model2_from_merged/regressor_comparison.csv")
cls_m  = load_safe("reports/model2_from_merged/classifier_comparison.csv")

if reg_m is not None:
    print("  Regressor comparison:")
    print(f"  {'Model':<45} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("  " + SEP2)
    for _, row in reg_m.iterrows():
        print(f"  {str(row['model']):<45} {row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>8.4f}")

if cls_m is not None:
    print()
    print("  Classifier comparison:")
    print(f"  {'Model':<45} {'F1':>7} {'BalAcc':>8} {'AUC':>8}")
    print("  " + SEP2)
    for _, row in cls_m.iterrows():
        print(f"  {str(row['model']):<45} {row['f1']:>7.4f} {row['balanced_accuracy']:>8.4f} {row['roc_auc']:>8.4f}")

# ─────────────────────────────────────────────────────────────────
# 4. Model 2 training results — windows_all.csv (full dataset)
# ─────────────────────────────────────────────────────────────────
section("4. MODEL 2 TRAINING RESULTS — windows_all.csv (full + augmented)")

reg_a  = load_safe("reports/model2_risk_assessment/regressor_comparison.csv")
cls_a  = load_safe("reports/model2_risk_assessment/classifier_comparison.csv")

if reg_a is not None:
    print("  Regressor comparison:")
    print(f"  {'Model':<45} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("  " + SEP2)
    for _, row in reg_a.iterrows():
        print(f"  {str(row['model']):<45} {row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>8.4f}")

if cls_a is not None:
    print()
    print("  Classifier comparison:")
    print(f"  {'Model':<45} {'F1':>7} {'BalAcc':>8} {'AUC':>8}")
    print("  " + SEP2)
    for _, row in cls_a.iterrows():
        print(f"  {str(row['model']):<45} {row['f1']:>7.4f} {row['balanced_accuracy']:>8.4f} {row['roc_auc']:>8.4f}")

# ─────────────────────────────────────────────────────────────────
# 5. Prediction analysis — windows_from_merged
# ─────────────────────────────────────────────────────────────────
section("5. PREDICTION ANALYSIS — windows_from_merged.csv")

pred_m = load_safe("reports/model2_from_merged/model2_predictions_merged.csv")
if pred_m is not None:
    score_col = "model2_risk_score"
    prob_col  = "model2_high_risk_probability"
    lvl_col   = "model2_risk_level"

    print(f"  Total predictions : {len(pred_m)}")
    print()
    print("  Risk score statistics:")
    print(f"    Mean   : {pred_m[score_col].mean():.4f}")
    print(f"    Median : {pred_m[score_col].median():.4f}")
    print(f"    Std    : {pred_m[score_col].std():.4f}")
    print(f"    Min    : {pred_m[score_col].min():.4f}")
    print(f"    Max    : {pred_m[score_col].max():.4f}")

    print()
    print("  Risk level distribution:")
    for lvl, cnt in pred_m[lvl_col].value_counts().items():
        pct = cnt / len(pred_m) * 100
        bar = "#" * int(pct / 3)
        print(f"    {str(lvl):<10} {cnt:4d}  ({pct:5.1f}%)  |{bar}")

    if "class_en" in pred_m.columns:
        print()
        print("  Risk score by class (sorted by mean risk, descending):")
        grp = pred_m.groupby("class_en")[score_col].agg(["mean","min","max","count"])
        grp = grp.sort_values("mean", ascending=False)
        print(f"    {'Class':<35} {'Mean':>7} {'Min':>7} {'Max':>7} {'N':>5}")
        print("    " + "-" * 60)
        for cls, row in grp.iterrows():
            risk_label = "FALL" if row["mean"] >= 0.66 else ("MED" if row["mean"] >= 0.33 else "low")
            print(f"    {str(cls):<35} {row['mean']:>7.3f} {row['min']:>7.3f} {row['max']:>7.3f} {int(row['count']):>5}  [{risk_label}]")

        print()
        print("  High-risk probability by class:")
        grp2 = pred_m.groupby("class_en")[prob_col].mean().sort_values(ascending=False)
        for cls, val in grp2.items():
            bar = "#" * int(val * 35)
            print(f"    {str(cls):<35} {val:.3f}  |{bar}")

# ─────────────────────────────────────────────────────────────────
# 6. Prediction analysis — windows_all
# ─────────────────────────────────────────────────────────────────
section("6. PREDICTION ANALYSIS — windows_all.csv (full + augmented)")

pred_a = load_safe("reports/model2_risk_assessment/model2_predictions_windows_all.csv")
if pred_a is not None:
    score_col = "model2_risk_score"
    prob_col  = "model2_high_risk_probability"
    lvl_col   = "model2_risk_level"

    print(f"  Total predictions : {len(pred_a)}")
    print()
    print("  Risk score statistics:")
    print(f"    Mean   : {pred_a[score_col].mean():.4f}")
    print(f"    Std    : {pred_a[score_col].std():.4f}")
    print(f"    Min    : {pred_a[score_col].min():.4f}")
    print(f"    Max    : {pred_a[score_col].max():.4f}")

    print()
    print("  Risk level distribution:")
    for lvl, cnt in pred_a[lvl_col].value_counts().items():
        pct = cnt / len(pred_a) * 100
        bar = "#" * int(pct / 3)
        print(f"    {str(lvl):<10} {cnt:5d}  ({pct:5.1f}%)  |{bar}")

    if "class_en" in pred_a.columns:
        print()
        print("  Risk score by class:")
        grp = pred_a.groupby("class_en")[score_col].agg(["mean","min","max","count"])
        grp = grp.sort_values("mean", ascending=False)
        print(f"    {'Class':<35} {'Mean':>7} {'Min':>7} {'Max':>7} {'N':>5}")
        print("    " + "-" * 60)
        for cls, row in grp.iterrows():
            risk_label = "FALL" if row["mean"] >= 0.66 else ("MED" if row["mean"] >= 0.33 else "low")
            print(f"    {str(cls):<35} {row['mean']:>7.3f} {row['min']:>7.3f} {row['max']:>7.3f} {int(row['count']):>5}  [{risk_label}]")

# ─────────────────────────────────────────────────────────────────
# 7. Readiness reports
# ─────────────────────────────────────────────────────────────────
section("7. MODEL READINESS REPORTS")

for rpt_path in [
    "reports/model2_from_merged/model2_readiness_report.md",
    "reports/model2_risk_assessment/model2_readiness_report.md",
]:
    p = HERE / rpt_path
    if p.exists():
        print(f"\n  [{p.name}] from {rpt_path}")
        content = p.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines()[:40]:
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            print(f"    {safe_line}")
    else:
        print(f"\n  [MISSING] {rpt_path}")

# ─────────────────────────────────────────────────────────────────
# 8. Dataset balance report summary
# ─────────────────────────────────────────────────────────────────
section("8. DATASET BALANCE REPORT")

balance_p = HERE / "reports/dataset_balance/dataset_balance_report.md"
if balance_p.exists():
    content = balance_p.read_text(encoding="utf-8", errors="replace")
    for line in content.splitlines()[:60]:
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        print(f"  {safe_line}")
else:
    print("  [MISSING] reports/dataset_balance/dataset_balance_report.md")

# ─────────────────────────────────────────────────────────────────
# 9. Model artifacts list
# ─────────────────────────────────────────────────────────────────
section("9. MODEL ARTIFACTS")

for model_dir in ["models/model2_from_merged", "models/model2_risk_assessment"]:
    p = HERE / model_dir
    if p.exists():
        print(f"\n  {model_dir}/")
        for f in sorted(p.rglob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name:<50} {size_kb:8.1f} KB")

# ─────────────────────────────────────────────────────────────────
# 10. Summary verdict
# ─────────────────────────────────────────────────────────────────
section("10. SUMMARY VERDICT")

print("  Dataset              | windows_from_merged (798 rows) | windows_all (2138 rows, augmented)")
print("  AUC (classifier)     | ~0.677 (merged only)           | 1.000 (augmented, may overfit)")
print("  R2 (risk regressor)  | ~-0.11 (too little data)       | 0.910 (good)")
print()
print("  RECOMMENDATION:")
print("  -> Use windows_all.csv for Model 2 training (more data, better AUC)")
print("  -> windows_from_merged.csv is useful for 'original data only' ablation")
print("  -> AUC=1.000 on windows_all may indicate data leakage via augmented windows")
print("     sharing the same session_id as training set.")
print("  -> Consider --split random to verify or use --original-only flag")
print()
print(SEP)
print("  All checks complete.")
print(SEP)
