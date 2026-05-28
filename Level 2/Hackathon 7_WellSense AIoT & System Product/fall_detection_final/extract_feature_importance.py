"""extract_feature_importance.py - ดึง Feature Importance จาก Model 2"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models" / "model2_combined"

bundle   = joblib.load(MODEL_DIR / "model2_risk_bundle.joblib")
feat_cols = bundle["feature_columns"]

print("=" * 68)
print("  Model 2 — Feature Importance Extraction")
print("=" * 68)
print(f"  Best Regressor : {bundle['best_regressor_name']}")
print(f"  Best Classifier: {bundle['best_classifier_name']}")
print(f"  Feature columns: {len(feat_cols)}")
print(f"  Final fit rows : {bundle.get('final_fit_rows','?')}")

# ── Metrics ───────────────────────────────────────────────────
reg_metrics = bundle.get("regressor_metrics", {})
clf_metrics = bundle.get("classifier_metrics", {})
print("\n  Regressor Metrics:")
if isinstance(reg_metrics, dict):
    for k, v in reg_metrics.items():
        print(f"    {k}: {v}")
elif isinstance(reg_metrics, list):
    for item in reg_metrics:
        print(f"    {item}")

print("\n  Classifier Metrics:")
if isinstance(clf_metrics, dict):
    for k, v in clf_metrics.items():
        print(f"    {k}: {v}")
elif isinstance(clf_metrics, list):
    for item in clf_metrics:
        print(f"    {item}")

# ── RF Classifier Feature Importance ─────────────────────────
print("\n" + "=" * 68)
print("  Feature Importance: Random Forest Classifier")
print("=" * 68)
rf_clf = joblib.load(MODEL_DIR / "random_forest_classifier.joblib")
try:
    if hasattr(rf_clf, 'named_steps'):
        est = list(rf_clf.named_steps.values())[-1]
    else:
        est = rf_clf
    imp = est.feature_importances_
    rf_imp = pd.Series(imp, index=feat_cols[:len(imp)]).sort_values(ascending=False)
    print(f"\n  {'Rank':<5} {'Feature':<26} {'Importance':>11}  {'Bar'}")
    print("  " + "-" * 65)
    for rank, (feat, val) in enumerate(rf_imp.items(), 1):
        bar = "#" * int(val * 80)
        print(f"  {rank:<5} {feat:<26} {val:>11.5f}  [{bar}]")
    # Export top 20 as JSON
    top20 = rf_imp.head(20).to_dict()
    with open(HERE / "reports" / "feature_importance_rf.json", "w") as f:
        json.dump({"method": "rf_classifier", "importances": top20}, f, indent=2)
    print("\n  -> Saved: reports/feature_importance_rf.json")
except Exception as e:
    print(f"  Error: {e}")

# ── LR Classifier Coefficients ────────────────────────────────
print("\n" + "=" * 68)
print("  Feature Coefficients: Logistic Regression (Best Classifier)")
print("=" * 68)
lr_clf = joblib.load(MODEL_DIR / "logistic_regression_classifier.joblib")
try:
    if hasattr(lr_clf, 'named_steps'):
        est = list(lr_clf.named_steps.values())[-1]
    else:
        est = lr_clf
    coef = np.abs(est.coef_[0])
    lr_imp = pd.Series(coef, index=feat_cols[:len(coef)]).sort_values(ascending=False)
    print(f"\n  {'Rank':<5} {'Feature':<26} {'|Coef|':>11}  {'Bar'}")
    print("  " + "-" * 65)
    for rank, (feat, val) in enumerate(lr_imp.items(), 1):
        bar = "#" * int(val * 5)
        print(f"  {rank:<5} {feat:<26} {val:>11.4f}  [{bar}]")
    top20_lr = lr_imp.head(20).to_dict()
    with open(HERE / "reports" / "feature_importance_lr.json", "w") as f:
        json.dump({"method": "lr_abs_coefficient", "importances": top20_lr}, f, indent=2)
    print("\n  -> Saved: reports/feature_importance_lr.json")
except Exception as e:
    print(f"  LR Error: {e}")

# ── RF Regressor Feature Importance ──────────────────────────
print("\n" + "=" * 68)
print("  Feature Importance: Random Forest Regressor (Risk Score)")
print("=" * 68)
rf_reg = joblib.load(MODEL_DIR / "random_forest_regressor.joblib")
try:
    if hasattr(rf_reg, 'named_steps'):
        est = list(rf_reg.named_steps.values())[-1]
    else:
        est = rf_reg
    imp = est.feature_importances_
    rfr_imp = pd.Series(imp, index=feat_cols[:len(imp)]).sort_values(ascending=False)
    print(f"\n  {'Rank':<5} {'Feature':<26} {'Importance':>11}  {'Bar'}")
    print("  " + "-" * 65)
    for rank, (feat, val) in enumerate(rfr_imp.items(), 1):
        bar = "#" * int(val * 80)
        print(f"  {rank:<5} {feat:<26} {val:>11.5f}  [{bar}]")
except Exception as e:
    print(f"  Regressor Error: {e}")

print("\n" + "=" * 68)
print("  Done!")
print("=" * 68)
