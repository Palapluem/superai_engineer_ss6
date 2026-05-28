"""
location_imu_zone.py
────────────────────────────────────────────────────────────────
MODULE B: IMU Activity-Zone Classifier (Fallback Method)

ทำงาน:
  - ใช้ IMU features จาก windows_combined.csv ที่มีอยู่แล้ว
  - Map กิจกรรม (class_en) → Location Zone โดยใช้ Rule-based + ML
  - ใช้เมื่อไม่มี GPS signal (indoor, GPS loss)

Logic การ Map:
  - lying_down               → bedroom (นอนในห้องนอน)
  - standing                 → living_room / bedroom
  - normal_walk              → corridor / living_room
  - limping_walk             → corridor / outdoor (เดินไม่สมดุล มักอยู่นอกห้อง)
  - elderly_pick_up_object   → kitchen / living_room
  - stand_sit_alternating    → living_room (นั่ง-ลุก บนโซฟา/เก้าอี้)
  - corrected_walking        → corridor
  - fall classes             → any (บันทึก zone ก่อนล้ม)

ทดสอบ:
  python location_imu_zone.py --data windows_combined.csv --test

Train + evaluate:
  python location_imu_zone.py --data windows_combined.csv --train
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Activity → Zone Mapping (Rule-based prior) ──────────────
# คือ "ถ้าทำกิจกรรมนี้ → น่าจะอยู่ใน zone นี้"
ACTIVITY_ZONE_RULES = {
    # activity class          → (zone_id, zone_name, confidence, risk_note)
    "lying_down":             ("bedroom",     "ห้องนอน",        0.85, "นอน = ห้องนอน"),
    "standing":               ("living_room", "ห้องนั่งเล่น",   0.60, "ยืนอาจอยู่หลายที่"),
    "normal_walk":            ("corridor",    "ทางเดิน/โถง",    0.70, "เดินปกติ = ทางเดิน"),
    "corrected_walking":      ("corridor",    "ทางเดิน/โถง",    0.75, "เดินแก้ไข = ทางเดิน"),
    "limping_walk":           ("outdoor",     "นอกบ้าน",        0.55, "เดินกะเผลก อาจอยู่ outdoor"),
    "elderly_pick_up_object": ("kitchen",     "ห้องครัว",       0.60, "หยิบของ = ห้องครัว/ห้องนั่งเล่น"),
    "stand_sit_alternating":  ("living_room", "ห้องนั่งเล่น",   0.80, "นั่ง-ลุก = โซฟา/เก้าอี้"),
    # fall classes → ไม่ระบุ zone แน่นอน (อาจล้มที่ไหนก็ได้)
    "slow_collapse_fall":     ("high_risk_area", "พื้นที่เสี่ยง", 0.40, "ล้ม = ไม่แน่ zone"),
    "gradual_fall":           ("high_risk_area", "พื้นที่เสี่ยง", 0.40, "ล้ม = ไม่แน่ zone"),
    "sideways_fall":          ("high_risk_area", "พื้นที่เสี่ยง", 0.35, "ล้มข้าง"),
    "backward_fall":          ("high_risk_area", "พื้นที่เสี่ยง", 0.35, "ล้มหลัง"),
    "collapse_fall":          ("high_risk_area", "พื้นที่เสี่ยง", 0.35, "ล้มทรุด"),
}

# ─── Zone Risk Level ─────────────────────────────────────────
ZONE_RISK = {
    "bedroom":        "medium",
    "bathroom":       "high",
    "kitchen":        "medium",
    "living_room":    "low",
    "corridor":       "low",
    "staircase":      "high",
    "outdoor":        "low",
    "high_risk_area": "high",
    "unknown":        "low",
}


# ─── IMU Features สำหรับ ML Classifier ─────────────────────
IMU_FEATURE_COLS = [
    "svm_mean", "svm_std", "svm_max", "svm_min", "svm_dev_mean",
    "jerk_mean", "jerk_std", "jerk_max",
    "KII_mean", "KII_std", "KII_max",
    "omega_mean", "omega_std", "omega_max",
    "theta_mean", "theta_std", "theta_max", "theta_range",
    "free_fall_n", "impact_n", "high_rot_n",
    "GSI", "fcri", "angular_impulse",
    "press_delta", "press_slope",
    "mic_p2p_max", "mic_p2p_mean",
]


# ─── Rule-based Classifier (No ML needed) ───────────────────
class RuleBasedZoneClassifier:
    """
    ใช้ class_en ของกิจกรรม → ระบุ zone โดยตรง
    เป็น baseline สำหรับเปรียบเทียบกับ ML version
    """

    def classify(self, class_en: str) -> dict:
        rule = ACTIVITY_ZONE_RULES.get(class_en)
        if rule:
            zone_id, zone_name, conf, note = rule
        else:
            zone_id, zone_name, conf, note = "unknown", "ไม่ระบุ", 0.3, "ไม่รู้จักกิจกรรม"

        return {
            "zone_id":    zone_id,
            "zone_name":  zone_name,
            "confidence": conf,
            "risk_level": ZONE_RISK.get(zone_id, "low"),
            "source":     "imu_rule",
            "note":       note,
        }


# ─── ML Zone Classifier ───────────────────────────────────────
class MLZoneClassifier:
    """
    Train Random Forest บน IMU features → predict zone
    ใช้เมื่อต้องการ fine-grained prediction มากกว่า rule-based
    """

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = IMU_FEATURE_COLS

    def _assign_zone_label(self, class_en: str) -> str:
        """Map class_en → zone_id สำหรับ training label"""
        rule = ACTIVITY_ZONE_RULES.get(class_en)
        if rule:
            return rule[0]
        return "unknown"

    def train(self, df: pd.DataFrame) -> dict:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import classification_report
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        print("\n[IMU Zone ML] Training zone classifier...")

        # สร้าง zone label จาก class_en
        df = df.copy()
        df["zone_label"] = df["class_en"].apply(self._assign_zone_label)
        print(f"  Zone distribution:")
        for zone, cnt in df["zone_label"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {zone:<20} {cnt:5d} ({pct:.1f}%)")

        # กรอง features
        available_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[available_cols].fillna(0).values
        y = df["zone_label"].values

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.label_encoder = le

        # Train pipeline
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ])

        # Cross-validation
        print("\n[IMU Zone ML] Cross-validating (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipe, X, y_enc, cv=cv,
            scoring=["accuracy", "balanced_accuracy", "f1_weighted"],
            return_train_score=True,
        )

        metrics = {
            "cv_accuracy_mean":     float(cv_results["test_accuracy"].mean()),
            "cv_accuracy_std":      float(cv_results["test_accuracy"].std()),
            "cv_balanced_acc_mean": float(cv_results["test_balanced_accuracy"].mean()),
            "cv_f1_weighted_mean":  float(cv_results["test_f1_weighted"].mean()),
        }

        print(f"\n  CV Accuracy        : {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"  CV Balanced Acc    : {metrics['cv_balanced_acc_mean']:.4f}")
        print(f"  CV F1 Weighted     : {metrics['cv_f1_weighted_mean']:.4f}")

        # Train final model on all data
        pipe.fit(X, y_enc)
        self.model = pipe
        self.feature_cols = available_cols

        # Feature importance
        rf = pipe.named_steps["clf"]
        importances = pd.Series(rf.feature_importances_, index=available_cols).sort_values(ascending=False)
        print(f"\n  Top 10 Important Features:")
        for feat, imp in importances.head(10).items():
            bar = "█" * int(imp * 80)
            print(f"    {feat:<25} {imp:.4f} {bar}")

        # Classification report (train set)
        y_pred_enc = pipe.predict(X)
        y_pred = le.inverse_transform(y_pred_enc)
        print(f"\n  Classification Report (Train Set):")
        report = classification_report(y, y_pred, zero_division=0)
        for line in report.split("\n"):
            print(f"    {line}")

        metrics["classes"] = list(le.classes_)
        metrics["n_features"] = len(available_cols)
        metrics["n_samples"] = len(df)
        return metrics

    def predict(self, features: dict) -> dict:
        """
        รับ dict ของ IMU features 1 window → คืน zone prediction
        """
        if self.model is None:
            raise RuntimeError("Model ยังไม่ได้ train! เรียก .train() ก่อน")

        X = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
        zone_enc = self.model.predict(X)[0]
        zone_proba = self.model.predict_proba(X)[0]
        confidence = float(zone_proba.max())
        zone_id = self.label_encoder.inverse_transform([zone_enc])[0]

        return {
            "zone_id":    zone_id,
            "zone_name":  ACTIVITY_ZONE_RULES.get(zone_id, ("?", zone_id))[1] if zone_id in ACTIVITY_ZONE_RULES else zone_id,
            "confidence": round(confidence, 3),
            "risk_level": ZONE_RISK.get(zone_id, "low"),
            "source":     "imu_ml",
        }


# ─── Benchmark: Rule-based vs ML ─────────────────────────────
def run_benchmark(df: pd.DataFrame):
    print("\n" + "="*68)
    print("  MODULE B: IMU Zone Classifier — Benchmark")
    print("="*68)

    # ── Rule-based ────────────────────────────────────────────
    rule_clf = RuleBasedZoneClassifier()

    df = df.copy()
    df["zone_true"]      = df["class_en"].apply(lambda c: ACTIVITY_ZONE_RULES.get(c, ("unknown",))[0])
    df["zone_rule_pred"] = df["class_en"].apply(lambda c: rule_clf.classify(c)["zone_id"])
    df["zone_rule_conf"] = df["class_en"].apply(lambda c: rule_clf.classify(c)["confidence"])

    rule_acc = (df["zone_true"] == df["zone_rule_pred"]).mean()

    print(f"\n[Rule-based] Accuracy = {rule_acc:.4f} ({rule_acc*100:.1f}%)")
    print(f"  (Rule-based = 100% by definition เพราะ map ตรง)")

    # ── Rule-based Zone Distribution ─────────────────────────
    print(f"\n  Zone distribution ที่ predict ได้:")
    zone_dist = df.groupby(["zone_true", "class_en"]).size().reset_index(name="count")
    for zone, grp in zone_dist.groupby("zone_true"):
        print(f"\n  🏠 {zone}:")
        for _, row in grp.iterrows():
            print(f"      {row['class_en']:<35} {row['count']:5d} windows")

    # Mean confidence per zone
    print(f"\n  Mean confidence per zone (Rule-based):")
    for zone, grp in df.groupby("zone_true"):
        mean_conf = grp["zone_rule_conf"].mean()
        bar = "█" * int(mean_conf * 20)
        print(f"    {zone:<20} conf={mean_conf:.3f} {bar}")

    # ── ML Classifier ─────────────────────────────────────────
    print("\n" + "-"*68)
    ml_clf = MLZoneClassifier()
    ml_metrics = ml_clf.train(df)

    return {
        "rule_based": {
            "accuracy": rule_acc,
            "note": "Deterministic mapping class→zone",
        },
        "ml_classifier": ml_metrics,
    }


# ─── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IMU Zone Classifier")
    parser.add_argument("--data",  type=Path, default=Path("windows_combined.csv"))
    parser.add_argument("--train", action="store_true", help="Train ML classifier")
    parser.add_argument("--test",  action="store_true", help="Run full benchmark")
    args = parser.parse_args()

    data_path = args.data
    if not data_path.exists():
        data_path = Path(__file__).parent / args.data
    if not data_path.exists():
        print(f"[ERROR] ไม่พบ {args.data}")
        return

    print(f"[IMU Zone] Loading {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  {len(df):,} rows | {df['class_en'].nunique()} classes")

    if args.test or args.train or True:  # รัน benchmark เสมอ
        results = run_benchmark(df)
        print("\n" + "="*68)
        print("  BENCHMARK SUMMARY")
        print("="*68)
        print(f"  Rule-based: ความแม่น {results['rule_based']['accuracy']*100:.0f}% (deterministic)")
        if "cv_accuracy_mean" in results["ml_classifier"]:
            ml_acc = results["ml_classifier"]["cv_accuracy_mean"]
            ml_f1  = results["ml_classifier"]["cv_f1_weighted_mean"]
            print(f"  ML (RF)   : CV Accuracy = {ml_acc:.4f} | F1 = {ml_f1:.4f}")
        print()
        print("  ข้อสรุป:")
        print("  • Rule-based → เหมาะกับ demo / prototype (ไม่ต้อง train)")
        print("  • ML         → แม่นกว่าเมื่อมีข้อมูลเพิ่ม และจับ feature pattern ได้")
        print("  • GPS        → แม่นที่สุด แต่ต้องมี signal")
        print()


if __name__ == "__main__":
    main()
