"""
wellsense_serial_sender.py
──────────────────────────────────────────────────────────────
Host PC script: โหลด Model 2, อ่าน live features, แล้วส่ง
risk_level,score,impact ผ่าน Serial ไปยัง Arduino UNO Q

ใช้งาน:
  python wellsense_serial_sender.py --port COM3 --input live_window.csv

หรือทดสอบ simulate โดยไม่มีบอร์ด:
  python wellsense_serial_sender.py --simulate
"""

import argparse
import time
import sys
import os
import csv
import json
from pathlib import Path

# ─── Optional imports ─────────────────────────────────────
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[WARNING] pyserial ไม่ได้ติดตั้ง — ทำงานได้เฉพาะ --simulate mode")
    print("          ติดตั้งด้วย: pip install pyserial\n")

try:
    import joblib
    import pandas as pd
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("[WARNING] joblib/pandas/numpy ไม่ได้ติดตั้ง — ไม่สามารถโหลดโมเดลได้")

# ─── Constants ────────────────────────────────────────────
HERE         = Path(__file__).parent
MODEL_DIR    = HERE / "fall_detection_final" / "models" / "model2_combined"
BUNDLE_PATH  = MODEL_DIR / "model2_risk_bundle.joblib"
FEATURES_JSON = MODEL_DIR / "feature_columns.json"

# ─── Risk Level thresholds ────────────────────────────────
THRESHOLD_HIGH   = 0.65   # risk_score ≥ 0.65 → high
THRESHOLD_MEDIUM = 0.35   # risk_score ≥ 0.35 → medium (ต่ำกว่า = low)
IMPACT_THRESHOLD = 0.80   # impact_event ≥ 0.80 → trigger buzzer

# ─── Load Model ───────────────────────────────────────────
def load_model(model_path: Path):
    if not HAS_ML:
        raise RuntimeError("ต้องติดตั้ง joblib, pandas, numpy ก่อน")
    if not model_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path}")
    bundle = joblib.load(model_path)
    print(f"[Model] โหลดสำเร็จ: {model_path.name}")
    return bundle

# ─── Predict Single Window ────────────────────────────────
def predict_window(bundle: dict, feature_row: dict) -> dict:
    """
    รับ dict ของ features 1 window → คืน dict ผลลัพธ์

    Returns:
        {
          "risk_level": "low" | "medium" | "high",
          "risk_score": float,
          "high_risk_prob": float,
          "impact_event": float
        }
    """
    # โหลดชื่อ features
    feature_cols = bundle.get("feature_columns", list(feature_row.keys()))
    clf          = bundle.get("high_risk_classifier")
    reg          = bundle.get("risk_regressor")

    # สร้าง feature vector
    X = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in feature_cols}])

    # Predict
    risk_score      = float(reg.predict(X)[0]) if reg is not None else 0.5
    high_risk_prob  = float(clf.predict_proba(X)[0][1]) if clf is not None else 0.5

    # Determine level
    if risk_score >= THRESHOLD_HIGH:
        risk_level = "high"
    elif risk_score >= THRESHOLD_MEDIUM:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Impact event score (ใช้ feature ถ้ามี)
    impact_event = float(feature_row.get("impact_n", 0))
    # Normalize impact_n (ปกติ 0–5) → 0.0–1.0
    impact_score = min(impact_event / 5.0, 1.0)

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 4),
        "high_risk_prob": round(high_risk_prob, 4),
        "impact_event": round(impact_score, 4),
    }

# ─── Format Serial Message ────────────────────────────────
def format_serial_msg(result: dict) -> str:
    """สร้าง message ตาม protocol: 'risk_level,risk_score,impact_event\n'"""
    return f"{result['risk_level']},{result['risk_score']},{result['impact_event']}\n"

# ─── Send to Arduino ─────────────────────────────────────
def send_to_arduino(ser, message: str):
    """ส่ง message ผ่าน Serial port"""
    ser.write(message.encode("ascii"))
    print(f"[Serial → UNO Q] {message.strip()}")

# ─── Simulate Mode ────────────────────────────────────────
def run_simulate_mode():
    """
    ทดสอบโดยไม่มีบอร์ด — แสดงผลเป็น text เท่านั้น
    ส่ง test sequence: low → medium → high → high+impact
    """
    print("\n" + "="*60)
    print("  WellSense LED Traffic Light — SIMULATE MODE")
    print("  (ไม่มีบอร์ด Arduino — แสดงผลที่จะส่งเท่านั้น)")
    print("="*60 + "\n")

    test_cases = [
        {"risk_level": "low",    "risk_score": 0.12, "impact_event": 0.05},
        {"risk_level": "low",    "risk_score": 0.28, "impact_event": 0.02},
        {"risk_level": "medium", "risk_score": 0.45, "impact_event": 0.10},
        {"risk_level": "medium", "risk_score": 0.58, "impact_event": 0.15},
        {"risk_level": "high",   "risk_score": 0.72, "impact_event": 0.30},
        {"risk_level": "high",   "risk_score": 0.88, "impact_event": 0.50},
        {"risk_level": "high",   "risk_score": 0.91, "impact_event": 0.95},  # ← impact!
        {"risk_level": "low",    "risk_score": 0.18, "impact_event": 0.03},  # กลับสู่ปกติ
    ]

    LED_DISPLAY = {
        "low":    "🟢 GREEN  (ติดค้าง)",
        "medium": "🟡 YELLOW (กะพริบช้า 1 Hz)",
        "high":   "🔴 RED    (กะพริบเร็ว 4 Hz)",
    }

    for i, result in enumerate(test_cases, 1):
        msg = format_serial_msg(result)
        led = LED_DISPLAY[result["risk_level"]]
        impact_flag = " ⚠️ IMPACT OVERRIDE + BUZZER!" if result["impact_event"] >= 0.8 else ""
        print(f"[Step {i:02d}] Serial: {msg.strip():35s} → LED: {led}{impact_flag}")
        time.sleep(1.5)

    print("\n[Simulate] เสร็จสิ้น — ทดสอบครบทุก state แล้ว\n")

# ─── Live Mode (อ่าน CSV + ส่ง Serial) ──────────────────
def run_live_mode(port: str, model_path: Path, input_csv: Path, baud: int = 9600):
    """
    อ่าน features จาก CSV ทีละแถว แล้วส่งผ่าน Serial ไป UNO Q
    (ใช้สำหรับ demo / replay data)
    """
    if not HAS_SERIAL:
        print("[ERROR] ต้องติดตั้ง pyserial ก่อน: pip install pyserial")
        sys.exit(1)

    if not HAS_ML:
        print("[ERROR] ต้องติดตั้ง joblib, pandas, numpy ก่อน")
        sys.exit(1)

    bundle = load_model(model_path)

    print(f"[Serial] เปิด port {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # รอ Arduino reset
        print(f"[Serial] เชื่อมต่อ {port} สำเร็จ\n")
    except serial.SerialException as e:
        print(f"[ERROR] ไม่สามารถเปิด {port}: {e}")
        sys.exit(1)

    print(f"[CSV] อ่านข้อมูลจาก: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[CSV] พบ {len(df)} windows — เริ่มส่งข้อมูล...\n")

    for idx, row in df.iterrows():
        feature_row = row.to_dict()
        result = predict_window(bundle, feature_row)
        msg = format_serial_msg(result)
        send_to_arduino(ser, msg)

        # อ่านผลจาก Arduino กลับมา (optional)
        if ser.in_waiting:
            arduino_reply = ser.readline().decode("ascii", errors="replace").strip()
            print(f"[UNO Q → PC] {arduino_reply}")

        time.sleep(0.5)  # ส่งทุก 0.5 วินาที (ตาม window step)

    ser.close()
    print("\n[Done] ส่งข้อมูลครบทุก window แล้ว")

# ─── Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="WellSense Serial Sender — ส่ง risk_level ไปยัง Arduino UNO Q"
    )
    parser.add_argument("--port",     type=str, default="COM3",
                        help="Serial port เช่น COM3 (Windows) หรือ /dev/ttyACM0 (Linux)")
    parser.add_argument("--baud",     type=int, default=9600,
                        help="Baud rate (ต้องตรงกับ Serial.begin() ใน Arduino)")
    parser.add_argument("--model",    type=Path, default=BUNDLE_PATH,
                        help="Path ไปยัง model2_risk_bundle.joblib")
    parser.add_argument("--input",    type=Path, default=None,
                        help="CSV ไฟล์ที่มี windowed features (ถ้าไม่ระบุ = simulate)")
    parser.add_argument("--simulate", action="store_true",
                        help="รันแบบ simulate (ไม่ต้องมีบอร์ด Arduino)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  WellSense AIoT — Serial Sender v1.0")
    print("  Hackathon 7 | SPAI11")
    print("="*60)

    if args.simulate or args.input is None:
        run_simulate_mode()
    else:
        run_live_mode(
            port       = args.port,
            model_path = args.model,
            input_csv  = args.input,
            baud       = args.baud,
        )

if __name__ == "__main__":
    main()
