# 🩺 Model 2 — WellSense Mobility Risk Assessment

### สรุปผลสำหรับทีม / Integration Guide สำหรับ UNO Q Board

---

## 1. บทบาทของ Model 2 ในระบบ WellSense

```
[Arduino Nano 33 BLE Sense]
   IMU: ax, ay, az, gx, gy, gz (20 Hz)
   Env: press, mic
           |
           v  (Sliding Window 2 sec, Step 0.5 sec)
[Feature Extraction: build_features_from_raw.py]
   svm_mean, jerk_mean, KII_mean, omega_mean, ...
           |
           v
[Model 1]                     [Model 2]
Binary: Fall / No Fall    Continuous: Risk Score 0.0–1.0
                              + Risk Level: low / medium / high
                              + High-Risk Probability
           |                        |
           v                        v
      [Dashboard / Alert]    [UNO Q Board / Heatmap]
```

> **Model 2 ≠ Model 1**  
> Model 1 ตรวจว่า "ล้มหรือไม่ล้ม" (binary)  
> Model 2 ประเมินว่า "เสี่ยงแค่ไหน" (continuous 0.0–1.0) — เหมาะกับ heatmap / dashboard risk visualization

---

## 2. ผลการ Training (สรุปสั้น)

### Dataset ที่ใช้ Train: `windows_combined.csv`

| รายการ | ค่า |
|---|---|
| จำนวน windows | **2,598 windows** |
| จำนวน class | **11 classes** (ครบทุกประเภทกิจกรรม) |
| Train rows | 2,110 |
| Test rows | 488 |

### ผล Model (Best Models)

| Model | Metric | ค่า |
|---|---|---|
| **Risk Classifier** (Logistic Regression) | **AUC** | **0.965** 🟢 |
| Risk Classifier | Balanced Accuracy | 0.894 |
| Risk Classifier | F1 Score | 0.902 |
| Risk Classifier | Precision | 0.864 |
| Risk Classifier | Recall | 0.944 |
| Risk Regressor (Random Forest) | MAE | 0.157 |
| Risk Regressor | R² | 0.328 |

> **ข้อสังเกต**: AUC = 0.965 หมายความว่าโมเดลแยก high-risk vs low-risk ได้ถูกต้อง 96.5% ของเวลา

### Risk Score ต่อ Class (Mean ± Range)

| Class | Mean Risk | Level | ทดสอบจริง? |
|---|---|---|---|
| `sideways_fall` | ~0.87 | 🔴 HIGH | ✅ บนบอร์ด |
| `backward_fall` | ~0.87 | 🔴 HIGH | ✅ บนบอร์ด |
| `slow_collapse_fall` | ~0.76 | 🔴 HIGH | ✅ บนบอร์ด |
| `gradual_fall` | ~0.77 | 🔴 HIGH | ✅ บนบอร์ด |
| `stand_sit_alternating` | ~0.51 | 🟡 MEDIUM | ✅ บนบอร์ด |
| `limping_walk` | ~0.46 | 🟡 MEDIUM | ✅ บนบอร์ด |
| `elderly_pick_up_object` | ~0.38 | 🟡 MEDIUM | ✅ บนบอร์ด |
| `normal_walk` | ~0.33 | 🟡 MEDIUM | ✅ บนบอร์ด |
| `corrected_walking` | ~0.29 | 🟢 LOW | ✅ บนบอร์ด |
| `lying_down` | ~0.16 | 🟢 LOW | ✅ บนบอร์ด |
| `standing` | ~0.11 | 🟢 LOW | ✅ บนบอร์ด |

---

## 3. Model Output ที่ส่งไป Dashboard / UNO Q

เมื่อรัน prediction แต่ละ window จะได้ผลลัพธ์ดังนี้:

```json
{
  "window_start_ts": "2026-05-28T00:02:52",
  "window_end_ts":   "2026-05-28T00:02:54",
  "session_id":      "slow_collapse_fall_orig",
  "class_en":        "slow_collapse_fall",
  "model2_risk_score":        0.789,
  "model2_high_risk_probability": 0.975,
  "model2_risk_level":        "high",
  "rule_feature_risk_score":  0.832,
  "component_gait_motion":    0.971,
  "component_rotation_balance": 0.773,
  "component_posture_transition": 0.971,
  "component_impact_event":   1.000,
  "component_physio_stress":  0.149
}
```

### Field ที่ UNO Q น่าจะต้องการ

| Field | ค่า | ใช้งาน |
|---|---|---|
| `model2_risk_score` | 0.0 – 1.0 | แสดง gauge / heatmap intensity |
| `model2_risk_level` | `low` / `medium` / `high` | แสดงสี LED / alert |
| `model2_high_risk_probability` | 0.0 – 1.0 | trigger alert threshold |
| `component_impact_event` | 0.0 – 1.0 | detect ช่วงที่กระแทก |

---

## 4. Input Features ที่ต้องส่งให้โมเดล (46 features)

> Model 2 รับ **windowed features** ไม่ใช่ raw sensor โดยตรง  
> ต้องผ่าน Feature Extraction ก่อนเสมอ

### IMU Features (บังคับ — ต้องมี)

```
svm_mean, svm_std, svm_max, svm_min, svm_dev_mean
jerk_mean, jerk_std, jerk_max, jerk_min, jerk_energy, jerk_sparsity
KII_mean, KII_std, KII_max, KII_min
omega_mean, omega_std, omega_max, omega_min
theta_mean, theta_std, theta_max, theta_min, theta_range
free_fall_n, impact_n, high_rot_n
GSI, fcri, angular_impulse
press_delta, press_slope
mic_p2p_max, mic_p2p_mean
```

### PPG Features (Optional — ใส่ 0 ได้ถ้าไม่มี sensor)

```
hr_mean, hr_max, hr_delta, hr_spike
spo2_min, spo2_mean, rmssd, sdnn, hr_accel
osi, css_max, css_mean
```

> ℹ️ โมเดลมี `SimpleImputer` อยู่ใน pipeline → ถ้าไม่มี PPG ใส่ค่าเป็น 0 หรือ NaN ได้

---

## 5. วิธี Integrate กับ UNO Q Board

### Option A: Python Inference บน Host → ส่งผลไป UNO Q

```powershell
# ฝั่ง PC / Raspberry Pi
python predict_model2_risk_assessment.py `
  --model .\models\model2_combined\model2_risk_bundle.joblib `
  --input .\live_window.csv `
  --output .\live_prediction.json
```

แล้ว parse JSON ส่งผ่าน Serial / MQTT ไป UNO Q

### Option B: Export Model เป็น ONNX → รันบน UNO Q โดยตรง

```python
# ถ้า UNO Q รอง ONNX Runtime ได้:
from skl2onnx import convert_sklearn
import joblib, onnx

bundle = joblib.load("models/model2_combined/model2_risk_bundle.joblib")
clf = bundle["high_risk_classifier"]
# Export เป็น .onnx แล้วโหลดบน board
```

### Option C: Export feature weights → hardcode บน board (simplest)

ถ้า UNO Q มี RAM จำกัด อาจ hardcode แค่ Logistic Regression weights:

```python
import joblib, numpy as np
bundle = joblib.load("models/model2_combined/model2_risk_bundle.joblib")
clf = bundle["high_risk_classifier"]
print("Coefficients:", clf.coef_)   # ส่งค่านี้ไป hardcode บน board
print("Intercept   :", clf.intercept_)
```

---

## 6. Inference Speed

| Mode | เวลา/window |
|---|---|
| Single window | ~77 ms |
| Full batch (Python) | ~0.05–0.13 ms |

> ✅ ไม่มีปัญหา real-time ที่ 20 Hz (sampling rate 50ms/sample, window 2 sec)

---

## 7. Model Files ที่ต้องส่งให้ทีม

```
fall_detection_final/
├── models/
│   └── model2_combined/
│       ├── model2_risk_bundle.joblib     ← หลัก (9.6 MB)
│       ├── feature_columns.json          ← รายชื่อ features ทั้ง 46
│       ├── final_risk_regressor.joblib   ← Risk Score 0.0–1.0
│       └── final_high_risk_classifier.joblib ← High/Low Risk binary
├── model2_risk_common.py                 ← Helper functions (ต้องส่งด้วย)
├── predict_model2_risk_assessment.py     ← Predict script
└── build_features_from_raw.py           ← Feature extraction from raw
```

---

## 8. ❓ คำถามที่ต้องถามพี่ในทีม (สำหรับ UNO Q Integration)

### เรื่อง Hardware / Communication

- [ ] **UNO Q board รัน Python ได้ไหม?** หรือต้องใช้ firmware ภาษา C/C++?
- [ ] **RAM/Flash ของ UNO Q มีเท่าไหร่?** — Logistic Regression ต้องการ RAM ~50KB, Random Forest ~1.5MB
- [ ] **UNO Q รับ input ผ่านช่องทางไหน?** Serial? I2C? SPI? WiFi/BLE?
- [ ] **ต้องรัน inference บน UNO Q เอง** หรือแค่รับ **ผลลัพธ์** (risk score / risk level) จาก host PC/RPi?

### เรื่อง Data Flow

- [ ] **UNO Q อ่าน sensor อะไร?** IMU เหมือนกันไหม หรือต่างจาก Nano 33 BLE Sense?
- [ ] **Sample rate ของ UNO Q?** ถ้าต่างจาก 20 Hz ต้องปรับ window size
- [ ] **Windowing ทำที่ไหน?** บน UNO Q หรือ host PC? — เพราะ feature extraction ต้องใช้ numpy
- [ ] **ต้องส่ง risk score กี่ Hz?** ทุก window (0.5 sec) หรือแค่ตอน alert?

### เรื่อง Output / Dashboard

- [ ] **Dashboard รับข้อมูลแบบไหน?** WebSocket? MQTT? REST API?
- [ ] **Heatmap ใช้ค่า `model2_risk_score`** (0.0–1.0) หรือ `model2_risk_level` (low/medium/high)?
- [ ] **Alert threshold** — ตั้ง trigger ที่ probability เท่าไหร่? แนะนำ `model2_high_risk_probability >= 0.7`

### เรื่อง Model Deployment

- [ ] **ต้อง export เป็น format ไหน?** `.joblib` / `.onnx` / `.tflite` / hardcoded weights?
- [ ] **รองรับ ONNX Runtime บน UNO Q ไหม?** ถ้าได้จะง่ายมาก
- [ ] **ต้อง retrain บน data ที่ collect จาก UNO Q sensor ด้วยไหม?** หรือใช้โมเดลเดิมได้เลย?

---

## 9. ข้อจำกัดที่ต้องแจ้งทีม

> [!WARNING]
> **Model 2 ไม่ใช่ clinical ground-truth**  
> เป็น proxy risk score สำหรับ prototype ไม่ใช่การวินิจฉัยทางการแพทย์

> [!IMPORTANT]
> PPG features (hr_mean, spo2_min ฯลฯ) ยังเป็น 0 ทั้งหมดในชุดข้อมูลนี้  
> ถ้า UNO Q มี PPG sensor → แจ้งด้วยเพื่อ retrain พร้อมข้อมูล PPG จริง

> [!NOTE]
> AUC = 0.965 วัดบน test set 488 windows  
> ควรทดสอบซ้ำบน live data จาก UNO Q board จริงก่อน deploy production

---

## 10. Quick Start — ทดสอบเองก่อนส่งให้ทีม

```powershell
# 1. Train โมเดลใหม่บน combined dataset
python .\train_model2_risk_assessment.py `
  --data .\windows_combined.csv `
  --run-name model2_combined

# 2. ทดสอบ predict
python .\predict_model2_risk_assessment.py `
  --model .\models\model2_combined\model2_risk_bundle.joblib `
  --input .\windows_combined.csv `
  --output .\reports\model2_combined\predictions_test.csv

# 3. ตรวจสอบ readiness
python .\evaluate_model2_readiness.py `
  --data .\windows_combined.csv `
  --model .\models\model2_combined\model2_risk_bundle.joblib `
  --output-dir .\reports\model2_combined

# 4. สร้าง features จาก raw data ใหม่
python .\build_features_from_raw.py `
  --input merged_dataset_full.csv `
  --output windows_from_merged.csv
```

---

*Last updated: 28 May 2025 | WellSense AIoT — Hackathon 7 | SPAI11*
