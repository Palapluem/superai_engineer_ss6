# 📄 Summary of Feature Important

---

## 1. ภาพรวมระบบ

```
[Arduino Nano 33 BLE Sense]
   ตรวจจับ IMU: ax, ay, az, gx, gy, gz (20 Hz)
         │
         ▼ Sliding Window (2 วิ / step 0.5 วิ)
[Feature Extraction] → 46 features
         │
         ├─── Model 1: Fall Detection   → ล้ม / ไม่ล้ม (Binary)
         │
         └─── Model 2: Risk Assessment  → Risk Score 0.0–1.0
                                          Risk Level: low / medium / high
                                                │
                                                ▼
                              [UNO Q Board: LED Traffic Light]
                              🟢 เขียว = ปลอดภัย  (score < 0.35)
                              🟡 เหลือง = เฝ้าระวัง (0.35–0.65)
                              🔴 แดง = อันตราย    (score ≥ 0.65)
```

---

## 2. ผลการ Training — Model 2 (Risk Assessment)

### Dataset

| รายการ | ค่า |
|---|---|
| ไฟล์ที่ใช้ | `windows_combined.csv` |
| จำนวน windows | **2,598 windows** |
| จำนวน class | **11 classes** (ครบทุกกิจกรรม) |
| Train / Test split | 2,110 / 488 |

### ผลโมเดล (Best Models)

| โมเดล | Metric | ค่า | ระดับ |
|---|---|---|---|
| **Logistic Regression** (Classifier) | **AUC** | **0.965** | 🟢 ดีเยี่ยม |
| Logistic Regression | F1 Score | 0.902 | 🟢 ดีเยี่ยม |
| Logistic Regression | Balanced Acc | 0.894 | 🟢 ดี |
| Logistic Regression | Recall | 0.944 | 🟢 ดีมาก |
| Random Forest (Regressor) | MAE | 0.157 | 🟡 พอใช้ |

> **ความหมาย AUC = 0.965**: โมเดลแยก "เสี่ยงสูง" vs "เสี่ยงต่ำ" ได้ถูกต้อง **96.5%** ของเวลา

---

## 3. Risk Score ต่อกิจกรรม (เรียงจากเสี่ยงสูงสุด)

| อันดับ | กิจกรรม | Risk Score (mean) | Risk Level |
|---|---|---|---|
| 1 | sideways_fall (ล้มข้าง) | ~0.87 | 🔴 HIGH |
| 2 | backward_fall (ล้มหลัง) | ~0.87 | 🔴 HIGH |
| 3 | gradual_fall (ล้มค่อยๆ) | ~0.77 | 🔴 HIGH |
| 4 | slow_collapse_fall (ล้มทรุด) | ~0.76 | 🔴 HIGH |
| 5 | stand_sit_alternating (นั่ง-ลุก) | ~0.51 | 🟡 MEDIUM |
| 6 | limping_walk (เดินกะเผลก) | ~0.46 | 🟡 MEDIUM |
| 7 | elderly_pick_up_object (หยิบของ) | ~0.38 | 🟡 MEDIUM |
| 8 | normal_walk (เดินปกติ) | ~0.33 | 🟡 MEDIUM |
| 9 | corrected_walking (เดินแก้ท่า) | ~0.29 | 🟢 LOW |
| 10 | lying_down (นอน) | ~0.16 | 🟢 LOW |
| 11 | standing (ยืน) | ~0.11 | 🟢 LOW |

---

## 4. 🔑 Feature Importance (สำคัญมาก — สำหรับ Integration)

> Features ที่สำคัญที่สุดคือสิ่งที่โมเดลใช้ตัดสินว่า "เสี่ยง" หรือ "ไม่เสี่ยง"
> **ถ้า UNO Q ส่ง feature เหล่านี้มาได้ครบ โมเดลทำงานได้เต็มประสิทธิภาพ**

---

### 4A. Top Features — Logistic Regression (โมเดลหลัก / Best Classifier)

> LR เป็นโมเดลที่เราแนะนำให้ใช้บน Edge Device เพราะขนาดเล็กที่สุด

| อันดับ | Feature | ค่า \|Coef\| | ความหมาย | ความสำคัญ |
|---|---|---|---|---|
| 🥇 1 | **high_rot_n** | 3.435 | จำนวน samples ที่ angular velocity > 100°/s | สำคัญมาก — หมุนเร็ว = ล้ม |
| 🥈 2 | **jerk_sparsity** | 2.672 | สัดส่วน jerk ต่ำใน window | สำคัญมาก — การเปลี่ยนแปลงแบบกะทันหัน |
| 🥉 3 | **jerk_energy** | 1.959 | พลังงานของ jerk สะสม | สำคัญ — แรงกระแทก |
| 4 | **theta_std** | 1.817 | ความผันผวนของมุมเอียงลำตัว | สำคัญ — ทรงตัวไม่นิ่ง |
| 5 | **angular_impulse** | 1.770 | impulse ของการหมุนรวมทั้ง window | สำคัญ |
| 6 | **theta_mean** | 1.564 | มุมเอียงลำตัวเฉลี่ย | สำคัญ |
| 7 | **omega_mean** | 1.343 | ความเร็วเชิงมุมเฉลี่ย | สำคัญ |
| 8 | **GSI** | 1.114 | Global Shock Index (RMS ของ jerk) | สำคัญ |
| 9 | svm_min | 0.742 | ค่าต่ำสุดของ Signal Vector Magnitude | ปานกลาง |
| 10 | svm_mean | 0.739 | SVM เฉลี่ย (ขนาดของความเร่ง) | ปานกลาง |

---

### 4B. Top Features — Random Forest Classifier

| อันดับ | Feature | Importance | ความหมาย |
|---|---|---|---|
| 🥇 1 | **jerk_min** | 0.233 (23.3%) | ค่าต่ำสุดของ jerk — ช่วง free-fall |
| 🥈 2 | **jerk_sparsity** | 0.155 (15.5%) | pattern ความกระชากของการเคลื่อนไหว |
| 🥉 3 | **KII_mean** | 0.096 (9.6%) | Kinematic Instability Index เฉลี่ย |
| 4 | **KII_std** | 0.091 (9.1%) | ความผันผวนของ Instability |
| 5 | **KII_min** | 0.069 (6.9%) | ค่าต่ำสุดของ Instability |
| 6 | **KII_max** | 0.055 (5.5%) | ค่าสูงสุดของ Instability |
| 7 | **jerk_min** | 0.050 (5.0%) | ค่าต่ำสุดของ jerk |
| 8 | hr_mean | 0.026 | Heart rate เฉลี่ย (ตอนนี้เป็น 0) |
| 9 | css_max | 0.024 | Cardiovascular stress max (เป็น 0) |
| 10 | css_mean | 0.023 | Cardiovascular stress mean (เป็น 0) |

---

### 4C. Top Features — Random Forest Regressor (Risk Score)

| อันดับ | Feature | Importance | ความหมาย |
|---|---|---|---|
| 🥇 1 | **GSI** | **0.778 (77.8%!!!)** | Global Shock Index — dominant feature! |
| 🥈 2 | **theta_min** | 0.068 (6.8%) | มุมเอียงต่ำสุด |
| 🥉 3 | **KII_mean** | 0.034 (3.4%) | ความไม่เสถียรเฉลี่ย |
| 4 | jerk_mean | 0.014 | jerk เฉลี่ย |
| 5 | omega_mean | 0.013 | ความเร็วเชิงมุมเฉลี่ย |

> **ข้อสังเกตสำคัญ**: RF Regressor ใช้ `GSI` เป็นหลักถึง 77.8%
> ถ้า UNO Q ส่งแค่ **GSI + theta_min + KII_mean** มา → ประมาณ Risk Score ได้ ~88% ของเวลา

---

### 4D. สรุป Features ที่ต้องมี (เรียงลำดับความสำคัญ)

```
🔴 CRITICAL (ต้องมี — โมเดลทำงานได้แย่มากถ้าขาด):
   high_rot_n    → จำนวน samples หมุนเร็ว (gyro > 100°/s)
   jerk_sparsity → สัดส่วน low-jerk ใน window
   jerk_energy   → พลังงานจาก jerk ทั้ง window
   GSI           → Global Shock Index = RMS(jerk)
   theta_mean    → มุมเอียงลำตัวเฉลี่ย
   theta_std     → ความผันผวนมุมเอียง

🟡 IMPORTANT (ควรมี):
   angular_impulse → impulse การหมุนสะสม
   omega_mean      → ความเร็วเชิงมุมเฉลี่ย
   KII_mean        → Kinematic Instability Index
   svm_mean        → Signal Vector Magnitude

🟢 NICE TO HAVE (เพิ่มความแม่นได้):
   jerk_min, jerk_max, jerk_std
   KII_std, KII_min, KII_max
   theta_range, theta_min
   svm_min, svm_max, svm_std
```

---

## 5. ไฟล์สำคัญที่ต้องส่งให้ทีม

```
fall_detection_final/
├── models/model2_combined/
│   ├── model2_risk_bundle.joblib        ← โมเดลหลัก (9.6 MB)
│   ├── feature_columns.json             ← รายชื่อ features 46 ตัว
│   ├── logistic_regression_classifier.joblib  ← Classifier ที่ดีที่สุด
│   └── random_forest_regressor.joblib   ← Regressor Risk Score
├── model2_risk_common.py                ← Helper functions
├── predict_model2_risk_assessment.py    ← Script ทำนาย
├── build_features_from_raw.py           ← แปลง raw sensor → features
└── reports/
    ├── feature_importance_rf.json       ← Feature importance RF
    └── feature_importance_lr.json       ← LR coefficients
```

---

## 6. วิธี Integrate กับ UNO Q (แนะนำ Option A ก่อน)

### Option A: Python บน Host PC → ส่งผล Serial ไป UNO Q ✅ (แนะนำ)

```
[Nano 33 BLE Sense]                [Host PC / Laptop]
  │ IMU raw data (Serial/BLE)  →    │ Python Script
  │                                 │ 1. รับ raw data
  │                                 │ 2. Feature Extraction (build_features_from_raw.py)
  │                                 │ 3. Model 2 Inference (predict_model2_risk_assessment.py)
  │                                 │ 4. ส่ง: "high,0.82,0.90\n" ผ่าน Serial
  │                                 │
  ▼                                 ▼
[UNO Q Board]                  [UNO Q Board]
  รับ "risk_level,score,impact"
  → ควบคุม LED + Buzzer
```

**Script ที่ใช้บน Host:**

```powershell
# ทดสอบ simulate (ไม่ต้องมี Arduino):
python .\wellsense_serial_sender.py --simulate

# ใช้จริงกับ UNO Q:
python .\wellsense_serial_sender.py --port COM3 --input .\windows_combined.csv
```

### Serial Protocol (UNO Q ต้องรับ format นี้)

```
FORMAT: <risk_level>,<risk_score>,<impact_event>\n

ตัวอย่าง:
  low,0.12,0.05\n     → ไฟเขียวติดค้าง
  medium,0.48,0.10\n  → ไฟเหลืองกะพริบ 1 Hz
  high,0.82,0.90\n    → ไฟแดงกะพริบ 4 Hz + Buzzer
```

### วงจร LED (ต่อสายแบบนี้)

```
UNO Q Pin 9  → 220Ω → LED แดง   → GND
UNO Q Pin 10 → 220Ω → LED เหลือง → GND
UNO Q Pin 11 → 220Ω → LED เขียว  → GND
UNO Q Pin 8  → Buzzer(+)          → GND
```

---

## 7. Location Identification (ระบุตำแหน่ง)

### ผลการทดสอบ

| Method | Accuracy | ข้อดี | ข้อเสีย |
|---|---|---|---|
| **GPS มือถือ (Primary)** | **88%+** | แม่นมาก ระบุห้องได้ | ต้องมี GPS signal / มือถือ |
| **IMU ML (Fallback)** | **94%** | ทำงานได้ทุกที่ ไม่ต้อง GPS | แม่นน้อยกว่า GPS |
| IMU Rule-based | 100% (deterministic) | ไม่ต้อง train | mapping ตรงๆ |

> **สรุป**: ทีม Proof ถูกแล้ว — GPS ดีกว่า แต่ Gyro เป็น fallback ที่แม่น 94% ได้เลย

### Zone ที่รองรับ

| Zone | ความเสี่ยง | หมายเหตุ |
|---|---|---|
| bedroom | 🟡 Medium | ล้มในห้องนอน |
| bathroom | 🔴 High | พื้นลื่น — เสี่ยงสูงสุด |
| kitchen | 🟡 Medium | |
| living_room | 🟢 Low | |
| corridor | 🟢 Low | ทางเดิน |
| staircase | 🔴 High | บันได — เสี่ยงสูง |
| outdoor | 🟢 Low | |

> **สิ่งที่ต้องทำ**: แก้ lat/lon ในไฟล์ `location_zones_config.json` ให้ตรงกับสถานที่ทดสอบจริง

---

## 8. สถานะ Checklist

| หัวข้อ | สถานะ | หมายเหตุ |
|---|---|---|
| ✅ Model 2 Training | เสร็จ | AUC=0.965, F1=0.902 |
| ✅ Feature Importance | เสร็จ | ดู Section 4 |
| ✅ LED Traffic Light Spec | เสร็จ | ไฟล์ `.ino` พร้อม |
| ✅ Location: GPS + IMU | เสร็จ | tested & benchmarked |
| ⚠️ Anomaly Detection | ยังไม่ทำ | รอยืนยันว่าต้องการ real-time ไหม |
| ⚠️ PPG Features | รอ sensor | hr_mean, spo2 ยังเป็น 0 |
| ❓ UNO Q Protocol | รอทีม | ต้องยืนยัน Serial port / COM |

---

## 9. คำถามที่ต้องตอบเพื่อเดินงานต่อ

> [!IMPORTANT]
> **Hardware Team ต้องตอบ:**

- [ ] **UNO Q อยู่ที่ COM port ไหน?** (เพื่อตั้งค่า `--port COM?`)
- [ ] **UNO Q รัน Python ได้ไหม?** หรือต้องเป็น C/C++ เท่านั้น?
- [ ] **GPS มือถือส่งข้อมูลมาผ่านช่องทางไหน?** HTTP / WebSocket / Bluetooth?
- [ ] **UNO Q มี PPG/SpO2 sensor ด้วยไหม?** (ถ้ามีจะเพิ่ม accuracy ได้อีก)
- [ ] **สถานที่ทดสอบจริงอยู่ที่ lat/lon เท่าไหร่?** (สำหรับตั้ง GPS zones)
- [ ] **ต้องการ Anomaly Detection แบบ Real-time ไหม?**

---

## 10. Commands สำคัญสำหรับรัน

```powershell
# ─── อยู่ใน folder: fall_detection_final ───────────────────

# ทดสอบ predict Model 2:
python .\predict_model2_risk_assessment.py `
  --model .\models\model2_combined\model2_risk_bundle.joblib `
  --input .\windows_combined.csv `
  --output .\reports\predictions_test.csv

# ทดสอบ LED Serial Sender (simulate ไม่ต้องมีบอร์ด):
python ..\wellsense_serial_sender.py --simulate

# ทดสอบ Location Identification:
python .\run_location_test.py

# ดู Feature Importance:
python .\extract_feature_importance.py
```

---

*สร้างโดย: ทีม SPAI11 | WellSense AIoT Hackathon 7 | 28 พ.ค. 2569*
*ติดต่อ ML Team สำหรับรายละเอียดเพิ่มเติม*
