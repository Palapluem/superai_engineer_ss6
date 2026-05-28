# 📋 WellSense Model 2 — Checklist & Feature Roadmap

> **วันที่อัปเดต**: 28 พ.ค. 2569 | ทีม SPAI11 | Hackathon 7

---

## ✅ สถานะ Model 2 Core (เสร็จแล้ว)

| รายการ | สถานะ | รายละเอียด |
|---|---|---|
| Dataset: `windows_combined.csv` | ✅ Done | 2,598 windows, 11 class |
| Feature Extraction (46 features) | ✅ Done | IMU: ครบ / PPG: เป็น 0 ทั้งหมด |
| Model Training (Logistic Reg + RF) | ✅ Done | AUC = 0.965, F1 = 0.902 |
| Model Files (`.joblib`) | ✅ Done | `model2_risk_bundle.joblib` |
| Integration Guide | ✅ Done | `model2_uno_q_integration_guide.md` |
| Risk Score Output (0.0–1.0) | ✅ Done | low / medium / high |
| LED Traffic Light Spec | ✅ Done | ดู Section 4 ด้านล่าง |

---

## 🔍 Checklist 3 หัวข้อที่ต้องเช็ค

### 1. 🚨 Anomaly Detection

> **สถานะ: ❌ ยังไม่ได้ทำ — ต้องเพิ่มใน Phase ถัดไป**

**คือ**: ตรวจจับพฤติกรรมที่ "ผิดปกติ" แบบที่ไม่เคยเห็นในข้อมูล training เช่น:
- ล้มแบบใหม่ที่โมเดลไม่รู้จัก
- อุปกรณ์สั่นแรงผิดปกติ (เช่น หกล้มกระแทกพื้น)
- ค่า sensor หลุด range ที่เคยเห็น

**แนวทางทำ (สำหรับทีม):**

```python
# Option A: Isolation Forest (ง่ายที่สุด)
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_features)
anomaly_score = iso.decision_function(X_new)  # ยิ่งต่ำ = ยิ่งผิดปกติ
is_anomaly = iso.predict(X_new)  # -1 = anomaly, 1 = normal

# Option B: ใช้ Reconstruction Error (Autoencoder)
# → ทำ DL ซับซ้อนกว่า แต่จับ pattern ได้ดีกว่า
```

**คำถามที่ต้องถามทีม:**
- [ ] ต้องการ Anomaly Detection แบบ Real-time ไหม?
- [ ] มีตัวอย่าง "anomaly" ให้เทรนไหม หรือทำแบบ Unsupervised?
- [ ] เมื่อตรวจเจอ anomaly ต้องทำอะไร? Alert? Log? แสดงบน Dashboard?

---

### 2. 🧩 จะเทรนโมเดลด้วย Feature เพิ่มไหม?

> **สถานะ: ⚠️ รอข้อมูล — PPG Features ยังเป็น 0 ทั้งหมด**

**Feature ปัจจุบัน (46 ตัว):**
- ✅ **IMU Features (34 ตัว)**: svm, jerk, KII, omega, theta, GSI, press, mic — ใช้งานได้ครบ
- ❌ **PPG Features (12 ตัว)**: hr_mean, hr_max, hr_delta, spo2_min, spo2_mean, rmssd, sdnn ฯลฯ — **เป็น 0 ทั้งหมด** เพราะยังไม่มี sensor จริง

**Feature ที่น่าเพิ่มในอนาคต:**

| Feature Group | ตัวอย่าง | ต้องการ Sensor | ความยาก |
|---|---|---|---|
| PPG / Heart Rate | hr_mean, rmssd, sdnn | Pulse Oximeter | Medium |
| Location | room_id, floor, zone | GPS / BLE Beacon | High |
| Time Pattern | hour_of_day, day_of_week | RTC | Low |
| Cumulative Load | steps_today, activity_intensity | Pedometer | Medium |
| Environmental | temperature, humidity | DHT sensor | Low |

**คำถามที่ต้องถามทีม:**
- [ ] UNO Q มี PPG / SpO2 sensor ติดมาด้วยไหม?
- [ ] ต้องการ retrain พร้อม PPG จริงไหม? (ต้องเก็บข้อมูลใหม่)
- [ ] Feature ไหนที่ Hardware ทีมมีอยู่แล้วบน UNO Q?

---

### 3. 📍 Location Identification

> **สถานะ: ❌ ยังไม่ได้ทำ — ต้องออกแบบ Architecture ใหม่**

**คือ**: ระบุว่าผู้สวมใส่อยู่ที่ไหน เช่น:
- ห้องนอน / ห้องน้ำ / ห้องครัว
- ชั้นไหนของบ้าน
- ใกล้บันได / ใกล้เตียง

**แนวทางทำ (ขึ้นกับ Hardware ที่มี):**

```
Option A: BLE RSSI Fingerprinting (ง่ายสุด สำหรับ indoor)
  → ติด BLE Beacon หลายจุดในบ้าน
  → วัด RSSI จากแต่ละ Beacon บน UNO Q / Nano 33 BLE Sense
  → Train classifier: RSSI vector → room label
  → แนะนำ: k-NN หรือ Random Forest (ง่าย, แม่น indoor)

Option B: GPS (เหมาะกับ outdoor)
  → UNO Q + GPS Module (NEO-6M ราคาถูก)
  → ใช้ geofencing: ถ้าพิกัดอยู่ใน polygon → "บ้าน", "สวน", ฯลฯ

Option C: IMU-based Activity Zone (ไม่ต้องการ Hardware เพิ่ม)
  → ใช้ pattern ของ sensor เดิม
  → เช่น ถ้าก้าวยก-ลงซ้ำ = บันได, นอนนิ่ง = เตียง
  → ความแม่นน้อยกว่า แต่ไม่ต้องเพิ่ม sensor
```

**คำถามที่ต้องถามทีม:**
- [ ] ต้องการ Location ระดับไหน? (ห้อง / ชั้น / พิกัด GPS?)
- [ ] บ้านที่ใช้ทดสอบมี BLE Access Point ไหม?
- [ ] UNO Q มี GPS module ด้วยไหม?
- [ ] Location ต้องแสดงบน Dashboard ด้วยไหม?

---

## 🚦 Section 4: ระบบสัญญาณไฟจราจร (LED Traffic Light)

> **สำหรับส่งให้ทีม Hardware / Arduino UNO Q**

### สถาปัตยกรรมการทำงาน

```
[Arduino Nano 33 BLE Sense]
  ↓ ส่ง IMU raw data ผ่าน Serial / BLE
[Host PC / Raspberry Pi]
  ↓ รัน Python: Feature Extraction + Model 2 Inference
  ↓ ส่งผล: risk_level = "low" / "medium" / "high"
[Arduino UNO Q]
  ↓ รับค่า risk_level ผ่าน Serial
  → ควบคุม LED ตามตาราง
```

### นิยามสัญญาณ

| สี LED | ความหมาย | เงื่อนไข (Risk Score) | พฤติกรรม LED |
|---|---|---|---|
| 🟢 **เขียว** | ปกติ — ปลอดภัย | `risk_score < 0.35` (low) | ติดค้าง |
| 🟡 **เหลือง** | เฝ้าระวัง — สะดุด/เสี่ยง | `0.35 ≤ risk_score < 0.65` (medium) | กะพริบช้า 1 Hz |
| 🔴 **แดง** | อันตราย — ล้มแล้ว/HIGH RISK | `risk_score ≥ 0.65` (high) | กะพริบเร็ว 4 Hz |

**Event-based Override (ล้มฉับพลัน):**

| เหตุการณ์ | การตอบสนอง |
|---|---|
| `component_impact_event ≥ 0.8` | บังคับไฟแดงกะพริบเร็ว 5 วินาที |
| `model2_high_risk_probability ≥ 0.9` | เพิ่ม Buzzer Alert + ไฟแดง |

---

### วงจร LED (การต่อสาย UNO Q)

```
UNO Q Pin 9  → Resistor 220Ω → LED แดง  → GND
UNO Q Pin 10 → Resistor 220Ω → LED เหลือง → GND
UNO Q Pin 11 → Resistor 220Ω → LED เขียว → GND
UNO Q Pin 8  → Buzzer (+)                → GND  (optional)
```

> ⚠️ หากใช้ LED ภายนอก ต้องใช้ Resistor 220Ω ทุกสาย มิฉะนั้น LED จะขาด

---

### Arduino Sketch (C++ สำหรับ UNO Q)

ดูไฟล์: `wellsense_led_traffic_light.ino`

---

### Protocol Serial ที่ Host PC ต้องส่งมา

```
FORMAT: <risk_level>,<risk_score>,<impact_event>\n

ตัวอย่าง:
  low,0.12,0.05\n       → ไฟเขียว
  medium,0.48,0.10\n    → ไฟเหลืองกะพริบ
  high,0.82,0.90\n      → ไฟแดงกะพริบเร็ว + buzzer
  high,0.91,0.98\n      → ไฟแดง + buzzer (impact override)
```

---

## 📌 สรุปสิ่งที่ต้องทำต่อ (Priority)

| Priority | งาน | ขึ้นกับ |
|---|---|---|
| 🔴 HIGH | ทดสอบ LED Sketch บน UNO Q จริง | Hardware team |
| 🔴 HIGH | เชื่อม Serial Protocol: Host Python → UNO Q | Software + Hardware |
| 🟡 MEDIUM | Anomaly Detection (Isolation Forest) | ML team |
| 🟡 MEDIUM | เพิ่ม Location Identification | ต้องรู้ Hardware ที่มี |
| 🟢 LOW | Retrain พร้อม PPG Features จริง | ต้องเก็บ data ใหม่ |

---

*Last updated: 28 พ.ค. 2569 | WellSense AIoT Hackathon 7 | SPAI11*
