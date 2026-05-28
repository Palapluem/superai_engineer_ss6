# สรุป Model 2: Assessment Risk

## 1. Model 2 คืออะไร

Model 2 ไม่ใช่โมเดลจับว่า "ล้ม / ไม่ล้ม" แบบ Model 1

Model 2 คือชั้นประเมินความเสี่ยงของการเคลื่อนไหวจาก feature เช่น jerk, omega, theta, GSI, FCRI และ PPG proxy เพื่อให้ได้:

- `model2_risk_score` ช่วง 0.0-1.0
- `model2_high_risk_probability`
- `model2_risk_level` เป็น `low`, `medium`, `high`

คำอธิบายสั้น ๆ:

> Model 1 ใช้ตรวจจับเหตุการณ์ล้ม ณ ตอนนั้น ส่วน Model 2 ใช้ประเมินความเสี่ยง/ความไม่มั่นคงของ movement window เพื่อใช้ทำ dashboard, alert, และ heatmap พื้นที่เสี่ยง

## 2. ทำไม Model 2 ยังเป็น Proxy Risk

ใน dataset ตอนนี้ยังไม่มี label ทางคลินิกว่า "คนนี้มีโอกาสล้มในอนาคตกี่เปอร์เซ็นต์" หรือ expert fall-risk score จริง ๆ

ดังนั้น Model 2 ตอนนี้จึงสร้าง target แบบ domain-informed proxy:

```text
weighted_score = 0.60 * feature_risk_score + 0.40 * activity_prior_score
prior_floor    = 0.95 * activity_prior_score
model2_risk_target = max(weighted_score, prior_floor)
```

จุดประสงค์คือให้ได้ risk score ที่อธิบายได้และใช้ prototype ได้ก่อน โดยยังต้องเก็บข้อมูลจริงเพิ่มเพื่อ validate ต่อ

## 3. Feature ที่ใช้

Model 2 ใช้ feature numeric จาก `windows_all.csv` จำนวน 46 features

กลุ่ม feature หลัก:

| กลุ่ม | ตัวอย่าง feature | ความหมาย |
|---|---|---|
| gait/motion | `jerk_mean`, `jerk_max`, `svm_std` | การเคลื่อนไหวกระชากหรือไม่นิ่ง |
| rotation/balance | `omega_mean`, `omega_max`, `angular_impulse` | การหมุนตัว/เสียสมดุล |
| posture transition | `theta_range`, `KII_mean`, `KII_max` | การเปลี่ยนท่าทางหรือเอียงตัว |
| impact/fall-like | `GSI`, `fcri`, `impact_n`, `free_fall_n` | สัญญาณลักษณะใกล้ fall/impact |
| PPG/physio proxy | `hr_delta`, `hr_spike`, `osi`, `spo2_min` | สัญญาณกายภาพประกอบ |

## 4. ผลที่รันแล้ว

ใช้ group split ตาม `session_id` เพื่อให้ประเมินเข้มกว่า random split

Regressor ที่ดีที่สุด:

```text
random_forest_regressor
MAE  = 0.0298
RMSE = 0.0703
R2   = 0.9096
```

High-risk classifier:

```text
random_forest_classifier
F1 = 1.0000
Balanced accuracy = 1.0000
ROC-AUC = 1.0000
```

หมายเหตุ: คะแนน classifier สูงมากเพราะ high-risk proxy target แยกจาก feature ได้ชัดใน dataset นี้ จึงไม่ควรพูดว่าเป็น clinical performance จริง

Readiness check ล่าสุด:

```text
Class count เทียบกับ image.png: PASS
Feature completeness: PASS
Required features: 46/46
Missing feature: ไม่มี
Feature ที่มี null: svm_dev_mean 338 rows
```

`svm_dev_mean` มีค่าว่างบางแถว แต่ pipeline ใช้ median imputation อยู่แล้ว จึง predict ได้

## 5. Risk score ที่ได้โดยประมาณ

ค่ากลางของ prediction ตาม class:

| Class | Median risk |
|---|---:|
| standing | 0.117 |
| lying_down | 0.123 |
| corrected_walking | 0.291 |
| normal_walk | 0.324 |
| elderly_pick_up_object | 0.368 |
| limping_walk | 0.450 |
| stand_sit_alternating | 0.496 |
| gradual_fall | 0.746 |
| slow_collapse_fall | 0.779 |
| backward_fall | 0.874 |
| sideways_fall | 0.874 |

ตีความ:

- static ต่ำ
- walking ปกติอยู่แถว low ถึง medium ต่ำ
- pick object, limping, stand-sit เป็น medium
- fall-like activity เป็น high

## 5.1 Inference time

วัดจาก `model2_risk_bundle.joblib` บนเครื่อง Windows เครื่องนี้ โดย predict ทั้ง risk regressor และ high-risk classifier

| Batch size | Mean batch time | Mean per window | ตีความ |
|---:|---:|---:|---|
| 1 | 77.9 ms | 77.9 ms/window | ใช้กับ real-time ทีละ window |
| 10 | 73.7 ms | 7.37 ms/window | ถ้าทำนายเป็นชุดเล็ก |
| 100 | 74.3 ms | 0.743 ms/window | batch inference เร็วมาก |
| 2138 | 102.0 ms | 0.048 ms/window | ทั้ง dataset |

ใน dataset นี้ window ยาวประมาณ 1.95 วินาที และ stride ประมาณ 0.5 วินาที ดังนั้น inference ทีละ window ประมาณ 0.078 วินาที ถือว่าเร็วกว่า sampling window มาก

คำพูดแนะนำ:

> บนเครื่อง Windows ตอนนี้ Model 2 ใช้เวลาทำนายประมาณ 78 ms ต่อหนึ่ง feature window ถ้ารันทีละ window ซึ่งน้อยกว่าความยาว window 1.95 วินาทีมาก จึงทันสำหรับ prototype real-time บน edge/server แต่ถ้าจะลงไมโครคอนโทรลเลอร์ตรง ๆ ต้องแปลงโมเดลหรือลอง benchmark บนบอร์ดจริงอีกครั้ง

## 6. ไฟล์สำคัญ

Train:

```powershell
python .\train_model2_risk_assessment.py `
  --data .\windows_all.csv `
  --run-name model2_risk_assessment
```

Predict:

```powershell
python .\predict_model2_risk_assessment.py `
  --model .\models\model2_risk_assessment\model2_risk_bundle.joblib `
  --input .\windows_all.csv `
  --output .\reports\model2_risk_assessment\model2_predictions_windows_all.csv
```

Model bundle:

```text
models\model2_risk_assessment\model2_risk_bundle.joblib
```

Report:

```text
reports\model2_risk_assessment\summary_report.md
reports\model2_risk_assessment\risk_formula_config.json
reports\model2_risk_assessment\model2_predictions_windows_all.csv
reports\model2_risk_assessment\model2_readiness_report.md
reports\model2_risk_assessment\model2_inference_benchmark.csv
reports\model2_risk_assessment\model2_feature_completeness.csv
reports\model2_risk_assessment\model2_class_count_check.csv
```

ตรวจ class counts, feature completeness, inference time:

```powershell
python .\evaluate_model2_readiness.py `
  --data .\windows_all.csv `
  --model .\models\model2_risk_assessment\model2_risk_bundle.joblib `
  --output-dir .\reports\model2_risk_assessment
```

ตรวจ imbalance และ missing values ของไฟล์ต้นทาง:

```powershell
python .\analyze_dataset_balance.py `
  --base-dir . `
  --output-dir .\reports\dataset_balance
```

ไฟล์ report:

```text
reports\dataset_balance\dataset_balance_report.md
reports\dataset_balance\windows_all_class_distribution.png
reports\dataset_balance\class_distribution_across_files.png
reports\dataset_balance\top_missing_columns.png
```

## 7. ประโยคแนะนำสำหรับพูดกับรุ่นพี่

> ตอนนี้ Model 2 ถูกแยกจาก Model 1 แล้วครับ Model 1 เป็น binary fall/no-fall detector ส่วน Model 2 เป็น assessment risk layer ที่ใช้ feature อย่าง jerk, omega, theta, GSI, FCRI และ PPG proxy เพื่อประเมิน risk score 0-1 ต่อ movement window โดยตอนนี้ target ยังเป็น proxy risk เพราะ dataset ยังไม่มี clinical fall-risk label จริง แต่ pipeline พร้อมสำหรับ prototype และสามารถนำ output ไปทำ dashboard/heatmap ได้

## 8. สิ่งที่ควรทำต่อ

- เก็บข้อมูลจริงจาก Nano/UNO Q เพิ่ม โดยให้มี normal walk, limping, stand-sit, pick object, static หลายรอบ
- ให้ทีม/รุ่นพี่ช่วยนิยาม risk label จริง เช่น low/medium/high จาก protocol
- เอา Model 2 ไป predict live window แล้วเทียบกับ observation จริง
- รวม `model2_risk_score + x,y` เพื่อทำ heatmap พื้นที่เสี่ยงในบ้าน
