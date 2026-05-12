# CatBoost Demand Forecasting Summary

ไฟล์ notebook หลัก:

`C:\Users\Natth\Downloads\super-ai-engineer-season-6-coffee-chain-hackathon\catboost_leakage_safe_baseline.ipynb`

## เป้าหมาย

สร้างโมเดลพยากรณ์ยอดขายรายวันระดับ:

```text
store_id × category × forecast_date × horizon
```

โดยมี 3 horizons:

```text
1d, 7d, 1m
```

เราใช้ **CatBoost 3 โมเดลแยกตาม horizon** เพื่อควบคุม data leakage ได้ง่ายและอธิบายชัดตอน code review

## Target

Target สร้างตามนิยามของโจทย์:

```text
TRANSACTION + ORDER + PRODUCT
```

แล้ว aggregate เป็น:

```text
store_id, category, date -> sum(units_sold)
```

วันไหนไม่มี transaction จะเติมยอดขายเป็น `0`

## Data Restriction

Notebook นี้ไม่ใช้ไฟล์ที่โจทย์ห้าม:

```text
test/TRANSACTION.csv
test/ORDER.csv
test/INVENTORY.csv
```

ใช้ test lookup เฉพาะไฟล์ที่โจทย์อนุญาต:

```text
test/DATE_DIM.csv
test/STORE.csv
test/PRODUCT.csv
test/PROMOTION.csv
test/LOCAL_EVENT.csv
```

## Leakage-Safe Rule

หลักสำคัญที่สุดคือ:

```text
anchor_date = forecast_date - horizon
```

Feature ที่มาจากยอดขายหรือ stockout ต้องใช้ข้อมูลได้ถึง `anchor_date` เท่านั้น

ตัวอย่าง:

```text
forecast_date = 2024-11-01
horizon = 7d
anchor_date = 2024-10-25
```

ดังนั้น `sales_lag0` คือยอดขายวันที่ `2024-10-25` ไม่ใช่ยอดขายวันที่ `2024-11-01`

สำหรับ test window ถ้า `decision_date` อยู่ใน Nov-Dec จะ cap history ที่:

```text
2024-10-31
```

เพื่อไม่ใช้ข้อมูล time-varying ของช่วง test ที่โจทย์ไม่ได้แจก

## Feature Groups

### 1. Historical Sales Features

ใช้ยอดขายย้อนหลังของ `store_id × category` เดิม

```text
sales_lag0
sales_lag7
sales_lag14
sales_lag21
sales_lag28
sales_lag35
sales_lag56
sales_lag84
```

ใช้จับยอดขายล่าสุดและ weekly seasonality

### 2. Rolling Demand Features

คำนวณจากยอดขายย้อนหลังถึง `anchor_date`

```text
sales_roll_mean_7
sales_roll_mean_14
sales_roll_mean_28
sales_roll_mean_56
sales_roll_std_28
sales_roll_std_56
sales_roll_max_28
sales_roll_max_56
sales_roll_nonzero_7
sales_roll_nonzero_14
sales_roll_nonzero_28
sales_roll_nonzero_56
```

ใช้จับ:

- demand level
- volatility
- spike
- intermittent demand

### 3. Trend and Stability Features

```text
sales_trend_7_vs_28
sales_cv_28
```

ความหมาย:

- `sales_trend_7_vs_28`: ค่าเฉลี่ย 7 วันล่าสุด ลบค่าเฉลี่ย 28 วัน
- `sales_cv_28`: ความผันผวนเทียบกับค่าเฉลี่ย

### 4. Stockout-Aware Features

ใช้ข้อมูลจาก `train/INVENTORY.csv` เท่านั้น

```text
stockout_lag0
stockout_rate_roll_28
closing_stock_roll_mean_28
sample_weight
```

สำคัญ:

- ไม่ใช้ `INVENTORY.units_sold` เป็น target
- ไม่ uncensor demand
- ไม่แก้ยอดขายจริง
- ใช้ stockout เพื่อลดน้ำหนัก row ที่อาจถูก truncate เพราะของหมด

### 5. Calendar Features

จาก `DATE_DIM`

```text
day_of_week
week_number
month
quarter
year
is_weekend
is_holiday
holiday_name
is_school_break
is_payday
is_rainy_season
```

ใช้จับ pattern ตามวันหยุด payday ฤดูกาล และวันในสัปดาห์

### 6. Promotion Features

จาก `PROMOTION`

```text
promo_active
promo_count
promo_discount_mean
promo_discount_max
promo_email_sent
promo_social_campaign
promo_type
```

ใช้จับผลของโปรโมชั่น เช่น ส่วนลด, ซื้อ 1 แถม 1, email campaign, social campaign

### 7. Local Event Features

จาก `LOCAL_EVENT`

```text
event_count
event_type
```

ใช้จับผลจาก event ใกล้สาขา เช่น concert, market, sports, cultural event

### 8. Store Features

จาก `STORE`

```text
neighborhood_type
seating_capacity
has_drive_through
staff_count
open_time
close_time
open_hour
close_hour
store_age_days
```

ใช้ให้โมเดลเข้าใจความต่างของแต่ละสาขา

### 9. Category / Product Features

สรุปจาก `PRODUCT` เป็นระดับ category

```text
product_count
avg_base_price
max_base_price
min_base_price
seasonal_rate
limited_rate
serve_type_nunique
```

ใช้บอกลักษณะของ category เช่น ราคาเฉลี่ย จำนวนสินค้า และสัดส่วน seasonal/limited products

## Categorical Features for CatBoost

CatBoost รับ categorical features โดยตรง จึงไม่ต้องทำ one-hot encoding

```text
store_id
category
day_of_week
holiday_name
neighborhood_type
has_drive_through
open_time
close_time
promo_type
event_type
```

ข้อดี:

- ประหยัด memory
- เหมาะกับข้อมูลตาราง
- เรียนรู้ interaction เช่น `store_id × category × promo_type` ได้ดี

## Validation Result

ใช้ time-based validation:

```text
Train: ก่อน 2024-09-01
Valid: 2024-09-01 ถึง 2024-10-31
```

ผล validation:

```text
Overall MAE ≈ 8.6940
```

แยกตาม horizon:

```text
1d ≈ 8.4442
7d ≈ 8.8352
1m ≈ 8.8028
```

จากกราฟ diagnostic:

- horizon ทั้ง 3 ตัวไม่ต่างกันมาก
- Coffee มี MAE สูงสุด เพราะยอดขายใหญ่และผันผวนกว่า category อื่น
- error distribution มี long tail ซึ่งเป็นปกติของ retail demand

## Final Training

หลัง validation มี cell:

```text
Retrain Final Models with Full Jan-Oct Data
```

Cell นี้ retrain โมเดลสุดท้ายด้วยข้อมูลทั้งหมดถึง:

```text
2024-10-31
```

เหตุผลคือโมเดล validation ใช้ train แค่ก่อน `2024-09-01` แต่ submission จริงควรใช้ข้อมูล Sep-Oct ด้วย

## Output

ไฟล์ submission ที่ได้:

```text
submission_catboost.csv
```

ผ่าน sanity checks:

```text
25,620 rows
2 columns: id, units_sold_predicted
id ตรงกับ sample submission
ไม่มี missing values
ไม่มีค่าติดลบ
```

## วิธีรัน

เปิด notebook:

```text
catboost_leakage_safe_baseline.ipynb
```

แล้วกด Run All

ถ้ารันไปแล้วบางส่วน ให้รันตั้งแต่ cell นี้ลงไปก่อน export:

```text
Retrain Final Models with Full Jan-Oct Data
```

จากนั้นรันต่อจนถึง cell สร้าง `submission_catboost.csv`

