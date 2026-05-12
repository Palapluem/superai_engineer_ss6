# THE ULTIMATE BLUEPRINT: Demand Forecasting Hackathon

แผนการดำเนินงานฉบับสมบูรณ์และละเอียดที่สุด (Comprehensive Master Plan) ที่ครอบคลุม Insight ทุกมิติ ตั้งแต่การดึงมูลค่าสูงสุดจากข้อมูลภายใน (Internal Data), การทำ Spatial EDA ด้วย DBSCAN, การจับคู่สาขากับเว็บไซต์, และการปฏิบัติตามกฎเหล็กของการแข่งขันอย่างเคร่งครัด

---

## Part 1: Data Integration & Web Scraping Mapping
การผสมผสานข้อมูลที่ต้องรับมือกับความไม่สมบูรณ์ (Missing External Data):

1. **Store ID Matching (การจับคู่สาขากับเว็บไซต์ Cafe Amazon):**
   - นำ Store ID ในข้อมูล `STORE` ไปแมพกับพิกัดที่ Scrape มาจาก [cafe-amazon.com/our-store](https://www.cafe-amazon.com/our-store)
   - **Critical Insight:** ในโลกความเป็นจริง จะต้องเจอ "สาขาที่ไม่ปรากฏบนเว็บ" (เช่น สาขาชั่วคราว, สาขาในสำนักงานปิด, บูธงาน Event หรือเพิ่งปิดตัว)
   - *Action:* สร้างฟีเจอร์ `is_unlisted_store = 1` ถ้าสาขานั้นไม่มีในเว็บ และจะใช้ค่าเฉลี่ยของสาขาประเภทเดียวกัน (Imputation by Format) สำหรับข้อมูลพิกัดหรือระยะทางที่หายไป
2. **External Data Integration:**
   - รวมข้อมูลสภาพเศรษฐกิจ (Macro Economics), ปริมาณนักท่องเที่ยว (mots.go.th), ราคาน้ำมัน, และสภาพอากาศ (อุณหภูมิ/ฝน) โดยแมพผ่านวันที่ (`date`) และคลัสเตอร์พื้นที่ (`dbscan_cluster_id`)

---

## Part 2: Heavy Internal Feature Engineering (The Core Engine)
ข้อมูลภายใน (Internal Data) คือหัวใจหลักที่จะทำคะแนนได้ดีที่สุด นี่คือสมการตัวแปรที่เราจะสร้าง:

1. **Strict Horizon Lags (แกนหลักของ Time Series):** 
   - เลื่อนข้อมูลอดีต (Shift) ตาม Horizon อย่างเคร่งครัด เพื่อป้องกันการโกง/Leakage
   - สำหรับ Horizon `H` (1, 7, หรือ 30): สร้าง `Lag_H`, `Lag_{H+1}`, `Lag_{H+6}`, `Lag_{H+7}`, `Lag_{H+14}`, `Lag_{H+28}`
2. **Rolling Statistics (ความจำและความผันผวนของสาขา):**
   - สร้าง `Rolling_Mean_7`, `Rolling_Mean_14`, `Rolling_Std_14`, `Rolling_Max_30` โดยคำนวณบนข้อมูลที่ถูกทำ `Lag_H` แล้วเท่านั้น
3. **Hierarchical Features (ดึงพฤติกรรมภาพรวม):**
   - คำนวณ `Average_Sales_by_Format_and_Date`: ยอดขายเฉลี่ยของกลุ่ม "ปั๊มน้ำมัน", "ห้าง", "Standalone" ในวันนั้นๆ ในอดีต (Shifted)
4. **Promo Cannibalization Flags (จับผิดโปรโมชั่น):**
   - `is_merch_cannibalized_by_drink`: 1 ถ้ามีโปรแจกของฟรี (แถมแก้ว) และหมวดปัจจุบันคือ Merchandise 
   - `cross_category_lift_multiplier`: ค่าสัมประสิทธิ์ที่ได้จากการทำ EDA (หมวด A จัดโปร ส่งผลให้หมวด B ขายดีขึ้นหรือแย่ลงกี่เปอร์เซ็นต์)
5. **Stockout Awareness:**
   - ดึงข้อมูลจากตาราง `INVENTORY` (ถ้ามี) สร้างฟีเจอร์ `is_stock_low_yesterday` เพื่อเตือนโมเดลถึงแนวโน้มของหมด

---

## Part 3: Spatial & Quantitative Deep EDA (การวิเคราะห์ด้วยสถิติและแผนที่)
ก่อนเทรนโมเดล เราต้องวิเคราะห์สมมติฐานทางธุรกิจ 5 ด้านเพื่อคัดกรอง Features:

1. **Geospatial Mapping (DBSCAN Clustering):**
   - ยิงพิกัด Lat/Lon เข้าอัลกอริทึม DBSCAN หา "ย่านความหนาแน่น"
   - สร้างฟีเจอร์ `competitor_density` (จำนวนสาขาคาเฟ่อเมซอนในรัศมี 3 กม.)
   - *Insight:* ดูผลกระทบของการแย่งลูกค้ากันเอง (Cannibalization by Proximity)
2. **The "EV & Gas" Effect:**
   - พล็อต Bar Chart/ANOVA เทียบยอดขายเฉลี่ย และ Basket Size ระหว่าง "ปั๊มที่มีตู้ชาร์จ EV" vs "ปั๊มธรรมดา" 
3. **Traffic Jam & Opening Hours (Rush Hour Dynamics):**
   - หาค่า Correlation เชิงตัวเลข (Pearson) ระหว่างดัชนีรถติดในกทม. กับยอดขายในสาขาที่ไม่ได้เปิด 24 ชม.
4. **Tourist vs Local (The "Fleeing Bangkok" Paradigm):**
   - เจาะลึก Time Series พล็อตยอดขายช่วง "เทศกาลสงกรานต์/ปีใหม่" ดูการดิ่งลงของสาขาในห้างกทม. และการพุ่งขึ้นของสาขาปั๊มน้ำมันขาออก
5. **22-Month YoY & Seasonality Validation:**
   - พล็อตทับซ้อน (Overlay) ยอดขายเดือน พ.ย.-ธ.ค. ของปี 2023 เทียบกับเดือนอื่นๆ และใช้ `seasonal_decompose` ถอดค่าฤดูกาลเพื่อจับเทรนด์เทศกาลส่งท้ายปี

---

## Part 4: Modeling & Validation Strategy (The Final Pipeline)

1. **Validation Split (จำลองการส่ง Kaggle):**
   - Train Set: ข้อมูลเดือนที่ 1 - 20
   - Validation Set: ข้อมูลเดือนที่ 21 - 22 (จำลองสถานการณ์จริงที่จะต้องทายเดือนที่ 23 - 24)
2. **Model Architecture:**
   - ใช้ **LightGBM** (หรือ CatBoost) เนื่องจากจัดการ Categorical Features และ Missing Values ได้ดีที่สุด
   - **Objective:** `Tweedie Loss` (Variance Power ~1.2) จัดการวันยอดตกหรือยอดขายเป็น 0 โดยไม่ทำให้ Bias
3. **Execution Rule:**
   - **Direct Forecasting:** สร้าง 3 โมเดลแยกอิสระ (Model 1d, Model 7d, Model 30d) 
   - **Rule of Honor:** ไม่มีการอ่าน `test/TRANSACTION.csv`, `ORDER.csv`, `INVENTORY.csv` โดยเด็ดขาด 

> [!IMPORTANT]
> **User Review Required**
> นี่คือ **"The Ultimate Blueprint"** ที่ผสมผสาน:
> 1. การเน้นหนักที่ Internal Data Features (Lags, Hierarchical, Promo Multipliers)
> 2. การแก้ปัญหาโลกจริง (สาขาที่หาไม่เจอบนเว็บ Cafe Amazon `is_unlisted_store`)
> 3. การทำ Spatial Clustering (DBSCAN) และ External Macro Boosters
> 
> ถ้านาย **Approve** ความละเอียดและครอบคลุมระดับ Masterpiece ของแผนฉบับสมบูรณ์นี้แล้ว ผมพร้อมเต็มที่ที่จะเริ่มเขียนสคริปต์ Python ในแต่ละขั้นตอนมารอรับข้อมูลเลยครับ! ลุยกันเลยไหมครับ?
