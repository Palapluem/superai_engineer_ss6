# ExpresSo Demand Forecasting - Deep EDA Add-on (Codex)

รายงานนี้ต่อยอดจาก `EDA_SUMMARY_จากเพื่อน.md` และ notebook เดิม โดยยึดข้อมูลภายใน dataset เป็นหลัก และคำนวณ effect หลายตัวแบบ normalized เทียบกับ baseline `store × category × day_of_week` เพื่อลด bias จากร้าน/หมวดที่ยอดขายสูงอยู่แล้ว

## 0) Ground Rules จากโจทย์

- Target ที่ถูกต้องคือ `TRANSACTION.units_sold` join `ORDER` และ `PRODUCT` แล้ว aggregate เป็น `store_id × category × date`
- Forecast window คือ `2024-11-01` ถึง `2024-12-31`; public = November, private = December; metric = MAE
- ห้ามใช้ `INVENTORY.units_sold` เป็น target; ใช้ได้เป็น stockout/context feature และต้องระวัง horizon leakage
- `PROMOTION`, `LOCAL_EVENT`, `DATE_DIM`, `STORE`, `PRODUCT` เป็น lookup ที่รู้ล่วงหน้าได้ แต่ lag/rolling จากยอดขายต้อง shift ตาม horizon

## 1) Data Audit

| table | rows | cols |
| --- | --- | --- |
| test/DATE_DIM.csv | 731 | 12 |
| test/LOCAL_EVENT.csv | 1,440 | 5 |
| test/PRODUCT.csv | 60 | 7 |
| test/PROMOTION.csv | 31,314 | 10 |
| test/STORE.csv | 20 | 8 |
| train/CUSTOMER.csv | 6,000 | 5 |
| train/DATE_DIM.csv | 731 | 12 |
| train/INVENTORY.csv | 804,000 | 9 |
| train/LOCAL_EVENT.csv | 1,440 | 5 |
| train/ORDER.csv | 1,376,133 | 7 |
| train/PRODUCT.csv | 60 | 7 |
| train/PROMOTION.csv | 31,314 | 10 |
| train/STORE.csv | 20 | 8 |
| train/TRANSACTION.csv | 2,858,050 | 6 |

- Train order date range: `2023-01-01` to `2024-10-31` = 670 days
- Full target grid after zero-fill: `20 stores × 7 categories × 670 days = 93,800` rows
- Forecast grid: `20 × 7 × 61 × 3 horizons = 25,620` rows

Lookup train/test byte check:
| lookup | sha256 status | train rows | test rows |
| --- | --- | --- | --- |
| DATE_DIM.csv | same | 731 | 731 |
| STORE.csv | same | 20 | 20 |
| PRODUCT.csv | same | 60 | 60 |
| PROMOTION.csv | same | 31,314 | 31,314 |
| LOCAL_EVENT.csv | same | 1,440 | 1,440 |

## 2) Target Shape และ Demand Concentration

| metric | value |
| --- | --- |
| mean | 42.1 |
| median | 23.0 |
| p75 | 47.0 |
| p90 | 104.0 |
| p95 | 158.0 |
| p99 | 277.0 |
| max | 1,607 |
| zero cell-days | 1,460 |

| category | total units | avg units/store/day | revenue | revenue/unit |
| --- | --- | --- | --- | --- |
| Coffee | 1,942,678 | 145.0 | 100,514,913.8 | 51.7 |
| Tea | 609,945 | 45.5 | 26,913,414.8 | 44.1 |
| Bakery | 476,108 | 35.5 | 20,557,950.7 | 43.2 |
| Savory Bakery | 376,799 | 28.1 | 17,955,232.7 | 47.7 |
| Chocolate & Milk | 277,784 | 20.7 | 13,410,454.2 | 48.3 |
| Juice & Smoothie | 167,674 | 12.5 | 8,705,314.8 | 51.9 |
| Merchandise | 98,361 | 7.3 | 21,342,516.9 | 217.0 |

**Insight:** distribution หนักขวาชัดมาก จึงควรดู MAE แยก category/store และทำ model หรือ post-process แยกหมวด โดย Coffee เป็น volume anchor ส่วน Merchandise เป็น low-volume/intermittent tail ที่ MAE ดิบเล็กแต่ ratio error ง่าย

## 3) Store, Location, Service, Capacity

| store | type | units/day | orders/day | units/staff/day | orders/seat/day | cap | staff | hours | loyal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | mall | 537.5 | 190.9 | 41.3 | 2.7 | 71 | 13 | 11.0 | 297 |
| 10 | tourist | 469.3 | 170.5 | 52.1 | 2.5 | 69 | 9 | 14.0 | 294 |
| 9 | tourist | 429.4 | 155.9 | 61.3 | 2.5 | 62 | 7 | 14.0 | 284 |
| 5 | urban_residential | 407.0 | 146.4 | 37.0 | 2.4 | 60 | 11 | 14.0 | 335 |
| 15 | tourist | 397.7 | 143.8 | 44.2 | 2.5 | 58 | 9 | 14.0 | 298 |
| 2 | tourist | 355.9 | 129.1 | 50.8 | 2.6 | 49 | 7 | 14.0 | 277 |
| 14 | university | 333.4 | 112.5 | 41.7 | 1.8 | 61 | 8 | 13.0 | 291 |
| 8 | urban_residential | 313.9 | 113.2 | 34.9 | 2.2 | 51 | 9 | 14.0 | 307 |
| 3 | university | 309.3 | 104.1 | 44.2 | 2.0 | 52 | 7 | 13.0 | 280 |
| 17 | office | 308.8 | 90.4 | 34.3 | 1.8 | 50 | 9 | 13.0 | 280 |
| 19 | university | 298.8 | 100.9 | 33.2 | 2.2 | 46 | 9 | 14.0 | 288 |
| 20 | transit | 272.1 | 88.2 | 34.0 | 2.0 | 45 | 8 | 15.0 | 279 |
| 1 | university | 218.2 | 73.6 | 54.6 | 1.9 | 39 | 4 | 15.0 | 319 |
| 16 | transit | 215.3 | 69.9 | 21.5 | 1.3 | 54 | 10 | 15.0 | 283 |
| 7 | hospital | 203.6 | 71.0 | 50.9 | 1.9 | 37 | 4 | 13.5 | 336 |
| 18 | hospital | 181.6 | 63.1 | 45.4 | 2.3 | 28 | 4 | 13.5 | 307 |
| 11 | gas_station | 173.0 | 62.7 | 43.3 | 2.2 | 28 | 4 | 14.0 | 309 |
| 12 | gas_station | 169.5 | 61.4 | 33.9 | 2.2 | 28 | 5 | 14.0 | 292 |
| 4 | hospital | 163.9 | 56.9 | 32.8 | 2.1 | 27 | 5 | 13.5 | 320 |
| 13 | gas_station | 136.5 | 49.8 | 34.1 | 2.3 | 22 | 4 | 14.0 | 324 |

| neighborhood_type | stores | avg units/day/store | avg orders/day/store | avg seats | avg hours |
| --- | --- | --- | --- | --- | --- |
| mall | 1 | 537.5 | 190.9 | 71.0 | 11.0 |
| tourist | 4 | 413.1 | 149.8 | 59.5 | 14.0 |
| urban_residential | 2 | 360.5 | 129.8 | 55.5 | 14.0 |
| office | 1 | 308.8 | 90.4 | 50.0 | 13.0 |
| university | 4 | 289.9 | 97.8 | 49.5 | 13.8 |
| transit | 2 | 243.7 | 79.0 | 49.5 | 15.0 |
| hospital | 3 | 183.0 | 63.7 | 30.7 | 13.5 |
| gas_station | 3 | 159.7 | 58.0 | 26.0 | 14.0 |

| correlation vs units/day | r |
| --- | --- |
| seating_capacity | 0.9 |
| staff_count | 0.8 |
| operating_hours | -0.4 |
| loyal_customer_count | -0.3 |

**Insight:** capacity และ staff_count เป็น demand proxy ที่แข็งกว่า operating hours เพียว ๆ. `loyal_customer_count` ใน registry นี้ใกล้กันมากและ correlation ไม่ดี จึงควรใช้เป็น weak/static context มากกว่า driver หลัก. ควรสร้าง `units_per_staff_lag`, `orders_per_seat_lag`, `store_type × weekend`, `store_type × hour_peak` และ cluster store จาก sales mix + operational profile

## 4) Calendar, Weather Proxy, Holiday, Payday

| factor | raw off | raw on | raw lift | rel off | rel on | rel lift |
| --- | --- | --- | --- | --- | --- | --- |
| Weekend | 40.7 | 45.6 | 12.1 | 1.0 | 1.0 | 0.0 |
| Holiday | 41.2 | 64.8 | 57.2 | 1.0 | 1.5 | 56.6 |
| Payday | 41.4 | 51.6 | 24.5 | 1.0 | 1.2 | 25.2 |
| Rainy season | 45.9 | 32.0 | -30.4 | 1.1 | 0.8 | -28.6 |
| School break | 38.9 | 47.7 | 22.7 | 0.9 | 1.1 | 20.8 |

### Weekend by category (baseline-normalized)
| category | off rel | on rel | lift |
| --- | --- | --- | --- |
| Savory Bakery | 1.0 | 1.0 | 0.0 |
| Chocolate & Milk | 1.0 | 1.0 | 0.0 |
| Tea | 1.0 | 1.0 | -0.0 |
| Juice & Smoothie | 1.0 | 1.0 | -0.0 |
| Merchandise | 1.0 | 1.0 | -0.0 |
| Coffee | 1.0 | 1.0 | -0.0 |
| Bakery | 1.0 | 1.0 | -0.0 |

### Holiday by category (baseline-normalized)
| category | off rel | on rel | lift |
| --- | --- | --- | --- |
| Merchandise | 1.0 | 1.6 | 58.7 |
| Bakery | 1.0 | 1.5 | 58.2 |
| Tea | 1.0 | 1.5 | 57.5 |
| Coffee | 1.0 | 1.5 | 56.8 |
| Juice & Smoothie | 1.0 | 1.5 | 55.7 |
| Savory Bakery | 1.0 | 1.5 | 55.7 |
| Chocolate & Milk | 1.0 | 1.5 | 53.7 |

### Payday by category (baseline-normalized)
| category | off rel | on rel | lift |
| --- | --- | --- | --- |
| Merchandise | 1.0 | 1.3 | 28.5 |
| Savory Bakery | 1.0 | 1.2 | 26.4 |
| Chocolate & Milk | 1.0 | 1.2 | 25.0 |
| Tea | 1.0 | 1.2 | 24.3 |
| Juice & Smoothie | 1.0 | 1.2 | 24.3 |
| Bakery | 1.0 | 1.2 | 24.1 |
| Coffee | 1.0 | 1.2 | 24.0 |

### Rainy season by category (baseline-normalized)
| category | off rel | on rel | lift |
| --- | --- | --- | --- |
| Juice & Smoothie | 1.1 | 0.8 | -27.2 |
| Tea | 1.1 | 0.8 | -27.9 |
| Chocolate & Milk | 1.1 | 0.8 | -28.6 |
| Merchandise | 1.1 | 0.8 | -28.8 |
| Coffee | 1.1 | 0.8 | -29.0 |
| Bakery | 1.1 | 0.8 | -29.2 |
| Savory Bakery | 1.1 | 0.8 | -29.4 |

Top holiday/category normalized spikes:
| holiday | category | holiday dates | rel to same store-cat-DOW |
| --- | --- | --- | --- |
| วันพ่อแห่งชาติ | Coffee | 1 | 2.0 |
| วันรัฐธรรมนูญ | Savory Bakery | 1 | 2.0 |
| วันรัฐธรรมนูญ | Bakery | 1 | 2.0 |
| วันสงกรานต์ | Bakery | 6 | 2.0 |
| วันสงกรานต์ | Merchandise | 6 | 2.0 |
| วันพ่อแห่งชาติ | Tea | 1 | 2.0 |
| วันรัฐธรรมนูญ | Tea | 1 | 2.0 |
| วันรัฐธรรมนูญ | Coffee | 1 | 2.0 |
| วันสงกรานต์ | Savory Bakery | 6 | 2.0 |
| วันรัฐธรรมนูญ | Merchandise | 1 | 2.0 |
| วันพ่อแห่งชาติ | Bakery | 1 | 2.0 |
| วันสงกรานต์ | Juice & Smoothie | 6 | 2.0 |
| วันพ่อแห่งชาติ | Juice & Smoothie | 1 | 2.0 |
| วันสงกรานต์ | Coffee | 6 | 2.0 |
| วันพ่อแห่งชาติ | Merchandise | 1 | 2.0 |

Forecast-window known calendar:
| date | day | holiday_name |
| --- | --- | --- |
| 2024-12-05 | Thursday | วันพ่อแห่งชาติ |
| 2024-12-10 | Tuesday | วันรัฐธรรมนูญ |

- Forecast paydays: 2024-11-15, 2024-11-30, 2024-12-15, 2024-12-31

**Insight:** `is_rainy_season` เป็น weather proxy แบบหยาบ ไม่ใช่ rainfall จริง. ถ้าเพิ่ม external weather ให้ใช้ forecast/observed ที่ public และทำเป็น feature ที่รู้ได้ ณ decision day; แต่ใน dataset ตอนนี้ interaction ระหว่าง rainy season × category/store type ยังให้ signal ที่ไม่ควรทิ้ง

## 5) Promotion, Price, Marketing Channel

| promo | raw no | raw yes | raw lift | rel no | rel yes | rel lift |
| --- | --- | --- | --- | --- | --- | --- |
| any promo | 36.0 | 51.4 | 42.6 | 0.9 | 1.1 | 20.7 |

| category | train promo cell-days | rel no promo | rel on promo | rel lift |
| --- | --- | --- | --- | --- |
| Coffee | 5,940 | 0.9 | 1.1 | 24.2 |
| Bakery | 5,256 | 0.9 | 1.1 | 21.3 |
| Merchandise | 5,103 | 0.9 | 1.1 | 21.2 |
| Chocolate & Milk | 5,303 | 0.9 | 1.1 | 21.1 |
| Savory Bakery | 5,161 | 0.9 | 1.1 | 20.6 |
| Tea | 5,475 | 0.9 | 1.1 | 19.4 |
| Juice & Smoothie | 4,776 | 0.9 | 1.1 | 17.5 |

Promotion design distribution:
| promo_type | rows | avg discount |
| --- | --- | --- |
| แต้มx2 | 12,900 | 0.0 |
| สมาชิกใหม่ | 10,260 | 16.8 |
| ลดราคา | 4,267 | 23.9 |
| ซื้อ1แถม1 | 2,809 | 50.0 |
| ชุดคู่ลด | 1,078 | 22.6 |

Top promo-type × category normalized effects:
| promo_type | category | cell-days | rel |
| --- | --- | --- | --- |
| ซื้อ1แถม1 | Savory Bakery | 1,314 | 1.3 |
| ซื้อ1แถม1 | Tea | 1,371 | 1.3 |
| ซื้อ1แถม1 | Coffee | 1,497 | 1.3 |
| ซื้อ1แถม1 | Merchandise | 1,255 | 1.3 |
| ซื้อ1แถม1 | Chocolate & Milk | 1,302 | 1.3 |
| ซื้อ1แถม1 | Bakery | 1,288 | 1.3 |
| ซื้อ1แถม1 | Juice & Smoothie | 1,006 | 1.3 |
| ลดราคา | Merchandise | 1,649 | 1.2 |
| ลดราคา | Chocolate & Milk | 1,807 | 1.2 |
| ลดราคา | Coffee | 1,979 | 1.2 |
| ลดราคา | Savory Bakery | 1,606 | 1.1 |
| ลดราคา | Bakery | 1,648 | 1.1 |
| ชุดคู่ลด | Bakery | 746 | 1.1 |
| ชุดคู่ลด | Merchandise | 612 | 1.1 |
| ลดราคา | Tea | 1,868 | 1.1 |
| ลดราคา | Juice & Smoothie | 1,566 | 1.1 |
| ชุดคู่ลด | Coffee | 1,332 | 1.1 |
| ชุดคู่ลด | Savory Bakery | 720 | 1.1 |
| ชุดคู่ลด | Chocolate & Milk | 720 | 1.1 |
| ชุดคู่ลด | Juice & Smoothie | 477 | 1.1 |

Marketing channel proxy:
| channel | category | cell-days | rel |
| --- | --- | --- | --- |
| email+social | Coffee | 2,101 | 1.1 |
| email+social | Savory Bakery | 1,779 | 1.1 |
| email+social | Chocolate & Milk | 1,799 | 1.1 |
| social_only | Merchandise | 1,098 | 1.1 |
| email+social | Bakery | 1,786 | 1.1 |
| email+social | Merchandise | 1,696 | 1.1 |
| email_only | Coffee | 1,744 | 1.1 |
| social_only | Chocolate & Milk | 1,106 | 1.1 |
| email_only | Merchandise | 1,594 | 1.1 |
| social_only | Savory Bakery | 1,120 | 1.1 |
| email_only | Bakery | 1,601 | 1.1 |
| email_only | Tea | 1,672 | 1.1 |
| social_only | Bakery | 1,124 | 1.1 |
| social_only | Juice & Smoothie | 987 | 1.1 |
| email+social | Juice & Smoothie | 1,599 | 1.1 |
| social_only | Tea | 1,188 | 1.1 |
| email+social | Tea | 1,886 | 1.1 |
| email_only | Savory Bakery | 1,572 | 1.1 |
| social_only | Coffee | 1,290 | 1.1 |
| email_only | Chocolate & Milk | 1,632 | 1.1 |

Transaction discount-applied distribution:
| discount_applied bin | lines | units | unit share |
| --- | --- | --- | --- |
| 0% | 2,402,150 | 3,319,296 | 84.0% |
| 1-10% | 81,939 | 113,354 | 2.9% |
| 11-20% | 183,159 | 253,087 | 6.4% |
| 21-30% | 75,632 | 104,671 | 2.7% |
| 31-40% | 26,480 | 36,601 | 0.9% |
| 41%+ | 88,690 | 122,340 | 3.1% |

- Promo schedule exposure train: 37,014/93,800 cell-days = 39.5%
- Promo schedule exposure forecast: 2,887/8,540 cell-days = 33.8%

| category | train promo % cell-days | forecast promo % cell-days | pp change |
| --- | --- | --- | --- |
| Chocolate & Milk | 39.6 | 36.0 | -3.6 |
| Savory Bakery | 38.5 | 34.8 | -3.7 |
| Juice & Smoothie | 35.6 | 30.2 | -5.5 |
| Coffee | 44.3 | 38.2 | -6.1 |
| Tea | 40.9 | 34.2 | -6.7 |
| Bakery | 39.2 | 32.2 | -7.0 |
| Merchandise | 38.1 | 31.1 | -7.0 |

Merchandise cannibalization proxy during Coffee/Tea promo: rel no drink promo = 0.906, rel drink promo = 1.113, effect = 22.9%

**Insight:** อย่าดู promo เป็น binary เดียว ให้แตกเป็น `max_discount`, `promo_type`, `email/social`, `category promo exposure`, `days_since/until promo`, และ `other drink promo active` เพื่อจับ cannibalization/cross-sell

## 6) Events, Festivals, Local Demand Shocks

| event | raw mean | rel mean | lift vs no-event |
| --- | --- | --- | --- |
| no event | 39.9 | 1.0 | 0 |
| has event | 63.1 | 1.4 | 49.7 |

| category | rel no event | rel event | lift |
| --- | --- | --- | --- |
| Juice & Smoothie | 1.0 | 1.5 | 52.7 |
| Tea | 1.0 | 1.4 | 50.6 |
| Savory Bakery | 1.0 | 1.4 | 50.3 |
| Bakery | 1.0 | 1.4 | 49.6 |
| Chocolate & Milk | 1.0 | 1.4 | 49.5 |
| Coffee | 1.0 | 1.4 | 49.0 |
| Merchandise | 1.0 | 1.4 | 46.2 |

Top event-type × category normalized effects:
| event_type | category | cell-days | rel |
| --- | --- | --- | --- |
| food_festival | Juice & Smoothie | 198 | 2.0 |
| food_festival | Savory Bakery | 198 | 2.0 |
| food_festival | Merchandise | 198 | 2.0 |
| food_festival | Bakery | 198 | 2.0 |
| food_festival | Tea | 198 | 2.0 |
| food_festival | Chocolate & Milk | 198 | 2.0 |
| food_festival | Coffee | 198 | 2.0 |
| music_festival | Chocolate & Milk | 129 | 1.6 |
| music_festival | Juice & Smoothie | 129 | 1.6 |
| music_festival | Merchandise | 129 | 1.6 |
| music_festival | Tea | 129 | 1.6 |
| music_festival | Coffee | 129 | 1.6 |
| concert | Juice & Smoothie | 110 | 1.6 |
| music_festival | Savory Bakery | 129 | 1.6 |
| music_festival | Bakery | 129 | 1.6 |
| concert | Savory Bakery | 110 | 1.6 |
| concert | Chocolate & Milk | 110 | 1.5 |
| concert | Bakery | 110 | 1.5 |
| concert | Tea | 110 | 1.5 |
| concert | Coffee | 110 | 1.5 |

Forecast-window event count by type:
| event_type | train count | forecast count |
| --- | --- | --- |
| book_fair | 120 | 14 |
| concert | 111 | 13 |
| convention | 130 | 13 |
| cultural | 216 | 18 |
| food_festival | 201 | 16 |
| market | 260 | 14 |
| music_festival | 130 | 13 |
| sports | 161 | 10 |

First forecast-window events:
| date | store | event_type |
| --- | --- | --- |
| 2024-11-01 | 1 | convention |
| 2024-11-12 | 1 | book_fair |
| 2024-11-18 | 1 | music_festival |
| 2024-11-19 | 1 | book_fair |
| 2024-12-02 | 1 | concert |
| 2024-12-13 | 1 | convention |
| 2024-11-01 | 2 | music_festival |
| 2024-11-19 | 2 | cultural |
| 2024-12-08 | 2 | market |
| 2024-12-29 | 2 | food_festival |
| 2024-11-21 | 3 | concert |
| 2024-12-10 | 3 | market |
| 2024-12-28 | 3 | cultural |
| 2024-12-30 | 3 | sports |
| 2024-11-24 | 4 | food_festival |
| 2024-12-23 | 4 | book_fair |
| 2024-12-31 | 4 | music_festival |
| 2024-11-01 | 5 | music_festival |
| 2024-11-21 | 5 | food_festival |
| 2024-12-06 | 5 | music_festival |
| 2024-12-26 | 5 | book_fair |
| 2024-11-01 | 6 | convention |
| 2024-11-02 | 6 | cultural |
| 2024-11-17 | 6 | concert |
| 2024-11-28 | 6 | sports |
| 2024-12-07 | 6 | food_festival |
| 2024-12-11 | 6 | food_festival |
| 2024-11-06 | 7 | food_festival |
| 2024-11-18 | 7 | music_festival |
| 2024-12-01 | 7 | concert |

**Insight:** event feature ควรมี `has_event`, `event_type`, `event_count_store_day`, `store_type × event_type`, และควร normalize/validate ด้วย same-store weekday baseline เพราะ event ถูกสุ่มไปตกที่ร้าน volume ต่างกัน

## 7) Traffic Proxy และ Hour-of-Day Behavior

| hour | orders | order share | avg units/order |
| --- | --- | --- | --- |
| 6 | 33,962 | 2.5% | 2.8 |
| 7 | 72,838 | 5.3% | 3.1 |
| 8 | 96,954 | 7.0% | 3.2 |
| 9 | 104,502 | 7.6% | 3.1 |
| 10 | 113,242 | 8.2% | 2.9 |
| 11 | 112,660 | 8.2% | 2.8 |
| 12 | 136,486 | 9.9% | 2.7 |
| 13 | 115,533 | 8.4% | 2.8 |
| 14 | 98,094 | 7.1% | 3.0 |
| 15 | 104,778 | 7.6% | 2.9 |
| 16 | 99,775 | 7.3% | 2.8 |
| 17 | 90,382 | 6.6% | 2.8 |
| 18 | 76,303 | 5.5% | 2.7 |
| 19 | 57,922 | 4.2% | 2.7 |
| 20 | 36,431 | 2.6% | 2.7 |
| 21 | 26,271 | 1.9% | 2.7 |

| neighborhood | top order hours |
| --- | --- |
| gas_station | 7:00 (16,842), 8:00 (14,498), 9:00 (11,848) |
| hospital | 8:00 (13,688), 9:00 (12,477), 12:00 (12,386) |
| mall | 12:00 (14,260), 13:00 (13,612), 16:00 (12,380) |
| office | 8:00 (8,282), 9:00 (7,353), 15:00 (6,253) |
| tourist | 12:00 (41,744), 11:00 (41,298), 10:00 (38,266) |
| transit | 8:00 (14,293), 7:00 (13,242), 18:00 (12,151) |
| university | 12:00 (32,788), 10:00 (31,326), 11:00 (27,244) |
| urban_residential | 8:00 (15,535), 12:00 (15,048), 13:00 (13,681) |

**Insight:** traffic ไม่มี column ตรง ๆ แต่ใช้ proxy ได้จาก `hour`, `neighborhood_type`, `open_time`, `is_weekend`, `event`, `payday`, และ store type เช่น transit/gas_station. สำหรับ daily forecasting ให้ทำ aggregate จากอดีต เช่น `lag_morning_order_share`, `lag_peak_hour_units`, `weekday_commute_store_type`

## 8) SKU, Menu, Seasonal/Limited Edition

| product_id | name | category | serve | price | units | revenue | seasonal | limited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | ลาเต้อเมซอน | Coffee | เย็น | 60.0 | 128,348 | 7,370,520.0 |  |  |
| 8 | คาปูชิโน่ | Coffee | เย็น | 60.0 | 128,259 | 7,373,580.0 |  |  |
| 13 | แบล็คคอฟฟี่ | Coffee | เย็น | 50.0 | 128,065 | 6,161,735.0 |  |  |
| 2 | อเมซอน | Coffee | เย็น | 55.0 | 127,548 | 6,723,675.8 |  |  |
| 5 | เอสเปรสโซ | Coffee | เย็น | 50.0 | 127,405 | 6,131,180.0 |  |  |
| 6 | เอสเปรสโซ | Coffee | ปั่น | 55.0 | 125,014 | 6,599,958.8 |  |  |
| 3 | อเมซอน | Coffee | ปั่น | 60.0 | 123,558 | 7,118,508.0 |  |  |
| 11 | มอคค่า | Coffee | ร้อน | 55.0 | 113,031 | 5,972,851.5 |  |  |
| 7 | คาปูชิโน่ | Coffee | ร้อน | 50.0 | 112,082 | 5,398,300.0 |  |  |
| 9 | ลาเต้อเมซอน | Coffee | ร้อน | 50.0 | 111,843 | 5,352,702.5 |  |  |
| 12 | แบล็คคอฟฟี่ | Coffee | ร้อน | 40.0 | 111,335 | 4,283,258.0 |  |  |
| 1 | อเมซอน | Coffee | ร้อน | 45.0 | 110,597 | 4,767,093.0 |  |  |

| category | seasonal units | regular units | seasonal unit share |
| --- | --- | --- | --- |
| Coffee | 385,200 | 1,557,478 | 19.8% |
| Tea | 56,974 | 552,971 | 9.3% |
| Chocolate & Milk | 57,433 | 220,351 | 20.7% |
| Juice & Smoothie | 0 | 167,674 | 0.0% |
| Bakery | 56,977 | 419,131 | 12.0% |
| Savory Bakery | 0 | 376,799 | 0.0% |
| Merchandise | 0 | 98,361 | 0.0% |

Trend YoY Jan-Oct 2024 vs Jan-Oct 2023:
| category | 2023 avg units/cell | 2024 avg units/cell | YoY |
| --- | --- | --- | --- |
| Coffee | 139.6 | 145.3 | 4.1 |
| Tea | 44.0 | 45.6 | 3.7 |
| Chocolate & Milk | 19.9 | 20.8 | 4.5 |
| Juice & Smoothie | 12.1 | 12.5 | 3.9 |
| Bakery | 34.2 | 35.7 | 4.5 |
| Savory Bakery | 27.1 | 28.1 | 3.6 |
| Merchandise | 7.1 | 7.4 | 3.9 |

Recent drift Aug-Oct 2024 vs May-Jul 2024:
| category | May-Jul avg units/cell | Aug-Oct avg units/cell | change |
| --- | --- | --- | --- |
| Coffee | 126.8 | 133.0 | 4.8 |
| Tea | 39.8 | 41.7 | 4.6 |
| Chocolate & Milk | 18.2 | 19.1 | 5.5 |
| Juice & Smoothie | 11.1 | 11.5 | 4.0 |
| Bakery | 31.0 | 32.6 | 4.9 |
| Savory Bakery | 24.4 | 25.9 | 5.9 |
| Merchandise | 6.3 | 6.8 | 8.1 |

**Insight:** โจทย์ forecast category แต่ SKU still matters เพราะ SKU mix, seasonal/limited flags, และ stockout เกิดที่ product level. ควร aggregate SKU signals ขึ้นเป็น category เช่น `seasonal_unit_share_28d`, `limited_sku_promo_count`, `top_sku_stockout_rate`

## 9) Customer Segments, Frequency, Density, Behavior

| segment | orders | units | revenue | units/order | revenue/order |
| --- | --- | --- | --- | --- | --- |
| walk_in | 825,292 | 2,367,416.0 | 125,501,312.0 | 2.9 | 152.1 |
| member | 550,841 | 1,581,933.0 | 83,898,485.9 | 2.9 | 152.3 |

- Walk-in orders: 825,292/1,376,133 = 60.0%; member/customer-id orders = 40.0%
- Active registered customers appearing in orders: 4,353/6,000; one-time = 33.8%, repeat >=2 = 66.2%, high-frequency >=10 = 18.5%
- Coffee attachment: Bakery in coffee orders = 19.8%, Savory Bakery in coffee orders = 15.4%

Top basket category combinations:
| basket categories | orders | share |
| --- | --- | --- |
| Coffee | 469,511 | 34.1% |
| Tea | 118,280 | 8.6% |
| Coffee + Tea | 103,102 | 7.5% |
| Coffee + Bakery | 84,661 | 6.2% |
| Coffee + Savory Bakery | 58,254 | 4.2% |
| Chocolate & Milk | 48,068 | 3.5% |
| Coffee + Chocolate & Milk | 38,755 | 2.8% |
| Bakery | 33,172 | 2.4% |
| Savory Bakery | 27,859 | 2.0% |
| Juice & Smoothie | 27,542 | 2.0% |
| Coffee + Bakery + Savory Bakery | 26,082 | 1.9% |
| Coffee + Juice & Smoothie | 21,738 | 1.6% |

Payment behavior:
| payment | orders | share | revenue | units | revenue/order |
| --- | --- | --- | --- | --- | --- |
| QR/PromptPay | 530,266 | 38.5% | 81,001,628.7 | 1,526,891 | 152.8 |
| กระเป๋าเงินดิจิทัล | 383,558 | 27.9% | 59,062,084.1 | 1,112,753 | 154.0 |
| บัตรเครดิต | 294,278 | 21.4% | 44,352,018.4 | 838,556 | 150.7 |
| เงินสด | 168,031 | 12.2% | 24,984,066.6 | 471,149 | 148.7 |

Registered customer density by preferred store:
| store | type | loyal customers | same-neighborhood customers | local share |
| --- | --- | --- | --- | --- |
| 7 | hospital | 336 | 336 | 100.0% |
| 5 | urban_residential | 335 | 335 | 100.0% |
| 13 | gas_station | 324 | 324 | 100.0% |
| 4 | hospital | 320 | 320 | 100.0% |
| 1 | university | 319 | 319 | 100.0% |
| 11 | gas_station | 309 | 309 | 100.0% |
| 18 | hospital | 307 | 307 | 100.0% |
| 8 | urban_residential | 307 | 307 | 100.0% |
| 15 | tourist | 298 | 298 | 100.0% |
| 6 | mall | 297 | 297 | 100.0% |
| 10 | tourist | 294 | 294 | 100.0% |
| 12 | gas_station | 292 | 292 | 100.0% |
| 14 | university | 291 | 291 | 100.0% |
| 19 | university | 288 | 288 | 100.0% |
| 9 | tourist | 284 | 284 | 100.0% |
| 16 | transit | 283 | 283 | 100.0% |
| 3 | university | 280 | 280 | 100.0% |
| 17 | office | 280 | 280 | 100.0% |
| 20 | transit | 279 | 279 | 100.0% |
| 2 | tourist | 277 | 277 | 100.0% |

**Insight:** ไม่มี review text ใน dataset ดังนั้น `พฤติกรรมและ review` ต้องใช้ behavior proxy: repeat frequency, basket mix, payment, member/walk-in ratio, and basket attachment. ส่วน `home_neighborhood_type` ตรงกับ preferred store 100% ใน registry นี้ จึงไม่ช่วยแยก local/non-local เท่าไร. Feature ที่น่าทำคือ rolling member ratio/store, repeat customer density, coffee-bakery attachment lag, payment mix lag

## 10) Inventory, Stockout, Recorded Demand

- Inventory row stockout rate: 52,939/804,000 = 6.6%
- Inventory.units_sold exact match vs transaction SKU-store-day units: 103,382/804,000 = 12.9%; mean absolute diff = 3.79 units
| store | type | stockout rows | inv rows | stockout % |
| --- | --- | --- | --- | --- |
| 19 | university | 2,799 | 40,200 | 7.0 |
| 9 | tourist | 2,789 | 40,200 | 6.9 |
| 8 | urban_residential | 2,716 | 40,200 | 6.8 |
| 17 | office | 2,713 | 40,200 | 6.7 |
| 20 | transit | 2,708 | 40,200 | 6.7 |
| 6 | mall | 2,703 | 40,200 | 6.7 |
| 10 | tourist | 2,691 | 40,200 | 6.7 |
| 4 | hospital | 2,688 | 40,200 | 6.7 |
| 1 | university | 2,658 | 40,200 | 6.6 |
| 12 | gas_station | 2,650 | 40,200 | 6.6 |
| 5 | urban_residential | 2,646 | 40,200 | 6.6 |
| 11 | gas_station | 2,628 | 40,200 | 6.5 |
| 3 | university | 2,625 | 40,200 | 6.5 |
| 2 | tourist | 2,604 | 40,200 | 6.5 |
| 7 | hospital | 2,601 | 40,200 | 6.5 |
| 15 | tourist | 2,591 | 40,200 | 6.4 |
| 16 | transit | 2,585 | 40,200 | 6.4 |
| 13 | gas_station | 2,543 | 40,200 | 6.3 |
| 18 | hospital | 2,538 | 40,200 | 6.3 |
| 14 | university | 2,463 | 40,200 | 6.1 |

| category | stockout rows | inv rows | stockout % |
| --- | --- | --- | --- |
| Juice & Smoothie | 6,052 | 67,000 | 9.0 |
| Tea | 9,733 | 120,600 | 8.1 |
| Chocolate & Milk | 8,169 | 107,200 | 7.6 |
| Coffee | 17,062 | 227,800 | 7.5 |
| Savory Bakery | 5,899 | 93,800 | 6.3 |
| Bakery | 5,415 | 93,800 | 5.8 |
| Merchandise | 609 | 93,800 | 0.6 |

Store-category day with any product stockout: rel no stockout = 0.938, rel stockout = 1.118, apparent effect = 19.2%

**Insight:** stockout rows are censored/recorded sales, not true demand. สำหรับ modeling ให้ลอง `sample_weight` ต่ำลงบน stockout-exposed rows และทำ feature `lag_stockout_rate`, `stockout_count_top_skus`, `days_since_stockout` แยกตาม horizon

## 11) Forecast Window Drift Watchlist

- Public LB = November 2024, Private LB = December 2024: อย่าจูนจาก Nov อย่างเดียว
- Calendar known in forecast window มี holiday/payday/event/promo exposure ที่ควร join จาก lookup ได้ตรง ๆ
- Compare promo exposure table ใน Section 5 เพื่อดูว่า category ไหนมี campaign density ใน Nov-Dec สูง/ต่ำกว่าประวัติ
- December มี holiday effect และ tourist/high-season possibility จึงควร validate แบบ time split ที่มี late-year holdout เช่น Oct 2024 และ backtest Nov/Dec-like windows จาก 2023

## 12) Feature Backlog ที่ควรเพิ่มใน Notebook

| block | features | why |
| --- | --- | --- |
| Leakage-safe lag | `lag_7/14/28/35/60/90`, rolling shifted by horizon | core signal |
| Baseline-normalized target | `y / mean(store,category,dow)` as diagnostic/feature | separates structural demand from shocks |
| Store profile | `store_type`, capacity, staff, hours, drive_through, age, loyal density | location/service/capacity |
| Calendar | DOW, month, holiday_name, payday, school_break, rainy_season | seasonality and event spikes |
| Promo | active, max_discount, promo_type, channels, days_until/since, category exposure | planned future lookup |
| Cross-promo | drink promo active while Merchandise target | cannibalization/cross-sell |
| Event | has_event, event_type, event_count, store_type×event_type | local shocks |
| Stockout | lag stockout rate/product/category, any_stockout cell | recorded demand censoring |
| Customer | rolling member ratio, repeat density, basket attachment | behavior/frequency |
| Traffic proxy | hour mix lag, peak share, store_type×weekend | commute/location behavior |
| SKU mix | seasonal share, limited SKU count, top SKU stockout/promo | category-level hidden composition |
| Drift | recent 28/56/90-day trend, YoY same month, forecast promo/event density | public/private robustness |

## 13) External Source Ideas

External data should be used only if public, cited, and available at prediction time. Strong candidates:
| source family | what to borrow | EDA use |
| --- | --- | --- |
| Kaggle/Rossmann Store Sales | retail daily sales affected by promotion, holidays, seasonality, locality | supports store/calendar/promo EDA framing |
| M5 Forecasting | hierarchical daily retail series with price, promotions, day-of-week, special events | supports global model + exogenous variables |
| Kaggle Coffee Shop Sales datasets | hour-of-day, product, payment, revenue style coffee transaction analysis | supports coffee behavior dashboard ideas |
| Thai Meteorological Department / Thailand.go.th | rainy season roughly mid-May to mid-October | validate `is_rainy_season` and optional rainfall feature |
| TomTom Traffic Index Bangkok | rush-hour congestion and Bangkok mobility patterns | optional city-level traffic proxy; use cautiously |
| Bank of Thailand / public holiday calendars | official/special holiday announcements | holiday_name QA and long-weekend features |

Suggested citation URLs:

- Rossmann Store Sales Kaggle: https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales
- Store Item Demand Forecasting Kaggle: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Coffee Shop Sales Kaggle: https://www.kaggle.com/datasets/xavierberge/coffee-shop-sales-dataset
- M5 Competition University of Nicosia: https://www.unic.ac.cy/iff/research/forecasting/m-competitions/m5/
- TMD Climate: https://www.tmd.go.th/en/climate/climateSubpage
- Thailand.go.th Seasons: https://www.thailand.go.th/public/useful-information-detail/009_142
- TomTom Bangkok Traffic Index: https://www.tomtom.com/traffic-index/bangkok-traffic/
- Bank of Thailand Financial Institution Holidays: https://www.bot.or.th/th/financial-institutions-holiday.html

## 14) Notebook Presentation Checklist

- Add charts with raw mean and normalized lift side by side for promo/event/holiday/weather
- Add store profile scatter: capacity/staff/hours/loyal density vs avg units/day with labels
- Add heatmaps: store × category demand, event_type × category lift, promo_type × category lift
- Add forecast-window calendar strip: Nov-Dec holiday/payday/event/promo count by date
- Add MAE diagnostic plan: by horizon/category/store/stockout/promo/event
- Add explicit leakage note before feature engineering cells

---

Generated by `deep_eda_report.py`. Internal dataset read from `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\super-ai-engineer-season-6-coffee-chain-hackathon`.
