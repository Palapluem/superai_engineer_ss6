# **ExpresSo NB Hackathon — CLAUDE.md**

Demand Forecasting at Scale · Kaggle Competition · Deadline: 2026-05-15 08:00 ICT

---

## **Competition Brief**

| Item | Detail |
| ----- | ----- |
| Task | Forecast daily units sold per (store × category × day × horizon) |
| Stores | 20 stores in Bangkok metropolitan area |
| Categories | Coffee, Tea, Chocolate & Milk, Juice & Smoothie, Bakery, Savory Bakery, Merchandise |
| Forecast window | 2024-11-01 → 2024-12-31 (61 days) |
| Horizons | 1d (next-day), 7d (week-ahead), 30d (month-ahead) |
| Total rows | 25,620 (20 × 7 × 61 × 3\) |
| Metric | MAE (lower is better) |
| Compute budget | Google Colab free tier · 2 CPU cores · 12 GB RAM · ≤ 2 hours |
| Public LB | November 2024 (12,600 rows) |
| Private LB | December 2024 (13,020 rows) — final rank |
| Submissions | 4 per day · highest score counts |

---

## **Data Files**

### **Train folder (`train/`)**

| Table | Rows | Key columns |
| ----- | ----- | ----- |
| TRANSACTION | 2,858,050 | transaction\_id, order\_id, product\_id, units\_sold, discount\_applied, revenue |
| ORDER | 1,376,133 | order\_id, store\_id, date, hour, customer\_id, is\_member, payment\_method |
| INVENTORY | 804,000 | store\_id, product\_id, date, opening\_stock, units\_received, units\_sold, closing\_stock, is\_stockout |
| PROMOTION | 31,314 | product\_id, store\_id, start\_date, end\_date, promo\_type, discount\_pct, email\_sent, social\_campaign |
| LOCAL\_EVENT | 1,440 | store\_id, date, event\_name, event\_type |
| CUSTOMER | 6,000 | customer\_id, registration\_date, home\_neighborhood\_type, is\_member, preferred\_store\_id |
| DATE\_DIM | 731 | date, day\_of\_week, week\_number, month, quarter, year, is\_weekend, is\_holiday, holiday\_name, is\_school\_break, is\_payday, is\_rainy\_season |
| STORE | 20 | store\_id, neighborhood\_type, seating\_capacity, has\_drive\_through, staff\_count, open\_time, close\_time, opened\_date |
| PRODUCT | 60 | product\_id, product\_name, category, serve\_type, base\_price, is\_seasonal, is\_limited\_edition |

### **Test folder (`test/`) — available for forecast window**

* DATE\_DIM.csv, STORE.csv, PRODUCT.csv, PROMOTION.csv, LOCAL\_EVENT.csv, sample\_submission\_with\_id.csv  
* ⚠️ `test/TRANSACTION.csv`, `test/ORDER.csv`, `test/INVENTORY.csv` do NOT exist — referencing them \= disqualification

### **Training period**

* 2023-01-01 → 2024-10-31 (670 days, 22 months)

---

## **Critical Rules & Gotchas**

### **1\. Feature lag leakage (most common disqualification)**

Horizon 1d  → features must use data from ≤ date \- 1  (lag\_1+ allowed)  
Horizon 7d  → features must use data from ≤ date \- 7  (lag\_8+ allowed)  
Horizon 30d → features must use data from ≤ date \- 30 (lag\_31+ allowed)

* `lag_1` in a 7d model \= instant disqualification  
* A 7-day rolling mean in a 30d model \= instant disqualification  
* Build **three completely separate feature pipelines**, one per horizon

### **2\. Build target from TRANSACTION, not INVENTORY**

target \= (  
    TRANSACTION  
    .merge(ORDER\[\["order\_id", "store\_id", "date"\]\], on="order\_id")  
    .merge(PRODUCT\[\["product\_id", "category"\]\], on="product\_id")  
    .groupby(\["store\_id", "category", "date"\])\["units\_sold"\]  
    .sum()  
)

* `INVENTORY.units_sold` only matches TXN in \~12% of rows — noisy, do not use as target  
* Use INVENTORY only for `is_stockout` flag

### **3\. Data integrity notes**

* `discount_applied` is 0–100 scale (percent), NOT a fraction  
* \~60% of orders have `customer_id = null` (walk-ins) — structural, not missing data  
* Test stockout rate is 8.20% vs train 6.58% — slight drift  
* Zero-sale rows: 0% in training data (every store sells every category every day)

### **4\. Stockout handling**

\# Down-weight stockout rows so model isn't pulled toward truncated values  
sample\_weight \= np.where(is\_stockout, 0.3, 1.0)  
\# Fix: INVENTORY has product\_id not category — need to join PRODUCT first  
inventory\_with\_cat \= inventory.merge(product\[\["product\_id", "category"\]\], on="product\_id")  
stockout \= inventory\_with\_cat.groupby(\["store\_id", "category", "date"\])\["is\_stockout"\].max()

---

## **Store Profiles**

| store\_id | neighborhood\_type | volume tier | notes |
| ----- | ----- | ----- | ----- |
| 6 | mall | HIGH (360k) | Highest volume store |
| 10 | tourist | HIGH (314k) | New store (Nov 2022\) — growth ramp |
| 9 | tourist | HIGH (288k) |  |
| 5 | urban\_residential | MID (273k) |  |
| 15 | tourist | MID (266k) |  |
| 2 | tourist | MID (238k) |  |
| 14 | university | MID (223k) |  |
| 8 | urban\_residential | MID (210k) |  |
| 3 | university | MID (207k) |  |
| 17 | office | MID (207k) | Weekday-heavy, weekend dip |
| 19 | university | MID (200k) |  |
| 20 | transit | MID (182k) |  |
| 1 | university | LOW (146k) |  |
| 16 | transit | LOW (144k) |  |
| 7 | hospital | LOW (136k) | Opens 06:30 |
| 18 | hospital | LOW (122k) | Opens 06:30 |
| 11 | gas\_station | LOW (116k) | Has drive-through |
| 12 | gas\_station | LOW (114k) |  |
| 4 | hospital | LOW (110k) | Opens 06:30 |
| 13 | gas\_station | LOW (91k) | Has drive-through · Lowest volume |

**Volume disparity: Store 6 (mall) \= 4× Store 13 (gas station)**

---

## **Category Summary Stats**

| Category | Mean | Median | Std | Max | Priority |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Coffee | 146.7 | 127 | 88.9 | 1,607 | Highest — most volatile |
| Tea | 46.1 | 39 | 31.8 | 589 | Tracks Coffee seasonally |
| Bakery | 36.0 | 31 | 23.2 | 398 | Steady |
| Savory Bakery | 28.5 | 24 | 18.7 | 291 | Lunch-driven |
| Chocolate & Milk | 21.0 | 18 | 15.1 | 313 | Cool-season lift |
| Juice & Smoothie | 12.7 | 11 | 9.6 | 159 | Inverse rainy-season pattern |
| Merchandise | 7.6 | 6 | 5.5 | 95 | Lowest — gifting season Dec |

---

## **EDA Key Findings**

### **Seasonality**

* **Dominant FFT periods: 335 days (annual) and 167.5 days (semi-annual)**  
* Monthly pattern: April is highest (Songkran Thai New Year), Jun–Aug is trough (rainy season), Dec is second highest  
* Coffee: April avg 201, June low 107 (+88% swing)  
* STL trend strength: 0.707 (strong) · STL seasonal strength: 0.224 (moderate)

### **Stationarity**

* 89.3% of (store × category) series are stationary (ADF p\<0.05 and KPSS p≥0.05)  
* No differencing needed — raw sales values are directly modelable

### **Autocorrelation**

lag\_1:  median AC \= 0.301  ← strongest lag feature  
lag\_7:  median AC \= 0.249  ← strong weekly pattern  
lag\_14: median AC \= 0.186  
lag\_30: median AC \= 0.021  ← near zero\! 30d model cannot rely on lags

### **Holiday Effect (from EDA)**

* Average uplift on holiday vs non-holiday days: **\+57% across ALL categories**  
* Coffee: 143.5 non-holiday → 225.8 holiday (+57%)  
* Bakery: 35.1 → 56.1 (+60%)  
* **is\_holiday is the single strongest feature in the dataset**

### **Structural Breaks**

* CUSUM breaks cluster on: Apr 15–23 (Songkran) and Dec 9 (Constitution Day)  
* 52.9% of series show heteroscedasticity (variance instability) → confirms quantile loss over MSE

### **Day of Week**

* Sat/Sun: \~10–15% above Mon–Fri average  
* Mon–Fri are essentially flat (within 2% of each other)  
* Office store 17 has inverse pattern — weekday heavy, weekend dip

### **Promotions**

* Types: แต้มx2 (double points, 41%), สมาชิกใหม่ (new member, 33%), ลดราคา (discount, 14%), ซื้อ1แถม1 (BOGO, 9%), ชุดคู่ลด (bundle, 3%)  
* Global promo effect on volume: near zero (double-points loyalty promos dilute signal)  
* Use `promo_type` as categorical feature — don't use binary `has_promo`

---

## **Feature Engineering Plan**

### **Tier 1 — Build First (all horizons, horizon-safe)**

\# Calendar (from DATE\_DIM — already encoded, just join)  
is\_holiday, holiday\_name, is\_weekend, is\_school\_break, is\_payday, is\_rainy\_season

\# Store profile (from STORE)  
neighborhood\_type, seating\_capacity, staff\_count, has\_drive\_through

\# Product profile (from PRODUCT)  
base\_price, is\_seasonal, is\_limited\_edition

\# Promotions (from PROMOTION — test window provided)  
promo\_type (categorical), discount\_pct, email\_sent, social\_campaign

\# Local events (from LOCAL\_EVENT — test window provided)  
event\_type (categorical), has\_event (binary)

### **Tier 2 — Lag & Rolling Features (horizon-safe windows)**

\# For 1d model (lag\_1+ allowed)  
lag\_1, lag\_2, lag\_7, lag\_14  
rolling\_mean\_7, rolling\_mean\_14, rolling\_std\_7

\# For 7d model (lag\_8+ allowed)  
lag\_7, lag\_8, lag\_14, lag\_21  
rolling\_mean\_14, rolling\_mean\_28, rolling\_std\_14

\# For 30d model (lag\_31+ allowed) — lags almost useless (AC=0.021)  
lag\_30, lag\_31, lag\_60, lag\_90  
rolling\_mean\_60, rolling\_mean\_90  
\# → Rely primarily on calendar \+ Fourier features

### **Tier 3 — Fourier Seasonality (critical for 30d model)**

import numpy as np

\# Add Fourier features for dominant periods (335d and 168d)  
for period in \[335, 168, 30, 7\]:  
    df\[f'sin\_{period}'\] \= np.sin(2 \* np.pi \* df\['day\_index'\] / period)  
    df\[f'cos\_{period}'\] \= np.cos(2 \* np.pi \* df\['day\_index'\] / period)

### **Tier 4 — Interaction Features**

\# High value interactions  
neighborhood\_type × is\_weekend          \# office store weekend dip  
neighborhood\_type × is\_holiday          \# tourist/mall spike vs hospital  
neighborhood\_type × is\_school\_break     \# university stores: Dec drop-off  
promo\_type × category                   \# BOGO effect differs by category  
is\_rainy\_season × category              \# Coffee drops, Juice less affected  
is\_holiday × category                   \# each category responds differently  
event\_type × neighborhood\_type          \# market near university vs mall

### **Tier 5 — Additional Features**

\# Store growth trajectory  
days\_since\_store\_opened      \# Store 10 opened Nov 2022 — still growing

\# DTW store clustering  
store\_dtw\_cluster            \# 3-4 clusters from DTW on normalized sales history

\# Stockout feature (fixed join)  
is\_stockout\_lag1             \# stockout yesterday → suppressed demand today

\# December-specific  
is\_gifting\_season \= (month \== 12\) & (day \>= 10\)   \# Merchandise spike

\# Named holiday individual flags (Dec private LB)  
is\_loy\_krathong    \# Nov 15, 2024  
is\_fathers\_day     \# Dec 5, 2024 (Thai National Day)  
is\_constitution\_day \# Dec 10, 2024  
is\_nye             \# Dec 31, 2024 — extreme outlier

---

## **Important Bangkok Calendar Dates (Nov–Dec 2024\)**

| Date | Event | Expected Impact |
| ----- | ----- | ----- |
| Nov 15 | Loy Krathong | \+57% avg, tourist/mall stores highest |
| Nov (mid-late) | Thai university final exams | Coffee spike at university stores |
| Dec 5 | Father's Day / National Day | Public holiday, mall traffic surge |
| Dec 10 | Constitution Day | Public holiday |
| Dec (semester end) | University semester ends | Student stores drop Dec onwards |
| Dec 10–31 | Corporate gifting season | Merchandise lift at office stores |
| Dec 24–26 | Christmas week | Mall stores significant lift |
| Dec 31 | New Year's Eve | Extreme outlier — Central World area |

---

## **Modeling Strategy**

### **Architecture: Three Separate Models**

Horizon 1d  → LightGBM (quantile α=0.5) · lag features available  
Horizon 7d  → LightGBM (quantile α=0.5) · blend recursive \+ non-recursive  
Horizon 30d → LightGBM (quantile α=0.5) · calendar \+ Fourier dominant

### **Loss Function Decision**

* **Primary: quantile loss (α=0.5)** — directly optimizes MAE, robust to heteroscedasticity  
* Zero-sale rate is 0% → Tweedie's zero-inflation benefit does NOT apply here  
* Test Tweedie as challenger — pick whichever wins in CV  
* Do NOT use MSE/RMSE as primary objective

### **LightGBM Parameters (starting point)**

lgb\_params \= {  
    'objective': 'quantile',  
    'alpha': 0.5,  
    'learning\_rate': 0.05,  
    'num\_leaves': 128,  
    'min\_child\_samples': 20,  
    'feature\_fraction': 0.8,  
    'bagging\_fraction': 0.8,  
    'bagging\_freq': 1,  
    'verbose': \-1,  
}

### **Recursive vs Non-Recursive (from M5 winner insights)**

* Recursive generally outperforms non-recursive overall  
* Non-recursive has best score on held-out CV (lower variance)  
* **Blend both** for robustness — most important for 7d and 30d horizons

### **Ensemble**

Final prediction \= 0.7 × LightGBM \+ 0.3 × XGBoost

* Train XGBoost only after LightGBM pipeline is finalized  
* Use `tree_method='hist'` in XGBoost for speed  
* Only add if time budget allows and CV improves

---

## **Validation Strategy**

### **Three-Fold Time-Based CV (mimic public/private split)**

CV1: Train Jan 2023 – Jul 2024 · Validate Aug 2024  
CV2: Train Jan 2023 – Aug 2024 · Validate Sep 2024  
CV3: Train Jan 2023 – Sep 2024 · Validate Oct 2024  ← most important  
Public LB: November 2024  
Private LB: December 2024  ← final rank

### **Key rules**

* Focus on **std across CV folds, not just mean** — consistent model beats a lucky one  
* Trust Oct CV score (CV3) over November public LB  
* Do NOT overfit to November public leaderboard  
* December has Father's Day \+ NYE — models missing these will underforecast

---

## **DTW Application**

from dtaidistance import dtw  
from scipy.cluster.hierarchy import linkage, fcluster

\# Normalize series before DTW (compare shape, not scale)  
series\_normalized \= \[(s \- s.mean()) / s.std() for s in store\_series\]

\# Compute pairwise distance matrix — 20×20 \= fast  
distance\_matrix \= dtw.distance\_matrix\_fast(series\_normalized)

\# Cluster into 3–4 groups  
Z \= linkage(distance\_matrix, method='ward')  
store\_clusters \= fcluster(Z, t=3, criterion='maxclust')  
\# → Add cluster label as categorical feature in LightGBM

Use DTW on training period only (Jan 2023 – Oct 2024\) — no leakage.

---

## **Post-Processing**

\# Required: clip all predictions to non-negative  
predictions \= np.clip(predictions, 0, None)

\# Optional: per-(store, category) bias correction for December drift  
\# Compute ratio of recent 30-day mean to model's predicted mean  
\# Apply as multiplicative scaling factor

---

## **Submission Format**

id format: {store\_id}\_{category}\_{forecast\_date}\_{horizon}  
example:   1\_Bakery\_2024-11-01\_1d

Columns: id (string), units\_sold\_predicted (float ≥ 0\)  
Required rows: 25,620

---

## **Time Allocation**

| Day | Activity |
| ----- | ----- |
| Day 1–2 | Canonical target build \+ EDA \+ LightGBM baseline \+ first submission |
| Day 2–3 | Tier 1 features \+ leakage audit \+ three CV folds |
| Day 3 | Holiday flags (individual named) \+ Fourier features \+ Bangkok calendar |
| Day 3–4 | Tier 2 lag features \+ store archetype features |
| Day 4–5 | Tier 3 interaction features \+ DTW clustering |
| Day 5 | Recursive vs non-recursive blend |
| Day 6 | XGBoost secondary model \+ final ensemble |
| Day 7 | Colab end-to-end runtime test \+ notebook cleanup \+ final submission |

---

## **Colab Checklist (before final submission)**

* \[ \] Notebook runs end-to-end in \< 2 hours on Colab free tier  
* \[ \] No reference to `test/TRANSACTION.csv`, `test/ORDER.csv`, `test/INVENTORY.csv`  
* \[ \] All predictions are non-negative (clipped)  
* \[ \] Submission has exactly 25,620 rows  
* \[ \] No data leakage in lag features (check all three horizons)  
* \[ \] Notebook is reproducible (fixed random seeds)  
* \[ \] Pre-trained models downloadable from public source if used

---

## **References**

* Competition: https://www.kaggle.com/competitions/super-ai-engineer-season-6-coffee-chain-hackathon  
* M5 Poisson baseline (mayer79): https://www.kaggle.com/code/mayer79/m5-forecast-poisson-loss-top-10  
* M5 Tweedie approach: https://www.christophenicault.com/post/m5\_forecasting\_accuracy/  
* M5 results paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9232271/  
* Prompt engineering: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview

# **ExpresSo NB — Feature Registry**

Reference: CLAUDE.md · Thai Guide · EDA Charts viz\_01–viz\_07 · Correlation Analysis Last updated from full conversation context · Deadline 2026-05-15 08:00 ICT

---

## **Summary**

| Tier | Category | Count | Source |
| ----- | ----- | ----- | ----- |
| 1 | Calendar signals | 13 | DATE\_DIM \+ `holidays` library |
| 2 | Fourier seasonality | 8 | Derived from day\_index |
| 3 | Lag / rolling | 13 | Derived from units\_sold |
| 4 | Store \+ Product static | 8 | STORE, PRODUCT tables |
| 5 | Promotions \+ Events | 16 | PROMOTION, LOCAL\_EVENT, INVENTORY |
| 6 | Interaction features | 10 | Cross-products of above |
| — | Categorical encodings | 4 | neighborhood\_type, store\_tier, category, day\_of\_week |
| ❌ | **Dropped (confirmed noise)** | 4 | See Section 7 |
| **Total** |  | **44 features** |  |

---

## **Horizon availability key**

| Symbol | Meaning |
| ----- | ----- |
| `[1d]` | 1-day horizon only — min\_lag \= 1 |
| `[7d]` | 7-day horizon only — min\_lag \= 7 |
| `[30d]` | 30-day horizon only — min\_lag \= 30 |
| `[ALL]` | Available in all three horizons |

---

## **Tier 1 — Calendar Features**

Source: `DATE_DIM` \+ Python `holidays` library (Thailand, years 2023–2025) All available in `[ALL]` horizons — no lag constraint applies to calendar features

| Feature | r (Coffee) | Evidence | Notes |
| ----- | ----- | ----- | ----- |
| `is_rainy_season` | **−0.600** | viz\_06 | Strongest feature overall. Suppresses Coffee more than holidays boost. Already in DATE\_DIM. |
| `is_holiday` | **\+0.436** | viz\_06 | \+28.7% mean lift confirmed in viz\_03. Uniform across all 7 categories (+27.7–29.7%). |
| `is_school_break` | **\+0.407** | viz\_06 | 2nd strongest calendar signal. Captures Songkran window \+ university semester end. |
| `is_national` | **\+0.304** | viz\_06 | National holidays subset. Use alongside is\_holiday. |
| `is_payday` | **\+0.239** | viz\_06 | Use `DATE_DIM.is_payday` — do NOT build manual encoding (days 24–28 was wrong direction). |
| `is_weekend` | **\+0.205** | viz\_05, viz\_06 | \+12.1% confirmed lift. Mon–Fri are flat — binary is sufficient, no 7-DOW dummies needed. |
| `is_pre_holiday` | derived | CLAUDE.md | Days −3 to −1 before a holiday. Sales ramp up 3 days before Loy Krathong / Father's Day. |
| `is_post_holiday` | derived | CLAUDE.md | Days \+1 to \+2 after holiday. Captures hangover effect. |
| `holiday_ramp` | derived | CLAUDE.md | Integer 0–4 sliding ramp into nearest holiday. Encodes build-up more smoothly than binary. |
| `is_loy_krathong` | derived | holidays lib | Nov 15 2024 — inside public LB window. Tourist/mall stores spike heavily. |
| `is_fathers_day` | derived | holidays lib | Dec 5 2024 — inside private LB window. National public holiday. |
| `is_nye` | derived | CLAUDE.md | Dec 31 — extreme outlier day. Central World / Iconsiam area stores. |
| `is_gifting_season` | derived | CLAUDE.md | Dec 10–31. Merchandise category lift at office stores (Store 17). |

**Construction:**

import holidays as hol\_lib  
import numpy as np

th\_hols \= hol\_lib.Thailand(years=\[2023, 2024, 2025\])

\# Named holiday flags  
df\["is\_loy\_krathong"\] \= df\["date"\].isin(  
    \[d for d, n in th\_hols.items() if "ลอยกระทง" in n\]  
).astype(int)  
df\["is\_fathers\_day"\]  \= df\["date"\].isin(  
    \[d for d, n in th\_hols.items() if "พ่อ" in n\]  
).astype(int)  
df\["is\_nye"\]          \= ((df\["date"\].dt.month \== 12\) &  
                          (df\["date"\].dt.day   \== 31)).astype(int)  
df\["is\_gifting\_season"\] \= ((df\["date"\].dt.month \== 12\) &  
                            (df\["date"\].dt.day   \>= 10)).astype(int)

\# Ramp features  
hol\_dates \= sorted(\[pd.Timestamp(d) for d in th\_hols.keys()\])  
df\["days\_from\_holiday"\] \= df\["date"\].apply(  
    lambda d: min((d \- h).days for h in hol\_dates, key=abs)  
)  
df\["is\_pre\_holiday"\]  \= df\["days\_from\_holiday"\].between(-3, \-1).astype(int)  
df\["is\_post\_holiday"\] \= df\["days\_from\_holiday"\].between(1, 2).astype(int)  
df\["holiday\_ramp"\]    \= np.clip(-df\["days\_from\_holiday"\], 0, 4\)

---

## **Tier 2 — Fourier Seasonality Features**

Source: Derived from `day_index` (days since training start) Available in `[ALL]` horizons — **critical for 30d model** where lag AC ≈ 0.021 Dominant periods confirmed by FFT analysis: 335d (annual) and 167.5d (semi-annual)

| Feature | Period | Evidence | Notes |
| ----- | ----- | ----- | ----- |
| `sin_335d` / `cos_335d` | 335 days | FFT analysis | Annual cycle — highest amplitude FFT component |
| `sin_168d` / `cos_168d` | 168 days | FFT analysis | Semi-annual — 2nd highest FFT amplitude |
| `sin_30d` / `cos_30d` | 30 days | CLAUDE.md | Monthly cycle — captures payday-like patterns |
| `sin_7d` / `cos_7d` | 7 days | CLAUDE.md | Weekly cycle — complements lag\_7 for 30d where lag is restricted |

**Construction:**

df \= df.sort\_values("date").reset\_index(drop=True)  
df\["day\_index"\] \= (df\["date"\] \- df\["date"\].min()).dt.days

for period in \[335, 168, 30, 7\]:  
    df\[f"sin\_{period}d"\] \= np.sin(2 \* np.pi \* df\["day\_index"\] / period)  
    df\[f"cos\_{period}d"\] \= np.cos(2 \* np.pi \* df\["day\_index"\] / period)

**Note:** For the test window (Nov–Dec 2024), `day_index` must be computed relative to the same training start date (2023-01-01), not the test start date.

---

## **Tier 3 — Lag / Rolling Features**

Source: Derived from `units_sold` (canonical target) **Horizon-specific — do NOT share lag features across horizons (leakage rule)** Autocorrelation by lag (from stationarity analysis): lag\_1=0.301, lag\_7=0.249, lag\_30=0.021

### **1d Horizon `[1d]` — min\_lag \= 1**

| Feature | AC | Notes |
| ----- | ----- | ----- |
| `lag_1` | 0.301 | Strongest lag signal |
| `lag_2` | \~0.28 |  |
| `lag_7` | 0.249 | Weekly same-day pattern |
| `lag_8` | \~0.23 |  |
| `roll_mean_7` | — | 7-day rolling mean (shifted by 1\) |
| `roll_mean_14` | — | 2-week rolling mean |
| `roll_std_7` | — | Volatility feature |

### **7d Horizon `[7d]` — min\_lag \= 7**

| Feature | Notes |
| ----- | ----- |
| `lag_7` | Most recent allowed lag |
| `lag_8` |  |
| `lag_14` | Bi-weekly pattern |
| `lag_21` | 3-week lookback |
| `roll_mean_14` | 2-week rolling mean (shifted by 7\) |
| `roll_mean_28` | Monthly rolling mean |
| `roll_std_14` | Volatility |

### **30d Horizon `[30d]` — min\_lag \= 30**

| Feature | Notes |
| ----- | ----- |
| `lag_30` | Most recent allowed — AC=0.021, very weak |
| `lag_31` |  |
| `lag_60` | 2-month lookback |
| `roll_mean_60` | Primary lag signal for 30d — 2-month average |
| `roll_mean_90` | 3-month average — most stable for long horizon |

**Critical:** lag\_30 AC ≈ 0.021 means the 30d model relies almost entirely on Tier 1 (calendar) and Tier 2 (Fourier) features. Lags are included but will have low importance. Do not expect the 30d model to match 1d MAE.

**Construction:**

def add\_lags(df, min\_lag, windows):  
    df \= df.sort\_values(\["store\_id", "category", "date"\])  
    grp \= df.groupby(\["store\_id", "category"\])\["units\_sold"\]

    for lag in range(min\_lag, min\_lag \+ 5):  
        df\[f"lag\_{lag}"\] \= grp.shift(lag)

    for w in windows:  
        shifted \= grp.shift(min\_lag)  
        df\[f"roll\_mean\_{w}"\] \= shifted.transform(  
            lambda x: x.rolling(w, min\_periods=1).mean()  
        )  
        df\[f"roll\_std\_{w}"\] \= shifted.transform(  
            lambda x: x.rolling(w, min\_periods=2).std()  
        ).fillna(0)  
    return df

df\_1d  \= add\_lags(master.copy(), min\_lag=1,  windows=\[7, 14\])  
df\_7d  \= add\_lags(master.copy(), min\_lag=7,  windows=\[14, 28\])  
df\_30d \= add\_lags(master.copy(), min\_lag=30, windows=\[60, 90\])

---

## **Tier 4 — Store \+ Product Static Features**

Source: `STORE` table, `PRODUCT` table Available in `[ALL]` horizons — static, no time dimension

| Feature | Type | Evidence | Notes |
| ----- | ----- | ----- | ----- |
| `neighborhood_type` | categorical | viz\_03, store profiles | 8 types: university, tourist, mall, office, hospital, gas\_station, transit, urban\_residential |
| `store_tier` | categorical | CLAUDE.md EDA | high (Store 6,9,10) / mid / low — 4× volume gap between Store 6 and Store 13 |
| `seating_capacity` | int | CLAUDE.md | Range 22–71. Proxy for store scale. Correlates with Coffee volume. |
| `staff_count` | int | CLAUDE.md | Range 4–13. Operational capacity signal. |
| `days_since_opened` | int | CLAUDE.md | Store 10 opened Nov 2022 — still on growth ramp. Captures maturity trajectory. |
| `category` | categorical | — | 7 categories: Coffee, Tea, Bakery, Savory Bakery, Chocolate & Milk, Juice & Smoothie, Merchandise |
| `avg_base_price` | float | PRODUCT | Average base price per category — demand elasticity proxy |
| `day_of_week` | categorical | viz\_05 | Mon–Sun. Mon–Fri essentially flat, Sat–Sun \+12.1% lift. |

**Construction:**

store\["opened\_date"\]  \= pd.to\_datetime(store\["opened\_date"\])  
store\["store\_tier"\]   \= store\["store\_id"\].map(STORE\_TIER)  \# see CLAUDE.md

\# For train: days since opened relative to each row's date  
df\["days\_since\_opened"\] \= (df\["date"\] \- df\["opened\_date"\]).dt.days

\# For test: days since opened relative to forecast start (Nov 1 2024\)  
\# to avoid future leakage  
df\["days\_since\_opened"\] \= (pd.Timestamp("2024-11-01") \- df\["opened\_date"\]).dt.days

cat\_product \= (  
    product.groupby("category")  
    .agg(avg\_base\_price=("base\_price", "mean"))  
    .reset\_index()  
)

---

## **Tier 5 — Promotion \+ Event \+ Stockout Features**

Source: `PROMOTION`, `LOCAL_EVENT`, `INVENTORY → PRODUCT` join Test window versions provided in `test/` — byte-identical lookup tables

### **Promotions (from PROMOTION table)**

| Feature | Notes |
| ----- | ----- |
| `has_promo` | Any promotion active on store-date |
| `has_bogo` | ซื้อ1แถม1 — BOGO, strongest volume driver per category |
| `has_discount` | ลดราคา — direct price discount |
| `has_points_x2` | แต้มx2 — double loyalty points (weak volume effect, high frequency) |
| `max_discount` | Highest `discount_pct` active on store-date |
| `email_sent` | Digital reach amplifier — check if email promo lifts more than non-email |
| `social_campaign` | Social media amplifier |

**Note:** Global promo effect is near zero because double-points (loyalty) promos dilute the signal. Use `promo_type` as categorical or encode each type separately — do NOT use a single binary `has_promo` alone.

### **Local Events (from LOCAL\_EVENT table)**

| Feature | r (Coffee) | Lift % | Notes |
| ----- | ----- | ----- | ----- |
| `event_has_any` | \+0.260 | — | Composite strongest event predictor |
| `event_food_festival` | \+0.227 | **\+130%** | Single most impactful event type |
| `event_music_festival` | \+0.117 | \+85% |  |
| `event_cultural` | \+0.117 | \+65% |  |
| `event_concert` | \+0.101 | \+82% |  |
| `event_market` | \+0.090 | \+45% |  |
| `event_convention` | \+0.042 | \+30% | Borderline — keep |
| `event_sports` | \+0.021 | \+20% | Weak but mostly significant |

### **Stockout (from INVENTORY → PRODUCT join)**

| Feature | Notes |
| ----- | ----- |
| `is_stockout_lag1` | Stockout yesterday → suppressed demand carry-over today. Down-weight stockout rows with `sample_weight=0.3` during training. |

**Critical fix — INVENTORY has no category column:**

\# WRONG — INVENTORY has product\_id not category  
stockout \= inventory.groupby(\["store\_id", "date"\])\["is\_stockout"\].max()

\# CORRECT — join PRODUCT first to get category  
stockout \= (  
    inventory  
    .merge(product\[\["product\_id", "category"\]\], on="product\_id", how="left")  
    .groupby(\["store\_id", "category", "date"\])\["is\_stockout"\]  
    .max()  
    .reset\_index()  
)

**Promotion expansion (daily grain):**

rows \= \[\]  
for \_, r in promotion.iterrows():  
    for d in pd.date\_range(r\["start\_date"\], r\["end\_date"\], freq="D"):  
        rows.append({  
            "store\_id":  r\["store\_id"\],  
            "date":      d,  
            "has\_bogo":  int("ซื้อ1แถม1" in str(r\["promo\_type"\])),  
            "has\_discount": int("ลดราคา" in str(r\["promo\_type"\])),  
            "has\_points\_x2": int("แต้มx2" in str(r\["promo\_type"\])),  
            "discount\_pct": r\["discount\_pct"\],  
            "email\_sent": int(r\["email\_sent"\]),  
            "social\_campaign": int(r\["social\_campaign"\]),  
        })  
promo\_daily \= pd.DataFrame(rows).groupby(\["store\_id","date"\]).max().reset\_index()

---

## **Tier 6 — Interaction Features**

Source: Cross-products of calendar × store\_type × category Confirmed by viz\_06 (calendar heatmap) and viz\_07 (event heatmap)

| Feature | Formula | Evidence | Notes |
| ----- | ----- | ----- | ----- |
| `rainy_x_coffee` | `is_rainy_season × (category=="Coffee")` | r=−0.60 | Strongest interaction. Rainy season suppresses Coffee more than any other signal. |
| `rainy_x_juice` | `is_rainy_season × (category=="Juice & Smoothie")` | viz\_06 | Opposite pattern to Coffee — less rainy-season sensitivity. |
| `weekend_x_office` | `is_weekend × (neighborhood_type=="office")` | viz\_05 | Store 17 (office) dips on weekends while others rise. |
| `weekend_x_mall` | `is_weekend × (neighborhood_type=="mall")` | viz\_05 | Store 6 (mall) has the strongest weekend surge. |
| `school_break_x_uni` | `is_school_break × (neighborhood_type=="university")` | viz\_06 | University stores spike during exam period, then drop after semester ends (December). |
| `gifting_x_merch` | `is_gifting_season × (category=="Merchandise")` | CLAUDE.md | Dec 10–31 gifting season lifts Merchandise at office stores. |
| `nye_x_tourist` | `is_nye × (neighborhood_type∈["tourist","mall"])` | CLAUDE.md | Dec 31 extreme outlier — only tourist/mall stores are meaningfully affected. |
| `loy_x_tourist` | `is_loy_krathong × (neighborhood_type∈["tourist","mall"])` | CLAUDE.md | Nov 15 Loy Krathong — tourist stores spike, hospital/gas\_station stores do not. |
| `food_festival_x_store` | `event_food_festival × store_id` | viz\_07 | r=+0.22, \+130% lift. Location-specific — a food festival near Store 6 (mall) drives far more absolute units than near Store 13 (gas station). |
| `holiday_x_coffee` | `is_holiday × (category=="Coffee")` | viz\_03 | Coffee has largest absolute MAE impact on holidays (2,836→3,639 units). |

**Construction:**

df\["rainy\_x\_coffee"\]     \= df\["is\_rainy\_season"\] \* (df\["category"\] \== "Coffee").astype(int)  
df\["rainy\_x\_juice"\]      \= df\["is\_rainy\_season"\] \* (df\["category"\] \== "Juice & Smoothie").astype(int)  
df\["weekend\_x\_office"\]   \= df\["is\_weekend"\] \* (df\["neighborhood\_type"\] \== "office").astype(int)  
df\["weekend\_x\_mall"\]     \= df\["is\_weekend"\] \* (df\["neighborhood\_type"\] \== "mall").astype(int)  
df\["school\_break\_x\_uni"\] \= df\["is\_school\_break"\] \* (df\["neighborhood\_type"\] \== "university").astype(int)  
df\["gifting\_x\_merch"\]    \= df\["is\_gifting\_season"\] \* (df\["category"\] \== "Merchandise").astype(int)  
df\["nye\_x\_tourist"\]      \= df\["is\_nye"\] \* df\["neighborhood\_type"\].isin(\["tourist","mall"\]).astype(int)  
df\["loy\_x\_tourist"\]      \= df\["is\_loy\_krathong"\] \* df\["neighborhood\_type"\].isin(\["tourist","mall"\]).astype(int)  
df\["food\_festival\_x\_store"\] \= df\["event\_food\_festival"\] \* df\["store\_id"\]  
df\["holiday\_x\_coffee"\]   \= df\["is\_holiday"\] \* (df\["category"\] \== "Coffee").astype(int)

---

## **Categorical Features (LightGBM native encoding)**

These are passed as `categorical_feature` to LightGBM — no one-hot encoding needed.

| Feature | Values | Notes |
| ----- | ----- | ----- |
| `neighborhood_type` | university, tourist, mall, office, hospital, gas\_station, transit, urban\_residential | 8 types |
| `store_tier` | high, mid, low | 3 tiers |
| `category` | 7 product categories | Core model dimension |
| `day_of_week` | Monday … Sunday | Mon–Fri flat, Sat–Sun \+12.1% |

CAT\_FEATURES \= \["neighborhood\_type", "store\_tier", "category", "day\_of\_week"\]

\# Cast before LightGBM  
for c in CAT\_FEATURES:  
    df\[c\] \= df\[c\].astype("category")

---

## **Section 7 — Confirmed Dropped Features**

These were explicitly tested and found to add no signal. Do not include.

| Feature | Reason | Evidence |
| ----- | ----- | ----- |
| `event_book_fair` | r=0.001–0.006, ALL cells marked `ns` (p\>0.05) | viz\_07 |
| `is_buddhist` | r=0.000 across all categories — absorbed into `is_holiday` | viz\_06 |
| `is_cultural` | r=0.000 across all categories — absorbed into `is_holiday` | viz\_06 |
| Manual payday week (days 24–28) | DATE\_DIM.is\_payday already exists and is more accurate. Manual encoding showed **lower** sales in payday week — opposite of assumption. | EDA numerical output |

---

## **Section 8 — Feature Availability by Horizon**

| Feature Group | 1d | 7d | 30d | Notes |
| ----- | ----- | ----- | ----- | ----- |
| Calendar (Tier 1\) | ✅ | ✅ | ✅ | No lag constraint |
| Fourier (Tier 2\) | ✅ | ✅ | ✅ | **Critical for 30d** |
| lag\_1, lag\_2 | ✅ | ❌ | ❌ | Leakage if used in 7d/30d |
| lag\_7, lag\_8 | ✅ | ✅ | ❌ |  |
| lag\_14–lag\_28 | ✅ | ✅ | ❌ |  |
| lag\_30–lag\_60 | ✅ | ✅ | ✅ | Very weak AC for 30d |
| roll\_mean\_7/14 | ✅ | ✅ | ❌ |  |
| roll\_mean\_28/60 | ✅ | ✅ | ✅ |  |
| roll\_mean\_90 | ✅ | ✅ | ✅ | Primary 30d lag signal |
| Store static (Tier 4\) | ✅ | ✅ | ✅ |  |
| Promotions (Tier 5\) | ✅ | ✅ | ✅ | test/ versions provided |
| Events (Tier 5\) | ✅ | ✅ | ✅ | test/ versions provided |
| Stockout lag\_1 | ✅ | ❌ | ❌ | Only valid for 1d |
| Interactions (Tier 6\) | ✅ | ✅ | ✅ | Derived from calendar |

---

## **Section 9 — Feature Priority for Private Leaderboard (December)**

December-specific signals that most teams will miss:

| Feature | Date | Why it matters |
| ----- | ----- | ----- |
| `is_fathers_day` | Dec 5 | Public holiday — \+28.7% expected lift |
| `is_gifting_season` | Dec 10–31 | Corporate gifting — Merchandise spike |
| `is_constitution_day` | Dec 10 | Public holiday |
| `is_nye` | Dec 31 | Extreme outlier — tourist/mall stores |
| `school_break_x_uni` | All Dec | Semester ends → university stores drop off |
| `nye_x_tourist` | Dec 31 | Location-specific extreme spike |
| Fourier 335d/168d | All Dec | December is on the upswing of annual cycle |

---

## **Section 10 — Loss Function**

| Category | Recommended loss | Reason |
| ----- | ----- | ----- |
| All categories | `quantile, alpha=0.5` | Directly optimizes MAE. Zero-sale rate \= 0% so Tweedie's zero-inflation benefit does not apply. |
| Challenger | `tweedie, power=1.1` | Test in CV — pick whichever wins. |

params \= {  
    "objective":    "quantile",  
    "alpha":        0.5,   \# median regression \= MAE-optimal  
    "metric":       "mae",  
    "learning\_rate": 0.05,  
    "num\_leaves":   128,  
    "min\_child\_samples": 20,  
    "feature\_fraction":  0.8,  
    "bagging\_fraction":  0.8,  
    "bagging\_freq":  1,  
    "verbose":      \-1,  
}

---

## **Quick Reference — Feature Count by Model**

| Model | Calendar | Fourier | Lags | Store/Product | Promo/Event | Interactions | Total |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1d | 13 | 8 | 7 | 8 | 16 | 10 | **62** |
| 7d | 13 | 8 | 5 | 8 | 16 | 10 | **60** |
| 30d | 13 | 8 | 3 | 8 | 16 | 10 | **58** |

Categorical features (4) are shared across all models and passed separately to LightGBM.

