# ExpresSo NB Hackathon — EDA Summary
**Demand Forecasting at Scale** | Metric: MAE | Deadline: 2026-05-15

---

## Problem Overview

| Item | Detail |
|------|--------|
| Task | Forecast `units_sold` per **(store × category × day)** at 3 horizons |
| Horizons | `1d` (next-day), `7d` (week-ahead), `30d` (month-ahead) |
| Forecast window | 2024-11-01 → 2024-12-31 (61 days) |
| Output rows | 20 × 7 × 61 × 3 = **25,620 rows** |
| Public LB | November 2024 |
| Private LB | December 2024 |
| Metric | **MAE** (lower = better, negatives clipped to 0) |

---

## Data Overview

| Table | Rows | Key columns |
|-------|------|-------------|
| TRANSACTION | 2,858,050 | transaction_id, order_id, product_id, units_sold, discount_applied, revenue |
| ORDER | 1,376,133 | order_id, store_id, date, hour, customer_id, is_member, payment_method |
| INVENTORY | 804,000 | store_id, product_id, date, opening_stock, units_sold, closing_stock, **is_stockout** |
| PROMOTION | 31,314 | promo_id, store_id, product_id, start/end_date, promo_type, discount_pct |
| LOCAL_EVENT | 1,440 | store_id, date, event_name, event_type |
| CUSTOMER | 6,000 | customer_id, home_neighborhood_type, preferred_store_id |
| DATE_DIM | 731 | date, day_of_week, is_weekend, is_holiday, is_payday, is_rainy_season |
| STORE | 20 | store_id, neighborhood_type, seating_capacity, has_drive_through, staff_count |
| PRODUCT | 60 | product_id, category, serve_type, base_price, is_seasonal |

### Target construction
```python
target = (TRANSACTION
          .merge(ORDER[["order_id","store_id","date"]], on="order_id")
          .merge(PRODUCT[["product_id","category"]], on="product_id")
          .groupby(["store_id","category","date"])["units_sold"]
          .sum())
# 92,340 daily observations | 2023-01-01 → 2024-10-31
```

---

## Target Distribution

| Stat | Value |
|------|-------|
| Mean | 42.8 units |
| Median | 23.0 units |
| Max | 1,607 units |
| Zero-sales days | ~0% (all combos recorded) |

> Heavy right skew → use **log1p transform** or tree-based models (LightGBM/XGBoost)

---

## Category Rankings (avg daily units/store)

| Rank | Category | Avg daily units |
|------|----------|----------------|
| 1 | Coffee | 146.7 |
| 2 | Tea | 46.1 |
| 3 | Bakery | 36.0 |
| 4 | Savory Bakery | 28.5 |
| 5 | Chocolate & Milk | 21.0 |
| 6 | Juice & Smoothie | 12.7 |
| 7 | Merchandise | 7.6 |

> Train **per-category models** or use category as a strong categorical feature

---

## Calendar Effects

| Feature | Effect |
|---------|--------|
| Weekend | Slight uplift vs weekday |
| Holiday | Visible spike |
| Payday | Modest uplift |
| Rainy season | Slight suppression |
| Peak hours | 8:00–11:00 (morning rush) |

---

## Promotion Impact

- **Active promo → 1.32× uplift** in units sold
- Discount range: 10–50%, most common at 30%
- Top promo types: ชุดคู่ลด (bundle), loyalty, flash sale
- `discount_applied` in TRANSACTION is **0–100 (percent)**, not a fraction

---

## Local Events Impact

- Events create above-average sales spikes
- Top event types by uplift: food_festival, market, cultural
- ~1,440 event records across 670 training days

---

## Stockout Analysis

| Metric | Value |
|--------|-------|
| Train stockout rate | **6.58%** |
| Test expected rate | **8.20%** (slight drift) |
| Worst stores | Stores 3, 7, 14 (highest rates) |
| Worst category | Merchandise (highest stockout freq) |

> **Critical:** Use `sample_weight ≈ 0.3` on stockout rows — recorded value is truncated demand

---

## Customer Analysis

| Metric | Value |
|--------|-------|
| Walk-in (null customer_id) | ~60% of all orders |
| Member orders | ~40% |
| Top payment | กระเป๋าเงินดิจิทัล, QR/PromptPay |

> Walk-ins are structural, not missing data — do NOT impute

---

## DBSCAN Clustering Results

### Store behaviour clusters (sales profile only)
- **2 clusters** + 4 noise outliers
- Cluster 0: high-volume stores (university, hospital zones)
- Cluster 1: mid-volume stores (tourist, gas_station zones)

### Store clusters (with store attributes)
- **5 clusters** (eps=2.0, min_samples=2)
- Driven by: neighborhood_type + seating capacity + sales mix

### Store × Category demand segments
- **4 segments** across 140 (store, category) cells
- Segment 0: high-demand Coffee/Tea cells
- Segment 1: moderate multi-category cells
- Segment 2: low-volume niche categories
- Segment 3: sporadic/event-driven cells

> Add `cluster_id` as a categorical feature — free signal

---

## Leakage Rules (Critical)

```
Decision day = day model runs | Forecast date = future target day

Horizon  Max usable date for features
-------  ----------------------------
1d       ≤ forecast_date − 1 day
7d       ≤ forecast_date − 7 days
30d      ≤ forecast_date − 30 days
```

**Safe lag features per horizon:**

| Horizon | Safe lags | Safe rolling windows |
|---------|-----------|---------------------|
| 1d | lag_7, lag_14, lag_28, lag_35 | rolling_7 shifted ≥1 |
| 7d | lag_7, lag_14, lag_28, lag_35 | rolling_7 shifted ≥7 |
| 30d | lag_35, lag_60, lag_90 | rolling_30 shifted ≥35 |

> Judges will check code — leakage = disqualification

---

## Missing Data

| Column | Missing % | Note |
|--------|-----------|------|
| `DATE_DIM.holiday_name` | 96.2% | Only populated on holidays — expected |
| `ORDER.customer_id` | 59.97% | Walk-ins — structural, not missing |

---

## Recommended Feature Set (Rank #1)

```python
# 1. Lag features (horizon-safe)
lag_7, lag_14, lag_28, lag_35, lag_60, lag_90

# 2. Rolling statistics (horizon-safe windows)
rolling_28_mean, rolling_28_std, rolling_28_median

# 3. Calendar
day_of_week, month, week_of_year,
is_holiday, is_weekend, is_payday, is_rainy_season

# 4. Promotion
has_promo, max_discount_pct, promo_type_encoded,
days_since_last_promo, days_until_next_promo

# 5. Local events
has_event, event_type_encoded

# 6. Store profile
neighborhood_type, seating_capacity, staff_count,
has_drive_through, dbscan_cluster_id

# 7. Stockout signal
lag_is_stockout_7, rolling_stockout_rate_28

# 8. Year-over-year
same_dow_same_week_last_year_mean
```

---

## Recommended Model Stack

| Step | Approach |
|------|----------|
| Model | **LightGBM** (one model per horizon, or add `horizon` as feature) |
| Transform | `log1p(y)` → predict → `expm1()` |
| Stockout | `sample_weight=0.3` on `is_stockout=True` rows |
| Validation | Time-series CV — train up to Sep 2024, validate Oct 2024 |
| Ensemble | LightGBM + XGBoost + simple seasonal baseline (weighted avg) |
| Post-process | Clip predictions to `max(0, pred)` |

> Trust your CV — overfitting to Nov public LB won't help on Dec private LB

---

## EDA Images (`pp/eda/`)

| File | Content |
|------|---------|
| `01_overview.png` | Table sizes + target distribution |
| `02_target_distribution.png` | Log-scale dist + percentile curve |
| `03_daily_total_sales.png` | Full time-series (all stores/categories) |
| `04_category_timeseries.png` | 7-day rolling mean by category |
| `04b_category_totals.png` | Total units by category |
| `05_store_totals.png` | Total units by store |
| `05b_store_category_heatmap.png` | Mean units heatmap (store × category) |
| `05c_neighborhood_sales.png` | Sales by store neighborhood type |
| `06_temporal_patterns.png` | Day-of-week + month patterns |
| `06b_hourly_orders.png` | Order count by hour |
| `06c_calendar_effects.png` | Holiday / payday / weekend / rainy effects |
| `07_promotion_impact.png` | Promo uplift + promo type counts |
| `07b_discount_distribution.png` | Discount % distribution |
| `08_event_impact.png` | Event uplift + by event type |
| `09_stockout_analysis.png` | Stockout rate by store + category |
| `09b_stockout_temporal.png` | Stockout by month + day-of-week |
| `10_customer_analysis.png` | Member vs walk-in + neighborhood + payment |
| `10b_member_sales.png` | Units sold member vs walk-in |
| `11_autocorrelation.png` | ACF lags 1–30 (daily total sales) |
| `12_units_vs_revenue.png` | Units vs revenue scatter |
| `13_dbscan_store_clusters.png` | DBSCAN store clusters (PCA 2D) + profile |
| `13b_dbscan_store_attrs.png` | DBSCAN with store attributes |
| `14_dbscan_cat_store.png` | DBSCAN store × category segments |
| `15_lag_boundary.png` | Lag cutoff visualisation per horizon |
| `16_correlation_heatmap.png` | Feature correlation matrix |
| `17_missing_data.png` | Missing data by table.column |
| `18_summary_insights.png` | Full text summary card |
