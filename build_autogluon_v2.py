import json, os

def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 5}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": [src]}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src if isinstance(src, list) else [src]}

cells = [
md("# 🚀 AutoGluon v2 — Ultimate Demand Forecasting\nผสมทุก insight จากทุกไฟล์: Leakage-Safe Lags, Momentum, Stockout, Events, Annual Lags, B1G1, Interaction Features\n**Run on GPU (T4x2). ไม่ต้องติดตั้งอะไรล่วงหน้า**"),

code("""\
import subprocess, sys
try:
    import autogluon.tabular
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon", "-q"])
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
"""),

code("""\
# ─── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path("/kaggle/input/competitions/super-ai-engineer-season-6-coffee-chain-hackathon")
TR, TE = DATA_DIR / "train", DATA_DIR / "test"
print("Data:", DATA_DIR)

def rc(p): return pd.read_csv(p).rename(columns=str.strip)

txn       = rc(TR/"TRANSACTION.csv")
order     = rc(TR/"ORDER.csv")
inventory = rc(TR/"INVENTORY.csv")
product_tr = rc(TR/"PRODUCT.csv")
product_te = rc(TE/"PRODUCT.csv")
store_tr   = rc(TR/"STORE.csv")
store_te   = rc(TE/"STORE.csv")
date_tr    = rc(TR/"DATE_DIM.csv")
date_te    = rc(TE/"DATE_DIM.csv")
promo_tr   = rc(TR/"PROMOTION.csv")
promo_te   = rc(TE/"PROMOTION.csv")
event_tr   = rc(TR/"LOCAL_EVENT.csv")
event_te   = rc(TE/"LOCAL_EVENT.csv")
sample     = rc(DATA_DIR/"sample_submission_with_id.csv")

for df, cols in [(order,["date"]), (inventory,["date"]), (promo_tr,["start_date","end_date"]),
                 (promo_te,["start_date","end_date"]), (event_tr,["date"]), (event_te,["date"])]:
    for c in cols:
        if c in df.columns: df[c] = pd.to_datetime(df[c])
"""),

code("""\
# ─── Target ─────────────────────────────────────────────────────────
product = pd.concat([product_tr, product_te]).drop_duplicates("product_id")
target = (txn[["order_id","product_id","units_sold"]]
          .merge(order[["order_id","store_id","date"]], on="order_id")
          .merge(product[["product_id","category"]], on="product_id")
          .groupby(["store_id","category","date"], as_index=False)["units_sold"].sum())
target["date"] = pd.to_datetime(target["date"])

STORES = sorted(target["store_id"].unique())
CATS   = sorted(target["category"].unique())
DATES  = pd.date_range("2023-01-01", "2024-10-31", freq="D")

grid = pd.MultiIndex.from_product([STORES, CATS, DATES],
       names=["store_id","category","date"]).to_frame(index=False)
full = grid.merge(target, on=["store_id","category","date"], how="left").fillna({"units_sold":0})
print(f"Grid: {full.shape}")
"""),

code("""\
# ─── Store Features ─────────────────────────────────────────────────
store = pd.concat([store_tr, store_te]).drop_duplicates("store_id")
store["open_h"]  = pd.to_numeric(store["open_time"].str.split(":").str[0], errors="coerce").fillna(7)
store["close_h"] = pd.to_numeric(store["close_time"].str.split(":").str[0], errors="coerce").fillna(21)
store["op_hours"]      = store["close_h"] - store["open_h"]
store["cap_per_staff"] = store["seating_capacity"] / store["staff_count"].replace(0, np.nan)
store["has_drive"]     = store["has_drive_through"].astype(str).str.lower().isin(["true","1","y"]).astype(int)
store["store_age"]     = (pd.Timestamp("2024-11-01") - pd.to_datetime(store["opened_date"])).dt.days
store["is_mall"]       = (store["neighborhood_type"] == "mall").astype(int)
store["is_tourist"]    = (store["neighborhood_type"] == "tourist").astype(int)
store["is_university"] = (store["neighborhood_type"] == "university").astype(int)
store["is_hospital"]   = (store["neighborhood_type"] == "hospital").astype(int)
nbr = pd.get_dummies(store["neighborhood_type"], prefix="nbr").astype(int)
for c in ["nbr_airport","nbr_business","nbr_hospital","nbr_mall","nbr_residential","nbr_tourist","nbr_university"]:
    if c not in nbr: nbr[c] = 0
store_feats = pd.concat([store[["store_id","seating_capacity","staff_count","op_hours",
                                  "cap_per_staff","has_drive","store_age",
                                  "is_mall","is_tourist","is_university","is_hospital"]], nbr], axis=1)
full = full.merge(store_feats, on="store_id", how="left")
"""),

code("""\
# ─── Calendar Features ──────────────────────────────────────────────
date_all = pd.concat([date_tr, date_te]).drop_duplicates("date")
date_all["date"] = pd.to_datetime(date_all["date"])
date_all["dow"]  = date_all["date"].dt.dayofweek
date_all["doy"]  = date_all["date"].dt.dayofyear
date_all["dom"]  = date_all["date"].dt.day
date_all["woy"]  = date_all["date"].dt.isocalendar().week.astype(int)
date_all["month"]    = date_all["date"].dt.month
date_all["quarter"]  = date_all["date"].dt.quarter
date_all["sin_dow"]  = np.sin(2*np.pi*date_all["dow"]/7)
date_all["cos_dow"]  = np.cos(2*np.pi*date_all["dow"]/7)
date_all["sin_doy"]  = np.sin(2*np.pi*date_all["doy"]/365.25)
date_all["cos_doy"]  = np.cos(2*np.pi*date_all["doy"]/365.25)
date_all["is_dec"]   = (date_all["month"] == 12).astype(int)
date_all["is_nov"]   = (date_all["month"] == 11).astype(int)
date_all["is_newyear"] = date_all["date"].dt.strftime("%m-%d").isin(["12-31","01-01"]).astype(int)
for c in ["is_weekend","is_holiday","is_payday","is_school_break","is_rainy_season"]:
    if c in date_all.columns:
        date_all[c] = date_all[c].astype(str).str.lower().isin(["true","1","y"]).astype(int)
    else:
        date_all[c] = 0

cal_cols = ["date","dow","doy","dom","woy","month","quarter","sin_dow","cos_dow",
            "sin_doy","cos_doy","is_dec","is_nov","is_newyear",
            "is_weekend","is_holiday","is_payday","is_school_break","is_rainy_season"]
full = full.merge(date_all[cal_cols], on="date", how="left")
"""),

code("""\
# ─── Promo Features ─────────────────────────────────────────────────
promo = pd.concat([promo_tr, promo_te]).drop_duplicates()
promo = promo.merge(product[["product_id","category"]], on="product_id", how="left")
promo["start_date"] = pd.to_datetime(promo["start_date"])
promo["end_date"]   = pd.to_datetime(promo["end_date"])

# detect B1G1
camp_col = next((c for c in ["campaign_name","campaign_type","promo_type"] if c in promo.columns), None)
promo["is_b1g1"] = 0
if camp_col:
    promo["is_b1g1"] = promo[camp_col].fillna("").str.lower().str.contains("1แถม1|buy1get1|b1g1").astype(int)

p_rows = []
for _, r in promo.iterrows():
    if pd.isna(r["start_date"]) or pd.isna(r.get("category")): continue
    s = max(r["start_date"], pd.Timestamp("2023-01-01"))
    e = min(r["end_date"],   pd.Timestamp("2024-10-31"))
    if e < s: continue
    sid = int(r["store_id"]) if not pd.isna(r.get("store_id")) else None
    for d in pd.date_range(s, e):
        row = {"store_id": sid or -1, "category": r["category"], "date": d,
               "promo_active": 1, "discount_pct": float(r.get("discount_pct",0) or 0),
               "is_b1g1": int(r["is_b1g1"])}
        p_rows.append(row)

if p_rows:
    pf = pd.DataFrame(p_rows)
    pf_grp = pf.groupby(["store_id","category","date"], as_index=False).agg(
        promo_active=("promo_active","max"), promo_count=("promo_active","sum"),
        max_discount=("discount_pct","max"), mean_discount=("discount_pct","mean"),
        is_b1g1=("is_b1g1","max"))
    # global promos (store_id=-1) → broadcast
    global_p = pf_grp[pf_grp["store_id"]==-1].drop(columns="store_id")
    store_p  = pf_grp[pf_grp["store_id"]!=-1]
    full = full.merge(store_p, on=["store_id","category","date"], how="left")
    full = full.merge(global_p, on=["category","date"], how="left", suffixes=("","_g"))
    for c in ["promo_active","promo_count","max_discount","mean_discount","is_b1g1"]:
        gc = c+"_g"
        if gc in full.columns:
            full[c] = full[c].combine_first(full[gc])
            full.drop(columns=gc, inplace=True)
full[["promo_active","promo_count","max_discount","mean_discount","is_b1g1"]] = \
    full[["promo_active","promo_count","max_discount","mean_discount","is_b1g1"]].fillna(0)
"""),

code("""\
# ─── Event Features ─────────────────────────────────────────────────
event = pd.concat([event_tr, event_te]).drop_duplicates()
event["date"] = pd.to_datetime(event["date"])
ev_type = pd.get_dummies(event["event_type"], prefix="evt").astype(int)
event = pd.concat([event[["store_id","date"]], ev_type], axis=1)
ev_agg = event.groupby(["store_id","date"], as_index=False).agg(
    n_events=("store_id","count"), **{c: (c,"max") for c in ev_type.columns})
ev_agg["has_event"] = 1
full = full.merge(ev_agg, on=["store_id","date"], how="left")
full["n_events"]  = full["n_events"].fillna(0)
full["has_event"] = full["has_event"].fillna(0)
for c in [c for c in full.columns if c.startswith("evt_")]:
    full[c] = full[c].fillna(0)
print("Event cols:", [c for c in full.columns if c.startswith("evt_")])
"""),

code("""\
# ─── Stockout Features ──────────────────────────────────────────────
inv = inventory.merge(product[["product_id","category"]], on="product_id", how="left")
inv["date"] = pd.to_datetime(inv["date"])
stockout = inv.groupby(["store_id","category","date"], as_index=False).agg(
    stockout_rate=("is_stockout","mean"),
    avg_closing_stock=("closing_stock","mean"))
full = full.merge(stockout, on=["store_id","category","date"], how="left").fillna(
    {"stockout_rate":0, "avg_closing_stock":0})
"""),

code("""\
# ─── Sorting for lag/rolling ─────────────────────────────────────────
full = full.sort_values(["store_id","category","date"]).reset_index(drop=True)
g = full.groupby(["store_id","category"])

# Category OHE
cat_d = pd.get_dummies(full["category"], prefix="cat").astype(int)
full = pd.concat([full, cat_d], axis=1)

# Interaction features
full["b1g1_x_coffee"]   = full["is_b1g1"] * full.get("cat_Coffee", 0)
full["b1g1_x_tea"]      = full["is_b1g1"] * full.get("cat_Tea", 0)
full["rainy_x_coffee"]  = full["is_rainy_season"] * full.get("cat_Coffee", 0)
full["rainy_x_juice"]   = full["is_rainy_season"] * full.get("cat_Juice & Smoothie", 0)
full["holiday_x_coffee"]= full["is_holiday"] * full.get("cat_Coffee", 0)
full["event_x_bakery"]  = full["has_event"] * full.get("cat_Bakery", 0)
full["mall_x_weekend"]  = full["is_mall"] * full["is_weekend"]
full["tourist_x_holiday"]= full["is_tourist"] * full["is_holiday"]
full["cap_x_weekend"]   = full["seating_capacity"] * full["is_weekend"]
full["payday_x_coffee"] = full["is_payday"] * full.get("cat_Coffee", 0)

# Store-level total (cross-category context)
store_day = full.groupby(["store_id","date"])["units_sold"].sum().reset_index().rename(columns={"units_sold":"store_total"})
store_day["store_total_lag1"]  = store_day.groupby("store_id")["store_total"].shift(1)
store_day["store_total_lag7"]  = store_day.groupby("store_id")["store_total"].shift(7)
store_day["store_roll_mean_7"] = store_day.groupby("store_id")["store_total"].transform(lambda x: x.shift(1).rolling(7,min_periods=1).mean())
full = full.merge(store_day[["store_id","date","store_total_lag1","store_total_lag7","store_roll_mean_7"]],
                  on=["store_id","date"], how="left")
"""),

code("""\
# ─── Product-level features ──────────────────────────────────────────
agg_dict = {
    "product_count": ("product_id", "nunique"),
    "avg_base_price": ("base_price", "mean"),
    "max_base_price": ("base_price", "max"),
}
if "is_seasonal" in product.columns:
    agg_dict["seasonal_rate"] = ("is_seasonal", "mean")
prod_feats = product.groupby("category").agg(**agg_dict).reset_index()
full = full.merge(prod_feats, on="category", how="left")
"""),

code("""\
# ─── Stockout lag features (horizon-specific, computed later) ─────────
# Stored separately for leakage-safe merge in panel builder
print("Base features ready. Full shape:", full.shape)
base_feature_cols = [c for c in full.columns if c not in ["store_id","category","date","units_sold"]]
print(f"Base feature count: {len(base_feature_cols)}")
"""),

md("## Build Horizon-Specific Panels (Leakage-Safe)"),

code("""\
HORIZON_MAP = {"1d":1, "7d":7, "1m":30}

def make_horizon_panel(full, horizon_str, stores, cats, dates):
    H = HORIZON_MAP[horizon_str]
    df = full.copy().sort_values(["store_id","category","date"])

    # ── Strict leakage-safe lags ─────────────────────────────────────
    for k in [0, 1, 2, 7, 14, 21, 28, 35, 56, 84]:
        lag = H + k
        col = f"lag_{lag}"
        df[col] = df.groupby(["store_id","category"])["units_sold"].shift(lag)

    # Annual lags (same week last year → Nov-Dec 2024 maps to Nov-Dec 2023)
    for ann in [364, 365, 366]:
        df[f"lag_{ann}"] = df.groupby(["store_id","category"])["units_sold"].shift(ann)

    # ── Momentum (from friend's 43-feature set) ──────────────────────
    lag_h   = df.groupby(["store_id","category"])["units_sold"].shift(H)
    lag_h7  = df.groupby(["store_id","category"])["units_sold"].shift(H+7)
    lag_h14 = df.groupby(["store_id","category"])["units_sold"].shift(H+14)
    df["lag_ratio_7_14"]  = lag_h7  / (lag_h14  + 1e-5)
    df["lag_diff_7_14"]   = lag_h7  - lag_h14
    df["lag_pct_7_14"]    = (lag_h7 - lag_h14) / (lag_h14 + 1e-5)
    df["lag_diff_h_h7"]   = lag_h   - lag_h7
    df["lag_ratio_h_h7"]  = lag_h   / (lag_h7  + 1e-5)

    # ── Rolling stats (shifted >= H, leakage-safe) ───────────────────
    for w in [7, 14, 28, 56]:
        df[f"roll_mean_{w}"] = df.groupby(["store_id","category"])["units_sold"].transform(
            lambda x, _H=H, _w=w: x.shift(_H).rolling(_w, min_periods=1).mean())
        df[f"roll_std_{w}"]  = df.groupby(["store_id","category"])["units_sold"].transform(
            lambda x, _H=H, _w=w: x.shift(_H).rolling(_w, min_periods=1).std())
        df[f"roll_max_{w}"]  = df.groupby(["store_id","category"])["units_sold"].transform(
            lambda x, _H=H, _w=w: x.shift(_H).rolling(_w, min_periods=1).max())
        df[f"roll_min_{w}"]  = df.groupby(["store_id","category"])["units_sold"].transform(
            lambda x, _H=H, _w=w: x.shift(_H).rolling(_w, min_periods=1).min())
        df[f"roll_nonzero_{w}"] = df.groupby(["store_id","category"])["units_sold"].transform(
            lambda x, _H=H, _w=w: (x.shift(_H) > 0).rolling(_w, min_periods=1).sum())

    # Trend feature (from catboost_summary)
    df["trend_7_vs_28"] = df["roll_mean_7"] - df["roll_mean_28"]
    df["cv_28"]         = df["roll_std_28"] / (df["roll_mean_28"] + 1e-5)
    df["momentum_7_28"] = df["roll_mean_7"] / (df["roll_mean_28"] + 1e-5)
    df["momentum_14_56"]= df["roll_mean_14"] / (df["roll_mean_56"] + 1e-5)

    # ── Stockout lags ────────────────────────────────────────────────
    df["stockout_lag0"]     = df.groupby(["store_id","category"])["stockout_rate"].shift(H)
    df["stockout_lag7"]     = df.groupby(["store_id","category"])["stockout_rate"].shift(H+7)
    df["stockout_roll_28"]  = df.groupby(["store_id","category"])["stockout_rate"].transform(
        lambda x: x.shift(H).rolling(28, min_periods=1).mean())

    # ── Sample weight (stockout rows downweighted) ───────────────────
    df["sample_weight"] = 1.0 - 0.7 * df["stockout_lag0"].fillna(0).clip(0,1)

    df["horizon"] = horizon_str
    df = df.fillna(0)
    return df

panels = {}
for h_str in ["1d","7d","1m"]:
    panels[h_str] = make_horizon_panel(full, h_str, STORES, CATS, DATES)
    ncols = len([c for c in panels[h_str].columns if c not in ["store_id","category","date","units_sold","sample_weight","horizon"]])
    print(f"Horizon {h_str}: {ncols} features")
"""),

md("## AutoGluon Training — Time-Machine Split (Jan-Oct 2023 → Nov-Dec 2023)"),

code("""\
from tqdm.notebook import tqdm
from IPython.display import display, HTML

# AutoGluon requires sample_weight as a COLUMN NAME in the DataFrame (not an array)
EXCLUDE = {"date","units_sold","sample_weight","horizon"}
MODELS  = {}
VAL_RESULTS = {}
LEADERBOARDS = {}

# ⏳ ตั้งค่าเวลาในการรัน (ยิ่งนาน โมเดลยิ่งแม่นยำขึ้น เพราะมีเวลาสร้าง Ensemble หลายชั้น)
TIME_LIMIT_VALID = 900    # 15 นาที ต่อ 1 Horizon (สำหรับ Validation)
TIME_LIMIT_RETRAIN = 900  # 15 นาที ต่อ 1 Horizon (สำหรับ Retrain ข้อมูลทั้งหมด)

# ใช้ tqdm เพื่อแสดง Progress Bar
for h_str in tqdm(["1d","7d","1m"], desc="Training Horizons"):
    print(f"\\n{'='*50}\\nTraining AutoGluon — Horizon: {h_str}\\n{'='*50}")
    df = panels[h_str].copy()
    df["store_id"] = df["store_id"].astype(str)
    df["category"] = df["category"].astype(str)
    H = HORIZON_MAP[h_str]
    
    TRAIN   = df[df["date"] < "2023-11-01"].copy()
    VALID   = df[df["date"].between("2023-11-01","2023-12-31")].copy()
    RETRAIN = df[df["date"] <= "2024-10-31"].copy()

    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    label_col = "units_sold"
    train_cols = feat_cols + [label_col]  # sample_weight not supported in this AG version

    print(f"  Features: {len(feat_cols)}")
    print(f"  Train rows: {len(TRAIN):,}  |  Valid rows: {len(VALID):,}")

    # ── Validation predictor (Time-Machine Split) ────────────────────
    # เพื่อให้สอดคล้องกับ Test set เราต้องเอาคุณสมบัติที่ Cap ที่ 2023-10-31 สำหรับ VALID
    VALID_CAPPED = VALID.copy()
    VALID_CAPPED["decision_date"] = VALID_CAPPED["date"] - pd.to_timedelta(H, unit="D")
    VALID_CAPPED["anchor_date"]   = VALID_CAPPED["decision_date"].clip(upper=pd.Timestamp("2023-10-31"))
    
    # ดึงฟีเจอร์จาก panel ณ วัน anchor_date เพื่อจำลองการทำแท้จริง
    # แปลง store_id/category ของ df_capped_temp ให้เป็น str
    df_temp = df.copy()
    df_temp["store_id"] = df_temp["store_id"].astype(str)
    df_temp["category"] = df_temp["category"].astype(str)
    
    VALID_CAPPED["store_id"] = VALID_CAPPED["store_id"].astype(str)
    VALID_CAPPED["category"] = VALID_CAPPED["category"].astype(str)
    
    merge_cols = list(set(["store_id","category","anchor_date"] + feat_cols))
    VALID_FEATS = VALID_CAPPED[["store_id","category","date","anchor_date"]].merge(
        df_temp.rename(columns={"date":"anchor_date"})[merge_cols],
        on=["store_id","category","anchor_date"],
        how="left"
    ).fillna(0)
    # ใส่ target กลับคืน
    VALID_FEATS[label_col] = VALID[label_col].values

    # อัปเดตข้อมูล Known-Future ของวันคาดการณ์จริง (forecast_date)
    fc_cal = df[cal_cols].copy()
    VALID_FEATS = VALID_FEATS.drop(columns=[c for c in cal_cols if c != "date" and c in VALID_FEATS.columns], errors="ignore")
    VALID_FEATS = VALID_FEATS.merge(fc_cal, on="date", how="left")

    predictor = TabularPredictor(
        label=label_col,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        path=f"ag_v2_{h_str}",
        verbosity=1, # ลดความรกของ Log ระหว่างรัน
    ).fit(
        TRAIN[train_cols],
        presets="best_quality",
        time_limit=TIME_LIMIT_VALID,
        num_gpus=1, # 🚀 เปิดใช้ GPU สำหรับการเทรนความเร็วสูง!
        hyperparameters={
            "GBM": [
                {"objective": "tweedie", "tweedie_variance_power": 1.1, "extra_trees": False},
                {"objective": "tweedie", "tweedie_variance_power": 1.5, "extra_trees": True},
            ],
            "CAT": [
                {"loss_function": "MAE",  "depth": 8},
                {"loss_function": "RMSE", "depth": 6},
            ],
            "XGB": [{"objective": "reg:tweedie", "tweedie_variance_power": 1.2}],
            "NN_TORCH": {},
        },
    )

    # ── Evaluate on capped validation set ────────────────────────────
    val_pred = predictor.predict(VALID_FEATS[feat_cols]).clip(lower=0)
    val_mae  = (val_pred - VALID_FEATS[label_col]).abs().mean()
    VAL_RESULTS[h_str] = val_mae
    print(f"  [Val MAE {h_str}]: {val_mae:.4f}")

    # Per-model leaderboard (top 5)
    lb = predictor.leaderboard(VALID_FEATS[feat_cols + [label_col]], silent=True)
    LEADERBOARDS[h_str] = lb
    print(lb[["model","score_test","score_val"]].head(5).to_string(index=False))
    lb.to_csv(f"leaderboard_{h_str}.csv", index=False)

    # ── Retrain on full Jan2023–Oct2024 ─────────────────────────────
    print("  Retraining on full data...")
    predictor_full = TabularPredictor(
        label=label_col,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        path=f"ag_v2_{h_str}_full",
        verbosity=0,
    ).fit(
        RETRAIN[train_cols],
        presets="best_quality",
        time_limit=TIME_LIMIT_RETRAIN,
        num_gpus=1, # 🚀 เปิดใช้ GPU สำหรับการเทรนความเร็วสูง!
    )
    MODELS[h_str] = (predictor_full, feat_cols)
    print(f"  Retrain complete")

print("\\n=== Validation Summary ===")
for h, v in VAL_RESULTS.items():
    print(f"  {h}: MAE = {v:.4f}")

# 📊 แสดง Benchmark Table
html_out = "<h2>🏆 AutoGluon Benchmark Table</h2>"
for h, lb in LEADERBOARDS.items():
    html_out += f"<h3>Horizon: {h} (Best MAE: {abs(lb['score_test'].iloc[0]):.4f})</h3>"
    # แปลงคะแนนลบของ AutoGluon กลับเป็นบวกเพื่อให้อ่านง่าย
    display_lb = lb[["model", "score_test", "score_val", "fit_time", "pred_time_test"]].copy()
    display_lb["score_test"] = display_lb["score_test"].abs()
    display_lb["score_val"] = display_lb["score_val"].abs()
    display_lb = display_lb.rename(columns={"score_test": "Test MAE", "score_val": "Val MAE"})
    html_out += display_lb.head(5).to_html(index=False, classes="table table-striped table-hover")
display(HTML(html_out))
"""),

md("## Inference + Submission"),

code("""\
def parse_sample(sample):
    out = sample.copy()
    parsed = out["id"].str.extract(r"^(\\d+)_(.+)_(\\d{4}-\\d{2}-\\d{2})_(1d|7d|1m)$")
    out[["store_id","category","forecast_date","horizon"]] = parsed
    out["store_id"]       = out["store_id"].astype(int)
    out["forecast_date"]  = pd.to_datetime(out["forecast_date"])
    out["h"]              = out["horizon"].map(HORIZON_MAP)
    out["decision_date"]  = out["forecast_date"] - pd.to_timedelta(out["h"], unit="D")
    out["anchor_date"]    = out["decision_date"].clip(upper=pd.Timestamp("2024-10-31"))
    return out

test_meta = parse_sample(sample)
submission = sample[["id"]].copy()
submission["units_sold_predicted"] = np.nan

for h_str in ["1d","7d","1m"]:
    H = HORIZON_MAP[h_str]
    predictor_full, feat_cols = MODELS[h_str]
    panel = panels[h_str]

    sub_h = test_meta[test_meta["horizon"] == h_str].copy()

    # Merge panel features at anchor_date
    # เพื่อป้องกันชื่อคอลัมน์ซ้ำซ้อนและเพื่อให้ชนิดข้อมูล (dtype) ตรงกัน
    panel_temp = panel.copy()
    panel_temp["store_id"] = panel_temp["store_id"].astype(str)
    panel_temp["category"] = panel_temp["category"].astype(str)
    
    sub_h["store_id"] = sub_h["store_id"].astype(str)
    sub_h["category"] = sub_h["category"].astype(str)
    
    merge_cols = list(set(["store_id","category","anchor_date"] + feat_cols))
    test_rows = sub_h.merge(
        panel_temp.rename(columns={"date":"anchor_date"})[merge_cols],
        on=["store_id","category","anchor_date"],
        how="left"
    ).fillna(0)

    # Override calendar features to reflect FORECAST date (future-known)
    fc_cal = date_all[cal_cols].rename(columns={"date":"forecast_date"})
    test_rows = test_rows.drop(columns=[c for c in cal_cols if c != "date" and c in test_rows.columns], errors="ignore")
    test_rows = test_rows.merge(fc_cal, on="forecast_date", how="left")

    # บังคับแปลงประเภทของฟีเจอร์ใน test_rows ให้ตรงกับที่โมเดลต้องการ
    test_rows["store_id"] = test_rows["store_id"].astype(str)
    test_rows["category"] = test_rows["category"].astype(str)

    preds = predictor_full.predict(test_rows[feat_cols]).clip(lower=0)
    # บันทึกผลลัพธ์โดยอ้างอิงช่วงดัชนีของ horizon นั้น ๆ
    submission.loc[test_meta["horizon"]==h_str, "units_sold_predicted"] = preds.values

submission["units_sold_predicted"] = submission["units_sold_predicted"].fillna(0).clip(lower=0)
submission.to_csv("submission_autogluon_v2.csv", index=False)
print(f"✅ Saved submission_autogluon_v2.csv  shape={submission.shape}")
print(submission.describe())
"""),
]

path = os.path.join(
    r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon",
    "hackathon-5-autogluon-v2.ipynb"
)
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb(cells), f, indent=2, ensure_ascii=False)
print("[SUCCESS] hackathon-5-autogluon-v2.ipynb generated!")
