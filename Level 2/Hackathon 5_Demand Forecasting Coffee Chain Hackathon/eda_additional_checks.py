"""
Additional EDA checks for the ExpresSo demand forecasting dataset.

This script focuses on EDA areas not fully covered in the main notebook:
- Data integrity and join coverage
- Zero-inflation and sparsity by store-category
- Train vs forecast coverage drift
- Promotion depth and overlap
- Discount depth vs demand response
- Product lifecycle (new vs inactive SKUs)
- Stockout spell lengths
- Customer tenure and member share shifts
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})
sns.set_theme(style="whitegrid")

RUN_DATA_QUALITY = True
RUN_SPARSITY = True
RUN_COVERAGE_DRIFT = True
RUN_PROMO_DEPTH = True
RUN_DISCOUNT_ELASTICITY = True
RUN_PRODUCT_LIFECYCLE = True
RUN_STOCKOUT_SPELLS = True
RUN_CUSTOMER_TENURE = True

OUTPUT_DIR = Path("eda_additional_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_base() -> Path:
    candidates = [
        Path(r"C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\super-ai-engineer-season-6-coffee-chain-hackathon"),
        Path(__file__).resolve().parent / "super-ai-engineer-season-6-coffee-chain-hackathon",
        Path("/kaggle/input/competitions/super-ai-engineer-season-6-coffee-chain-hackathon"),
        Path("/kaggle/input/super-ai-engineer-season-6-coffee-chain-hackathon"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset folder not found. Check the local path or Kaggle input.")


def load_data(base: Path) -> dict:
    train = base / "train"
    test = base / "test"
    data = {
        "txn": pd.read_csv(train / "TRANSACTION.csv"),
        "order": pd.read_csv(train / "ORDER.csv", parse_dates=["date"]),
        "prod": pd.read_csv(train / "PRODUCT.csv"),
        "store": pd.read_csv(train / "STORE.csv", parse_dates=["opened_date"]),
        "promo": pd.read_csv(train / "PROMOTION.csv", parse_dates=["start_date", "end_date"]),
        "cust": pd.read_csv(train / "CUSTOMER.csv", parse_dates=["registration_date"]),
        "date_dim": pd.read_csv(train / "DATE_DIM.csv", parse_dates=["date"]),
        "event": pd.read_csv(train / "LOCAL_EVENT.csv", parse_dates=["date"]),
        "inventory": pd.read_csv(train / "INVENTORY.csv", parse_dates=["date"]),
        "test_date": pd.read_csv(test / "DATE_DIM.csv", parse_dates=["date"]),
        "test_promo": pd.read_csv(test / "PROMOTION.csv", parse_dates=["start_date", "end_date"]),
        "test_event": pd.read_csv(test / "LOCAL_EVENT.csv", parse_dates=["date"]),
        "test_prod": pd.read_csv(test / "PRODUCT.csv"),
        "test_store": pd.read_csv(test / "STORE.csv", parse_dates=["opened_date"]),
    }
    sample_path = base / "sample_submission_with_id.csv"
    data["sample_submission"] = pd.read_csv(sample_path) if sample_path.exists() else None
    return data


def build_line(data: dict) -> pd.DataFrame:
    txn = data["txn"]
    order = data["order"]
    prod = data["prod"]

    order_cols = [
        c for c in ["order_id", "store_id", "date", "customer_id", "hour", "is_member", "payment_method"]
        if c in order.columns
    ]
    prod_cols = [
        c for c in ["product_id", "category", "base_price", "is_seasonal", "is_limited_edition", "serve_type"]
        if c in prod.columns
    ]

    line = txn.merge(order[order_cols], on="order_id", how="left")
    line = line.merge(prod[prod_cols], on="product_id", how="left")
    return line


def build_daily(line: pd.DataFrame) -> pd.DataFrame:
    revenue_col = "revenue" if "revenue" in line.columns else None
    agg_map = {
        "units_sold": ("units_sold", "sum"),
        "n_orders": ("order_id", "nunique"),
        "n_customers": ("customer_id", "nunique"),
    }
    if revenue_col:
        agg_map["revenue"] = ("revenue", "sum")
    else:
        agg_map["revenue"] = ("units_sold", "count")

    daily = (
        line.groupby(["store_id", "category", "date"], observed=True)
        .agg(**agg_map)
        .reset_index()
    )
    return daily


def build_daily_full(daily: pd.DataFrame, data: dict) -> pd.DataFrame:
    store_ids = sorted(data["store"]["store_id"].dropna().unique())
    categories = sorted(data["prod"]["category"].dropna().unique())
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

    idx = pd.MultiIndex.from_product(
        [store_ids, categories, all_dates],
        names=["store_id", "category", "date"],
    )
    out = daily.set_index(["store_id", "category", "date"]).reindex(idx, fill_value=0).reset_index()
    return out


def save_table(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_fig(name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def data_quality_checks(data: dict, line: pd.DataFrame) -> None:
    txn = data["txn"]
    order = data["order"]
    prod = data["prod"]
    store = data["store"]

    results = []

    def dup_count(df: pd.DataFrame, subset, label: str) -> None:
        if isinstance(subset, str):
            subset = [subset]
        if not all(col in df.columns for col in subset):
            return
        dup = int(df.duplicated(subset=subset).sum())
        results.append({"check": f"duplicates_{label}", "count": dup})

    dup_count(order, "order_id", "order_id")
    dup_count(txn, "transaction_id", "transaction_id")
    dup_count(txn, ["order_id", "product_id"], "order_id_product_id")
    dup_count(prod, "product_id", "product_id")
    dup_count(store, "store_id", "store_id")

    if "order_id" in order.columns and "order_id" in txn.columns:
        missing_orders = order.loc[~order["order_id"].isin(txn["order_id"]), "order_id"].nunique()
        missing_txn = txn.loc[~txn["order_id"].isin(order["order_id"]), "order_id"].nunique()
        results.append({"check": "orders_missing_transactions", "count": int(missing_orders)})
        results.append({"check": "transactions_missing_orders", "count": int(missing_txn)})

    if "product_id" in txn.columns and "product_id" in prod.columns:
        missing_prod = txn.loc[~txn["product_id"].isin(prod["product_id"]), "product_id"].nunique()
        results.append({"check": "transactions_missing_product", "count": int(missing_prod)})

    if "store_id" in order.columns and "store_id" in store.columns:
        missing_store = order.loc[~order["store_id"].isin(store["store_id"]), "store_id"].nunique()
        results.append({"check": "orders_missing_store", "count": int(missing_store)})

    for col in ["units_sold", "revenue", "discount_applied"]:
        if col in txn.columns:
            neg = int((txn[col] < 0).sum())
            zero = int((txn[col] == 0).sum())
            results.append({"check": f"neg_{col}", "count": neg})
            results.append({"check": f"zero_{col}", "count": zero})

    if "hour" in order.columns:
        bad_hours = int((~order["hour"].between(0, 23)).sum())
        results.append({"check": "order_hour_out_of_range", "count": bad_hours})

    if "revenue" in line.columns and "base_price" in line.columns and "units_sold" in line.columns:
        denom = (line["units_sold"] * line["base_price"]).replace(0, np.nan)
        ratio = (line["revenue"] / denom).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            pct = ratio.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).reset_index()
            pct.columns = ["quantile", "revenue_to_base_price_ratio"]
            save_table(pct, "revenue_price_ratio_quantiles")

    results_df = pd.DataFrame(results)
    print("Data quality checks:")
    print(results_df.sort_values("count", ascending=False))
    save_table(results_df, "data_quality_checks")


def zero_inflation(daily_full: pd.DataFrame) -> None:
    zero_rate = (
        daily_full.groupby(["store_id", "category"], observed=True)["units_sold"]
        .apply(lambda s: (s == 0).mean())
        .reset_index(name="zero_rate")
    )
    save_table(zero_rate.sort_values("zero_rate", ascending=False), "zero_rate_by_store_category")

    plt.figure(figsize=(10, 4))
    sns.histplot(zero_rate["zero_rate"], bins=30, color="#457b9d")
    plt.title("Zero-rate distribution across store-category")
    plt.xlabel("share of zero-sales days")
    save_fig("zero_rate_distribution")


def parse_submission_id(sample_df: pd.DataFrame) -> pd.DataFrame:
    parsed = sample_df[["id"]].copy()
    pieces = parsed["id"].str.rsplit("_", n=2, expand=True)
    parsed["store_category"] = pieces[0]
    parsed["date"] = pd.to_datetime(pieces[1])
    parsed["horizon"] = pieces[2]
    left = parsed["store_category"].str.split("_", n=1, expand=True)
    parsed["store_id"] = left[0].astype(int)
    parsed["category"] = left[1]
    return parsed.drop(columns=["store_category"])


def expand_promo(promo_df: pd.DataFrame, product_df: pd.DataFrame) -> pd.DataFrame:
    if promo_df.empty:
        return pd.DataFrame(columns=["store_id", "category", "date", "promo_id", "discount_pct", "promo_type"])
    p = promo_df.merge(product_df[["product_id", "category"]], on="product_id", how="left").copy()
    p["date"] = p.apply(lambda r: pd.date_range(r["start_date"], r["end_date"], freq="D"), axis=1)
    p = p.explode("date")
    return p


def coverage_drift(data: dict, daily: pd.DataFrame) -> None:
    sample = data["sample_submission"]
    if sample is not None:
        parsed = parse_submission_id(sample)
        train_keys = daily[["store_id", "category"]].drop_duplicates()
        miss = parsed[["store_id", "category"]].drop_duplicates().merge(
            train_keys,
            on=["store_id", "category"],
            how="left",
            indicator=True,
        )
        missing_keys = miss[miss["_merge"].eq("left_only")][["store_id", "category"]]
        save_table(missing_keys, "submission_pairs_missing_in_train")

    train_event_types = set(data["event"]["event_type"].dropna().astype(str).unique())
    test_event_types = set(data["test_event"]["event_type"].dropna().astype(str).unique())
    new_events = sorted(test_event_types - train_event_types)
    event_df = pd.DataFrame({"new_event_types_in_test": new_events})
    save_table(event_df, "new_event_types_in_test")

    train_promo_types = set(data["promo"]["promo_type"].dropna().astype(str).unique())
    test_promo_types = set(data["test_promo"]["promo_type"].dropna().astype(str).unique())
    new_promos = sorted(test_promo_types - train_promo_types)
    promo_df = pd.DataFrame({"new_promo_types_in_test": new_promos})
    save_table(promo_df, "new_promo_types_in_test")

    train_days = data["date_dim"]["date"].nunique()
    store_count = data["store"]["store_id"].nunique()
    event_daily = data["event"].groupby(["store_id", "date"], observed=True).size().reset_index(name="event_count")
    train_event_rate = event_daily["event_count"].gt(0).sum() / (store_count * train_days)

    forecast_days = data["test_date"]["date"].nunique()
    test_store_count = data["test_store"]["store_id"].nunique()
    test_event_daily = data["test_event"].groupby(["store_id", "date"], observed=True).size().reset_index(name="event_count")
    test_event_rate = test_event_daily["event_count"].gt(0).sum() / (test_store_count * forecast_days)

    drift_summary = pd.DataFrame(
        [
            {"metric": "event_rate_train", "value": train_event_rate},
            {"metric": "event_rate_forecast", "value": test_event_rate},
        ]
    )
    save_table(drift_summary, "event_rate_train_vs_forecast")

    promo_daily_train = expand_promo(data["promo"], data["prod"])
    promo_daily_test = expand_promo(data["test_promo"], data["test_prod"])

    train_promo_rate = len(promo_daily_train) / (store_count * train_days * data["prod"]["category"].nunique())
    test_promo_rate = len(promo_daily_test) / (test_store_count * forecast_days * data["test_prod"]["category"].nunique())

    promo_rate_df = pd.DataFrame(
        [
            {"metric": "promo_rate_train", "value": train_promo_rate},
            {"metric": "promo_rate_forecast", "value": test_promo_rate},
        ]
    )
    save_table(promo_rate_df, "promo_rate_train_vs_forecast")


def promo_depth_and_overlap(data: dict) -> None:
    promo = data["promo"].copy()
    promo["duration_days"] = (promo["end_date"] - promo["start_date"]).dt.days + 1
    save_table(promo[["promo_id", "duration_days", "discount_pct", "promo_type"]], "promo_duration_summary")

    promo_daily = expand_promo(promo, data["prod"])
    if promo_daily.empty:
        return

    store_day = (
        promo_daily.groupby(["store_id", "date"], observed=True)
        .agg(
            promo_product_count=("product_id", "nunique"),
            promo_campaign_count=("promo_id", "nunique"),
            max_discount=("discount_pct", "max"),
            mean_discount=("discount_pct", "mean"),
        )
        .reset_index()
    )
    save_table(store_day.sort_values("promo_campaign_count", ascending=False), "promo_overlap_store_day")

    plt.figure(figsize=(10, 4))
    sns.histplot(store_day["promo_campaign_count"], bins=20, color="#2a9d8f")
    plt.title("Promo overlap per store-day")
    plt.xlabel("number of active promos")
    save_fig("promo_overlap_hist")


def discount_elasticity(data: dict, daily: pd.DataFrame) -> None:
    promo_daily = expand_promo(data["promo"], data["prod"])
    if promo_daily.empty:
        return

    promo_cat = (
        promo_daily.groupby(["store_id", "category", "date"], observed=True)
        .agg(max_discount=("discount_pct", "max"), mean_discount=("discount_pct", "mean"))
        .reset_index()
    )

    merged = daily.merge(promo_cat, on=["store_id", "category", "date"], how="left")
    merged["max_discount"] = merged["max_discount"].fillna(0)

    bins = [-0.1, 0, 5, 10, 20, 35, 50, 100]
    labels = ["0", "0-5", "5-10", "10-20", "20-35", "35-50", ">50"]
    merged["discount_bin"] = pd.cut(merged["max_discount"], bins=bins, labels=labels)

    discount_summary = (
        merged.groupby(["category", "discount_bin"], observed=True)
        .agg(avg_units=("units_sold", "mean"), median_units=("units_sold", "median"), n_days=("units_sold", "size"))
        .reset_index()
    )
    save_table(discount_summary, "discount_elasticity_by_category")

    plt.figure(figsize=(12, 6))
    pivot = discount_summary.pivot(index="category", columns="discount_bin", values="avg_units")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Average units sold by discount bin and category")
    save_fig("discount_elasticity_heatmap")


def product_lifecycle(line: pd.DataFrame, data: dict) -> None:
    if "date" not in line.columns:
        return
    prod = data["prod"]
    train_end = line["date"].max()

    sku = (
        line.groupby(["product_id", "category"], observed=True)
        .agg(
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            units=("units_sold", "sum"),
        )
        .reset_index()
    )
    sku["days_active"] = (sku["last_sale"] - sku["first_sale"]).dt.days + 1
    sku["days_since_last_sale"] = (train_end - sku["last_sale"]).dt.days

    recent_new = sku[sku["first_sale"] >= train_end - pd.Timedelta(days=90)]
    recently_inactive = sku[sku["days_since_last_sale"] >= 90]

    save_table(sku.sort_values("units", ascending=False), "sku_lifecycle_summary")
    save_table(recent_new, "sku_new_last_90_days")
    save_table(recently_inactive, "sku_inactive_90_days")

    plt.figure(figsize=(10, 4))
    sns.histplot(sku["days_since_last_sale"], bins=30, color="#e76f51")
    plt.title("Days since last sale (SKU)")
    plt.xlabel("days since last sale")
    save_fig("sku_days_since_last_sale")

    if "is_seasonal" in prod.columns:
        seasonal_ids = prod.loc[prod["is_seasonal"].astype(bool), "product_id"]
        seasonal_share = sku.assign(is_seasonal=sku["product_id"].isin(seasonal_ids))
        seasonal_share = seasonal_share.groupby("category", observed=True)["is_seasonal"].mean().reset_index()
        save_table(seasonal_share, "category_seasonal_sku_share")


def stockout_spells(data: dict) -> None:
    inv = data["inventory"].copy()
    if "is_stockout" not in inv.columns:
        return

    inv["is_stockout_int"] = inv["is_stockout"].astype(int)
    inv = inv.sort_values(["store_id", "product_id", "date"])
    inv["spell_id"] = (
        inv.groupby(["store_id", "product_id"], observed=True)["is_stockout_int"]
        .diff()
        .ne(0)
        .cumsum()
    )
    spells = inv[inv["is_stockout_int"].eq(1)].groupby(
        ["store_id", "product_id", "spell_id"], observed=True
    ).size().reset_index(name="stockout_days")

    save_table(spells.sort_values("stockout_days", ascending=False), "stockout_spells")

    plt.figure(figsize=(10, 4))
    sns.histplot(spells["stockout_days"], bins=30, color="#264653")
    plt.title("Stockout spell length distribution")
    plt.xlabel("days in stockout")
    save_fig("stockout_spell_length_hist")


def customer_tenure(line: pd.DataFrame, data: dict) -> None:
    order = data["order"].copy()
    cust = data["cust"].copy()

    if "customer_id" not in order.columns or "customer_id" not in cust.columns:
        return

    order_units = line.groupby("order_id", observed=True)["units_sold"].sum().reset_index(name="units_per_order")
    merged = order.merge(order_units, on="order_id", how="left")
    merged = merged.merge(cust[["customer_id", "registration_date"]], on="customer_id", how="left")
    merged = merged[merged["customer_id"].notna()].copy()

    merged["tenure_days"] = (merged["date"] - merged["registration_date"]).dt.days
    merged = merged[merged["tenure_days"].notna()]

    bins = [-1, 30, 90, 180, 365, 730, 5000]
    labels = ["0-30", "31-90", "91-180", "181-365", "366-730", ">730"]
    merged["tenure_bin"] = pd.cut(merged["tenure_days"], bins=bins, labels=labels)

    tenure_summary = (
        merged.groupby("tenure_bin", observed=True)
        .agg(
            orders=("order_id", "nunique"),
            avg_units_per_order=("units_per_order", "mean"),
            member_share=("is_member", "mean"),
        )
        .reset_index()
    )
    save_table(tenure_summary, "customer_tenure_summary")

    plt.figure(figsize=(10, 4))
    sns.barplot(data=tenure_summary, x="tenure_bin", y="orders", color="#1d3557")
    plt.title("Orders by customer tenure")
    plt.xlabel("tenure (days)")
    save_fig("orders_by_tenure")


def main() -> None:
    base = find_base()
    data = load_data(base)
    line = build_line(data)
    daily = build_daily(line)
    daily_full = build_daily_full(daily, data)

    if RUN_DATA_QUALITY:
        data_quality_checks(data, line)
    if RUN_SPARSITY:
        zero_inflation(daily_full)
    if RUN_COVERAGE_DRIFT:
        coverage_drift(data, daily)
    if RUN_PROMO_DEPTH:
        promo_depth_and_overlap(data)
    if RUN_DISCOUNT_ELASTICITY:
        discount_elasticity(data, daily)
    if RUN_PRODUCT_LIFECYCLE:
        product_lifecycle(line, data)
    if RUN_STOCKOUT_SPELLS:
        stockout_spells(data)
    if RUN_CUSTOMER_TENURE:
        customer_tenure(line, data)


if __name__ == "__main__":
    main()
