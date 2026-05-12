from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "super-ai-engineer-season-6-coffee-chain-hackathon"
TRAIN = DATA_ROOT / "train"
TEST = DATA_ROOT / "test"
OUT = HERE / "EDA_DEEP_DIVE_Codex.md"


CATEGORY_ORDER = [
    "Coffee",
    "Tea",
    "Chocolate & Milk",
    "Juice & Smoothie",
    "Bakery",
    "Savory Bakery",
    "Merchandise",
]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def boolish(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fmt_num(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:,.{digits}f}"
    return str(value)


def pct(numer: float, denom: float, digits: int = 1) -> str:
    if not denom:
        return "-"
    return f"{100 * numer / denom:,.{digits}f}%"


def lift_pct(base: float, new: float) -> float | None:
    if not base:
        return None
    return (new - base) / base * 100


def table(headers: list[str], rows: list[list[object]], digits: int = 1) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_num(x, digits) for x in row) + " |")
    return "\n".join(lines)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def add_agg(store: dict, key, value: float, count: int = 1):
    item = store[key]
    item[0] += value
    item[1] += count


def mean_from_agg(item: list[float]) -> float:
    return item[0] / item[1] if item[1] else 0.0


def main() -> None:
    row_counts: dict[str, int] = {}
    schemas: dict[str, list[str]] = {}
    for split_dir in [TRAIN, TEST]:
        for path in sorted(split_dir.glob("*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = sum(1 for _ in reader)
            key = f"{split_dir.name}/{path.name}"
            row_counts[key] = rows
            schemas[key] = header

    store: dict[int, dict] = {}
    for r in read_csv(TRAIN / "STORE.csv"):
        sid = int(r["store_id"])
        open_h, open_m = map(int, r["open_time"].split(":"))
        close_h, close_m = map(int, r["close_time"].split(":"))
        operating_hours = (close_h + close_m / 60) - (open_h + open_m / 60)
        store[sid] = {
            **r,
            "store_id": sid,
            "seating_capacity": int(r["seating_capacity"]),
            "has_drive_through": boolish(r["has_drive_through"]),
            "staff_count": int(r["staff_count"]),
            "opened_date": parse_date(r["opened_date"]),
            "open_hour_float": open_h + open_m / 60,
            "close_hour_float": close_h + close_m / 60,
            "operating_hours": operating_hours,
            "staff_per_seat": int(r["staff_count"]) / int(r["seating_capacity"]),
        }

    products: dict[int, dict] = {}
    category_seen = []
    for r in read_csv(TRAIN / "PRODUCT.csv"):
        pid = int(r["product_id"])
        cat = r["category"]
        if cat not in category_seen:
            category_seen.append(cat)
        products[pid] = {
            **r,
            "product_id": pid,
            "base_price": float(r["base_price"]),
            "is_seasonal": boolish(r["is_seasonal"]),
            "is_limited_edition": boolish(r["is_limited_edition"]),
        }
    categories = [c for c in CATEGORY_ORDER if c in category_seen]
    category_idx = {c: i for i, c in enumerate(categories)}

    date_dim: dict[str, dict] = {}
    for r in read_csv(TRAIN / "DATE_DIM.csv"):
        date_dim[r["date"]] = {
            **r,
            "date": parse_date(r["date"]),
            "month": int(r["month"]),
            "year": int(r["year"]),
            "week_number": int(r["week_number"]),
            "is_weekend": boolish(r["is_weekend"]),
            "is_holiday": boolish(r["is_holiday"]),
            "is_school_break": boolish(r["is_school_break"]),
            "is_payday": boolish(r["is_payday"]),
            "is_rainy_season": boolish(r["is_rainy_season"]),
        }

    customers: dict[int, dict] = {}
    preferred_store_count = Counter()
    local_customer_count = Counter()
    home_neighborhood_count = Counter()
    for r in read_csv(TRAIN / "CUSTOMER.csv"):
        cid = int(float(r["customer_id"]))
        psid = int(r["preferred_store_id"])
        home = r["home_neighborhood_type"]
        customers[cid] = {
            **r,
            "customer_id": cid,
            "preferred_store_id": psid,
            "registration_date": parse_date(r["registration_date"]),
        }
        preferred_store_count[psid] += 1
        home_neighborhood_count[home] += 1
        if store[psid]["neighborhood_type"] == home:
            local_customer_count[psid] += 1

    order_info: dict[int, tuple[int, str, int, int | None, bool, str]] = {}
    order_dates: list[date] = []
    order_count_by_store_date = Counter()
    order_count_by_store = Counter()
    order_count_by_hour = Counter()
    order_count_by_neighborhood_hour = Counter()
    order_count_by_weekend_hour = Counter()
    walkin_orders = 0
    member_orders = 0
    payment_order_count = Counter()
    customer_order_count = Counter()
    customer_store_count = Counter()
    for r in read_csv(TRAIN / "ORDER.csv"):
        oid = int(r["order_id"])
        sid = int(r["store_id"])
        dstr = r["date"]
        d = parse_date(dstr)
        hour = int(r["hour"])
        customer_id = int(float(r["customer_id"])) if r["customer_id"].strip() else None
        is_member = boolish(r["is_member"])
        payment = r["payment_method"]
        order_info[oid] = (sid, dstr, hour, customer_id, is_member, payment)
        order_dates.append(d)
        order_count_by_store_date[(sid, dstr)] += 1
        order_count_by_store[sid] += 1
        order_count_by_hour[hour] += 1
        order_count_by_neighborhood_hour[(store[sid]["neighborhood_type"], hour)] += 1
        order_count_by_weekend_hour[(date_dim[dstr]["is_weekend"], hour)] += 1
        payment_order_count[payment] += 1
        if customer_id is None:
            walkin_orders += 1
        else:
            member_orders += 1
            customer_order_count[customer_id] += 1
            customer_store_count[(customer_id, sid)] += 1

    train_start = min(order_dates)
    train_end = max(order_dates)
    train_dates = [d.isoformat() for d in daterange(train_start, train_end)]
    forecast_start = date(2024, 11, 1)
    forecast_end = date(2024, 12, 31)
    forecast_dates = [d.isoformat() for d in daterange(forecast_start, forecast_end)]
    stores = sorted(store)
    full_cell_count = len(stores) * len(categories) * len(train_dates)
    forecast_cell_count = len(stores) * len(categories) * len(forecast_dates)

    daily_units = Counter()
    daily_revenue = defaultdict(float)
    daily_member_units = Counter()
    daily_walkin_units = Counter()
    sku_units = Counter()
    sku_revenue = defaultdict(float)
    sku_day_units = Counter()
    category_units = Counter()
    category_revenue = defaultdict(float)
    category_discount_units = Counter()
    category_discounted_line_count = Counter()
    discount_bin_units = Counter()
    discount_bin_lines = Counter()
    order_units = Counter()
    order_revenue = defaultdict(float)
    order_masks = Counter()
    order_line_count = Counter()
    line_count = 0
    revenue_total = 0.0
    units_total = 0
    for r in read_csv(TRAIN / "TRANSACTION.csv"):
        line_count += 1
        oid = int(r["order_id"])
        pid = int(r["product_id"])
        units = int(r["units_sold"])
        discount = float(r["discount_applied"])
        revenue = float(r["revenue"])
        sid, dstr, _hour, customer_id, _is_member, _payment = order_info[oid]
        product = products[pid]
        cat = product["category"]
        daily_units[(sid, cat, dstr)] += units
        daily_revenue[(sid, cat, dstr)] += revenue
        sku_units[pid] += units
        sku_revenue[pid] += revenue
        sku_day_units[(sid, pid, dstr)] += units
        category_units[cat] += units
        category_revenue[cat] += revenue
        if discount > 0:
            category_discount_units[cat] += units
            category_discounted_line_count[cat] += 1
        if discount == 0:
            dbin = "0%"
        elif discount <= 10:
            dbin = "1-10%"
        elif discount <= 20:
            dbin = "11-20%"
        elif discount <= 30:
            dbin = "21-30%"
        elif discount <= 40:
            dbin = "31-40%"
        else:
            dbin = "41%+"
        discount_bin_units[dbin] += units
        discount_bin_lines[dbin] += 1
        order_units[oid] += units
        order_revenue[oid] += revenue
        order_masks[oid] |= 1 << category_idx[cat]
        order_line_count[oid] += 1
        if customer_id is None:
            daily_walkin_units[(sid, cat, dstr)] += units
        else:
            daily_member_units[(sid, cat, dstr)] += units
        revenue_total += revenue
        units_total += units

    target_values = []
    store_total_units = Counter()
    store_total_revenue = defaultdict(float)
    store_total_orders = Counter()
    store_day_units = Counter()
    category_day_units = Counter()
    month_category_units = Counter()
    month_category_cells = Counter()
    month_flag_units = Counter()
    month_flag_total_units = Counter()
    baseline_sum_count = defaultdict(lambda: [0.0, 0])

    for dstr in train_dates:
        dim = date_dim[dstr]
        month_key = f"{dim['year']}-{dim['month']:02d}"
        dow = dim["day_of_week"]
        for sid in stores:
            store_total_orders[sid] += order_count_by_store_date[(sid, dstr)]
            for cat in categories:
                units = daily_units[(sid, cat, dstr)]
                revenue = daily_revenue[(sid, cat, dstr)]
                target_values.append(units)
                store_total_units[sid] += units
                store_total_revenue[sid] += revenue
                store_day_units[(sid, dstr)] += units
                category_day_units[(cat, dstr)] += units
                month_category_units[(month_key, cat)] += units
                month_category_cells[(month_key, cat)] += 1
                add_agg(baseline_sum_count, (sid, cat, dow), units)

    target_values_sorted = sorted(target_values)
    def quantile(q: float) -> float:
        if not target_values_sorted:
            return 0.0
        idx = (len(target_values_sorted) - 1) * q
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return float(target_values_sorted[lo])
        return target_values_sorted[lo] * (hi - idx) + target_values_sorted[hi] * (idx - lo)

    baseline_mean = {
        key: mean_from_agg(value) for key, value in baseline_sum_count.items()
    }

    def rel_value(sid: int, cat: str, dstr: str, units: float) -> float:
        base = baseline_mean[(sid, cat, date_dim[dstr]["day_of_week"])]
        return units / base if base > 0 else 0.0

    factor_raw = defaultdict(lambda: [0.0, 0])
    factor_rel = defaultdict(lambda: [0.0, 0])
    factor_cat_raw = defaultdict(lambda: [0.0, 0])
    factor_cat_rel = defaultdict(lambda: [0.0, 0])
    flag_cols = ["is_weekend", "is_holiday", "is_payday", "is_rainy_season", "is_school_break"]
    for dstr in train_dates:
        dim = date_dim[dstr]
        for sid in stores:
            for cat in categories:
                units = daily_units[(sid, cat, dstr)]
                rel = rel_value(sid, cat, dstr, units)
                for flag in flag_cols:
                    state = dim[flag]
                    add_agg(factor_raw, (flag, state), units)
                    add_agg(factor_rel, (flag, state), rel)
                    add_agg(factor_cat_raw, (flag, cat, state), units)
                    add_agg(factor_cat_rel, (flag, cat, state), rel)

    holiday_rel = defaultdict(lambda: [0.0, 0])
    holiday_raw = defaultdict(lambda: [0.0, 0])
    holiday_dates = defaultdict(set)
    for dstr in train_dates:
        hname = date_dim[dstr]["holiday_name"].strip()
        if not hname:
            continue
        holiday_dates[hname].add(dstr)
        for sid in stores:
            for cat in categories:
                units = daily_units[(sid, cat, dstr)]
                add_agg(holiday_raw, (hname, cat), units)
                add_agg(holiday_rel, (hname, cat), rel_value(sid, cat, dstr, units))

    promo_daily = defaultdict(
        lambda: {
            "count": 0,
            "max_discount": 0.0,
            "types": set(),
            "email": False,
            "social": False,
            "products": set(),
        }
    )
    promo_type_rows = Counter()
    promo_type_discount = defaultdict(lambda: [0.0, 0])
    promo_duration = []
    for r in read_csv(TRAIN / "PROMOTION.csv"):
        pid = int(r["product_id"])
        sid = int(r["store_id"])
        cat = products[pid]["category"]
        start = parse_date(r["start_date"])
        end = parse_date(r["end_date"])
        ptype = r["promo_type"]
        discount = float(r["discount_pct"])
        promo_type_rows[ptype] += 1
        add_agg(promo_type_discount, ptype, discount)
        promo_duration.append((end - start).days + 1)
        for d in daterange(start, end):
            dstr = d.isoformat()
            if dstr not in date_dim:
                continue
            item = promo_daily[(sid, cat, dstr)]
            item["count"] += 1
            item["max_discount"] = max(item["max_discount"], discount)
            item["types"].add(ptype)
            item["email"] = item["email"] or boolish(r["email_sent"])
            item["social"] = item["social"] or boolish(r["social_campaign"])
            item["products"].add(pid)

    promo_raw = defaultdict(lambda: [0.0, 0])
    promo_rel = defaultdict(lambda: [0.0, 0])
    promo_cat_raw = defaultdict(lambda: [0.0, 0])
    promo_cat_rel = defaultdict(lambda: [0.0, 0])
    promo_type_effect = defaultdict(lambda: [0.0, 0])
    promo_channel_effect = defaultdict(lambda: [0.0, 0])
    discount_effect = defaultdict(lambda: [0.0, 0])
    train_promo_cell_count = 0
    forecast_promo_cell_count = 0
    forecast_promo_by_cat = Counter()
    train_promo_by_cat = Counter()
    for dstr in train_dates:
        for sid in stores:
            for cat in categories:
                units = daily_units[(sid, cat, dstr)]
                rel = rel_value(sid, cat, dstr, units)
                pdata = promo_daily.get((sid, cat, dstr))
                active = pdata is not None
                if active:
                    train_promo_cell_count += 1
                    train_promo_by_cat[cat] += 1
                add_agg(promo_raw, active, units)
                add_agg(promo_rel, active, rel)
                add_agg(promo_cat_raw, (cat, active), units)
                add_agg(promo_cat_rel, (cat, active), rel)
                if pdata:
                    for ptype in pdata["types"]:
                        add_agg(promo_type_effect, (ptype, cat), rel)
                    channel = (
                        "email+social" if pdata["email"] and pdata["social"]
                        else "email_only" if pdata["email"]
                        else "social_only" if pdata["social"]
                        else "no_channel_flag"
                    )
                    add_agg(promo_channel_effect, (channel, cat), rel)
                    md = pdata["max_discount"]
                    if md <= 10:
                        dbin = "10%"
                    elif md <= 20:
                        dbin = "20%"
                    elif md <= 30:
                        dbin = "30%"
                    elif md <= 40:
                        dbin = "40%"
                    else:
                        dbin = "50%"
                    add_agg(discount_effect, (dbin, cat), rel)

    for dstr in forecast_dates:
        for sid in stores:
            for cat in categories:
                if (sid, cat, dstr) in promo_daily:
                    forecast_promo_cell_count += 1
                    forecast_promo_by_cat[cat] += 1

    drink_promo_dates = {
        (sid, dstr)
        for (sid, cat, dstr), value in promo_daily.items()
        if cat in {"Coffee", "Tea"} and train_start.isoformat() <= dstr <= train_end.isoformat()
    }
    merch_cannibal = defaultdict(lambda: [0.0, 0])
    for dstr in train_dates:
        for sid in stores:
            active = (sid, dstr) in drink_promo_dates
            units = daily_units[(sid, "Merchandise", dstr)]
            rel = rel_value(sid, "Merchandise", dstr, units)
            add_agg(merch_cannibal, active, rel)

    event_daily = defaultdict(list)
    event_type_counter = Counter()
    forecast_event_counter = Counter()
    train_event_counter = Counter()
    for r in read_csv(TRAIN / "LOCAL_EVENT.csv"):
        sid = int(r["store_id"])
        dstr = r["date"]
        etype = r["event_type"]
        event_daily[(sid, dstr)].append(etype)
        event_type_counter[etype] += 1
        if train_start.isoformat() <= dstr <= train_end.isoformat():
            train_event_counter[etype] += 1
        if forecast_start.isoformat() <= dstr <= forecast_end.isoformat():
            forecast_event_counter[etype] += 1

    event_raw = defaultdict(lambda: [0.0, 0])
    event_rel = defaultdict(lambda: [0.0, 0])
    event_cat_rel = defaultdict(lambda: [0.0, 0])
    event_type_cat_rel = defaultdict(lambda: [0.0, 0])
    for dstr in train_dates:
        for sid in stores:
            etypes = event_daily.get((sid, dstr), [])
            has_event = bool(etypes)
            for cat in categories:
                units = daily_units[(sid, cat, dstr)]
                rel = rel_value(sid, cat, dstr, units)
                add_agg(event_raw, has_event, units)
                add_agg(event_rel, has_event, rel)
                add_agg(event_cat_rel, (cat, has_event), rel)
                for etype in set(etypes):
                    add_agg(event_type_cat_rel, (etype, cat), rel)

    inv_rows = 0
    inv_stockout = 0
    stockout_store = defaultdict(lambda: [0, 0])
    stockout_cat = defaultdict(lambda: [0, 0])
    stockout_month_cat = defaultdict(lambda: [0, 0])
    inv_match = 0
    inv_abs_diff = 0
    inv_compared = 0
    stockout_sales_rel = defaultdict(lambda: [0.0, 0])
    for r in read_csv(TRAIN / "INVENTORY.csv"):
        inv_rows += 1
        sid = int(r["store_id"])
        pid = int(r["product_id"])
        dstr = r["date"]
        cat = products[pid]["category"]
        is_stockout = boolish(r["is_stockout"])
        inv_units = int(r["units_sold"])
        txn_units = sku_day_units[(sid, pid, dstr)]
        inv_compared += 1
        inv_abs_diff += abs(inv_units - txn_units)
        if inv_units == txn_units:
            inv_match += 1
        stockout_store[sid][1] += 1
        stockout_cat[cat][1] += 1
        dim = date_dim[dstr]
        stockout_month_cat[(f"{dim['year']}-{dim['month']:02d}", cat)][1] += 1
        if is_stockout:
            inv_stockout += 1
            stockout_store[sid][0] += 1
            stockout_cat[cat][0] += 1
            stockout_month_cat[(f"{dim['year']}-{dim['month']:02d}", cat)][0] += 1

    # Approximate store-category stockout exposure: any product in category stocked out that day.
    stockout_cell = set()
    for r in read_csv(TRAIN / "INVENTORY.csv"):
        if boolish(r["is_stockout"]):
            sid = int(r["store_id"])
            pid = int(r["product_id"])
            cat = products[pid]["category"]
            stockout_cell.add((sid, cat, r["date"]))
    for dstr in train_dates:
        for sid in stores:
            for cat in categories:
                active = (sid, cat, dstr) in stockout_cell
                rel = rel_value(sid, cat, dstr, daily_units[(sid, cat, dstr)])
                add_agg(stockout_sales_rel, active, rel)

    order_hour_units = defaultdict(lambda: [0.0, 0])
    order_neighborhood_hour_units = defaultdict(lambda: [0.0, 0])
    member_order_agg = defaultdict(lambda: [0.0, 0, 0.0])
    payment_agg = defaultdict(lambda: [0, 0.0, 0])
    basket_mask_count = Counter()
    category_count_per_order = Counter()
    coffee_orders = 0
    coffee_with_bakery = 0
    coffee_with_savory = 0
    for oid, units in order_units.items():
        sid, dstr, hour, customer_id, is_member, payment = order_info[oid]
        revenue = order_revenue[oid]
        ncat = int(order_masks[oid].bit_count())
        category_count_per_order[ncat] += 1
        basket_mask_count[order_masks[oid]] += 1
        add_agg(order_hour_units, hour, units)
        add_agg(order_neighborhood_hour_units, (store[sid]["neighborhood_type"], hour), units)
        member_key = "member" if customer_id is not None and is_member else "walk_in"
        member_order_agg[member_key][0] += units
        member_order_agg[member_key][1] += 1
        member_order_agg[member_key][2] += revenue
        payment_agg[payment][0] += 1
        payment_agg[payment][1] += revenue
        payment_agg[payment][2] += units
        mask = order_masks[oid]
        if mask & (1 << category_idx["Coffee"]):
            coffee_orders += 1
            if mask & (1 << category_idx["Bakery"]):
                coffee_with_bakery += 1
            if mask & (1 << category_idx["Savory Bakery"]):
                coffee_with_savory += 1

    store_rows = []
    for sid in stores:
        days = len(train_dates)
        orders_per_day = store_total_orders[sid] / days
        units_per_day = store_total_units[sid] / days
        rows = {
            "store_id": sid,
            "type": store[sid]["neighborhood_type"],
            "units_per_day": units_per_day,
            "orders_per_day": orders_per_day,
            "units_per_staff_day": units_per_day / store[sid]["staff_count"],
            "orders_per_seat_day": orders_per_day / store[sid]["seating_capacity"],
            "capacity": store[sid]["seating_capacity"],
            "staff": store[sid]["staff_count"],
            "hours": store[sid]["operating_hours"],
            "drive": store[sid]["has_drive_through"],
            "age_days": (train_end - store[sid]["opened_date"]).days,
            "loyal_customers": preferred_store_count[sid],
            "local_loyal_customers": local_customer_count[sid],
        }
        store_rows.append(rows)

    corr_capacity = pearson(
        [r["capacity"] for r in store_rows],
        [r["units_per_day"] for r in store_rows],
    )
    corr_staff = pearson(
        [r["staff"] for r in store_rows],
        [r["units_per_day"] for r in store_rows],
    )
    corr_hours = pearson(
        [r["hours"] for r in store_rows],
        [r["units_per_day"] for r in store_rows],
    )
    corr_loyal = pearson(
        [r["loyal_customers"] for r in store_rows],
        [r["units_per_day"] for r in store_rows],
    )

    neighborhood_agg = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for r in store_rows:
        item = neighborhood_agg[r["type"]]
        item[0] += r["units_per_day"]
        item[1] += r["orders_per_day"]
        item[2] += 1
        item[3] += r["capacity"]
        item[4] += r["hours"]

    month_rows = []
    for cat in categories:
        u23 = c23 = u24 = c24 = 0
        for (month_key, c), units in month_category_units.items():
            if c != cat:
                continue
            year, month = map(int, month_key.split("-"))
            cells = month_category_cells[(month_key, c)]
            if month <= 10 and year == 2023:
                u23 += units
                c23 += cells
            if month <= 10 and year == 2024:
                u24 += units
                c24 += cells
        avg23 = u23 / c23 if c23 else 0
        avg24 = u24 / c24 if c24 else 0
        month_rows.append([cat, avg23, avg24, lift_pct(avg23, avg24)])

    recent_rows = []
    for cat in categories:
        may_jul_u = may_jul_c = aug_oct_u = aug_oct_c = 0
        for (month_key, c), units in month_category_units.items():
            if c != cat:
                continue
            year, month = map(int, month_key.split("-"))
            if year != 2024:
                continue
            cells = month_category_cells[(month_key, c)]
            if 5 <= month <= 7:
                may_jul_u += units
                may_jul_c += cells
            if 8 <= month <= 10:
                aug_oct_u += units
                aug_oct_c += cells
        a = may_jul_u / may_jul_c if may_jul_c else 0
        b = aug_oct_u / aug_oct_c if aug_oct_c else 0
        recent_rows.append([cat, a, b, lift_pct(a, b)])

    seasonal_units = Counter()
    seasonal_total = Counter()
    for pid, units in sku_units.items():
        p = products[pid]
        key = "seasonal" if p["is_seasonal"] else "regular"
        seasonal_units[(p["category"], key)] += units
        seasonal_total[p["category"]] += units

    top_sku_rows = []
    for pid, units in sku_units.most_common(12):
        p = products[pid]
        top_sku_rows.append([
            pid,
            p["product_name"],
            p["category"],
            p["serve_type"],
            p["base_price"],
            units,
            sku_revenue[pid],
            "Y" if p["is_seasonal"] else "",
            "Y" if p["is_limited_edition"] else "",
        ])

    cust_counts = list(customer_order_count.values())
    one_time = sum(1 for x in cust_counts if x == 1)
    repeat = sum(1 for x in cust_counts if x >= 2)
    high_freq = sum(1 for x in cust_counts if x >= 10)
    active_members = len(cust_counts)

    forecast_holidays = [
        (dstr, date_dim[dstr]["day_of_week"], date_dim[dstr]["holiday_name"])
        for dstr in forecast_dates
        if date_dim[dstr]["is_holiday"]
    ]
    forecast_paydays = [dstr for dstr in forecast_dates if date_dim[dstr]["is_payday"]]
    forecast_events = [
        [dstr, sid, ",".join(sorted(set(event_daily[(sid, dstr)])))]
        for (sid, dstr) in sorted(event_daily)
        if forecast_start.isoformat() <= dstr <= forecast_end.isoformat()
    ]

    hash_rows = []
    for name in ["DATE_DIM.csv", "STORE.csv", "PRODUCT.csv", "PROMOTION.csv", "LOCAL_EVENT.csv"]:
        hash_rows.append([
            name,
            "same" if sha256(TRAIN / name) == sha256(TEST / name) else "different",
            row_counts[f"train/{name}"],
            row_counts[f"test/{name}"],
        ])

    lines: list[str] = []
    lines.append("# ExpresSo Demand Forecasting - Deep EDA Add-on (Codex)")
    lines.append("")
    lines.append(
        "รายงานนี้ต่อยอดจาก `EDA_SUMMARY_จากเพื่อน.md` และ notebook เดิม โดยยึดข้อมูลภายใน dataset เป็นหลัก "
        "และคำนวณ effect หลายตัวแบบ normalized เทียบกับ baseline `store × category × day_of_week` "
        "เพื่อลด bias จากร้าน/หมวดที่ยอดขายสูงอยู่แล้ว"
    )
    lines.append("")
    lines.append("## 0) Ground Rules จากโจทย์")
    lines.append("")
    lines.append(
        "- Target ที่ถูกต้องคือ `TRANSACTION.units_sold` join `ORDER` และ `PRODUCT` แล้ว aggregate เป็น `store_id × category × date`"
    )
    lines.append(
        "- Forecast window คือ `2024-11-01` ถึง `2024-12-31`; public = November, private = December; metric = MAE"
    )
    lines.append(
        "- ห้ามใช้ `INVENTORY.units_sold` เป็น target; ใช้ได้เป็น stockout/context feature และต้องระวัง horizon leakage"
    )
    lines.append(
        "- `PROMOTION`, `LOCAL_EVENT`, `DATE_DIM`, `STORE`, `PRODUCT` เป็น lookup ที่รู้ล่วงหน้าได้ แต่ lag/rolling จากยอดขายต้อง shift ตาม horizon"
    )
    lines.append("")
    lines.append("## 1) Data Audit")
    lines.append("")
    rows = []
    for key in sorted(row_counts):
        rows.append([key, row_counts[key], len(schemas[key])])
    lines.append(table(["table", "rows", "cols"], rows, digits=0))
    lines.append("")
    lines.append(f"- Train order date range: `{train_start}` to `{train_end}` = {len(train_dates):,} days")
    lines.append(f"- Full target grid after zero-fill: `{len(stores)} stores × {len(categories)} categories × {len(train_dates)} days = {full_cell_count:,}` rows")
    lines.append(f"- Forecast grid: `{len(stores)} × {len(categories)} × {len(forecast_dates)} × 3 horizons = {forecast_cell_count * 3:,}` rows")
    lines.append("")
    lines.append("Lookup train/test byte check:")
    lines.append(table(["lookup", "sha256 status", "train rows", "test rows"], hash_rows, digits=0))
    lines.append("")
    lines.append("## 2) Target Shape และ Demand Concentration")
    lines.append("")
    target_stats = [
        ["mean", sum(target_values) / len(target_values)],
        ["median", quantile(0.50)],
        ["p75", quantile(0.75)],
        ["p90", quantile(0.90)],
        ["p95", quantile(0.95)],
        ["p99", quantile(0.99)],
        ["max", max(target_values)],
        ["zero cell-days", target_values.count(0)],
    ]
    lines.append(table(["metric", "value"], target_stats))
    lines.append("")
    cat_rows = []
    for cat in categories:
        avg = category_units[cat] / (len(stores) * len(train_dates))
        rev_per_unit = category_revenue[cat] / category_units[cat]
        cat_rows.append([cat, category_units[cat], avg, category_revenue[cat], rev_per_unit])
    cat_rows.sort(key=lambda r: r[2], reverse=True)
    lines.append(table(["category", "total units", "avg units/store/day", "revenue", "revenue/unit"], cat_rows))
    lines.append("")
    lines.append(
        "**Insight:** distribution หนักขวาชัดมาก จึงควรดู MAE แยก category/store และทำ model หรือ post-process แยกหมวด "
        "โดย Coffee เป็น volume anchor ส่วน Merchandise เป็น low-volume/intermittent tail ที่ MAE ดิบเล็กแต่ ratio error ง่าย"
    )
    lines.append("")
    lines.append("## 3) Store, Location, Service, Capacity")
    lines.append("")
    store_table = sorted(store_rows, key=lambda r: r["units_per_day"], reverse=True)
    lines.append(table(
        ["store", "type", "units/day", "orders/day", "units/staff/day", "orders/seat/day", "cap", "staff", "hours", "loyal"],
        [
            [
                r["store_id"], r["type"], r["units_per_day"], r["orders_per_day"],
                r["units_per_staff_day"], r["orders_per_seat_day"],
                r["capacity"], r["staff"], r["hours"], r["loyal_customers"],
            ]
            for r in store_table
        ],
    ))
    lines.append("")
    nb_rows = []
    for nb, v in sorted(neighborhood_agg.items(), key=lambda kv: kv[1][0] / kv[1][2], reverse=True):
        nb_rows.append([nb, v[2], v[0] / v[2], v[1] / v[2], v[3] / v[2], v[4] / v[2]])
    lines.append(table(["neighborhood_type", "stores", "avg units/day/store", "avg orders/day/store", "avg seats", "avg hours"], nb_rows))
    lines.append("")
    lines.append(table(
        ["correlation vs units/day", "r"],
        [
            ["seating_capacity", corr_capacity],
            ["staff_count", corr_staff],
            ["operating_hours", corr_hours],
            ["loyal_customer_count", corr_loyal],
        ],
    ))
    lines.append("")
    lines.append(
        "**Insight:** capacity และ staff_count เป็น demand proxy ที่แข็งกว่า operating hours เพียว ๆ. "
        "`loyal_customer_count` ใน registry นี้ใกล้กันมากและ correlation ไม่ดี จึงควรใช้เป็น weak/static context มากกว่า driver หลัก. "
        "ควรสร้าง `units_per_staff_lag`, `orders_per_seat_lag`, `store_type × weekend`, `store_type × hour_peak` "
        "และ cluster store จาก sales mix + operational profile"
    )
    lines.append("")
    lines.append("## 4) Calendar, Weather Proxy, Holiday, Payday")
    lines.append("")
    factor_rows = []
    labels = {
        "is_weekend": "Weekend",
        "is_holiday": "Holiday",
        "is_payday": "Payday",
        "is_rainy_season": "Rainy season",
        "is_school_break": "School break",
    }
    for flag in flag_cols:
        base_raw = mean_from_agg(factor_raw[(flag, False)])
        on_raw = mean_from_agg(factor_raw[(flag, True)])
        base_rel = mean_from_agg(factor_rel[(flag, False)])
        on_rel = mean_from_agg(factor_rel[(flag, True)])
        factor_rows.append([
            labels[flag],
            base_raw,
            on_raw,
            lift_pct(base_raw, on_raw),
            base_rel,
            on_rel,
            lift_pct(base_rel, on_rel),
        ])
    lines.append(table(["factor", "raw off", "raw on", "raw lift", "rel off", "rel on", "rel lift"], factor_rows))
    lines.append("")
    for flag in ["is_weekend", "is_holiday", "is_payday", "is_rainy_season"]:
        rows = []
        for cat in categories:
            off = mean_from_agg(factor_cat_rel[(flag, cat, False)])
            on = mean_from_agg(factor_cat_rel[(flag, cat, True)])
            rows.append([cat, off, on, lift_pct(off, on)])
        rows.sort(key=lambda r: (r[3] if r[3] is not None else -999), reverse=True)
        lines.append(f"### {labels[flag]} by category (baseline-normalized)")
        lines.append(table(["category", "off rel", "on rel", "lift"], rows))
        lines.append("")
    h_rows = []
    for (hname, cat), agg in holiday_rel.items():
        h_rows.append([hname, cat, len(holiday_dates[hname]), mean_from_agg(agg)])
    h_rows.sort(key=lambda r: r[3], reverse=True)
    lines.append("Top holiday/category normalized spikes:")
    lines.append(table(["holiday", "category", "holiday dates", "rel to same store-cat-DOW"], h_rows[:15]))
    lines.append("")
    lines.append("Forecast-window known calendar:")
    lines.append(table(["date", "day", "holiday_name"], forecast_holidays, digits=0))
    lines.append("")
    lines.append(f"- Forecast paydays: {', '.join(forecast_paydays)}")
    lines.append("")
    lines.append(
        "**Insight:** `is_rainy_season` เป็น weather proxy แบบหยาบ ไม่ใช่ rainfall จริง. "
        "ถ้าเพิ่ม external weather ให้ใช้ forecast/observed ที่ public และทำเป็น feature ที่รู้ได้ ณ decision day; "
        "แต่ใน dataset ตอนนี้ interaction ระหว่าง rainy season × category/store type ยังให้ signal ที่ไม่ควรทิ้ง"
    )
    lines.append("")
    lines.append("## 5) Promotion, Price, Marketing Channel")
    lines.append("")
    promo_summary_rows = []
    no_promo_raw = mean_from_agg(promo_raw[False])
    on_promo_raw = mean_from_agg(promo_raw[True])
    no_promo_rel = mean_from_agg(promo_rel[False])
    on_promo_rel = mean_from_agg(promo_rel[True])
    promo_summary_rows.append(["any promo", no_promo_raw, on_promo_raw, lift_pct(no_promo_raw, on_promo_raw), no_promo_rel, on_promo_rel, lift_pct(no_promo_rel, on_promo_rel)])
    lines.append(table(["promo", "raw no", "raw yes", "raw lift", "rel no", "rel yes", "rel lift"], promo_summary_rows))
    lines.append("")
    rows = []
    for cat in categories:
        off = mean_from_agg(promo_cat_rel[(cat, False)])
        on = mean_from_agg(promo_cat_rel[(cat, True)])
        rows.append([cat, train_promo_by_cat[cat], off, on, lift_pct(off, on)])
    rows.sort(key=lambda r: (r[4] if r[4] is not None else -999), reverse=True)
    lines.append(table(["category", "train promo cell-days", "rel no promo", "rel on promo", "rel lift"], rows))
    lines.append("")
    ptype_rows = []
    for ptype, cnt in promo_type_rows.most_common():
        ptype_rows.append([ptype, cnt, mean_from_agg(promo_type_discount[ptype])])
    lines.append("Promotion design distribution:")
    lines.append(table(["promo_type", "rows", "avg discount"], ptype_rows))
    lines.append("")
    pte_rows = []
    for (ptype, cat), agg in promo_type_effect.items():
        if agg[1] >= 20:
            pte_rows.append([ptype, cat, agg[1], mean_from_agg(agg)])
    pte_rows.sort(key=lambda r: r[3], reverse=True)
    lines.append("Top promo-type × category normalized effects:")
    lines.append(table(["promo_type", "category", "cell-days", "rel"], pte_rows[:20]))
    lines.append("")
    channel_rows = []
    for (channel, cat), agg in promo_channel_effect.items():
        if agg[1] >= 20:
            channel_rows.append([channel, cat, agg[1], mean_from_agg(agg)])
    channel_rows.sort(key=lambda r: r[3], reverse=True)
    lines.append("Marketing channel proxy:")
    lines.append(table(["channel", "category", "cell-days", "rel"], channel_rows[:20]))
    lines.append("")
    disc_rows = []
    for dbin in ["0%", "1-10%", "11-20%", "21-30%", "31-40%", "41%+"]:
        disc_rows.append([dbin, discount_bin_lines[dbin], discount_bin_units[dbin], pct(discount_bin_units[dbin], units_total)])
    lines.append("Transaction discount-applied distribution:")
    lines.append(table(["discount_applied bin", "lines", "units", "unit share"], disc_rows, digits=0))
    lines.append("")
    lines.append(
        f"- Promo schedule exposure train: {train_promo_cell_count:,}/{full_cell_count:,} cell-days = {pct(train_promo_cell_count, full_cell_count)}"
    )
    lines.append(
        f"- Promo schedule exposure forecast: {forecast_promo_cell_count:,}/{forecast_cell_count:,} cell-days = {pct(forecast_promo_cell_count, forecast_cell_count)}"
    )
    lines.append("")
    fpromo_rows = []
    for cat in categories:
        train_rate = train_promo_by_cat[cat] / (len(stores) * len(train_dates))
        test_rate = forecast_promo_by_cat[cat] / (len(stores) * len(forecast_dates))
        fpromo_rows.append([cat, train_rate * 100, test_rate * 100, test_rate * 100 - train_rate * 100])
    fpromo_rows.sort(key=lambda r: r[3], reverse=True)
    lines.append(table(["category", "train promo % cell-days", "forecast promo % cell-days", "pp change"], fpromo_rows))
    lines.append("")
    off = mean_from_agg(merch_cannibal[False])
    on = mean_from_agg(merch_cannibal[True])
    lines.append(
        f"Merchandise cannibalization proxy during Coffee/Tea promo: rel no drink promo = {off:.3f}, rel drink promo = {on:.3f}, effect = {fmt_num(lift_pct(off, on))}%"
    )
    lines.append("")
    lines.append(
        "**Insight:** อย่าดู promo เป็น binary เดียว ให้แตกเป็น `max_discount`, `promo_type`, `email/social`, "
        "`category promo exposure`, `days_since/until promo`, และ `other drink promo active` เพื่อจับ cannibalization/cross-sell"
    )
    lines.append("")
    lines.append("## 6) Events, Festivals, Local Demand Shocks")
    lines.append("")
    ev_no = mean_from_agg(event_rel[False])
    ev_yes = mean_from_agg(event_rel[True])
    lines.append(table(
        ["event", "raw mean", "rel mean", "lift vs no-event"],
        [
            ["no event", mean_from_agg(event_raw[False]), ev_no, 0],
            ["has event", mean_from_agg(event_raw[True]), ev_yes, lift_pct(ev_no, ev_yes)],
        ],
    ))
    lines.append("")
    ev_cat_rows = []
    for cat in categories:
        off = mean_from_agg(event_cat_rel[(cat, False)])
        on = mean_from_agg(event_cat_rel[(cat, True)])
        ev_cat_rows.append([cat, off, on, lift_pct(off, on)])
    ev_cat_rows.sort(key=lambda r: (r[3] if r[3] is not None else -999), reverse=True)
    lines.append(table(["category", "rel no event", "rel event", "lift"], ev_cat_rows))
    lines.append("")
    ev_type_rows = []
    for (etype, cat), agg in event_type_cat_rel.items():
        if agg[1] >= 10:
            ev_type_rows.append([etype, cat, agg[1], mean_from_agg(agg)])
    ev_type_rows.sort(key=lambda r: r[3], reverse=True)
    lines.append("Top event-type × category normalized effects:")
    lines.append(table(["event_type", "category", "cell-days", "rel"], ev_type_rows[:20]))
    lines.append("")
    lines.append("Forecast-window event count by type:")
    lines.append(table(["event_type", "train count", "forecast count"], [[k, train_event_counter[k], forecast_event_counter[k]] for k in sorted(event_type_counter)], digits=0))
    lines.append("")
    lines.append("First forecast-window events:")
    lines.append(table(["date", "store", "event_type"], forecast_events[:30], digits=0))
    lines.append("")
    lines.append(
        "**Insight:** event feature ควรมี `has_event`, `event_type`, `event_count_store_day`, "
        "`store_type × event_type`, และควร normalize/validate ด้วย same-store weekday baseline เพราะ event ถูกสุ่มไปตกที่ร้าน volume ต่างกัน"
    )
    lines.append("")
    lines.append("## 7) Traffic Proxy และ Hour-of-Day Behavior")
    lines.append("")
    hour_rows = []
    total_orders = sum(order_count_by_hour.values())
    for hour, cnt in sorted(order_count_by_hour.items()):
        avg_units = mean_from_agg(order_hour_units[hour])
        hour_rows.append([hour, cnt, pct(cnt, total_orders), avg_units])
    lines.append(table(["hour", "orders", "order share", "avg units/order"], hour_rows))
    lines.append("")
    peak_rows = []
    neighborhoods = sorted({s["neighborhood_type"] for s in store.values()})
    for nb in neighborhoods:
        counts = [(hour, order_count_by_neighborhood_hour[(nb, hour)]) for hour in range(24)]
        top = sorted(counts, key=lambda x: x[1], reverse=True)[:3]
        peak_rows.append([nb, ", ".join(f"{h}:00 ({c:,})" for h, c in top)])
    lines.append(table(["neighborhood", "top order hours"], peak_rows, digits=0))
    lines.append("")
    lines.append(
        "**Insight:** traffic ไม่มี column ตรง ๆ แต่ใช้ proxy ได้จาก `hour`, `neighborhood_type`, `open_time`, "
        "`is_weekend`, `event`, `payday`, และ store type เช่น transit/gas_station. สำหรับ daily forecasting ให้ทำ aggregate จากอดีต เช่น "
        "`lag_morning_order_share`, `lag_peak_hour_units`, `weekday_commute_store_type`"
    )
    lines.append("")
    lines.append("## 8) SKU, Menu, Seasonal/Limited Edition")
    lines.append("")
    lines.append(table(["product_id", "name", "category", "serve", "price", "units", "revenue", "seasonal", "limited"], top_sku_rows))
    lines.append("")
    seas_rows = []
    for cat in categories:
        seasonal = seasonal_units[(cat, "seasonal")]
        regular = seasonal_units[(cat, "regular")]
        total = seasonal_total[cat]
        seas_rows.append([cat, seasonal, regular, pct(seasonal, total)])
    lines.append(table(["category", "seasonal units", "regular units", "seasonal unit share"], seas_rows, digits=0))
    lines.append("")
    lines.append("Trend YoY Jan-Oct 2024 vs Jan-Oct 2023:")
    lines.append(table(["category", "2023 avg units/cell", "2024 avg units/cell", "YoY"], month_rows))
    lines.append("")
    lines.append("Recent drift Aug-Oct 2024 vs May-Jul 2024:")
    lines.append(table(["category", "May-Jul avg units/cell", "Aug-Oct avg units/cell", "change"], recent_rows))
    lines.append("")
    lines.append(
        "**Insight:** โจทย์ forecast category แต่ SKU still matters เพราะ SKU mix, seasonal/limited flags, และ stockout เกิดที่ product level. "
        "ควร aggregate SKU signals ขึ้นเป็น category เช่น `seasonal_unit_share_28d`, `limited_sku_promo_count`, `top_sku_stockout_rate`"
    )
    lines.append("")
    lines.append("## 9) Customer Segments, Frequency, Density, Behavior")
    lines.append("")
    member_rows = []
    for key in ["walk_in", "member"]:
        agg = member_order_agg[key]
        member_rows.append([key, agg[1], agg[0], agg[2], agg[0] / agg[1] if agg[1] else 0, agg[2] / agg[1] if agg[1] else 0])
    lines.append(table(["segment", "orders", "units", "revenue", "units/order", "revenue/order"], member_rows))
    lines.append("")
    lines.append(
        f"- Walk-in orders: {walkin_orders:,}/{len(order_info):,} = {pct(walkin_orders, len(order_info))}; member/customer-id orders = {pct(member_orders, len(order_info))}"
    )
    lines.append(
        f"- Active registered customers appearing in orders: {active_members:,}/{len(customers):,}; one-time = {pct(one_time, active_members)}, repeat >=2 = {pct(repeat, active_members)}, high-frequency >=10 = {pct(high_freq, active_members)}"
    )
    lines.append(
        f"- Coffee attachment: Bakery in coffee orders = {pct(coffee_with_bakery, coffee_orders)}, Savory Bakery in coffee orders = {pct(coffee_with_savory, coffee_orders)}"
    )
    basket_rows = []
    for mask, cnt in basket_mask_count.most_common(12):
        cats = [cat for cat in categories if mask & (1 << category_idx[cat])]
        basket_rows.append([" + ".join(cats), cnt, pct(cnt, len(order_units))])
    lines.append("")
    lines.append("Top basket category combinations:")
    lines.append(table(["basket categories", "orders", "share"], basket_rows, digits=0))
    lines.append("")
    pay_rows = []
    for payment, agg in sorted(payment_agg.items(), key=lambda kv: kv[1][0], reverse=True):
        pay_rows.append([payment, agg[0], pct(agg[0], len(order_units)), agg[1], agg[2], agg[1] / agg[0]])
    lines.append("Payment behavior:")
    lines.append(table(["payment", "orders", "share", "revenue", "units", "revenue/order"], pay_rows))
    lines.append("")
    pref_rows = []
    for sid, cnt in preferred_store_count.most_common():
        pref_rows.append([sid, store[sid]["neighborhood_type"], cnt, local_customer_count[sid], pct(local_customer_count[sid], cnt)])
    lines.append("Registered customer density by preferred store:")
    lines.append(table(["store", "type", "loyal customers", "same-neighborhood customers", "local share"], pref_rows, digits=0))
    lines.append("")
    lines.append(
        "**Insight:** ไม่มี review text ใน dataset ดังนั้น `พฤติกรรมและ review` ต้องใช้ behavior proxy: repeat frequency, basket mix, payment, "
        "member/walk-in ratio, and basket attachment. ส่วน `home_neighborhood_type` ตรงกับ preferred store 100% ใน registry นี้ "
        "จึงไม่ช่วยแยก local/non-local เท่าไร. Feature ที่น่าทำคือ rolling member ratio/store, repeat customer density, "
        "coffee-bakery attachment lag, payment mix lag"
    )
    lines.append("")
    lines.append("## 10) Inventory, Stockout, Recorded Demand")
    lines.append("")
    lines.append(
        f"- Inventory row stockout rate: {inv_stockout:,}/{inv_rows:,} = {pct(inv_stockout, inv_rows)}"
    )
    lines.append(
        f"- Inventory.units_sold exact match vs transaction SKU-store-day units: {inv_match:,}/{inv_compared:,} = {pct(inv_match, inv_compared)}; mean absolute diff = {inv_abs_diff / inv_compared:.2f} units"
    )
    so_store_rows = []
    for sid, (so, total) in sorted(stockout_store.items(), key=lambda kv: kv[1][0] / kv[1][1], reverse=True):
        so_store_rows.append([sid, store[sid]["neighborhood_type"], so, total, so / total * 100])
    lines.append(table(["store", "type", "stockout rows", "inv rows", "stockout %"], so_store_rows[:20]))
    lines.append("")
    so_cat_rows = []
    for cat, (so, total) in sorted(stockout_cat.items(), key=lambda kv: kv[1][0] / kv[1][1], reverse=True):
        so_cat_rows.append([cat, so, total, so / total * 100])
    lines.append(table(["category", "stockout rows", "inv rows", "stockout %"], so_cat_rows))
    lines.append("")
    off = mean_from_agg(stockout_sales_rel[False])
    on = mean_from_agg(stockout_sales_rel[True])
    lines.append(
        f"Store-category day with any product stockout: rel no stockout = {off:.3f}, rel stockout = {on:.3f}, apparent effect = {fmt_num(lift_pct(off, on))}%"
    )
    lines.append("")
    lines.append(
        "**Insight:** stockout rows are censored/recorded sales, not true demand. สำหรับ modeling ให้ลอง `sample_weight` ต่ำลงบน stockout-exposed rows "
        "และทำ feature `lag_stockout_rate`, `stockout_count_top_skus`, `days_since_stockout` แยกตาม horizon"
    )
    lines.append("")
    lines.append("## 11) Forecast Window Drift Watchlist")
    lines.append("")
    lines.append("- Public LB = November 2024, Private LB = December 2024: อย่าจูนจาก Nov อย่างเดียว")
    lines.append("- Calendar known in forecast window มี holiday/payday/event/promo exposure ที่ควร join จาก lookup ได้ตรง ๆ")
    lines.append("- Compare promo exposure table ใน Section 5 เพื่อดูว่า category ไหนมี campaign density ใน Nov-Dec สูง/ต่ำกว่าประวัติ")
    lines.append("- December มี holiday effect และ tourist/high-season possibility จึงควร validate แบบ time split ที่มี late-year holdout เช่น Oct 2024 และ backtest Nov/Dec-like windows จาก 2023")
    lines.append("")
    lines.append("## 12) Feature Backlog ที่ควรเพิ่มใน Notebook")
    lines.append("")
    feature_rows = [
        ["Leakage-safe lag", "`lag_7/14/28/35/60/90`, rolling shifted by horizon", "core signal"],
        ["Baseline-normalized target", "`y / mean(store,category,dow)` as diagnostic/feature", "separates structural demand from shocks"],
        ["Store profile", "`store_type`, capacity, staff, hours, drive_through, age, loyal density", "location/service/capacity"],
        ["Calendar", "DOW, month, holiday_name, payday, school_break, rainy_season", "seasonality and event spikes"],
        ["Promo", "active, max_discount, promo_type, channels, days_until/since, category exposure", "planned future lookup"],
        ["Cross-promo", "drink promo active while Merchandise target", "cannibalization/cross-sell"],
        ["Event", "has_event, event_type, event_count, store_type×event_type", "local shocks"],
        ["Stockout", "lag stockout rate/product/category, any_stockout cell", "recorded demand censoring"],
        ["Customer", "rolling member ratio, repeat density, basket attachment", "behavior/frequency"],
        ["Traffic proxy", "hour mix lag, peak share, store_type×weekend", "commute/location behavior"],
        ["SKU mix", "seasonal share, limited SKU count, top SKU stockout/promo", "category-level hidden composition"],
        ["Drift", "recent 28/56/90-day trend, YoY same month, forecast promo/event density", "public/private robustness"],
    ]
    lines.append(table(["block", "features", "why"], feature_rows, digits=0))
    lines.append("")
    lines.append("## 13) External Source Ideas")
    lines.append("")
    lines.append(
        "External data should be used only if public, cited, and available at prediction time. Strong candidates:"
    )
    source_rows = [
        ["Kaggle/Rossmann Store Sales", "retail daily sales affected by promotion, holidays, seasonality, locality", "supports store/calendar/promo EDA framing"],
        ["M5 Forecasting", "hierarchical daily retail series with price, promotions, day-of-week, special events", "supports global model + exogenous variables"],
        ["Kaggle Coffee Shop Sales datasets", "hour-of-day, product, payment, revenue style coffee transaction analysis", "supports coffee behavior dashboard ideas"],
        ["Thai Meteorological Department / Thailand.go.th", "rainy season roughly mid-May to mid-October", "validate `is_rainy_season` and optional rainfall feature"],
        ["TomTom Traffic Index Bangkok", "rush-hour congestion and Bangkok mobility patterns", "optional city-level traffic proxy; use cautiously"],
        ["Bank of Thailand / public holiday calendars", "official/special holiday announcements", "holiday_name QA and long-weekend features"],
    ]
    lines.append(table(["source family", "what to borrow", "EDA use"], source_rows, digits=0))
    lines.append("")
    lines.append("Suggested citation URLs:")
    lines.append("")
    lines.append("- Rossmann Store Sales Kaggle: https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales")
    lines.append("- Store Item Demand Forecasting Kaggle: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset")
    lines.append("- Coffee Shop Sales Kaggle: https://www.kaggle.com/datasets/xavierberge/coffee-shop-sales-dataset")
    lines.append("- M5 Competition University of Nicosia: https://www.unic.ac.cy/iff/research/forecasting/m-competitions/m5/")
    lines.append("- TMD Climate: https://www.tmd.go.th/en/climate/climateSubpage")
    lines.append("- Thailand.go.th Seasons: https://www.thailand.go.th/public/useful-information-detail/009_142")
    lines.append("- TomTom Bangkok Traffic Index: https://www.tomtom.com/traffic-index/bangkok-traffic/")
    lines.append("- Bank of Thailand Financial Institution Holidays: https://www.bot.or.th/th/financial-institutions-holiday.html")
    lines.append("")
    lines.append("## 14) Notebook Presentation Checklist")
    lines.append("")
    lines.append("- Add charts with raw mean and normalized lift side by side for promo/event/holiday/weather")
    lines.append("- Add store profile scatter: capacity/staff/hours/loyal density vs avg units/day with labels")
    lines.append("- Add heatmaps: store × category demand, event_type × category lift, promo_type × category lift")
    lines.append("- Add forecast-window calendar strip: Nov-Dec holiday/payday/event/promo count by date")
    lines.append("- Add MAE diagnostic plan: by horizon/category/store/stockout/promo/event")
    lines.append("- Add explicit leakage note before feature engineering cells")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"Generated by `{Path(__file__).name}`. Internal dataset read from `{DATA_ROOT}`.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
