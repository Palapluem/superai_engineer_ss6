# %% [markdown]
# # The Ultimate Blueprint: Demand Forecasting Coffee Chain Hackathon
# สคริปต์นี้เป็นการรวบรวมเทคนิคระดับ Masterpiece สำหรับการแข่ง Hackathon ตามแผน "Ultimate Blueprint" 
# ครอบคลุมตั้งแต่:
# 1. การจัดการสาขาที่หาไม่เจอบนเว็บ Cafe Amazon (Unlisted Stores)
# 2. การทำ Spatial EDA ด้วย DBSCAN
# 3. การหาจุด Cannibalization ของโปรโมชั่น (แจกแก้วฟรีทำยอด Merchandise ตก)
# 4. Feature Engineering: Lags & Rolling ที่ป้องกัน Data Leakage 100%

# %% [markdown]
# ## 1. Imports & Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import DBSCAN
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import os
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# เปลี่ยน Path ให้ตรงกับโฟลเดอร์ Dataset ของคุณ
DATA_DIR = './super-ai-engineer-season-6-coffee-chain-hackathon/train'

# %% [markdown]
# ## 2. Data Loading & Handling Unlisted Stores (Missing External Data)

# %%
print("Loading data...")
try:
    txn = pd.read_csv(os.path.join(DATA_DIR, 'TRANSACTION.csv'))
    order = pd.read_csv(os.path.join(DATA_DIR, 'ORDER.csv'), parse_dates=['date'])
    product = pd.read_csv(os.path.join(DATA_DIR, 'PRODUCT.csv'))
    store = pd.read_csv(os.path.join(DATA_DIR, 'STORE.csv'))
    promo = pd.read_csv(os.path.join(DATA_DIR, 'PROMOTION.csv'), parse_dates=['start_date', 'end_date'])
    # ถ้ามี external data (coordinates.csv) สามารถโหลดเพิ่มตรงนี้
    # coords = pd.read_csv(os.path.join(DATA_DIR, 'COORDINATES.csv'))
except Exception as e:
    print(f"File loading error: {e}")

# สมมติฐาน: ข้อมูลสาขาบางสาขาไม่มีบนเว็บ Cafe Amazon
# สร้าง Mock Data สำหรับพิกัด (Lat/Lon) เพื่อทดสอบ DBSCAN (หากมีข้อมูลจริงให้ใช้แทนได้เลย)
np.random.seed(42)
store['latitude'] = np.random.uniform(13.5, 13.9, len(store)) # พิกัดกทม.
store['longitude'] = np.random.uniform(100.3, 100.8, len(store))

# สร้าง Flag `is_unlisted_store` (สมมติว่าสาขาที่ 4 และ 5 ไม่มีข้อมูลบนเว็บ)
store['is_unlisted_store'] = store['store_id'].apply(lambda x: 1 if x in [4, 5] else 0)

print(f"Found {store['is_unlisted_store'].sum()} unlisted stores.")

# %% [markdown]
# ## 3. Data Integration: สร้างตารางยอดขายรายวัน (Target Variable)

# %%
# Merge Transaction เข้ากับ Order เพื่อเอา date และ store_id
df = txn.merge(order[['order_id', 'store_id', 'date']], on='order_id', how='left')
# Merge กับ Product เพื่อเอา Category
df = df.merge(product[['product_id', 'category']], on='product_id', how='left')

# Aggregate ยอดขายตาม (store_id, category, date)
daily_sales = df.groupby(['store_id', 'category', 'date'])['units_sold'].sum().reset_index()

# เติมเต็มวันที่หายไป (Reindexing) เพื่อให้ Time Series ไม่แหว่ง
all_dates = pd.date_range(start=daily_sales['date'].min(), end=daily_sales['date'].max())
multi_index = pd.MultiIndex.from_product(
    [daily_sales['store_id'].unique(), daily_sales['category'].unique(), all_dates],
    names=['store_id', 'category', 'date']
)
daily_sales = daily_sales.set_index(['store_id', 'category', 'date']).reindex(multi_index, fill_value=0).reset_index()

# เอาข้อมูล Store กลับมาประกอบ
daily_sales = daily_sales.merge(store, on='store_id', how='left')

print("Daily Sales DataFrame Shape:", daily_sales.shape)

# %% [markdown]
# ## 4. Ultimate Deep EDA

# %%
# 4.1 DBSCAN Spatial Clustering (หาทำเลทองและจุดแย่งยอดขาย)
print("Running DBSCAN Clustering...")
coords = store[['latitude', 'longitude']].values
# แปลงองศาเป็นเรเดียนสำหรับการคำนวณ Haversine (1 rad ≈ 6371 km)
kms_per_radian = 6371.0088
epsilon = 3 / kms_per_radian # รัศมี 3 กิโลเมตร
db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

store['dbscan_cluster_id'] = db.labels_
print("DBSCAN Clusters found:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))

# 4.2 The EV / Neighborhood Capacity Effect
plt.figure(figsize=(10, 6))
sns.boxplot(data=daily_sales, x='neighborhood_type', y='units_sold', hue='category')
plt.title('Sales Capacity by Neighborhood Type (The EV / Mall Effect)')
plt.show()

# 4.3 Promo Cannibalization (โปรแจกแก้วฟรี กระทบ Merchandise ไหม?)
# สร้าง Dummy Promo Flag (ในงานจริงให้ Join กับ PROMOTION.csv)
daily_sales['is_free_cup_promo'] = np.where((daily_sales['date'] >= '2024-03-01') & (daily_sales['date'] <= '2024-03-15'), 1, 0)

merch_sales = daily_sales[daily_sales['category'] == 'Merchandise'].copy()
promo_impact = merch_sales.groupby('is_free_cup_promo')['units_sold'].mean()

print("\n--- Cannibalization Analysis ---")
print(f"Avg Merchandise Sales (No Promo): {promo_impact[0]:.2f}")
print(f"Avg Merchandise Sales (During Free Cup Promo): {promo_impact[1]:.2f}")
if promo_impact[1] < promo_impact[0]:
    print(">> INSIGHT: Cannibalization Confirmed! 'Free Cup' promotion decreases Merchandise sales.")

# %% [markdown]
# ## 5. Heavy Internal Feature Engineering (The Core Engine)
# ป้องกัน Data Leakage 100% ด้วยการอิงตาม `horizon` อย่างเข้มงวด

# %%
def engineer_features(df, horizon):
    df_feat = df.copy()
    df_feat.sort_values(['store_id', 'category', 'date'], inplace=True)
    
    print(f"Engineering features for Horizon: {horizon}d")
    
    # 1. Temporal Features
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
    
    # 2. Strict Horizon Lags
    # เราสามารถสร้าง Lag ได้แค่ข้อมูลที่มีอยู่จริง ณ วันทำนาย (เช่น ทายล่วงหน้า 7 วัน Lag ต้อง >= 7)
    lags = [0, 1, 2, 7, 14, 28] 
    
    for lag in lags:
        actual_lag = horizon + lag 
        col_name = f'sales_lag_{actual_lag}'
        df_feat[col_name] = df_feat.groupby(['store_id', 'category'])['units_sold'].shift(actual_lag)
    
    # 3. Rolling Statistics (on Lagged Data ONLY!)
    base_lag = horizon # lag ตัวแรกที่เราสามารถใช้ได้
    windows = [7, 14, 28]
    
    for w in windows:
        # Rolling Mean
        df_feat[f'rolling_mean_{w}'] = df_feat.groupby(['store_id', 'category'])[f'sales_lag_{base_lag}'].transform(lambda x: x.rolling(w).mean())
        # Rolling Std (Volatility index)
        df_feat[f'rolling_std_{w}'] = df_feat.groupby(['store_id', 'category'])[f'sales_lag_{base_lag}'].transform(lambda x: x.rolling(w).std())
    
    # 4. Cannibalization Flag
    # 1 ถ้ากำลังทำนาย Merchandise และมีโปรแจกของฟรี
    df_feat['is_merch_cannibalized'] = ((df_feat['category'] == 'Merchandise') & (df_feat['is_free_cup_promo'] == 1)).astype(int)
    
    # 5. Spatial Feature (DBSCAN)
    # สมมติ Merge จากตาราง Store แล้ว (มี dbscan_cluster_id แล้ว)
    
    # Drop NAs จากการทำ Lag
    df_feat.dropna(subset=[f'sales_lag_{horizon+28}', f'rolling_mean_28'], inplace=True)
    return df_feat

# สร้างข้อมูลพร้อมเทรนสำหรับโมเดล 1d
df_1d = engineer_features(daily_sales, horizon=1)

# %% [markdown]
# ## 6. LightGBM Tweedie Modeling (Validation & Training)

# %%
print("\n--- Training LightGBM Model (Tweedie Loss) ---")

# เข้ารหัส Categorical
df_1d['category_cat'] = df_1d['category'].astype('category').cat.codes
df_1d['neighborhood_cat'] = df_1d['neighborhood_type'].astype('category').cat.codes

features = [c for c in df_1d.columns if c.startswith('sales_lag_') or c.startswith('rolling_')]
features += ['dayofweek', 'month', 'is_weekend', 'category_cat', 'neighborhood_cat', 'dbscan_cluster_id', 'is_merch_cannibalized', 'is_unlisted_store']

# Validation Split (เดือน 21-22 ไว้ Test)
max_date = df_1d['date'].max()
val_cutoff = max_date - pd.Timedelta(days=60) # แบ่ง 2 เดือนสุดท้ายเป็น Validation

train_df = df_1d[df_1d['date'] < val_cutoff]
val_df = df_1d[df_1d['date'] >= val_cutoff]

X_train, y_train = train_df[features], train_df['units_sold']
X_val, y_val = val_df[features], val_df['units_sold']

# สร้าง lgb.Dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# พารามิเตอร์ลับจาก M5 (Tweedie)
params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.15, # จัดการ Zero-inflated
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'random_state': 42,
    'verbose': -1
}

model_1d = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# ประเมินผล
preds = model_1d.predict(X_val)
mae = mean_absolute_error(y_val, preds)
print(f"\n[Validation Result] MAE (1d Horizon): {mae:.4f}")

# Feature Importance
lgb.plot_importance(model_1d, max_num_features=15, figsize=(10, 6), title='Top 15 Most Important Features')
plt.show()

print("\n✅ Ultimate Pipeline Execution Completed!")
