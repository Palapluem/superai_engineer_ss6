# %% [markdown]
# # Demand Forecasting Coffee Chain Hackathon Pipeline
# สคริปต์นี้ถูกออกแบบมาเพื่อรันบน Google Colab (หรือรันผ่าน VS Code ที่รองรับ `# %%` เป็น Cell ของ Jupyter Notebook)
# ประกอบด้วยขั้นตอนตั้งแต่ Data Loading, Deep EDA (6 ขั้นตอน) และ Feature Engineering & Modeling ตามเทคนิค Kaggle M5 Forecasting (Tweedie Loss, Hierarchical Features, Direct Forecasting)

# %% [markdown]
# ## 0. Imports & Configurations
# เตรียม Library ที่จำเป็น (หากไม่มีตัวไหน ให้ใช้ `!pip install <library_name>`)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import gc

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# กำหนดสไตล์ของกราฟ
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# ตั้งค่า Path ของข้อมูล (แก้ให้ตรงกับ path จริงใน Colab เช่น '/content/drive/MyDrive/Hackathon_Data/')
DATA_DIR = './' 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# %% [markdown]
# ## 1. Data Loading and Target Construction
# โหลดข้อมูลและสร้าง Target Variable `units_sold` โดยต้องระวังเรื่อง Date Casting

# %%
def load_data(is_train=True):
    """
    ฟังก์ชันสำหรับโหลดข้อมูลจากโฟลเดอร์ train หรือ test
    """
    base_dir = TRAIN_DIR if is_train else TEST_DIR
    
    print(f"Loading data from {base_dir}...")
    try:
        # ตารางพื้นฐานที่มีทั้ง Train และ Test
        date_dim = pd.read_csv(os.path.join(base_dir, 'DATE_DIM.csv'), parse_dates=['date'])
        product = pd.read_csv(os.path.join(base_dir, 'PRODUCT.csv'))
        store = pd.read_csv(os.path.join(base_dir, 'STORE.csv'))
        promo = pd.read_csv(os.path.join(base_dir, 'PROMOTION.csv'), parse_dates=['start_date', 'end_date'])
        event = pd.read_csv(os.path.join(base_dir, 'LOCAL_EVENT.csv'), parse_dates=['event_date'])
        
        # ตารางที่มีเฉพาะใน Train (Transactions, Orders, Inventory)
        if is_train:
            txn = pd.read_csv(os.path.join(base_dir, 'TRANSACTION.csv'), parse_dates=['date'])
            order = pd.read_csv(os.path.join(base_dir, 'ORDER.csv'), parse_dates=['order_date'])
            inventory = pd.read_csv(os.path.join(base_dir, 'INVENTORY.csv'), parse_dates=['date'])
            
            # รวม Target: units_sold = จำนวน transactions (สมมติให้ 1 txn = 1 unit) 
            # *หมายเหตุ: ต้องตรวจสอบจากข้อมูลจริงว่ามี quantity หรือไม่
            # ในที่นี้สมมติรวมยอดขายจาก Transaction ตาม (store_id, product_id, date)
            txn_grouped = txn.groupby(['store_id', 'product_id', 'date']).size().reset_index(name='units_sold')
            
            # นำ Category จากตาราง PRODUCT มา Join
            txn_grouped = txn_grouped.merge(product[['product_id', 'category']], on='product_id', how='left')
            
            # รวมยอดขายระดับ (store_id, category, date) ตามโจทย์
            daily_sales = txn_grouped.groupby(['store_id', 'category', 'date'])['units_sold'].sum().reset_index()
            
            return daily_sales, date_dim, product, store, promo, event, inventory
        else:
            return None, date_dim, product, store, promo, event, None
            
    except FileNotFoundError as e:
        print(f"Warning: File not found - {e}")
        return None

# สมมติโหลดข้อมูล Train
daily_sales, date_dim, product, store, promo, event, inventory = load_data(is_train=True)

# %% [markdown]
# ## 2. Deep EDA - Part 1: Data Exploration (สถิติพื้นฐาน & ข้อมูลสูญหาย)
# ตรวจสอบภาพรวมของข้อมูลเป้าหมาย

# %%
if daily_sales is not None:
    print("=== Daily Sales Overview ===")
    print(daily_sales.info())
    display(daily_sales.describe())
    
    # ตรวจสอบวันที่ยอดขายเป็น 0 (Zero-inflated / Intermittent demand)
    zeros = (daily_sales['units_sold'] == 0).sum()
    total = len(daily_sales)
    print(f"\nZero-sales days: {zeros} out of {total} ({(zeros/total)*100:.2f}%)")
    
    # ตรวจสอบ Missing Values
    print("\nMissing values in daily_sales:")
    print(daily_sales.isnull().sum())

# %% [markdown]
# ## 3. Deep EDA - Part 2: Time Plot & Seasonal Plots
# พล็อตกราฟ Time Series สังเกต Trend และสร้างฟีเจอร์เวลาเพื่อวิเคราะห์ฤดูกาล (Seasonality)

# %%
if daily_sales is not None:
    # --- Time Plot ---
    # รวมยอดขายทุกสาขาและหมวดหมู่ เพื่อดูเทรนด์รวมรายวัน
    total_daily = daily_sales.groupby('date')['units_sold'].sum().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(total_daily['date'], total_daily['units_sold'], color='#2c3e50', linewidth=1.5)
    plt.title('Total Daily Units Sold (All Stores & Categories)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Units Sold', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # --- Feature Extraction for Seasonality ---
    daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['is_weekend'] = daily_sales['dayofweek'].isin([5, 6]).astype(int)
    
    # --- Box Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(data=daily_sales, x='dayofweek', y='units_sold', ax=axes[0], palette='Blues')
    axes[0].set_title('Sales Distribution by Day of Week (0=Mon, 6=Sun)', fontsize=14)
    
    sns.boxplot(data=daily_sales, x='category', y='units_sold', ax=axes[1], palette='Set2')
    axes[1].set_title('Sales Distribution by Category', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Deep EDA - Part 3: Time Series Decomposition & Lag Analysis
# แยกส่วนประกอบ Trend/Seasonality และดูกราฟ ACF/PACF เพื่อเลือก Lag Features

# %%
if daily_sales is not None:
    # วิเคราะห์ Decomposition สาขา 1 หมวดหมู่ 1 เป็นตัวอย่าง
    sample_ts = daily_sales[(daily_sales['store_id'] == 1) & (daily_sales['category'] == 'Coffee')].copy()
    if not sample_ts.empty:
        sample_ts.set_index('date', inplace=True)
        # ตรวจสอบว่ามีข้อมูลพอที่จะวิเคราะห์ (อย่างน้อย 14 วัน)
        if len(sample_ts) >= 14:
            decomposition = seasonal_decompose(sample_ts['units_sold'], model='additive', period=7)
            
            fig = decomposition.plot()
            fig.set_size_inches(14, 8)
            plt.suptitle('Time Series Decomposition (Store 1, Coffee)', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # --- Lag Analysis (ACF & PACF) ---
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            plot_acf(sample_ts['units_sold'], lags=35, ax=axes[0], title='Autocorrelation (ACF)')
            plot_pacf(sample_ts['units_sold'], lags=35, ax=axes[1], title='Partial Autocorrelation (PACF)')
            plt.tight_layout()
            plt.show()

# %% [markdown]
# ## 5. Feature Engineering (ตามหลัก Kaggle M5)
# สร้าง Lag, Rolling Stats, M5 Hierarchical Features
# **ระวัง Data Leakage:** การ Shift (Lag) จะต้องสัมพันธ์กับ Horizon ปัจจุบัน 
# (เช่น ถ้า Horizon 7d -> Lag ต้องเริ่มที่ t-7 เป็นอย่างต่ำ)

# %%
def create_features(df, horizon):
    """
    df: DataFrame ที่มียอดขายรายวันเรียงตามวันที่แล้ว 
        (ต้องกรอกข้อมูลให้ครบทุกวันด้วยวิธี Reindexing ก่อนเรียกใช้ฟังก์ชันนี้)
    horizon: 1, 7, หรือ 30
    """
    df = df.copy()
    df.sort_values(['store_id', 'category', 'date'], inplace=True)
    
    # 1. M5 Hierarchical Aggregation Features (ก่อนทำการ Shift)
    # หาราคาหรือยอดขายเฉลี่ยระดับ Category ในอดีต (ต้อง shift ตาม horizon ด้วย)
    cat_avg = df.groupby(['category', 'date'])['units_sold'].transform('mean')
    store_avg = df.groupby(['store_id', 'date'])['units_sold'].transform('mean')
    
    df['cat_avg_sales'] = cat_avg
    df['store_avg_sales'] = store_avg
    
    # 2. Lag Features (Shift >= horizon)
    # สมมติถ้าเป็น 1d: ข้อมูลย้อนหลังได้ 1 วัน
    # ถ้าเป็น 7d: ข้อมูลย้อนหลังได้ 7 วัน
    lags = [0, 1, 2, 7, 14, 28] 
    
    for lag in lags:
        # ระยะ lag จริง = horizon + lag
        actual_lag = horizon + lag 
        
        # Lag ของ units_sold
        df[f'sales_lag_{actual_lag}'] = df.groupby(['store_id', 'category'])['units_sold'].shift(actual_lag)
        # Lag ของ Hierarchical Features
        df[f'cat_avg_lag_{actual_lag}'] = df.groupby(['store_id', 'category'])['cat_avg_sales'].shift(actual_lag)
    
    # 3. Rolling Statistics (Mean, Std)
    # Rolling บน Lag เพื่อป้องกัน Leakage
    windows = [7, 14, 28]
    base_lag = horizon # lag ตัวแรกที่ใช้ได้
    
    for w in windows:
        df[f'rolling_mean_{w}'] = df.groupby(['store_id', 'category'])[f'sales_lag_{base_lag}'].transform(lambda x: x.rolling(w).mean())
        df[f'rolling_std_{w}'] = df.groupby(['store_id', 'category'])[f'sales_lag_{base_lag}'].transform(lambda x: x.rolling(w).std())

    # 4. Temporal & Event Features (ไม่ผิดกฎ Leakage เพราะใช้วันที่ปัจจุบัน)
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Drop rows with NA from lagging
    df.dropna(subset=[f'sales_lag_{horizon+28}', f'rolling_mean_28'], inplace=True)
    
    return df

# หมายเหตุ: ในทางปฏิบัติ ควร Join DATE_DIM, PROMOTION, LOCAL_EVENT เข้ามาเป็นฟีเจอร์ด้วย

# %% [markdown]
# ## 6. Modeling Pipeline: LightGBM (Tweedie Loss) & Direct Forecasting
# แบ่ง Train/Validation แบบ Time-Series Split และรันโมเดลทีละ Horizon

# %%
# สร้างตัวอย่างโมเดล 1d
if daily_sales is not None:
    print("Preparing features for Horizon: 1d...")
    df_1d = create_features(daily_sales, horizon=1)
    
    # เลือก Features 
    features = [col for col in df_1d.columns if col not in ['date', 'units_sold', 'cat_avg_sales', 'store_avg_sales', 'category']]
    
    # เข้ารหัส Categorical
    df_1d['category_cat'] = df_1d['category'].astype('category').cat.codes
    features.append('category_cat')
    
    # Time-Series Validation Split (สมมติให้ 14 วันสุดท้ายเป็น Validation)
    max_date = df_1d['date'].max()
    val_cutoff = max_date - pd.Timedelta(days=14)
    
    train_df = df_1d[df_1d['date'] < val_cutoff]
    val_df = df_1d[df_1d['date'] >= val_cutoff]
    
    X_train, y_train = train_df[features], train_df['units_sold']
    X_val, y_val = val_df[features], val_df['units_sold']
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # --- LightGBM Dataset ---
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # --- Model Parameters (Tweedie Loss for Intermittent Demand) ---
    params = {
        'objective': 'tweedie', # เหมาะสำหรับยอดขายที่มีเลข 0 เยอะ 
        'tweedie_variance_power': 1.1, # ค่าที่ใช้บ่อยใน M5 (1.1 - 1.5)
        'metric': 'mae', # Metric ตัดสิน Hackathon
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'n_jobs': 2, # รัน 2 core ตาม Colab free tier
        'random_state': 42
    }
    
    print("Training LightGBM for 1d Horizon...")
    model_1d = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # --- Evaluation ---
    preds = model_1d.predict(X_val)
    # Tweedie loss จะพ่นค่าเป็นบวก (Exponential) ทำให้ไม่มีติดลบโดยปริยาย
    mae = mean_absolute_error(y_val, preds)
    print(f"\nValidation MAE (1d Horizon): {mae:.4f}")
    
    # Feature Importance
    lgb.plot_importance(model_1d, max_num_features=15, figsize=(10, 6), title='Top 15 Important Features (1d Horizon)')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## การนำไปใช้จริงใน Hackathon
# 1. ทำแบบเดียวกันสำหรับ Horizon 7d (`horizon=7`) และ 30d (`horizon=30`)
# 2. นำฟีเจอร์จาก `PROMOTION` (เช่น มีส่วนลดไหม) มาใส่
# 3. จัดการ Stockout โดยกำหนด weight ใน `lgb.Dataset(..., weight=...)` ให้กับแถวที่ไม่เกิด stockout ให้มีความสำคัญสูงกว่า
