import os
import json

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in src.split('\n')]}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in src.split('\n')]}

cells = [
md("# 🕵️‍♂️ Advanced EDA: Churn, Promos, and Behavioral Patterns\nThis notebook generates 21 deep insight visualizations based on the raw dataset."),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)

print('Loading datasets...')
path = r'super-ai-engineer-season-6-coffee-chain-hackathon/train/'
try:
    txn = pd.read_csv(path + 'TRANSACTION.csv')
except FileNotFoundError:
    # Fallback to local path structure if running outside Kaggle
    path = r'Level 2/Hackathon 5_Demand Forecasting Coffee Chain Hackathon/super-ai-engineer-season-6-coffee-chain-hackathon/train/'
    txn = pd.read_csv(path + 'TRANSACTION.csv')

order = pd.read_csv(path + 'ORDER.csv')
cust = pd.read_csv(path + 'CUSTOMER.csv')
promo = pd.read_csv(path + 'PROMOTION.csv')
store = pd.read_csv(path + 'STORE.csv')
prod = pd.read_csv(path + 'PRODUCT.csv')
date_dim = pd.read_csv(path + 'DATE_DIM.csv')
event = pd.read_csv(path + 'LOCAL_EVENT.csv')

order['date'] = pd.to_datetime(order['date'])
date_dim['date'] = pd.to_datetime(date_dim['date'])
promo['start_date'] = pd.to_datetime(promo['start_date'])
promo['end_date'] = pd.to_datetime(promo['end_date'])

# Merge core dataframe
df = txn.merge(order, on='order_id', how='left')
df = df.merge(prod, on='product_id', how='left')
df = df.merge(store, on='store_id', how='left')
print('Data loaded! Shape:', df.shape)
"""),

md("## 1. Member vs. Non-Member Buying Patterns"),
code("""\
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# G1: Revenue Split
rev_split = df.groupby('is_member')['revenue'].sum()
axes[0,0].pie(rev_split, labels=['Walk-in', 'Member'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
axes[0,0].set_title('G1: Total Revenue by Membership')

# G2: Basket Size (Units per Order)
basket = df.groupby(['order_id', 'is_member'])['units_sold'].sum().reset_index()
sns.boxplot(data=basket, x='is_member', y='units_sold', ax=axes[0,1], showfliers=False)
axes[0,1].set_title('G2: Basket Size (Units per Order)')

# G3: Hourly Visit Pattern
hourly = df.groupby(['hour', 'is_member'])['order_id'].nunique().reset_index()
sns.lineplot(data=hourly, x='hour', y='order_id', hue='is_member', ax=axes[1,0], marker='o')
axes[1,0].set_title('G3: Order Frequency by Hour of Day')

# G4: Category Preference
cat_pref = df.groupby(['category', 'is_member'])['units_sold'].sum().unstack()
cat_pref = cat_pref.div(cat_pref.sum(axis=1), axis=0) * 100
cat_pref.plot(kind='bar', stacked=True, ax=axes[1,1], colormap='Set2')
axes[1,1].set_title('G4: Category Preference Share (%)')

plt.tight_layout()
plt.show()

# G5: Payment Method Distribution
plt.figure(figsize=(10,5))
sns.countplot(data=df.drop_duplicates('order_id'), y='payment_method', hue='is_member')
plt.title('G5: Payment Method Preference')
plt.show()
"""),

md("## 2. Promotion Effectiveness & Cannibalization"),
code("""\
# Prepare promo data mapped to daily store level
daily_sales = df.groupby(['store_id', 'date', 'category'])['units_sold'].sum().reset_index()
daily_sales = daily_sales.merge(date_dim[['date', 'day_of_week']], on='date', how='left')

promo_expanded = []
for _, r in promo.iterrows():
    dates = pd.date_range(r['start_date'], r['end_date'])
    for d in dates:
        promo_expanded.append({
            'store_id': r['store_id'], 'date': d, 'product_id': r['product_id'],
            'promo_type': r['promo_type'], 'discount_pct': r['discount_pct'],
            'email_sent': r['email_sent'], 'social_campaign': r['social_campaign']
        })
promo_df = pd.DataFrame(promo_expanded).drop_duplicates(subset=['store_id', 'date', 'product_id'])
promo_df = promo_df.merge(prod[['product_id', 'category']], on='product_id', how='left')
promo_daily = promo_df.groupby(['store_id', 'date', 'category']).agg({
    'promo_type': 'first', 'discount_pct': 'max', 'email_sent': 'max', 'social_campaign': 'max'
}).reset_index()

daily_promo = daily_sales.merge(promo_daily, on=['store_id', 'date', 'category'], how='left')
daily_promo['has_promo'] = daily_promo['promo_type'].notna()

fig, axes = plt.subplots(3, 2, figsize=(22, 20))

# G6: Overall Lift by Promo Type
sns.boxplot(data=daily_promo[daily_promo['has_promo']], x='promo_type', y='units_sold', ax=axes[0,0], showfliers=False)
axes[0,0].set_title('G6: Units Sold by Promotion Type')

# G7: Discount Depth Non-Linearity
daily_promo['discount_tier'] = pd.cut(daily_promo['discount_pct'], bins=[0, 10, 20, 30, 40, 50, 100], labels=['1-10%', '11-20%', '21-30%', '31-40%', '41-50%', '50%+'])
sns.barplot(data=daily_promo, x='discount_tier', y='units_sold', ax=axes[0,1])
axes[0,1].set_title('G7: Average Units Sold by Discount Tier')

# G8: Promo Lift by Day of Week
sns.lineplot(data=daily_promo, x='day_of_week', y='units_sold', hue='has_promo', marker='o', ax=axes[1,0])
axes[1,0].set_title('G8: Promotion Lift by Day of Week (0=Mon, 6=Sun)')

# G9: Marketing Channel Effectiveness
daily_promo['channel'] = 'None'
daily_promo.loc[daily_promo['email_sent'] & ~daily_promo['social_campaign'], 'channel'] = 'Email Only'
daily_promo.loc[~daily_promo['email_sent'] & daily_promo['social_campaign'], 'channel'] = 'Social Only'
daily_promo.loc[daily_promo['email_sent'] & daily_promo['social_campaign'], 'channel'] = 'Both'
sns.barplot(data=daily_promo[daily_promo['has_promo']], x='channel', y='units_sold', ax=axes[1,1])
axes[1,1].set_title('G9: Impact of Marketing Channels')

# G10: Neighborhood Promo Responsiveness
daily_promo = daily_promo.merge(store[['store_id', 'neighborhood_type']], on='store_id', how='left')
sns.barplot(data=daily_promo, y='neighborhood_type', x='units_sold', hue='has_promo', ax=axes[2,0])
axes[2,0].set_title('G10: Promo Responsiveness by Neighborhood')

# G11: Promo Cannibalization (Coffee Promo vs Bakery Sales)
coffee_promo_dates = daily_promo[(daily_promo['category']=='Coffee') & daily_promo['has_promo']][['store_id', 'date']].drop_duplicates()
coffee_promo_dates['coffee_promo_active'] = True
bakery_sales = daily_promo[daily_promo['category']=='Bakery'].merge(coffee_promo_dates, on=['store_id', 'date'], how='left')
bakery_sales['coffee_promo_active'] = bakery_sales['coffee_promo_active'].fillna(False)
sns.boxplot(data=bakery_sales, x='coffee_promo_active', y='units_sold', ax=axes[2,1], showfliers=False)
axes[2,1].set_title('G11: Bakery Sales when Coffee is on Promo (Cannibalization?)')

plt.tight_layout()
plt.show()
"""),

md("## 3. Customer Retention & Churn (Cohort Analysis)"),
code("""\
# G12: Monthly Active Users (MAU)
order['month_year'] = order['date'].dt.to_period('M')
mau = order.dropna(subset=['customer_id']).groupby('month_year')['customer_id'].nunique()
plt.figure(figsize=(15, 5))
mau.plot(kind='bar', color='teal')
plt.title('G12: Monthly Active Users (MAU) Trend')
plt.xticks(rotation=45)
plt.show()

# G13: Cohort Retention Heatmap
cohorts = order.dropna(subset=['customer_id'])[['customer_id', 'date']].copy()
cohorts['cohort_month'] = cohorts.groupby('customer_id')['date'].transform('min').dt.to_period('M')
cohorts['order_month'] = cohorts['date'].dt.to_period('M')
cohort_group = cohorts.groupby(['cohort_month', 'order_month'])['customer_id'].nunique().reset_index()
cohort_group['period_number'] = (cohort_group['order_month'] - cohort_group['cohort_month']).apply(lambda x: x.n)
cohort_pivot = cohort_group.pivot(index='cohort_month', columns='period_number', values='customer_id')
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

plt.figure(figsize=(16, 10))
sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=1.0)
plt.title('G13: Customer Cohort Retention Heatmap')
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.show()

# G14 & G15: High-Value Customer Drop-off & Revenue Concentration
customer_revenue = df.dropna(subset=['customer_id']).groupby('customer_id')['revenue'].sum().sort_values(ascending=False)
cumulative_revenue = customer_revenue.cumsum() / customer_revenue.sum()

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

axes[0].plot(np.arange(1, len(cumulative_revenue)+1) / len(cumulative_revenue), cumulative_revenue.values)
axes[0].fill_between(np.arange(1, len(cumulative_revenue)+1) / len(cumulative_revenue), cumulative_revenue.values, alpha=0.3)
axes[0].set_title('G14: Lorenz Curve (Revenue Concentration)')
axes[0].set_xlabel('% of Customers')
axes[0].set_ylabel('% of Cumulative Revenue')
axes[0].plot([0,1], [0,1], 'k--') # 80/20 rule line

# Track top 5% Whales over time
top_5_pct_threshold = customer_revenue.quantile(0.95)
whales = customer_revenue[customer_revenue >= top_5_pct_threshold].index
whale_orders = order[order['customer_id'].isin(whales)]
whale_mau = whale_orders.groupby('month_year')['customer_id'].nunique()
whale_mau.plot(kind='line', marker='o', ax=axes[1], color='crimson', linewidth=2)
axes[1].set_title('G15: Active "Whales" (Top 5% Spenders) Over Time')
axes[1].set_ylabel('Number of Active Whales')

plt.tight_layout()
plt.show()
"""),

md("## 4. Deep Temporal & Store Insights"),
code("""\
fig, axes = plt.subplots(3, 2, figsize=(22, 18))

# G16: Heatmap of Sales (Day of Month vs Month of Year)
daily_total = df.groupby('date')['revenue'].sum().reset_index()
daily_total['day'] = daily_total['date'].dt.day
daily_total['month'] = daily_total['date'].dt.month
sales_cal = daily_total.pivot_table(index='day', columns='month', values='revenue', aggfunc='mean')
sns.heatmap(sales_cal, cmap='rocket_r', ax=axes[0,0])
axes[0,0].set_title('G16: Average Daily Revenue (Day vs Month)')

# G17: Store Revenue vs Seating Capacity
store_rev = df.groupby('store_id')['revenue'].sum().reset_index()
store_rev = store_rev.merge(store, on='store_id')
sns.scatterplot(data=store_rev, x='seating_capacity', y='revenue', size='staff_count', sizes=(50, 400), hue='neighborhood_type', ax=axes[0,1], alpha=0.7)
axes[0,1].set_title('G17: Store Revenue vs Capacity (Bubble = Staff Count)')

# G18: Rainy Season Impact across Store Types
rainy_df = df.merge(date_dim[['date', 'is_rainy_season']], on='date')
rainy_impact = rainy_df.groupby(['neighborhood_type', 'is_rainy_season'])['revenue'].mean().reset_index()
sns.barplot(data=rainy_impact, x='neighborhood_type', y='revenue', hue='is_rainy_season', ax=axes[1,0])
axes[1,0].set_title('G18: Impact of Rainy Season by Neighborhood')

# G19: Event Type Impact
event_impact = df.merge(event[['store_id', 'date', 'event_type']], on=['store_id', 'date'], how='left')
event_impact['event_type'] = event_impact['event_type'].fillna('No Event')
sns.boxplot(data=event_impact, x='event_type', y='revenue', ax=axes[1,1], showfliers=False)
axes[1,1].set_title('G19: Revenue Distribution by Local Event Type')

# G20: Staff Efficiency (Revenue per Staff vs Store Age)
store_rev['opened_date'] = pd.to_datetime(store_rev['opened_date'])
store_rev['store_age_days'] = (pd.Timestamp('2024-11-01') - store_rev['opened_date']).dt.days
store_rev['rev_per_staff'] = store_rev['revenue'] / store_rev['staff_count']
sns.regplot(data=store_rev, x='store_age_days', y='rev_per_staff', ax=axes[2,0])
axes[2,0].set_title('G20: Staff Efficiency vs Store Age')

# G21: Discount Applied Distribution
sns.histplot(df[df['discount_applied'] > 0]['discount_applied'], bins=20, kde=True, ax=axes[2,1], color='purple')
axes[2,1].set_title('G21: Distribution of Applied Discounts (%)')

plt.tight_layout()
plt.show()
"""),
]

path = os.path.join(
    r"Level 2/Hackathon 5_Demand Forecasting Coffee Chain Hackathon",
    "hackathon-5-eda-advanced.ipynb"
)

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(path), exist_ok=True)

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb(cells), f, indent=2, ensure_ascii=False)
print("[SUCCESS] hackathon-5-eda-advanced.ipynb generated!")
