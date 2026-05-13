import json

old_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-lv2-nb-expresson-hackathon (4).ipynb"
out_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-MASTER-SPLIT.ipynb"

print("Loading original notebook...")
with open(old_nb_path, 'r', encoding='utf-8') as f:
    old_nb = json.load(f)

old_cells = old_nb.get('cells', [])

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in src.split('\n')]}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in src.split('\n')]}

# Introduction
intro_md = md("""\
---
# 🚀 ADVANCED EDA MODULE: 40 Deep Graphs & 5 Tabular Insights
**Format**: One Graph/Table per Cell. Includes ultra-deep, unexpected data relationships.
""")

# Load Data Code
data_code = code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (10, 6)

print('Loading datasets for Advanced EDA...')
path = r'super-ai-engineer-season-6-coffee-chain-hackathon/train/'
try:
    txn = pd.read_csv(path + 'TRANSACTION.csv')
except FileNotFoundError:
    path = r'Level 2/Hackathon 5_Demand Forecasting Coffee Chain Hackathon/super-ai-engineer-season-6-coffee-chain-hackathon/train/'
    txn = pd.read_csv(path + 'TRANSACTION.csv')

order = pd.read_csv(path + 'ORDER.csv')
cust = pd.read_csv(path + 'CUSTOMER.csv')
promo = pd.read_csv(path + 'PROMOTION.csv')
store = pd.read_csv(path + 'STORE.csv')
prod = pd.read_csv(path + 'PRODUCT.csv')
date_dim = pd.read_csv(path + 'DATE_DIM.csv')
event = pd.read_csv(path + 'LOCAL_EVENT.csv')
inventory = pd.read_csv(path + 'INVENTORY.csv')

order['date'] = pd.to_datetime(order['date'])
date_dim['date'] = pd.to_datetime(date_dim['date'])
promo['start_date'] = pd.to_datetime(promo['start_date'])
promo['end_date'] = pd.to_datetime(promo['end_date'])

# Merge core dataframe
df = txn.merge(order, on='order_id', how='left')
df = df.merge(prod, on='product_id', how='left')
df = df.merge(store, on='store_id', how='left')
print('Data loaded successfully!')
""")

cells = [intro_md, data_code]

# --- SECTION 1: Member vs Non-Member ---
cells.append(md("## 1. Member vs. Non-Member Buying Patterns"))
cells.append(md("### G1: Total Revenue by Membership\nIllustrates the exact revenue split between Walk-in customers and registered Members."))
cells.append(code("""\
rev_split = df.groupby('is_member')['revenue'].sum()
plt.figure(figsize=(6,6))
plt.pie(rev_split, labels=['Walk-in', 'Member'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('G1: Total Revenue by Membership')
plt.show()
"""))

cells.append(md("### G2: Basket Size (Units per Order)\nCompares the average number of items purchased per transaction."))
cells.append(code("""\
basket = df.groupby(['order_id', 'is_member'])['units_sold'].sum().reset_index()
plt.figure(figsize=(8,5))
sns.boxplot(data=basket, x='is_member', y='units_sold', showfliers=False)
plt.title('G2: Basket Size (Units per Order)')
plt.show()
"""))

cells.append(md("### G3: Order Frequency by Hour of Day\nReveals peak visitation hours."))
cells.append(code("""\
hourly = df.groupby(['hour', 'is_member'])['order_id'].nunique().reset_index()
plt.figure(figsize=(10,5))
sns.lineplot(data=hourly, x='hour', y='order_id', hue='is_member', marker='o')
plt.title('G3: Order Frequency by Hour of Day')
plt.show()
"""))

cells.append(md("### G4: Category Preference Share (%)\nShows if Members lean more heavily towards Coffee or Bakery items."))
cells.append(code("""\
cat_pref = df.groupby(['category', 'is_member'])['units_sold'].sum().unstack()
cat_pref = cat_pref.div(cat_pref.sum(axis=1), axis=0) * 100
cat_pref.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8,5))
plt.title('G4: Category Preference Share (%)')
plt.ylabel('Percentage')
plt.show()
"""))

cells.append(md("### G5: Payment Method Preference\nDisplays transaction settlement behaviors."))
cells.append(code("""\
plt.figure(figsize=(8,5))
sns.countplot(data=df.drop_duplicates('order_id'), y='payment_method', hue='is_member')
plt.title('G5: Payment Method Preference')
plt.show()
"""))

# --- SECTION 2: Promotion Effectiveness ---
cells.append(md("## 2. Promotion Effectiveness & Cannibalization"))
cells.append(code("""\
# Pre-computation for Section 2
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
"""))

cells.append(md("### G6: Units Sold by Promotion Type\nBoxplots showing the volume impact of different promo types."))
cells.append(code("""\
plt.figure(figsize=(10,6))
sns.boxplot(data=daily_promo[daily_promo['has_promo']], x='promo_type', y='units_sold', showfliers=False)
plt.title('G6: Units Sold by Promotion Type')
plt.show()
"""))

cells.append(md("### G7: Average Units Sold by Discount Tier\nIdentifies the Discount Depth Sweet Spot."))
cells.append(code("""\
daily_promo['discount_tier'] = pd.cut(daily_promo['discount_pct'], bins=[0, 10, 20, 30, 40, 50, 100], labels=['1-10%', '11-20%', '21-30%', '31-40%', '41-50%', '50%+'])
plt.figure(figsize=(10,6))
sns.barplot(data=daily_promo, x='discount_tier', y='units_sold')
plt.title('G7: Average Units Sold by Discount Tier')
plt.show()
"""))

cells.append(md("### G8: Promotion Lift by Day of Week\nHighlights the Mid-Week Magic."))
cells.append(code("""\
plt.figure(figsize=(10,6))
sns.lineplot(data=daily_promo, x='day_of_week', y='units_sold', hue='has_promo', marker='o')
plt.title('G8: Promotion Lift by Day of Week (0=Mon, 6=Sun)')
plt.show()
"""))

cells.append(md("### G9: Impact of Marketing Channels\nCompares Email vs Social Media."))
cells.append(code("""\
daily_promo['channel'] = 'None'
daily_promo.loc[daily_promo['email_sent'] & ~daily_promo['social_campaign'], 'channel'] = 'Email Only'
daily_promo.loc[~daily_promo['email_sent'] & daily_promo['social_campaign'], 'channel'] = 'Social Only'
daily_promo.loc[daily_promo['email_sent'] & daily_promo['social_campaign'], 'channel'] = 'Both'
plt.figure(figsize=(8,6))
sns.barplot(data=daily_promo[daily_promo['has_promo']], x='channel', y='units_sold')
plt.title('G9: Impact of Marketing Channels')
plt.show()
"""))

cells.append(md("### G10: Promo Responsiveness by Neighborhood\nIdentifies which store demographics react most aggressively."))
cells.append(code("""\
daily_promo_nh = daily_promo.merge(store[['store_id', 'neighborhood_type']], on='store_id', how='left')
plt.figure(figsize=(10,6))
sns.barplot(data=daily_promo_nh, y='neighborhood_type', x='units_sold', hue='has_promo')
plt.title('G10: Promo Responsiveness by Neighborhood')
plt.show()
"""))

cells.append(md("### G11: Bakery Sales when Coffee is on Promo (Cannibalization?)\nTests cross-category elasticity."))
cells.append(code("""\
coffee_promo_dates = daily_promo[(daily_promo['category']=='Coffee') & daily_promo['has_promo']][['store_id', 'date']].drop_duplicates()
coffee_promo_dates['coffee_promo_active'] = True
bakery_sales = daily_promo[daily_promo['category']=='Bakery'].merge(coffee_promo_dates, on=['store_id', 'date'], how='left')
bakery_sales['coffee_promo_active'] = bakery_sales['coffee_promo_active'].fillna(False)
plt.figure(figsize=(8,6))
sns.boxplot(data=bakery_sales, x='coffee_promo_active', y='units_sold', showfliers=False)
plt.title('G11: Bakery Sales when Coffee is on Promo (Cannibalization?)')
plt.show()
"""))

# --- SECTION 3: Customer Retention & Churn ---
cells.append(md("## 3. Customer Retention, Churn & The Whale Phenomenon"))

cells.append(md("### G12: Monthly Active Users (MAU) Trend\nOverall platform engagement over time."))
cells.append(code("""\
order['month_year'] = order['date'].dt.to_period('M')
mau = order.dropna(subset=['customer_id']).groupby('month_year')['customer_id'].nunique()
plt.figure(figsize=(12, 5))
mau.plot(kind='bar', color='teal')
plt.title('G12: Monthly Active Users (MAU) Trend')
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(md("### G13: Customer Cohort Retention Heatmap\nTracks groups of customers based on acquisition month."))
cells.append(code("""\
cohorts = order.dropna(subset=['customer_id'])[['customer_id', 'date']].copy()
cohorts['cohort_month'] = cohorts.groupby('customer_id')['date'].transform('min').dt.to_period('M')
cohorts['order_month'] = cohorts['date'].dt.to_period('M')
cohort_group = cohorts.groupby(['cohort_month', 'order_month'])['customer_id'].nunique().reset_index()
cohort_group['period_number'] = (cohort_group['order_month'] - cohort_group['cohort_month']).apply(lambda x: x.n)
cohort_pivot = cohort_group.pivot(index='cohort_month', columns='period_number', values='customer_id')
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

plt.figure(figsize=(14, 8))
sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=1.0)
plt.title('G13: Customer Cohort Retention Heatmap')
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.show()
"""))

cells.append(md("### G14: Lorenz Curve (Revenue Concentration)\nProves the Pareto Principle (80/20 rule)."))
cells.append(code("""\
customer_revenue = df.dropna(subset=['customer_id']).groupby('customer_id')['revenue'].sum().sort_values(ascending=False)
cumulative_revenue = customer_revenue.cumsum() / customer_revenue.sum()

plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(cumulative_revenue)+1) / len(cumulative_revenue), cumulative_revenue.values)
plt.fill_between(np.arange(1, len(cumulative_revenue)+1) / len(cumulative_revenue), cumulative_revenue.values, alpha=0.3)
plt.title('G14: Lorenz Curve (Revenue Concentration)')
plt.xlabel('% of Customers')
plt.ylabel('% of Cumulative Revenue')
plt.plot([0,1], [0,1], 'k--')
plt.show()
"""))

cells.append(md("### G15: Active Whales Over Time\nTracks the Top 5% of highest-spending B2B/Reseller customers."))
cells.append(code("""\
top_5_pct_threshold = customer_revenue.quantile(0.95)
whales = customer_revenue[customer_revenue >= top_5_pct_threshold].index
whale_orders = order[order['customer_id'].isin(whales)]
whale_mau = whale_orders.groupby('month_year')['customer_id'].nunique()
plt.figure(figsize=(10,5))
whale_mau.plot(kind='line', marker='o', color='crimson', linewidth=2)
plt.title('G15: Active "Whales" (Top 5% Spenders) Over Time')
plt.ylabel('Number of Active Whales')
plt.show()
"""))

# --- SECTION 4: Temporal Insights ---
cells.append(md("## 4. Deep Temporal & Store Insights"))

cells.append(md("### G16: Average Daily Revenue (Day vs Month Heatmap)\nUncovers macro calendar patterns."))
cells.append(code("""\
daily_total = df.groupby('date')['revenue'].sum().reset_index()
daily_total['day'] = daily_total['date'].dt.day
daily_total['month'] = daily_total['date'].dt.month
sales_cal = daily_total.pivot_table(index='day', columns='month', values='revenue', aggfunc='mean')
plt.figure(figsize=(10,8))
sns.heatmap(sales_cal, cmap='rocket_r')
plt.title('G16: Average Daily Revenue (Day vs Month)')
plt.show()
"""))

cells.append(md("### G17: Store Revenue vs Capacity\nExamines efficiency vs neighborhood."))
cells.append(code("""\
store_rev = df.groupby('store_id')['revenue'].sum().reset_index()
store_rev = store_rev.merge(store, on='store_id')
plt.figure(figsize=(10,6))
sns.scatterplot(data=store_rev, x='seating_capacity', y='revenue', size='staff_count', sizes=(50, 400), hue='neighborhood_type', alpha=0.7)
plt.title('G17: Store Revenue vs Capacity (Bubble = Staff Count)')
plt.show()
"""))

cells.append(md("### G18: Impact of Rainy Season by Neighborhood\nHow bad weather affects different store types."))
cells.append(code("""\
rainy_df = df.merge(date_dim[['date', 'is_rainy_season']], on='date')
rainy_impact = rainy_df.groupby(['neighborhood_type', 'is_rainy_season'])['revenue'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(data=rainy_impact, x='neighborhood_type', y='revenue', hue='is_rainy_season')
plt.title('G18: Impact of Rainy Season by Neighborhood')
plt.show()
"""))

cells.append(md("### G19: Revenue Distribution by Local Event Type\nMonetary impact of specific events."))
cells.append(code("""\
event_impact = df.merge(event[['store_id', 'date', 'event_type']], on=['store_id', 'date'], how='left')
event_impact['event_type'] = event_impact['event_type'].fillna('No Event')
plt.figure(figsize=(10,6))
sns.boxplot(data=event_impact, x='event_type', y='revenue', showfliers=False)
plt.title('G19: Revenue Distribution by Local Event Type')
plt.show()
"""))

cells.append(md("### G20: Staff Efficiency vs Store Age\nDo mature stores operate more efficiently?"))
cells.append(code("""\
store_rev['opened_date'] = pd.to_datetime(store_rev['opened_date'])
store_rev['store_age_days'] = (pd.Timestamp('2024-11-01') - store_rev['opened_date']).dt.days
store_rev['rev_per_staff'] = store_rev['revenue'] / store_rev['staff_count']
plt.figure(figsize=(10,6))
sns.regplot(data=store_rev, x='store_age_days', y='rev_per_staff')
plt.title('G20: Staff Efficiency vs Store Age')
plt.show()
"""))

cells.append(md("### G21: Distribution of Applied Discounts (%)\nHistogram showing frequency of discount depths."))
cells.append(code("""\
plt.figure(figsize=(10,6))
sns.histplot(df[df['discount_applied'] > 0]['discount_applied'], bins=20, kde=True, color='purple')
plt.title('G21: Distribution of Applied Discounts (%)')
plt.show()
"""))

# --- SECTION 5: Super-Deep Insights ---
cells.append(md("## 5. Super-Deep Insights: Unexpected Relationships"))

cells.append(md("### G22: Payday Splurge Effect by Neighborhood\nPayday effect amplified in Office/Hospital locations."))
cells.append(code("""\
payday_df = df.merge(date_dim[['date', 'is_payday']], on='date')
payday_impact = payday_df.groupby(['neighborhood_type', 'is_payday'])['revenue'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=payday_impact, x='neighborhood_type', y='revenue', hue='is_payday', palette='coolwarm')
plt.title('G22: Payday Splurge Effect by Neighborhood')
plt.show()
"""))

cells.append(md("### G23: Do Coffee Sales Increase when Merchandise is Sold Out? (Substitution)\nRunning out of merch drives coffee sales?"))
cells.append(code("""\
stockout_df = df.merge(inventory[['store_id', 'product_id', 'date', 'is_stockout']], on=['store_id', 'product_id', 'date'], how='left')
stockout_daily = stockout_df.groupby(['store_id', 'date', 'category'])['is_stockout'].max().unstack().fillna(False)
coffee_sales = df[df['category']=='Coffee'].groupby(['store_id', 'date'])['units_sold'].sum().reset_index()
coffee_vs_merch_stockout = coffee_sales.merge(stockout_daily[['Merchandise']], left_on=['store_id', 'date'], right_index=True, how='left')
plt.figure(figsize=(8, 6))
sns.boxplot(data=coffee_vs_merch_stockout, x='Merchandise', y='units_sold', showfliers=False, palette='Set2')
plt.title('G23: Coffee Sales vs Merchandise Stockout')
plt.xlabel('Merchandise Stockout (True/False)')
plt.show()
"""))

cells.append(md("### G24: Staff Stress vs Basket Size\nHigh orders per staff = smaller baskets (rushed service)."))
cells.append(code("""\
staff_stress = df.groupby(['store_id', 'date']).agg({'order_id': 'nunique', 'units_sold': 'sum', 'staff_count': 'first'}).reset_index()
staff_stress['orders_per_staff'] = staff_stress['order_id'] / staff_stress['staff_count']
staff_stress['basket_size'] = staff_stress['units_sold'] / staff_stress['order_id']
plt.figure(figsize=(10, 6))
sns.regplot(data=staff_stress, x='orders_per_staff', y='basket_size', scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('G24: Staff Stress vs Basket Size')
plt.show()
"""))

cells.append(md("### G25: Macro Trend - Average Price Per Unit\nTime-series view of menu inflation and price sensitivity."))
cells.append(code("""\
price_trend = df.groupby('date').agg({'revenue': 'sum', 'units_sold': 'sum'}).reset_index()
price_trend['avg_price_per_unit'] = price_trend['revenue'] / price_trend['units_sold']
plt.figure(figsize=(14, 5))
sns.lineplot(data=price_trend, x='date', y='avg_price_per_unit', color='darkgreen')
plt.title('G25: Macro Trend - Average Price Per Unit Over 22 Months')
plt.show()
"""))

# --- SECTION 6: Ultra-Deep Insights ---
cells.append(md("## 6. Ultra-Deep Insights: Inventory, Customers, Autocorrelation"))

cells.append(md("### G26: Customer LTV by Home Neighborhood"))
cells.append(code("""\
ltv = df.dropna(subset=['customer_id']).groupby('customer_id')['revenue'].sum().reset_index()
ltv = ltv.merge(cust[['customer_id', 'home_neighborhood_type']], on='customer_id')
plt.figure(figsize=(10,6))
sns.boxplot(data=ltv, x='home_neighborhood_type', y='revenue', showfliers=False)
plt.title('G26: Customer LTV by Home Neighborhood')
plt.show()
"""))

cells.append(md("### G27: Cross-Store Shopping Behavior"))
cells.append(code("""\
cross_store = order.dropna(subset=['customer_id']).groupby('customer_id')['store_id'].nunique().value_counts().sort_index()
plt.figure(figsize=(8,6))
cross_store.plot(kind='bar', color='coral')
plt.title('G27: Number of Unique Stores Visited per Customer')
plt.show()
"""))

cells.append(md("### G28: Avg LTV by Registration Year (Vintage)"))
cells.append(code("""\
vintage = ltv.merge(cust[['customer_id', 'registration_date']], on='customer_id')
vintage['reg_year'] = pd.to_datetime(vintage['registration_date']).dt.year
plt.figure(figsize=(8,6))
sns.barplot(data=vintage, x='reg_year', y='revenue')
plt.title('G28: Avg LTV by Registration Year (Vintage)')
plt.show()
"""))

cells.append(md("### G29: Category Co-occurrence Heatmap"))
cells.append(code("""\
basket_matrix = df.groupby(['order_id', 'category'])['units_sold'].sum().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)
co_occurrence = basket_matrix.T.dot(basket_matrix)
np.fill_diagonal(co_occurrence.values, 0)
plt.figure(figsize=(8,6))
sns.heatmap(co_occurrence, cmap='Blues', annot=True, fmt='d')
plt.title('G29: Category Co-occurrence Heatmap')
plt.show()
"""))

cells.append(md("### G30: Serve Type Performance in Rainy Season"))
cells.append(code("""\
serve_weather = df.merge(date_dim[['date', 'is_rainy_season']], on='date')
serve_weather = serve_weather.groupby(['serve_type', 'is_rainy_season'])['units_sold'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(data=serve_weather, x='serve_type', y='units_sold', hue='is_rainy_season')
plt.title('G30: Serve Type Performance in Rainy Season')
plt.show()
"""))

cells.append(md("### G31: Seasonal vs Regular Item Sales (7-Day MA)"))
cells.append(code("""\
seasonal_perf = df.groupby(['date', 'is_seasonal'])['units_sold'].sum().unstack().rolling(7).mean()
plt.figure(figsize=(12,5))
seasonal_perf.plot()
plt.title('G31: Seasonal vs Regular Item Sales (7-Day MA)')
plt.show()
"""))

cells.append(md("### G32: Stockout Duration Distribution (Days)"))
cells.append(code("""\
inventory['stockout_block'] = (inventory['is_stockout'] != inventory['is_stockout'].shift(1)).cumsum()
stockout_duration = inventory[inventory['is_stockout'] == True].groupby(['store_id', 'product_id', 'stockout_block']).size()
plt.figure(figsize=(8,5))
sns.histplot(stockout_duration[stockout_duration < 10], bins=9, discrete=True, color='red')
plt.title('G32: Stockout Duration Distribution (Days)')
plt.show()
"""))

cells.append(md("### G33: Stockout Rate (%) by Neighborhood"))
cells.append(code("""\
inv_store = inventory.merge(store[['store_id', 'neighborhood_type']], on='store_id')
stockout_rate = inv_store.groupby('neighborhood_type')['is_stockout'].mean().reset_index()
stockout_rate['is_stockout'] *= 100
plt.figure(figsize=(10,6))
sns.barplot(data=stockout_rate, x='neighborhood_type', y='is_stockout')
plt.title('G33: Stockout Rate (%) by Neighborhood')
plt.show()
"""))

cells.append(md("### G34: Closing Stock Buffer vs Units Sold"))
cells.append(code("""\
plt.figure(figsize=(8,6))
sns.scatterplot(data=inventory.sample(10000), x='closing_stock', y='units_sold', hue='is_stockout', alpha=0.3)
plt.title('G34: Closing Stock Buffer vs Units Sold')
plt.show()
"""))

cells.append(md("### G35: Local Event Impact by Category"))
cells.append(code("""\
event_cat = df.merge(event[['store_id', 'date', 'event_type']], on=['store_id', 'date'], how='left')
event_cat['has_event'] = event_cat['event_type'].notna()
event_cat_lift = event_cat.groupby(['category', 'has_event'])['units_sold'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(data=event_cat_lift, x='category', y='units_sold', hue='has_event')
plt.title('G35: Local Event Impact by Category')
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(md("### G36: Daily Order Volume (Drive-Through vs Standard)"))
cells.append(code("""\
dt_impact = df.groupby(['store_id', 'date']).agg({'order_id': 'nunique', 'has_drive_through': 'first'})
plt.figure(figsize=(8,6))
sns.boxplot(data=dt_impact, x='has_drive_through', y='order_id', showfliers=False)
plt.title('G36: Daily Order Volume (Drive-Through vs Standard)')
plt.show()
"""))

cells.append(md("### G37: Weekend Revenue Uplift % by Neighborhood"))
cells.append(code("""\
wknd_df = df.merge(date_dim[['date', 'is_weekend']], on='date')
wknd_impact = wknd_df.groupby(['neighborhood_type', 'is_weekend'])['revenue'].mean().unstack()
wknd_impact['uplift_pct'] = (wknd_impact[True] - wknd_impact[False]) / wknd_impact[False] * 100
plt.figure(figsize=(10,6))
wknd_impact['uplift_pct'].plot(kind='bar', color='teal')
plt.title('G37: Weekend Revenue Uplift % by Neighborhood')
plt.show()
"""))

cells.append(md("### G38: Day-of-Week Volatility (Stores 1-5 Sample)"))
cells.append(code("""\
dow_vol = df.merge(date_dim[['date', 'day_of_week']], on='date')
dow_vol_store = dow_vol.groupby(['store_id', 'date', 'day_of_week'])['revenue'].sum().reset_index()
plt.figure(figsize=(10,6))
sns.boxplot(data=dow_vol_store[dow_vol_store['store_id'] <= 5], x='day_of_week', y='revenue', hue='store_id', showfliers=False)
plt.title('G38: Day-of-Week Volatility (Stores 1-5 Sample)')
plt.show()
"""))

cells.append(md("### G39: Autocorrelation (ACF) of Overall Daily Demand"))
cells.append(code("""\
from statsmodels.graphics.tsaplots import plot_acf
daily_demand = df.groupby('date')['units_sold'].sum()
plt.figure(figsize=(12,5))
plot_acf(daily_demand, lags=30, ax=plt.gca())
plt.title('G39: Autocorrelation (ACF) of Overall Daily Demand')
plt.show()
"""))

cells.append(md("### G40: Revenue Volatility (Standard Deviation) by Store"))
cells.append(code("""\
store_volatility = df.groupby(['store_id', 'date'])['revenue'].sum().groupby('store_id').std().sort_values()
plt.figure(figsize=(8,8))
store_volatility.plot(kind='barh', color='purple')
plt.title('G40: Revenue Volatility (Standard Deviation) by Store')
plt.show()
"""))

# --- SECTION 7: TABULAR INSIGHTS ---
cells.append(md("## 7. 🗂️ Unexpected Tabular Insights (The 'Crazy' Tables)\nDeep, table-based analysis showcasing extreme, counter-intuitive data points."))

cells.append(md("### T1: The 'Anti-Promo' Stores\nStores where running a promotion paradoxically *decreased* their average daily revenue."))
cells.append(code("""\
promo_effect = daily_promo.groupby(['store_id', 'has_promo'])['units_sold'].mean().unstack()
promo_effect['promo_lift_pct'] = (promo_effect[True] - promo_effect[False]) / promo_effect[False] * 100
anti_promo = promo_effect[promo_effect['promo_lift_pct'] < 0].sort_values('promo_lift_pct')
anti_promo_stores = anti_promo.reset_index().merge(store[['store_id', 'neighborhood_type']], on='store_id')
anti_promo_stores.style.background_gradient(cmap='Reds_r', subset=['promo_lift_pct']).set_caption("T1: Stores with NEGATIVE Promotion Lift")
"""))

cells.append(md("### T2: The Customer 'Polygamy' Index\nTop customers who visit the highest number of distinct store locations, showing extreme brand loyalty but zero location loyalty."))
cells.append(code("""\
customer_loyalty = order.dropna(subset=['customer_id']).groupby('customer_id').agg(
    total_spend=('revenue', 'sum'),
    unique_stores=('store_id', 'nunique'),
    total_orders=('order_id', 'nunique')
)
polygamists = customer_loyalty[customer_loyalty['unique_stores'] >= 5].sort_values('total_spend', ascending=False).head(10)
polygamists.style.background_gradient(cmap='Greens', subset=['total_spend']).highlight_max(subset=['unique_stores'], color='yellow').set_caption("T2: High-Value Customers visiting 5+ Locations")
"""))

cells.append(md("### T3: The 'Ghost' Inventory Matrix\nProducts that show as 'In Stock' (Closing Stock > 0) but generated exactly 0 units sold across the network, indicating potential data errors or extremely dead stock."))
cells.append(code("""\
ghost_stock = inventory[(inventory['is_stockout'] == False) & (inventory['closing_stock'] > 10) & (inventory['units_sold'] == 0)]
ghost_summary = ghost_stock.groupby('product_id').agg(
    zero_sales_days=('date', 'nunique'),
    avg_closing_stock=('closing_stock', 'mean')
).sort_values('zero_sales_days', ascending=False).head(10)
ghost_summary = ghost_summary.merge(prod[['product_id', 'product_name', 'category']], on='product_id')
ghost_summary.style.background_gradient(cmap='Oranges').set_caption("T3: Products with High Stock but ZERO Sales")
"""))

cells.append(md("### T4: The 'Whale' Diet (Basket Composition)\nWhat exactly do the top 1% of spenders buy compared to the bottom 50%?"))
cells.append(code("""\
top_1_pct_threshold = customer_revenue.quantile(0.99)
whales_1 = customer_revenue[customer_revenue >= top_1_pct_threshold].index
df['is_whale'] = df['customer_id'].isin(whales_1)

whale_diet = df.groupby(['is_whale', 'category'])['units_sold'].sum().unstack()
whale_diet = whale_diet.div(whale_diet.sum(axis=1), axis=0) * 100
whale_diet.index = ['Bottom 99%', 'Top 1% Whales']
whale_diet.style.background_gradient(cmap='Blues', axis=1).format("{:.1f}%").set_caption("T4: Category Purchase Breakdown (%)")
"""))

cells.append(md("### T5: Weather-Defying Branches\nStores that actually perform *better* on rainy days than on dry days (Positive Rain Coefficient)."))
cells.append(code("""\
rain_effect = rainy_df.groupby(['store_id', 'is_rainy_season'])['revenue'].mean().unstack()
rain_effect['rain_uplift_pct'] = (rain_effect[True] - rain_effect[False]) / rain_effect[False] * 100
weather_defying = rain_effect[rain_effect['rain_uplift_pct'] > 5].sort_values('rain_uplift_pct', ascending=False)
weather_defying_stores = weather_defying.reset_index().merge(store[['store_id', 'neighborhood_type', 'has_drive_through']], on='store_id')
weather_defying_stores.style.background_gradient(cmap='Blues', subset=['rain_uplift_pct']).set_caption("T5: Stores that Profit from the Rain")
"""))

print("Merging new split cells with original notebook...")
final_cells = old_cells + cells

final_nb = {
    "cells": final_cells,
    "metadata": old_nb.get("metadata", {}),
    "nbformat": old_nb.get("nbformat", 4),
    "nbformat_minor": old_nb.get("nbformat_minor", 5)
}

with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(final_nb, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Notebook with SPLIT cells and Tabular Insights generated at:\n{out_nb_path}")
