import json
import os

# Paths
old_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-lv2-nb-expresson-hackathon (4).ipynb"
adv_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-eda-advanced.ipynb"
out_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-MASTER.ipynb"

print("Loading existing notebooks...")
with open(old_nb_path, 'r', encoding='utf-8') as f:
    old_nb = json.load(f)

with open(adv_nb_path, 'r', encoding='utf-8') as f:
    adv_nb = json.load(f)

old_cells = old_nb.get('cells', [])
adv_cells = adv_nb.get('cells', [])

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in src.split('\n')]}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in src.split('\n')]}

print("Generating super-deep insight cells...")
super_deep_cells = [
md("## 5. 🧠 Super-Deep Insights: Unexpected Data Relationships (Machine Learning Features)"),
code("""\
# G22: The 'Payday x High-Value Location' Interaction
# Do people splurge more on paydays specifically at office/hospital locations?
payday_df = df.merge(date_dim[['date', 'is_payday']], on='date')
payday_impact = payday_df.groupby(['neighborhood_type', 'is_payday'])['revenue'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=payday_impact, x='neighborhood_type', y='revenue', hue='is_payday', palette='coolwarm')
plt.title('G22: Payday Splurge Effect by Neighborhood (Unexpected Interaction)')
plt.show()

# G23: Stockout Domino Effect
# If Merchandise stocks out, do they buy more Coffee?
try:
    inventory = pd.read_csv(path + 'INVENTORY.csv')
    stockout_df = df.merge(inventory[['store_id', 'product_id', 'date', 'is_stockout']], on=['store_id', 'product_id', 'date'], how='left')
    stockout_daily = stockout_df.groupby(['store_id', 'date', 'category'])['is_stockout'].max().unstack().fillna(False)
    coffee_sales = df[df['category']=='Coffee'].groupby(['store_id', 'date'])['units_sold'].sum().reset_index()
    coffee_vs_merch_stockout = coffee_sales.merge(stockout_daily[['Merchandise']], left_on=['store_id', 'date'], right_index=True, how='left')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=coffee_vs_merch_stockout, x='Merchandise', y='units_sold', showfliers=False, palette='Set2')
    plt.title('G23: Do Coffee Sales Increase when Merchandise is Sold Out? (Substitution Effect)')
    plt.xlabel('Merchandise Stockout (True/False)')
    plt.show()
except:
    pass

# G24: Staff Stress Index
# High orders per staff member -> does it lead to smaller basket sizes? (Rushed service reduces upsells)
staff_stress = df.groupby(['store_id', 'date']).agg({'order_id': 'nunique', 'units_sold': 'sum', 'staff_count': 'first'}).reset_index()
staff_stress['orders_per_staff'] = staff_stress['order_id'] / staff_stress['staff_count']
staff_stress['basket_size'] = staff_stress['units_sold'] / staff_stress['order_id']
plt.figure(figsize=(10, 6))
sns.regplot(data=staff_stress, x='orders_per_staff', y='basket_size', scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('G24: Staff Stress vs Basket Size (Does rushed service reduce upsells?)')
plt.show()

# G25: Macro Trend - Price Sensitivity
# Has the average revenue per unit increased? (Inflation/Menu changes)
price_trend = df.groupby('date').agg({'revenue': 'sum', 'units_sold': 'sum'}).reset_index()
price_trend['avg_price_per_unit'] = price_trend['revenue'] / price_trend['units_sold']
plt.figure(figsize=(14, 5))
sns.lineplot(data=price_trend, x='date', y='avg_price_per_unit', color='darkgreen')
plt.title('G25: Macro Trend - Average Price Per Unit Over 22 Months')
plt.show()
""")
]

print("Merging notebooks...")
final_cells = old_cells + [md("---"), md("# 🚀 ADVANCED EDA MODULE (Deep Insights from Raw Data)")] + adv_cells + super_deep_cells

final_nb = {
    "cells": final_cells,
    "metadata": old_nb.get("metadata", {}),
    "nbformat": old_nb.get("nbformat", 4),
    "nbformat_minor": old_nb.get("nbformat_minor", 5)
}

with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(final_nb, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Combined MASTER Notebook generated at:\n{out_nb_path}")
