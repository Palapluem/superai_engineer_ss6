import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Load data exactly like cell 1
path = r'Level 2/Hackathon 5_Demand Forecasting Coffee Chain Hackathon/super-ai-engineer-season-6-coffee-chain-hackathon/train/'
txn = pd.read_csv(path + 'TRANSACTION.csv')
order = pd.read_csv(path + 'ORDER.csv', parse_dates=['date'])
cust = pd.read_csv(path + 'CUSTOMER.csv', parse_dates=['registration_date'])
promo = pd.read_csv(path + 'PROMOTION.csv', parse_dates=['start_date', 'end_date'])
store = pd.read_csv(path + 'STORE.csv', parse_dates=['opened_date'])
prod = pd.read_csv(path + 'PRODUCT.csv')
date_dim = pd.read_csv(path + 'DATE_DIM.csv', parse_dates=['date'])
event = pd.read_csv(path + 'LOCAL_EVENT.csv', parse_dates=['date'])
inventory = pd.read_csv(path + 'INVENTORY.csv', parse_dates=['date'])

df = txn.merge(order, on='order_id', how='left')
df = df.merge(prod, on='product_id', how='left')
df = df.merge(store, on='store_id', how='left')

# Prepare promo_df exactly like cell 13
promo_expanded = []
for _, r in promo.iterrows():
    dates = pd.date_range(r['start_date'], r['end_date'])
    for d in dates:
        promo_expanded.append({
            'store_id': r['store_id'], 'date': d, 'product_id': r['product_id'],
            'promo_type': r['promo_type'], 'discount_pct': r['discount_pct']
        })
promo_df = pd.DataFrame(promo_expanded).drop_duplicates(subset=['store_id', 'date', 'product_id'])

# Disable plt.show
plt.show = lambda: None

# Find Section 8 cells
sec8_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and 'Section 8' in str(cell.get('source')):
        sec8_found = True
    if sec8_found and cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        print("Executing code snippet...")
        exec(src, globals())

print("All Section 8 charts executed without errors!")
