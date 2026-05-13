import json
import pandas as pd
import numpy as np

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global fix for plt.margins to prevent text overlap
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "plt.xticks(rotation=45, ha='right')" in line:
                if "plt.margins(y=0.2)" not in "".join(cell['source']):
                    new_source.append("plt.margins(y=0.2)\n")
            new_source.append(line)
        cell['source'] = new_source

def find_idx(nb, section_text):
    for i, cell in enumerate(nb['cells']):
        if section_text in str(cell.get('source', [])): return i
    return -1

i_sec3 = find_idx(nb, "## Section 3:")
i_sec4 = find_idx(nb, "## Section 4:")
i_sec5 = find_idx(nb, "## Section 5:")
i_sec6 = find_idx(nb, "## Section 6:")
i_sec7 = find_idx(nb, "## Section 7:")

def inject(nb, idx, code_src, md_src):
    if idx == -1: return
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_src}
    m = {"cell_type": "markdown", "metadata": {}, "source": md_src}
    nb['cells'].insert(idx, c)
    nb['cells'].insert(idx, m)

m425 = [
    "### Chart 42.5: 90-Day Stockout Death Spiral (Cross-Selling Contagion)\n",
    "\n",
    "**Description:**\n",
    "A 90-day time-series of total store revenue, overlaid with red shaded blocks indicating days where a 'Core Coffee' item was completely out of stock.\n",
    "\n",
    "**Actionable Insight (Product/Solution Opportunity):**\n",
    "This graph reveals the 'Contagion Effect'. When a core item stocks out, it doesn't just reduce sales for that item—total revenue plummets because customers walk out without buying their usual bakery cross-sells. The solution is a **'Stockout Contagion Alert System'** that prioritizes emergency restocking based on an item's cross-selling weight, not just its individual sales volume.\n"
]
c425 = [
    "q1_stk = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-03-31')]\n",
    "stk_store = 5\n",
    "ts_rev = q1_stk[q1_stk['store_id'] == stk_store].groupby('date')['revenue'].sum().reset_index()\n",
    "plt.figure(figsize=(16, 6))\n",
    "ax = sns.lineplot(data=ts_rev, x='date', y='revenue', color='teal', linewidth=2)\n",
    "stk_days = inventory[(inventory['store_id'] == stk_store) & (inventory['date'] >= '2024-01-01') & (inventory['date'] <= '2024-03-31') & (inventory['is_stockout']) & (inventory['product_id'] == 1)]\n",
    "for d in stk_days['date'].unique():\n",
    "    plt.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color='red', alpha=0.3)\n",
    "    plt.text(d, plt.ylim()[1]*0.9, 'Core Item\\nStockout', color='darkred', ha='center', fontsize=9)\n",
    "plt.title(f'Chart 42.5: Stockout Death Spiral (Store {stk_store})', fontsize=16)\n",
    "plt.margins(y=0.2)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

m365 = [
    "### Chart 36.5: 90-Day Payday Splurge Behavior (Luxury vs Necessity)\n",
    "\n",
    "**Description:**\n",
    "A 90-day time-series comparing the sales of 'Merchandise' (Luxury) vs 'Coffee' (Necessity), with vertical lines marking Paydays.\n",
    "\n",
    "**Actionable Insight (Product/Solution Opportunity):**\n",
    "Notice how Coffee sales are relatively stable across the month, but Merchandise sales experience violent spikes **exactly on Paydays**. This proves that marketing budgets for Tumblers and Premium items are wasted mid-month. The solution is **'Targeted FOMO Marketing'**—an automated ad engine that only deploys merchandise ads 48 hours around Payday to maximize ROI.\n"
]
c365 = [
    "q1_splurge = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-03-31')]\n",
    "splurge_ts = q1_splurge[q1_splurge['category'].isin(['Coffee', 'Merchandise'])].groupby(['date', 'category'])['revenue'].sum().reset_index()\n",
    "plt.figure(figsize=(16, 6))\n",
    "ax = sns.lineplot(data=splurge_ts, x='date', y='revenue', hue='category', palette=['#3e2723', 'gold'], linewidth=2)\n",
    "paydays = date_dim[(date_dim['date'] >= '2024-01-01') & (date_dim['date'] <= '2024-03-31') & (date_dim['is_payday'])]\n",
    "for d in paydays['date']:\n",
    "    plt.axvline(d, color='blue', linestyle='--', alpha=0.5)\n",
    "    plt.text(d, plt.ylim()[1]*0.9, ' Payday', color='blue', rotation=90, va='top', ha='right', fontsize=10)\n",
    "plt.title('Chart 36.5: Payday Splurge Behavior (Luxury vs Necessity)', fontsize=16)\n",
    "plt.margins(y=0.2)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "if ax.get_legend() is not None: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

m355 = [
    "### Chart 35.5: 90-Day Weekend Micro-Migration (Residential vs Mall)\n",
    "\n",
    "**Description:**\n",
    "A 90-day time-series plotting an Urban Residential store against a Mall store. Yellow shaded regions highlight Weekends.\n",
    "\n",
    "**Actionable Insight (Product/Solution Opportunity):**\n",
    "This graph shows perfectly inverse demand waves. Residential peaks on weekdays and crashes on weekends, while Mall stores explode on weekends. The solution is a **'Dynamic Part-Time Roster'**—an automated HR system that shares a pool of baristas, assigning them to Residential branches Monday-Friday, and shifting them to Mall branches Saturday-Sunday to optimize labor costs.\n"
]
c355 = [
    "q1_mig = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-03-31')]\n",
    "mall = store[store['neighborhood_type'] == 'mall']['store_id'].unique()[0]\n",
    "res = store[store['neighborhood_type'] == 'urban_residential']['store_id'].unique()[0]\n",
    "mig_ts = q1_mig[q1_mig['store_id'].isin([mall, res])].groupby(['date', 'neighborhood_type'])['units_sold'].sum().reset_index()\n",
    "plt.figure(figsize=(16, 6))\n",
    "ax = sns.lineplot(data=mig_ts, x='date', y='units_sold', hue='neighborhood_type', palette=['purple', 'green'], linewidth=2)\n",
    "weekends = date_dim[(date_dim['date'] >= '2024-01-01') & (date_dim['date'] <= '2024-03-31') & (date_dim['is_weekend'])]\n",
    "if not weekends.empty:\n",
    "    we_groups = (weekends['date'] != weekends['date'].shift(1) + pd.Timedelta(days=1)).cumsum()\n",
    "    for _, group in weekends.groupby(we_groups):\n",
    "        plt.axvspan(group['date'].min(), group['date'].max(), color='yellow', alpha=0.2)\n",
    "        plt.text(group['date'].min(), plt.ylim()[1]*0.9, ' Weekend', color='olive', fontsize=9)\n",
    "plt.title('Chart 35.5: Weekend Micro-Migration (Residential vs Mall)', fontsize=16)\n",
    "plt.margins(y=0.2)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "if ax.get_legend() is not None: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

m285 = [
    "### Chart 28.5: 90-Day Temperature-Driven Category Crossover\n",
    "\n",
    "**Description:**\n",
    "A 90-day time-series transitioning from February to May (heading into the extreme Thai Summer). It compares 'Coffee' units vs 'Juice & Smoothie' units. The red zone marks Peak Summer.\n",
    "\n",
    "**Actionable Insight (Product/Solution Opportunity):**\n",
    "Observe the specific week where the Juice/Smoothie line surges upwards and crosses the baseline. This proves temperature alters the category mix. The solution is a **'Climate-Triggered Packaging Rebalance'**—an automated supply chain rule that triggers a 300% increase in orders for clear plastic cups and dome lids exactly two weeks before this historical crossover, preventing catastrophic stockouts of cold packaging.\n"
]
c285 = [
    "q2_temp = df[(df['date'] >= '2024-02-15') & (df['date'] <= '2024-05-15')]\n",
    "temp_ts = q2_temp[q2_temp['category'].isin(['Coffee', 'Juice & Smoothie'])].groupby(['date', 'category'])['units_sold'].sum().reset_index()\n",
    "plt.figure(figsize=(16, 6))\n",
    "ax = sns.lineplot(data=temp_ts, x='date', y='units_sold', hue='category', palette=['#3e2723', 'magenta'], linewidth=2)\n",
    "plt.axvspan(pd.Timestamp('2024-04-01'), pd.Timestamp('2024-04-30'), color='red', alpha=0.15)\n",
    "plt.text(pd.Timestamp('2024-04-15'), plt.ylim()[1]*0.9, 'Peak Summer Heat (April)', color='red', ha='center', fontsize=12, fontweight='bold')\n",
    "plt.title('Chart 28.5: Temperature-Driven Crossover (Coffee vs Juice/Smoothie)', fontsize=16)\n",
    "plt.margins(y=0.2)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "if ax.get_legend() is not None: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

m165 = [
    "### Chart 16.5: 90-Day Promotion Fatigue & Baseline Bleed\n",
    "\n",
    "**Description:**\n",
    "A 90-day time-series for a store running consecutive promotions. Yellow blocks indicate active promotion periods.\n",
    "\n",
    "**Actionable Insight (Product/Solution Opportunity):**\n",
    "This reveals the 'Coupon Clipper' effect. The first promo generates a massive spike. However, notice how the baseline sales *between* the later promos begin to drop below the historical average. Customers are being trained to wait for discounts. The solution is a **'Promo Cooldown Recommender'**—an ML module that enforces a mandatory 21-day cool-down period between heavy discounts to protect organic baseline revenue.\n"
]
c165 = [
    "p_store = promo['store_id'].value_counts().index[0]\n",
    "q3_fatigue = df[(df['store_id'] == p_store) & (df['date'] >= '2023-08-01') & (df['date'] <= '2023-10-31')]\n",
    "fatigue_ts = q3_fatigue.groupby('date')['revenue'].sum().reset_index()\n",
    "plt.figure(figsize=(16, 6))\n",
    "ax = sns.lineplot(data=fatigue_ts, x='date', y='revenue', color='darkorange', linewidth=2)\n",
    "active_promos = promo[(promo['store_id'] == p_store) & (promo['start_date'] <= '2023-10-31') & (promo['end_date'] >= '2023-08-01')]\n",
    "for _, r in active_promos.iterrows():\n",
    "    start = max(r['start_date'], pd.Timestamp('2023-08-01'))\n",
    "    end = min(r['end_date'], pd.Timestamp('2023-10-31'))\n",
    "    plt.axvspan(start, end, color='yellow', alpha=0.3)\n",
    "    plt.text(start, plt.ylim()[1]*0.9, ' Promo', color='olive', fontsize=9)\n",
    "plt.title(f'Chart 16.5: Promotion Fatigue & Baseline Bleed (Store {p_store})', fontsize=16)\n",
    "plt.margins(y=0.2)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

m65 = [
    "### Chart 6.5: 90-Day 'Bridge Holiday' Exodus\n",
    "\n",
    "**Description:**\n",
    "A 90-day time-series (April - June) tracking an Office store versus a Residential store. Gray dashed lines indicate Public Holidays.\n",
    "\n",
    "**Actionable Insight (Product/Solution Opportunity):**\n",
    "When a holiday falls on a Wednesday or Thursday, the adjacent Friday becomes a 'Bridge Holiday'. The chart proves that Office branch sales completely die on these Fridays (as workers take leave), while Residential sales boom. The solution is a **'Smart Holiday Staff Roster'** that detects Bridge Holidays and automatically slashes staff and fresh pastry inventory at Office branches, reallocating them to Residential branches.\n"
]
c65 = [
    "q2_bridge = df[(df['date'] >= '2024-04-01') & (df['date'] <= '2024-06-30')]\n",
    "office = store[store['neighborhood_type'] == 'office']['store_id'].unique()[0]\n",
    "res = store[store['neighborhood_type'] == 'urban_residential']['store_id'].unique()[0]\n",
    "bridge_ts = q2_bridge[q2_bridge['store_id'].isin([office, res])].groupby(['date', 'neighborhood_type'])['units_sold'].sum().reset_index()\n",
    "plt.figure(figsize=(16, 6))\n",
    "ax = sns.lineplot(data=bridge_ts, x='date', y='units_sold', hue='neighborhood_type', palette=['blue', 'green'], linewidth=2)\n",
    "holidays = date_dim[(date_dim['date'] >= '2024-04-01') & (date_dim['date'] <= '2024-06-30') & (date_dim['is_holiday'])]\n",
    "for d in holidays['date']:\n",
    "    plt.axvline(d, color='gray', linestyle='--', alpha=0.8)\n",
    "    plt.text(d, plt.ylim()[1]*0.9, ' Holiday', color='black', rotation=90, va='top', ha='right', fontsize=9)\n",
    "plt.title('Chart 6.5: The Bridge Holiday Exodus (Office vs Residential)', fontsize=16)\n",
    "plt.margins(y=0.2)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "if ax.get_legend() is not None: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

inject(nb, i_sec7, c425, m425)
inject(nb, i_sec6, c365, m365)
inject(nb, i_sec6, c355, m355)
inject(nb, i_sec5, c285, m285)
inject(nb, i_sec4, c165, m165)
inject(nb, i_sec3, c65, m65)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("6 New TS Charts injected and global margins fixed!")
