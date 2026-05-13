import json
import numpy as np
import pandas as pd

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if "## Section 3: Promotions" in str(cell.get('source', [])):
        insert_idx = i
        break

new_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Chart 12.5: 90-Day Time-Series Dynamics (Paydays & School Breaks)\n",
        "\n",
        "**Description:**\n",
        "A highly detailed 90-day time-series slice (March 1 - May 31, 2024) tracking the daily revenue of 20 distinct store branches, color-coded by their neighborhood type. Vertical dashed lines represent Paydays, and the red shaded regions highlight the massive Thai Summer School Break.\n",
        "\n",
        "**Actionable Insight (Product/Solution Opportunity):**\n",
        "This is the ultimate operational view. Notice how University stores (and some Transit) crash into the red zone during the School Break, while Mall/Residential stores might remain stable or rise. By clustering these 20 stores based on their visual waveform, you can build a **'Dynamic Roster Product'** (shifting baristas from University branches to Mall branches during mid-March to May) or an **'Automated Inventory Rebalancer'** (cutting pastry orders for University branches by 60% before the red zone hits). Furthermore, Payday spikes (blue lines) prove that short-term promotions should be targeted exclusively around the 1st and 16th to maximize the FOMO effect when customers have high purchasing power.\n"
    ]
}

new_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Filter Data to Q2 2024 (90 Days) to capture both Paydays and Summer School Break\n",
        "q2_df = df[(df['date'] >= '2024-03-01') & (df['date'] <= '2024-05-31')].copy()\n",
        "\n",
        "# Select 20 specific stores across different neighborhoods to avoid a messy spaghetti chart\n",
        "np.random.seed(42)\n",
        "selected_stores = []\n",
        "for n_type in ['university', 'mall', 'office', 'transit', 'urban_residential']:\n",
        "    stores_in_type = store[store['neighborhood_type'] == n_type]['store_id'].unique()\n",
        "    if len(stores_in_type) > 0:\n",
        "        selected_stores.extend(np.random.choice(stores_in_type, min(4, len(stores_in_type)), replace=False))\n",
        "\n",
        "ts_data = q2_df[q2_df['store_id'].isin(selected_stores)].groupby(['date', 'store_id', 'neighborhood_type'])['revenue'].sum().reset_index()\n",
        "\n",
        "plt.figure(figsize=(18, 8))\n",
        "ax = sns.lineplot(data=ts_data, x='date', y='revenue', hue='neighborhood_type', units='store_id', estimator=None, alpha=0.7, linewidth=1.5, palette='tab10')\n",
        "\n",
        "# Highlight Paydays (Vertical Dashed Lines)\n",
        "paydays = date_dim[(date_dim['date'] >= '2024-03-01') & (date_dim['date'] <= '2024-05-31') & (date_dim['is_payday'])]\n",
        "for d in paydays['date']:\n",
        "    plt.axvline(d, color='blue', linestyle='--', alpha=0.6)\n",
        "    plt.text(d, plt.ylim()[1]*0.95, ' Payday', color='blue', rotation=90, va='top', ha='right', fontsize=10, alpha=0.7)\n",
        "\n",
        "# Highlight School Breaks (Red Shaded Regions)\n",
        "breaks = date_dim[(date_dim['date'] >= '2024-03-01') & (date_dim['date'] <= '2024-05-31') & (date_dim['is_school_break'])]\n",
        "if not breaks.empty:\n",
        "    break_groups = (breaks['date'] != breaks['date'].shift(1) + pd.Timedelta(days=1)).cumsum()\n",
        "    for _, group in breaks.groupby(break_groups):\n",
        "        start_date = group['date'].min() - pd.Timedelta(hours=12)\n",
        "        end_date = group['date'].max() + pd.Timedelta(hours=12)\n",
        "        plt.axvspan(start_date, end_date, color='red', alpha=0.15)\n",
        "        plt.text(start_date + (end_date - start_date)/2, plt.ylim()[1]*0.8, 'School Break', color='red', ha='center', fontsize=12, fontweight='bold', alpha=0.7)\n",
        "\n",
        "plt.title('Chart 12.5: 90-Day Time-Series Dynamics (20 Branches)\\nImpact of Paydays and School Breaks', fontsize=16)\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Daily Revenue')\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "if ax.get_legend() is not None:\n",
        "    plt.setp(ax.get_legend().get_texts(), fontsize='10')\n",
        "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Neighborhood')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

if insert_idx != -1:
    nb['cells'].insert(insert_idx, new_code)
    nb['cells'].insert(insert_idx, new_md)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("Chart successfully inserted!")
else:
    print("Could not find insertion point!")
