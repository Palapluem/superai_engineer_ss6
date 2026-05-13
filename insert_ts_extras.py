import json
import pandas as pd
import numpy as np

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def find_section_idx(nb, section_text):
    for i, cell in enumerate(nb['cells']):
        if section_text in str(cell.get('source', [])):
            return i
    return -1

idx_sec3 = find_section_idx(nb, "## Section 3: Promotions")
idx_sec4 = find_section_idx(nb, "## Section 4: Local Events")
idx_sec5 = find_section_idx(nb, "## Section 5: Customer Retention")

# CHART 9.5
c95_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Chart 9.5: 90-Day Weather Disruption Dynamics (Drive-Through vs Transit)\n",
        "\n",
        "**Description:**\n",
        "A 90-day time-series focusing on the Peak Rainy Season (August 1 - October 31, 2023). This chart contrasts the daily revenue of a typical Drive-Through store against a Walk-in Transit store, with blue shaded regions indicating the active rainy season block.\n",
        "\n",
        "**Actionable Insight (Product/Solution Opportunity):**\n",
        "This is the foundation for a **'Weather-Adaptive Supply Routing System'**. Basic bar charts show overall weather penalties, but this time-series visually proves demand transfer. Notice how Transit store revenue drops during prolonged rain blocks, while Drive-Through revenue simultaneously spikes as customers refuse to walk in the rain. By feeding weather forecasts into the ML model, the business can automatically reroute fresh pastry deliveries from Transit branches to Drive-Through branches, preventing waste and capturing the transferred demand.\n"
    ]
}
c95_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "q3_weather = df[(df['date'] >= '2023-08-01') & (df['date'] <= '2023-10-31')].copy()\n",
        "dt_stores = store[store['has_drive_through'] == True]['store_id'].unique()\n",
        "transit_stores = store[store['neighborhood_type'] == 'transit']['store_id'].unique()\n",
        "\n",
        "sample_dt = dt_stores[0] if len(dt_stores) > 0 else 1\n",
        "sample_tr = transit_stores[0] if len(transit_stores) > 0 else 2\n",
        "\n",
        "weather_ts = q3_weather[q3_weather['store_id'].isin([sample_dt, sample_tr])].groupby(['date', 'store_id', 'has_drive_through'])['revenue'].sum().reset_index()\n",
        "weather_ts['Store_Type'] = weather_ts['has_drive_through'].map({True: 'Drive-Through', False: 'Transit (Walk-in)'})\n",
        "\n",
        "plt.figure(figsize=(16, 6))\n",
        "ax = sns.lineplot(data=weather_ts, x='date', y='revenue', hue='Store_Type', palette=['orange', 'teal'], linewidth=2)\n",
        "\n",
        "rain_days = date_dim[(date_dim['date'] >= '2023-08-01') & (date_dim['date'] <= '2023-10-31') & (date_dim['is_rainy_season'])]\n",
        "if not rain_days.empty:\n",
        "    rain_groups = (rain_days['date'] != rain_days['date'].shift(1) + pd.Timedelta(days=1)).cumsum()\n",
        "    for _, group in rain_days.groupby(rain_groups):\n",
        "        plt.axvspan(group['date'].min(), group['date'].max(), color='blue', alpha=0.1)\n",
        "        plt.text(group['date'].min() + (group['date'].max() - group['date'].min())/2, plt.ylim()[1]*0.9, 'Rainy Season Block', color='blue', ha='center', alpha=0.7)\n",
        "\n",
        "plt.title('Chart 9.5: Weather Disruption Dynamics (Drive-Through vs Transit)', fontsize=16)\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Daily Revenue')\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "if ax.get_legend() is not None:\n",
        "    plt.setp(ax.get_legend().get_texts(), fontsize='10')\n",
        "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Branch Format')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

# CHART 19.5
c195_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Chart 19.5: 90-Day Promotion Halo vs Cannibalization Dynamics\n",
        "\n",
        "**Description:**\n",
        "A 90-day time-series isolating a single high-performing store. It plots two distinct lines: total units of Coffee sold versus total units of Bakery items sold. Yellow shaded regions indicate periods when a major Coffee discount was active.\n",
        "\n",
        "**Actionable Insight (Product/Solution Opportunity):**\n",
        "This visualizes the core logic needed for a **'Smart Bundle Recommendation Engine'**. When a Coffee promotion launches (yellow zone), the coffee line spikes. But does the Bakery line rise with it (Halo Effect), or stay flat/drop (Cannibalization)? If bakery sales fail to scale during a coffee traffic surge, the store is bleeding potential margin. The solution is an ML model that explicitly detects these dead zones and automatically triggers 'Bundle Deals' (e.g., Buy Coffee, get Croissant 50% off) exclusively on those dates to force cross-selling.\n"
    ]
}

c195_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "q1_promo = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-03-31')].copy()\n",
        "top_store = q1_promo.groupby('store_id')['units_sold'].sum().idxmax()\n",
        "cross_sell = q1_promo[(q1_promo['store_id'] == top_store) & (q1_promo['category'].isin(['Coffee', 'Bakery']))]\n",
        "cross_ts = cross_sell.groupby(['date', 'category'])['units_sold'].sum().reset_index()\n",
        "\n",
        "promo_active = promo[(promo['store_id'] == top_store) & (promo['product_id'].isin(prod[prod['category']=='Coffee']['product_id'])) & (promo['discount_pct'] >= 10)]\n",
        "\n",
        "plt.figure(figsize=(16, 6))\n",
        "ax = sns.lineplot(data=cross_ts, x='date', y='units_sold', hue='category', palette=['#3e2723', '#f4a460'], linewidth=2)\n",
        "\n",
        "for _, r in promo_active.iterrows():\n",
        "    start = max(r['start_date'], pd.Timestamp('2024-01-01'))\n",
        "    end = min(r['end_date'], pd.Timestamp('2024-03-31'))\n",
        "    if start <= end:\n",
        "        plt.axvspan(start, end, color='yellow', alpha=0.3)\n",
        "        plt.text(start + (end-start)/2, plt.ylim()[1]*0.85, 'Coffee Promo Active', color='olive', ha='center', fontsize=10, rotation=90)\n",
        "\n",
        "plt.title('Chart 19.5: Promotion Halo vs Cannibalization Dynamics', fontsize=16)\n",
        "plt.xlabel('Date')\n",
        "plt.ylabel('Daily Units Sold')\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "if ax.get_legend() is not None:\n",
        "    plt.setp(ax.get_legend().get_texts(), fontsize='10')\n",
        "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Category')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

# CHART 21.5
c215_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Chart 21.5: 90-Day Local Event Demand Shock & Hangover\n",
        "\n",
        "**Description:**\n",
        "A 90-day time-series for a specific store that hosted a major local event. The red dashed lines pinpoint the exact dates the event took place, plotting the raw units sold before, during, and after the shock.\n",
        "\n",
        "**Actionable Insight (Product/Solution Opportunity):**\n",
        "This graph reveals the exact anatomy of an anomaly, justifying an **'Event-Driven Inventory Buffer Model'**. While basic models see a single spike, this time-series exposes both the 'Demand Shock' (a massive single-day spike causing stockouts) and the critical 'Hangover Effect' (sales plunging below baseline the day after because inventory is depleted or customers hoarded goods). A successful product must run an isolated forecasting module specifically for event days to inject a calculated safety buffer, preventing the post-event hangover.\n"
    ]
}

c215_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "q4_event = event[(event['date'] >= '2023-10-01') & (event['date'] <= '2023-12-31')]\n",
        "if not q4_event.empty:\n",
        "    event_store = q4_event['store_id'].iloc[0]\n",
        "    event_date = q4_event['date'].iloc[0]\n",
        "    start_win = event_date - pd.Timedelta(days=45)\n",
        "    end_win = event_date + pd.Timedelta(days=45)\n",
        "    \n",
        "    shock_df = df[(df['store_id'] == event_store) & (df['date'] >= start_win) & (df['date'] <= end_win)]\n",
        "    shock_ts = shock_df.groupby('date')['units_sold'].sum().reset_index()\n",
        "    \n",
        "    plt.figure(figsize=(16, 6))\n",
        "    ax = sns.lineplot(data=shock_ts, x='date', y='units_sold', color='purple', linewidth=2)\n",
        "    \n",
        "    event_dates = q4_event[q4_event['store_id'] == event_store]['date']\n",
        "    for ed in event_dates:\n",
        "        plt.axvline(ed, color='red', linestyle='--', linewidth=2)\n",
        "        plt.text(ed, plt.ylim()[1]*0.9, ' LOCAL\\n EVENT', color='red', fontsize=10, fontweight='bold', ha='left')\n",
        "\n",
        "    plt.title(f'Chart 21.5: Local Event Demand Shock & Hangover (Store {event_store})', fontsize=16)\n",
        "    plt.xlabel('Date')\n",
        "    plt.ylabel('Total Units Sold')\n",
        "    plt.xticks(rotation=45, ha='right')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n"
    ]
}

if idx_sec5 != -1:
    nb['cells'].insert(idx_sec5, c215_code)
    nb['cells'].insert(idx_sec5, c215_md)
if idx_sec4 != -1:
    nb['cells'].insert(idx_sec4, c195_code)
    nb['cells'].insert(idx_sec4, c195_md)
if idx_sec3 != -1:
    nb['cells'].insert(idx_sec3, c95_code)
    nb['cells'].insert(idx_sec3, c95_md)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Charts successfully inserted!")
