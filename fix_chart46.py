import json

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Chart 46 code cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Chart 46' in str(cell.get('source')):
        # Code cell is i+1
        code_cell = nb['cells'][i+1]
        new_source = [
            "df_c46 = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-10-31')].copy()\n",
            "daily_nh = df_c46.groupby(['date', 'neighborhood_type'])['revenue'].sum().unstack(fill_value=0)\n",
            "daily_nh_ma = daily_nh.rolling(7, min_periods=1).mean()\n",
            "evt_days = event[(event['date'] >= '2024-01-01') & (event['date'] <= '2024-10-31')]['date'].unique()\n",
            "\n",
            "fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios':[1.3, 1]})\n",
            "ax0 = axes[0]\n",
            "sns.lineplot(data=daily_nh_ma, palette='Set1', linewidth=2, ax=ax0)\n",
            "for ed in evt_days:\n",
            "    if ed in daily_nh_ma.index:\n",
            "        ax0.plot(ed, daily_nh_ma.loc[ed, 'mall'] if 'mall' in daily_nh_ma.columns else daily_nh_ma.iloc[:,0].loc[ed], marker='*', color='black', markersize=14)\n",
            "ax0.plot([], [], marker='*', color='black', linestyle='None', label='Local Event Marker', markersize=12)\n",
            "ax0.set_title('Chart 46: Local Event Cannibalization Timeline (Top: 7-day MA Revenue | Bottom: Promo Revenue Split)', fontsize=16, pad=15)\n",
            "ax0.set_ylabel('7-day MA Revenue')\n",
            "ax0.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "\n",
            "ax1 = axes[1]\n",
            "# Merge with promo_df created earlier in Section 3\n",
            "if 'promo_df' in globals():\n",
            "    df_c46 = df_c46.merge(promo_df[['store_id', 'date', 'product_id', 'promo_type']], on=['store_id', 'date', 'product_id'], how='left')\n",
            "    df_c46['rev_type'] = np.where(df_c46['promo_type'].notna(), 'Promo Revenue', 'Walk-in Revenue')\n",
            "else:\n",
            "    # Fallback approximation if run isolated\n",
            "    df_c46['rev_type'] = 'Walk-in Revenue'\n",
            "\n",
            "daily_split = df_c46.groupby(['date', 'rev_type'])['revenue'].sum().unstack(fill_value=0)\n",
            "daily_split.plot(kind='bar', stacked=True, color=['#80cbc4', '#e57373'], ax=ax1, width=1.0, legend=False)\n",
            "ticks = ax1.get_xticks()\n",
            "step = max(1, len(ticks) // 15)\n",
            "ax1.set_xticks(ticks[::step])\n",
            "ax1.set_xticklabels([daily_split.index[t].strftime('%Y-%m-%d') for t in ticks[::step]], rotation=45, ha='right')\n",
            "ax1.set_ylabel('Daily Revenue')\n",
            "ax1.set_xlabel('Date')\n",
            "ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "plt.margins(y=0.2)\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
        code_cell['source'] = new_source
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Chart 46 fixed!")
