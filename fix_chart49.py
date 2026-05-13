import json

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Chart 49 code cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Chart 49' in str(cell.get('source')):
        code_cell = nb['cells'][i+1]
        new_source = [
            "df_c49 = df[df['customer_id'].notna()].merge(cust[['customer_id', 'registration_date']], on='customer_id', how='left')\n",
            "df_c49['vintage'] = df_c49['registration_date'].dt.year.fillna(2022).astype(int).astype(str)\n",
            "df_c49['is_we'] = np.where(df_c49['date'].dt.dayofweek >= 5, 'Weekend', 'Weekday')\n",
            "vint_nh = df_c49.groupby(['is_we', 'vintage', 'neighborhood_type'])['units_sold'].sum().unstack(fill_value=0)\n",
            "vint_pct = vint_nh.div(vint_nh.sum(axis=1), axis=0) * 100\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)\n",
            "we_list = ['Weekday', 'Weekend']\n",
            "# Expand palette to 10 colors to guarantee coverage for all neighborhood types\n",
            "nh_cols = ['#3e2723', '#d84315', '#ad1457', '#2e7d32', '#1565c0', '#8d6e63', '#6a1b29', '#004d40', '#e65100', '#37474f']\n",
            "\n",
            "for i, we in enumerate(we_list):\n",
            "    ax = axes[i]\n",
            "    we_data = vint_pct.loc[we]\n",
            "    we_data.plot(kind='barh', stacked=True, color=nh_cols[:len(we_data.columns)], ax=ax, width=0.6, legend=False)\n",
            "    ax.set_title(f'{we} Transactions by Vintage', fontsize=14, fontweight='bold')\n",
            "    ax.set_ylabel('Registration Vintage' if i==0 else '')\n",
            "    ax.set_xlabel('Neighborhood Share (%)')\n",
            "    ax.set_xlim(0, 100)\n",
            "    for container in ax.containers:\n",
            "        labels = [f'{w:.0f}%' if w > 6 else '' for w in container.datavalues]\n",
            "        ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)\n",
            "\n",
            "handles = [plt.Rectangle((0,0),1,1, color=nh_cols[idx]) for idx in range(len(vint_pct.columns))]\n",
            "fig.legend(handles, vint_pct.columns, bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=len(vint_pct.columns), fontsize=12)\n",
            "plt.suptitle('Chart 49: Cross-Store Habit Evolution by Registration Vintage', fontsize=20, fontweight='bold', y=0.98)\n",
            "plt.margins(y=0.2)\n",
            "plt.tight_layout(rect=[0, 0.05, 1, 0.95])\n",
            "plt.show()\n"
        ]
        code_cell['source'] = new_source
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Chart 49 fixed!")
