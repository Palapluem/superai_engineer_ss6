import json
import re

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-MASTER-SPLIT.ipynb"

print(f"Loading notebook: {nb_path}")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Regex to match emojis (basic range covering most common emojis used)
emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        new_source = []
        for line in cell.get('source', []):
            # Targeted replacements for professional tone
            line = line.replace("🚀 ADVANCED EDA MODULE: 40 Deep Graphs & 5 Tabular Insights", "ADVANCED EDA MODULE: 40 Analytical Visualizations & 5 Tabular Insights")
            line = line.replace("## 7. 🗂️ Unexpected Tabular Insights (The 'Crazy' Tables)", "## 7. Unexpected Tabular Insights (Data Anomalies)")
            line = line.replace("📌 Description:", "Description:")
            line = line.replace("💡 Actionable Insight (Why it matters):", "Actionable Insight:")
            
            # Remove any remaining emojis
            line = emoji_pattern.sub(r'', line)
            
            new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("[SUCCESS] Notebook professionalized. All emojis removed.")
