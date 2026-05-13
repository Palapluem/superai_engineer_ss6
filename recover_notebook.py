import json

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

fixed_count = 0
for cell in nb['cells']:
    src = cell.get('source')
    if isinstance(src, list) and len(src) > 5:
        # Check if this cell is fragmented (mostly 1-2 char strings)
        if all(len(c) <= 2 for c in src[:5]):
            recovered_str = "".join([c[0] if len(c)==2 and c[1]=='\n' else c for c in src])
            cell['source'] = recovered_str.splitlines(keepends=True)
            fixed_count += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Successfully recovered {fixed_count} cells!")
