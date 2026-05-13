import json

path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-lv2-nb-expresson-hackathon (4).ipynb"
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for c in nb['cells']:
    if c['cell_type'] == 'code':
        code_cells.append(''.join(c['source']))

with open('parsed_old_nb.py', 'w', encoding='utf-8') as f:
    f.write('\n\n# -------- NEW CELL -------- \n\n'.join(code_cells))
