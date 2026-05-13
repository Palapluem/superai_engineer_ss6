import json

path = r'Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-lv2-nb-expresson-hackathon (4).ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch_notebook_dump.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        ctype = c['cell_type']
        src = "".join(c.get('source', []))[:100].replace('\n', '\\n')
        out.write(f"Cell {i} [{ctype}]: {src}\n")
