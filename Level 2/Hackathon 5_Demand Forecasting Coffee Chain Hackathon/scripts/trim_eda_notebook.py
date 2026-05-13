import json
import re
from pathlib import Path

# Config
notebook_path = Path(r"Level 2/Hackathon 5_Demand Forecasting Coffee Chain Hackathon/hackathon-5-EDA-PURE-MASTER.ipynb")
out_path = notebook_path.with_name(notebook_path.stem + "-trimmed.ipynb")
keep_nums = {1,2,41,42,43,44,45,46,47}
season_keywords = ("season", "seasonality", "weekday", "dayofweek", "weekly")

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
cells = nb.get('cells', [])

heading_re = re.compile(r"###\s*Graph\s*(\d+)[:\s]", re.IGNORECASE)
keep_indices = set()

# find indices of graph headings
graph_positions = []
for i, c in enumerate(cells):
    if c.get('cell_type') != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    m = heading_re.search(src)
    if m:
        num = int(m.group(1))
        graph_positions.append((i, num, src))

# choose which graph heading indices to keep
season_candidate = None
for idx, num, src in graph_positions:
    if num in keep_nums:
        keep_indices.add(idx)
    else:
        lower = src.lower()
        if any(k in lower for k in season_keywords) and season_candidate is None:
            season_candidate = idx

if season_candidate is not None:
    keep_indices.add(season_candidate)

# include following code cell (if exists) for each kept heading
final_cells = []

# include all cells up to first kept heading (to preserve imports/defs)
if keep_indices:
    first_keep = min(keep_indices)
    for i in range(0, first_keep):
        final_cells.append(cells[i])

# now append kept headings and their next cell (if code)
for idx in sorted(keep_indices):
    final_cells.append(cells[idx])
    if idx + 1 < len(cells) and cells[idx+1].get('cell_type') == 'code':
        final_cells.append(cells[idx+1])

# minimal safety: if final_cells empty, copy original
if not final_cells:
    final_cells = cells

out_nb = nb.copy()
out_nb['cells'] = final_cells

out_path.write_text(json.dumps(out_nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f"Trimmed notebook written to: {out_path}")
