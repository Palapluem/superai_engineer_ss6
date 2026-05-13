import json
import re

nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Identify blocks
blocks = []
current_block = []
block_type = "header" # header, section, chart
block_num = 0

for cell in nb['cells']:
    src = cell.get('source', [])
    if not src:
        current_block.append(cell)
        continue
    
    text = src[0].strip()
    if text.startswith("## Section"):
        if current_block:
            blocks.append((block_type, block_num, current_block))
        current_block = [cell]
        block_type = "section"
        match = re.search(r'Section (\d+)', text)
        block_num = float(match.group(1)) if match else block_num + 1
    elif text.startswith("### Chart"):
        if current_block:
            blocks.append((block_type, block_num, current_block))
        current_block = [cell]
        block_type = "chart"
        match = re.search(r'Chart (\d+\.?\d*)', text)
        block_num = float(match.group(1)) if match else block_num + 0.1
    elif text.startswith("### Table"):
        if current_block:
            blocks.append((block_type, block_num, current_block))
        current_block = [cell]
        block_type = "table"
        match = re.search(r'Table (\d+\.?\d*)', text)
        block_num = float(match.group(1)) if match else block_num + 0.1
    else:
        current_block.append(cell)

if current_block:
    blocks.append((block_type, block_num, current_block))

# Now we sort the blocks.
# We want header to be first.
# Then sections in order.
# Within each section, charts and tables in order. But wait, tables are in Section 7.
# Let's just assign a global sorting key to each block.
# Header = (0, 0, 0)
# Section N = (N, 0, 0)
# Chart X = (Section_of_X, 1, X)
# Table Y = (7, 2, Y)

def get_section_for_chart(chart_num):
    if chart_num <= 5: return 1
    if chart_num <= 12.5: return 2
    if chart_num <= 19.5: return 3
    if chart_num <= 26: return 4
    if chart_num <= 36.5: return 5
    if chart_num <= 42.5: return 6
    return 7

sorted_blocks = []
for b_type, b_num, cells in blocks:
    if b_type == "header":
        key = (0, 0, 0)
    elif b_type == "section":
        key = (b_num, 0, 0)
    elif b_type == "chart":
        sec = get_section_for_chart(b_num)
        key = (sec, 1, b_num)
    elif b_type == "table":
        key = (7, 2, b_num)
    sorted_blocks.append((key, cells))

sorted_blocks.sort(key=lambda x: x[0])

new_cells = []
for key, cells in sorted_blocks:
    new_cells.extend(cells)

nb['cells'] = new_cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Notebook cells sorted!")
