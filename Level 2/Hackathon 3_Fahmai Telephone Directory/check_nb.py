import json

nb = json.load(open('Hackathon 3_LV2_Fahmai Telephone Directory.ipynb', encoding='utf-8'))

all_src = []
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        all_src.append(src)

full = '\n'.join(all_src)

print("=== Notebook Pre-flight Check ===")
print()

# 1. row.get() on pandas Series bug
if 'row.get(' in full:
    print("[BUG] row.get() detected - pandas Series does not have .get(), use row['col'] instead")
else:
    print("[OK] No row.get() issue")

# 2. detect_thai defined but never called (dead code)
import re
calls = re.findall(r'detect_thai\(', full)
defs  = re.findall(r'def detect_thai', full)
if defs and len(calls) == 0:
    print("[WARN] detect_thai() is defined but never called (dead code, harmless)")
else:
    print("[OK] detect_thai usage OK:", len(calls), "calls")

# 3. Check TOOL schema defined before df_emp
tool_idx = full.find('GREP_CSV_TOOL')
emp_idx  = full.find('df_emp = pd.read_csv')
if tool_idx < emp_idx:
    print("[BUG] GREP_CSV_TOOL defined before df_emp is loaded")
else:
    print("[OK] GREP_CSV_TOOL defined after df_emp")

# 4. Check SYSTEM_PROMPT is a string concatenation (no triple-quote issues)
if 'SYSTEM_PROMPT = (' in full:
    print("[OK] SYSTEM_PROMPT uses string concatenation (safe)")
elif 'SYSTEM_PROMPT = """' in full:
    print("[WARN] SYSTEM_PROMPT uses triple-quote - check for escape issues")

# 5. Check model name
if 'typhoon-v2.5' in full:
    import re
    models = re.findall(r"typhoon-v2\.5[\\w-]*", full)
    print(f"[OK] Model found: {set(models)}")
else:
    print("[BUG] No typhoon-v2.5 model name found!")

# 6. Check submission format
if '"id"' in full and '"response"' in full:
    print("[OK] Submission columns id+response present")
else:
    print("[BUG] Missing id or response column in submission")

# 7. Check grade.py call
if 'grade.py' in full and 'subprocess' in full:
    print("[OK] Local validation with grade.py present")
else:
    print("[WARN] No local validation cell found")

# 8. Check for any stray unicode box chars that would break on Windows terminals
box_chars = [c for c in full if ord(c) in range(0x2500, 0x2600)]
if box_chars:
    print(f"[WARN] Unicode box-drawing chars found in code cells: {len(box_chars)} - OK in Jupyter but may break on terminals")
else:
    print("[OK] No box-drawing chars in code cells")

# 9. Check cell count
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells   = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"\n[INFO] Total cells: {len(nb['cells'])} ({len(code_cells)} code, {len(md_cells)} markdown)")

print()
print("=== Check Complete ===")
