import json, re

nb = json.load(open('Hackathon 3_LV2_Fahmai Telephone Directory.ipynb', encoding='utf-8'))
full = ''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')

print("=== FINAL VERIFICATION ===")
print()

# 1. Model name
models = re.findall(r'typhoon[a-z0-9._-]+', full)
print("Model strings:", set(models))

# 2. Submission columns
has_id   = ("'id'" in full) or ('"id"' in full)
has_resp = ("'response'" in full) or ('"response"' in full)
print("Has 'id' column:", has_id)
print("Has 'response' column:", has_resp)

# 3. row.get bug
print("row.get() bug:", "row.get(" in full)

# 4. Cell count
code_n = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md_n   = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f"Cells: {len(nb['cells'])} total ({code_n} code, {md_n} markdown)")

# 5. Key functions present
for fn in ['grep_csv', 'run_agent', 'process_question', 'SYSTEM_PROMPT', 'GREP_CSV_TOOL', 'submission_df']:
    print(f"  {'OK' if fn in full else 'MISSING'}: {fn}")

# 6. Kaggle path
print("Kaggle path:", '/kaggle/input/' in full)

# 7. grade.py validation
print("grade.py validation:", 'grade.py' in full)
