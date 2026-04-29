import json
from pathlib import Path
import re

# Paths
source_nb_path = Path(r"d:\superai_engineer_ss6\Level 2\Hackathon 3_Fahmai Telephone Directory\Hackathon 3_LV2_Fahmai Telephone Directory_Agentic_Detailed.ipynb")
target_nb_path = Path(r"d:\superai_engineer_ss6\Level 2\Hackathon 3_Fahmai Telephone Directory\Hackathon 3_LV2_Fahmai Telephone Directory_Agentic_Detailed copy.ipynb")

# User's specified Kaggle path
kaggle_prefix = "/kaggle/input/competitions/super-ai-engineer-season-6-fahmai-telephone-directory/"

def strip_emojis(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    emojis = ["🏗️", "💡", "✅", "⚠️", "🚀", "📊", "🤖"]
    for e in emojis:
        text = text.replace(e, "")
    return text.strip()

with open(source_nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

header_cells = []
config_cells = []
db_cells = []
main_cells = []

current_section = None

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        content = "".join(cell['source']).lower()
        if 'architecture' in content or 'outline' in content:
            current_section = 'header'
        elif 'configuration' in content:
            current_section = 'config'
        elif 'database' in content or 'data management' in content:
            current_section = 'db'
        elif 'orchestrator' in content or 'initialization' in content or 'stage' in content:
            current_section = 'main'
            
    if current_section == 'header': header_cells.append(cell)
    elif current_section == 'config': config_cells.append(cell)
    elif current_section == 'db': db_cells.append(cell)
    elif current_section == 'main': main_cells.append(cell)

new_cells = header_cells + config_cells + db_cells + main_cells

for cell in new_cells:
    if cell['cell_type'] == 'markdown':
        cell['source'] = [strip_emojis(line) for line in cell['source']]
    elif cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_import_block = False
        
        for line in source:
            # 1. Update Kaggle Paths ONLY in assignments
            # Check for: EMPLOYEES_CSV: str = "..." or EMPLOYEES_CSV = "..."
            if re.match(r'^\s*EMPLOYEES_CSV[:\w\s]*=', line):
                line = re.sub(r'=\s*["\'].*?["\']', f'= "{kaggle_prefix}employees.csv"', line)
            elif re.match(r'^\s*QUESTIONS_CSV[:\w\s]*=', line):
                line = re.sub(r'=\s*["\'].*?["\']', f'= "{kaggle_prefix}questions.csv"', line)
            
            # 2. Fix Kaggle Secrets for API Key
            if 'TYPHOON_API_KEY: str =' in line:
                line = "try:\n    from kaggle_secrets import UserSecretsClient\n    TYPHOON_API_KEY = UserSecretsClient().get_secret(\"TYPHOON_API_KEY\")\nexcept:\n    " + line
            
            # 3. Fix OutStream error
            if 'sys.stdout = io.TextIOWrapper' in line:
                continue
            
            # 4. Remove internal file imports (handle multi-line block)
            if 'from config import (' in line or 'from database import (' in line:
                skip_import_block = True
                continue
            if skip_import_block:
                if ')' in line:
                    skip_import_block = False
                continue
            if 'from config import' in line or 'from database import' in line:
                continue
                
            new_source.append(line)
        cell['source'] = new_source

nb['cells'] = new_cells

# 5. Title Update
if nb['cells'] and nb['cells'][0]['cell_type'] == 'markdown':
    nb['cells'][0]['source'][0] = "# [Super AI Engineer Season 6] Mini Hackathon 3 Level 2\n"
    nb['cells'][0]['source'][1] = "## FahMai Telephone Directory: Professional Agentic Pipeline (Kaggle Edition)\n"

with open(target_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook fixed with targeted regex and multi-line import handling.")
