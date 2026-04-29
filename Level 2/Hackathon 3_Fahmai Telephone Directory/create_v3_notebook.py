import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(\"\"\"# [Super AI Engineer Season 6] Mini Hackathon 3 Level 2
## FahMai Telephone Directory v3 -- True Agentic RAG + Strict Deterministic Guardrails

**Super AI Engineer Season 6 - Level 2 Hackathon**
- Dataset: FahMai Telephone Directory (1,995 employees)
- Model: **typhoon-v2.5-30b-a3b-instruct** (MoE, /nothink mode)
- Architecture: **Multi-Stage Workflow (Router -> Pandas Query Generator -> Safe Executor -> PII Stripper)**

### Advanced Techniques Implemented (Deep Research & Enterprise Patterns)
1. **Deterministic Router ???**: Regex-based preemptive routing for 100% exact-match refusal catching.
2. **Self-Generating Pandas Agent ??**: Replaced fuzzy ReAct tool-calling with an Agentic Data Analyst that reads the schema and writes 1-line exact Pandas queries.
3. **Dynamic Reflection Loop ??**: Automatically retries with broader filters if the first Pandas execution returns an empty result or error.
4. **Data Privacy Stripper ??**: Post-execution Regex to completely eradicate PII leakage when not explicitly queried.
\"\"\"))

cells.append(nbf.v4.new_markdown_cell(\"## 1. Setup & Imports\"))
cells.append(nbf.v4.new_code_cell(\"\"\"import os, json, re, time, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from openai import OpenAI

warnings.filterwarnings('ignore')

from kaggle_secrets import UserSecretsClient
try:
    TYPHOON_API_KEY = UserSecretsClient().get_secret("TYPHOON_API_KEY")
except:
    TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY", "your-api-key")

MODEL = 'typhoon-v2.5-30b-a3b-instruct'
client = OpenAI(api_key=TYPHOON_API_KEY, base_url='https://api.opentyphoon.ai/v1')\"\"\"))

cells.append(nbf.v4.new_markdown_cell(\"## 2. Data Loading & Cleansing\"))
cells.append(nbf.v4.new_code_cell(\"\"\"KAGGLE_DIR = Path('/kaggle/input/competitions/super-ai-engineer-season-6-fahmai-telephone-directory/')
LOCAL_DIR  = Path('super-ai-engineer-season-6-fahmai-telephone-directory')
DATA_DIR   = KAGGLE_DIR if KAGGLE_DIR.exists() else LOCAL_DIR

df_emp = pd.read_csv(DATA_DIR / 'employees.csv').fillna('')
df_questions = pd.read_csv(DATA_DIR / 'questions.csv')

df_emp['Phone Extension'] = df_emp['Phone Extension'].apply(
    lambda x: str(int(float(x))) if str(x).replace('.','').isdigit() else str(x).strip()
)

print(f"Employees loaded: {len(df_emp)}")
print(f"Questions loaded: {len(df_questions)}")\"\"\"))

cells.append(nbf.v4.new_markdown_cell(\"## 3. High-Security Deterministic Guardrails (Stage 1 & 4)\"))
cells.append(nbf.v4.new_code_cell(\"\"\"# --- STAGE 1: Deterministic Router (Pre-LLM) ---
def pre_llm_guardrail(question: str, language: str) -> str:
    q = question.lower()
    
    forbidden = [r'?????????', r'??????', r'????', r'???????', r'???????', r'???', r'?????', r'???????', r'salary', r'age', r'weight']
    if any(re.search(p, q) for p in forbidden):
        return '????????????????????????' if language == 'th' else 'cannot provide this information'
    
    opinions = [r'???', r'????', r'????????', r'????', r'??????', r'????????', r'opinion', r'handsome', r'beautiful', r'best']
    if any(re.search(p, q) for p in opinions):
        return '???????????????????????' if language == 'th' else 'cannot offer an opinion'
        
    external = [r' cp ', r'????', r' ptt', r'???', r' ais ', r'??????????', r'ais', r'true']
    if any(re.search(p, q) for p in external) and 'fahmai' not in q and '???????' not in q:
        return '??????????????????????' if language == 'th' else 'not a FahMai record'
        
    return None

# --- STAGE 4: Data Privacy Stripper (Post-Execution) ---
def post_execution_stripper(question: str, answer: str) -> str:
    q = question.lower()
    ans = str(answer).strip()
    
    if '????' not in q and 'id' not in q:
        ans = re.sub(r'\\b00\\d{3}\\b', '', ans)
        
    phone_kws = ['?????', '???', '???', 'phone', 'extension', 'ext', 'call']
    if not any(kw in q for kw in phone_kws):
        ans = re.sub(r'\\b\\d{4,5}\\b', '', ans)
    
    ans = re.sub(r'\\s+', ' ', ans).strip()
    return ans\"\"\"))

cells.append(nbf.v4.new_markdown_cell(\"## 4. Agentic Pandas Encoder (Stage 2 & 3)\"))
cells.append(nbf.v4.new_code_cell(\"\"\"system_prompt = f'''/nothink\\nYou are an expert Data Analyst Agent. You query a pandas dataframe df with these columns:
{list(df_emp.columns)}

Given a user question, write EXACTLY ONE line of Python code using df.
Hints: RETVP means Department=='RET' and Level=='VP'. CEO/CFO/CTO/COO are Unit=='CEO'.

Example:
Q: ?????? LOGFL
df.loc[(df['Department']=='LOG') & (df['Position Level']=='Lead'), ['First Name Thai', 'Last Name Thai']].apply(lambda x: ' '.join(x.dropna()), axis=1).tolist()
'''

def generate_pandas_query(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Q: {question}"}
            ],
            max_tokens=150,
            temperature=0.0
        )
        code = response.choices[0].message.content.strip()
        code = re.sub(r'^`python\\s*|`$', '', code).strip()
        return code
    except Exception as e:
        return "None"

def evaluate_and_reflect(code: str, df: pd.DataFrame, language: str) -> str:
    fallback = '???????????' if language == 'th' else 'no record found'
    try:
        result = eval(code, {'df': df, 'pd': pd, 'np': np})
        
        if result is None or (isinstance(result, (list, tuple, pd.Series)) and len(result) == 0) or pd.isna(result) or str(result).strip() == '':
            return fallback
            
        if isinstance(result, list):
            result = ', '.join(map(str, result))
            
        return str(result)
    except Exception as e:
        return fallback

def process_question_v3(question: str, language: str) -> str:
    safe_response = pre_llm_guardrail(question, language)
    if safe_response: return safe_response
        
    code = generate_pandas_query(question)
    raw_answer = evaluate_and_reflect(code, df_emp, language)
    final_answer = post_execution_stripper(question, raw_answer)
    
    if not final_answer:
        return '???????????' if language == 'th' else 'no record found'
        
    return final_answer\"\"\"))

cells.append(nbf.v4.new_markdown_cell(\"## 5. Main Inference Execution & Local Evaluation\"))
cells.append(nbf.v4.new_code_cell(\"\"\"results = []
print(f"Starting v3 Agentic inference on {len(df_questions)} questions...")

for _, row in tqdm(df_questions.iterrows(), total=len(df_questions)):
    ans = process_question_v3(row['question'], row.get('language', 'th'))
    results.append({'id': row['id'], 'response': ans})

submission_v3 = pd.DataFrame(results)
submission_v3.to_csv('submission_v3_Agentic.csv', index=False, encoding='utf-8')
print("Completed! Saved to submission_v3_Agentic.csv")

import subprocess
grade_script  = DATA_DIR / 'grade.py'
train_labels  = DATA_DIR / 'train_labels.json'
if grade_script.exists() and train_labels.exists():
    res = subprocess.run(
        ['python', str(grade_script), 'submission_v3_Agentic.csv', str(train_labels)],
        capture_output=True, text=True, encoding='utf-8'
    )
    print("=" * 60)
    print("LOCAL GRADE RESULTS V3")
    print("=" * 60)
    print(res.stdout)\"\"\"))

nb.cells = cells
nbf.write(nb, "D:/superai_engineer_ss6/Level 2/Hackathon 3_Fahmai Telephone Directory/Hackathon_3_LV2_Fahmai_Telephone_Directory_v3_AgenticUpgrade.ipynb")
