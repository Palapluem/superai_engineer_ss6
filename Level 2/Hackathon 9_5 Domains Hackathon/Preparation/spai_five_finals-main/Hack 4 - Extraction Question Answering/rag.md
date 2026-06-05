RAG Pipeline using Qwen Embedding + LLM (Open-Domain QA over Thai Text)

1. Install dependencies

pip install transformers sentence-transformers faiss-cpu pandas

2. Load and split context documents into chunks

import os
import glob
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

# Load context
CONTEXT_DIR = "./context"
context_chunks = []
context_ids = []
chunk_size = 300  # characters
stride = 100

for file_path in glob.glob(os.path.join(CONTEXT_DIR, '*.txt')):
    filename = os.path.basename(file_path)
    with open(file_path, encoding='utf-8') as f:
        text = f.read()
    # Chunking
    for i in range(0, len(text), stride):
        chunk = text[i:i+chunk_size]
        if len(chunk) < 50:
            continue
        context_chunks.append(chunk)
        context_ids.append(f"{filename}__{i}")

3. Embed chunks with Qwen Embedding (0.6B)

from transformers import AutoTokenizer, AutoModel

model_name = "Qwen/Qwen-embed-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

@torch.no_grad()
def embed_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
    return embeddings.cpu().numpy()

# Create embeddings
all_embeddings = []
batch_size = 16
for i in range(0, len(context_chunks), batch_size):
    batch = context_chunks[i:i+batch_size]
    embs = embed_texts(batch)
    all_embeddings.append(embs)

all_embeddings = np.vstack(all_embeddings)

4. Build FAISS index

index = faiss.IndexFlatL2(all_embeddings.shape[1])
index.add(all_embeddings)

5. Define RAG Inference Function

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a chat model like Qwen1.5-1.8B or Qwen1.5-4B
rag_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto")
rag_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")


def answer_question_rag(question, top_k=3):
    # Embed question
    q_emb = embed_texts([question])
    D, I = index.search(q_emb, top_k)
    relevant_chunks = [context_chunks[i] for i in I[0]]

    context_concat = "\n---\n".join(relevant_chunks)
    prompt = f"<|user|>\n{question}\n<|context|>\n{context_concat}\n<|assistant|>"

    inputs = rag_tokenizer(prompt, return_tensors="pt").to(rag_model.device)
    outputs = rag_model.generate(**inputs, max_new_tokens=128)
    response = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

6. Run on test.csv

test_df = pd.read_csv("test.csv")
answers = test_df['question'].apply(answer_question_rag)
test_df['answer'] = answers
test_df.to_csv("rag_answers.csv", index=False)


---

✅ Summary:

Chunks your .txt context files

Embeds them with Qwen-embed-0.6B

Builds a FAISS index for fast retrieval

Answers questions with Qwen-Chat using RAG-style prompt


Let me know if you want to stream responses, speed it up with GPU batching, or use Qwen3!

