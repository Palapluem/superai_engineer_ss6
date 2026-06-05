import os import glob import pandas as pd

Lexical retrieval with BM25

from rank_bm25 import BM25Okapi

Dense retrieval with SentenceTransformers

from sentence_transformers import SentenceTransformer import numpy as np from sklearn.metrics.pairwise import cosine_similarity

Parameters

CONTEXT_DIR = "./context/"  # path to folder of .txt files TEST_FILE = "./test.csv"    # path to test.csv TOP_K = 3                     # number of contexts to retrieve per question

1. Load contexts

contexts = {} doc_texts = [] doc_names = [] for path in glob.glob(os.path.join(CONTEXT_DIR, "*.txt")): name = os.path.basename(path) with open(path, "r", encoding="utf-8") as f: text = f.read() contexts[name] = text doc_texts.append(text) doc_names.append(name)

2a. Build BM25 index

Tokenize by whitespace or Thai tokenizer if needed

tokenized_corpus = [doc.split() for doc in doc_texts] bm25 = BM25Okapi(tokenized_corpus)

2b. Build dense embeddings

model = SentenceTransformer('intfloat/multilingual-e5-base') doc_embeddings = model.encode(doc_texts, convert_to_numpy=True)

3. Read test questions

test_df = pd.read_csv(TEST_FILE)

4. Retrieval functions

def retrieve_bm25(question, top_k=TOP_K): tokens = question.split() scores = bm25.get_scores(tokens) top_indices = np.argsort(scores)[::-1][:top_k] return [(doc_names[i], contexts[doc_names[i]], scores[i]) for i in top_indices]

def retrieve_dense(question, top_k=TOP_K): q_emb = model.encode([question], convert_to_numpy=True) sims = cosine_similarity(q_emb, doc_embeddings)[0] top_indices = np.argsort(sims)[::-1][:top_k] return [(doc_names[i], contexts[doc_names[i]], float(sims[i])) for i in top_indices]

5. Inference loop: get top contexts for each question

results = [] for idx, row in test_df.iterrows(): q = row['question'] # Choose either BM25 or Dense candidates = retrieve_dense(q) # Format for filename, context, score in candidates: results.append({ 'question': q, 'context_file': filename, 'score': score, 'context_snippet': context[:200].replace("\n", " ") + '...' })

6. Convert to DataFrame

output_df = pd.DataFrame(results)

Keep top_k per question

output_df = output_df.groupby('question').head(TOP_K)

7. Save

output_df.to_csv("retrieved_contexts.csv", index=False, encoding='utf-8-sig')

Usage:

- Run this script after placing your test.csv and context/ folder

- It produces 'retrieved_contexts.csv' with columns: question, context_file, score, snippet

Next: feed these (question, context_file, context_text) pairs into your QA model for answer extraction.
