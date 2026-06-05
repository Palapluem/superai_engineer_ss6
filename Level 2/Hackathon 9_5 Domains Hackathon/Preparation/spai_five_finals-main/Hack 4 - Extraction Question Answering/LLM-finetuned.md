Thai Extractive QA with Qwen3 + Unsloth Fine-tuning

1. Install dependencies (run in Colab or local)

pip install unsloth transformers datasets peft bitsandbytes accelerate

2. Load and preprocess the training data

import pandas as pd
from datasets import Dataset

train_df = pd.read_csv("train.csv")

# Load context files
import os
CONTEXT_PATH = "./context"
context_map = {fn: open(os.path.join(CONTEXT_PATH, fn), encoding="utf-8").read() for fn in os.listdir(CONTEXT_PATH)}

# Prepare data for SFT (prompt format)
def create_prompt(row):
    context = context_map[row['file']]
    return {
        "prompt": f"<|user|>\n{row['question']}\n<|context|>\n{context}\n<|end|>\n",
        "response": row['answer']
    }

prompt_data = train_df.apply(create_prompt, axis=1)
dataset = Dataset.from_pandas(pd.DataFrame(prompt_data.tolist()))

3. Load Qwen3-4B with Unsloth

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen1.5-4B-Chat",
    max_seq_length = 2048,
    dtype = "auto",
    load_in_4bit = True
)

4. Tokenize with prompt template

def formatting_func(example):
    return [
        f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"
    ]

tokenized = tokenizer.apply_chat_template(
    dataset,
    tokenize=True,
    add_generation_prompt=False,
    function_to_apply=formatting_func,
    return_tensors="pt"
)

5. Fine-tune with PEFT (LoRA)

from unsloth import get_peft_model
from transformers import TrainingArguments, Trainer

model = get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0.05)

training_args = TrainingArguments(
    output_dir="./qwen-qa-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
)

trainer.train()

6. Inference on test.csv

test_df = pd.read_csv("test.csv")

# Load and concatenate context (or use retriever to pick one)
retrieved_df = pd.read_csv("retrieved_contexts.csv")  # from previous step

# Group by question, pick best context
test_pairs = retrieved_df.groupby("question").first().reset_index()

def generate_answer(prompt):
    input_text = f"<|user|>\n{prompt['question']}\n<|context|>\n{context_map[prompt['context_file']]}\n<|end|>\n<|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

answers = test_pairs.apply(generate_answer, axis=1)
test_pairs['answer'] = answers
test_pairs.to_csv("qa_predictions.csv", index=False)


---

🔧 Summary:

Fine-tunes Qwen1.5-4B using Unsloth + LoRA

Uses <|user|>...<|context|>...<|assistant|> prompt template

Retrieves context using your previous pipeline

Outputs answers into qa_predictions.csv


Let me know if you want to switch to a smaller model like Phi-3 or use a single file training format.

