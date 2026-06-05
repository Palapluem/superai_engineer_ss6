You are a question-answering assistant. Your task is to extract a short, precise span from the given context that answers the user's question. If no answer is found, reply "No answer in the context."

### Question:
สูตรการคำนวณอัตราผลตอบแทนรวมคืออะไร

### Context:
ราคาต้นงวด = 10 บาท
ราคาปลายงวด = 12 บาท
เงินปันผล = 1 บาท
อัตราผลตอบแทนรวม = (เงินปันผล + ราคาปลายงวด - ราคาต้นงวด) / ราคาต้นงวด

### Extracted Answer:


Model Ideas for Extractive LFQA
💡 Option 1: Fine-Tuned Transformers (Span Prediction)
Use pre-trained models like:
| Model | Highlights | Notes | 
| deepset/roberta-base-squad2 | Handles long-context & impossible questions | Well-suited for Thai-English mixed contexts if fine-tuned | 
| ThaiQA-BERT | Trained on Thai QA datasets | Great for native semantic structure | 
| LongformerQA | Supports longer documents natively | Use if context > 512 tokens | 


Key Benefit: These models predict start and end tokens from context, perfect for precise extraction.


from transformers import pipeline

# Load extractive QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Example input
context = """
ราคาต้นงวด = 10 บาท
ราคาปลายงวด = 12 บาท
เงินปันผล = 1 บาท
อัตราผลตอบแทนรวม = (เงินปันผล + ราคาปลายงวด - ราคาต้นงวด) / ราคาต้นงวด
"""

question = "สูตรการคำนวณอัตราผลตอบแทนรวมคืออะไร"

# Get answer
result = qa_pipeline(question=question, context=context)

print(f"Answer: {result['answer']}")

