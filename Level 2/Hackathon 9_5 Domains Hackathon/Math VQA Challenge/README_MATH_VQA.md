# Thai Math VQA Challenge Runbook

Notebook หลัก:

- `5-domains-hackathon-math-vqa-challenge.ipynb`

Dataset path ที่รองรับ:

- Kaggle: `/kaggle/input/competitions/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen`
- Local: `Level 2/Hackathon 9_5 Domains Hackathon/Math VQA Challenge/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen`

## วิธีรันเร็ว

1. เปิด notebook แล้วรันตั้งแต่ต้นจนจบ
2. ค่า default คือ `RUN_VLM_INFERENCE = False` เพื่อสร้าง baseline submission ที่ submit/check format ได้ทันที
3. output หลักจะอยู่ที่ `submission.csv`
4. baseline แยกเก็บที่ `submission_baseline_most_common.csv`

## วิธีเปิดใช้ Open-Weights VLM

1. เพิ่ม model weights แบบ open weights เข้า Kaggle Notebook ผ่าน Add Model/Add Dataset หรือใช้ model ที่ดาวน์โหลดไว้เอง
2. ตั้ง `MODEL_DIR` หรือ environment variable `VLM_MODEL_DIR` ให้ชี้ไปยัง folder ของ model
3. เปิด `RUN_VLM_INFERENCE = True`
4. รัน validation subset ก่อนเสมอ แล้วค่อยรัน test ครบ 420 รูป

ห้ามใช้ hosted/commercial inference API เพื่อผลิตคำตอบ เช่น OpenAI, Claude, Gemini, Bedrock, Azure OpenAI, OpenRouter, Together, Replicate, Groq, Hugging Face Inference API หรือบริการเทียบเท่า

## สิ่งที่ notebook ทำให้แล้ว

- load `train.csv`, `test.csv`, `sample_submission.csv`
- resolve image path ทั้งแบบ Kaggle `images/{id}.jpg` และ local `images/images/{id}.jpg`
- preview image และตรวจ image size
- implement answer normalizer สำหรับ local validation
- สร้าง most-common baseline
- เตรียม prompt/post-processing สำหรับ Math VQA
- เตรียม VLM inference function พร้อม cache/resume
- สร้าง submission ตาม format `id,answer`
- run submit safety checks ก่อนจบ

## Next Experiments

- ลอง Qwen2-VL/Qwen2.5-VL, InternVL หรือ MiniCPM-V ที่รันด้วย weights local
- เพิ่ม self-consistency แล้ว majority vote จาก normalized answer
- เพิ่ม local OCR เช่น PaddleOCR/EasyOCR/Tesseract Thai แล้วใช้ open-weights reasoning LLM ช่วยแก้โจทย์ text-heavy
- ทำ error analysis แยกกลุ่มโจทย์เรขาคณิต เศษส่วน รากที่สอง choice และหน่วย
