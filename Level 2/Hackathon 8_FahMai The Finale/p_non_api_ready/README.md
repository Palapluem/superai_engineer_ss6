# Bank Statement OCR API Ready

## Evaluation API Contract

This folder is prepared for the FahMai OCR back-test API described in
`SS6-Week4_InferanceAPI_Eval-20260603_1915.pdf`.

Primary endpoint:

```text
POST /ocr
```

Expected JSON body:

```json
{
  "id": "BS-KBANK-OPER-2567-03",
  "header": "BASE64_ENCODED_PNG",
  "transaction": ["BASE64_ENCODED_PNG"]
}
```

`transactions` is also accepted as an alias for `transaction`. The API saves
the request as `{id}_header.png` and `{id}_transactions_pN.png`, runs the
FahMai PaddleOCR crop pipeline, and returns the Kaggle `pred_json` object:

```json
{
  "id": "BS-KBANK-OPER-2567-03",
  "answer": {
    "L0_account_id": "...",
    "L0_bank": "...",
    "L1_BT-..._business_event_date": "...",
    "L1_BT-..._amount_thb": "..."
  }
}
```

Runtime defaults for the RTX 5090/B200-style evaluation environment:

```text
OCR_WORKERS=1
OCR_QUEUE_MAXSIZE=128
OCR_JOB_TIMEOUT_SECONDS=900
OCR_DEVICE=gpu:0
CUDA_VISIBLE_DEVICES=0
```

Run:

```bash
bash run_api_b200.sh
```

or on Windows:

```bat
run_api.bat
```

## API Preflight / Log Check

After the server is running, use:

```powershell
python check_api_ready.py --url http://127.0.0.1:8000
```

This checks `/health`, OpenAPI routes, verifies that only `/ocr` is exposed,
sends an invalid-base64 probe that should return HTTP 422, tails server logs,
and lists recent persisted OCR runs.

To validate base64 encoding from local render files without calling OCR:

```powershell
python check_api_ready.py --sample-artifact BS-BBL-OPER-2567-01 --max-transactions 1 --skip-invalid-base64
```

To send a real OCR smoke request:

```powershell
python check_api_ready.py --sample-artifact BS-BBL-OPER-2567-01 --max-transactions 1 --send --persist --ocr-timeout 900
```

To smoke-test the ModelHarbor `/ocr` endpoint in one command:

```powershell
python test_ocr_endpoint_once.py
```

The default endpoint is `http://swarm-manager.modelharbor.com:57444/ocr` and
the default sample is `BS-BBL-OPER-2567-11`. The command fails if
`response.answer` is empty or all values are blank. Use `--min-answer-fields 0
--min-nonempty-fields 0` only when you want to check connectivity without
validating extraction output.

For deployment, keep these bundled files with `p_non_api_ready`:

```text
submission_template_OCR.csv
ocr_pipeline_runtime/
```

If the remote endpoint returns `{"answer": {}}` almost instantly, the exposed
port is still running an old/stub service or a copy without the bundled
template/runtime files. Restart the server from this folder after deploying the
full package.

โฟลเดอร์นี้เป็นโครงสำหรับเอา pipeline OCR ไปทำ API รับรูป statement แล้วคืนข้อมูล `headers`, `transactions`, `artifact_schema`, และ metadata

## Input Filename Contract

ระบบ route bank/layout จากชื่อไฟล์ ดังนั้นควรส่งชื่อไฟล์เดิม:

```text
BS-KBANK-OPER-2567-03_header.png
BS-KBANK-OPER-2567-03_transactions_p1.png
BS-OPER-PTY-CTRL-2567-11_header.png
BS-OPER-PTY-CTRL-2567-11_transactions_p1.png
```

รองรับไฟล์ `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

## ติดตั้ง

แนะนำให้ใช้ environment ที่ติดตั้ง PaddleOCR/PaddlePaddle แล้ว หากยังไม่มี:

```powershell
cd D:\sup_ai\level2_hack3_agentic\p_non_api_ready
pip install -r requirements.txt
```

ถ้าใช้ GPU ให้ติดตั้ง `paddlepaddle-gpu` ให้ตรง CUDA ของเครื่องแยกต่างหากตาม environment ที่ใช้อยู่

## รันแบบ Local Folder

```powershell
cd D:\sup_ai\level2_hack3_agentic\p_non_api_ready
python run_folder.py --input-dir D:\sup_ai\level2_hack3_agentic\10_files_test --output-dir D:\sup_ai\level2_hack3_agentic\p_non_api_ready\runs\test_10_files
```

Output:

```text
runs\test_10_files\transactions_canonical.csv
runs\test_10_files\headers.csv
runs\test_10_files\ocr_debug.csv
runs\test_10_files\artifact_schema_summary.csv
runs\test_10_files\by_bank\*.csv
runs\test_10_files\by_artifact\*.csv
runs\test_10_files\run_meta.json
```

## รัน API

```powershell
cd D:\sup_ai\level2_hack3_agentic\p_non_api_ready
uvicorn api:app --host 0.0.0.0 --port 8000
```

หรือ double click:

```text
run_api.bat
```

บน Linux/B200 server:

```bash
cd /path/to/p_non_api_ready
bash run_api_b200.sh
```

ค่า environment ที่ใช้คุม queue:

```text
OCR_WORKERS=1
OCR_QUEUE_MAXSIZE=128
OCR_JOB_TIMEOUT_SECONDS=900
CUDA_VISIBLE_DEVICES=0
PORT=8000
```

แนะนำเริ่มที่ `OCR_WORKERS=1` ก่อน เพราะ pipeline OCR ใช้ GPU/VRAM หนักและมี global config ภายใน pipeline เดิม หากวัดแล้ว VRAM ยังเหลือค่อยเพิ่มเป็น `2`

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

## ยิง API `/ocr`

Input หลักสำหรับ back-test เป็น JSON ที่มี `id`, `header`, และ `transactions`:

```json
{
  "id": "case_001",
  "header": {
    "content": "<base64>",
    "mime_type": "image/png"
  },
  "transactions": [
    {
      "content": "<base64>",
      "mime_type": "image/png"
    },
    {
      "content": "<base64>",
      "mime_type": "image/png"
    }
  ],
  "persist": true
}
```

`header` รับได้ 1 ภาพ/PDF และ `transactions` รับได้หลายภาพ/PDF เช่น 1-10 หน้า ระบบจะตั้งชื่อไฟล์ภายในเป็น `{id}_header.png` และ `{id}_transactions_pN.png` เพื่อให้ pipeline เดิมแยก header/transaction ได้

ยังรองรับ JSON แบบไฟล์เดี่ยวที่มี Image/PDF แบบ base64-encoded:

```json
{
  "filename": "BS-KBANK-OPER-2567-03_header.png",
  "content": "<base64>",
  "mime_type": "image/png",
  "persist": true
}
```

หรือส่งหลายไฟล์พร้อมกัน:

```json
{
  "persist": true,
  "files": [
    {
      "filename": "BS-KBANK-OPER-2567-03_header.png",
      "content": "<base64>",
      "mime_type": "image/png"
    },
    {
      "filename": "BS-KBANK-OPER-2567-03_transactions_p1.png",
      "content": "<base64>",
      "mime_type": "image/png"
    }
  ]
}
```

รองรับ field base64 ชื่อ `content`, `base64`, `data`, `image`, หรือ `pdf` และรองรับ data URL เช่น `data:image/png;base64,...`

```powershell
curl -X POST "http://127.0.0.1:8000/ocr" `
  -H "Content-Type: application/json" `
  -d "{\"filename\":\"BS-KBANK-OPER-2567-03_header.png\",\"content\":\"<base64>\",\"mime_type\":\"image/png\",\"persist\":true}"
```

Response จะเป็น JSON:

```json
{
  "id": "case_001",
  "answer": {
    "output_dir": "...",
    "transactions": [],
    "headers": [],
    "artifact_schema": [],
    "meta": {},
    "debug_count": 0,
    "request_id": "case_001",
    "queue_wait_seconds": 0.0,
    "processing_seconds": 0.0,
    "worker_id": 1
  }
}
```

ถ้า `persist=true` ไฟล์ CSV/debug จะถูกเขียนไว้ที่:

```text
D:\sup_ai\level2_hack3_agentic\p_non_api_ready\runs\{id}_{run_uuid}
```

ถ้าส่ง PDF (`mime_type = application/pdf` หรือ filename ลงท้าย `.pdf`) server จะแปลงแต่ละหน้าเป็น PNG ก่อนส่งเข้า pipeline OCR เดิม โดยต้องติดตั้ง `PyMuPDF` จาก `requirements.txt`

## Queue Behavior

- `POST /ocr` จะ enqueue งานก่อน แล้วรอผล OCR กลับใน request เดิม
- ถ้ามี 10 requests เข้ามาทุก 5 วินาที server จะค่อย ๆ ประมวลผลตามจำนวน `OCR_WORKERS`
- ถ้าคิวเต็มจะตอบ `429` พร้อม code `ocr_queue_full`
- ถ้างานเกิน timeout จะตอบ `504` พร้อม code `ocr_timeout`
- ไม่ควรใช้หลาย process ของ uvicorn (`--workers > 1`) กับ GPU pipeline นี้ เพราะแต่ละ process จะโหลด OCR/GPU memory แยกกันและทำให้ VRAM พุ่ง

## Output Fields

`transactions_canonical.csv`:

```text
artifact_id
bank
family
account_id
account_number
account_role
currency
page
row_index
business_event_date
effective_time
transaction_type
direction
amount_thb
balance_after_thb
channel
description
date_raw
amount_raw
balance_raw
description_raw
parse_status
```

`headers.csv`:

```text
artifact_id
bank
family
account_id
account_number
account_number_raw
account_number_score
account_role
currency
```

## Notes

- `description` ของ KBANK ถูกตัด prefix `k plus` ออกแล้ว
- `bbl_compact` ใช้ `description = k plus` ตาม rule ล่าสุด
- `bbl_sparse` และ `scb_sparse` filter เหลือ transaction row หลัก และคำนวณ amount จาก opening/ending balance
- pipeline นี้ยังเป็น extraction API ไม่ใช่ตัว build `submission.csv` เต็ม เพราะการ map `BT-...` slot ต้องใช้ schema/template เพิ่ม
