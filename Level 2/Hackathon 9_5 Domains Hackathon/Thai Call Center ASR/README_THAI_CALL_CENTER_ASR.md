# Thai Call Center ASR Runbook

Notebook หลัก:

- `5-domains-hackathon-thai-call-center-asr.ipynb`

Dataset path ที่รองรับ:

- Kaggle: `/kaggle/input/competitions/individual-test-thai-call-center-asr`
- Local: `Level 2/Hackathon 9_5 Domains Hackathon/Thai Call Center ASR/individual-test-thai-call-center-asr`

## วิธีรันบน Kaggle

1. เปิด GPU runtime
2. ถ้า Kaggle ยังไม่มี dependency ให้เปิด install cell ด้านบนของ notebook
3. ถ้ามี internet/model download ได้ ให้ใช้ default model ได้เลย
4. ถ้าไม่มี internet ให้ Add model weights เป็น Kaggle Dataset/Model แล้วตั้ง `ASR_MODEL_DIR` ให้ชี้ไปยัง folder นั้น
5. รัน notebook จากบนลงล่าง
6. ไฟล์ส่งจริงจะออกที่ `/kaggle/working/submission.csv`

## Pipeline ที่ใช้

- อ่าน `sample_submission.csv` เป็นแกนหลัก
- หา audio path อัตโนมัติจากทุก `.wav` ใต้ dataset root
- แยก `base_key` จากชื่อไฟล์ โดยตัด suffix `_phone`, `_noise`, `_fast`, `_slow`, `_pitch`
- เลือกเสียงตัวแทนต่อหนึ่ง base utterance โดยชอบ original ก่อน
- ถอดเสียงเฉพาะ representative audio แล้ว copy transcript กลับไปทุก variant ในกลุ่มเดียวกัน
- post-process transcript แบบเบา ๆ เพื่อเอา punctuation/special tokens ออก แต่ไม่ลบคำพูดจริง เช่น `ครับ`, `ค่ะ`, `อืม`, `เอ่อ`
- cache prediction ที่ `asr_representative_predictions.csv` เพื่อ resume ได้
- สร้าง submission ตาม column จริงของ sample submission
- run safety checks ก่อนจบ

## Smoke Test Local

ถ้าต้องการตรวจ format โดยไม่โหลด ASR model:

```powershell
$env:RUN_ASR="0"
```

แล้ว execute notebook จะได้ `submission_format_check.csv` สำหรับตรวจ row/column/order เท่านั้น ไม่ใช่ไฟล์แข่งขันจริง

## Model Notes

ค่า default ใน notebook:

- `biodatlab/distill-whisper-th-large-v3`

fallback:

- `openai/whisper-large-v3-turbo`

สามารถเปลี่ยนได้ผ่าน `ASR_MODEL_NAME`, `ASR_FALLBACK_MODEL_NAME`, หรือ `ASR_MODEL_DIR`

## Next Experiments

- ลอง Thai Whisper checkpoint หลายตัวแล้วเทียบ public score
- รัน second model เฉพาะ representative audio ที่ transcript สั้นผิดปกติ
- เพิ่ม domain correction จาก error analysis เช่น banking terms
- ปรับ batch size ตาม GPU memory
- ใช้ cache ทุกครั้งเพื่อลดเวลารันซ้ำ
