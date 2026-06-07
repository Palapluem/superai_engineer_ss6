# Codex CLI Prompt Templates For 5 Domains Hackathon

ไฟล์นี้เป็น prompt bank สำหรับใช้กับ Codex CLI ตอนเจอโจทย์จริง 5 โดเมน: Data Science, Computer Vision, IoT, Signal Processing, และ Natural Language Processing

ใช้หลักนี้เสมอ:

- แนบ resource ด้วย path/link ให้ชัดเจน
- บอก runtime ว่าจะใช้ Kaggle, Colab, หรือ local
- บอก output path ที่ต้องการให้ Codex สร้างไฟล์
- ให้ Codex อิงโจทย์เก่าใน `Preparation` โดยเน้น SS5 ก่อน แล้วค่อยใช้ SS4 เป็น reference รอง
- ให้ Codex ทำ baseline ที่ submit ได้ก่อน จากนั้นค่อยเพิ่ม improvement

## เปิด Codex CLI

จาก root repo:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6"
codex -m gpt-5.5 -c model_reasoning_effort="high" --sandbox workspace-write --ask-for-approval on-request
```

ถ้างานหนักมาก เช่น OCR/CV pipeline, time-series leakage, ensemble หลายโมเดล:

```powershell
codex -m gpt-5.5 -c model_reasoning_effort="xhigh" --sandbox workspace-write --ask-for-approval on-request
```

ถ้าจะรันแบบสั่งครั้งเดียว:

```powershell
codex exec --sandbox workspace-write --cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6" "PASTE_PROMPT_HERE"
```

## Resource Block ที่ให้เติมทุกครั้ง

ใช้ block นี้เป็นหัว prompt ทุกโจทย์:

```text
บริบทการแข่งขัน:
- งาน: Super AI Engineer Season 6 Level 2 - 5 Domains Hackathon
- Domain: [Data Science / Computer Vision / IoT / Signal Processing / NLP]
- Runtime เป้าหมาย: [Kaggle / Colab / Local]
- Dataset link/path: [ใส่ link หรือ path dataset]
- Statement/path โจทย์: [ใส่รายละเอียดโจทย์ หรือ path ไฟล์โจทย์]
- Problem instruction paste zone:
  """
  [แปะ instruction/statement ของโจทย์จริงทั้งหมดตรงนี้]
  [เช่น background, task, input files, output format, metric, rules, constraints, leaderboard note]
  [ถ้ามีหลายหน้า/หลาย section ให้แปะให้ครบ อย่าสรุปเองก่อน]
  """
- Output notebook path: [เช่น Level 2/Hackathon 9_5 Domains Hackathon/workspace/<domain>/<name>.ipynb]
- Output submission path: [เช่น .../submission.csv หรือ /kaggle/working/submission.csv]
- Metric ที่ใช้วัดผล: [accuracy / F1 / RMSE / MAE / mAP / BLEU / WER / custom]
- Submission format: [sample_submission.csv path หรือรายละเอียด column]
- Time budget: [เช่น 2 ชั่วโมง / 4 ชั่วโมง / ต้องได้ baseline ใน 30 นาที]
- ข้อจำกัด: [ห้ามใช้ internet / ใช้ GPU ได้ / memory limit / ต้องรันใน Kaggle]
- Reference เก่า:
  - ใช้ SS5 เป็นหลักจาก `Level 2/Hackathon 9_5 Domains Hackathon/Preparation`
  - ใช้ SS4 เป็น reference รองถ้าหา pattern จาก SS5 ไม่เจอ
- Notebook style reference:
  - ให้รูปแบบ Markdown, ลำดับหัวข้อ, และโครง code cell ใกล้เคียงกับ notebook ใน `Level 1/Hackathon 4_5 Domains Hackathon`
  - ใช้ style แบบ Level 1: title cell -> `# 1. Setup & Imports` -> `# 2. Data Loading & Initial Inspection` -> preprocessing/feature/model -> evaluation -> `# 6. Prediction & Submission Generation`
  - Markdown อธิบายเป็นภาษาไทยกระชับเหมือน notebook Level 1 และ code แยก cell เป็นขั้นตอน ไม่ยัดทุกอย่างใน cell เดียว

สิ่งที่ต้องการ:
1. อ่านโจทย์และ dataset/sample submission
2. หาโจทย์เก่าที่ใกล้เคียงที่สุดใน Preparation โดยเน้น SS5
3. สร้าง notebook/script ที่รันได้จริงตาม runtime เป้าหมาย
4. ทำ baseline ที่ submit ได้ก่อน
5. เพิ่ม validation ที่กัน data leakage
6. สร้าง submission file ตาม format
7. เขียน README สั้น ๆ อธิบายวิธีรันและสิ่งที่ควรลองต่อ
8. ก่อนจบให้ใช้ $scrutinize ตรวจความเสี่ยงก่อน submit
```

## Master Prompt: เริ่มโจทย์ใหม่

ใช้เมื่อเพิ่งได้โจทย์ใหม่และยังไม่รู้จะเริ่มยังไง:

```text
[วาง Resource Block ตรงนี้]

ช่วยทำงานแบบ end-to-end สำหรับโจทย์นี้:

Phase 1 - Understand
- อ่านโจทย์, dataset structure, sample submission, metric
- สรุปว่า task คืออะไร, input/output คืออะไร, leakage risk อยู่ตรงไหน
- หา reference เก่าที่ใกล้ที่สุดจาก `Level 2/Hackathon 9_5 Domains Hackathon/Preparation`
- ระบุว่า reference มาจาก SS5 หรือ SS4 และเอา pattern อะไรมาใช้ได้

Phase 2 - Build Baseline
- สร้าง notebook ตาม output path ที่ระบุ
- Notebook ต้องมี markdown อธิบายแต่ละขั้นสั้น ๆ
- ต้องมี cell สำหรับ setup/import, load data, EDA quick check, preprocess, train, validate, inference, save submission
- ถ้าเป็น Kaggle ให้ใช้ path แบบ `/kaggle/input/...` และ save ที่ `/kaggle/working/submission.csv`
- ถ้าเป็น Colab ให้มี cell mount Drive/ตั้ง path แบบแก้ได้ง่าย
- ถ้าเป็น local ให้ใช้ relative path จาก repo

Phase 3 - Improve
- เพิ่ม improvement ที่คุ้มเวลา 2-3 อย่าง
- อย่าเพิ่ม dependency หนักถ้าไม่จำเป็น
- ทำให้ reproducible ด้วย seed
- จัด logging/print metric ให้ดูง่าย

Phase 4 - Submit Safety
- ตรวจ column/order/type ของ submission
- เช็กจำนวนแถวเท่ากับ sample_submission
- ใช้ $scrutinize review notebook/submission pipeline ก่อนจบ

Done when:
- มี notebook ที่เปิดได้จริง
- มี submission.csv ที่ตรง format
- มี summary ว่ารัน cell ไหนก่อนหลัง
- มี next experiments 3 ข้อถ้ามีเวลาต่อ
```

## Prompt: สแกนโจทย์เก่าเพื่อหา Reference

ใช้ก่อนเขียนโมเดล ถ้าอยากให้ Codex หา pattern จาก SS5/SS4:

```text
ช่วยหาโจทย์เก่าที่คล้ายกับโจทย์นี้จากโฟลเดอร์:
`Level 2/Hackathon 9_5 Domains Hackathon/Preparation`

โจทย์ใหม่:
- Domain: [ใส่ domain]
- Task: [สรุปโจทย์]
- Dataset columns/files: [ใส่คร่าว ๆ]
- Metric: [ใส่ metric]

ให้ทำ:
1. ค้น reference จาก SS5 ก่อน
2. ถ้า SS5 ไม่พอ ค่อยดู SS4
3. สรุปไฟล์ notebook/README/dataset ที่เกี่ยวข้องพร้อม path
4. บอก pattern ที่เอามาใช้ได้ เช่น preprocessing, split, model, metric, submission
5. บอกสิ่งที่ไม่ควร copy ตรง ๆ เพราะอาจผิดโจทย์หรือเสี่ยง leakage
```

## Data Science Prompt

ใช้กับ tabular classification/regression, structured data, business data:

```text
[วาง Resource Block ตรงนี้]

นี่คือโจทย์ Data Science / Tabular ML

ช่วยสร้าง Kaggle/Colab-ready notebook โดยเน้น:
- อ่าน train/test/sample_submission
- ตรวจ target, missing values, categorical/numerical columns, duplicate rows
- ตรวจ metric และ direction ว่าต้อง maximize/minimize
- ทำ validation split ที่เหมาะสม และระวัง leakage
- ถ้ามี group/time/id ให้พิจารณา GroupKFold หรือ temporal split แทน random split
- สร้าง baseline เร็ว:
  - classification: LogisticRegression/RandomForest/LightGBM/CatBoost ถ้ามี
  - regression: Ridge/RandomForest/LightGBM/CatBoost ถ้ามี
- preprocess:
  - numeric impute + scale ถ้าจำเป็น
  - categorical encode อย่างปลอดภัย
  - handle class imbalance ถ้า metric เป็น F1/recall
- train/validate/inference
- สร้าง submission.csv ตรง sample_submission

ให้ทำไฟล์:
- notebook: [ใส่ output notebook path]
- submission: [ใส่ output submission path]
- README: [ใส่ output README path]

ก่อนจบให้รายงาน:
- best validation score
- features ที่ใช้
- leakage risks
- next experiments ที่คุ้มเวลาที่สุด 3 อย่าง
```

## Computer Vision Prompt

ใช้กับ image classification, object detection, segmentation, OCR, image retrieval:

```text
[วาง Resource Block ตรงนี้]

นี่คือโจทย์ Computer Vision

ช่วยสร้าง notebook ที่รันได้จริง โดยเริ่มจากตรวจ dataset structure:
- train image path, labels, test image path, sample_submission
- task type: classification / detection / segmentation / OCR / retrieval
- image size, class distribution, corrupted images
- metric: accuracy/F1/mAP/Dice/IoU/custom

ให้หา reference เก่าจาก Preparation โดยเน้น SS5 ก่อน โดยเฉพาะ notebook CV ที่ใช้ pretrained model

Baseline ที่ต้องมี:
- reproducible seed
- dataset/dataloader หรือ simple generator
- train/valid split ที่ไม่ leak
- augmentation เบา ๆ
- pretrained model ถ้าเหมาะสม เช่น timm/torchvision/ultralytics
- training loop หรือ framework ที่ง่ายสุดและรันทัน
- inference test set
- create submission.csv

ถ้าเป็น Kaggle:
- ใช้ `/kaggle/input/...`
- save model/checkpoint เท่าที่จำเป็นใน `/kaggle/working`
- save submission ที่ `/kaggle/working/submission.csv`

ถ้าเป็น Colab:
- มี cell mount Google Drive
- มี config cell รวม path ทั้งหมดไว้บนสุด

ก่อนจบ:
- preview batch image + labels
- print validation metric
- ตรวจ submission shape/order/type
- ใช้ $scrutinize ตรวจ pipeline ก่อน submit
```

## IoT Prompt

ใช้กับ sensor data, wearable, device logs, tabular time-series, anomaly detection:

```text
[วาง Resource Block ตรงนี้]

นี่คือโจทย์ IoT / Sensor ML

ช่วยสร้าง notebook โดยเน้น:
- อ่านข้อมูล sensor/time-series/device/session/user
- ตรวจ sampling rate, timestamp, missing time, duplicated timestamp
- ตรวจว่า split ต้องกัน leakage ตาม user/device/session หรือ time หรือไม่
- สร้าง features:
  - raw statistical features: mean/std/min/max/median/IQR
  - rolling/window features
  - frequency features ถ้ามีสัญญาณ periodic
  - lag/diff features
- baseline:
  - tabular features + LightGBM/CatBoost/RandomForest
  - ถ้า sequence ชัดเจน ค่อยเพิ่ม 1D CNN/LSTM แบบง่าย
- validation:
  - GroupKFold ตาม subject/device/session ถ้ามี
  - temporal split ถ้าเป็น forecasting
- inference และ submission

ให้หา reference เก่า SS5 IoT/Data Science ใน Preparation ก่อน แล้วสรุปว่าเอา feature engineering หรือ split strategy อะไรมาใช้ได้

ก่อนจบให้เช็ก:
- target leakage จาก timestamp/id/future data
- train/test distribution shift
- submission format
- next experiments 3 อย่าง
```

## Signal Processing Prompt

ใช้กับ audio, EEG, vibration, radio burst, waveform, spectrogram, time-frequency:

```text
[วาง Resource Block ตรงนี้]

นี่คือโจทย์ Signal Processing

ช่วยสร้าง notebook โดยเน้น:
- อ่านไฟล์ signal/audio/waveform และ metadata
- ตรวจ sampling rate, duration, channel count, label distribution
- ทำ quick visualization: waveform, histogram length, spectrogram ตัวอย่าง
- เลือก representation:
  - handcrafted features: mean/std/RMS/zero-crossing/energy
  - FFT/STFT/Mel-spectrogram ถ้าเป็น audio/waveform
  - bandpower/features เฉพาะ domain ถ้าเป็น EEG/sensor
- baseline:
  - feature table + classical ML ก่อน
  - spectrogram image + CNN/pretrained model ถ้าคุ้มเวลา
  - 1D CNN ถ้า sequence length manageable
- validation:
  - group split ถ้ามี subject/session/source
  - ระวัง segment จากไฟล์เดียวกันหลุดไปทั้ง train/valid
- inference + submission

ให้หา reference จาก Preparation โดยเน้น SS5 Signal Processing ก่อน แล้ว SS4 เป็นรอง

ก่อนจบให้สรุป:
- representation ที่เลือกและเหตุผล
- validation metric
- leakage risks
- submit path
```

## NLP Prompt

ใช้กับ classification, NER, QA, retrieval, summarization, translation, word segmentation:

```text
[วาง Resource Block ตรงนี้]

นี่คือโจทย์ Natural Language Processing

ช่วยสร้าง notebook โดยเริ่มจาก:
- อ่าน train/test/sample_submission
- ตรวจ task type: classification / NER / QA / retrieval / generation / translation / word segmentation
- ตรวจภาษา: Thai / English / mixed
- ตรวจ label distribution, text length, missing text, duplicated text
- metric: F1/accuracy/EM/BLEU/ROUGE/WER/custom

ให้หา reference เก่าใน Preparation โดยเน้น SS5 NLP ก่อน เช่น QA, word segmentation, NER, retrieval

Baseline ที่ต้องมี:
- clean text แบบไม่ทำลายข้อมูลสำคัญ
- split ที่ไม่ leak duplicate/near-duplicate text
- simple baseline:
  - TF-IDF + LogisticRegression/LinearSVC สำหรับ classification
  - rule/simple tokenizer สำหรับ word segmentation ถ้าเวลาเหลือน้อย
  - BM25/TF-IDF retrieval baseline สำหรับ QA/retrieval
- stronger baseline ถ้า runtime พร้อม:
  - WangchanBERTa / multilingual transformer / sentence-transformers ตามโจทย์
- inference + create submission.csv

ถ้าเป็น Kaggle/Colab:
- รวม path/config ไว้ cell แรก
- อย่าพึ่งโหลดโมเดลใหญ่มากถ้าไม่แน่ใจ internet/GPU
- cache/output ให้อยู่ใน working path

ก่อนจบ:
- show ตัวอย่าง prediction 10 แถว
- check submission format
- ใช้ $scrutinize ตรวจก่อน submit
```

## Prompt: ให้สร้าง Notebook จากศูนย์

ใช้เมื่อมีข้อมูลครบแล้วและอยากให้ Codex generate `.ipynb`:

```text
[วาง Resource Block ตรงนี้]

ช่วยสร้าง Jupyter Notebook ที่เปิดได้จริงด้วย `nbformat`

Notebook requirements:
- ชื่อไฟล์: [output notebook path]
- มี markdown title, task summary, metric, data paths
- ใช้รูปแบบการเขียน Markdown และ code organization เหมือน notebook ตัวอย่างใน `Level 1/Hackathon 4_5 Domains Hackathon`
- โครงหัวข้อควรใกล้เคียง Level 1:
  1. `# 1. Setup & Imports`
  2. `# 2. Data Loading & Initial Inspection`
  3. `# 3. Data Preprocessing` หรือ domain-specific preparation
  4. `# 4. Feature Engineering / Model Preparation`
  5. `# 5. Model Training & Evaluation`
  6. `# 6. Prediction & Submission Generation`
- ทุก major section ต้องมี Markdown ภาษาไทยสั้น ๆ อธิบายว่า cell ถัดไปทำอะไรและทำเพื่ออะไร
- Code style ให้เหมือน Level 1: ใช้ config/path cell ชัดเจน, `display(...)`/`print(...)` ตรวจ shape/head/value_counts, ใช้ `tqdm` เมื่อ loop ยาว, และเซฟ submission ตอนท้ายพร้อม print path
- มี config cell รวม path/seed/runtime
- มี cells:
  1. install/import
  2. load data
  3. quick EDA
  4. preprocessing
  5. validation split
  6. baseline model
  7. validation metric
  8. train final/inference
  9. save submission
  10. sanity checks
- ต้องไม่ hard-code path กระจัดกระจาย ให้แก้ path ได้จาก config cell เดียว
- ถ้าเป็น Kaggle ให้ default path เป็น `/kaggle/input/...` และ `/kaggle/working/submission.csv`
- ถ้าเป็น Colab ให้มี optional mount Drive cell

หลังสร้างไฟล์:
- validate ว่า notebook JSON เปิดได้
- สรุปวิธีรัน cell order
```

## Prompt: Kaggle Submission Safety Check

ใช้ก่อนกด submit ทุกครั้ง:

```text
ใช้ $scrutinize ตรวจ readiness ก่อนส่ง Kaggle

ตรวจไฟล์:
- notebook: [path notebook]
- submission: [path submission.csv]
- sample submission: [path sample_submission.csv]
- statement: [path/link statement]

ช่วยตรวจ:
1. submission มีจำนวนแถวเท่ากับ sample_submission ไหม
2. column name/order/type ตรงไหม
3. prediction มี NaN/inf/empty/string ผิดชนิดไหม
4. index/id ตรงกับ test/sample submission ไหม
5. validation split มี leakage ไหม
6. inference ใช้ preprocessing เดียวกับ train ไหม
7. มี path ที่ใช้ได้เฉพาะ local แต่ Kaggle จะหาไม่เจอไหม
8. มี dependency ที่ Kaggle/Colab ไม่มีและไม่ได้ install ไหม
9. จุดที่น่าจะ fail ตอน rerun notebook ตั้งแต่ต้นคืออะไร

ถ้าพบปัญหา ให้แก้ไฟล์ให้เลย แล้ว rerun sanity checks ที่จำเป็น
```

## Prompt: Colab Runtime Setup

ใช้เมื่อต้องการ notebook สำหรับ Colab:

```text
ช่วยปรับ notebook นี้ให้ใช้บน Google Colab ได้:
- notebook path: [path]
- dataset path ใน Drive: [path]
- output path: [path]

ต้องมี:
- cell mount Google Drive
- config cell สำหรับ DATA_DIR, OUTPUT_DIR, SUBMISSION_PATH
- install cell เฉพาะ dependencies ที่จำเป็น
- fallback ถ้าไม่มี GPU
- save submission ไปยัง path ที่กำหนด
- markdown อธิบายว่าต้องแก้ path ตรงไหนก่อนรัน

อย่าทำให้ notebook ผูกกับ path local Windows เว้นแต่ใส่เป็น optional
```

## Prompt: Kaggle Runtime Setup

ใช้เมื่อต้องการ notebook สำหรับ Kaggle:

```text
ช่วยปรับ notebook นี้ให้ใช้บน Kaggle ได้:
- notebook path: [path]
- Kaggle dataset input path: [/kaggle/input/...]
- output submission path: /kaggle/working/submission.csv

ต้องมี:
- config cell รวม DATA_DIR/OUTPUT_DIR/SUBMISSION_PATH
- ไม่ใช้ path local Windows
- ไม่ต้อง mount Drive
- ไม่ download resource จาก internet เว้นแต่โจทย์อนุญาต
- save submission ที่ `/kaggle/working/submission.csv`
- print `ls /kaggle/input` และ check file exists
- sanity check sample_submission ก่อน save
```

## Prompt: Debug ระหว่างแข่ง

ใช้เมื่อ error/score แปลก:

```text
ใช้ $debug-mantra ช่วย debug ปัญหานี้

อาการ:
- [error/metric แปลก/submit fail]

ไฟล์ที่เกี่ยวข้อง:
- notebook/script: [path]
- dataset/sample: [path]
- log/error: [paste หรือ path]

ช่วยทำตามนี้:
1. reproduce ปัญหาให้ได้
2. trace fail path ว่าพังจาก cell/function ไหน
3. ตั้ง hypothesis และ falsify ทีละข้อ
4. cross-reference กับโจทย์, sample_submission, และ reference เก่าใน Preparation
5. แก้เฉพาะจุดที่จำเป็น
6. rerun sanity check หลังแก้
```

## Prompt: หลังได้ Baseline แล้วให้ Improve

```text
ตอนนี้ baseline รันได้แล้ว

ไฟล์:
- notebook: [path]
- submission: [path]
- validation score: [score]
- metric: [metric]
- runtime budget ที่เหลือ: [เวลา]

ช่วยเลือก improvement ที่คุ้มที่สุดไม่เกิน 3 อย่าง โดย:
- อิงจากโจทย์และ validation result
- หลีกเลี่ยง leakage
- ไม่ทำให้ notebook ซับซ้อนเกินจน rerun ไม่ทัน
- ถ้าเพิ่ม ensemble ให้ยัง save submission ง่าย
- ถ้าเพิ่ม feature/model ให้เก็บ baseline เดิมไว้เปรียบเทียบ

หลังแก้ให้สรุปว่าอะไรเปลี่ยน และต้องรัน cell ไหนใหม่
```

## Prompt: Final Handoff ในแต่ละโจทย์

ใช้ตอนจบแต่ละข้อเพื่อบันทึก state:

```text
ช่วยทำ final handoff สำหรับโจทย์นี้

ไฟล์:
- notebook: [path]
- submission: [path]
- README/log: [path]

สรุปเป็น Markdown:
1. task summary
2. dataset paths
3. metric
4. best validation score
5. model/features ที่ใช้
6. exact submission path
7. command/cell order ที่ต้องรัน
8. risks ที่ยังเหลือ
9. next experiments ถ้ามีเวลา

บันทึกเป็น:
[path เช่น Level 2/Hackathon 9_5 Domains Hackathon/workspace/<domain>/HANDOFF.md]
```

## One-Shot Prompt สำหรับวันสอบ

ถ้าต้องการ prompt เดียวที่ใช้ได้กับทุก domain ให้ paste ตัวนี้แล้วเติมช่องว่าง:

```text
เรากำลังทำ Super AI Engineer SS6 Level 2 - 5 Domains Hackathon

Domain:
[Data Science / Computer Vision / IoT / Signal Processing / NLP]

โจทย์:
[paste รายละเอียดโจทย์]

Dataset:
[link/path dataset]

Sample submission:
[path sample_submission.csv หรือรายละเอียด format]

Runtime:
[Kaggle / Colab / Local]

Output ที่ต้องสร้าง:
- notebook: [path]
- submission: [path]
- README/handoff: [path]

Reference:
- ค้นโจทย์เก่าใน `Level 2/Hackathon 9_5 Domains Hackathon/Preparation`
- ใช้ SS5 เป็นหลัก
- ใช้ SS4 เป็นรองเมื่อจำเป็น

ขอให้ทำแบบนี้:
1. วิเคราะห์โจทย์ + metric + submission format
2. หา reference เก่าที่ใกล้ที่สุดและสรุป pattern ที่ใช้ได้
3. สร้าง notebook ที่รันได้จริงตาม runtime
4. ทำ baseline submit ได้ก่อน
5. เพิ่ม validation ที่กัน leakage
6. ทำ inference และสร้าง submission.csv
7. ใช้ $scrutinize ตรวจความเสี่ยงก่อนจบ
8. สรุปวิธีรันและ next experiments

เน้นความสำคัญ:
- submission ต้อง format ถูก
- notebook ต้อง rerun จากต้นได้
- path ต้องเหมาะกับ Kaggle/Colab ตามที่ระบุ
- ถ้าไม่มั่นใจอย่าเดา ให้ inspect ไฟล์จริงก่อน
```

# รูปแบบโจทย์และข้อมูลแบบละเอียด
In this individual assessment, you will build a system that reads an image of a Thai-language mathematics problem and outputs the correct answer. The dataset contains 700 problems drawn from Thai math curricula across multiple grade levels — primary, lower-secondary, and upper-secondary — and includes a mix of arithmetic, algebra, geometry, combinatorics, and word problems.

Each problem is provided as a single image. Problems may include:

Thai natural-language text describing the question.
Diagrams (geometric figures, number lines, tables, charts).
Mathematical notation, including fractions, square roots, and angle measures.
Your task is to produce a CSV of predicted answers — one row per problem — that maximizes exact-match accuracy against the hidden ground-truth answers, after a deterministic normalization step (described in detail in the Evaluation tab).

Goal
Given an image of a Thai math problem, predict the answer.

A trivial baseline that answers "2" for every problem scores approximately 4–6%. Your submission should comfortably beat this. Strong solutions are expected to combine a capable vision-language model (or OCR + LLM pipeline) with careful prompting and answer normalization.

Why this matters
Thai-language mathematical reasoning is an underexplored intersection of:

Vision-language modeling — understanding figures, equations, and Thai text in a single image.
Numerical and symbolic reasoning — performing multi-step calculations correctly.
Low-resource language modeling — Thai is far less represented than English in most pretrained models.
Solving this challenge requires more than calling a single API — you will need to engineer how the model sees the problem, reasons about it, and formats its final answer.

What you will submit
A single CSV file with two columns: id and answer, covering every row in test.csv. The exact format is described in Evaluation → Submission File.

Eligibility & rules
This competition is restricted to participants of the Super AI Engineer SS6 program and is an individual assessment — no teams. See the Rules tab for the full set of constraints, including allowed external data, pretrained models, submission limits, and prohibited actions.

Start

9 hours ago
Close

5 hours to go
Description
The problem
Each example in this competition is a single image containing a complete Thai-language mathematics problem. The image carries the entire context the model needs to answer:

The problem statement (Thai natural language).
Any accompanying diagram — triangles, circles, coordinate grids, number tables, etc.
Any necessary mathematical notation, drawn in the original textbook typesetting.
The model's task is to produce a single short answer string per problem. Answers vary in form across the dataset:

Plain integers — e.g. "36", "227", "-2"
Decimals — e.g. "2.8125"
Numbers with Thai units — e.g. "20 ตารางเซนติเมตร", "30 องศา", "15 หน่วย"
LaTeX expressions — e.g. "$6\sqrt{3}$", "$\frac{17}{10}$"
Short Thai phrases — e.g. "ขาดทุน ร้อยละ 1", "ข้อ (จ)"
The Evaluation tab describes exactly how these forms are normalized before comparison, so you do not need to match the host's exact string — you only need to match it up to normalization.

Data composition
The 700 problems are drawn from 9 source buckets (identified by a 3-digit prefix in the original filename), which correspond roughly to grade level and problem set.

Adversarial Split Warning: While the train.csv set contains a representative mix of all 9 buckets, the test set features an Adversarial Split. Certain buckets are evaluated exclusively on the Public Leaderboard, while others are evaluated exclusively on the Private Leaderboard. Beware of overfitting to the Public Leaderboard difficulty mix!

Bucket	Count
101	26
102	27
103	125
104	134
105	101
116	105
118	38
120	30
122	114
Files
train.csv — id, image_path, answer for 280 problems.
test.csv — id, image_path for 420 problems (no answers).
sample_submission.csv — a valid submission that predicts "2" for every row.
images/{id}.jpg — the 700 problem images, named by id.
image_path is a path relative to the dataset root, e.g. images/42.jpg.

What approaches work
We expect competitive solutions to combine several ideas:

⚠️ Commercial vision-language model APIs are not allowed in this competition. You may not call hosted services such as Claude, GPT-4o, Gemini, or any other paid or free commercial inference API to produce your predictions. See the Rules tab for the precise definition. All inference must run from open-weights models that you load and run yourself (locally, on Kaggle Notebooks, or on your own infrastructure).

Open-weights vision-language models
An open-weights VLM (for example Qwen-VL, InternVL, Llama-3.2-Vision, MiniCPM-V, Pixtral, DeepSeek-VL, Molmo) can read the image directly and reason about the problem. Things to think about:

Prompt design — instruct the model to extract the problem, reason step by step, and output only the final answer in a constrained format.
Few-shot examples — draw 2–5 demonstrations from train.csv to anchor both the reasoning style and the answer format.
Self-consistency — sample multiple chains and take a majority vote.
Quantization — for larger open-weights models, use 4-bit / 8-bit loading (bitsandbytes, AWQ, GPTQ) to fit on a single GPU.
OCR + LLM pipelines
If you prefer not to rely on a single VLM, you can:

Run a Thai-capable OCR (Tesseract Thai, EasyOCR, PaddleOCR) to extract the problem text.
Pass the text — and optionally a description of the figure — to a strong open-weights reasoning LLM (Qwen, Llama, DeepSeek, Typhoon, etc.) that you run yourself.
Combine with a separate vision pass for any geometric content.
Hybrid approaches
The strongest entries will likely combine open-weights VLM perception with explicit symbolic reasoning (e.g. SymPy), post-process the model's output through your own normalizer, and validate against the held-out portion of train.csv before submitting.

Tips and pitfalls
Match the normalizer. A correct answer that is formatted wrongly will be scored wrong. Build your own copy of the normalizer (see Evaluation), run it on your predictions, and check against train.csv locally before each submission.
Watch units. Many ground-truth answers carry Thai units (ตารางเซนติเมตร, องศา, วิธี, จำนวน, etc.) that the normalizer strips. You can submit "20" or "20 ตารางเซนติเมตร" and both will normalize the same way — but only if you stay within the recognized unit list.
Thai vs Arabic digits. The normalizer translates ๐–๙ to 0–9. Don't rely on this for answers — just always submit Arabic digits.
Use your daily submission budget wisely. Five submissions per day is enough to validate small prompt or model tweaks. Use most of your iteration on local evaluation against train.csv.
Acknowledgements
This dataset is curated from publicly available Thai mathematics problems for the purpose of evaluating Super AI Engineer SS6 candidates. Questions and ground-truth answers are used here for educational assessment only.

Evaluation
Submissions are evaluated on accuracy — the fraction of test problems for which your predicted answer exactly matches the ground-truth answer, after a deterministic normalization step that is applied to both sides before comparison.

Formally, for a test set of $N$ problems with ground-truth answers $y_1, \dots, y_N$ and your predictions $\hat{y}_1, \dots, \hat{y}_N$:


The Public Leaderboard is computed on approximately 34% of the test set; the Private Leaderboard on the remaining 66%. Final standings are determined by the Private Leaderboard.

Normalization
Both the prediction and the ground truth pass through the same normalize function before comparison. The function is deterministic, public, and applied identically to every submission. The steps, in order, are:

Lowercase and strip. Outer whitespace is removed, and the string is lowercased.
Thai digits → Arabic digits. Characters ๐ ๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙ are mapped to 0–9.
Strip math delimiters. Dollar signs $ (used for inline LaTeX) are removed.
Strip recognized units. Common Thai and English unit words that do not change the value are removed — for example ตารางเซนติเมตร, ลูกบาศก์หน่วย, เซนติเมตร, องศา, หน่วย, จำนวน, วิธี, แบบ, ค่า, ร้อยละ, ดอลลาร์, บาท, degrees, square centimeters, years old, and others.
Expand LaTeX. A fixed set of LaTeX macros is rewritten to plain text:
\frac{a}{b} → (a)/(b)
\sqrt{x} → sqrt(x)
\pi → pi
\times, \cdot → *
\div → /
\pm → +-
\overrightarrow{AB}, \overline{AB}, \vec{AB} → AB
\left, \right, \,, \;, \:, \! → removed
Drop structural characters. Whitespace and the characters { } \ , are removed.
Drop redundant parentheses around pure integers. (3) → 3. This means sqrt(3) collapses to sqrt3 on both sides, so they still match.
Integer canonicalization. If the result parses as a number with zero fractional part, it is rewritten in integer form. "2.0" and "2" both normalize to "2".
Worked examples
Raw answer	After normalization
20 ตารางเซนติเมตร	20
30 องศา	30
15 หน่วย	15
$6\sqrt{3}$	6sqrt3
$\frac{17}{10}$	17/10
$\frac{3\sqrt{3}}{2}$	3sqrt3/2
๒๕	25
2.0	2
80 degrees	80
You do not need to reproduce the host's exact formatting — you only need to produce a string that normalizes to the same canonical form.

Submission File
For each id in test.csv, you must predict a single string in the answer column. The file must contain a header and have the following format:

id,answer
0,20
1,1
2,6sqrt(3)
3,10
4,69
etc.
Constraints:

The file must contain exactly two columns with the header id,answer.
Every id in test.csv must appear exactly once in your submission.
id is treated as a string. Do not strip leading zeros or change the format.
answer may contain Thai characters, LaTeX, units, or plain numbers — the normalizer handles all of them.
Empty answer cells are scored as wrong.
Local validation
A trivial baseline submission that predicts "2" for every test row is provided as sample_submission.csv. It scores approximately 6.3% on the Public Leaderboard and 3.9% on the Private Leaderboard. Use it to verify your submission pipeline end-to-end before iterating on a real solution.

For local validation, hold out a small slice of train.csv and score yourself with the same normalization rules — your local score will be a strong predictor of the public score, since the train, public, and private splits are stratified the same way.

The dataset contains 700 Thai-language mathematics problems, each represented as a single image plus a short string answer. The data has been split into a training set (with answers) and a test set (without answers). Your task is to predict the answer for every problem in the test set.

Files
File	Rows	Description
train.csv	280	Training problems — id, image_path, and ground-truth answer.
test.csv	420	Test problems — id and image_path only. Predict the answer.
sample_submission.csv	420	Example submission in the correct format (predicts "2" for every row).
images/	700	All problem images, one JPEG per problem, named {id}.jpg.
Columns
train.csv
Column	Type	Description
id	string	Unique identifier for the problem. Use this to match images and submission rows.
image_path	string	Path to the problem image, relative to the dataset root (e.g. images/42.jpg).
answer	string	The ground-truth answer. May be a plain number, a number with a Thai unit, a LaTeX expression, or a short Thai phrase.
test.csv
Column	Type	Description
id	string	Unique identifier for the problem.
image_path	string	Path to the problem image, relative to the dataset root.
sample_submission.csv
Column	Type	Description
id	string	Must match every id in test.csv.
answer	string	Your predicted answer for that id.
Images
Format: JPEG
Color mode: RGB
Dimensions: vary by problem — most are roughly 1000–1500 px wide and 500–800 px tall.
Content: each image contains the complete problem — Thai text, any diagrams (geometric figures, tables, number lines, coordinate planes), and any required mathematical notation. The image is self-contained; no external context is needed.
To load an image in Python:

from PIL import Image
img = Image.open("images/0.jpg")
img.show()
Example rows from train.csv
id,image_path,answer
0,images/0.jpg,20 ตารางเซนติเมตร
1,images/1.jpg,1
2,images/2.jpg,$6\sqrt{3}$
3,images/3.jpg,10
5,images/5.jpg,$(\frac{17}{10}, \frac{289}{20})$
9,images/9.jpg,30 องศา
Answer forms
Ground-truth answers in train.csv appear in several forms. The Evaluation tab describes how each form is normalized before scoring, so you only need to match up to normalization.

Form	Example	Frequency
Plain integer	36, 227, -2	Most common
Decimal	2.8125	Occasional
Number + Thai unit	20 ตารางเซนติเมตร, 30 องศา	Common
LaTeX fraction or square root	$\frac{17}{10}$, $6\sqrt{3}$	~9% of training set
Short Thai phrase	ขาดทุน ร้อยละ 1, ข้อ (จ)	Rare
Thai digits	๒๕	Rare
Sentinel	<n/a>	A handful of rows
Data composition
The 700 problems come from 9 source buckets, identified by a 3-digit prefix in the original problem filenames. The buckets correspond approximately to grade level and source collection.

Notes on the data
Problems are drawn from publicly available Thai mathematics curricula and are used here solely for educational assessment within the Super AI Engineer SS6 program.
No problem appears in more than one split. Each id is unique across the entire dataset.
The test set contains no answer column. Any submission that depends on reading an answer from test.csv will fail.
There is no leakage between train and test: the same problem (image) does not appear in both.
Loading the data
A minimal example to get started:

# Make a trivial submission
import pandas as pd
from PIL import Image

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# Look at one example
row = train.iloc[0]
img = Image.open(row["image_path"])
print(row["id"], "→", row["answer"])
img.show()

# Make a trivial submission
sub = pd.DataFrame({"id": test["id"], "answer": "2"})
sub.to_csv("submission.csv", index=False)

Competition Rules
Super AI Engineer SS6 — Individual Test Thai Math Visual QA

By submitting to this competition, you agree to be bound by the rules below. The competition host's decisions are final in all matters relating to the competition.

1. Eligibility
1.1 This competition is restricted to enrolled participants of the Super AI Engineer SS6 program. Invitations are issued by the program organizers; uninvited Kaggle users may view the competition page but may not join, submit, or appear on the leaderboard.

1.2 Participants must be at least 13 years of age and meet Kaggle's general Terms of Service.

1.3 Employees, contractors, and immediate family of the competition host, judges, or evaluation team are not eligible for top placements but may join for practice.

2. Individual Participation
2.1 This is an individual assessment. Teams are not permitted. Each participant must work alone, on their own Kaggle account, and submit independently.

2.2 You may not form or join teams in the Kaggle UI for this competition. Any team merger attempts will result in disqualification of all involved accounts.

2.3 You must not collaborate with any other participant — including, but not limited to: discussing solution approaches, sharing prompts, sharing code, sharing model outputs, sharing predictions, sharing notebooks privately, or coordinating submissions.

3. One Account Per Participant
3.1 You may submit from only one Kaggle account. Submitting from multiple accounts, or having another person submit on your behalf, is strictly forbidden.

3.2 Use of duplicate, alternate, or proxy accounts will result in immediate disqualification of all related accounts and may result in removal from the Super AI Engineer SS6 program.

4. Submissions
4.1 Each submission must be a valid CSV with the exact header id,answer, containing one row for every id in test.csv. Malformed submissions will fail to score.

4.2 Submission limit: 5 submissions per day per participant.

4.3 At the end of the competition you may select up to 2 submissions to be used for Private Leaderboard scoring. If you do not select any, Kaggle will use your two highest-scoring submissions on the Public Leaderboard by default.

4.4 Late submissions (after the official close time) will not be scored.

5. External Data and Pretrained Models
5.1 External datasets: Permitted. You may use any publicly available dataset (textbooks, prior math competition archives, open VQA datasets, etc.) to develop your solution, provided the data was publicly available before the competition start date and you do not include data that overlaps with the test set.

5.2 Pretrained models — open weights only. You may use any pretrained model whose weights are openly published and downloadable (for example Llama, Qwen, Qwen-VL, InternVL, MiniCPM-V, Pixtral, DeepSeek-VL, Typhoon, Molmo). You may run these models locally, on Kaggle Notebooks, on Colab, or on your own infrastructure.

5.3 Commercial / hosted inference APIs are PROHIBITED. You may not call any hosted model-serving API — paid or free — to produce predictions for this competition. This includes, but is not limited to: Anthropic Claude, OpenAI GPT-4o / GPT-4 / o1 / any OpenAI model, Google Gemini / PaLM / Vertex AI, AWS Bedrock, Azure OpenAI, Cohere, Mistral La Plateforme, Together AI, Replicate, Fireworks, Groq, Perplexity, DeepInfra, OpenRouter, Hugging Face Inference API / Inference Endpoints, and any equivalent service. The model performing inference must be one whose weights you have downloaded and are running yourself.

5.4 Open-source tooling APIs are still permitted. OCR libraries that run locally (Tesseract, EasyOCR, PaddleOCR), search-augmentation against your own local indices, and any other local computation are allowed.

5.5 Disclosure: You must briefly disclose the external data and pretrained models you used in the description of your final selected submission. Failure to disclose may result in disqualification.

6. Prohibited Actions
The following actions are strictly forbidden and will result in disqualification:

6.1 Hand-labeling the test set — manually inspecting test images and writing answers by hand, in whole or in part. Submissions must be produced by an automated system.

6.2 Probing the leaderboard — using the submission limit to extract per-id answers (e.g., binary-search techniques against the public LB).

6.3 Sourcing the ground truth — attempting to identify the original source of any test problem (e.g., from a textbook or competition archive) in order to look up its answer.

6.4 Sharing solutions — sharing code, prompts, predictions, notebooks, or any solution component with other participants during the competition, by any channel (private chat, screen-share, code repositories, etc.).

6.5 Reverse engineering the evaluation system, scraping non-public Kaggle endpoints, or otherwise attempting to obtain information not intended to be public.

6.6 Plagiarism — submitting work substantially copied from public Kaggle notebooks, GitHub repositories, or any other source without your own meaningful contribution.

6.7 Routing predictions through a commercial API — using any wrapper, proxy, agent framework, browser-automation, or third-party tool that ultimately invokes a prohibited commercial inference API (see §5.3) to produce or refine your submission. The prohibition applies to the act of using the model, regardless of how the call is dressed up.

7. Notebooks and Code Sharing
7.1 During the competition, you may use Kaggle Notebooks privately for your own development.

7.2 Do not publicly share notebooks containing competition-specific solutions while the competition is open. Public notebooks containing baseline exploration (data loading, EDA only) are allowed.

7.3 After the competition closes, top participants may be asked to publish their solution notebook for the program's educational benefit.

8. Winner Requirements
8.1 To be eligible for top placement, the top 5 participants on the Private Leaderboard must:

(a) Submit a clean, runnable notebook or repository reproducing their final submission within 7 days of the competition close. (b) Provide a written description (1–2 pages) of their approach, including the external data and pretrained models used. (c) Make their submission reproducible — fix random seeds, pin dependencies, and document the environment. (d) Be available for a short verification interview with the program organizers if requested.

8.2 Failure to meet these requirements within the deadline will result in forfeiting the placement, and the next-ranked eligible participant will be moved up.

9. Evaluation
9.1 Submissions are scored by accuracy — exact match between your predicted answer and the ground truth, after applying the normalization rules described in the competition Overview. The normalization is deterministic and the same for every participant.

9.2 The Public Leaderboard is computed on approximately 35% of the test set and is visible throughout the competition.

9.3 The Private Leaderboard is computed on the remaining 65% of the test set and is revealed only after the competition closes. Final rankings are determined by the Private Leaderboard.

10. Disqualification
10.1 The competition host reserves the right to disqualify any participant, at any time, for violating these rules. Disqualification may be applied with or without prior notice.

10.2 Disqualified submissions will be removed from both leaderboards.

10.3 Disqualification from this competition may also result in removal from the Super AI Engineer SS6 program, at the program's discretion.

11. General Provisions
11.1 The competition host may modify these rules at any time for clarity, fairness, or to address unforeseen issues. Material changes will be announced on the competition forum.

11.2 In the event of a tie on the Private Leaderboard, the participant who reached the tied score earlier (by submission timestamp) will be ranked higher.

11.3 All decisions of the competition host and the Super AI Engineer SS6 program organizers are final.

11.4 By participating, you grant the competition host the right to use your username, ranking, and (with your consent) your solution writeup for educational and promotional purposes within the Super AI Engineer SS6 program.

12. Contact
Questions about the rules should be posted on the competition's Discussion forum so all participants receive the same information. Private questions about eligibility or suspected violations may be sent to the program organizers.