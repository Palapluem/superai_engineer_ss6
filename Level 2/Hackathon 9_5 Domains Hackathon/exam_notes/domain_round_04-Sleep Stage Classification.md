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

# รายละเอียดของโจทย์
train/ - This folder contains the training set with 83 CSV files. Each file includes continuous numerical values collected from multiple sensors, resampled to 16 Hz. The continuous signals can be considered as continuous segments, with each 30-second segment having the same label.

test_segment/ - This process segments the test data for each subject ID into 30-second intervals, ensuring that each segment is associated with a single label. The segmented data corresponds to the "id" column in the sample submission file. sample_submission.csv - This CSV file serves as a sample submission. It includes the filenames corresponding to subject_segment IDs and the predicted labels.