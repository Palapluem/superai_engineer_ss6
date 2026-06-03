# PaddleOCR Fast Pipeline

This folder contains the production OCR pipeline for the FahMai OCR task.
The final path uses PaddleOCR crop recognition and deterministic layout rules.
Full-page generative OCR is not used by the final submission pipeline.

## Notebook Entry Point

Open:

```text
..\notebooks\02_run_fast_ocr_pipeline.ipynb
```

Run the quick-path cells to rebuild:

```text
..\ocr_outputs\submission_OCR_image_reviewed.csv
```

from existing checkpoints.

## Fast Checkpoint Rebuild

From this folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_build_submission_from_checkpoints.ps1
```

This rebuilds the final CSV from:

- `..\submission_template_OCR.csv`
- `..\ocr_outputs\fast_*` checkpoint folders
- `..\ocr_outputs\audits\image_grounded\manual_ground_truth_annotations.csv`

It does not require the 4GB render dataset.

## Full OCR Rerun

Use this only when the public render dataset is available:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_exam_fast_ocr.ps1
```

The launcher uses GPU Paddle statement lanes when `C:\fahmai_paddle_gpu` is
ready, and CPU lanes for fixed-layout non-bank forms and general documents.

## Engine Summary

- Bank statements: PaddleOCR `TextRecognition`, model
  `en_PP-OCRv5_mobile_rec`
- Fixed receipt/vendor/warranty forms: PaddleOCR fixed crops,
  `en_PP-OCRv5_mobile_rec` and `th_PP-OCRv5_mobile_rec`
- General documents: native PDF text extraction or deterministic public-layout
  rules

## Validation

```powershell
C:\fahmai_paddle\Scripts\python.exe .\validate_submission.py ..\ocr_outputs\submission_OCR_image_reviewed.csv
```

The validator checks artifact order, JSON schema, and the required
date-only format for bank-statement `business_event_date` fields.

## Read-only Benchmark

`..\ocr_outputs\PaddleOCR.csv` is a read-only diagnostic benchmark. It must not
be used as a source of submitted values.
