# FahMai OCR CSV Reproduction Package

This package contains the PaddleOCR checkpoint outputs, verified image
annotations, and a single notebook entry point for recreating:

```text
ocr_outputs/submission_OCR_image_reviewed.csv
```

## One-Notebook Path

Open:

```text
notebooks/02_run_fast_ocr_pipeline.ipynb
```

Run all cells. The notebook rebuilds the CSV from the included PaddleOCR
checkpoints, applies verified image annotations, validates the submission
schema, and checks that bank-statement `business_event_date` fields are
date-only. No full-page generative OCR is used.

This is the only file you need to run to recreate the final CSV.

## Optional Script Equivalent

From the package root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ocr_pipeline\run_build_submission_from_checkpoints.ps1
```

This uses the included checkpoint folders and does not require the 4GB public
render dataset. It is kept only as a command-line equivalent of the notebook
path.

## Full OCR Rerun

The full OCR rerun still needs the public render dataset under:

```text
fahmai_renders_with_json/
```

or the short junction:

```text
C:\fahmai_ocr_data
```

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ocr_pipeline\run_exam_fast_ocr.ps1
```

## Engine Summary

- Bank statements: PaddleOCR `TextRecognition`, model `en_PP-OCRv5_mobile_rec`
- Fixed non-bank forms: PaddleOCR fixed crops, models `en_PP-OCRv5_mobile_rec`
  and `th_PP-OCRv5_mobile_rec`
- General PDFs/documents: native PDF text extraction or deterministic layout
  rules
