from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = Path(os.environ.get("OCR_PIPELINE_PATH", PROJECT_DIR / "paddle_api_ocr_test_pipeline.py"))
DEFAULT_RUNS_DIR = PROJECT_DIR / "p_non_api_ready" / "runs"
DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
_PIPELINE_MODULE: Any | None = None


def _load_pipeline_module() -> Any:
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is not None:
        return _PIPELINE_MODULE

    pipeline_path = PIPELINE_PATH
    if not pipeline_path.exists():
        fallback_path = PACKAGE_DIR / "paddle_api_ocr_test_pipeline.py"
        if fallback_path.exists():
            pipeline_path = fallback_path
    if not pipeline_path.exists():
        raise RuntimeError(
            "OCR pipeline file not found. Put paddle_api_ocr_test_pipeline.py next to this package, "
            "in the parent folder, or set OCR_PIPELINE_PATH."
        )

    spec = importlib.util.spec_from_file_location("paddle_api_ocr_test_pipeline_runtime", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PIPELINE_MODULE = module
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_answers(path: Path) -> dict[str, dict[str, str]]:
    payload = _read_json(path)
    answers = payload.get("answers", payload)
    if not isinstance(answers, dict):
        return {}
    return {
        str(artifact_id): answer
        for artifact_id, answer in answers.items()
        if isinstance(answer, dict)
    }


def run_ocr_on_folder(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    clean_output: bool = True,
) -> dict[str, Any]:
    """Run OCR pipeline on a folder of statement images.

    Expected filenames:
    - BS-..._header.png
    - BS-..._transactions_p1.png
    - BS-..._transactions_p2.png

    Returns parsed headers, transactions, debug rows, and metadata.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = DEFAULT_RUNS_DIR / f"run_{uuid.uuid4().hex[:12]}"
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = _load_pipeline_module()

    module.INPUT_DIR = input_dir
    module.OUTPUT_DIR = output_dir
    module.CROP_DIR = output_dir / "debug_crops"
    module.BY_BANK_DIR = output_dir / "by_bank"
    module.BY_ARTIFACT_DIR = output_dir / "by_artifact"
    module.TRANSACTIONS_CSV = output_dir / "transactions_canonical.csv"
    module.HEADERS_CSV = output_dir / "headers.csv"
    module.OCR_DEBUG_CSV = output_dir / "ocr_debug.csv"
    module.ARTIFACT_SCHEMA_CSV = output_dir / "artifact_schema_summary.csv"
    module.META_JSON = output_dir / "run_meta.json"
    module.ANSWERS_JSON = output_dir / "answers.json"

    for directory in [module.OUTPUT_DIR, module.CROP_DIR, module.BY_BANK_DIR, module.BY_ARTIFACT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    module.main()

    answers = _read_answers(module.ANSWERS_JSON)
    result = {
        "output_dir": str(output_dir),
        "answers": answers,
        "transactions": _read_csv(module.TRANSACTIONS_CSV),
        "headers": _read_csv(module.HEADERS_CSV),
        "artifact_schema": _read_csv(module.ARTIFACT_SCHEMA_CSV),
        "debug": _read_csv(module.OCR_DEBUG_CSV),
        "meta": _read_json(module.META_JSON),
    }
    if len(answers) == 1:
        result["answer"] = next(iter(answers.values()))
    return result


def run_ocr_on_uploaded_files(
    files: list[tuple[str, bytes]],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run OCR on uploaded file bytes.

    `files` is a list of `(filename, bytes)` pairs. Filenames are used for
    artifact routing, so keep the original render names.
    """
    with tempfile.TemporaryDirectory(prefix="bank_ocr_upload_") as temp:
        input_dir = Path(temp)
        for filename, content in files:
            safe_name = Path(filename).name
            if not safe_name:
                continue
            (input_dir / safe_name).write_bytes(content)
        return run_ocr_on_folder(input_dir=input_dir, output_dir=output_dir, clean_output=True)
