from __future__ import annotations

import asyncio
import base64
import binascii
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException

from ocr_service import DEFAULT_RUNS_DIR, run_ocr_on_uploaded_files


app = FastAPI(
    title="Bank Statement OCR API",
    version="0.2.0",
    description="PaddleOCR endpoint for FahMai OCR back-test bank-statement extraction.",
)

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}
WORKER_SHUTDOWN_SENTINEL = object()


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


DEFAULT_QUEUE_MAXSIZE = _env_int("OCR_QUEUE_MAXSIZE", default=64, minimum=1)
DEFAULT_WORKERS = _env_int("OCR_WORKERS", default=1, minimum=1)
DEFAULT_JOB_TIMEOUT_SECONDS = _env_int("OCR_JOB_TIMEOUT_SECONDS", default=600, minimum=1)


@dataclass
class OcrJob:
    request_id: str
    uploaded: list[tuple[str, bytes]]
    output_dir: Path | None
    future: asyncio.Future[dict[str, Any]]
    enqueued_at: float


ocr_queue: asyncio.Queue[OcrJob | object] | None = None
worker_tasks: list[asyncio.Task[None]] = []


@app.get("/health")
def health() -> dict[str, str]:
    queue_size = ocr_queue.qsize() if ocr_queue is not None else 0
    return {
        "status": "ok",
        "queue_size": str(queue_size),
        "workers": str(DEFAULT_WORKERS),
        "queue_maxsize": str(DEFAULT_QUEUE_MAXSIZE),
    }


@app.on_event("startup")
async def startup_ocr_workers() -> None:
    global ocr_queue
    ocr_queue = asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)
    for worker_index in range(DEFAULT_WORKERS):
        worker_tasks.append(asyncio.create_task(_ocr_worker(worker_index + 1)))


@app.on_event("shutdown")
async def shutdown_ocr_workers() -> None:
    if ocr_queue is None:
        return
    for _ in worker_tasks:
        await ocr_queue.put(WORKER_SHUTDOWN_SENTINEL)
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    worker_tasks.clear()


async def _ocr_worker(worker_id: int) -> None:
    if ocr_queue is None:
        return
    while True:
        job = await ocr_queue.get()
        try:
            if job is WORKER_SHUTDOWN_SENTINEL:
                return
            assert isinstance(job, OcrJob)
            if job.future.cancelled():
                continue

            started_at = time.monotonic()
            try:
                result = await asyncio.to_thread(
                    run_ocr_on_uploaded_files,
                    job.uploaded,
                    job.output_dir,
                )
                result["request_id"] = job.request_id
                result["queue_wait_seconds"] = round(started_at - job.enqueued_at, 3)
                result["processing_seconds"] = round(time.monotonic() - started_at, 3)
                result["worker_id"] = worker_id
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
        finally:
            ocr_queue.task_done()


def _decode_base64_content(content: str) -> bytes:
    encoded = content.strip()
    if "," in encoded and encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    encoded = "".join(encoded.split())
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_base64", "message": "File content must be valid base64."},
        ) from exc


def _suffix_from_payload(filename: str, mime_type: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    if mime_type:
        normalized_mime = mime_type.split(";", 1)[0].strip().lower()
        mime_suffixes = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "application/pdf": ".pdf",
        }
        if normalized_mime in mime_suffixes:
            return mime_suffixes[normalized_mime]
    if content.startswith(b"%PDF"):
        return ".pdf"
    return ".png"


def _pdf_pages_to_images(filename: str, content: bytes) -> list[tuple[str, bytes]]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pdf_support_missing",
                "message": "PDF input requires PyMuPDF. Install requirements.txt before running the server.",
            },
        ) from exc

    stem = Path(filename).stem or "upload"
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_pdf", "message": f"Cannot read PDF file: {filename}"},
        ) from exc

    images: list[tuple[str, bytes]] = []
    with document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            page_suffix = "" if document.page_count == 1 else f"_p{page_index + 1}"
            images.append((f"{stem}{page_suffix}.png", pixmap.tobytes("png")))
    return images


def _safe_request_id(value: Any) -> str:
    request_id = str(value or uuid.uuid4().hex[:12]).strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in request_id)
    return safe[:96] or uuid.uuid4().hex[:12]


def _file_entry(value: Any, default_filename: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"filename": default_filename, "content": value}
    if isinstance(value, dict):
        entry = dict(value)
        if not entry.get("filename") and not entry.get("name"):
            suffix = ".pdf" if entry.get("pdf") else Path(default_filename).suffix or ".png"
            entry["filename"] = f"{Path(default_filename).stem}{suffix}"
        return entry
    raise HTTPException(
        status_code=422,
        detail={"code": "invalid_file_item", "message": "File payload must be a base64 string or an object."},
    )


def _structured_payload_files(payload: dict[str, Any], request_id: str) -> list[dict[str, Any]] | None:
    has_header = "header" in payload
    transaction_value = (
        payload.get("transactions")
        if "transactions" in payload
        else payload.get("transaction")
        if "transaction" in payload
        else payload.get("transection")
        if "transection" in payload
        else payload.get("transections")
    )
    has_transactions = transaction_value is not None
    if not has_header and not has_transactions:
        return None

    files: list[dict[str, Any]] = []
    if has_header:
        files.append(_file_entry(payload["header"], f"{request_id}_header.png"))

    if has_transactions:
        transaction_items = transaction_value if isinstance(transaction_value, list) else [transaction_value]
        for index, transaction in enumerate(transaction_items, start=1):
            files.append(_file_entry(transaction, f"{request_id}_transactions_p{index}.png"))

    return files


def _normalize_payload_files(payload: dict[str, Any], request_id: str) -> list[tuple[str, bytes]]:
    raw_files = _structured_payload_files(payload, request_id)
    if raw_files is None:
        raw_files = payload.get("files")
    if raw_files is None:
        raw_files = [_file_entry(payload, f"{request_id}.png")]
    if not isinstance(raw_files, list) or not raw_files:
        raise HTTPException(
            status_code=400,
            detail={"code": "missing_files", "message": "Send a file object or a non-empty files list."},
        )

    normalized: list[tuple[str, bytes]] = []
    for index, raw_file in enumerate(raw_files, start=1):
        if isinstance(raw_file, str):
            raw_file = _file_entry(raw_file, f"{request_id}_file_{index}.png")
        if not isinstance(raw_file, dict):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_file_item", "message": "Each files item must be an object."},
            )

        content = (
            raw_file.get("content")
            or raw_file.get("base64")
            or raw_file.get("data")
            or raw_file.get("image")
            or raw_file.get("pdf")
        )
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_base64",
                    "message": "Each file must include content, base64, data, image, or pdf.",
                },
            )

        default_suffix = ".pdf" if raw_file.get("pdf") else ".png"
        filename = str(raw_file.get("filename") or raw_file.get("name") or f"upload_{index}{default_suffix}")
        mime_type = str(raw_file.get("mime_type") or raw_file.get("content_type") or "")
        if not mime_type and raw_file.get("pdf"):
            mime_type = "application/pdf"
        decoded = _decode_base64_content(content)
        suffix = _suffix_from_payload(filename, mime_type, decoded)
        if suffix not in SUPPORTED_INPUT_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsupported_file_type", "message": f"Unsupported file type for {filename}"},
            )

        safe_stem = Path(filename).stem or f"upload_{index}"
        safe_filename = Path(filename).name
        if suffix == ".pdf":
            normalized.extend(_pdf_pages_to_images(f"{safe_stem}.pdf", decoded))
        else:
            normalized.append((safe_filename if Path(safe_filename).suffix else f"{safe_stem}{suffix}", decoded))

    return normalized


def _payload_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _response_without_debug(result: dict[str, Any]) -> dict[str, Any]:
    result["debug_count"] = len(result.pop("debug", []))
    return result


def _kaggle_answer_from_result(result: dict[str, Any], request_id: str) -> dict[str, Any]:
    answer = result.get("answer")
    if isinstance(answer, dict):
        return answer

    answers = result.get("answers")
    if isinstance(answers, dict):
        selected = answers.get(request_id)
        if isinstance(selected, dict):
            return selected
        for selected in answers.values():
            if isinstance(selected, dict):
                return selected

    return _response_without_debug(result)


@app.post("/ocr")
async def ocr_base64(
    payload: Annotated[dict[str, Any], Body(description="Base64-encoded image/PDF file or files.")],
) -> dict:
    request_id = _safe_request_id(payload.get("id") or payload.get("request_id"))
    persist = _payload_bool(payload.get("persist"), default=True)
    output_dir = DEFAULT_RUNS_DIR / f"{request_id}_{uuid.uuid4().hex[:8]}" if persist else None

    uploaded = await asyncio.to_thread(_normalize_payload_files, payload, request_id)
    if ocr_queue is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "queue_not_ready", "message": "OCR queue is not ready."},
        )

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    job = OcrJob(
        request_id=request_id,
        uploaded=uploaded,
        output_dir=output_dir,
        future=future,
        enqueued_at=time.monotonic(),
    )
    try:
        ocr_queue.put_nowait(job)
    except asyncio.QueueFull as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "ocr_queue_full",
                "message": "OCR queue is full. Retry later.",
                "queue_maxsize": DEFAULT_QUEUE_MAXSIZE,
            },
        ) from exc

    try:
        result = await asyncio.wait_for(future, timeout=DEFAULT_JOB_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        future.cancel()
        raise HTTPException(
            status_code=504,
            detail={
                "code": "ocr_timeout",
                "message": f"OCR job exceeded {DEFAULT_JOB_TIMEOUT_SECONDS} seconds.",
                "id": request_id,
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "ocr_failed", "message": str(exc)},
        ) from exc

    answer = _kaggle_answer_from_result(result, request_id)
    return {"id": request_id, "answer": answer}
