from __future__ import annotations

import shutil
from pathlib import Path

import fitz
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "outputs" / "Foundation_AI_QUIZ_100_Review.docx"
OUT_DIR = ROOT / "rendered_docx"
PDF = OUT_DIR / "Foundation_AI_QUIZ_100_Review.pdf"


def export_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PDF.exists():
        PDF.unlink()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(str(DOCX.resolve()), ReadOnly=True)
        doc.ExportAsFixedFormat(str(PDF.resolve()), 17)  # wdExportFormatPDF
    finally:
        if doc is not None:
            doc.Close(False)
        word.Quit()


def render_pages() -> None:
    for path in OUT_DIR.glob("page-*.png"):
        path.unlink()
    doc = fitz.open(PDF)
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        pix.save(OUT_DIR / f"page-{idx:03d}.png")
    print(f"PDF: {PDF}")
    print(f"Rendered pages: {doc.page_count}")


def main() -> None:
    if not DOCX.exists():
        raise FileNotFoundError(DOCX)
    export_pdf()
    render_pages()


if __name__ == "__main__":
    main()
