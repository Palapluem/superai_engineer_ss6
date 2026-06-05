from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract(pdf_name: str, out_name: str) -> None:
    pdf = BASE / pdf_name
    out_root = ROOT / out_name
    crops = out_root / "question_crops"
    pages = out_root / "page_previews"
    contact = out_root / "contact_sheets"
    clean_dir(crops)
    clean_dir(pages)
    clean_dir(contact)

    doc = fitz.open(pdf)
    blocks = []
    for page_no, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        pix.save(pages / f"page_{page_no:03d}.png")
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") == 1:
                bbox = fitz.Rect(block["bbox"])
                # Keep real question screenshots, not tiny decorations.
                if bbox.width >= 200 and bbox.height >= 40:
                    blocks.append((page_no, bbox.y0, bbox.x0, bbox))
    blocks.sort(key=lambda x: (x[0], x[1], x[2]))

    for qid, (page_no, _y, _x, bbox) in enumerate(blocks, start=1):
        page = doc[page_no - 1]
        outdir = crops / f"q{qid:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=bbox, alpha=False)
        pix.save(outdir / "question_full.png")

    for start in range(1, len(blocks) + 1, 10):
        imgs = []
        for qid in range(start, min(len(blocks), start + 9) + 1):
            p = crops / f"q{qid:03d}" / "question_full.png"
            if p.exists():
                img = Image.open(p).convert("RGB")
                scale = min(1.0, 1180 / img.width)
                if scale != 1.0:
                    img = img.resize((int(img.width * scale), int(img.height * scale)))
                imgs.append(img)
        if not imgs:
            continue
        width = max(i.width for i in imgs)
        height = sum(i.height + 18 for i in imgs)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for img in imgs:
            sheet.paste(img, (0, y))
            y += img.height + 18
        out = contact / f"q{start:03d}_{min(len(blocks), start + 9):03d}.png"
        sheet.save(out)
        print(out)

    print(pdf_name, "blocks", len(blocks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    extract(args.pdf, args.out)


if __name__ == "__main__":
    main()
