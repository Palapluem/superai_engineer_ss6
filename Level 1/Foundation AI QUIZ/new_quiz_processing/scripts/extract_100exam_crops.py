from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
PDF = BASE / "100Exam_Lv1.pdf"
PAGES = ROOT / "100exam_pages"
CROPS = ROOT / "100exam_crops"
CONTACT = ROOT / "100exam_contact_sheets"


def clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def render_pages() -> list[Path]:
    clean(PAGES)
    doc = fitz.open(PDF)
    paths: list[Path] = []
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        out = PAGES / f"page_{idx:03d}.png"
        pix.save(out)
        paths.append(out)
    return paths


def detect_labels(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue labels like "ข้อที่ N".
    mask = cv2.inRange(hsv, np.array([85, 60, 80]), np.array([115, 255, 255]))
    mask[:, int(img.shape[1] * 0.55):] = 0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 30 <= w <= 95 and 25 <= h <= 60 and x < img.shape[1] * 0.35:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    dedup = []
    for box in boxes:
        if not dedup or abs(box[1] - dedup[-1][1]) > 40:
            dedup.append(box)
    return dedup


def crop_questions(paths: list[Path]) -> None:
    clean(CROPS)
    qid = 1
    for page_path in paths:
        img = cv2.imread(str(page_path))
        boxes = detect_labels(img)
        pil = Image.open(page_path).convert("RGB")
        w, h = pil.size
        for i, (x, y, bw, bh) in enumerate(boxes):
            next_y = boxes[i + 1][1] if i + 1 < len(boxes) else h - 70
            top = max(0, y - 35)
            bottom = min(h, next_y - 18)
            left = max(0, x - 65)
            # Crop only the question/choices area, excluding the right note grid.
            right = min(w, int(w * 0.62))
            crop = pil.crop((left, top, right, bottom))
            outdir = CROPS / f"q{qid:03d}"
            outdir.mkdir(parents=True, exist_ok=True)
            crop.save(outdir / "question_full.png")
            qid += 1
    print("cropped", qid - 1)


def make_contact_sheets() -> None:
    clean(CONTACT)
    for start in range(1, 101, 10):
        imgs = []
        for qid in range(start, min(100, start + 9) + 1):
            p = CROPS / f"q{qid:03d}" / "question_full.png"
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
        out = CONTACT / f"q{start:03d}_{min(100, start + 9):03d}.png"
        sheet.save(out)
        print(out)


def main() -> None:
    paths = render_pages()
    crop_questions(paths)
    make_contact_sheets()


if __name__ == "__main__":
    main()
