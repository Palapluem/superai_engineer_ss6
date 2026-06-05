from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
PDF_PATH = BASE / "SuperAI SS5_Foundation AI QUIZ.pdf"
OUT_ROOT = ROOT / "ss5"
STRIPS_DIR = OUT_ROOT / "source_strips"
CROPS_DIR = OUT_ROOT / "question_crops"
ASSETS_DIR = OUT_ROOT / "assets"
CONTACT_DIR = OUT_ROOT / "contact_sheets"


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_strips() -> list[Path]:
    clean_dir(STRIPS_DIR)
    doc = fitz.open(PDF_PATH)
    paths: list[Path] = []
    for idx, img in enumerate(doc[0].get_images(full=True), start=1):
        xref = img[0]
        data = doc.extract_image(xref)
        ext = data["ext"]
        out = STRIPS_DIR / f"strip_{idx:02d}.{ext}"
        out.write_bytes(data["image"])
        paths.append(out)
    return paths


def stitch_strips(paths: list[Path]) -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    imgs = [Image.open(path).convert("RGB") for path in paths]
    width = max(img.width for img in imgs)
    height = sum(img.height for img in imgs)
    stitched = Image.new("RGB", (width, height), "white")
    y = 0
    for img in imgs:
        stitched.paste(img, (0, y))
        y += img.height
    out = ASSETS_DIR / "ss5_full_stitched.png"
    stitched.save(out)
    return out


def detect_question_tops(stitched_path: Path) -> list[int]:
    img = cv2.imread(str(stitched_path))
    if img is None:
        raise RuntimeError(f"Failed to read {stitched_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 70, 80]), np.array([115, 255, 255]))
    mask[:, 130:] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 15 <= x <= 45 and 25 <= w <= 75 and 18 <= h <= 38:
            candidates.append((x, y, w, h))
    candidates.sort(key=lambda box: box[1])
    tops: list[int] = []
    for _x, y, _w, _h in candidates:
        if not tops or y - tops[-1] > 80:
            tops.append(y)
    return tops


def crop_questions(stitched_path: Path, tops: list[int]) -> None:
    clean_dir(CROPS_DIR)
    img = Image.open(stitched_path).convert("RGB")
    width, height = img.size
    for idx, top in enumerate(tops, start=1):
        next_top = tops[idx] if idx < len(tops) else height
        crop_top = max(0, top - 20)
        crop_bottom = min(height, next_top - 12)
        outdir = CROPS_DIR / f"q{idx:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        crop = img.crop((0, crop_top, width, crop_bottom))
        crop.save(outdir / "question_full.png")


def make_contact_sheets() -> None:
    clean_dir(CONTACT_DIR)
    for start in range(1, 101, 10):
        imgs = []
        for qid in range(start, min(100, start + 9) + 1):
            p = CROPS_DIR / f"q{qid:03d}" / "question_full.png"
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
        out = CONTACT_DIR / f"q{start:03d}_{min(100, start + 9):03d}.png"
        sheet.save(out)
        print(out)


def main() -> None:
    paths = extract_strips()
    stitched = stitch_strips(paths)
    tops = detect_question_tops(stitched)
    crop_questions(stitched, tops)
    make_contact_sheets()
    (OUT_ROOT / "detected_question_tops.txt").write_text(
        "\n".join(str(v) for v in tops), encoding="utf-8"
    )
    print(f"Extracted {len(paths)} source strips")
    print(f"Detected/cropped {len(tops)} questions")


if __name__ == "__main__":
    main()
