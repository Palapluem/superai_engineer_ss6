from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import cv2
import easyocr


warnings.filterwarnings("ignore", message=".*pin_memory.*")

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "ss5" / "question_crops"
OUT = ROOT / "extracted" / "ss5_ocr"
OUT.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_selected_option(image: Path) -> dict:
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"selected_index": None, "scores": []}
    mask = (img < 190).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 30 <= x <= 65 and 9 <= w <= 18 and 9 <= h <= 18 and y > 40:
            cx = x + w // 2
            cy = y + h // 2
            patch = img[max(0, cy - 4) : cy + 5, max(0, cx - 4) : cx + 5]
            dark = int((patch < 120).sum())
            circles.append({"y": cy, "dark": dark, "mean": float(patch.mean())})
    circles.sort(key=lambda r: r["y"])
    if not circles:
        return {"selected_index": None, "scores": []}
    best_i = max(range(len(circles)), key=lambda i: circles[i]["dark"])
    selected = best_i + 1 if circles[best_i]["dark"] >= 12 else None
    return {"selected_index": selected, "scores": circles}


def ocr_question(reader: easyocr.Reader, qn: int) -> dict:
    image = CROPS / f"q{qn:03d}" / "question_full.png"
    raw = reader.readtext(
        str(image),
        detail=1,
        paragraph=False,
        y_ths=0.55,
        x_ths=1.0,
        text_threshold=0.5,
        low_text=0.25,
    )
    lines = []
    for box, text, conf in raw:
        if not text.strip():
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append(
            {
                "text": clean(text),
                "conf": float(conf),
                "x0": min(xs),
                "y0": min(ys),
                "x1": max(xs),
                "y1": max(ys),
            }
        )
    lines.sort(key=lambda r: (r["y0"], r["x0"]))
    selected = detect_selected_option(image)
    return {
        "source": "SuperAI SS5_Foundation AI QUIZ.pdf",
        "number": qn,
        "crop": str(image.relative_to(ROOT)).replace("\\", "/"),
        "selected_index": selected["selected_index"],
        "selected_scores": selected["scores"],
        "text": "\n".join(r["text"] for r in lines),
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()

    reader = easyocr.Reader(["th", "en"], gpu=False, verbose=False)
    items = []
    for qn in range(args.start, args.end + 1):
        item = ocr_question(reader, qn)
        items.append(item)
        (OUT / f"q{qn:03d}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"q{qn:03d}: {len(item['lines'])} lines selected={item['selected_index']}")
    combined = OUT / f"q{args.start:03d}_{args.end:03d}.json"
    combined.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(combined)


if __name__ == "__main__":
    main()
