from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import cv2
import easyocr
import numpy as np


warnings.filterwarnings("ignore", message=".*pin_memory.*")

ROOT = Path(__file__).resolve().parents[1]


def clean(text: str) -> str:
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_selected_option(image: Path) -> dict:
    img = cv2.imread(str(image))
    if img is None:
        return {"selected_index": None, "scores": []}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (gray < 210).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / max(h, 1)
        if 0 <= x <= 75 and 8 <= w <= 22 and 8 <= h <= 22 and 0.75 <= ratio <= 1.35 and y > 30:
            cx = x + w // 2
            cy = y + h // 2
            patch = gray[max(0, cy - 5) : cy + 6, max(0, cx - 5) : cx + 6]
            hpatch = hsv[max(0, cy - 6) : cy + 7, max(0, cx - 6) : cx + 7]
            blue = int(((hpatch[:, :, 0] >= 85) & (hpatch[:, :, 0] <= 120) & (hpatch[:, :, 1] > 60) & (hpatch[:, :, 2] > 80)).sum())
            dark = int((patch < 160).sum())
            circles.append({"x": x, "y": y, "cy": cy, "blue": blue, "dark": dark, "mean": float(patch.mean())})
    circles.sort(key=lambda r: r["cy"])
    if not circles:
        return {"selected_index": None, "scores": []}
    best_i = max(range(len(circles)), key=lambda i: (circles[i]["blue"], circles[i]["dark"]))
    best = circles[best_i]
    selected = best_i + 1 if best["blue"] >= 20 or best["dark"] >= 18 else None
    return {"selected_index": selected, "scores": circles}


def ocr_question(reader: easyocr.Reader, exam: str, qn: int) -> dict:
    image = ROOT / exam / "question_crops" / f"q{qn:03d}" / "question_full.png"
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
        "source": f"{exam.upper()}.pdf",
        "number": qn,
        "crop": str(image.relative_to(ROOT)).replace("\\", "/"),
        "selected_index": selected["selected_index"],
        "selected_scores": selected["scores"],
        "text": "\n".join(r["text"] for r in lines),
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", required=True, choices=["exam02", "exam03"])
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()

    out_dir = ROOT / "extracted" / f"{args.exam}_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["th", "en"], gpu=False, verbose=False)
    items = []
    for qn in range(args.start, args.end + 1):
        item = ocr_question(reader, args.exam, qn)
        items.append(item)
        (out_dir / f"q{qn:03d}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{args.exam} q{qn:03d}: {len(item['lines'])} lines selected={item['selected_index']}")
    combined = out_dir / f"q{args.start:03d}_{args.end:03d}.json"
    combined.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(combined)


if __name__ == "__main__":
    main()
