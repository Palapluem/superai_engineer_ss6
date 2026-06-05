from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import easyocr


warnings.filterwarnings("ignore", message=".*pin_memory.*")

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "100exam_crops"
OUT = ROOT / "extracted" / "100exam_ocr"
OUT.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def image_for(qn: int) -> Path:
    return CROPS / f"q{qn:03d}" / "question_full.png"


def ocr_question(reader: easyocr.Reader, qn: int) -> dict:
    image = image_for(qn)
    raw = reader.readtext(
        str(image),
        detail=1,
        paragraph=False,
        y_ths=0.55,
        x_ths=1.2,
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
    plain = "\n".join(r["text"] for r in lines)
    return {
        "source": "100Exam_Lv1.pdf",
        "number": qn,
        "crop": str(image.relative_to(ROOT)).replace("\\", "/"),
        "text": plain,
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()

    reader = easyocr.Reader(["th", "en"], gpu=False, verbose=False)
    all_items = []
    for qn in range(args.start, args.end + 1):
        item = ocr_question(reader, qn)
        all_items.append(item)
        (OUT / f"q{qn:03d}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"q{qn:03d}: {len(item['lines'])} lines")

    combined = OUT / f"q{args.start:03d}_{args.end:03d}.json"
    combined.write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(combined)


if __name__ == "__main__":
    main()
