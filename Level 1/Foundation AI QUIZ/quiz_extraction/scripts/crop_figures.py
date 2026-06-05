from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "question_crops"


# Coordinates are in each question_full.png pixel space: left, top, right, bottom.
FIGURE_BOXES = {
    20: (90, 38, 660, 235),
    74: (85, 68, 510, 225),
    94: (120, 55, 820, 435),
    95: (85, 55, 585, 235),
}


def main() -> None:
    for qid, box in FIGURE_BOXES.items():
        src = CROPS_DIR / f"q{qid:03d}" / "question_full.png"
        img = Image.open(src).convert("RGB")
        out = img.crop(box)
        dest = src.parent / "figure_1.png"
        out.save(dest)
        print(dest, out.size)


if __name__ == "__main__":
    main()
