from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "question_crops"
SHEETS_DIR = ROOT / "contact_sheets"


def main() -> None:
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(1, 101, 10):
        end = min(100, start + 9)
        crops = []
        for idx in range(start, end + 1):
            img = Image.open(CROPS_DIR / f"q{idx:03d}" / "question_full.png").convert("RGB")
            img = ImageOps.expand(img, border=(0, 0, 0, 12), fill="white")
            crops.append(img)
        width = max(img.width for img in crops)
        height = sum(img.height for img in crops)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for img in crops:
            sheet.paste(img, (0, y))
            y += img.height
        out = SHEETS_DIR / f"questions_{start:03d}_{end:03d}.png"
        sheet.save(out)
        print(out, sheet.size)


if __name__ == "__main__":
    main()
