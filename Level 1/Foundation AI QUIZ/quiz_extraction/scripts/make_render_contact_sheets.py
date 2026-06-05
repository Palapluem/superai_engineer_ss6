from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RENDER_DIR = ROOT / "rendered_docx"
SHEET_DIR = RENDER_DIR / "contact_sheets"


def main() -> None:
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(RENDER_DIR.glob("page-*.png"))
    thumb_w = 330
    margin = 24
    label_h = 28
    cols = 3
    rows = 4
    per_sheet = cols * rows
    for sheet_idx in range(0, len(pages), per_sheet):
        chunk = pages[sheet_idx:sheet_idx + per_sheet]
        thumbs = []
        for path in chunk:
            img = Image.open(path).convert("RGB")
            scale = thumb_w / img.width
            thumb = img.resize((thumb_w, int(img.height * scale)))
            thumb = ImageOps.expand(thumb, border=1, fill="#999999")
            canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), "white")
            canvas.paste(thumb, (0, label_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 6), path.stem, fill="black")
            thumbs.append(canvas)
        cell_w = thumb_w + 2 + margin
        cell_h = max(t.height for t in thumbs) + margin
        sheet = Image.new("RGB", (cols * cell_w + margin, rows * cell_h + margin), "white")
        for i, thumb in enumerate(thumbs):
            x = margin + (i % cols) * cell_w
            y = margin + (i // cols) * cell_h
            sheet.paste(thumb, (x, y))
        start = sheet_idx + 1
        end = sheet_idx + len(chunk)
        out = SHEET_DIR / f"render_pages_{start:03d}_{end:03d}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
