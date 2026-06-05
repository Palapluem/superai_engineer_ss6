from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / "extracted"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "answer_by_category_figures"


def find_pdf() -> Path:
    matches = [p for p in BASE.glob("*.pdf") if p.stat().st_size == 692857]
    if not matches:
        raise FileNotFoundError("answer-by-category PDF not found")
    return matches[0]


def clean(text: str) -> str:
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def yellow_rects(page: fitz.Page) -> list[fitz.Rect]:
    rects = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        rect = drawing.get("rect")
        if rect and fill and fill[0] > 0.9 and fill[1] > 0.55 and fill[2] < 0.5:
            rects.append(fitz.Rect(rect))
    return rects


def line_rows(page: fitz.Page, page_no: int) -> list[dict]:
    rects = yellow_rects(page)
    rows = []
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = clean("".join(s["text"] for s in spans))
            bbox = fitz.Rect(line["bbox"])
            highlighted = any(bbox.intersects(r) for r in rects)
            colors = {s["color"] for s in spans if s["text"].strip()}
            rows.append(
                {
                    "page": page_no,
                    "text": text,
                    "x0": float(bbox.x0),
                    "y0": float(bbox.y0),
                    "y1": float(bbox.y1),
                    "highlighted": highlighted,
                    "colors": sorted(colors),
                }
            )
    rows.sort(key=lambda r: (r["page"], r["y0"], r["x0"]))
    return rows


def parse(pdf: Path) -> list[dict]:
    doc = fitz.open(pdf)
    questions = []
    current = None
    current_opt = None
    category = ""
    q_re = re.compile(r"^(\d{1,3})\)\s*(.*)")
    opt_re = re.compile(r"^([1-6])\.\s*(.*)")

    def finish_opt() -> None:
        nonlocal current_opt
        if current is not None and current_opt is not None:
            current["options"].append(current_opt)
        current_opt = None

    for page_no, page in enumerate(doc, start=1):
        for row in line_rows(page, page_no):
            text = row["text"]
            # Section headings are colored and appear before a question on each page.
            if row["colors"] != [0] and not q_re.match(text):
                category = text
                continue

            qm = q_re.match(text)
            if qm and row["x0"] < 90:
                finish_opt()
                if current:
                    questions.append(current)
                current = {
                    "source": "[เฉลย] ข้อสอบ แยกหมวด.pdf",
                    "number": int(qm.group(1)),
                    "source_id": f"CAT-{int(qm.group(1)):03d}",
                    "category": category,
                    "question": qm.group(2).strip(),
                    "options": [],
                    "answer_label": "",
                    "answer": "",
                    "answer_note": "",
                    "highlighted_body": [],
                    "page": page_no,
                }
                continue
            if current is None:
                continue

            om = opt_re.match(text)
            if om and row["x0"] >= 80:
                finish_opt()
                current_opt = {
                    "label": om.group(1),
                    "text": om.group(2).strip(),
                    "is_answer": row["highlighted"],
                }
                if row["highlighted"]:
                    current["answer_label"] = current_opt["label"]
                    current["answer"] = current_opt["text"]
                continue

            if current_opt is not None and (row["x0"] >= 80 or row["highlighted"]):
                current_opt["text"] = clean(current_opt["text"] + " " + text)
                if row["highlighted"]:
                    current_opt["is_answer"] = True
                    current["answer_label"] = current_opt["label"]
                    current["answer"] = current_opt["text"]
            else:
                if row["highlighted"]:
                    current["highlighted_body"].append(text)
                current["question"] = clean(current["question"] + " " + text)

    finish_opt()
    if current:
        questions.append(current)
    for q in questions:
        if not q["answer"] and q.get("highlighted_body"):
            q["answer_label"] = "text"
            q["answer"] = clean(" ".join(q["highlighted_body"]))
            q["answer_note"] = "answer highlighted in body text, not a numbered option"
        if q["number"] == 13 and not q["answer"]:
            q["answer_label"] = "2"
            q["answer"] = "Random Forest"
            q["answer_note"] = "source page has no highlight for this item; filled from standard Bagging-method answer"
            for opt in q["options"]:
                if opt["label"] == "2":
                    opt["is_answer"] = True
        if q["number"] == 55 and not q["answer"]:
            q["answer_label"] = "unknown"
            q["answer"] = "ไม่มีเฉลยในไฟล์ [เฉลย] ข้อสอบ แยกหมวด.pdf (source ระบุว่า 'ข้ามก่อน')"
            q["answer_note"] = "source skipped this image-XOR question"
    return questions


def add_figure_crops(pdf: Path, questions: list[dict]) -> None:
    if FIG_DIR.exists():
        shutil.rmtree(FIG_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    by_page: dict[int, list[dict]] = {}
    for q in questions:
        by_page.setdefault(q["page"], []).append(q)
        q["figures"] = []

    for page_no, qs in by_page.items():
        page = doc[page_no - 1]
        rows = line_rows(page, page_no)
        starts = []
        for q in qs:
            pat = re.compile(rf"^{q['number']}\)")
            y_candidates = [r["y0"] for r in rows if pat.match(r["text"]) and r["x0"] < 90]
            if y_candidates:
                starts.append((q, y_candidates[0]))
        starts.sort(key=lambda x: x[1])
        image_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 1]
        for block in image_blocks:
            bbox = fitz.Rect(block["bbox"])
            owner = None
            for idx, (q, y0) in enumerate(starts):
                y1 = starts[idx + 1][1] if idx + 1 < len(starts) else page.rect.y1
                if y0 <= bbox.y0 < y1:
                    owner = q
                    break
            if owner is None:
                continue
            outdir = FIG_DIR / f"q{owner['number']:03d}"
            outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / f"figure_{len(owner['figures']) + 1}.png"
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=bbox, alpha=False)
            pix.save(out)
            owner["figures"].append(str(out.relative_to(ROOT)).replace("\\", "/"))


def main() -> None:
    pdf = find_pdf()
    questions = parse(pdf)
    add_figure_crops(pdf, questions)
    out = OUT / "answer_by_category_questions.json"
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [q["number"] for q in questions if not q["answer"]]
    print(pdf.name)
    print("questions", len(questions))
    print("missing_answers", missing)
    print(out)


if __name__ == "__main__":
    main()
