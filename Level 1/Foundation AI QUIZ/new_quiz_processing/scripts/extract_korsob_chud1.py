from __future__ import annotations

import json
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / "extracted"
OUT.mkdir(parents=True, exist_ok=True)


def find_source() -> Path:
    # Avoid embedding Thai literal in shell wrappers; identify the small text PDF.
    matches = [p for p in BASE.glob("*.pdf") if p.stat().st_size == 284473]
    if not matches:
        raise FileNotFoundError("Could not locate ข้อสอบชุด1.pdf")
    return matches[0]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u200b", "")).strip()


def line_items(pdf: Path):
    doc = fitz.open(pdf)
    for page_no, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = clean("".join(s["text"] for s in spans))
                red = any(s["color"] == 16711680 for s in spans)
                x0 = min(s["bbox"][0] for s in spans)
                yield {"page": page_no, "text": text, "red": red, "x0": x0}


def parse(pdf: Path):
    rows = list(line_items(pdf))
    questions = []
    current = None
    current_opt = None
    q_re = re.compile(r"^(\d{1,3})\.\s*(.*)")
    opt_re = re.compile(r"^([a-e])\.\s*(.*)", re.I)
    numeric_opt_re = re.compile(r"^([1-5])\.\s*(.*)")

    def finish_opt():
        nonlocal current_opt
        if current is not None and current_opt is not None:
            current["options"].append(current_opt)
        current_opt = None

    for row in rows:
        text = row["text"]
        qm = q_re.match(text)
        om = opt_re.match(text)
        # Real question numbers in this PDF start around x=90. Nested lists and
        # numeric answer choices start farther right, so position prevents false
        # question splits for matching/TRUE-FALSE items.
        if qm and row["x0"] < 105:
            finish_opt()
            if current:
                questions.append(current)
            current = {
                "source": "ข้อสอบชุด1.pdf",
                "number": int(qm.group(1)),
                "question": qm.group(2).strip(),
                "options": [],
                "answer": "",
                "answer_label": "",
                "page": row["page"],
            }
            continue
        if current is None:
            continue
        if current["number"] == 75 and not om:
            nom = numeric_opt_re.match(text)
            if nom and row["x0"] >= 105:
                finish_opt()
                current_opt = {
                    "label": nom.group(1),
                    "text": nom.group(2).strip(),
                    "is_answer": row["red"],
                }
                if row["red"]:
                    current["answer_label"] = current_opt["label"]
                    current["answer"] = current_opt["text"]
                continue
        if om:
            finish_opt()
            current_opt = {
                "label": om.group(1).lower(),
                "text": om.group(2).strip(),
                "is_answer": row["red"],
            }
            if row["red"]:
                current["answer_label"] = current_opt["label"]
                current["answer"] = current_opt["text"]
            continue
        if current_opt is not None:
            current_opt["text"] = clean(current_opt["text"] + " " + text)
            if row["red"]:
                current_opt["is_answer"] = True
                current["answer_label"] = current_opt["label"]
                current["answer"] = current_opt["text"]
        else:
            current["question"] = clean(current["question"] + " " + text)

    finish_opt()
    if current:
        questions.append(current)
    return questions


def main() -> None:
    src = find_source()
    questions = parse(src)
    out = OUT / "korsob_chud1_questions.json"
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(src.name)
    print("questions", len(questions))
    print(out)
    missing = [q["number"] for q in questions if not q["answer"]]
    print("missing_answers", missing)


if __name__ == "__main__":
    main()
