import json
from pathlib import Path
import re

# Paths
fahmai_dir = Path(r"d:\superai_engineer_ss6\Level 2\Hackathon 3_Fahmai Telephone Directory")
target_nb_path = Path(r"d:\superai_engineer_ss6\Level 2\Hackathon 3_Fahmai Telephone Directory\Hackathon 3_LV2_Fahmai Telephone Directory_Agentic_Detailed.ipynb")

def get_file_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def split_content(content, split_markers):
    blocks = []
    lines = content.split('\n')
    current_block = []
    for line in lines:
        match_found = False
        for marker, title, desc in split_markers:
            if re.search(marker, line):
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
                match_found = True
                break
        if not match_found:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)
    return blocks

# Professional Stage Definitions (No Emojis)
main_splits = [
    (r"# STAGE 1", "Stage 1: Intent Routing and Security Implementation", 
     "ส่วนงานวิเคราะห์เจตนาและรักษาความปลอดภัยเชิงรุก โดยใช้การคัดกรองด้วยนิพจน์เรกูลาร์ (Regular Expressions) เพื่อตรวจสอบและปฏิเสธคำถามที่ละเมิดนโยบายความเป็นส่วนตัวหรือความพยายามในการโจมตีระบบ (Prompt Injection) ก่อนเข้าสู่กระบวนการประมวลผลหลัก"),
    (r"# STAGE 2 — Deterministic Fast-Path", "Stage 2: Deterministic Data Retrieval Architecture", 
     "กลไกการดึงข้อมูลแบบมีเงื่อนไขตายตัว (Fast-Path) ออกแบบมาเพื่อจัดการคำถามที่ระบุตัวตนพนักงานหรือโครงสร้างองค์กรอย่างชัดเจน โดยใช้กระบวนการค้นหาผ่านดัชนีข้อมูลในหน่วยความจำเพื่อป้องกันความเสี่ยงในการเกิดความผิดพลาดของข้อมูล (Hallucination)"),
    (r"# Formatters", "Stage 2 Methodology: Data Normalization and Formatting", 
     "กระบวนการจัดเตรียมและจัดรูปแบบข้อมูลผลลัพธ์ให้มีความคงเส้นคงวาตามมาตรฐานที่กำหนด โดยรองรับการจัดการข้อมูลพนักงานทั้งในรูปแบบรายบุคคลและรูปแบบรายการ ตามบริบทของคำถามทั้งภาษาไทยและภาษาอังกฤษ"),
    (r"def stage2_fast_path", "Stage 2 Execution: High-Precision Retrieval Logic", 
     "ส่วนการประมวลผลตรรกะการค้นหาข้อมูลแบบตรงตัว หากระบบตรวจพบความสอดคล้องของข้อมูลในส่วนนี้ จะดำเนินการส่งออกคำตอบทันทีเพื่อประสิทธิภาพสูงสุดในการใช้ทรัพยากรประมวลผล"),
    (r"# STAGE 3 — Typhoon Planner", "Stage 3: LLM-Based Dynamic Query Planning", 
     "ในกรณีที่คำถามมีความซับซ้อนเชิงตรรกะสูง ระบบจะใช้แบบจำลองภาษาขนาดใหญ่ (LLM) ทำหน้าที่วิเคราะห์และสร้างแผนผังการค้นหาข้อมูล (Pandas Query) เพื่อนำไปประมวลผลกับชุดข้อมูลจริง แทนการวิเคราะห์คำตอบจากความจำส่วนตัวของโมเดล"),
    (r"def stage4_answerer", "Stage 4: Contextual Synthesis and Answer Generation", 
     "ขั้นตอนการสังเคราะห์คำตอบจากชุดข้อมูลจริงที่ได้รับจากการประมวลผลในขั้นตอนก่อนหน้า โดยใช้แบบจำลองภาษาในการสรุปประเด็นให้มีความกระชับและตรงตามบริบทของคำถามต้นฉบับ"),
    (r"# Orchestrator", "System Orchestration and Pipeline Management", 
     "ส่วนการบริหารจัดการลำดับขั้นตอนการประมวลผล (Pipeline Orchestration) ตั้งแต่ขั้นตอนเริ่มต้นจนถึงการส่งออกข้อมูล โดยทำหน้าที่ควบคุมการไหลของข้อมูลระหว่าง Stages ต่างๆ อย่างเป็นระบบ"),
    (r"# STAGE 5 helpers", "Stage 5: Final Compliance and PII Security Guard", 
     "การตรวจสอบความถูกต้องและนโยบายความปลอดภัยขั้นสุดท้ายก่อนการส่งออกคำตอบ โดยทำหน้าที่ลบข้อมูลระบุตัวตน (Personally Identifiable Information) และตรวจสอบความถูกต้องของรูปแบบคำตอบให้เป็นไปตามมาตรฐานที่กำหนด"),
    (r"# Main loop", "Inference Implementation and Batch Processing", 
     "กระบวนการประมวลผลชุดคำถามทดสอบทั้งหมดแบบกลุ่ม (Batch Processing) พร้อมระบบบริหารจัดการข้อผิดพลาดและบันทึกการทำงาน เพื่อความเสถียรในการประมวลผลข้อมูลขนาดใหญ่")
]

cells = []

# ── Professional Header ──
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# [Super AI Engineer Season 6] Mini Hackathon 3 Level 2\n",
        "## FahMai Telephone Directory Pipeline: Agentic Detailed Implementation\n",
        "\n",
        "**Project Overview**\n",
        "- **Dataset:** FahMai Telephone Directory (1,995 employees)\n",
        "- **Architecture:** 5-Stage Hybrid Deterministic Pipeline\n",
        "\n",
        "---\n",
        "\n",
        "### 5-Stage Architecture Framework\n",
        "\n",
        "| Stage | Classification | Primary Objective | Methodology |\n",
        "|---|---|---|---|\n",
        "| Stage 1 | Intent Router | Security screening and intent classification | Regex & Pattern Matching |\n",
        "| Stage 2 | Fast-Path | High-precision factual data retrieval | Pandas Indexed Search |\n",
        "| Stage 3 | Query Planner | Dynamic query generation for complex requests | LLM Synthesis |\n",
        "| Stage 4 | Answerer | Data summarization and context matching | LLM Natural Language Generation |\n",
        "| Stage 5 | Compliance | Security auditing and PII removal | Automated Security Guard |\n",
        "\n",
        "---\n",
        "\n",
        "### Engineering Strategy and Methodology\n",
        "โครงสร้างระบบถูกออกแบบภายใต้หลักการ 'Deterministic Priority' โดยมุ่งเน้นการใช้ตรรกะทางโปรแกรมในการจัดการข้อมูลข้อเท็จจริง (Factual Data) เป็นอันดับแรก และประยุกต์ใช้แบบจำลองภาษาขนาดใหญ่ (LLM) เฉพาะในส่วนงานที่ต้องการความยืดหยุ่นสูง เพื่อให้มั่นใจในความถูกต้องของข้อมูลและความปลอดภัยของระบบในระดับสูงสุด\n",
        "\n",
        "---"
    ]
})

# ── Process main.py ──
main_content = get_file_content(fahmai_dir / "main.py")
main_blocks = split_content(main_content, main_splits)

# ── 0. Imports ──
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# System Environment and Initialization\n", "*การกำหนดค่าสภาพแวดล้อมและการเตรียมความพร้อมของระบบประมวลผล*"]
})
source_lines = [line + '\n' for line in main_blocks[0]]
if source_lines and source_lines[-1] == '\n': source_lines.pop()
cells.append({"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": source_lines})

# ── Stages ──
for block in main_blocks[1:]:
    block_title = ""
    block_desc = ""
    for marker, title, desc in main_splits:
        if any(re.search(marker, line) for line in block):
            block_title = title
            block_desc = desc
            break
    if block_title:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {block_title}\n", f"### {block_desc}"]
        })
    source_lines = [line + '\n' for line in block]
    if source_lines and source_lines[-1] == '\n': source_lines.pop()
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": source_lines})

# ── config & database ──
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# Global Security Configuration and Policy Definition\n", "*การกำหนดนโยบายความปลอดภัยและค่าคงที่สำหรับการบริหารจัดการระบบ*"]
})
config_content = get_file_content(fahmai_dir / "config.py")
source_lines = [line + '\n' for line in config_content.split('\n')]
if source_lines and source_lines[-1] == '\n': source_lines.pop()
cells.append({"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": source_lines})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# Optimized Data Management Engine\n", "*การบริหารจัดการชุดข้อมูลและการทำดัชนีเพื่อประสิทธิภาพในการสืบค้น*"]
})
db_content = get_file_content(fahmai_dir / "database.py")
source_lines = [line + '\n' for line in db_content.split('\n')]
if source_lines and source_lines[-1] == '\n': source_lines.pop()
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": source_lines})

# Write
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "nbformat": 4, "nbformat_minor": 4
}
with open(target_nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Success: {target_nb_path}")
