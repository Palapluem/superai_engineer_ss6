# Codex Skills For Hackathon 9_5

ติดตั้งแล้วจาก `thananon/9arm-skills`:

- `debug-mantra` - ใช้ตอน debug bug, error, failing test, API พัง, stack trace, หรือผลลัพธ์ไม่ตรง
- `scrutinize` - ใช้ก่อนส่งงาน เพื่อตรวจแผน, diff, PR, implementation, edge cases
- `post-mortem` - ใช้หลังแก้ bug สำเร็จและ validate แล้ว เพื่อเขียน RCA / root cause analysis
- `management-talk` - ใช้แปลงข้อความ technical ให้เป็น Slack, standup, email, หรือ summary สำหรับ PM/leadership

## Where They Are

- Codex auto-discovery: `C:\Users\CPE KMUTT\.codex\skills`
- Local exam copy: `Level 2\Hackathon 9_5 Domains Hackathon\codex-skills`

Restart Codex after installation so the new skills are picked up automatically.

## How To Use

เรียกชื่อ skill ใน prompt ตรง ๆ ได้เลย:

```text
ใช้ $debug-mantra ช่วย debug error นี้ให้หน่อย
```

```text
ใช้ $scrutinize review โค้ดที่แก้ล่าสุดก่อน submit
```

```text
ใช้ $post-mortem เขียน RCA จาก bug ที่เพิ่งแก้เสร็จ
```

```text
ใช้ $management-talk แปลง technical update นี้เป็น Slack update ให้ PM
```

ถ้าไม่อยากจำ `$` ให้พิมพ์ชื่อ skill เป็นภาษาไทยปนอังกฤษก็ได้ เช่น:

```text
ใช้ skill debug-mantra ตรวจ API ที่ fail ให้หน่อย
```

## Practical Examples

### 1. Debug API ที่รันไม่ผ่าน

ใช้เมื่อ API start ไม่ขึ้น, smoke test fail, response format ผิด, latency แปลก, หรือ log มี error

```text
ใช้ $debug-mantra ช่วย debug API นี้ให้หน่อย

อาการ:
- endpoint /predict ตอบ 500
- command ที่ใช้รันคือ python api.py
- smoke test คือ python check_api_ready.py

ช่วยทำตามขั้นตอน:
1. หา repro ที่รันซ้ำได้
2. trace ว่า fail ตรงไหน
3. สรุป hypothesis และวิธี falsify
4. แก้เฉพาะจุดที่จำเป็น แล้ว validate ให้ด้วย
```

ถ้ามี log ให้ paste ต่อท้ายได้เลย:

```text
ใช้ $debug-mantra วิเคราะห์ log นี้และแก้โค้ดให้หน่อย

[paste log ตรงนี้]
```

### 2. Review ก่อนส่งงาน

ใช้หลังแก้โค้ดเสร็จ แต่ก่อน submit หรือก่อนยิง evaluation

```text
ใช้ $scrutinize ตรวจงานล่าสุดก่อนส่ง

ช่วยดู:
- มี bug หรือ edge case ที่น่าพังไหม
- behavior ตรงกับโจทย์ไหม
- มีไฟล์ไหนที่แก้เกินจำเป็นไหม
- test หรือ validation ที่ควรรันคืออะไร
```

ถ้าอยากให้ตรวจเฉพาะไฟล์:

```text
ใช้ $scrutinize review ไฟล์ api.py และ check_api_ready.py
เน้น correctness, runtime error, input/output contract, และจุดที่อาจ fail ตอน deploy
```

### 3. สรุปให้ทีม/กรรมการ/PM

ใช้เมื่อต้องเขียนข้อความสั้น ๆ ที่ไม่ technical เกินไป

```text
ใช้ $management-talk แปลง update นี้เป็น Slack update สั้น ๆ ให้ PM

Technical update:
- fixed OCR API crash caused by missing model path fallback
- added readiness check
- validated with 10 sample images
- remaining risk: low confidence on handwritten edge cases
```

### 4. เขียน RCA หลังแก้ bug เสร็จ

ใช้หลังมีครบ 4 อย่าง: repro, root cause, fix, validation

```text
ใช้ $post-mortem เขียน RCA จากข้อมูลนี้

Repro:
- run python check_api_ready.py แล้ว /predict ตอบ 500

Root cause:
- model path config ชี้ไป folder เก่า ทำให้โหลด weights ไม่เจอ

Fix:
- เพิ่ม fallback path และ error message ตอน model missing

Validation:
- rerun check_api_ready.py ผ่าน
- tested 10 sample requests สำเร็จ
```

## Recommended Exam Flow

1. เปิด Codex จากโฟลเดอร์สอบ:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 9_5 Domains Hackathon"
```

2. เวลาเจอ error หรือผลลัพธ์แปลก ให้เริ่มด้วย:

```text
ใช้ $debug-mantra ช่วยทำ repro และหา root cause จาก log/ไฟล์นี้
```

3. หลังแก้โค้ดแล้ว ก่อน submit ให้ใช้:

```text
ใช้ $scrutinize ตรวจ changes ล่าสุดทั้งหมดก่อนส่งงาน
```

4. ถ้าต้องส่งสรุปหรืออธิบายให้คนอื่น:

```text
ใช้ $management-talk ทำสรุปแบบสั้นสำหรับ Slack/standup
```

5. ถ้าต้องเขียน RCA หลังแก้ bug:

```text
ใช้ $post-mortem เขียน root cause analysis จาก repro, cause, fix, validation
```

## Notes For Claude Skills In Codex

Codex ใช้ skill ที่มีโครง `SKILL.md` พร้อม YAML frontmatter `name` และ `description` ได้เหมือนกัน ถ้า skill ไม่มีคำสั่งเฉพาะ Claude หรือ script ที่ผูกกับ `~/.claude` โดยตรง

สำหรับ repo นี้ skill ทั้ง 4 ตัวมีแค่ `SKILL.md` ไม่มี script ที่ต้องรัน จึงปรับใช้กับ Codex ได้ตรง ๆ

Source:

- https://github.com/thananon/9arm-skills
- https://skillsmp.com/
