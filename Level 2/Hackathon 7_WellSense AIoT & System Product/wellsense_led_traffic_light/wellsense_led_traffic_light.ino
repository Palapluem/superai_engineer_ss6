/*
 * ============================================================
 *  WellSense AIoT — LED Traffic Light System
 *  Board   : Arduino UNO Q (หรือ Uno R4 WiFi/Minima)
 *  Purpose : รับ risk_level จาก Host PC ผ่าน Serial
 *            แล้วแสดงสัญญาณไฟจราจร 3 สี
 *
 *  WIRING (ต่อสายดังนี้):
 *    Pin 9  → Resistor 220Ω → LED แดง   → GND
 *    Pin 10 → Resistor 220Ω → LED เหลือง → GND
 *    Pin 11 → Resistor 220Ω → LED เขียว  → GND
 *    Pin 8  → Buzzer (+)                  → GND  [optional]
 *
 *  SERIAL PROTOCOL (Host PC → UNO Q):
 *    FORMAT: <risk_level>,<risk_score>,<impact_event>\n
 *    ตัวอย่าง:
 *      "low,0.12,0.05\n"     → ไฟเขียว
 *      "medium,0.48,0.10\n"  → ไฟเหลืองกะพริบช้า
 *      "high,0.82,0.50\n"    → ไฟแดงกะพริบเร็ว
 *      "high,0.91,0.98\n"    → ไฟแดง + Buzzer (impact event)
 *
 *  RISK LEVEL ที่ Host PC ต้องส่งมา:
 *    "low"    → risk_score < 0.35
 *    "medium" → 0.35 ≤ risk_score < 0.65
 *    "high"   → risk_score ≥ 0.65
 * ============================================================
 */

// ─── Pin Definitions ───────────────────────────────────────
#define PIN_LED_RED     9
#define PIN_LED_YELLOW  10
#define PIN_LED_GREEN   11
#define PIN_BUZZER      8   // ถ้าไม่มี Buzzer ให้ comment บรรทัดที่ใช้ BUZZER ออก

// ─── Timing Constants ─────────────────────────────────────
#define BLINK_SLOW_MS     500   // เหลือง: กะพริบทุก 500ms (1 Hz)
#define BLINK_FAST_MS     125   // แดง   : กะพริบทุก 125ms (4 Hz)
#define IMPACT_DURATION_MS 5000 // ไฟแดงบังคับ 5 วินาทีเมื่อ impact สูง
#define BUZZER_BEEP_MS    150   // ความยาว beep ของ buzzer
#define SERIAL_TIMEOUT_MS 3000  // ถ้าไม่ได้รับข้อมูลนาน 3 วิ → ไฟเหลือง warning

// ─── Risk Level Enum ──────────────────────────────────────
enum RiskLevel {
  RISK_UNKNOWN = -1,
  RISK_LOW     = 0,   // 🟢 เขียว
  RISK_MEDIUM  = 1,   // 🟡 เหลือง
  RISK_HIGH    = 2    // 🔴 แดง
};

// ─── Global State ─────────────────────────────────────────
RiskLevel currentRisk     = RISK_UNKNOWN;
float     currentScore    = 0.0;
float     impactEvent     = 0.0;
bool      impactOverride  = false;       // บังคับไฟแดงชั่วคราว
unsigned long impactStartMs = 0;
unsigned long lastSerialMs  = 0;
unsigned long lastBlinkMs   = 0;
bool          ledState       = false;    // สถานะ LED ปัจจุบัน (สำหรับ blink)
String        serialBuffer   = "";

// ─── Setup ────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);

  pinMode(PIN_LED_RED,    OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_GREEN,  OUTPUT);
  pinMode(PIN_BUZZER,     OUTPUT);

  // ทดสอบไฟตอนเริ่ม (startup sequence)
  startupSequence();

  lastSerialMs = millis();
  Serial.println("[WellSense] LED Traffic Light ready. Waiting for data...");
}

// ─── Main Loop ────────────────────────────────────────────
void loop() {
  // 1. อ่านข้อมูลจาก Serial (non-blocking)
  readSerialData();

  // 2. เช็ค impact override หมดเวลาหรือยัง
  if (impactOverride && (millis() - impactStartMs >= IMPACT_DURATION_MS)) {
    impactOverride = false;
    Serial.println("[WellSense] Impact override ended.");
  }

  // 3. เช็ค timeout (ถ้าไม่มีข้อมูลนาน → warning)
  if (millis() - lastSerialMs >= SERIAL_TIMEOUT_MS) {
    // ไม่มีข้อมูลนานเกินไป → แสดงไฟเหลืองเตือน
    blinkLED(PIN_LED_YELLOW, BLINK_SLOW_MS);
    allLedsOff();
    return;
  }

  // 4. แสดงไฟตาม risk level ปัจจุบัน
  updateLedDisplay();
}

// ─── Read Serial (Non-blocking) ───────────────────────────
void readSerialData() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      // ได้รับ 1 บรรทัดครบแล้ว → parse
      serialBuffer.trim();
      if (serialBuffer.length() > 0) {
        parseMessage(serialBuffer);
        lastSerialMs = millis();
      }
      serialBuffer = "";
    } else {
      serialBuffer += c;
    }
  }
}

// ─── Parse Message ─────────────────────────────────────────
// FORMAT: "low,0.12,0.05"  หรือ  "medium,0.48,0.10"  หรือ  "high,0.82,0.90"
void parseMessage(String msg) {
  // แยก field ด้วย comma
  int firstComma  = msg.indexOf(',');
  int secondComma = msg.indexOf(',', firstComma + 1);

  if (firstComma < 0) {
    Serial.println("[WellSense] ERROR: Invalid format. Expected: risk_level,score,impact");
    return;
  }

  String levelStr = msg.substring(0, firstComma);
  levelStr.trim();
  levelStr.toLowerCase();

  // Parse risk level
  if (levelStr == "low") {
    currentRisk = RISK_LOW;
  } else if (levelStr == "medium") {
    currentRisk = RISK_MEDIUM;
  } else if (levelStr == "high") {
    currentRisk = RISK_HIGH;
  } else {
    Serial.print("[WellSense] ERROR: Unknown risk level: ");
    Serial.println(levelStr);
    return;
  }

  // Parse risk score (optional)
  if (secondComma > firstComma) {
    String scoreStr  = msg.substring(firstComma + 1, secondComma);
    String impactStr = msg.substring(secondComma + 1);
    currentScore = scoreStr.toFloat();
    impactEvent  = impactStr.toFloat();

    // ตรวจ impact override
    if (impactEvent >= 0.8 || currentScore >= 0.90) {
      triggerImpactAlert();
    }
  }

  // Debug log
  Serial.print("[WellSense] RISK=");
  Serial.print(levelStr);
  Serial.print(" | SCORE=");
  Serial.print(currentScore, 3);
  Serial.print(" | IMPACT=");
  Serial.println(impactEvent, 3);
}

// ─── Update LED Display ───────────────────────────────────
void updateLedDisplay() {
  // Impact override บังคับไฟแดงกะพริบเร็ว
  if (impactOverride) {
    blinkLED(PIN_LED_RED, BLINK_FAST_MS);
    return;
  }

  switch (currentRisk) {
    case RISK_LOW:
      // 🟢 เขียว: ติดค้าง ปลอดภัย
      allLedsOff();
      digitalWrite(PIN_LED_GREEN, HIGH);
      break;

    case RISK_MEDIUM:
      // 🟡 เหลือง: กะพริบช้า — เฝ้าระวัง
      allLedsOff();
      blinkLED(PIN_LED_YELLOW, BLINK_SLOW_MS);
      break;

    case RISK_HIGH:
      // 🔴 แดง: กะพริบเร็ว — อันตราย
      allLedsOff();
      blinkLED(PIN_LED_RED, BLINK_FAST_MS);
      break;

    case RISK_UNKNOWN:
    default:
      // ยังไม่รู้ risk → เหลืองกะพริบช้า (standby warning)
      allLedsOff();
      blinkLED(PIN_LED_YELLOW, BLINK_SLOW_MS);
      break;
  }
}

// ─── Blink Helper (Non-blocking) ─────────────────────────
// เรียกซ้ำใน loop() — ไม่ใช้ delay()
void blinkLED(int pin, unsigned long intervalMs) {
  unsigned long now = millis();
  if (now - lastBlinkMs >= intervalMs) {
    lastBlinkMs = now;
    ledState = !ledState;
    digitalWrite(pin, ledState ? HIGH : LOW);
  }
}

// ─── All LEDs Off ─────────────────────────────────────────
void allLedsOff() {
  digitalWrite(PIN_LED_RED,    LOW);
  digitalWrite(PIN_LED_YELLOW, LOW);
  digitalWrite(PIN_LED_GREEN,  LOW);
  // หมายเหตุ: อย่า digitalWrite BUZZER ที่นี่ — จัดการแยกต่างหาก
}

// ─── Impact Alert ─────────────────────────────────────────
// เรียกเมื่อตรวจพบ impact สูง (ล้มฉับพลัน)
void triggerImpactAlert() {
  if (!impactOverride) {
    impactOverride = true;
    impactStartMs  = millis();
    Serial.println("[WellSense] ⚠️ IMPACT DETECTED — Override RED for 5 seconds!");

    // Buzzer alert: 3 beep สั้น
    for (int i = 0; i < 3; i++) {
      digitalWrite(PIN_BUZZER, HIGH);
      delay(BUZZER_BEEP_MS);
      digitalWrite(PIN_BUZZER, LOW);
      delay(BUZZER_BEEP_MS);
    }

    // บังคับไฟแดงทันที
    allLedsOff();
    digitalWrite(PIN_LED_RED, HIGH);
  }
}

// ─── Startup Sequence (ทดสอบไฟทุกดวง) ──────────────────
void startupSequence() {
  Serial.println("[WellSense] Startup LED test...");

  // เขียว 300ms
  digitalWrite(PIN_LED_GREEN, HIGH);
  delay(300);
  digitalWrite(PIN_LED_GREEN, LOW);

  // เหลือง 300ms
  digitalWrite(PIN_LED_YELLOW, HIGH);
  delay(300);
  digitalWrite(PIN_LED_YELLOW, LOW);

  // แดง 300ms
  digitalWrite(PIN_LED_RED, HIGH);
  delay(300);
  digitalWrite(PIN_LED_RED, LOW);

  // buzzer beep 1 ครั้ง
  digitalWrite(PIN_BUZZER, HIGH);
  delay(100);
  digitalWrite(PIN_BUZZER, LOW);

  Serial.println("[WellSense] Startup complete.");
}
