# UNO Q JSONL -> Dashboard replay

You already have BLE collection working on UNO Q:

```bash
cd ~/elderly_gait
source venv/bin/activate
cd src
python ble_collect.py --label log_ble
```

It saves JSONL like:

```text
/home/arduino/elderly_gait/data/log_ble_20260528_091142.jsonl
```

Each line has sensor fields such as:

```json
{"pc_ts":"2026-05-28T09:11:42.913","ms":316902,"ax":0.56,"ay":0.97,"az":-1.85,"gx":53.3,"gy":-103.6,"gz":-126.5}
```

## 1. Copy JSONL from UNO Q to Windows

Run this from Windows PowerShell:

```powershell
scp arduino@10.0.11.44:/home/arduino/elderly_gait/data/log_ble_20260528_091142.jsonl `
  "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\localization_sim\outputs_real\log_ble_20260528_091142.jsonl"
```

Use the real filename from UNO Q.

## 2. Convert JSONL to dashboard SensorReading JSON

From Windows PowerShell:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\localization_sim"

python tools\jsonl_to_dashboard_readings.py `
  --input-jsonl outputs_real\log_ble_20260528_091142.jsonl `
  --output-json "..\fall-detect-dashboard\public\live-readings.json"
```

This creates:

```text
fall-detect-dashboard/public/live-readings.json
```

The dashboard auto-loads this file. If the file is empty (`[]`), it falls back
to mock data.

## 3. Run dashboard

From Windows PowerShell:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall-detect-dashboard"

node .\node_modules\next\dist\bin\next dev -p 3000
```

Open:

```text
http://localhost:3000
```

The dashboard will replay `live-readings.json` every 1.2 seconds.

## 4. What the converter calculates

Input:

```text
ax, ay, az, gx, gy, gz from UNO Q JSONL
```

Output dashboard fields:

```text
timestamp          <- pc_ts
ax..gz             <- raw sensor values
instability_score  <- acceleration + gyro magnitude heuristic
fall_risk          <- instability + high-motion boost
near_fall          <- high fall_risk / high jerk-like motion
gait_speed         <- inverse of instability
sway               <- side/up acceleration proxy
cadence            <- estimated from instability
turning_velocity   <- inverse of gyro magnitude
x,y,room           <- demo floorplan replay path
```

This is a demo bridge, not the final clinical model. It lets us show that real
BLE sensor data can drive the dashboard now.

## 5. Repeat workflow

For every new test:

```text
1. Run ble_collect.py on UNO Q
2. Stop with Ctrl+C
3. Copy latest JSONL to Windows
4. Convert to public/live-readings.json
5. Refresh dashboard
```

## 6. Optional: check JSON count

```powershell
python -c "import json; print(len(json.load(open('..\fall-detect-dashboard\public\live-readings.json'))))"
```
