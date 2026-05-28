# Arduino Nano 33 BLE Sense -> preprocessing -> dashboard next steps

This is the practical path from the hardware board to a demo dashboard.

## Current assets

```text
localization_sim/
  arduino/wellsense_nano_logger/
    - Arduino serial logger for IMU + distance format
  localization_sim/
    - localization, gait, heatmap, dataset formatting scripts
  outputs_from_csv/
    - generated coordinate/gait/heatmap examples

fall-detect-dashboard/
  - Next.js dashboard
  - currently uses mock readings from hooks/useMockWebSocket.ts
```

## Goal

```text
Arduino Nano 33 BLE Sense
-> collect ax,ay,az,gx,gy,gz over serial
-> save raw CSV
-> preprocess into mobility state / fall risk / dashboard reading
-> test in local dashboard
-> later publish dashboard
```

## Phase 1: Verify Arduino sensor data

1. Upload a logger sketch to Arduino Nano 33 BLE Sense.
2. Confirm Serial Monitor prints rows like:

```text
time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid
0,0.01,-0.02,9.80,0.01,0.02,-0.01,1200,0
```

3. Save a 30-60 second CSV for each test action:

```text
static_standing.csv
static_sitting.csv
walking_normal.csv
walking_slow.csv
sit_to_stand.csv
stand_to_sit.csv
fall_like_soft.csv
```

4. Keep a simple naming rule:

```text
SE_D04_01_raw.csv
SE_D07_01_raw.csv
SE_F01_01_raw.csv
```

## Phase 2: Format and preprocess

Use the existing dataset formatter:

```bash
python -m localization_sim.dataset_formatter ^
  --input-csv path\to\SE_D04_01_raw.csv ^
  --output-dir dataset_formatted ^
  --subject-group SE ^
  --activity-code D04 ^
  --trial-id 01
```

Expected outputs:

```text
SE_D04_01_dataset_6ch.txt
SE_D04_01_preprocessed.csv
SE_D04_01_manifest.json
```

Use the localization/gait pipeline if the CSV includes distance fields:

```bash
python -m localization_sim.main ^
  --input-csv path\to\nano_log.csv ^
  --output-dir outputs_real ^
  --start-x 0.5 ^
  --start-y 0.5 ^
  --start-heading-deg 0
```

Expected outputs:

```text
outputs_real/localization_coordinates.csv
outputs_real/gait_windows.csv
outputs_real/risk_heatmap.csv
outputs_real/home_test_map.png
```

## Phase 3: Map preprocessing output to dashboard reading

The dashboard accepts this TypeScript shape:

```ts
type SensorReading = {
  timestamp: string;
  room: "Bedroom" | "Bathroom" | "Kitchen" | "Living Room" | "Hallway" | "Balcony";
  x: number;
  y: number;
  gait_speed: number;
  sway: number;
  cadence: number;
  turning_velocity: number;
  instability_score: number;
  fall_risk: number;
  near_fall: boolean;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
};
```

Minimal mapping from our files:

```text
timestamp          <- CSV time or current ISO time
x, y               <- localization_coordinates.csv
gait_speed         <- gait_windows.csv mean_speed_mps
sway               <- gait_windows.csv lateral_sway_rms_mps2
cadence            <- gait_windows.csv cadence_spm
turning_velocity   <- gait_windows.csv turn_rate_rms_rad_s converted/scaled
instability_score  <- gait_windows.csv risk_score * 100
fall_risk          <- risk_heatmap.csv heatmap_score * 100 or gait risk * 100
near_fall          <- fall_risk >= threshold
ax..gz             <- raw/preprocessed sensor CSV
room               <- map x,y to room/zone
```

For the current 4 m x 6 m demo room, a simple room mapping is enough:

```text
y < 1.2                         -> Hallway / doorway area
x < 1.4 and y < 2.4             -> Bedroom/static demo area
x > 2.4 and y > 3.8             -> Desk / living area
else                            -> Living Room
```

## Phase 4: Local dashboard testing

The dashboard path contains `&`, so on Windows use the direct Next command:

```bash
node .\node_modules\next\dist\bin\next dev -p 3000
```

Open:

```text
http://localhost:3000
```

Current available pages:

```text
/
/alerts
/analytics
/devices
```

## Phase 5: Replace mock stream with real/preprocessed stream

Current dashboard live data comes from:

```text
fall-detect-dashboard/hooks/useMockWebSocket.ts
fall-detect-dashboard/lib/mock-data.ts
```

Recommended next implementation:

1. Add a JSON file generated from preprocessing:

```text
fall-detect-dashboard/data/live-readings.json
```

2. Add a loader that converts JSON rows into `SensorReading`.
3. Change `useMockWebSocket` to replay `live-readings.json` instead of
   synthetic `generateReading`.
4. Later replace JSON replay with:

```text
Arduino serial -> small local bridge server -> dashboard fetch/WebSocket
```

## Demo script

1. Show Arduino serial values changing.
2. Save CSV from one walking/static/transition test.
3. Run formatter/preprocessor.
4. Show `home_test_map.png` and `risk_heatmap.png`.
5. Open dashboard local URL.
6. Explain that the current dashboard is replaying processed sensor readings.
7. For future work, replace replay with live serial bridge.

## What not to overbuild tonight

- Do not publish before local demo is stable.
- Do not force static posture labels like sitting vs standing vs lying.
- Do not train a new deep model until we have enough real CSV from the board.
- Do not connect Arduino directly to the browser; use CSV replay or a bridge
  server first.

## Recommended near-term tasks

```text
[ ] Upload Nano logger and collect 3-5 short CSV clips
[ ] Convert each clip with dataset_formatter.py
[ ] Run localization/gait pipeline for clips that include distance
[ ] Generate dashboard-ready JSON
[ ] Modify dashboard to replay dashboard-ready JSON
[ ] Run local demo on localhost:3000
[ ] Publish only after local replay is smooth
```
