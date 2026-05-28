# Indoor coordinate localization + gait risk demo

This is the first demo-ready pipeline for the WellSense prevention idea:

```text
known room map
+ known start coordinate
+ Arduino Nano-style IMU + Distance packets
+ rule-based indoor localization
+ gait window features
= estimated x,y coordinate + risk heatmap + summary report
```

The default run still uses software-simulated packets, but the same packet
format can now be produced by the Arduino sketch in
`arduino/wellsense_nano_logger`.

## Why this answers the assignment

- `x,y in house`: `localization_coordinates.csv` contains one estimated
  coordinate per packet.
- `Arduino Nano`: `arduino/wellsense_nano_logger/wellsense_nano_logger.ino`
  streams the Nano packet format.
- `Modulino IMU + Modulino Distance`: the packet uses acceleration, gyroscope,
  and ToF distance.
- `Can it work?`: `summary_report.txt` states demo readiness, warnings, and
  known blockers.
- `x,y + gait -> heatmap`: `gait_windows.csv`, `risk_heatmap.csv`, and
  `risk_heatmap.png` combine location and gait risk.

## Room model

Example room:

```text
width  = 4.0 m
height = 6.0 m
origin = bottom-left corner
start  = (0.5, 0.5), heading east
```

Known objects:

```text
Panel A = rectangle around (1.0, 4.8)
Panel B = rectangle around (0.8, 1.8)
Desk    = rectangle around (3.0, 4.5)
Doorway = bottom wall gap around x ~= 2.0, y = 0
```

## Run simulated demo

From this directory:

```bash
python -m localization_sim.main --output-dir outputs
```

Outputs:

```text
outputs/simulated_nano_packets.csv  # fake Arduino Nano serial stream
outputs/localization_coordinates.csv # estimated x,y per packet
outputs/estimated_path.csv           # truth vs estimate for simulated runs
outputs/gait_windows.csv             # windowed gait/risk features
outputs/risk_heatmap.csv             # dashboard-ready heatmap cells
outputs/summary_report.txt           # viability and blocker summary
outputs/trajectory.png               # true/estimated path plot
outputs/confidence.png               # confidence over time
outputs/risk_heatmap.png             # x,y + gait risk heatmap
outputs/home_test_map.png            # floor-plan view for device-test presentation
```

## Run with real Arduino CSV

Capture Serial output from `arduino/wellsense_nano_logger` and run:

```bash
python -m localization_sim.main --input-csv path/to/nano_log.csv --output-dir outputs_real
```

You can override the known start pose:

```bash
python -m localization_sim.main ^
  --input-csv path/to/nano_log.csv ^
  --output-dir outputs_real ^
  --start-x 0.5 ^
  --start-y 0.5 ^
  --start-heading-deg 0
```

## Packet format

```text
time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid
```

Units:

```text
ax, ay, az: m/s^2
gx, gy, gz: rad/s
distance_mm: millimeters
distance_valid: 1 or 0
```

The loader also accepts a few common aliases such as `timestamp_ms`,
`gyro_z`, `tof_mm`, and `tof_valid`.

## Format sensor CSV like the paper dataset

The paper-style dataset rows are headerless numeric rows ending with `;`.
For this project we use only the first 6 IMU channels:

```text
ax,ay,az,gx,gy,gz;
```

The formatter reads a raw sensor CSV like:

```text
pc_timestamp_iso,ms,ax,ay,az,gx,gy,gz,mx,my,mz,...
```

and exports:

```text
dataset_formatted/SE_D04_01_dataset_6ch.txt      # paper-like 6-channel rows
dataset_formatted/SE_D04_01_preprocessed.csv     # metadata + normalized SI units
dataset_formatted/SE_D04_01_manifest.json        # label/unit summary
```

Example:

```bash
python -m localization_sim.dataset_formatter ^
  --input-csv path\to\raw_sensor.csv ^
  --output-dir dataset_formatted ^
  --subject-group SE ^
  --activity-code D04 ^
  --trial-id 01
```

Subject groups:

```text
SA = young adult
SE = elderly
```

Activity labels:

```text
D01 = เดินจับของระหว่างทาง
D02 = เดินช้า
D03 = เดินกระเพก
D04 = เดินปกติ
D05 = นอน
D06 = ยืน
D07 = ลุกยืนสลับนั่ง
F01 = ค่อยๆล้ม
F02 = ล้มแบบค่อยๆทรุด
```

Unit handling:

```text
accelerometer input: auto-detect g or m/s^2
gyroscope input: auto-detect deg/s or rad/s
paper-like output: accel in milli-g, gyro in deg/s, integer rows
preprocessed CSV: accel in m/s^2, gyro in rad/s
```

## Current estimator

The estimator is deliberately simple and explainable:

1. Calibrate gyro bias while the subject is initially stationary.
2. Integrate `gz` to update heading.
3. Use IMU motion energy to classify stationary / moving / turning.
4. Predict movement using a nominal walking speed.
5. If ToF is valid, search nearby grid cells and choose the one whose expected
   raycast distance best matches the measured distance.
6. Output estimated coordinate and confidence.

This is not yet a particle filter. That should be the next version after this
pipeline is validated with real walking data.

## Current blockers to tell judges honestly

- Needs a known room map and known start pose.
- Heading can drift between distance correction events.
- Forward ToF only helps when a mapped wall/object is within range.
- Gait thresholds assume waist/L4-L5 placement; other placements need tuning.
- Real homes need up-to-date obstacle/layout data for meaningful heatmaps.

## Research notes

- Static vs transition activity notes:
  `docs/static_transition_research.md`
- Useful paper roadmap:
  `docs/paper_roadmap.md`
- Arduino-to-dashboard next steps:
  `docs/arduino_dashboard_next_steps.md`
- Nano 33 BLE Sense BLE connection:
  `docs/ble_nano_connection.md`
- UNO Q JSONL to dashboard replay:
  `docs/unoq_jsonl_to_dashboard.md`
