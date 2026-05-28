# Start here: Arduino Nano 33 BLE Sense sensor test

Use this when starting from Arduino IDE.

## 1. Install board support

In Arduino IDE:

```text
Tools -> Board -> Boards Manager
```

Search and install:

```text
Arduino Mbed OS Nano Boards
```

Then select your board:

```text
Tools -> Board -> Arduino Mbed OS Nano Boards -> Arduino Nano 33 BLE Sense
```

or, if your board is Rev2:

```text
Tools -> Board -> Arduino Mbed OS Nano Boards -> Arduino Nano 33 BLE Sense Rev2
```

## 2. Install the IMU library

Open:

```text
Tools -> Manage Libraries
```

If the board is original Nano 33 BLE Sense, install:

```text
Arduino_LSM9DS1
```

If the board is Nano 33 BLE Sense Rev2, install:

```text
Arduino_BMI270_BMM150
```

## 3. Open the right sketch

Original Nano 33 BLE Sense:

```text
arduino/nano33_ble_sense_imu_logger_rev1/nano33_ble_sense_imu_logger_rev1.ino
```

Nano 33 BLE Sense Rev2:

```text
arduino/nano33_ble_sense_imu_logger_rev2/nano33_ble_sense_imu_logger_rev2.ino
```

## 4. Select port

Plug in the board with USB, then select:

```text
Tools -> Port -> COMx
```

If no port appears:

```text
1. Change USB cable
2. Press reset on board twice quickly
3. Reopen Arduino IDE
```

## 5. Upload

Click the arrow upload button.

If upload works, open:

```text
Tools -> Serial Monitor
```

Set baud:

```text
115200
```

Expected output:

```text
# Arduino Nano 33 BLE Sense Rev2 IMU logger
# Units: ax/ay/az=m/s^2, gx/gy/gz=rad/s
time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid
102,0.12,-0.01,9.78,0.01,0.02,-0.01,1200,0
202,0.10,-0.03,9.82,0.00,0.01,-0.02,1200,0
```

`distance_mm,distance_valid` are placeholders here because this test uses only
the built-in Nano IMU. They keep the CSV compatible with the Python pipeline.

## 6. Save CSV for preprocessing

Copy the Serial Monitor output into a file, for example:

```text
SE_D04_01_raw.csv
```

Remove any extra Arduino IDE text before the CSV header if needed.

Then run:

```bash
python -m localization_sim.dataset_formatter ^
  --input-csv path\to\SE_D04_01_raw.csv ^
  --output-dir dataset_formatted ^
  --subject-group SE ^
  --activity-code D04 ^
  --trial-id 01
```

For the localization/gait pipeline:

```bash
python -m localization_sim.main ^
  --input-csv path\to\SE_D04_01_raw.csv ^
  --output-dir outputs_real ^
  --start-x 0.5 ^
  --start-y 0.5 ^
  --start-heading-deg 0
```

Since this IMU-only logger has no real distance sensor, position confidence may
drop over time. That is expected for the first sensor test.
