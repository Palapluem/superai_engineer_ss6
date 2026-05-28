# Bluetooth/BLE connection for Nano 33 BLE Sense

Nano 33 BLE Sense uses BLE. It is not the same as classic Bluetooth serial.
Pairing in Windows may show the device, but the useful part is subscribing to a
BLE characteristic from a BLE central app/script.

## Roles

```text
Nano 33 BLE Sense = BLE peripheral, advertises IMU data
Laptop / UNO Q    = BLE central, scans and subscribes
```

## Arduino sketch

Install libraries in Arduino IDE:

```text
ArduinoBLE
Arduino_LSM9DS1          # original Nano 33 BLE Sense
Arduino_BMI270_BMM150    # Nano 33 BLE Sense Rev2
```

Open one of these:

```text
arduino/nano33_ble_sense_ble_imu_rev1/nano33_ble_sense_ble_imu_rev1.ino
arduino/nano33_ble_sense_ble_imu_rev2/nano33_ble_sense_ble_imu_rev2.ino
```

Upload to the Nano. In Serial Monitor at `115200`, expected logs:

```text
# WellSense Nano BLE IMU Rev1
# Advertising as WellSenseNano
# CSV payload: time_ms,ax,ay,az,gx,gy,gz
```

BLE device name:

```text
WellSenseNano
```

BLE characteristic:

```text
19b10001-e8f2-537e-4f6c-d104768a1214
```

Payload:

```text
time_ms,ax,ay,az,gx,gy,gz
```

The Python bridge adds:

```text
distance_mm=1200,distance_valid=0
```

so the CSV stays compatible with the localization pipeline.

## Receive from Windows laptop

From `localization_sim`:

```bash
python -m pip install bleak
python tools/ble_nano_bridge.py --output-csv outputs_real/ble_nano_log.csv
```

Stop with `Ctrl+C`.

Then format for dataset:

```bash
python -m localization_sim.dataset_formatter ^
  --input-csv outputs_real/ble_nano_log.csv ^
  --output-dir dataset_formatted ^
  --subject-group SE ^
  --activity-code D04 ^
  --trial-id 01
```

Or run the localization/gait pipeline:

```bash
python -m localization_sim.main ^
  --input-csv outputs_real/ble_nano_log.csv ^
  --output-dir outputs_real ^
  --start-x 0.5 ^
  --start-y 0.5 ^
  --start-heading-deg 0
```

## Receive from UNO Q

SSH into UNO Q:

```bash
ssh arduino@10.0.11.44
```

Then install the BLE Python client:

```bash
python3 -m pip install --user bleak
```

Copy `tools/ble_nano_bridge.py` to the UNO Q, then run:

```bash
python3 ble_nano_bridge.py --output-csv ble_nano_log.csv
```

If BLE scan fails on Linux, check:

```bash
bluetoothctl
power on
scan on
```

You should see:

```text
WellSenseNano
```

## Important notes

- Do not expect a COM port from BLE. Nano 33 BLE Sense BLE is GATT, not classic
  serial.
- If the Nano was paired before, it can still fail if the sketch is not
  advertising the service above.
- For dashboard demo, CSV replay is safer than direct live BLE until the stream
  is stable.
