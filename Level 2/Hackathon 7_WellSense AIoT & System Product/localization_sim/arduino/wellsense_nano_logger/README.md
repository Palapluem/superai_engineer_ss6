# WellSense Nano Logger

Arduino sketch for streaming the same packet format used by the Python
localization pipeline:

```text
time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid
```

## Hardware

- Arduino Nano board supported by the Arduino Modulino library
- Modulino Movement / IMU
- Modulino Distance / ToF

## Arduino IDE libraries

Install these from Library Manager:

- Arduino Modulino
- Arduino LSM6DSOX
- STM32duino VL53L4CD
- STM32duino VL53L4ED

The sketch converts Movement acceleration from `g` to `m/s^2` and gyroscope
from `deg/s` to `rad/s`, matching the Python estimator units.

## Use with Python

1. Upload `wellsense_nano_logger.ino`.
2. Capture Serial Monitor output to CSV.
3. Run:

```bash
python -m localization_sim.main --input-csv path/to/nano_log.csv --output-dir outputs_real
```

The Python loader ignores `#` status lines before the CSV header.
