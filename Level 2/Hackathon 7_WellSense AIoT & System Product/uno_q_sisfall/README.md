# UNO Q SisFall Deployment

Generated from `models/sisfall_adxl345_itg3200_fall_vs_adl_model.json`.

## Files

- `uno_q_sisfall.ino`: Arduino sketch for the UNO Q microcontroller side.
- `sisfall_model.h`: scaler and logistic-regression weights exported as C++ constants.

## Model Input

The sketch expects CSV rows in this order:

```text
adxl345_x, adxl345_y, adxl345_z, itg3200_x, itg3200_y, itg3200_z
```

`INPUT_IS_RAW_BITS` is `true` by default, so raw SisFall-style values are converted into the physical
units used during training. Set it to `false` in the sketch if your sensor library already returns
ADXL345 acceleration in g and ITG3200 angular velocity in deg/s.

The model was trained with `sample_stride=4`, giving an effective
feature rate of about `50` Hz. The default window is `SISFALL_DEFAULT_WINDOW_SAMPLES`,
roughly 15 seconds of the original 200 Hz SisFall data.

## Upload

Open `uno_q_sisfall.ino` in Arduino App Lab or Arduino IDE, select the UNO Q MCU target, and upload.
Use the serial monitor at 115200 baud to feed rows or to verify predictions.

## Deployment Note

This is a file/window-level SisFall classifier. For a production fall detector, retrain on fixed-length
sliding windows collected from your actual sensor placement and sampling loop.
