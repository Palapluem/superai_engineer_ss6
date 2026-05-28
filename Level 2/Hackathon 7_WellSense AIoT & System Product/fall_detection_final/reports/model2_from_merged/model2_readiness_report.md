# Model 2 Readiness Report

This report checks class counts, feature completeness, and inference time for Model 2 risk assessment.

## Executive Summary

- Dataset rows: 798
- Model features required: 46
- Feature check: PASS
- Class count check against `image.png`: CHECK
- Single-window inference mean: 77.849 ms/window
- Full-batch inference mean: 0.125 ms/window

## Interpretation For Seniors

- Model 2 is not the binary fall/no-fall baseline.
- It estimates a mobility risk score from movement features such as jerk, omega, theta, GSI, FCRI, and optional PPG-derived features.
- The current output is a proxy risk score for prototype testing because the dataset does not include clinical future-fall labels.
- The score can be used for dashboard risk level and heatmap accumulation with `(x, y)` location.

## Class Count Check

| category        | class_en               | thai_name          |   expected_count_from_image |   actual_count_in_windows_all |   difference | match   |
|:----------------|:-----------------------|:-------------------|----------------------------:|------------------------------:|-------------:|:--------|
| fall            | slow_collapse_fall     | ล้มแบบค่อยๆทรุด       |                         189 |                           100 |          -89 | False   |
| fall            | gradual_fall           | ค่อยๆล้ม             |                          25 |                            53 |           28 | False   |
| fall            | sideways_fall          | ล้มข้าง              |                         806 |                             0 |         -806 | False   |
| fall            | backward_fall          | ล้มไปด้านหลัง         |                         805 |                             0 |         -805 | False   |
| activity        | normal_walk            | เดินปกติ             |                          29 |                            61 |           32 | False   |
| activity        | limping_walk           | เดินกระเพก          |                          80 |                           163 |           83 | False   |
| activity        | corrected_walking      | คนแก่เดิน            |                          59 |                           121 |           62 | False   |
| activity        | stand_sit_alternating  | ลุกยืนสลับนั่ง          |                          56 |                           115 |           59 | False   |
| activity        | elderly_pick_up_object | คนแก่จับของระหว่างทาง |                          63 |                           128 |           65 | False   |
| static_activity | standing               | ยืน                 |                           9 |                            20 |           11 | False   |
| static_activity | lying_down             | นอน                |                          17 |                            37 |           20 | False   |

## Feature Completeness

- Required features present: 46/46
- Required features numeric: 46/46
- Missing features: none
- Non-numeric features: none
- Null note: No null values in model features.

Features with null values:

none

## Inference Time

Measured on the current Windows machine with the saved `model2_risk_bundle.joblib`.

|   batch_size |   repeats |   mean_batch_ms |   median_batch_ms |   p95_batch_ms |   mean_per_window_ms |   median_per_window_ms |   windows_per_second_mean |
|-------------:|----------:|----------------:|------------------:|---------------:|---------------------:|-----------------------:|--------------------------:|
|            1 |       100 |         77.8494 |           77.3276 |        79.1872 |            77.8494   |              77.3276   |                   12.8453 |
|           10 |       100 |         60.6504 |           56.1749 |        73.7824 |             6.06504  |               5.61749  |                  164.879  |
|           50 |       100 |         79.5588 |           77.585  |        90.5113 |             1.59118  |               1.5517   |                  628.466  |
|          100 |       100 |         79.5731 |           77.8506 |        91.5161 |             0.795731 |               0.778506 |                 1256.71   |
|          798 |        20 |         99.7718 |           98.3122 |       110.323  |             0.125027 |               0.123198 |                 7998.25   |

## Window Timing From Dataset

- Median feature window length: 1.95 seconds
- Median stride between windows: 0.499 seconds

Practical reading:

- One prediction call takes about 0.0778 seconds per single window on this machine.
- Since each feature window represents about 1.95 seconds of sensor data, inference time is much smaller than the sensing window.

## Output Files

- `model2_class_count_check.csv`
- `model2_feature_completeness.csv`
- `model2_inference_benchmark.csv`
- `model2_readiness_summary.json`