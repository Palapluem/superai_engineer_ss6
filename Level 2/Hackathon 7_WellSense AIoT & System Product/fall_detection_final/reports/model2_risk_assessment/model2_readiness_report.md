# Model 2 Readiness Report

This report checks class counts, feature completeness, and inference time for Model 2 risk assessment.

## Executive Summary

- Dataset rows: 2138
- Model features required: 46
- Feature check: PASS
- Class count check against `image.png`: PASS
- Single-window inference mean: 77.030 ms/window
- Full-batch inference mean: 0.047 ms/window

## Interpretation For Seniors

- Model 2 is not the binary fall/no-fall baseline.
- It estimates a mobility risk score from movement features such as jerk, omega, theta, GSI, FCRI, and optional PPG-derived features.
- The current output is a proxy risk score for prototype testing because the dataset does not include clinical future-fall labels.
- The score can be used for dashboard risk level and heatmap accumulation with `(x, y)` location.

## Class Count Check

| category        | class_en               | thai_name          |   expected_count_from_image |   actual_count_in_windows_all |   difference | match   |
|:----------------|:-----------------------|:-------------------|----------------------------:|------------------------------:|-------------:|:--------|
| fall            | slow_collapse_fall     | ล้มแบบค่อยๆทรุด       |                         189 |                           189 |            0 | True    |
| fall            | gradual_fall           | ค่อยๆล้ม             |                          25 |                            25 |            0 | True    |
| fall            | sideways_fall          | ล้มข้าง              |                         806 |                           806 |            0 | True    |
| fall            | backward_fall          | ล้มไปด้านหลัง         |                         805 |                           805 |            0 | True    |
| activity        | normal_walk            | เดินปกติ             |                          29 |                            29 |            0 | True    |
| activity        | limping_walk           | เดินกระเพก          |                          80 |                            80 |            0 | True    |
| activity        | corrected_walking      | คนแก่เดิน            |                          59 |                            59 |            0 | True    |
| activity        | stand_sit_alternating  | ลุกยืนสลับนั่ง          |                          56 |                            56 |            0 | True    |
| activity        | elderly_pick_up_object | คนแก่จับของระหว่างทาง |                          63 |                            63 |            0 | True    |
| static_activity | standing               | ยืน                 |                           9 |                             9 |            0 | True    |
| static_activity | lying_down             | นอน                |                          17 |                            17 |            0 | True    |

## Feature Completeness

- Required features present: 46/46
- Required features numeric: 46/46
- Missing features: none
- Non-numeric features: none
- Null note: Some features contain null values; the model pipeline imputes them with median values.

Features with null values:

| feature      |   null_count |
|:-------------|-------------:|
| svm_dev_mean |          338 |

## Inference Time

Measured on the current Windows machine with the saved `model2_risk_bundle.joblib`.

|   batch_size |   repeats |   mean_batch_ms |   median_batch_ms |   p95_batch_ms |   mean_per_window_ms |   median_per_window_ms |   windows_per_second_mean |
|-------------:|----------:|----------------:|------------------:|---------------:|---------------------:|-----------------------:|--------------------------:|
|            1 |       100 |         77.0298 |           76.5426 |        78.7174 |           77.0298    |             76.5426    |                    12.982 |
|           10 |       100 |         76.9273 |           76.4585 |        79.5048 |            7.69273   |              7.64585   |                   129.993 |
|           50 |       100 |         76.3918 |           76.4889 |        78.9749 |            1.52784   |              1.52978   |                   654.521 |
|          100 |       100 |         81.3283 |           77.2558 |       104.449  |            0.813283  |              0.772558  |                  1229.58  |
|         2138 |        20 |        100.8    |           98.7473 |       111.037  |            0.0471471 |              0.0461868 |                 21210.2   |

## Window Timing From Dataset

- Median feature window length: 1.95 seconds
- Median stride between windows: 0.5 seconds

Practical reading:

- One prediction call takes about 0.0770 seconds per single window on this machine.
- Since each feature window represents about 1.95 seconds of sensor data, inference time is much smaller than the sensing window.

## Output Files

- `model2_class_count_check.csv`
- `model2_feature_completeness.csv`
- `model2_inference_benchmark.csv`
- `model2_readiness_summary.json`