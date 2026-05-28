# Model 2 Readiness Report

This report checks class counts, feature completeness, and inference time for Model 2 risk assessment.

## Executive Summary

- Dataset rows: 2138
- Model features required: 46
- Feature check: PASS
- Class count check against `image.png`: PASS
- Single-window inference mean: 102.677 ms/window
- Full-batch inference mean: 0.063 ms/window

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
|            1 |       100 |         102.677 |          100.399  |        123.945 |          102.677     |            100.399     |                   9.73926 |
|           10 |       100 |         105.545 |          101.107  |        139.055 |           10.5545    |             10.1107    |                  94.7461  |
|           50 |       100 |         100.92  |          100.999  |        120.268 |            2.0184    |              2.01998   |                 495.441   |
|          100 |       100 |         104.281 |           99.9768 |        132.218 |            1.04281   |              0.999768  |                 958.947   |
|         2138 |        20 |         134.842 |          128.587  |        172.543 |            0.0630692 |              0.0601437 |               15855.6     |

## Window Timing From Dataset

- Median feature window length: 1.95 seconds
- Median stride between windows: 0.5 seconds

Practical reading:

- One prediction call takes about 0.1027 seconds per single window on this machine.
- Since each feature window represents about 1.95 seconds of sensor data, inference time is much smaller than the sensing window.

## Output Files

- `model2_class_count_check.csv`
- `model2_feature_completeness.csv`
- `model2_inference_benchmark.csv`
- `model2_readiness_summary.json`