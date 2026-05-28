# Classical ML Training Summary: binary_fall_risk_rf_deploy

- Dataset: `windows_all.csv`
- Rows used: 2138
- Target: `label`
- Feature count: 46
- Best model: `random_forest`

## Class Distribution

- `0`: 313
- `1`: 1825

## Model Comparison

|   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro |   positive_class |   positive_recall |   positive_precision |   positive_f1 | model         |   train_rows |   test_rows |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_f1_macro_mean |   cv_f1_macro_std |   cv_f1_weighted_mean |   cv_f1_weighted_std |
|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|-----------------:|------------------:|---------------------:|--------------:|:--------------|-------------:|------------:|-------------------:|------------------:|----------------------------:|---------------------------:|-------------------:|------------------:|----------------------:|---------------------:|
|   0.992991 |            0.989324 |   0.986131 |      0.993013 |          0.983001 |       0.989324 |                1 |          0.994521 |             0.997253 |      0.995885 | random_forest |         1710 |         428 |           0.991813 |        0.00624391 |                    0.995205 |                 0.00365653 |           0.984175 |         0.0119299 |              0.991954 |           0.00610183 |

## Main Artifacts

- Best model bundle: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\models\binary_fall_risk_rf_deploy\best_model.joblib`
- Metrics CSV: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\binary_fall_risk_rf_deploy\model_comparison.csv`
- Dataset summary: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\binary_fall_risk_rf_deploy\dataset_summary.json`