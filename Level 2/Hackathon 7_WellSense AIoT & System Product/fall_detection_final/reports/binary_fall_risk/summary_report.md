# Classical ML Training Summary: binary_fall_risk

- Dataset: `windows_all.csv`
- Rows used: 2138
- Target: `label`
- Feature count: 46
- Best model: `xgboost`

## Class Distribution

- `0`: 313
- `1`: 1825

## Model Comparison

|   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro |   positive_class |   positive_recall |   positive_precision |   positive_f1 | model         |   train_rows |   test_rows |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_f1_macro_mean |   cv_f1_macro_std |   cv_f1_weighted_mean |   cv_f1_weighted_std |
|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|-----------------:|------------------:|---------------------:|--------------:|:--------------|-------------:|------------:|-------------------:|------------------:|----------------------------:|---------------------------:|-------------------:|------------------:|----------------------:|---------------------:|
|   0.995327 |            0.990694 |   0.990694 |      0.995327 |          0.990694 |       0.990694 |                1 |          0.99726  |             0.99726  |      0.99726  | xgboost       |         1710 |         428 |           0.991813 |        0.00725712 |                    0.990233 |                 0.00997422 |           0.983973 |        0.0140113  |              0.991905 |           0.00712737 |
|   0.992991 |            0.989324 |   0.986131 |      0.993013 |          0.983001 |       0.989324 |                1 |          0.994521 |             0.997253 |      0.995885 | random_forest |         1710 |         428 |           0.991813 |        0.00624391 |                    0.995205 |                 0.00365653 |           0.984175 |        0.0119299  |              0.991954 |           0.00610183 |
|   0.988318 |            0.973451 |   0.97658  |      0.988279 |          0.979773 |       0.973451 |                1 |          0.994521 |             0.991803 |      0.99316  | svm_rbf       |         1710 |         428 |           0.992398 |        0.0047509  |                    0.99389  |                 0.00458268 |           0.985156 |        0.00916273 |              0.992492 |           0.00466353 |

## Main Artifacts

- Best model bundle: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\models\binary_fall_risk\best_model.joblib`
- Metrics CSV: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\binary_fall_risk\model_comparison.csv`
- Dataset summary: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\binary_fall_risk\dataset_summary.json`