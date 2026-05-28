# Classical ML Training Summary: multiclass_activity

- Dataset: `windows_all.csv`
- Rows used: 2138
- Target: `class_en`
- Feature count: 46
- Best model: `random_forest`

## Class Distribution

- `backward_fall`: 805
- `corrected_walking`: 59
- `elderly_pick_up_object`: 63
- `gradual_fall`: 25
- `limping_walk`: 80
- `lying_down`: 17
- `normal_walk`: 29
- `sideways_fall`: 806
- `slow_collapse_fall`: 189
- `stand_sit_alternating`: 56
- `standing`: 9

## Model Comparison

|   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro | model         |   train_rows |   test_rows |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_f1_macro_mean |   cv_f1_macro_std |   cv_f1_weighted_mean |   cv_f1_weighted_std |
|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|:--------------|-------------:|------------:|-------------------:|------------------:|----------------------------:|---------------------------:|-------------------:|------------------:|----------------------:|---------------------:|
|   0.969626 |            0.879756 |   0.891538 |      0.968751 |          0.932089 |       0.879756 | random_forest |         1710 |         428 |           0.950292 |        0.0126781  |                    0.823452 |                  0.0382348 |           0.824042 |         0.0408189 |              0.949797 |           0.0122623  |
|   0.962617 |            0.845112 |   0.855578 |      0.961477 |          0.896723 |       0.845112 | xgboost       |         1710 |         428 |           0.960234 |        0.00935673 |                    0.839528 |                  0.0351113 |           0.838534 |         0.0383729 |              0.959241 |           0.00906499 |
|   0.766355 |            0.707534 |   0.722354 |      0.765448 |          0.760249 |       0.707534 | svm_rbf       |         1710 |         428 |           0.732164 |        0.0243657  |                    0.712206 |                  0.0263178 |           0.70522  |         0.0327836 |              0.73018  |           0.0249505  |

## Main Artifacts

- Best model bundle: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\models\multiclass_activity\best_model.joblib`
- Metrics CSV: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\multiclass_activity\model_comparison.csv`
- Dataset summary: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\multiclass_activity\dataset_summary.json`