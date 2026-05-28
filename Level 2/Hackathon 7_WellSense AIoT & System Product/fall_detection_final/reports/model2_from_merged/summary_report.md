# Model 2 Risk Assessment Summary

Model 2 is a mobility risk assessment layer. It is not the Model 1 fall/no-fall detector.

## Method

- Input: numeric window features from `windows_all.csv`.
- Target: domain-informed proxy risk score from motion, rotation, posture, impact, and PPG components.
- Output: risk score 0.0-1.0, high-risk probability, and low/medium/high risk level.
- Important limitation: this is not clinical future-fall probability because the dataset has no true longitudinal fall-risk labels.

## Dataset

- Rows: 798
- Feature count: 46
- Split: group
- Train rows: 552
- Test rows: 246

## Risk Level Counts

- `medium`: 419
- `low`: 219
- `high`: 160

## Best Models

- Best risk-score regressor: `random_forest_regressor`
- Best high-risk classifier: `random_forest_classifier`

## Regressor Comparison

|      mae |     rmse |        r2 | model                            |   train_rows |   test_rows |
|---------:|---------:|----------:|:---------------------------------|-------------:|------------:|
| 0.152616 | 0.165133 | -0.113534 | random_forest_regressor          |          552 |         246 |
| 0.155479 | 0.172935 | -0.221246 | hist_gradient_boosting_regressor |          552 |         246 |
| 0.185248 | 0.202825 | -0.679875 | ridge_regressor                  |          552 |         246 |

## Classifier Comparison

|   accuracy |   balanced_accuracy |        f1 |   precision |   recall |   roc_auc | model                          |   train_rows |   test_rows |
|-----------:|--------------------:|----------:|------------:|---------:|----------:|:-------------------------------|-------------:|------------:|
|   0.634146 |            0.569915 | 0.1       |   0.0555556 |      0.5 |  0.677119 | random_forest_classifier       |          552 |         246 |
|   0.654472 |            0.484746 | 0.0659341 |   0.037037  |      0.3 |  0.667373 | logistic_regression_classifier |          552 |         246 |

## Main Artifacts

- Model bundle: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\models\model2_from_merged\model2_risk_bundle.joblib`
- Training targets: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_from_merged\model2_training_targets.csv`
- Formula config: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_from_merged\risk_formula_config.json`
- Test predictions: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_from_merged\test_predictions_preview.csv`