# Model 2 Risk Assessment Summary

Model 2 is a mobility risk assessment layer. It is not the Model 1 fall/no-fall detector.

## Method

- Input: numeric window features from `windows_all.csv`.
- Target: domain-informed proxy risk score from motion, rotation, posture, impact, and PPG components.
- Output: risk score 0.0-1.0, high-risk probability, and low/medium/high risk level.
- Important limitation: this is not clinical future-fall probability because the dataset has no true longitudinal fall-risk labels.

## Dataset

- Rows: 2598
- Feature count: 46
- Split: group
- Train rows: 2110
- Test rows: 488

## Risk Level Counts

- `high`: 1953
- `medium`: 413
- `low`: 232

## Best Models

- Best risk-score regressor: `random_forest_regressor`
- Best high-risk classifier: `logistic_regression_classifier`

## Regressor Comparison

|      mae |     rmse |       r2 | model                            |   train_rows |   test_rows |
|---------:|---------:|---------:|:---------------------------------|-------------:|------------:|
| 0.156932 | 0.267312 | 0.327721 | random_forest_regressor          |         2110 |         488 |
| 0.158976 | 0.269082 | 0.31879  | hist_gradient_boosting_regressor |         2110 |         488 |
| 0.173986 | 0.244057 | 0.439603 | ridge_regressor                  |         2110 |         488 |

## Classifier Comparison

|   accuracy |   balanced_accuracy |       f1 |   precision |   recall |   roc_auc | model                          |   train_rows |   test_rows |
|-----------:|--------------------:|---------:|------------:|---------:|----------:|:-------------------------------|-------------:|------------:|
|   0.895492 |            0.894482 | 0.902111 |    0.863971 | 0.943775 |  0.965233 | logistic_regression_classifier |         2110 |         488 |
|   0.860656 |            0.858665 | 0.875    |    0.80678  | 0.955823 |  0.954437 | random_forest_classifier       |         2110 |         488 |

## Main Artifacts

- Model bundle: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\models\model2_combined\model2_risk_bundle.joblib`
- Training targets: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_combined\model2_training_targets.csv`
- Formula config: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_combined\risk_formula_config.json`
- Test predictions: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_combined\test_predictions_preview.csv`