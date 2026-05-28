# Model 2 Risk Assessment Summary

Model 2 is a mobility risk assessment layer. It is not the Model 1 fall/no-fall detector.

## Method

- Input: numeric window features from `windows_all.csv`.
- Target: domain-informed proxy risk score from motion, rotation, posture, impact, and PPG components.
- Output: risk score 0.0-1.0, high-risk probability, and low/medium/high risk level.
- Important limitation: this is not clinical future-fall probability because the dataset has no true longitudinal fall-risk labels.

## Dataset

- Rows: 2138
- Feature count: 46
- Split: group
- Train rows: 1692
- Test rows: 446

## Risk Level Counts

- `high`: 1825
- `medium`: 224
- `low`: 89

## Best Models

- Best risk-score regressor: `random_forest_regressor`
- Best high-risk classifier: `random_forest_classifier`

## Regressor Comparison

|       mae |      rmse |       r2 | model                            |   train_rows |   test_rows |
|----------:|----------:|---------:|:---------------------------------|-------------:|------------:|
| 0.0298234 | 0.0702853 | 0.909554 | random_forest_regressor          |         1692 |         446 |
| 0.0331653 | 0.0784819 | 0.887228 | hist_gradient_boosting_regressor |         1692 |         446 |
| 0.0342284 | 0.0672617 | 0.917168 | ridge_regressor                  |         1692 |         446 |

## Classifier Comparison

|   accuracy |   balanced_accuracy |   f1 |   precision |   recall |   roc_auc | model                          |   train_rows |   test_rows |
|-----------:|--------------------:|-----:|------------:|---------:|----------:|:-------------------------------|-------------:|------------:|
|          1 |                   1 |    1 |           1 |        1 |         1 | random_forest_classifier       |         1692 |         446 |
|          1 |                   1 |    1 |           1 |        1 |         1 | logistic_regression_classifier |         1692 |         446 |

## Main Artifacts

- Model bundle: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\models\model2_risk_assessment\model2_risk_bundle.joblib`
- Training targets: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_risk_assessment\model2_training_targets.csv`
- Formula config: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_risk_assessment\risk_formula_config.json`
- Test predictions: `C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final\reports\model2_risk_assessment\test_predictions_preview.csv`