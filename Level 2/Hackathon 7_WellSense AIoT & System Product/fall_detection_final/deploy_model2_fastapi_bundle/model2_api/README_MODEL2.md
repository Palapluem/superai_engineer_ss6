# Model 2: Risk Assessment

Model 2 is the risk assessment layer for the project. It is intentionally separate from Model 1.

## Model Roles

| Model | Owner / Role | Output |
|---|---|---|
| Model 1 | Baseline fall detector | Fall / no fall |
| Model 2 | This risk assessment layer | Risk score 0.0-1.0, low/medium/high risk |

Model 2 does not answer only "did the person fall right now?". It answers:

> How risky or unstable does this movement window look?

## Important Limitation

The current dataset does not contain true clinical labels such as "this person fell in the future" or expert-scored fall risk. Because of that, Model 2 uses a domain-informed proxy target.

Use this wording when presenting:

> Model 2 currently estimates a proxy mobility risk score from IMU/PPG-derived features. It is suitable for prototype testing and dashboard risk visualization, but it is not yet a clinically validated future-fall probability.

## Risk Target Design

The training script builds a `model2_risk_target` from:

- `feature_risk_score`: computed from feature components
- `activity_prior_score`: class/category prior used only to construct the training target

Formula:

```text
weighted_score = 0.60 * feature_risk_score + 0.40 * activity_prior_score
prior_floor    = 0.95 * activity_prior_score
raw_model2_risk_target = max(weighted_score, prior_floor)
model2_risk_target = conservative_guard(raw_model2_risk_target, feature_components)
```

The conservative guard prevents a single noisy channel from creating a high alert. A high-risk window now needs impact evidence or multiple elevated components.

Risk levels:

```text
low    : 0.00 <= score <= 0.35
medium : 0.35 < score <= 0.70
high   : score > 0.70
alert  : score > 0.70 and high-risk probability >= 0.70
```

## Feature Components

Model 2 uses numeric features only during inference.

| Component | Example features | Meaning |
|---|---|---|
| gait_motion | `jerk_mean`, `jerk_max`, `svm_std`, `jerk_energy` | Abrupt or unstable movement |
| rotation_balance | `omega_mean`, `omega_max`, `angular_impulse` | Turning / rotational instability |
| posture_transition | `theta_range`, `KII_mean`, `KII_max` | Large body tilt or posture change |
| impact_event | `GSI`, `fcri`, `impact_n`, `free_fall_n` | Fall-like shock or impact pattern |
| physio_stress | `hr_delta`, `hr_spike`, `osi`, `spo2_min` | Optional PPG/physiology stress proxy |

## Train Model 2

Open PowerShell:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final"

python .\train_model2_risk_assessment.py `
  --data .\windows_all.csv `
  --run-name model2_risk_assessment
```

The default split is group-based by `session_id`, which is stricter than a random window split.

## Predict With Model 2

```powershell
python .\predict_model2_risk_assessment.py `
  --model .\models\model2_risk_assessment\model2_risk_bundle.joblib `
  --input .\windows_all.csv `
  --output .\reports\model2_risk_assessment\model2_predictions_windows_all.csv
```

Output columns include:

- `raw_model2_risk_score`
- `model2_risk_score`
- `model2_high_risk_probability`
- `model2_risk_level`
- `model2_alert`
- `rule_feature_risk_score`
- guard columns such as `guard_reason`, `guard_high_component_count`
- component scores such as `component_gait_motion`, `component_rotation_balance`

## FastAPI / Dashboard

See [README_MODEL2_FASTAPI.md](README_MODEL2_FASTAPI.md) for the local API and dashboard integration.

Quick start:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model2_api.ps1
```

Then open:

```text
http://localhost:8000/docs
```

## Check Counts, Features, And Inference Time

Use this before presenting the model:

```powershell
python .\evaluate_model2_readiness.py `
  --data .\windows_all.csv `
  --model .\models\model2_risk_assessment\model2_risk_bundle.joblib `
  --output-dir .\reports\model2_risk_assessment
```

This creates:

- `reports\model2_risk_assessment\model2_readiness_report.md`
- `reports\model2_risk_assessment\model2_class_count_check.csv`
- `reports\model2_risk_assessment\model2_feature_completeness.csv`
- `reports\model2_risk_assessment\model2_inference_benchmark.csv`

## Analyze Dataset Imbalance

Use this when checking whether the source files are imbalanced or missing classes:

```powershell
python .\analyze_dataset_balance.py `
  --base-dir . `
  --output-dir .\reports\dataset_balance
```

This creates:

- `reports\dataset_balance\dataset_balance_report.md`
- `reports\dataset_balance\windows_all_class_distribution.png`
- `reports\dataset_balance\class_distribution_across_files.png`
- `reports\dataset_balance\dataset_value_counts.csv`
- `reports\dataset_balance\dataset_missing_values.csv`

## Main Artifacts

After training:

- `models\model2_risk_assessment\model2_risk_bundle.joblib`
- `reports\model2_risk_assessment\summary_report.md`
- `reports\model2_risk_assessment\risk_formula_config.json`
- `reports\model2_risk_assessment\model2_training_targets.csv`
- `reports\model2_risk_assessment\test_predictions_preview.csv`
- `reports\model2_risk_assessment\model2_readiness_report.md`

## How To Explain To Seniors

Short explanation:

> Model 1 is fall/no-fall detection. Model 2 is an assessment layer that converts IMU/PPG window features such as jerk, omega, tilt, GSI, and FCRI into a continuous risk score. Since we do not yet have clinical fall-risk labels, the current Model 2 uses a transparent proxy target and should be validated with real senior movement data later.
