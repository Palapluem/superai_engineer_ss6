# Fall Detection Classical ML Workflow

This folder is prepared for the recommended first training path:

- Train Classical ML from `windows_all.csv`
- Use feature columns such as `svm_mean`, `jerk_mean`, `KII_mean`, `omega_mean`, `theta_range`, `GSI`, `fcri`
- Compare Random Forest, SVM-RBF, and XGBoost
- Save reusable model bundles and reports under `models/` and `reports/`

## Why Start Here

`windows_all.csv` is the best first dataset for tonight because it already contains window-level features for all available classes. The raw sequence files can still be useful later for LSTM/CNN-1D, but this dataset size is small for deep learning and the current hybrid file only covers `slow_collapse_fall`.

## Terminal Setup

Open PowerShell and run:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final"
python -m pip install -r requirements_ml.txt
```

PowerShell uses the backtick character for multi-line commands. Do not use `^` here because that is for `cmd.exe`.

## Train Binary Fall-Risk Model

This is the main model for the dashboard/edge demo:

```powershell
python .\train_classical_ml.py `
  --data .\windows_all.csv `
  --target label `
  --run-name binary_fall_risk
```

In this dataset:

- `label=1` means fall-risk/fall window
- `label=0` means non-fall activity or static activity

Important outputs:

- `models\binary_fall_risk\best_model.joblib`
- `reports\binary_fall_risk\summary_report.md`
- `reports\binary_fall_risk\model_comparison.csv`
- `reports\binary_fall_risk\confusion_matrix_<model>.png`
- `reports\binary_fall_risk\feature_importance_<model>.csv`

## Train Multiclass Activity/Fall Model

Use this when you want to show individual activity/fall classes:

```powershell
python .\train_classical_ml.py `
  --data .\windows_all.csv `
  --target class_en `
  --run-name multiclass_activity
```

This target includes classes such as:

- `normal_walk`
- `limping_walk`
- `stand_sit_alternating`
- `standing`
- `lying_down`
- `gradual_fall`
- `slow_collapse_fall`
- `sideways_fall`
- `backward_fall`

## Run Prediction On Feature Windows

After training, test the model on any CSV with the same feature columns:

```powershell
python .\predict_window.py `
  --model .\models\binary_fall_risk\best_model.joblib `
  --input .\windows_all.csv `
  --output .\reports\binary_fall_risk\predictions_windows_all.csv
```

The prediction output includes:

- `predicted_label`
- `confidence`
- `fall_risk_score`
- `risk_level`

## Real-Time Integration Idea

For live Arduino/Nano/UNO Q flow:

1. Collect raw IMU readings: `ax, ay, az, gx, gy, gz`
2. Segment readings into short windows
3. Compute the same 46 features used by `windows_all.csv`
4. Send one feature row into `predict_window.py` or a small Python inference service
5. Publish `fall_risk_score`, `risk_level`, activity label, and latest `(x, y)` to the dashboard

For microcontroller deployment, Random Forest is the easiest first candidate to convert to a lightweight rule/tree format. Keep XGBoost/SVM as comparison baselines unless the edge runtime is Python-capable.
