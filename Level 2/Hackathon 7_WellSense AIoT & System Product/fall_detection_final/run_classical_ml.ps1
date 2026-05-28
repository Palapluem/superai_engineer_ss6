$ErrorActionPreference = "Stop"

python -m pip install -r requirements_ml.txt

python .\train_classical_ml.py `
  --data .\windows_all.csv `
  --target label `
  --run-name binary_fall_risk

python .\train_classical_ml.py `
  --data .\windows_all.csv `
  --target class_en `
  --run-name multiclass_activity

python .\predict_window.py `
  --model .\models\binary_fall_risk\best_model.joblib `
  --input .\windows_all.csv `
  --output .\reports\binary_fall_risk\predictions_windows_all.csv
