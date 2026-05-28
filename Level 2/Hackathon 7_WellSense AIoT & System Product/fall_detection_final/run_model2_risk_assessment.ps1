$ErrorActionPreference = "Stop"

python .\train_model2_risk_assessment.py `
  --data .\windows_all.csv `
  --run-name model2_risk_assessment

python .\predict_model2_risk_assessment.py `
  --model .\models\model2_risk_assessment\model2_risk_bundle.joblib `
  --input .\windows_all.csv `
  --output .\reports\model2_risk_assessment\model2_predictions_windows_all.csv `
  --json-output .\reports\model2_risk_assessment\model2_predictions_windows_all.json

python .\evaluate_model2_readiness.py `
  --data .\windows_all.csv `
  --model .\models\model2_risk_assessment\model2_risk_bundle.joblib `
  --output-dir .\reports\model2_risk_assessment
