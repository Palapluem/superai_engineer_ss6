# Dulae Model 2 Deploy Bundle

This bundle contains the deploy-ready Model 2 risk assessment API and the dashboard patch files.

## Folder Structure

```text
deploy_model2_fastapi_bundle/
  model2_api/
    model2_fastapi_app.py
    model2_service.py
    model2_live_features.py
    model2_risk_common.py
    build_features_from_raw.py
    requirements_ml.txt
    run_model2_api.ps1
    models/model2_risk_assessment/model2_risk_bundle.joblib
    reports/model2_risk_assessment/model2_readiness_report.md
    reports/model2_risk_assessment/model2_inference_benchmark.csv
    reports/model2_risk_assessment/risk_formula_config.json
  dashboard_patch/
    hooks/useMockWebSocket.ts
    lib/utils.ts
    lib/ai-engine.ts
    app/page.tsx
    components/RiskRoomSummaryCard.tsx
    components/RiskScoreCard.tsx
    components/floorplan/CondoFloorplanMap.tsx
```

## Model 2 Thresholds

```text
low    : 0.00 - 0.35
medium : > 0.35 - 0.70
high   : > 0.70
alert  : score > 0.70 and probability >= 0.70
```

The API includes a conservative guard to reduce false alerts from small board movement.

## Start Backend API

Open PowerShell in `model2_api`:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model2_api.ps1
```

Default URLs:

```text
API health: http://127.0.0.1:8000/health
API docs  : http://127.0.0.1:8000/docs
Dashboard latest reading: http://127.0.0.1:8000/dashboard/latest
```

## Test Raw IMU Prediction

```powershell
$samples = @()
for ($i = 0; $i -lt 40; $i++) {
  $samples += @{ ms = $i * 50; ax = 0.01 * ($i % 2); ay = 0.01; az = 1.0; gx = 0.5; gy = 0.2; gz = 0.1 }
}

$body = @{
  samples = $samples
  meta = @{ session_id = "deploy_calm_test"; room = "Hallway"; x = 286; y = 292 }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/model2/predict-raw `
  -ContentType "application/json" `
  -Body $body
```

Expected calm-window behavior:

```text
model2_risk_level = low
model2_alert = false
near_fall = false
```

## Dashboard Patch

Copy files from `dashboard_patch` into the dashboard project using the same relative paths.

Example:

```text
dashboard_patch/hooks/useMockWebSocket.ts
-> fall-detect-dashboard/hooks/useMockWebSocket.ts
```

After applying the patch, the dashboard polls:

```text
http://localhost:8000/dashboard/latest
```

If the API is not running, the dashboard falls back to mock data.

## Notes For Deployment

- The only required model file is `models/model2_risk_assessment/model2_risk_bundle.joblib`.
- `model2_fastapi_app.py` is the FastAPI entrypoint.
- `model2_service.py` contains the model loading and response mapping.
- `model2_risk_common.py` contains thresholds and conservative guard logic.
- `model2_live_features.py` converts raw IMU samples to Model 2 feature windows.
- `build_features_from_raw.py` is required by `model2_live_features.py`.
