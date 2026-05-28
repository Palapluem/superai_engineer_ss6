# Model 2 FastAPI Integration

Model 2 is the risk assessment layer. It does not replace Model 1 fall/no-fall.
The API returns a conservative risk score for dashboard and hardware testing.

## Thresholds

- `low`: `0.00 - 0.35`
- `medium`: `> 0.35 - 0.70`
- `high`: `> 0.70`
- `model2_alert`: `model2_risk_score > 0.70` and `model2_high_risk_probability >= 0.70`

The alert threshold was moved from the old 0.5-style decision point to 0.7.
Small board movements are guarded by a post-processing check: a high alert needs
impact evidence or multiple elevated feature components, not only one noisy spike.

## Start API

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall_detection_final"

powershell -ExecutionPolicy Bypass -File .\run_model2_api.ps1
```

API URL:

```text
http://localhost:8000
```

Swagger/OpenAPI:

```text
http://localhost:8000/docs
```

## Health Check

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Predict From Feature Window

Use this when the device/edge script has already calculated 46 Model 2 features.

```powershell
$body = @{
  features = @{
    svm_mean = 1.02
    svm_std = 0.03
    svm_max = 1.10
    svm_min = 0.94
    jerk_mean = 0.8
    jerk_std = 0.4
    jerk_max = 2.2
    omega_mean = 12
    omega_std = 5
    omega_max = 30
    theta_range = 8
    GSI = 0.9
    fcri = 0.25
  }
  meta = @{
    session_id = "demo_001"
    room = "Hallway"
    x = 286
    y = 292
  }
} | ConvertTo-Json -Depth 6

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/model2/predict-window `
  -ContentType "application/json" `
  -Body $body
```

Missing feature columns are filled by the same median imputer inside the trained
pipeline. For production, send all 46 features when possible.

## Predict From Raw IMU Window

Use this when the Nano/edge device sends raw samples.

```powershell
$samples = @(
  @{ ms = 0;   ax = 0.00; ay = 0.01; az = 1.00; gx = 0.5; gy = 0.2; gz = 0.1 },
  @{ ms = 50;  ax = 0.01; ay = 0.02; az = 1.01; gx = 0.7; gy = 0.1; gz = 0.0 },
  @{ ms = 100; ax = 0.02; ay = 0.02; az = 1.00; gx = 0.4; gy = 0.2; gz = 0.2 }
)

$body = @{
  samples = $samples
  meta = @{
    session_id = "raw_demo_001"
    room = "Hallway"
    x = 286
    y = 292
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/model2/predict-raw `
  -ContentType "application/json" `
  -Body $body
```

The API also returns `dashboard_reading`, which matches the dashboard
`SensorReading` shape.

## Connect Dashboard

Start the dashboard normally:

```powershell
cd "C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\Hackathon 7_WellSense AIoT & System Product\fall-detect-dashboard"
npm run dev
```

The dashboard polls:

```text
http://localhost:8000/dashboard/latest
```

If the API is not running, the dashboard falls back to mock data. If the API is
running, every successful `/model2/predict-window` or `/model2/predict-raw` call
updates the dashboard's latest live reading.
