from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model2_service import DEFAULT_MODEL_PATH, Model2RiskService


MODEL_PATH = os.getenv("MODEL2_MODEL_PATH", str(DEFAULT_MODEL_PATH))

app = FastAPI(
    title="Dulae Model 2 Risk Assessment API",
    version="1.0.0",
    description="Conservative mobility risk scoring for dashboard and hardware testing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = Model2RiskService(MODEL_PATH)
latest_dashboard_reading: dict[str, Any] | None = None


class FeatureWindowRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class FeatureBatchRequest(BaseModel):
    rows: list[dict[str, Any]]


class RawWindowRequest(BaseModel):
    samples: list[dict[str, Any]]
    meta: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "thresholds": service.config_payload()["thresholds"],
    }


@app.get("/model2/config")
def model2_config() -> dict[str, Any]:
    return service.config_payload()


@app.post("/model2/predict-window")
def predict_window(request: FeatureWindowRequest) -> dict[str, Any]:
    global latest_dashboard_reading
    try:
        row = {**request.features, **request.meta}
        result = service.predict_feature_rows([row])[0]
        latest_dashboard_reading = service.to_dashboard_reading(result, meta=request.meta)
        result["dashboard_reading"] = latest_dashboard_reading
        return result
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/model2/predict-batch")
def predict_batch(request: FeatureBatchRequest) -> dict[str, Any]:
    try:
        return {"predictions": service.predict_feature_rows(request.rows)}
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/model2/predict-raw")
def predict_raw(request: RawWindowRequest) -> dict[str, Any]:
    global latest_dashboard_reading
    try:
        result = service.predict_raw_samples(request.samples, request.meta)
        latest_dashboard_reading = result["dashboard_reading"]
        return result
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dashboard/latest")
def dashboard_latest() -> dict[str, Any]:
    if latest_dashboard_reading is not None:
        return latest_dashboard_reading
    return {
        "timestamp": "",
        "room": "Hallway",
        "x": 286,
        "y": 292,
        "gait_speed": 1.05,
        "sway": 1.4,
        "cadence": 106,
        "turning_velocity": 92,
        "instability_score": 18,
        "fall_risk": 18,
        "near_fall": False,
        "ax": 0,
        "ay": 0,
        "az": 1,
        "gx": 0,
        "gy": 0,
        "gz": 0,
    }
