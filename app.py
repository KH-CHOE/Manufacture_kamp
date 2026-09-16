"""FastAPI service for the Foundry Guard dashboard and prediction API."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from build_dashboard import MODEL_OUT
from storage import dashboard_payload, initialize, record_prediction, remote_enabled


BASE = Path(__file__).parent
DIST = BASE / "dist"
app = FastAPI(title="Foundry Guard API", version="0.1.0")


class PredictionRequest(BaseModel):
    values: dict[str, float | None] = Field(description="Sensor values keyed by feature name")
    event_at: datetime | None = None


def artifact() -> dict[str, Any]:
    if not MODEL_OUT.exists():
        raise HTTPException(503, "Model is not ready. Run: python3 scripts/train_model.py")
    return pickle.loads(MODEL_OUT.read_bytes())


@app.on_event("startup")
def startup() -> None:
    initialize()


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    return {"ok": True, "model_ready": MODEL_OUT.exists(), "storage": "supabase" if remote_enabled() else "sqlite"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return dashboard_payload(artifact())


@app.post("/api/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    saved = artifact()
    features = saved["features"]
    values = {feature: payload.values.get(feature, saved["medians"].get(feature)) for feature in features}
    frame = pd.DataFrame([values], columns=features).fillna(pd.Series(saved["medians"]))
    probability = float(saved["model"].predict_proba(frame)[0][1])
    specs = saved["specs"]
    outside = [feature for feature in specs if specs[feature]["low"] is not None and values.get(feature) is not None and (values[feature] < specs[feature]["low"] or values[feature] > specs[feature]["high"])]
    event_at = (payload.event_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    record_prediction(event_at, values, probability)
    return {"event_at": event_at, "defect_probability": round(probability, 4), "risk_level": "critical" if probability >= saved["threshold"] and any(specs[item]["grade"] == "A" for item in outside) else "warning" if probability >= saved["threshold"] or outside else "normal", "threshold": saved["threshold"], "out_of_band_features": outside}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DIST / "index.html")


app.mount("/", StaticFiles(directory=DIST), name="static")
