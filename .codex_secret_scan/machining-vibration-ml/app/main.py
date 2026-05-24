"""Optional FastAPI microservice for single-window wear prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_extraction import extract_window_features  # noqa: E402
from physics_features import extract_physics_features  # noqa: E402


MODEL_PATH = ROOT / "results" / "physics_guided_model.joblib"
app = FastAPI(title="Milling Wear Prediction API")


class PredictionRequest(BaseModel):
    sample_rate_hz: float = Field(..., gt=0)
    speed_rpm: float | None = None
    feed: float | None = None
    depth_of_cut: float | None = None
    tooth_count: int = 4
    vib_spindle: list[float]


def _load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Train a model first with: python evaluate.py --demo")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    artifact = _load_model()
    window = pd.DataFrame({"vib_spindle": payload.vib_spindle})
    features = extract_window_features(window, payload.sample_rate_hz)
    features.update(
        extract_physics_features(
            window["vib_spindle"].to_numpy(dtype=float),
            payload.sample_rate_hz,
            speed_rpm=payload.speed_rpm,
            tooth_count=payload.tooth_count,
        )
    )
    feature_columns = artifact["feature_columns"]
    row = pd.DataFrame([{column: features.get(column, 0.0) for column in feature_columns}])
    model = artifact["model"]
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    return {
        "prediction": prediction,
        "probabilities": {label: float(prob) for label, prob in zip(classes, probabilities)},
    }

