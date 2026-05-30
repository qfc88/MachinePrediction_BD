"""Prediction endpoints — single and batch prediction."""

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    SensorReading,
)
from src.config import get_config
from src.data.feature_engineering import add_interaction_features

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


def _prepare_features(reading: SensorReading, feature_names: list[str],
                      scaler) -> pd.DataFrame:
    """Convert a sensor reading into a feature vector aligned with training features."""
    # Build raw row
    raw = {
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
    }

    pdf = pd.DataFrame([raw])

    # Add interaction features (same as training)
    pdf = add_interaction_features(pdf)

    # One-hot encode Type
    for t in ["L", "M", "H"]:
        pdf[f"Type_{t}"] = int(reading.type == t)

    # Align to training feature set
    pdf = pdf.reindex(columns=feature_names, fill_value=0)
    pdf = pdf.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale sensor columns
    sensor_cols = [c for c in get_config()["data"]["sensor_columns"] if c in pdf.columns]
    if sensor_cols and scaler is not None:
        pdf[sensor_cols] = scaler.transform(pdf[sensor_cols])

    return pdf


def _get_risk_level(prob: float) -> str:
    if prob >= 0.9:
        return "critical"
    elif prob >= 0.7:
        return "high"
    elif prob >= 0.5:
        return "medium"
    return "low"


def _get_contributing_factors(features: pd.DataFrame, model, feature_names: list[str]) -> list[str]:
    """Identify top contributing factors using feature importance."""
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    return [feature_names[i] for i in top_idx if importances[i] > 0]


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(reading: SensorReading):
    """Predict failure probability for a single sensor reading."""
    from api.server import app_state

    model = app_state.get("model")
    scaler = app_state.get("scaler")
    feature_names = app_state.get("feature_names")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    features = _prepare_features(reading, feature_names, scaler)
    prob = model.predict_proba(features)[0, 1]
    threshold = get_config()["monitoring"]["failure_threshold"]

    return PredictionResponse(
        failure_probability=round(float(prob), 4),
        predicted_failure=prob >= threshold,
        risk_level=_get_risk_level(prob),
        contributing_factors=_get_contributing_factors(features, model, feature_names),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict failure for a batch of sensor readings."""
    from api.server import app_state

    model = app_state.get("model")
    scaler = app_state.get("scaler")
    feature_names = app_state.get("feature_names")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    predictions = []
    probs = []
    threshold = get_config()["monitoring"]["failure_threshold"]

    for reading in request.readings:
        features = _prepare_features(reading, feature_names, scaler)
        prob = float(model.predict_proba(features)[0, 1])
        probs.append(prob)
        predictions.append(PredictionResponse(
            failure_probability=round(prob, 4),
            predicted_failure=prob >= threshold,
            risk_level=_get_risk_level(prob),
            contributing_factors=_get_contributing_factors(features, model, feature_names),
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        batch_size=len(predictions),
        avg_failure_probability=round(np.mean(probs), 4) if probs else 0,
    )
