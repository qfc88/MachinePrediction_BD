"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Single equipment sensor reading for prediction."""
    air_temperature: float = Field(..., alias="Air temperature [K]", description="Air temperature in Kelvin")
    process_temperature: float = Field(..., alias="Process temperature [K]", description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., alias="Rotational speed [rpm]", description="Rotational speed in RPM")
    torque: float = Field(..., alias="Torque [Nm]", description="Torque in Newton-meters")
    tool_wear: int = Field(..., alias="Tool wear [min]", description="Tool wear in minutes")
    type: str = Field("M", alias="Type", description="Product quality type: L, M, or H")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Prediction result for a single reading."""
    failure_probability: float
    predicted_failure: bool
    risk_level: str
    contributing_factors: list[str] = []


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    readings: list[SensorReading]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResponse]
    batch_size: int
    avg_failure_probability: float


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    dask_status: str
    version: str


class StreamStatus(BaseModel):
    """Streaming pipeline status."""
    is_running: bool
    total_predictions: int
    failure_rate: float
    avg_probability: float
    total_alerts: int
    recent_alerts: list[dict] = []
