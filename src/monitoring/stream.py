"""Near-real-time streaming simulation using Dask."""

import logging
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import get_config
from src.monitoring.alerting import AlertManager

logger = logging.getLogger(__name__)


class StreamSimulator:
    """Simulates near-real-time equipment sensor data streaming.

    Reads from the dataset in configurable batches, mimicking
    a Kafka-like ingestion pipeline through Dask.
    """

    def __init__(self, data: pd.DataFrame, model, scaler, feature_names: list[str],
                 preprocess_fn=None):
        cfg = get_config()["monitoring"]
        self.data = data.reset_index(drop=True)
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.preprocess_fn = preprocess_fn
        self.batch_size = cfg["batch_size"]
        self.interval = cfg["stream_interval"]
        self.threshold = cfg["failure_threshold"]
        self.alert_manager = AlertManager()

        # State
        self.current_idx = 0
        self.predictions_log = deque(maxlen=5000)
        self.is_running = False

    def get_next_batch(self) -> pd.DataFrame | None:
        """Get next batch of sensor data (simulating stream ingestion)."""
        if self.current_idx >= len(self.data):
            self.current_idx = 0  # Loop for continuous simulation

        end_idx = min(self.current_idx + self.batch_size, len(self.data))
        batch = self.data.iloc[self.current_idx:end_idx].copy()
        self.current_idx = end_idx
        return batch

    def predict_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a batch. Returns batch with predictions appended."""
        # Align features with training set
        features = batch.reindex(columns=self.feature_names, fill_value=0)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale sensor columns
        sensor_cols = [c for c in get_config()["data"]["sensor_columns"] if c in features.columns]
        if sensor_cols and self.scaler is not None:
            features[sensor_cols] = self.scaler.transform(features[sensor_cols])

        # Predict
        proba = self.model.predict_proba(features)[:, 1]
        batch = batch.copy()
        batch["failure_probability"] = proba
        batch["predicted_failure"] = (proba >= self.threshold).astype(int)
        batch["prediction_time"] = datetime.now().isoformat()
        return batch

    def process_batch(self, batch: pd.DataFrame) -> dict:
        """Process one batch: predict + alert + log."""
        result = self.predict_batch(batch)

        # Log predictions
        for _, row in result.iterrows():
            self.predictions_log.append(row.to_dict())

        # Check alerts
        high_risk = result[result["failure_probability"] >= self.threshold]
        alerts = []
        for idx, row in high_risk.iterrows():
            eq_id = row.get("UDI") if "UDI" in row and pd.notna(row.get("UDI")) else f"MACHINE-{(int(idx) % 20) + 1:02d}"
            alert = self.alert_manager.check_and_alert(
                equipment_id=str(eq_id),
                probability=row["failure_probability"],
            )
            if alert:
                alerts.append(alert)

        return {
            "batch_size": len(batch),
            "predictions": result,
            "high_risk_count": len(high_risk),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
        }

    def run(self, max_batches: int | None = None, callback=None):
        """Run streaming loop. Optional callback(result) for each batch."""
        self.is_running = True
        batch_count = 0
        logger.info("Starting stream simulation (batch_size=%d, interval=%ds)",
                     self.batch_size, self.interval)

        try:
            while self.is_running:
                batch = self.get_next_batch()
                if batch is None or batch.empty:
                    break

                result = self.process_batch(batch)
                batch_count += 1

                if callback:
                    callback(result)

                logger.info("Batch %d: %d samples, %d high-risk, %d alerts",
                           batch_count, result["batch_size"],
                           result["high_risk_count"], len(result["alerts"]))

                if max_batches and batch_count >= max_batches:
                    break

                time.sleep(self.interval)
        finally:
            self.is_running = False
            logger.info("Stream simulation stopped after %d batches.", batch_count)

    def stop(self):
        self.is_running = False

    def get_stats(self) -> dict:
        """Get current streaming statistics."""
        if not self.predictions_log:
            return {"total": 0, "failure_rate": 0, "avg_probability": 0}

        probs = [p["failure_probability"] for p in self.predictions_log]
        failures = sum(1 for p in probs if p >= self.threshold)
        return {
            "total_predictions": len(self.predictions_log),
            "failure_rate": failures / len(self.predictions_log),
            "avg_probability": np.mean(probs),
            "max_probability": max(probs),
            "total_alerts": self.alert_manager.total_alerts,
        }
