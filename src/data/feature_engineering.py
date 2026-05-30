"""Feature engineering — extract meaningful features reflecting equipment health."""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


def add_rolling_features(pdf: pd.DataFrame, sensor_cols: list[str],
                         windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling mean and std for sensor columns."""
    if windows is None:
        windows = get_config()["feature_engineering"]["rolling_windows"]

    for col in sensor_cols:
        for w in windows:
            pdf[f"{col}_rolling_mean_{w}"] = pdf[col].rolling(window=w, min_periods=1).mean()
            pdf[f"{col}_rolling_std_{w}"] = pdf[col].rolling(window=w, min_periods=1).std().fillna(0)
    logger.info("Added rolling features: windows=%s, columns=%d", windows, len(sensor_cols))
    return pdf


def add_rate_of_change(pdf: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """Add rate of change (first derivative) features."""
    for col in sensor_cols:
        pdf[f"{col}_roc"] = pdf[col].diff().fillna(0)
    logger.info("Added rate-of-change features for %d columns", len(sensor_cols))
    return pdf


def add_interaction_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific interaction features for equipment health."""
    # Temperature difference — indicates heat dissipation efficiency
    if "Air temperature [K]" in pdf.columns and "Process temperature [K]" in pdf.columns:
        pdf["temp_diff"] = pdf["Process temperature [K]"] - pdf["Air temperature [K]"]

    # Power — approximation of mechanical power output
    if "Rotational speed [rpm]" in pdf.columns and "Torque [Nm]" in pdf.columns:
        pdf["power"] = pdf["Rotational speed [rpm]"] * pdf["Torque [Nm]"]

    # Torque-to-speed ratio — indicates load condition
    if "Rotational speed [rpm]" in pdf.columns and "Torque [Nm]" in pdf.columns:
        pdf["torque_speed_ratio"] = pdf["Torque [Nm]"] / (pdf["Rotational speed [rpm]"] + 1e-8)

    # Tool wear rate interaction
    if "Tool wear [min]" in pdf.columns and "Torque [Nm]" in pdf.columns:
        pdf["wear_torque"] = pdf["Tool wear [min]"] * pdf["Torque [Nm]"]

    # Overstrain indicator
    if "Tool wear [min]" in pdf.columns and "power" in pdf.columns:
        pdf["overstrain"] = pdf["Tool wear [min]"] * pdf["power"]

    logger.info("Added interaction features: temp_diff, power, torque_speed_ratio, wear_torque, overstrain")
    return pdf


def add_anomaly_indicators(pdf: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """Add z-score based anomaly indicator flags."""
    for col in sensor_cols:
        mean = pdf[col].mean()
        std = pdf[col].std()
        if std > 0:
            z = (pdf[col] - mean) / std
            pdf[f"{col}_anomaly"] = (z.abs() > 2).astype(int)
    logger.info("Added anomaly indicators for %d columns", len(sensor_cols))
    return pdf


def add_usage_cycle_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """Add usage cycle features based on tool wear patterns."""
    if "Tool wear [min]" not in pdf.columns:
        return pdf

    # Bin tool wear into lifecycle stages
    pdf["wear_stage"] = pd.cut(
        pdf["Tool wear [min]"],
        bins=[0, 50, 100, 150, 200, float("inf")],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(0).astype(int)

    # Cumulative operational time indicator
    pdf["cumulative_usage"] = pdf["Tool wear [min]"].cumsum() / len(pdf)

    logger.info("Added usage cycle features: wear_stage, cumulative_usage")
    return pdf


def run_feature_engineering(ddf: dd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Converts Dask → Pandas for feature computation.

    Returns a pandas DataFrame with all engineered features.
    """
    cfg = get_config()
    sensor_cols = cfg["data"]["sensor_columns"]

    logger.info("=== Starting Feature Engineering ===")

    # Convert to pandas for feature computation (dataset is small enough)
    pdf = ddf.compute()

    # Determine which sensor columns are still present after preprocessing
    available_sensors = [c for c in sensor_cols if c in pdf.columns]

    pdf = add_interaction_features(pdf)
    pdf = add_rolling_features(pdf, available_sensors)
    pdf = add_rate_of_change(pdf, available_sensors)
    pdf = add_anomaly_indicators(pdf, available_sensors)
    pdf = add_usage_cycle_features(pdf)

    logger.info("=== Feature Engineering Complete. Shape: %s ===", pdf.shape)
    return pdf
