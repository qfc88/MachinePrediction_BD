"""Data preprocessing — cleaning, encoding, scaling with Dask."""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import get_config

logger = logging.getLogger(__name__)


def handle_missing_values(ddf: dd.DataFrame) -> dd.DataFrame:
    """Handle missing values: numeric → median, categorical → mode."""
    cfg = get_config()["data"]
    sensor_cols = cfg["sensor_columns"]
    cat_cols = cfg["categorical_columns"]

    # Compute fill values once (small dataset, safe to compute)
    pdf = ddf[sensor_cols].describe().compute()
    medians = pdf.loc["50%"]

    for col in sensor_cols:
        if ddf[col].isnull().sum().compute() > 0:
            ddf[col] = ddf[col].fillna(medians[col])
            logger.info("Filled missing in '%s' with median %.2f", col, medians[col])

    for col in cat_cols:
        if ddf[col].isnull().sum().compute() > 0:
            mode_val = ddf[col].mode().compute().iloc[0]
            ddf[col] = ddf[col].fillna(mode_val)
            logger.info("Filled missing in '%s' with mode '%s'", col, mode_val)

    return ddf


def remove_outliers(ddf: dd.DataFrame, factor: float = 3.0) -> dd.DataFrame:
    """Remove outliers using IQR method on sensor columns."""
    cfg = get_config()["data"]
    sensor_cols = cfg["sensor_columns"]

    stats = ddf[sensor_cols].describe().compute()
    mask = None

    for col in sensor_cols:
        q1 = stats.loc["25%", col]
        q3 = stats.loc["75%", col]
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        col_mask = (ddf[col] >= lower) & (ddf[col] <= upper)
        mask = col_mask if mask is None else mask & col_mask

    before = len(ddf)
    ddf = ddf[mask]
    after = len(ddf)
    logger.info("Outlier removal: %d → %d rows (removed %d)", before, after, before - after)
    return ddf


def encode_categoricals(ddf: dd.DataFrame) -> dd.DataFrame:
    """One-hot encode categorical columns using Dask get_dummies."""
    cfg = get_config()["data"]
    cat_cols = cfg["categorical_columns"]

    # Dask requires categorical dtype before get_dummies
    ddf = ddf.categorize(columns=cat_cols)
    ddf = dd.get_dummies(ddf, columns=cat_cols, prefix=cat_cols)
    logger.info("Encoded categoricals: %s", cat_cols)
    return ddf


def scale_features(pdf: pd.DataFrame, sensor_cols: list[str], fit: bool = True,
                   scaler: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScale sensor columns. Returns (df, fitted_scaler).

    When fit=True, fits a new scaler. When fit=False, uses provided scaler (for inference).
    """
    if fit:
        scaler = StandardScaler()
        pdf[sensor_cols] = scaler.fit_transform(pdf[sensor_cols])
        logger.info("Fitted and applied StandardScaler on %d columns", len(sensor_cols))
    else:
        pdf[sensor_cols] = scaler.transform(pdf[sensor_cols])
        logger.info("Applied existing StandardScaler on %d columns", len(sensor_cols))
    return pdf, scaler


def drop_id_columns(ddf: dd.DataFrame) -> dd.DataFrame:
    """Drop ID columns that shouldn't be used as features."""
    cfg = get_config()["data"]
    id_cols = [c for c in cfg["id_columns"] if c in ddf.columns]
    if id_cols:
        ddf = ddf.drop(columns=id_cols)
        logger.info("Dropped ID columns: %s", id_cols)
    return ddf


def run_preprocessing(ddf: dd.DataFrame) -> dd.DataFrame:
    """Full preprocessing pipeline: missing → outliers → drop IDs → encode."""
    logger.info("=== Starting Preprocessing Pipeline ===")
    ddf = handle_missing_values(ddf)
    ddf = remove_outliers(ddf)
    ddf = drop_id_columns(ddf)
    ddf = encode_categoricals(ddf)
    logger.info("=== Preprocessing Complete. Columns: %s ===", list(ddf.columns))
    return ddf
