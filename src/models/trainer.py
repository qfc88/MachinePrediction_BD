"""Model training pipeline — Dask-ML and Scikit-learn based."""

import logging
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import get_config
from src.data.preprocessing import scale_features

logger = logging.getLogger(__name__)

# Lazy imports for optional heavy dependencies
_xgb = None
_lgb = None


def _get_xgb():
    global _xgb
    if _xgb is None:
        import xgboost as xgb
        _xgb = xgb
    return _xgb


def _get_lgb():
    global _lgb
    if _lgb is None:
        import lightgbm as lgb
        _lgb = lgb
    return _lgb


ALGORITHM_MAP = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
    "xgboost": lambda: _get_xgb().XGBClassifier(
        n_estimators=200, scale_pos_weight=28, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
    "lightgbm": lambda: _get_lgb().LGBMClassifier(
        n_estimators=200, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
    ),
}


def prepare_data(pdf: pd.DataFrame) -> dict:
    """Split data into train/test with optional SMOTE for imbalance handling.

    Returns dict with X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    cfg = get_config()
    target = cfg["data"]["target_column"]
    failure_types = cfg["data"]["failure_type_columns"]
    test_size = cfg["data"]["test_size"]
    random_state = cfg["data"]["random_state"]

    # Separate features and target
    drop_cols = [target] + [c for c in failure_types if c in pdf.columns]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]
    X = pdf[feature_cols].copy()
    y = pdf[target].copy()

    # Ensure all numeric
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)

    # Handle any remaining NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale
    sensor_cols = [c for c in get_config()["data"]["sensor_columns"] if c in X.columns]
    X, scaler = scale_features(X, sensor_cols, fit=True)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Handle imbalance
    imbalance_method = cfg["training"]["handle_imbalance"]
    if imbalance_method == "smote":
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info("Applied SMOTE: %d → %d training samples", len(y) - len(y_test), len(y_train))

    logger.info("Data split: train=%d, test=%d, features=%d", len(X_train), len(X_test), len(feature_names))
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "feature_names": feature_names,
    }


def train_single_model(name: str, X_train, y_train) -> tuple:
    """Train a single model. Returns (name, model, training_time)."""
    if name not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {name}")

    model = ALGORITHM_MAP[name]()
    logger.info("Training %s...", name)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("%s trained in %.2fs", name, elapsed)
    return name, model, elapsed


def train_all_models(data: dict) -> dict:
    """Train all configured algorithms. Returns {name: (model, time)}."""
    cfg = get_config()
    algorithms = cfg["training"]["algorithms"]
    results = {}

    for algo in algorithms:
        try:
            name, model, elapsed = train_single_model(algo, data["X_train"], data["y_train"])
            results[name] = {"model": model, "train_time": elapsed}
        except Exception as e:
            logger.error("Failed to train %s: %s", algo, e)

    # Ensemble: Voting Classifier (top 3 models)
    if len(results) >= 3:
        top3 = sorted(results.items(), key=lambda x: x[1]["train_time"])[:3]
        estimators = [(n, r["model"]) for n, r in top3]
        voting = VotingClassifier(estimators=estimators, voting="soft")
        start = time.time()
        voting.fit(data["X_train"], data["y_train"])
        elapsed = time.time() - start
        results["voting_ensemble"] = {"model": voting, "train_time": elapsed}
        logger.info("Voting ensemble trained in %.2fs", elapsed)

    return results
