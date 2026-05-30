"""Model registry — save, load, version management."""

import logging
import pickle
from datetime import datetime
from pathlib import Path

from src.config import get_config

logger = logging.getLogger(__name__)


def save_model(model, name: str, metadata: dict | None = None,
               model_dir: str | None = None) -> Path:
    """Save model + metadata to disk."""
    if model_dir is None:
        model_dir = get_config()["paths"]["models"]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pkl"
    filepath = model_dir / filename

    artifact = {
        "model": model,
        "name": name,
        "timestamp": timestamp,
        "metadata": metadata or {},
    }

    with open(filepath, "wb") as f:
        pickle.dump(artifact, f)

    logger.info("Saved model '%s' to %s", name, filepath)
    return filepath


def load_model(filepath: str | Path) -> dict:
    """Load model artifact from disk."""
    with open(filepath, "rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            f"File {filepath} is not a valid model artifact (got {type(artifact).__name__}). "
            "Expected a dict with 'model' key."
        )
    logger.info("Loaded model '%s' from %s", artifact["name"], filepath)
    return artifact


def get_latest_model(name: str | None = None, model_dir: str | None = None) -> dict | None:
    """Load the latest saved model, optionally filtered by name.

    Only looks for timestamped model files (e.g. gradient_boosting_20260527_155957.pkl),
    excluding non-model artifacts like scaler.pkl and feature_names.pkl.
    """
    if model_dir is None:
        model_dir = get_config()["paths"]["models"]
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    # Timestamped model files always follow the pattern *_YYYYMMDD_HHMMSS.pkl
    pattern = f"{name}_[0-9]*.pkl" if name else "*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*.pkl"
    files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        logger.warning("No saved models found matching '%s'", pattern)
        return None

    return load_model(files[0])


def save_scaler(scaler, model_dir: str | None = None) -> Path:
    """Save the fitted scaler for inference."""
    if model_dir is None:
        model_dir = get_config()["paths"]["models"]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filepath = model_dir / "scaler.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", filepath)
    return filepath


def load_scaler(model_dir: str | None = None):
    """Load fitted scaler."""
    if model_dir is None:
        model_dir = get_config()["paths"]["models"]
    filepath = Path(model_dir) / "scaler.pkl"
    with open(filepath, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def save_feature_names(feature_names: list[str], model_dir: str | None = None) -> Path:
    """Save feature names for inference consistency."""
    if model_dir is None:
        model_dir = get_config()["paths"]["models"]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filepath = model_dir / "feature_names.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(feature_names, f)
    logger.info("Saved %d feature names to %s", len(feature_names), filepath)
    return filepath


def load_feature_names(model_dir: str | None = None) -> list[str]:
    """Load feature names."""
    if model_dir is None:
        model_dir = get_config()["paths"]["models"]
    filepath = Path(model_dir) / "feature_names.pkl"
    with open(filepath, "rb") as f:
        return pickle.load(f)
