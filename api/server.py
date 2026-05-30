"""FastAPI prediction server — serves the trained model for real-time inference."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config
from src.models.registry import get_latest_model, load_scaler, load_feature_names

logger = logging.getLogger(__name__)

# Shared application state (loaded model, scaler, simulator)
app_state: dict = {}


def _load_model_artifacts():
    """Load latest model, scaler, and feature names into app_state."""
    try:
        artifact = get_latest_model()
        if artifact:
            app_state["model"] = artifact["model"]
            app_state["model_name"] = artifact["name"]
            logger.info("Loaded model: %s", artifact["name"])
        else:
            logger.warning("No saved model found. Prediction endpoints will return 503.")

        app_state["scaler"] = load_scaler()
        app_state["feature_names"] = load_feature_names()
        logger.info("Loaded scaler and feature names (%d features)", len(app_state["feature_names"]))
    except FileNotFoundError as e:
        logger.warning("Model artifacts not found: %s. Run training first.", e)


def _init_stream_simulator():
    """Initialize the streaming simulator with loaded model."""
    if app_state.get("model") is None:
        return

    try:
        import pandas as pd
        from src.monitoring.stream import StreamSimulator
        from src.config import get_config as _cfg

        # Use held-out test set (never seen during training) for honest streaming demo.
        # Falls back to full dataset if test set not yet generated (run main.py first).
        test_path = Path(_cfg()["paths"]["processed_data"]) / "stream_test_data.parquet"
        if test_path.exists():
            stream_data = pd.read_parquet(test_path)
            logger.info("Stream simulator using test set: %d rows (unseen during training)", len(stream_data))
        else:
            from src.data.loader import load_raw_data
            from src.data.preprocessing import run_preprocessing
            from src.data.feature_engineering import run_feature_engineering
            ddf = load_raw_data()
            ddf = run_preprocessing(ddf)
            stream_data = run_feature_engineering(ddf)
            logger.warning("stream_test_data.parquet not found — using full dataset. Run main.py to fix.")

        app_state["stream_simulator"] = StreamSimulator(
            data=stream_data,
            model=app_state["model"],
            scaler=app_state.get("scaler"),
            feature_names=app_state.get("feature_names", []),
        )
        logger.info("Stream simulator initialized.")
    except Exception as e:
        logger.error("Failed to initialize stream simulator: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model artifacts and init simulator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger.info("Starting Predictive Maintenance API...")
    _load_model_artifacts()
    _init_stream_simulator()
    yield
    logger.info("Shutting down API...")
    from src.data.loader import shutdown_dask
    shutdown_dask()


app = FastAPI(
    title="Predictive Maintenance API",
    description="Equipment failure prediction and real-time monitoring system powered by Dask.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from api.routes.monitor import router as monitor_router

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(monitor_router)


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("api.server:app", host=cfg["api"]["host"],
                port=cfg["api"]["port"], reload=False)
