"""CLI script — Run streaming simulation standalone."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data.loader import load_raw_data
from src.data.preprocessing import run_preprocessing
from src.data.feature_engineering import run_feature_engineering
from src.models.registry import get_latest_model, load_scaler, load_feature_names
from src.monitoring.stream import StreamSimulator


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("stream")

    # Load model
    artifact = get_latest_model()
    if artifact is None:
        logger.error("No saved model found. Run 'python main.py' first.")
        sys.exit(1)

    model = artifact["model"]
    scaler = load_scaler()
    feature_names = load_feature_names()

    # Load and prepare data
    ddf = load_raw_data()
    ddf = run_preprocessing(ddf)
    pdf = run_feature_engineering(ddf)

    # Run simulation
    simulator = StreamSimulator(
        data=pdf, model=model, scaler=scaler, feature_names=feature_names,
    )

    def on_batch(result):
        if result["high_risk_count"] > 0:
            logger.warning("⚠️  %d high-risk detections in batch!", result["high_risk_count"])

    logger.info("Starting stream simulation... Press Ctrl+C to stop.")
    try:
        simulator.run(callback=on_batch)
    except KeyboardInterrupt:
        simulator.stop()
        logger.info("Stopped.")
        stats = simulator.get_stats()
        logger.info("Final stats: %s", stats)


if __name__ == "__main__":
    main()
