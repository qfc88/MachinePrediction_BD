"""E2E pipeline orchestrator — train, evaluate, save, and optionally serve."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_config
from src.data.loader import load_raw_data, save_processed_data, get_dask_client, shutdown_dask
from src.data.preprocessing import run_preprocessing
from src.data.feature_engineering import run_feature_engineering
from src.models.trainer import prepare_data, train_all_models
from src.models.evaluator import (
    evaluate_all_models,
    cross_validate_model,
    get_best_model,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
    plot_model_comparison,
)
from src.models.registry import save_model, save_scaler, save_feature_names


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(get_config()["paths"]["logs"]) / "pipeline.log"),
        ],
    )


def run_pipeline():
    """Execute the full training pipeline end-to-end."""
    cfg = get_config()
    logger = logging.getLogger("pipeline")

    # ── Step 1: Init Dask ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Initializing Dask Cluster")
    logger.info("=" * 60)
    client = get_dask_client()
    logger.info("Dask dashboard: %s", client.dashboard_link)

    # ── Step 2: Load Data ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Loading Raw Data")
    logger.info("=" * 60)
    ddf = load_raw_data()

    # ── Step 3: Preprocess ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Preprocessing")
    logger.info("=" * 60)
    ddf = run_preprocessing(ddf)
    save_processed_data(ddf)

    # ── Step 4: Feature Engineering ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Feature Engineering")
    logger.info("=" * 60)
    pdf = run_feature_engineering(ddf)

    # ── Step 5: Prepare & Train ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Model Training")
    logger.info("=" * 60)
    data = prepare_data(pdf)
    trained_models = train_all_models(data)

    # Save training artifacts
    save_scaler(data["scaler"])
    save_feature_names(data["feature_names"])

    # Save test set for streaming simulation (data model never saw during training)
    processed_dir = Path(cfg["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_stream_path = processed_dir / "stream_test_data.parquet"
    test_rows = pdf.loc[data["X_test"].index].copy()
    test_rows.to_parquet(test_stream_path, index=False)
    logger.info("Saved test stream data: %d rows → %s", len(test_rows), test_stream_path)

    # ── Step 6: Evaluate ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Model Evaluation")
    logger.info("=" * 60)
    comparison_df = evaluate_all_models(trained_models, data["X_test"], data["y_test"])
    best_name, best_model = get_best_model(trained_models, comparison_df)

    # Cross-validate best model
    cv_result = cross_validate_model(
        best_model, data["X_train"], data["y_train"],
        cv_folds=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["scoring"],
    )
    logger.info("Best model CV: %.4f ± %.4f", cv_result["mean"], cv_result["std"])

    # ── Step 7: Save Best Model ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Saving Best Model")
    logger.info("=" * 60)
    save_model(best_model, best_name, metadata={
        "cv_f1_mean": cv_result["mean"],
        "cv_f1_std": cv_result["std"],
        "test_metrics": comparison_df[comparison_df["name"] == best_name].iloc[0].to_dict(),
    })

    # ── Step 8: Generate Visualizations ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Generating Visualizations")
    logger.info("=" * 60)
    plot_confusion_matrices(trained_models, data["X_test"], data["y_test"])
    plot_roc_curves(trained_models, data["X_test"], data["y_test"])
    plot_precision_recall_curves(trained_models, data["X_test"], data["y_test"])
    plot_model_comparison(comparison_df)

    # Feature importance for best model
    plot_feature_importance(best_model, data["feature_names"])

    # Save comparison table
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    # ── Done ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Best model: %s (F1=%.4f, AUC=%.4f)",
                best_name,
                comparison_df[comparison_df["name"] == best_name].iloc[0]["f1"],
                comparison_df[comparison_df["name"] == best_name].iloc[0]["roc_auc"])
    logger.info("Artifacts saved to: %s", cfg["paths"]["models"])
    logger.info("Reports saved to: %s", cfg["paths"]["reports"])
    logger.info("=" * 60)

    shutdown_dask()
    return comparison_df


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
