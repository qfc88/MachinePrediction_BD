"""src.models package — training, evaluation, registry."""

from src.models.trainer import prepare_data, train_all_models, train_single_model
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
from src.models.registry import (
    save_model,
    load_model,
    get_latest_model,
    save_scaler,
    load_scaler,
    save_feature_names,
    load_feature_names,
)
