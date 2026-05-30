"""Model evaluation — metrics, cross-validation, visualization."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import get_config

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, name: str = "model") -> dict:
    """Evaluate a single model. Returns metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    logger.info("[%s] Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f AUC=%.4f",
                name, metrics["accuracy"], metrics["precision"],
                metrics["recall"], metrics["f1"], metrics["roc_auc"] or 0)
    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all trained models. Returns comparison DataFrame."""
    rows = []
    for name, info in trained_models.items():
        m = evaluate_model(info["model"], X_test, y_test, name)
        m["train_time"] = info["train_time"]
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    logger.info("Model comparison:\n%s", df.to_string())
    return df


def cross_validate_model(model, X, y, cv_folds: int = 5, scoring: str = "f1") -> dict:
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    result = {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores.tolist(),
    }
    logger.info("CV %s: %.4f ± %.4f", scoring, result["mean"], result["std"])
    return result


def get_best_model(trained_models: dict, comparison_df: pd.DataFrame):
    """Return the best model based on F1 score."""
    best_name = comparison_df.iloc[0]["name"]
    best_model = trained_models[best_name]["model"]
    logger.info("Best model: %s (F1=%.4f)", best_name, comparison_df.iloc[0]["f1"])
    return best_name, best_model


def plot_confusion_matrices(trained_models: dict, X_test, y_test, save_dir: str | None = None):
    """Plot confusion matrix for each model."""
    if save_dir is None:
        save_dir = Path(get_config()["paths"]["reports"]) / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(trained_models)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (name, info) in enumerate(trained_models.items()):
        y_pred = info["model"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Failure"]).plot(ax=axes[idx], cmap="Blues")
        axes[idx].set_title(name, fontsize=10)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrices to %s", save_dir / "confusion_matrices.png")


def plot_roc_curves(trained_models: dict, X_test, y_test, save_dir: str | None = None):
    """Plot ROC curves for all models on one figure."""
    if save_dir is None:
        save_dir = Path(get_config()["paths"]["reports"]) / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, info in trained_models.items():
        model = info["model"]
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves to %s", save_dir / "roc_curves.png")


def plot_precision_recall_curves(trained_models: dict, X_test, y_test, save_dir: str | None = None):
    """Plot Precision-Recall curves for all models."""
    if save_dir is None:
        save_dir = Path(get_config()["paths"]["reports"]) / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, info in trained_models.items():
        model = info["model"]
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curves to %s", save_dir / "precision_recall_curves.png")


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20,
                           save_dir: str | None = None):
    """Plot feature importance for tree-based models."""
    if save_dir is None:
        save_dir = Path(get_config()["paths"]["reports"]) / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.info("Model doesn't support feature_importances_, skipping.")
        return

    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(save_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance to %s", save_dir / "feature_importance.png")


def plot_model_comparison(comparison_df: pd.DataFrame, save_dir: str | None = None):
    """Plot bar chart comparing all models across metrics."""
    if save_dir is None:
        save_dir = Path(get_config()["paths"]["reports"]) / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_plot = comparison_df.set_index("name")[metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison to %s", save_dir / "model_comparison.png")
