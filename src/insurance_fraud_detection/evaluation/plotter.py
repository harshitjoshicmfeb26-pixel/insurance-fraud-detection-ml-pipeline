"""
evaluation/plotter.py — All plots for training history, evaluation, and comparison.

Generates:
  1. Loss & accuracy curves  (training vs validation)
  2. Confusion matrix heatmap
  3. ROC curve
  4. Precision-Recall curve
  5. Model comparison bar chart
  6. Threshold sensitivity plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

from ..config import PLOTS_DIR
from ..utils.logger import get_logger

log = get_logger(__name__)

# ── Shared plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE   = "#378ADD"
RED    = "#E24B4A"
GREEN  = "#1D9E75"
AMBER  = "#BA7517"


def plot_training_history(history, title: str = "Training History"):
    """
    Plot loss, accuracy, recall, and AUC curves over epochs.
    Helps diagnose: overfitting (train >> val), underfitting (both low).
    """
    h = history.history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, y=1.01)

    pairs = [
        ("loss",     "val_loss",     "Loss",           RED),
        ("accuracy", "val_accuracy", "Accuracy",       BLUE),
        ("recall",   "val_recall",   "Recall (Fraud)", GREEN),
        ("auc",      "val_auc",      "ROC-AUC",        AMBER),
    ]

    for ax, (train_key, val_key, label, color) in zip(axes.flat, pairs):
        if train_key in h:
            ax.plot(h[train_key], color=color, lw=1.5, label="Train")
        if val_key in h:
            ax.plot(h[val_key], color=color, lw=1.5, ls="--",
                    label="Validation", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    _save("training_history.png")


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Annotated heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))

    labels = np.array([
        [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
        [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"],
    ])

    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"],
                linewidths=0.5, ax=ax)

    ax.set_title(title, fontsize=13)

    # Annotate TN/FP/FN/TP meanings
    ax.set_xlabel("FN = missed fraud (costly!) | FP = false alarm (acceptable)")
    plt.tight_layout()
    _save("confusion_matrix.png")


def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray,
                   auc_score: float, model_name: str = "Tuned ANN"):
    """
    ROC curve: FPR vs TPR at all thresholds.
    Area under the curve (AUC) = single number representing overall performance.
    Random classifier has AUC = 0.5 (diagonal line).
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2,
            label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    _save("roc_curve.png")


def plot_precision_recall(y_test: np.ndarray, y_prob: np.ndarray,
                          avg_precision: float):
    """
    Precision-Recall curve — more informative than ROC for imbalanced data.
    Shows the tradeoff: higher recall → lower precision (and vice versa).
    The vertical dashed line shows your chosen threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=GREEN, lw=2)
    ax.axvline(x=0.70, color=RED, ls="--", alpha=0.7,
               label="Target recall = 0.70")
    ax.fill_between(recall, precision, alpha=0.1, color=GREEN)
    ax.set_xlabel("Recall (fraud catch rate)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {avg_precision:.3f})")
    ax.legend()
    plt.tight_layout()
    _save("precision_recall_curve.png")


def plot_model_comparison(comparison_dict: dict):
    """
    Bar chart comparing Baseline vs Tuned ANN across key metrics.

    Parameters
    ----------
    comparison_dict : {
        'Model': ['Baseline', 'Tuned ANN'],
        'Accuracy': [0.88, 0.91],
        'Fraud Recall': [0.45, 0.72],
        'ROC-AUC': [0.78, 0.88],
    }
    """
    import pandas as pd
    df = pd.DataFrame(comparison_dict).set_index("Model")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df.columns))
    width = 0.35
    colors = [RED, BLUE, GREEN, AMBER]

    for i, (model_name, row) in enumerate(df.iterrows()):
        bars = ax.bar(x + i * width, row.values * 100,
                      width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df.columns)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Baseline vs Tuned ANN")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    _save("model_comparison.png")


def plot_threshold_sensitivity(y_test: np.ndarray, y_prob: np.ndarray):
    """
    Show how Precision and Recall change as we vary the decision threshold.
    Helps choose the right operating point for the business.
    """
    thresholds = np.linspace(0.1, 0.9, 80)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, recalls,    color=GREEN, lw=2, label="Recall (fraud)")
    ax.plot(thresholds, precisions, color=BLUE,  lw=2, label="Precision (fraud)")
    ax.plot(thresholds, f1s,        color=AMBER, lw=2, label="F1 (fraud)", ls="--")
    ax.axvline(x=best_t, color=RED, ls=":", alpha=0.7,
               label=f"Best F1 threshold = {best_t:.2f}")
    ax.axvline(x=0.40, color="gray", ls=":", alpha=0.5,
               label="Our threshold = 0.40")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity Analysis")
    ax.legend()
    plt.tight_layout()
    _save("threshold_sensitivity.png")

    log.info(f"Best F1 threshold: {best_t:.2f}  |  F1 = {f1s[best_idx]:.4f}")
    return best_t


def _save(filename: str):
    """Save figure to outputs/plots/ and close."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved → {path}")
    plt.show()
    plt.close()
