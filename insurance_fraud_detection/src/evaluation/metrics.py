"""
evaluation/metrics.py — Compute and print all evaluation metrics.

Why not just accuracy?
  In fraud detection, the dataset is heavily imbalanced (~6% fraud).
  A model that always predicts 'not fraud' gets 94% accuracy — but is useless.
  We care about:
    - Recall (fraud)    : of all actual frauds, how many did we catch?  ← most important
    - Precision (fraud) : of all fraud predictions, how many were correct?
    - F1 (fraud)        : harmonic mean of precision and recall
    - ROC-AUC           : overall discrimination ability at all thresholds
"""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             threshold: float = 0.40) -> dict:
    """
    Full evaluation of a trained model.

    Parameters
    ----------
    threshold : decision boundary (default 0.40, not 0.50).
                Lower threshold = higher recall = catch more fraud.
                This is the key tuning lever for fraud detection.

    Returns
    -------
    metrics dict with all computed scores
    """
    # Raw probability of fraud
    y_prob = model.predict(X_test, verbose=0).flatten()

    # Apply threshold to get binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=["Legitimate", "Fraud"],
                                   output_dict=True)

    fraud_precision = report["Fraud"]["precision"]
    fraud_recall    = report["Fraud"]["recall"]
    fraud_f1        = report["Fraud"]["f1-score"]
    accuracy        = report["accuracy"]

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy,
        "fraud_precision": fraud_precision,
        "fraud_recall": fraud_recall,
        "fraud_f1": fraud_f1,
        "roc_auc": auc,
        "avg_precision": ap,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    _print_summary(metrics)
    return metrics


def _print_summary(m: dict):
    log.info("\n" + "="*55)
    log.info(f"  Decision threshold : {m['threshold']:.2f}")
    log.info(f"  Accuracy           : {m['accuracy']*100:.2f}%")
    log.info(f"  Fraud Precision    : {m['fraud_precision']*100:.2f}%")
    log.info(f"  Fraud Recall       : {m['fraud_recall']*100:.2f}%  ← KEY METRIC")
    log.info(f"  Fraud F1           : {m['fraud_f1']*100:.2f}%")
    log.info(f"  ROC-AUC            : {m['roc_auc']:.4f}")
    log.info(f"  Avg Precision (PR) : {m['avg_precision']:.4f}")
    cm = m["confusion_matrix"]
    log.info(f"\n  Confusion Matrix:")
    log.info(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    log.info(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    log.info(f"\n  FN = missed frauds — keep this low!")
    log.info("="*55 + "\n")
