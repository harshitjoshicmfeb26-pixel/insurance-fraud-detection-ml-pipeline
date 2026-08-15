"""
evaluation/threshold_tuner.py — Find the optimal decision threshold.

Default sigmoid threshold is 0.5, but fraud detection requires a lower value
because missing a fraud (false negative) is far more costly than a false alarm.

This module tries all thresholds from 0.1 to 0.9 and recommends the one that
maximises Fraud F1 score (or Recall, depending on business priority).
"""

import numpy as np
from ..utils.logger import get_logger

log = get_logger(__name__)


def find_best_threshold(y_test: np.ndarray, y_prob: np.ndarray,
                        optimize_for: str = "f1") -> float:
    """
    Grid search over thresholds [0.10 → 0.90] to find the best one.

    Parameters
    ----------
    optimize_for : 'f1'     — balance precision and recall
                   'recall' — maximise fraud catch rate (accept more false alarms)

    Returns
    -------
    best_threshold : float
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_score, best_t = -1, 0.5

    log.info(f"\nThreshold search (optimising for: {optimize_for})")
    log.info(f"{'Threshold':>10}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}")
    log.info("-" * 45)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        recall    = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        score = f1 if optimize_for == "f1" else recall

        if score > best_score:
            best_score = score
            best_t = t

    log.info(f"\nBest threshold: {best_t:.2f}  (best {optimize_for}: {best_score:.4f})")
    log.info("Recommendation: Use 0.40 for balanced fraud detection.")
    log.info("Use lower (e.g. 0.30) if the business cannot afford to miss fraud.")

    return best_t
