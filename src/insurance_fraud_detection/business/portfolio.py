"""Label-dependent portfolio and investigator-capacity metrics."""

from math import ceil

import numpy as np
import pandas as pd


def _ordered(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if labels.size == 0:
        raise ValueError("labels and scores cannot be empty")
    order = np.argsort(-scores, kind="stable")
    return labels[order], scores[order]


def top_k_count(total_claims: int, percentage: float) -> int:
    """Use ceiling so a non-empty top segment is retained for small samples."""
    if total_claims <= 0 or not 0 < percentage <= 100:
        raise ValueError("total_claims must be positive and percentage must be in (0, 100]")
    return min(total_claims, max(1, ceil(total_claims * percentage / 100.0)))


def top_k_metrics(labels, scores, percentage: float) -> dict:
    """Calculate Fraud Capture, Precision, and Lift for the top K percent."""
    ordered_labels, _ = _ordered(labels, scores)
    count = top_k_count(len(ordered_labels), percentage)
    selected = ordered_labels[:count]
    total_fraud = int(ordered_labels.sum())
    fraud_selected = int(selected.sum())
    precision = fraud_selected / count
    prevalence = total_fraud / len(ordered_labels)
    return {
        "percentage": float(percentage),
        "claims_reviewed": count,
        "fraud_selected": fraud_selected,
        "total_claims": len(ordered_labels),
        "total_fraud": total_fraud,
        "fraud_capture": fraud_selected / total_fraud if total_fraud else 0.0,
        "precision": precision,
        "lift": precision / prevalence if prevalence else 0.0,
        "rounding": "ceil(K% × total claims)",
    }


def capacity_summary(labels, scores, percentages=(1, 5, 10, 20)) -> pd.DataFrame:
    return pd.DataFrame([top_k_metrics(labels, scores, p) for p in percentages])


def cumulative_gains(labels, scores, percentages=(1, 5, 10, 20, 30, 50, 100)) -> pd.DataFrame:
    return capacity_summary(labels, scores, percentages)[[
        "percentage", "claims_reviewed", "fraud_selected", "total_fraud",
        "fraud_capture"]].rename(columns={"fraud_capture": "fraud_captured_percentage"})


def lift_by_decile(labels, scores) -> pd.DataFrame:
    ordered_labels, _ = _ordered(labels, scores)
    total = len(ordered_labels)
    prevalence = ordered_labels.mean()
    deciles = np.ceil(np.arange(1, total + 1) * 10 / total).astype(int)
    rows = []
    for decile in range(1, 11):
        part = ordered_labels[deciles == decile]
        rate = float(part.mean()) if len(part) else 0.0
        rows.append({"decile": decile, "claims": len(part),
                     "fraud_cases": int(part.sum()), "fraud_rate": rate,
                     "lift": rate / prevalence if prevalence else 0.0})
    return pd.DataFrame(rows)
