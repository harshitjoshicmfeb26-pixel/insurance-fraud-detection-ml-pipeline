"""Validation-only ANN search with an explicit one-time final evaluation."""

import os
import time

import numpy as np
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score, f1_score)

from ..models.ann_v3 import ANNOptimizationConfig, build_optimized_ann
from ..models.ann_model import train
from ..utils.seed import set_all_seeds
from .benchmark import metrics_from_probabilities
from .threshold_tuner import find_best_threshold


def _validate_search_arrays(X_train, y_train, X_validation, y_validation):
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Training and validation features/labels must have matching lengths")
    if len(X_train) == 0 or len(X_validation) == 0:
        raise ValueError("Training and validation partitions must not be empty")


def _validation_record(config, model, history, validation_labels, validation_probabilities):
    threshold = find_best_threshold(validation_labels, validation_probabilities,
                                    optimize_for="f1")
    validation_pred = (validation_probabilities >= threshold).astype(int)
    return {
        **config.to_dict(),
        "epochs_run": len(history.history.get("loss", [])),
        "validation_pr_auc": float(average_precision_score(
            validation_labels, validation_probabilities)),
        "validation_roc_auc": float(roc_auc_score(
            validation_labels, validation_probabilities)),
        "validation_precision": float(precision_score(
            validation_labels, validation_pred, zero_division=0)),
        "validation_recall": float(recall_score(
            validation_labels, validation_pred, zero_division=0)),
        "validation_f1": float(f1_score(
            validation_labels, validation_pred, zero_division=0)),
        "validation_threshold": float(threshold),
        "model": model,
        "history": history.history,
        "fit_seconds": None,
    }


def run_validation_candidate(config: ANNOptimizationConfig, X_train, y_train,
                             X_validation, y_validation, random_state=42,
                             checkpoint_path=None):
    """Fit/evaluate one candidate using only training and validation data."""
    _validate_search_arrays(X_train, y_train, X_validation, y_validation)
    set_all_seeds(random_state)
    model = build_optimized_ann(X_train.shape[1], config)
    started = time.perf_counter()
    history = train(
        model, X_train, y_train, X_val=X_validation, y_val=y_validation,
        batch_size=config.batch_size, epochs=200,
        class_weight={0: 1.0, 1: config.fraud_class_weight},
        checkpoint_path=checkpoint_path or os.path.join(
            "outputs", "models", "optimization_candidates",
            f"{config.candidate_id}.keras"),
        monitor="val_auc", patience=20, include_reduce_lr=False,
        verbose=0,
    )
    validation_probabilities = model.predict(X_validation, verbose=0).reshape(-1)
    result = _validation_record(config, model, history, np.asarray(y_validation).astype(int),
                                validation_probabilities)
    result["fit_seconds"] = round(time.perf_counter() - started, 3)
    return result


def validation_sort_key(result):
    """Predetermined selection rule: PR-AUC, then F1, then ROC-AUC."""
    return (result["validation_pr_auc"], result["validation_f1"],
            result["validation_roc_auc"])


def select_validation_winner(results):
    if not results:
        raise ValueError("At least one validation result is required")
    return max(results, key=validation_sort_key)


def evaluate_selected_ann_once(model, test_features, test_labels, threshold):
    """Evaluate the frozen validation-selected model on the final test set once."""
    if model is None:
        raise ValueError("A trained winner model is required")
    probabilities = model.predict(test_features, verbose=0).reshape(-1)
    return metrics_from_probabilities(test_labels, probabilities, threshold)
