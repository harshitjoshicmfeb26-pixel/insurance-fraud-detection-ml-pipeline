"""Consistent validation-thresholded benchmark for classical classifiers."""

import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from ..models.classical import create_model_registry
from ..config import ANN_V3_SAVE_PATH, BATCH_SIZE, CANONICAL_ANN_SAVE_PATH, EPOCHS
from ..utils.logger import get_logger
from .threshold_tuner import find_best_threshold

log = get_logger(__name__)


def _probabilities(model, X):
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not provide predict_proba")
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def metrics_from_probabilities(y_true, y_prob, threshold):
    """Calculate all benchmark metrics from frozen test probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }


def evaluate_probability_model(name, validation_labels, validation_probabilities,
                               test_labels, test_probabilities,
                               optimize_for="f1"):
    """Select a validation threshold, then evaluate frozen test probabilities."""
    threshold = find_best_threshold(validation_labels, validation_probabilities,
                                    optimize_for=optimize_for)
    metrics = metrics_from_probabilities(test_labels, test_probabilities, threshold)
    metrics["model"] = name
    return metrics


def run_classical_benchmark(prepared, random_state=42, optimize_for="f1"):
    """Fit all registry models on the shared prepared data and evaluate them."""
    required = {"X_train", "y_train", "X_validation", "y_validation", "X_test", "y_test"}
    missing = required.difference(prepared)
    if missing:
        raise KeyError(f"Prepared data is missing: {sorted(missing)}")

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_validation, y_validation = prepared["X_validation"], prepared["y_validation"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    models = create_model_registry(random_state)
    results = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        validation_prob = _probabilities(model, X_validation)
        test_prob = _probabilities(model, X_test)
        metrics = evaluate_probability_model(name, y_validation, validation_prob,
                                             y_test, test_prob, optimize_for)
        metrics["fit_seconds"] = round(time.perf_counter() - started, 3)
        metrics["validation_pr_auc"] = float(average_precision_score(y_validation, validation_prob))
        metrics["estimator"] = model
        results[name] = metrics
        log.info("%s: threshold=%.2f ROC-AUC=%.4f PR-AUC=%.4f", name,
                 metrics["threshold"], metrics["roc_auc"], metrics["pr_auc"])
    return results


def run_ann_benchmark(prepared, random_state=42, optimize_for="f1",
                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                      checkpoint_path=CANONICAL_ANN_SAVE_PATH):
    """Train the lineage-preserving ANN using validation-only model selection."""
    from ..models.ann_model import build_model, train
    from ..utils.seed import set_all_seeds

    required = {"X_train", "y_train", "X_validation", "y_validation", "X_test", "y_test"}
    missing = required.difference(prepared)
    if missing:
        raise KeyError(f"Prepared data is missing: {sorted(missing)}")
    set_all_seeds(random_state)
    model = build_model(prepared["X_train"].shape[1])
    history = train(model, prepared["X_train"], prepared["y_train"],
                    X_val=prepared["X_validation"], y_val=prepared["y_validation"],
                    batch_size=batch_size, epochs=epochs, class_weight=None,
                    checkpoint_path=checkpoint_path)
    validation_prob = model.predict(prepared["X_validation"], verbose=0).reshape(-1)
    test_prob = model.predict(prepared["X_test"], verbose=0).reshape(-1)
    metrics = evaluate_probability_model(
        "ANN", prepared["y_validation"], validation_prob,
        prepared["y_test"], test_prob, optimize_for)
    metrics["estimator"] = model
    metrics["history"] = history.history
    metrics["epochs_requested"] = epochs
    metrics["epochs_run"] = len(history.history.get("loss", []))
    metrics["checkpoint_path"] = checkpoint_path
    return metrics


def run_ann_v3_benchmark(prepared, random_state=42, optimize_for="f1",
                         checkpoint_path=ANN_V3_SAVE_PATH):
    """Run notebook-faithful v3 once under the corrected protocol."""
    from ..models.ann_model import train
    from ..models.ann_v3 import (V3_BATCH_SIZE, V3_CLASS_WEIGHT, V3_EPOCHS,
                                 V3_PATIENCE, build_v3)
    from ..utils.seed import set_all_seeds

    required = {"X_train", "y_train", "X_validation", "y_validation", "X_test", "y_test"}
    missing = required.difference(prepared)
    if missing:
        raise KeyError(f"Prepared data is missing: {sorted(missing)}")
    set_all_seeds(random_state)
    model = build_v3(prepared["X_train"].shape[1])
    history = train(
        model, prepared["X_train"], prepared["y_train"],
        X_val=prepared["X_validation"], y_val=prepared["y_validation"],
        batch_size=V3_BATCH_SIZE, epochs=V3_EPOCHS,
        class_weight=V3_CLASS_WEIGHT, checkpoint_path=checkpoint_path,
        monitor="val_auc", patience=V3_PATIENCE, include_reduce_lr=False,
    )
    validation_prob = model.predict(prepared["X_validation"], verbose=0).reshape(-1)
    test_prob = model.predict(prepared["X_test"], verbose=0).reshape(-1)
    metrics = evaluate_probability_model(
        "ANN v3", prepared["y_validation"], validation_prob,
        prepared["y_test"], test_prob, optimize_for)
    metrics.update({"estimator": model, "history": history.history,
                    "epochs_requested": V3_EPOCHS,
                    "epochs_run": len(history.history.get("loss", [])),
                    "checkpoint_path": checkpoint_path,
                    "smote_strategy": 0.15,
                    "class_weight": {"0": 1.0, "1": 5.0}})
    return metrics


def results_table(results):
    """Return JSON/CSV-friendly scalar rows, excluding fitted estimators/arrays."""
    fields = ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
              "pr_auc", "threshold", "tn", "fp", "fn", "tp", "fit_seconds"]
    return [{field: result[field] for field in fields} for result in results.values()]
