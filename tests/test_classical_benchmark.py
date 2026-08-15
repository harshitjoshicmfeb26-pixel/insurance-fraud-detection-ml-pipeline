"""Lightweight tests for the five-model classical benchmark."""

import os
import sys
import inspect

import numpy as np
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from insurance_fraud_detection.config import (ANN_V3_SAVE_PATH,
                                               CANONICAL_ANN_SAVE_PATH,
                                               MODEL_SAVE_PATH,
                                               OPTIMIZED_ANN_SAVE_PATH)
from insurance_fraud_detection.evaluation.ann_optimization import (
    select_validation_winner, validation_sort_key)
from insurance_fraud_detection.evaluation.benchmark import (
    evaluate_probability_model,
    metrics_from_probabilities,
    results_table,
    run_classical_benchmark,
)
from insurance_fraud_detection.models.classical import MODEL_NAMES, create_model_registry
from insurance_fraud_detection.models.ann_v3 import (
    V3_BATCH_SIZE, V3_CLASS_WEIGHT, V3_EPOCHS, V3_HIDDEN_LAYERS,
    V3_L2_LAMBDA, V3_PATIENCE, V3_SMOTE_STRATEGY, build_v3,
)


def test_classical_registry_contains_all_required_models():
    registry = create_model_registry(random_state=7)
    assert list(registry) == MODEL_NAMES
    assert registry["SVM"].__class__.__name__ == "SVC"
    assert registry["SVM"].probability is True


def test_svc_is_fitted_and_returns_probabilities():
    registry = create_model_registry(random_state=7)
    X = np.random.RandomState(7).normal(size=(80, 6))
    y = np.tile([0, 1], 40)
    svc = registry["SVM"].fit(X, y)
    check_is_fitted(svc)
    assert svc.predict(X).shape == (80,)
    assert svc.predict_proba(X).shape == (80, 2)


def test_benchmark_returns_metrics_for_all_models_without_test_mutation():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(180, 8))
    y = np.tile([0, 0, 0, 1, 0, 0], 30)
    prepared = {
        "X_train": X[:120], "y_train": y[:120].copy(),
        "X_validation": X[120:150], "y_validation": y[120:150].copy(),
        "X_test": X[150:], "y_test": y[150:].copy(),
    }
    validation_before = prepared["y_validation"].copy()
    test_before = prepared["y_test"].copy()
    results = run_classical_benchmark(prepared, random_state=42)
    assert list(results) == MODEL_NAMES
    required = {"accuracy", "precision", "recall", "f1", "roc_auc",
                "pr_auc", "threshold", "tn", "fp", "fn", "tp"}
    for result in results.values():
        assert required.issubset(result)
        assert result["confusion_matrix"].shape == (2, 2)
        assert 0.10 <= result["threshold"] <= 0.90
    np.testing.assert_array_equal(prepared["y_validation"], validation_before)
    np.testing.assert_array_equal(prepared["y_test"], test_before)


def test_benchmark_contains_no_ann_model():
    assert all("ANN" not in name and "DNN" not in name for name in MODEL_NAMES)


def test_ann_probability_evaluation_uses_validation_threshold():
    validation_labels = np.array([0, 0, 1, 1])
    validation_probabilities = np.array([0.05, 0.20, 0.35, 0.40])
    test_labels = np.array([0, 1, 0, 1])
    test_probabilities = np.array([0.10, 0.30, 0.25, 0.45])
    result = evaluate_probability_model(
        "ANN", validation_labels, validation_probabilities,
        test_labels, test_probabilities)
    assert result["model"] == "ANN"
    assert result["threshold"] != 0.40
    assert result["confusion_matrix"].shape == (2, 2)


def test_ann_row_is_structured_and_checkpoint_is_separate():
    result = metrics_from_probabilities(
        np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.5)
    result["model"] = "ANN"
    result["fit_seconds"] = 1.0
    row = results_table({"ANN": result})[0]
    assert row["model"] == "ANN"
    assert CANONICAL_ANN_SAVE_PATH != MODEL_SAVE_PATH
    assert not CANONICAL_ANN_SAVE_PATH.endswith("fraud_detector_final.keras")


def test_ann_v3_architecture_and_configuration():
    model = build_v3(29)
    assert model.output_shape == (None, 1)
    assert [model.get_layer(f"dense_{i}").units for i in (1, 2)] == V3_HIDDEN_LAYERS
    assert [model.get_layer(f"dropout_{i}").rate for i in (1, 2)] == [0.4, 0.3]
    assert model.get_layer("dense_1").kernel_regularizer.l2 == V3_L2_LAMBDA
    assert V3_BATCH_SIZE == 64
    assert V3_EPOCHS == 200
    assert V3_PATIENCE == 20
    assert V3_SMOTE_STRATEGY == 0.15
    assert V3_CLASS_WEIGHT == {0: 1.0, 1: 5.0}


def test_unified_rows_can_include_both_ann_variants():
    base = metrics_from_probabilities(
        np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.5)
    rows = results_table({name: dict(base, model=name, fit_seconds=0.1)
                          for name in ["ANN", "ANN v3"]})
    assert [row["model"] for row in rows] == ["ANN", "ANN v3"]
    assert ANN_V3_SAVE_PATH != CANONICAL_ANN_SAVE_PATH
    assert not ANN_V3_SAVE_PATH.endswith("fraud_detector_final.keras")


def test_optimization_configuration_and_checkpoint_are_separate():
    from insurance_fraud_detection.models.ann_v3 import ANNOptimizationConfig

    config = ANNOptimizationConfig(
        "candidate", (96, 48), 0.005, (0.3, 0.2), 0.001, 64, 0.15, 5.0)
    values = config.to_dict()
    assert values["hidden_layers"] == [96, 48]
    assert values["dropout_rates"] == [0.3, 0.2]
    assert OPTIMIZED_ANN_SAVE_PATH.endswith("optimized_ann.keras")
    assert OPTIMIZED_ANN_SAVE_PATH not in {
        MODEL_SAVE_PATH, ANN_V3_SAVE_PATH, CANONICAL_ANN_SAVE_PATH}


def test_validation_search_has_no_final_test_arguments():
    from insurance_fraud_detection.evaluation.ann_optimization import run_validation_candidate

    parameters = set(inspect.signature(run_validation_candidate).parameters)
    assert "X_test" not in parameters
    assert "y_test" not in parameters


def test_validation_winner_uses_pr_auc_then_f1_then_roc_auc():
    results = [
        {"candidate_id": "f1", "validation_pr_auc": 0.20,
         "validation_f1": 0.90, "validation_roc_auc": 0.90},
        {"candidate_id": "pr", "validation_pr_auc": 0.30,
         "validation_f1": 0.10, "validation_roc_auc": 0.10},
        {"candidate_id": "tie", "validation_pr_auc": 0.30,
         "validation_f1": 0.20, "validation_roc_auc": 0.90},
    ]
    assert select_validation_winner(results)["candidate_id"] == "tie"
    assert validation_sort_key(results[1]) < validation_sort_key(results[2])
