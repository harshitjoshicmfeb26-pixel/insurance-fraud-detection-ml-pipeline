"""
train.py — Main entry point for the Insurance Fraud Detection project.

Run this file to execute the complete pipeline:
  1. Load raw data
  2. Preprocess (encode + scale + SMOTE)
  3. Train baseline ANN (no regularization)
  4. Train tuned ANN (BN + Dropout + L2 + ADAM)
  5. Compare optimizers (SGD / RMSProp / Momentum / ADAM)
  6. Evaluate and generate all plots + reports

Usage:
  python scripts/train.py

Outputs:
  outputs/models/fraud_detector.keras    — saved best model
  outputs/plots/*.png                    — all plots
  outputs/reports/classification_report.txt
"""

import os
import sys
import numpy as np

# ── Make src importable from project root ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from insurance_fraud_detection.config import (DECISION_THRESHOLD, REPORTS_DIR,
                                              EPOCHS, BATCH_SIZE,
                                              CLASS_WEIGHT, RANDOM_STATE)

from insurance_fraud_detection.utils import set_all_seeds, get_logger
from insurance_fraud_detection.data import (load_raw_data, prepare_training_data,
                                             save_processed)
from insurance_fraud_detection.models import (build_model, build_baseline,
                                              train as train_model,
                                              compare_optimizers)
from insurance_fraud_detection.evaluation import (evaluate, find_best_threshold,
                                                  plot_training_history,
                                                  plot_confusion_matrix,
                                                  plot_roc_curve,
                                                  plot_precision_recall,
                                                  plot_model_comparison,
                                                  plot_threshold_sensitivity)

log = get_logger("train")


def run_pipeline():
    """Execute the full training and evaluation pipeline."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_all_seeds(RANDOM_STATE)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("STEP 1: Loading raw data")
    log.info("="*60)

    df = load_raw_data()
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: Split raw data, fit preprocessing on train, and SMOTE train only
    # ═════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("STEP 2: Preprocessing")
    log.info("="*60)

    prepared = prepare_training_data(df, random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = prepared["X_train"], prepared["y_train"]
    X_train, y_train = prepared["X_train_original"], prepared["y_train_original"]
    X_validation, y_validation = prepared["X_validation"], prepared["y_validation"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    feature_names = prepared["feature_names"]
    prepared["preprocessor"].save()
    log.info(f"Training features: {X_train.shape}  |  Validation: {X_validation.shape}  |  Test: {X_test.shape}")
    log.info(f"Features ({len(feature_names)}): {feature_names}")

    # Save processed data for notebook use
    save_processed(X_train_sm, X_test, y_train_sm, y_test, X_validation, y_validation)

    input_dim = X_train_sm.shape[1]

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: Baseline Model (no regularization)
    # ═════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("STEP 3: Training Baseline ANN (no regularization)")
    log.info("="*60)

    baseline = build_baseline(input_dim)
    baseline.fit(
        X_train_sm, y_train_sm,
        validation_data=(X_validation, y_validation),
        epochs=30,
        batch_size=BATCH_SIZE,
        verbose=0,
        class_weight=CLASS_WEIGHT
    )

    log.info("\nBaseline Evaluation:")
    baseline_metrics = evaluate(baseline, X_test, y_test,
                                threshold=DECISION_THRESHOLD)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4: Tuned ANN (full regularization stack)
    # ═════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("STEP 4: Training Tuned ANN (BN + Dropout + L2 + ADAM)")
    log.info("="*60)

    model = build_model(input_dim)
    history = train_model(model, X_train_sm, y_train_sm,
                          X_val=X_validation, y_val=y_validation)

    # Training curves
    plot_training_history(history, title="Tuned ANN — Training History")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5: Optimizer Comparison Experiment
    # ═════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("STEP 5: Optimizer Comparison (SGD / Momentum / RMSProp / ADAM)")
    log.info("="*60)

    compare_optimizers(X_train_sm, y_train_sm, X_validation, y_validation,
                       epochs=30, batch_size=BATCH_SIZE)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6: Full Evaluation
    # ═════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("STEP 6: Evaluation and Plots")
    log.info("="*60)

    # Threshold tuning
    y_prob_validation = model.predict(X_validation, verbose=0).flatten()
    best_t = find_best_threshold(y_validation, y_prob_validation, optimize_for="f1")

    # Full metrics at chosen threshold
    log.info(f"\nFull evaluation at threshold = {DECISION_THRESHOLD}")
    tuned_metrics = evaluate(model, X_test, y_test,
                             threshold=DECISION_THRESHOLD)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(tuned_metrics["confusion_matrix"],
                          title="Tuned ANN — Confusion Matrix")

    plot_roc_curve(y_test, tuned_metrics["y_prob"],
                   tuned_metrics["roc_auc"])

    plot_precision_recall(y_test, tuned_metrics["y_prob"],
                          tuned_metrics["avg_precision"])

    plot_threshold_sensitivity(y_test, tuned_metrics["y_prob"])

    # ── Model comparison chart ─────────────────────────────────────────────
    plot_model_comparison({
        "Model": ["Baseline ANN", "Tuned ANN"],
        "Accuracy":      [baseline_metrics["accuracy"],
                          tuned_metrics["accuracy"]],
        "Fraud Recall":  [baseline_metrics["fraud_recall"],
                          tuned_metrics["fraud_recall"]],
        "ROC-AUC":       [baseline_metrics["roc_auc"],
                          tuned_metrics["roc_auc"]],
        "Fraud F1":      [baseline_metrics["fraud_f1"],
                          tuned_metrics["fraud_f1"]],
    })

    # ── Save text report ───────────────────────────────────────────────────
    _save_report(baseline_metrics, tuned_metrics, feature_names, best_t)

    log.info("\n" + "="*60)
    log.info("Pipeline complete. Check outputs/ for all results.")
    log.info("="*60)


def _save_report(baseline, tuned, features, best_t):
    """Write a human-readable summary report."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "classification_report.txt")

    lines = [
        "Insurance Fraud Detection — Project Report",
        "=" * 60,
        f"Total features: {len(features)}",
        f"Features: {features}",
        "",
        "Baseline ANN (1 hidden layer, SGD, no regularization):",
        f"  Accuracy       : {baseline['accuracy']*100:.2f}%",
        f"  Fraud Recall   : {baseline['fraud_recall']*100:.2f}%",
        f"  ROC-AUC        : {baseline['roc_auc']:.4f}",
        "",
        "Tuned ANN (3 hidden layers, BN + Dropout + L2 + ADAM):",
        f"  Accuracy       : {tuned['accuracy']*100:.2f}%",
        f"  Fraud Recall   : {tuned['fraud_recall']*100:.2f}%",
        f"  ROC-AUC        : {tuned['roc_auc']:.4f}",
        f"  Fraud F1       : {tuned['fraud_f1']*100:.2f}%",
        "",
        f"Optimal threshold (max F1): {best_t:.2f}",
        f"Operating threshold used  : {DECISION_THRESHOLD}",
        "",
        "Confusion Matrix (Tuned ANN):",
        f"  TN={tuned['confusion_matrix'][0,0]}  FP={tuned['confusion_matrix'][0,1]}",
        f"  FN={tuned['confusion_matrix'][1,0]}  TP={tuned['confusion_matrix'][1,1]}",
        "",
        "Note: FN = frauds missed (high business cost). Keep low.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"Report saved → {path}")


if __name__ == "__main__":
    run_pipeline()
