"""Run a staged validation-only ANN v3 optimization and one final test evaluation."""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from insurance_fraud_detection.config import (OPTIMIZED_ANN_SAVE_PATH,
                                               RANDOM_STATE)
from insurance_fraud_detection.data import load_raw_data, prepare_training_data
from insurance_fraud_detection.evaluation.ann_optimization import (
    evaluate_selected_ann_once, run_validation_candidate,
    select_validation_winner)
from insurance_fraud_detection.models.ann_v3 import ANNOptimizationConfig


def _config(candidate_id, hidden_layers=(64, 32), l2_lambda=0.01,
            dropout_rates=(0.4, 0.3), learning_rate=0.001, batch_size=64,
            smote_ratio=0.15, fraud_class_weight=5.0):
    return ANNOptimizationConfig(
        candidate_id=candidate_id, hidden_layers=tuple(hidden_layers),
        l2_lambda=l2_lambda, dropout_rates=tuple(dropout_rates),
        learning_rate=learning_rate, batch_size=batch_size,
        smote_ratio=smote_ratio, fraud_class_weight=fraud_class_weight)


def _stage1():
    return [
        _config("s1_ratio010_weight4", smote_ratio=0.10, fraud_class_weight=4.0),
        _config("s1_ratio015_weight5", smote_ratio=0.15, fraud_class_weight=5.0),
        _config("s1_ratio020_weight5", smote_ratio=0.20, fraud_class_weight=5.0),
        _config("s1_ratio025_weight6", smote_ratio=0.25, fraud_class_weight=6.0),
    ]


def _stage2(best):
    common = dict(smote_ratio=best.smote_ratio,
                  fraud_class_weight=best.fraud_class_weight)
    return [
        _config("s2_96x48", hidden_layers=(96, 48), l2_lambda=0.005,
                dropout_rates=(0.3, 0.2), **common),
        _config("s2_128x64", hidden_layers=(128, 64), l2_lambda=0.005,
                dropout_rates=(0.3, 0.2), **common),
        _config("s2_128x64x32", hidden_layers=(128, 64, 32), l2_lambda=0.005,
                dropout_rates=(0.3, 0.2, 0.2), **common),
        _config("s2_64x32_light", l2_lambda=0.001,
                dropout_rates=(0.2, 0.2), **common),
    ]


def _stage3(best):
    common = dict(hidden_layers=best.hidden_layers, l2_lambda=best.l2_lambda,
                  dropout_rates=best.dropout_rates,
                  smote_ratio=best.smote_ratio,
                  fraud_class_weight=best.fraud_class_weight)
    return [
        _config("s3_lr0005_batch64", learning_rate=0.0005, batch_size=64, **common),
        _config("s3_lr001_batch128", learning_rate=0.001, batch_size=128, **common),
        _config("s3_lr002_batch64", learning_rate=0.002, batch_size=64, **common),
    ]


def _scalar_row(result):
    return {key: value for key, value in result.items()
            if key not in {"model", "history", "fit_seconds"}}


def _prepare_for(config, raw_data):
    prepared = prepare_training_data(raw_data, random_state=RANDOM_STATE,
                                     smote_strategy=config.smote_ratio)
    return prepared


def main():
    raw_data = load_raw_data()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "reports", "generated")
    os.makedirs(output_dir, exist_ok=True)
    candidates = []
    results = []
    prepared_by_id = {}

    # Stage winners are selected only from validation PR-AUC/F1/ROC-AUC.
    stage1 = _stage1()
    for config in stage1:
        prepared = _prepare_for(config, raw_data)
        prepared_by_id[config.candidate_id] = prepared
        result = run_validation_candidate(
            config, prepared["X_train"], prepared["y_train"],
            prepared["X_validation"], prepared["y_validation"],
            random_state=RANDOM_STATE)
        results.append(result)
    best_stage1 = select_validation_winner(results)
    best_stage1_config = next(c for c in stage1
                              if c.candidate_id == best_stage1["candidate_id"])

    stage2 = _stage2(best_stage1_config)
    stage2_results = []
    for config in stage2:
        prepared = _prepare_for(config, raw_data)
        prepared_by_id[config.candidate_id] = prepared
        stage2_results.append(run_validation_candidate(
            config, prepared["X_train"], prepared["y_train"],
            prepared["X_validation"], prepared["y_validation"],
            random_state=RANDOM_STATE))
    results.extend(stage2_results)
    best_stage2 = select_validation_winner(stage2_results)
    best_stage2_config = next(c for c in stage2
                              if c.candidate_id == best_stage2["candidate_id"])

    stage3 = _stage3(best_stage2_config)
    for config in stage3:
        prepared = _prepare_for(config, raw_data)
        prepared_by_id[config.candidate_id] = prepared
        results.append(run_validation_candidate(
            config, prepared["X_train"], prepared["y_train"],
            prepared["X_validation"], prepared["y_validation"],
            random_state=RANDOM_STATE))

    winner = select_validation_winner(results)
    winner_config = next(c for c in stage1 + stage2 + stage3
                         if c.candidate_id == winner["candidate_id"])
    winner_model = winner["model"]
    winner_model.save(OPTIMIZED_ANN_SAVE_PATH)

    # This is the only point where the untouched final-test partition is read.
    winner_prepared = prepared_by_id[winner_config.candidate_id]
    final_metrics = evaluate_selected_ann_once(
        winner_model, winner_prepared["X_test"], winner_prepared["y_test"],
        winner["validation_threshold"])
    final_metrics["model"] = "Optimized ANN"
    final_metrics["candidate_id"] = winner_config.candidate_id
    final_metrics["configuration"] = winner_config.to_dict()
    final_metrics["selection_rule"] = "validation PR-AUC, then F1, then ROC-AUC"
    final_metrics["checkpoint_path"] = OPTIMIZED_ANN_SAVE_PATH

    tuning_rows = [_scalar_row(result) for result in results]
    tuning_csv = os.path.join(output_dir, "ann_v3_tuning.csv")
    with open(tuning_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuning_rows[0].keys())
        writer.writeheader()
        writer.writerows(tuning_rows)
    with open(os.path.join(output_dir, "ann_v3_tuning.json"), "w", encoding="utf-8") as handle:
        json.dump(tuning_rows, handle, indent=2)
    with open(os.path.join(output_dir, "optimized_ann_result.json"), "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in final_metrics.items()
                   if key not in {"y_prob", "y_pred", "confusion_matrix"}}, handle, indent=2)

    comparison_path = os.path.join(output_dir, "ann_v3_optimization_comparison.csv")
    reference_path = os.path.join(output_dir, "classical_benchmark.csv")
    with open(reference_path, newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    comparison_rows = reference_rows + [{key: final_metrics.get(key, "") for key in reference_rows[0]}]
    with open(comparison_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Evaluated ANN configurations: {len(results)}")
    print(f"Stage 1 winner: {best_stage1['candidate_id']}")
    print(f"Stage 2 winner: {best_stage2['candidate_id']}")
    print(f"Final winner: {winner_config.candidate_id}")
    print(json.dumps({key: value for key, value in winner.items()
                      if key.startswith("validation_") or key in {"candidate_id", "epochs_run"}}, indent=2))
    print("Final untouched-test result:")
    print(json.dumps({key: value for key, value in final_metrics.items()
                      if key in {"model", "accuracy", "precision", "recall", "f1",
                                 "roc_auc", "pr_auc", "threshold", "tn", "fp", "fn", "tp"}}, indent=2))
    print(f"Tuning history: {tuning_csv}")
    print(f"Optimized checkpoint: {OPTIMIZED_ANN_SAVE_PATH}")


if __name__ == "__main__":
    main()
