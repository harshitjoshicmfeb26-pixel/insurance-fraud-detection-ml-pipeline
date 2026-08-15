"""Run the reproducible seven-model fraud benchmark."""

import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from insurance_fraud_detection.config import RANDOM_STATE
from insurance_fraud_detection.data import load_raw_data, prepare_training_data
from insurance_fraud_detection.evaluation.benchmark import (results_table,
                                                              run_ann_benchmark,
                                                              run_ann_v3_benchmark,
                                                              run_classical_benchmark)
from insurance_fraud_detection.models.ann_v3 import V3_SMOTE_STRATEGY


def main():
    prepared = prepare_training_data(load_raw_data(), random_state=RANDOM_STATE)
    results = run_classical_benchmark(prepared, random_state=RANDOM_STATE)
    ann_started = time.perf_counter()
    ann_result = run_ann_benchmark(prepared, random_state=RANDOM_STATE)
    ann_result["fit_seconds"] = round(time.perf_counter() - ann_started, 3)
    results["ANN"] = ann_result
    prepared_v3 = prepare_training_data(
        load_raw_data(), random_state=RANDOM_STATE,
        smote_strategy=V3_SMOTE_STRATEGY)
    v3_started = time.perf_counter()
    v3_result = run_ann_v3_benchmark(prepared_v3, random_state=RANDOM_STATE)
    v3_result["fit_seconds"] = round(time.perf_counter() - v3_started, 3)
    results["ANN v3"] = v3_result
    rows = results_table(results)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "generated")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "classical_benchmark.csv")
    json_path = os.path.join(output_dir, "classical_benchmark.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    history = ann_result["history"]
    history_path = os.path.join(output_dir, "canonical_ann_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump({key: [float(value) for value in values]
                   for key, values in history.items()}, handle, indent=2)
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="training loss")
    plt.plot(history.get("val_loss", []), label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Canonical ANN training history")
    plt.legend()
    plt.tight_layout()
    history_plot = os.path.join(output_dir, "canonical_ann_history.png")
    plt.savefig(history_plot, dpi=150)
    plt.close()
    v3_history_path = os.path.join(output_dir, "ann_v3_history.json")
    with open(v3_history_path, "w", encoding="utf-8") as handle:
        json.dump({key: [float(value) for value in values]
                   for key, values in v3_result["history"].items()}, handle, indent=2)
    print("\nUnified benchmark (SMOTE training data; validation-selected threshold):")
    print("Model                   Accuracy  Precision  Recall  F1      ROC-AUC  PR-AUC  Threshold  TN   FP  FN  TP")
    for row in rows:
        print(f"{row['model']:<23} {row['accuracy']:.4f}    {row['precision']:.4f}     "
              f"{row['recall']:.4f}  {row['f1']:.4f}  {row['roc_auc']:.4f}   "
              f"{row['pr_auc']:.4f}  {row['threshold']:.2f}       {row['tn']:4d} {row['fp']:4d} "
              f"{row['fn']:3d} {row['tp']:3d}")
    print(f"\nCanonical ANN epochs: requested={ann_result['epochs_requested']} ran={ann_result['epochs_run']}")
    print(f"ANN v3 epochs: requested={v3_result['epochs_requested']} ran={v3_result['epochs_run']}")
    print(f"ANN v3 threshold: {v3_result['threshold']:.2f}")
    print(f"Canonical ANN checkpoint: {ann_result['checkpoint_path']}")
    print(f"ANN v3 checkpoint: {v3_result['checkpoint_path']}")
    print(f"Wrote {csv_path}, {json_path}, {history_path}, {history_plot}, and {v3_history_path}")


if __name__ == "__main__":
    main()
