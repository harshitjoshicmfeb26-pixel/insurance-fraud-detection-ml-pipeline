"""Generate reproducible investigation-capacity metrics without retraining."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from insurance_fraud_detection.business.inference import transform_with_bundle
from insurance_fraud_detection.business.portfolio import (
    capacity_summary, cumulative_gains, lift_by_decile)
from insurance_fraud_detection.data.loader import load_raw_data
from insurance_fraud_detection.data.preprocessor import split_raw_dataframe


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_DIR = os.path.join(ROOT, "deployment", "huggingface", "artifacts")
OUTPUT_DIR = os.path.join(ROOT, "reports", "generated", "business")


def main():
    raw = load_raw_data()
    splits = split_raw_dataframe(raw, random_state=42)
    bundle = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    model = joblib.load(os.path.join(ARTIFACT_DIR, "gradient_boosting_model.joblib"))
    transformed = transform_with_bundle(splits.X_test, bundle)
    probabilities = np.asarray(model.predict_proba(transformed)[:, 1], dtype=float)
    labels = splits.y_test.to_numpy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    capacity = capacity_summary(labels, probabilities)
    gains = cumulative_gains(labels, probabilities)
    deciles = lift_by_decile(labels, probabilities)
    top_k = capacity.copy()
    capacity.to_json(os.path.join(OUTPUT_DIR, "portfolio_metrics.json"), orient="records", indent=2)
    top_k.to_csv(os.path.join(OUTPUT_DIR, "top_k_metrics.csv"), index=False)
    gains.to_csv(os.path.join(OUTPUT_DIR, "fraud_capture_curve.csv"), index=False)
    deciles.to_csv(os.path.join(OUTPUT_DIR, "lift_by_decile.csv"), index=False)
    summary = {
        "source": "FraudDataset.csv final test split",
        "split": "canonical stratified 70/15/15, random_state=42",
        "model": "GradientBoostingClassifier deployment artifact",
        "threshold": 0.24,
        "test_claims": int(len(labels)),
        "test_fraud_cases": int(labels.sum()),
        "rounding": "ceil(K% × total claims)",
    }
    with open(os.path.join(OUTPUT_DIR, "generation_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"summary": summary, "metrics": capacity.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
