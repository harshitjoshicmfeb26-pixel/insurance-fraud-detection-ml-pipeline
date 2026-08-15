"""Build the proposed Hugging Face Gradient Boosting bundle without test access."""

import json
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from insurance_fraud_detection.config import RANDOM_STATE, SMOTE_STRATEGY, TARGET_COLUMN
from insurance_fraud_detection.data import load_raw_data
from insurance_fraud_detection.data.preprocessor import (
    CanonicalPreprocessor, apply_smote, split_raw_dataframe)
from insurance_fraud_detection.models.classical import create_model_registry


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "deployment", "huggingface")
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")


def main():
    raw_data = load_raw_data()
    splits = split_raw_dataframe(raw_data, random_state=RANDOM_STATE)
    preprocessor = CanonicalPreprocessor().fit(splits.X_train)

    # Only the training partition is transformed, resampled, and used to fit
    # the deployment model. Validation/test partitions are not read here.
    X_train = preprocessor.transform(splits.X_train)
    X_train_smote, y_train_smote = apply_smote(
        X_train, splits.y_train.to_numpy(), strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE)
    model = create_model_registry(random_state=RANDOM_STATE)["Gradient Boosting"]
    model.fit(X_train_smote, y_train_smote)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    bundle = {
        "feature_columns": list(preprocessor.feature_columns),
        "categorical_columns": list(preprocessor.categorical_columns),
        "numeric_columns": list(preprocessor.numeric_columns),
        "encoder": preprocessor.encoder,
        "scaler": preprocessor.scaler,
        "age_fill_value": preprocessor.cleaning_metadata["age_fill_value"],
        "preprocessing_contract": "canonical train-fitted encoder/scaler; training-only SMOTE",
    }
    joblib.dump(bundle, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "gradient_boosting_model.joblib"))

    metadata = {
        "model_name": "Gradient Boosting",
        "model_role": "selected deployment model based on F1 and PR-AUC",
        "selected_threshold": 0.24,
        "target_column": TARGET_COLUMN,
        "feature_count": len(preprocessor.feature_columns),
        "feature_order": list(preprocessor.feature_columns),
        "dropped_identifier_columns": ["PolicyNumber", "RepNumber", "Year"],
        "preprocessing_version": "canonical 70/15/15 pipeline",
        "smote_strategy": SMOTE_STRATEGY,
        "benchmark_metrics": {
            "accuracy": 0.8093,
            "precision": 0.1641,
            "recall": 0.5362,
            "f1": 0.2513,
            "roc_auc": 0.7941,
            "pr_auc": 0.2200,
        },
        "benchmark_rationale": (
            "Gradient Boosting led the fair benchmark on F1 and PR-AUC. "
            "This does not make it universally best."
        ),
        "artifacts": ["preprocessor.joblib", "gradient_boosting_model.joblib"],
        "inference_requires_raw_dataset": False,
    }
    with open(os.path.join(ARTIFACT_DIR, "deployment_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote deployment artifacts to {ARTIFACT_DIR}")
    print(f"Feature count: {len(preprocessor.feature_columns)}")
    print(f"Threshold: {metadata['selected_threshold']}")


if __name__ == "__main__":
    main()
