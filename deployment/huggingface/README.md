---
title: Insurance Fraud Risk Demo
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
---

# Insurance Fraud Risk Demo

This proposed Hugging Face deployment is a local migration bundle for the leakage-safe GitHub pipeline. It serves a Gradient Boosting classifier selected for its F1 and PR-AUC performance in the comparative benchmark.

## Model and threshold

The deployed model is `gradient_boosting_model.joblib`. It was trained using the canonical 29-feature preprocessing contract and training-only SMOTE. The validation-selected binary operating threshold is **0.24**.

Benchmark metrics for this selected deployment model:

- Accuracy: 0.8093
- Precision: 0.1641
- Recall: 0.5362
- F1: 0.2513
- ROC-AUC: 0.7941
- PR-AUC: 0.2200

Gradient Boosting is not presented as universally best. Random Forest has the highest ROC-AUC, Decision Tree has the highest recall, and ANN v3 remains the strongest deep-learning model. The deployment choice emphasizes the Gradient Boosting F1/PR-AUC balance.

## Data contract

The application accepts the same 29 logical predictor fields used by the canonical local pipeline. It applies the fitted encoder and scaler stored in `preprocessor.joblib` in the exact training feature order. The raw training CSV is not required at inference time.

The model output is a risk signal. “Higher fraud risk / Review recommended” does not mean fraud is confirmed; predictions should support investigation and human review.

The binary benchmark operating threshold is `0.24`. The interface presents separate explanatory bands: below `0.24` is Lower Fraud Risk, `0.24` to below `0.50` is Elevated Fraud Risk, and `0.50` or above is Higher Fraud Risk / Review Recommended. These presentation bands do not change the binary threshold.

The interface preserves the earlier Space's polished two-column layout, grouped Claim Timing, Vehicle, Policy and Driver, Policyholder, and Incident Details sections, constrained sliders/dropdowns, assessment panel, contextual review cues, and compact explanatory guidance. Its inference backend is the canonical Gradient Boosting bundle described above.

## Repository

Source repository: https://github.com/harshitjoshicmfeb26-pixel/insurance-fraud-detection-ml-pipeline

This is a proposed local migration only. The current public Space remains separate until a future, explicitly approved deployment update.
