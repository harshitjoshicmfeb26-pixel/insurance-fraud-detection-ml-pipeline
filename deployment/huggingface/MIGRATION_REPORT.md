# Hugging Face migration report

## Current live Space

The current public Space is a separate Gradio deployment using:

- `app.py`
- `fraud_detector_final.keras`
- `scaler.joblib`
- TensorFlow ANN inference
- notebook-derived hard-coded categorical mappings
- legacy UI risk bands: `<0.40` low, `0.40–0.59` medium, `>=0.60` high

## Proposed local Space

The proposed local bundle contains:

- `app.py`
- `artifacts/preprocessor.joblib`
- `artifacts/gradient_boosting_model.joblib`
- `artifacts/deployment_metadata.json`
- `requirements.txt`
- `README.md`

The new app uses the canonical 29-feature schema, fitted categorical encoder, fitted scaler, exact feature order, and Gradient Boosting trained under the fair pipeline. It does not load `FraudDataset.csv` during inference and does not require TensorFlow.

The presentation shell preserves the legacy app's two-column layout, grouped insurance sections, dropdowns, bounded sliders, assessment outputs, contextual cues, and explanatory guidance. The old ANN-specific claims and 0.40/0.60 risk bands were replaced. The new binary threshold remains `0.24`; UI bands are separately defined as below `0.24`, `0.24` to below `0.50`, and `0.50` or above.

## Future replacement plan

After explicit deployment approval, the proposed `app.py`, requirements, README, and three artifact files would replace the current Space's app, legacy ANN artifact, scaler, and associated documentation. The old files must not be removed until the new bundle has been uploaded, started, and smoke-tested in the Space.

The local repository artifacts `ann_v3.keras`, `optimized_ann.keras`, and `canonical_ann.keras` are unrelated to this proposed Gradient Boosting deployment and must remain separate.

## Compatibility risks

- The new Space must preserve the 29 raw field names and feature order.
- `preprocessor.joblib` and `gradient_boosting_model.joblib` must be uploaded together.
- Unknown categories are encoded safely as `-1`, but malformed numeric inputs are rejected.
- The threshold `0.24` is the benchmark operating threshold, not the old ANN UI band scheme.
- Any change to cleaning, mappings, scaling, threshold, or field names requires rebuilding and retesting the bundle.
- This local migration has not modified the public Space.
