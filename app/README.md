# Hugging Face deployment snapshot

app/app.py is a source snapshot of the currently deployed Gradio application.

Live Space: https://huggingface.co/spaces/harshitjoshiai/insurance-fraud-detector

The Hugging Face Space is deployed separately from this GitHub repository. Changes made here do not automatically update the live Space, and this Phase 1 work does not modify the Space.

The current deployment contract is:

- Model: fraud_detector_final.keras
- Scaler: scaler.joblib
- Input features: 29, in the exact order documented in docs/deployment-contract.md
- Low risk: probability < 0.40
- Medium risk: probability 0.40–0.59
- High risk: probability >= 0.60

The live Space currently stores the model and scaler alongside its app.py. Those artifacts are not silently renamed or relocated by this repository snapshot.
