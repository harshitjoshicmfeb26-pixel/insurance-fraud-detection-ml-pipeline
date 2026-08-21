# Insurance Fraud Detection ML Pipeline

An end-to-end insurance-claim fraud detection project comparing classical supervised learning with TensorFlow/Keras neural networks. The repository focuses on leakage-safe preprocessing, training-only class-imbalance handling, validation-based threshold selection, and transparent comparison of accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.

The results show that different models are strongest for different operational goals: Random Forest leads ROC-AUC, Gradient Boosting leads F1 and PR-AUC, Decision Tree leads recall, and ANN v3 is the strongest deep-learning model. No single model is universally best.

**Live Demo:** https://huggingface.co/spaces/harshitjoshiai/insurance-fraud-detector
**Repository:** https://github.com/harshitjoshicmfeb26-pixel/insurance-fraud-detection-ml-pipeline

The live Hugging Face Space is a separately deployed Gradio application. It currently serves the Gradient Boosting model using the canonical preprocessing pipeline; the broader project still evaluates both classical machine-learning and TensorFlow/Keras deep-learning models.

## Business problem

Fraudulent claims are a minority class in this dataset, so accuracy alone can hide poor fraud detection. Missing fraud creates financial risk, while false positives create investigation workload. The appropriate operating point depends on business cost:

- Higher recall can catch more fraudulent claims but may produce more alerts.
- Higher precision can reduce unnecessary investigations but may miss more fraud.
- ROC-AUC measures ranking discrimination across thresholds, while PR-AUC is especially informative when the positive class is rare.

The project therefore reports multiple metrics instead of treating one score as universally correct.

## Dataset

The local dataset is `data/raw/FraudDataset.csv`:

- 15,420 rows and 33 original columns
- Target: `FraudFound_P`
- 923 fraud cases, approximately 5.99%
- 29 model predictors after removing the target and these identifiers: `PolicyNumber`, `RepNumber`, and `Year`

The canonical preprocessing contract includes schema validation, replacement of `Age == 0` using a training-derived value, handling of invalid claim-day/month sentinel values, categorical encoding, and scaling. This is primarily cleaning and representation preparation; the project does not claim extensive domain-derived feature engineering.

The dataset source and redistribution licence have not been independently verified. Verify licensing before redistributing the raw CSV.

## Leakage-safe methodology

~~~
flowchart LR
    A[Raw insurance claims] --> B[Schema validation and cleaning]
    B --> C[Remove identifiers and separate target]
    C --> D[Stratified 70/15/15 split]
    D --> E[Fit preprocessing on training data only]
    E --> F[Transform validation and test]
    F --> G[Training-only SMOTE and model fitting]
    G --> H[Validation-based model and threshold decisions]
    H --> I[Untouched final-test evaluation]
    I --> J[Benchmark reports and model artifacts]
~~~

The corrected workflow is:

- Training: fit the encoder/scaler, apply SMOTE only to training data, and fit models.
- Validation: use validation probabilities for early stopping, model selection, and threshold selection.
- Final test: freeze the model and threshold, then evaluate once.

This replaced earlier notebook experimentation in which the test partition was reused during development. The notebook remains valuable development history, but the reusable evaluation path is implemented in `src/` and `scripts/`.

## Models evaluated

The primary comparison includes:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (`SVC`)
- Canonical ANN
- ANN v3

The models use the same canonical split, 29-feature schema, train-fitted preprocessing contract, validation strategy, and untouched final test partition. They are competing candidate approaches; the ANN is not assumed to be the winner.

## Primary benchmark results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8037 | 0.1595 | 0.5362 | 0.2458 | 0.7918 | 0.1594 | 0.53 |
| Decision Tree | 0.6498 | 0.1316 | 0.8696 | 0.2286 | 0.8017 | 0.2035 | 0.15 |
| Random Forest | 0.8020 | 0.1552 | 0.5217 | 0.2392 | 0.8065 | 0.1966 | 0.18 |
| Gradient Boosting | 0.8093 | 0.1641 | 0.5362 | 0.2513 | 0.7941 | 0.2200 | 0.24 |
| SVM | 0.7929 | 0.1395 | 0.4783 | 0.2160 | 0.7386 | 0.1352 | 0.23 |
| Canonical ANN | 0.8240 | 0.1542 | 0.4348 | 0.2277 | 0.7375 | 0.1381 | 0.27 |
| **ANN v3** | **0.8167** | **0.1643** | **0.5072** | **0.2482** | **0.8034** | **0.1793** | **0.64** |

Metric leaders:

- Highest accuracy: Canonical ANN (`0.8240`)
- Highest recall: Decision Tree (`0.8696`)
- Highest precision: ANN v3 (`0.1643`), narrowly above Gradient Boosting
- Highest F1: Gradient Boosting (`0.2513`)
- Highest ROC-AUC: Random Forest (`0.8065`)
- Highest PR-AUC: Gradient Boosting (`0.2200`)
- Strongest deep-learning model: ANN v3

These results do not establish a universally best model. Model choice depends on the cost of missed fraud versus false alerts.

### Confusion-matrix trade-off

At their validation-selected thresholds:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Decision Tree | 1,383 | 792 | 18 | 120 |
| Gradient Boosting | 1,798 | 377 | 64 | 74 |
| ANN v3 | 1,819 | 356 | 68 | 70 |

The Decision Tree catches substantially more fraud but creates far more false alerts. ANN v3 produces fewer false alerts but misses more fraud cases. Gradient Boosting provides the strongest F1/PR-AUC balance in this benchmark.

## Threshold tuning

The evaluation flow is:

~~~
validation probabilities -> select threshold -> freeze threshold -> final test evaluation
~~~

The default threshold of 0.50 is not assumed automatically. ANN v3 uses its validation-selected threshold of `0.64`; classical models receive their own validation-selected thresholds. These benchmark thresholds must not be confused with the live Hugging Face application's presentation bands. The live deployment's validation-selected operating threshold is `0.24`; its additional score bands are UX interpretation only.

## ANN development and model naming

The deep-learning work progressed from notebook experimentation to a corrected canonical ANN, then to a faithful ANN v3 reconstruction and a later validation-only optimization experiment.

### ANN v3 — selected deep-learning model

Artifact: `outputs/models/ann_v3.keras`

ANN v3 is the strongest deep-learning model, not the best overall fraud-detection model.

~~~
29 inputs
 -> Dense(64, ReLU) -> BatchNormalization -> Dropout(0.4)
 -> Dense(32, ReLU) -> BatchNormalization -> Dropout(0.3)
 -> Dense(1, Sigmoid)
~~~

Configuration:

- L2 regularization: `0.01`
- Optimizer: Adam, learning rate `0.001`
- Loss: binary cross-entropy
- SMOTE ratio: `0.15`
- Fraud class weight: `5.0` relative to class 0 weight `1.0`
- Batch size: `64`
- Maximum epochs: `200`
- Early stopping and checkpoint monitoring: validation AUC
- Patience: `20`

### Optimized ANN — later tuning experiment

Artifact: `outputs/models/optimized_ann.keras`

Eleven ANN configurations were evaluated using validation-only selection, with validation PR-AUC as the primary criterion followed by F1 and ROC-AUC. The validation winner used the 64-32 architecture, L2 `0.01`, dropout `0.4/0.3`, learning rate `0.001`, batch size `64`, SMOTE ratio `0.20`, fraud class weight `5.0`, and threshold `0.70`.

Its untouched-test result was accuracy `0.8111`, precision `0.1345`, recall `0.3986`, F1 `0.2011`, ROC-AUC `0.7740`, and PR-AUC `0.1772`. It did not outperform ANN v3, so ANN v3 remains the selected deep-learning model. This experiment demonstrates that stronger validation performance does not necessarily generalize to the final test set.

### Canonical ANN

Artifact: `outputs/models/canonical_ann.keras`

The canonical ANN is an earlier corrected reference ANN used in the primary benchmark. It is not the selected final ANN.

## Live Hugging Face demo

Live Space: https://huggingface.co/spaces/harshitjoshiai/insurance-fraud-detector

The live demo is a Gradio-based Gradient Boosting fraud-risk application. Gradient Boosting was selected for deployment because it achieved the highest F1 and PR-AUC in the comparative benchmark. It currently uses:

- `artifacts/gradient_boosting_model.joblib`
- `artifacts/preprocessor.joblib`
- `artifacts/deployment_metadata.json`

The validation-selected operating threshold is `0.24`. The interface displays these presentation bands:

- `< 0.24`: Lower Fraud Risk
- `0.24–<0.40`: Elevated Fraud Risk
- `0.40–<0.50`: High Review Priority
- `>= 0.50`: Very High Review Priority

The `0.24` value is the binary operating threshold selected on validation data; the additional score bands are presentation guidance and do not change that threshold. The score is a model-generated risk signal, not proof that a claim is fraudulent. Any future Space update must preserve the model filename, preprocessing artifact, feature order, categorical mappings, input field names, and operating threshold unless the Space is updated together with those changes.

## Business Decision-Support Layer

The validated classifier is also used as a human-in-the-loop claims-triage tool:

```text
Claim → Model score → Risk band → Recommended action → Human review
Claims portfolio → Batch scoring → Ranking → Investigation queue → Capacity analysis
```

The business layer does not retrain the model, change the 29-feature contract, or introduce claim severity or financial-loss estimates. It adds a reusable risk policy, structured decisions, batch scoring, transparent probability-based investigation ranking, and labeled portfolio analytics.

Recommended actions are advisory: Standard Claim Processing, Additional Verification, Fraud Analyst Review, and Priority Fraud Investigation. The model output is a risk signal and does not prove fraud or replace a claims or fraud analyst's decision.

### Actual investigator-capacity metrics

These results were generated from the frozen Gradient Boosting deployment artifact on the untouched 2,313-row final test set containing 138 known fraud cases. Top-K sizes use ceiling rounding.

| Review capacity | Claims reviewed | Fraud Capture | Precision | Lift |
|---:|---:|---:|---:|---:|
| Top 1% | 24 | 8.70% | 50.00% | 8.38 |
| Top 5% | 116 | 19.57% | 23.28% | 3.90 |
| Top 10% | 232 | 32.61% | 19.40% | 3.25 |
| Top 20% | 463 | 53.62% | 15.98% | 2.68 |

Fraud Capture @ K measures the share of known fraud cases inside the highest-scored K% of claims. Precision @ K measures the fraud rate within that reviewed segment. Lift compares that segment's fraud rate with the overall fraud prevalence. Outcome-dependent metrics are calculated only on labeled evaluation data.

Reproducible business outputs are generated by `scripts/generate_business_metrics.py` and stored under `reports/generated/business/`. The implementation and human-review policy are documented in `docs/business_decision_support.md`. SHAP explainability and the optional assumption-based cost simulator are intentionally deferred.

## Repository structure

~~~
.
├── app/                  # Local snapshot of the live Gradio application
├── data/                 # Raw dataset and dataset notes
├── docs/                 # Deployment contract and project documentation
├── notebooks/            # EDA, training scripts, and archived notebook history
├── outputs/              # Local model artifacts and generated model outputs
├── reports/              # Generated benchmark and tuning reports
├── scripts/              # Reproducible training, benchmark, and tuning entry points
├── src/                  # Reusable data, model, evaluation, and utility modules
├── tests/                # Unit and structural tests
├── requirements.txt
├── pyproject.toml
└── setup.py
~~~

`notebooks/archive/FinalCode123.ipynb` is preserved as the original experimentation and model-development notebook. Reusable logic was later moved into modular source files; the archived notebook should not be treated as the current production-style training entry point.

## Installation

~~~
git clone https://github.com/harshitjoshicmfeb26-pixel/insurance-fraud-detection-ml-pipeline.git
cd insurance-fraud-detection-ml-pipeline
~~~

Windows PowerShell:

~~~
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

macOS/Linux:

~~~
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Python 3.11 is the validated local environment for this project. The raw dataset must be available at `data/raw/FraudDataset.csv`.

## Testing and reproducible workflows

Run the test suite:

~~~
python -m pytest
~~~

The verified result after the business decision-support layer is 34 tests passed.

The suite currently reports two scikit-learn SVC deprecation warnings because the installed scikit-learn version is deprecating `SVC(probability=True)`. The compatibility replacement uses a different probability-calibration path and could change benchmark probabilities, so the warning is intentionally documented rather than silently changing the frozen results.

Reproducible entry points:

~~~
# Seven-model benchmark: five classical models, Canonical ANN, and ANN v3
python scripts/benchmark_classical.py

# Later validation-only ANN optimization experiment (not part of the primary table)
python scripts/optimize_ann_v3.py

# Original modular ANN training workflow
python scripts/train.py
~~~

The benchmark and optimization scripts train models and write local artifacts. They should not be run merely to use the live demo.

## Generated reports and artifacts

Important generated outputs include:

- `reports/generated/classical_benchmark.csv`
- `reports/generated/classical_benchmark.json`
- `reports/generated/ann_v3_history.json`
- `reports/generated/canonical_ann_history.json`
- `reports/generated/ann_v3_tuning.csv`
- `reports/generated/ann_v3_tuning.json`
- `reports/generated/optimized_ann_result.json`
- `reports/generated/business/portfolio_metrics.json`
- `reports/generated/business/top_k_metrics.csv`
- `reports/generated/business/fraud_capture_curve.csv`
- `reports/generated/business/lift_by_decile.csv`
- `reports/generated/business/generation_metadata.json`
- `outputs/models/ann_v3.keras` — selected deep-learning model
- `outputs/models/optimized_ann.keras` — later tuning experiment
- `outputs/models/canonical_ann.keras` — earlier corrected reference ANN

## Technology stack

Python, Pandas, NumPy, scikit-learn, imbalanced-learn/SMOTE, TensorFlow/Keras, Matplotlib, Seaborn, Joblib, Pytest, Gradio, and Hugging Face Spaces are used in the repository or deployment snapshot.

## Limitations

- The dataset is relatively small for deep learning, with 923 fraud examples.
- It is historical/public tabular data; predictions are not proof of fraud.
- False-positive and false-negative trade-offs remain business decisions.
- Concept drift and production monitoring are not implemented.
- The project does not use claim images, documents, text, voice, or telematics.
- Probability calibration is not implemented.
- The live Hugging Face demo is a separately deployed application and is not automatically updated by this repository.
- Dataset redistribution and licensing require verification.

Deep learning may become more competitive with substantially more minority examples, richer feature interactions, sequential behavior, or multimodal inputs. Larger data alone does not guarantee that an ANN will outperform classical ML; that must still be validated empirically.

## Future work

Potential next steps include SHAP explanations, calibrated probabilities, cost-sensitive thresholding, richer fraud signals, multimodal claim analysis, cross-validation, additional validation-only optimization, drift monitoring, a production API, and experiment tracking.

## Resume verification

The resume description is broadly defensible for the implemented preprocessing, label encoding, scaling, SMOTE, classical models, TensorFlow/Keras ANN, classification metrics, ROC-AUC, and comparative evaluation. SVM is genuinely instantiated, trained, used for probability prediction, and included in the benchmark.

The phrase **feature engineering** may be overstated: the current pipeline primarily performs cleaning, identifier removal, categorical encoding, and scaling rather than extensive derived-feature creation. A more precise resume phrase would be “data cleaning and preprocessing” unless meaningful domain-derived features are added later.

The project supports an end-to-end fraud-detection workflow and a live Gradient Boosting demonstration. The live Space is separate from the broader benchmark artifacts. No claim should describe ANN v3 as the overall best model or imply that deep learning outperformed classical ML.
