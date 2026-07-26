# Insurance Fraud Detection — Deep Neural Network Project
**PGCP-AI | ACTS Pune | Sessions 3–13**

## Project Overview
Binary classification project to detect fraudulent insurance claims using a Deep Neural Network (ANN).  
Dataset: Vehicle Insurance Claim Fraud Detection (Kaggle — shivamb/vehicle-claim-fraud-detection)

## Problem Statement
Insurance companies lose billions annually to fraudulent claims. Traditional rule-based systems have high false-positive rates and cannot adapt to evolving fraud patterns. This project builds a DNN that learns from 33 features (policy details, vehicle info, accident data) to classify each claim as fraudulent or legitimate.

## Syllabus Coverage
| Session | Topic | Where Used |
|---------|-------|-----------|
| 3–4 | Forward/Backward propagation, Cost function | `src/models/ann_model.py` |
| 5 | Sigmoid, Gradient Descent | Model output layer |
| 6 | Shallow → Deep NN, ReLU, Tanh | Hidden layers |
| 7–8 | L1/L2 Regularization, Frobenius norm | `kernel_regularizer` |
| 9 | Dropout, Early Stopping, Data Augmentation (SMOTE) | Training pipeline |
| 10 | Vanishing gradient → ReLU fix | Activation choice |
| 11 | Batch Normalization | `BatchNormalization()` |
| 12 | ADAM, Mini-batch gradient descent | `optimizer=Adam` |
| 13 | RMSProp, Momentum comparison | `src/models/optimizer_comparison.py` |

## File Structure
```
insurance_fraud_detection/
├── README.md
├── requirements.txt
├── setup.py
├── config.py                          # All hyperparameters and paths
│
├── data/
│   ├── raw/                           # Place downloaded CSV here
│   │   └── fraud_oracle.csv           # From Kaggle
│   └── processed/                     # Auto-generated after preprocessing
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Data cleaning and feature engineering
│   ├── 03_Model_Training.ipynb        # ANN training and experiments
│   └── 04_Evaluation.ipynb            # Results, plots, confusion matrix
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Load raw CSV, basic validation
│   │   ├── preprocessor.py            # Encode, scale, SMOTE
│   │   └── splitter.py                # Train/test split with stratification
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ann_model.py               # Main ANN architecture
│   │   ├── baseline_model.py          # Simple 1-layer baseline for comparison
│   │   └── optimizer_comparison.py    # SGD vs RMSProp vs ADAM experiment
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Precision, Recall, F1, ROC-AUC
│   │   ├── plotter.py                 # Loss curves, confusion matrix, ROC
│   │   └── threshold_tuner.py         # Find optimal decision threshold
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       └── seed.py                    # Reproducibility (random seeds)
│
├── outputs/
│   ├── models/                        # Saved .keras model files
│   ├── plots/                         # PNG plots (loss curves, ROC, etc.)
│   └── reports/                       # classification_report.txt
│
├── tests/
│   ├── test_preprocessor.py
│   └── test_model.py
│
└── train.py                           # Main entry point — run this to train
```

## Quick Start (3 commands)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle and place CSV in data/raw/
#    https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

# 3. Train the model
python train.py
```

## Expected Results
| Metric | Baseline ANN | Tuned ANN (with BN + Dropout + L2) |
|--------|-------------|--------------------------------------|
| Accuracy | ~88% | ~91% |
| Fraud Recall | ~45% | ~72% |
| ROC-AUC | ~0.78 | ~0.88 |

*Note: Fraud Recall is the key metric — catching fraud matters more than overall accuracy.*
