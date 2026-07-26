"""
config.py — Central configuration for Insurance Fraud Detection project.
Change hyperparameters here instead of hunting through multiple files.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
MODELS_DIR      = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR       = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR     = os.path.join(OUTPUTS_DIR, "reports")

RAW_CSV         = os.path.join(DATA_RAW_DIR, "fraud_oracle.csv")

# ─── Data ─────────────────────────────────────────────────────────────────────
TARGET_COLUMN   = "FraudFound_P"        # 1 = Fraud, 0 = Legitimate
TEST_SIZE       = 0.20                  # 80/20 split
RANDOM_STATE    = 42                    # for reproducibility
SMOTE_STRATEGY  = 0.5                   # ratio of minority:majority after SMOTE

# ─── Model Architecture ───────────────────────────────────────────────────────
INPUT_DIM       = None                  # auto-detected from data shape
HIDDEN_LAYERS   = [128, 64, 32]         # neurons per hidden layer
ACTIVATION      = "relu"                # hidden layer activation
OUTPUT_ACT      = "sigmoid"             # binary classification
DROPOUT_RATE    = 0.3                   # dropout after each hidden layer
L2_LAMBDA       = 0.001                 # L2 regularization strength
USE_BATCH_NORM  = True                  # apply BatchNormalization

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE   = 0.001
BATCH_SIZE      = 32
EPOCHS          = 100
PATIENCE        = 10                    # early stopping patience
CLASS_WEIGHT    = {0: 1.0, 1: 3.0}     # penalize missing fraud 3x more

# ─── Evaluation ───────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.40              # lower than 0.5 to catch more fraud
                                        # (higher recall, lower precision)

# ─── Saved Model ──────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "fraud_detector.keras")
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
