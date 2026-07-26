"""
data/preprocessor.py — Full preprocessing pipeline.

Steps:
  1. Drop irrelevant columns (policy number, date fields)
  2. Label-encode all categorical features
  3. StandardScaler on numerical features
  4. SMOTE oversampling on training set (handles class imbalance)
  5. Save processed arrays + fitted scaler to disk

Syllabus links:
  - StandardScaler     → Session 5 (Standardization)
  - SMOTE              → Session 9 (Data Augmentation for small/imbalanced data)
  - LabelEncoder       → basic preprocessing required for ANN
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from config import (TARGET_COLUMN, RANDOM_STATE, SMOTE_STRATEGY,
                    DATA_PROC_DIR, SCALER_SAVE_PATH)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns to drop — not useful as model features
COLUMNS_TO_DROP = [
    "PolicyNumber",       # unique identifier, no predictive value
    "RepNumber",          # agent rep number
    "PolicyReport",       # highly correlated with other fields
]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all object/category columns in-place.

    Why LabelEncoder not OneHotEncoder?
    ANN can handle ordinal-encoded categoricals well when combined with
    BatchNormalization. OneHot would massively inflate feature count here.

    Returns a copy of df with all columns as numeric types.
    """
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
        log.info(f"  Encoded: {col} → {df[col].nunique()} unique values")

    return df


def preprocess(df: pd.DataFrame, fit_scaler: bool = True,
               scaler: StandardScaler = None):
    """
    Main preprocessing function.

    Parameters
    ----------
    df          : Raw DataFrame from loader.py
    fit_scaler  : True when processing training data (fits a new scaler).
                  False when processing test/inference data (uses saved scaler).
    scaler      : Pre-fitted scaler (required when fit_scaler=False).

    Returns
    -------
    X           : np.ndarray of shape (n_samples, n_features)
    y           : np.ndarray of shape (n_samples,)
    scaler      : The fitted StandardScaler (save this for inference!)
    feature_names : list of feature column names (for interpretability)
    """
    df = df.copy()

    # ── Step 1: Drop irrelevant columns ──────────────────────────────────────
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)
    log.info(f"Dropped {len(cols_present)} irrelevant columns")

    # ── Step 2: Separate target ───────────────────────────────────────────────
    y = df[TARGET_COLUMN].values.astype(np.float32)
    X_df = df.drop(columns=[TARGET_COLUMN])

    # ── Step 3: Encode categoricals ───────────────────────────────────────────
    X_df = encode_categoricals(X_df)
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)
    log.info(f"Feature matrix shape after encoding: {X.shape}")

    # ── Step 4: Scale features ────────────────────────────────────────────────
    # CRITICAL: fit only on train data, transform on test
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        log.info(f"Scaler fitted and saved → {SCALER_SAVE_PATH}")
    else:
        if scaler is None:
            scaler = joblib.load(SCALER_SAVE_PATH)
            log.info(f"Scaler loaded from → {SCALER_SAVE_PATH}")
        X = scaler.transform(X)

    return X, y, scaler, feature_names


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                strategy: float = SMOTE_STRATEGY,
                random_state: int = RANDOM_STATE):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to training data.

    Why SMOTE instead of random oversampling?
    SMOTE generates synthetic fraud samples by interpolating between existing
    fraud samples in feature space — it adds NEW information rather than
    duplicating rows, which helps the model generalise better.

    Syllabus: Session 9 — Data Augmentation.

    Parameters
    ----------
    strategy  : float — target ratio of minority:majority class after resampling.
                0.5 means fraud will be ~50% as common as non-fraud.

    Returns
    -------
    X_resampled, y_resampled : balanced arrays ready for training
    """
    before = np.bincount(y_train.astype(int))
    log.info(f"Class distribution BEFORE SMOTE: {dict(enumerate(before))}")

    sm = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    after = np.bincount(y_res.astype(int))
    log.info(f"Class distribution AFTER SMOTE:  {dict(enumerate(after))}")

    return X_res.astype(np.float32), y_res.astype(np.float32)


def save_processed(X_train, X_test, y_train, y_test):
    """Save processed numpy arrays to data/processed/ for quick reloading."""
    os.makedirs(DATA_PROC_DIR, exist_ok=True)
    np.save(os.path.join(DATA_PROC_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROC_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_PROC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_PROC_DIR, "y_test.npy"),  y_test)
    log.info(f"Processed data saved to {DATA_PROC_DIR}/")


def load_processed():
    """Load previously saved processed arrays."""
    X_train = np.load(os.path.join(DATA_PROC_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_PROC_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_PROC_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_PROC_DIR, "y_test.npy"))
    log.info("Processed arrays loaded from disk")
    return X_train, X_test, y_train, y_test
