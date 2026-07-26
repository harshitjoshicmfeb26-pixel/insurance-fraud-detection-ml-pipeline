"""
tests/test_preprocessor.py — Unit tests for preprocessing pipeline.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import encode_categoricals, apply_smote
from src.data.splitter import split


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_df():
    """Small synthetic DataFrame mimicking the Kaggle structure."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "Age": np.random.randint(18, 70, n),
        "VehicleCategory": np.random.choice(["Sport", "Sedan", "Utility"], n),
        "PolicyType": np.random.choice(["All Perils", "Collision", "Liability"], n),
        "FraudFound_P": np.random.choice([0, 1], n, p=[0.94, 0.06]),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_encode_categoricals(dummy_df):
    """After encoding, all columns must be numeric."""
    df = dummy_df.drop(columns=["FraudFound_P"])
    encoded = encode_categoricals(df)
    assert all(encoded.dtypes != "object"), "Non-numeric columns remain after encoding"
    assert encoded.shape == df.shape, "Shape changed after encoding"


def test_smote_increases_minority(dummy_df):
    """SMOTE must increase the count of the minority class."""
    X = dummy_df.drop(columns=["FraudFound_P"]).values.astype(float)
    y = dummy_df["FraudFound_P"].values.astype(float)

    fraud_before = int(y.sum())
    X_res, y_res = apply_smote(X, y, strategy=0.5)
    fraud_after = int(y_res.sum())

    assert fraud_after > fraud_before, "SMOTE did not increase minority class"


def test_stratified_split(dummy_df):
    """Stratified split must preserve roughly equal fraud rate in both sets."""
    X = dummy_df.drop(columns=["FraudFound_P"]).values.astype(float)
    y = dummy_df["FraudFound_P"].values.astype(float)

    X_train, X_test, y_train, y_test = split(X, y)

    train_rate = y_train.mean()
    test_rate = y_test.mean()

    # Rates should be within 5% of each other
    assert abs(train_rate - test_rate) < 0.05, \
        f"Fraud rates diverged: train={train_rate:.3f}, test={test_rate:.3f}"


def test_no_data_leakage_in_split(dummy_df):
    """Train and test sets must not overlap."""
    X = dummy_df.drop(columns=["FraudFound_P"]).values.astype(float)
    y = dummy_df["FraudFound_P"].values.astype(float)

    X_train, X_test, y_train, y_test = split(X, y)

    # Convert rows to sets of tuples for overlap check
    train_set = set(map(tuple, X_train.tolist()))
    test_set  = set(map(tuple, X_test.tolist()))

    overlap = train_set & test_set
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping samples"
