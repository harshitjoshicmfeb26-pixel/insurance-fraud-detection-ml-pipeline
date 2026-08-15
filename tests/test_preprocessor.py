"""Tests for the leakage-safe Phase 2 data pipeline."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from insurance_fraud_detection.config import TARGET_COLUMN
from insurance_fraud_detection.data.loader import load_raw_data
from insurance_fraud_detection.data.preprocessor import (
    CanonicalPreprocessor,
    FEATURE_COLUMNS,
    apply_smote,
    encode_categoricals,
    prepare_training_data,
    split_raw_dataframe,
)
from insurance_fraud_detection.data.splitter import split_three_way


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture
def dummy_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "Age": np.random.randint(18, 70, n),
        "VehicleCategory": np.random.choice(["Sport", "Sedan", "Utility"], n),
        "PolicyType": np.random.choice(["All Perils", "Collision", "Liability"], n),
        "FraudFound_P": np.random.choice([0, 1], n, p=[0.94, 0.06]),
    })


def test_encode_categoricals(dummy_df):
    encoded = encode_categoricals(dummy_df.drop(columns=[TARGET_COLUMN]))
    assert all(encoded.dtypes != "object")
    assert encoded.shape == dummy_df.drop(columns=[TARGET_COLUMN]).shape


def test_smote_increases_minority(dummy_df):
    X = encode_categoricals(dummy_df.drop(columns=[TARGET_COLUMN])).to_numpy(dtype=float)
    y = dummy_df[TARGET_COLUMN].to_numpy(dtype=float)
    _, y_res = apply_smote(X, y, strategy=0.5)
    assert int(y_res.sum()) > int(y.sum())


def test_three_way_stratified_split_and_no_overlap():
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = np.tile([0] * 19 + [1], 10)
    train, validation, test, y_train, y_validation, y_test = split_three_way(X, y)
    assert (len(train), len(validation), len(test)) == (140, 30, 30)
    assert len(set(train[:, 0]) & set(validation[:, 0])) == 0
    assert len(set(train[:, 0]) & set(test[:, 0])) == 0
    assert len(set(validation[:, 0]) & set(test[:, 0])) == 0
    # With only ten positives, integer allocation makes a 1/30 vs 2/30
    # validation/test split the closest possible stratification.
    assert max(abs(part.mean() - y.mean()) for part in [y_train, y_validation, y_test]) <= 0.02


def test_target_separated_and_exact_feature_schema(raw_df):
    splits = split_raw_dataframe(raw_df)
    assert TARGET_COLUMN not in splits.X_train.columns
    assert list(splits.X_train.columns) == FEATURE_COLUMNS
    assert set(splits.X_train.columns).isdisjoint({"PolicyNumber", "RepNumber", "Year"})
    assert len(FEATURE_COLUMNS) == 29


def test_preprocessing_is_fitted_on_train_and_reused(raw_df):
    splits = split_raw_dataframe(raw_df)
    processor = CanonicalPreprocessor().fit(splits.X_train)
    means_before = processor.scaler.mean_.copy()
    categories_before = [x.copy() for x in processor.encoder.categories_]
    processor.transform(splits.X_validation)
    processor.transform(splits.X_test)
    np.testing.assert_array_equal(processor.scaler.mean_, means_before)
    for before, after in zip(categories_before, processor.encoder.categories_):
        np.testing.assert_array_equal(before, after)
    assert processor.scaler.n_samples_seen_ == len(splits.X_train)


def test_unknown_categories_are_safe_and_edge_cleaning_is_explicit(raw_df):
    splits = split_raw_dataframe(raw_df)
    processor = CanonicalPreprocessor().fit(splits.X_train)
    edge = splits.X_test.iloc[[0]].copy()
    edge.loc[:, "Make"] = "UnseenMake"
    edge.loc[:, "DayOfWeekClaimed"] = "0"
    edge.loc[:, "MonthClaimed"] = "0"
    edge.loc[:, "Age"] = 0
    edge.loc[:, "AgeOfPolicyHolder"] = "18 to 20"
    transformed = processor.transform(edge)
    assert transformed.shape == (1, 29)
    assert np.isfinite(transformed).all()
    assert processor.cleaning_metadata["age_fill_value"] > 0


def test_porche_is_not_silently_removed(raw_df):
    splits = split_raw_dataframe(raw_df)
    processor = CanonicalPreprocessor().fit(splits.X_train)
    edge = splits.X_test.iloc[[0]].copy()
    edge.loc[:, "Make"] = "Porche"
    transformed = processor.transform(edge)
    assert transformed.shape == (1, 29)
    assert "Make" in processor.feature_columns


def test_smote_changes_training_only(raw_df):
    prepared = prepare_training_data(raw_df)
    original_train_counts = np.bincount(prepared["y_train_original"].astype(int))
    validation_counts = np.bincount(prepared["y_validation"].astype(int))
    test_counts = np.bincount(prepared["y_test"].astype(int))
    assert len(prepared["y_train"]) > len(prepared["y_train_original"])
    np.testing.assert_array_equal(np.bincount(prepared["y_validation"].astype(int)), validation_counts)
    np.testing.assert_array_equal(np.bincount(prepared["y_test"].astype(int)), test_counts)
    assert np.array_equal(original_train_counts, np.bincount(prepared["y_train_original"].astype(int)))
