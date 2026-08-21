"""Leakage-safe canonical preprocessing for the insurance fraud dataset."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ..config import DATA_PROC_DIR, RANDOM_STATE, SCALER_SAVE_PATH, SMOTE_STRATEGY, TARGET_COLUMN
from ..utils.logger import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea",
    "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Sex",
    "MaritalStatus", "Age", "Fault", "PolicyType", "VehicleCategory",
    "VehiclePrice", "Deductible", "DriverRating", "Days_Policy_Accident",
    "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars", "BasePolicy",
]
IDENTIFIER_COLUMNS = ["PolicyNumber", "RepNumber", "Year"]
COLUMNS_TO_DROP = IDENTIFIER_COLUMNS
CLAIM_UNKNOWN_COLUMNS = ["DayOfWeekClaimed", "MonthClaimed"]


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def _validate_schema(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN] + IDENTIFIER_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_features(df: pd.DataFrame, age_fill_value: Optional[float] = None,
                   fit: bool = False):
    """Clean claim sentinels and replace Age==0 using a train-derived median."""
    cleaned = df.loc[:, FEATURE_COLUMNS].copy()
    metadata: Dict[str, float] = {}
    for col in CLAIM_UNKNOWN_COLUMNS:
        cleaned[col] = cleaned[col].astype(str).replace("0", "Unknown")
    age = pd.to_numeric(cleaned["Age"], errors="coerce")
    if fit:
        age_fill_value = float(age.mask(age.eq(0)).median())
        metadata["age_fill_value"] = age_fill_value
    if age_fill_value is None or not np.isfinite(age_fill_value):
        raise ValueError("A training-derived Age replacement value is required")
    cleaned["Age"] = age.mask(age.eq(0), age_fill_value)
    return cleaned, metadata


class CanonicalPreprocessor:
    """Encoder and scaler fitted only on training data and reusable for inference."""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.encoder = None
        self.scaler = None
        self.cleaning_metadata: Dict[str, float] = {}

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared, _ = clean_features(df, self.cleaning_metadata["age_fill_value"])
        return prepared

    def fit(self, X_train: pd.DataFrame):
        if list(X_train.columns) != self.feature_columns:
            raise ValueError("Input columns do not match the canonical feature order")
        prepared, self.cleaning_metadata = clean_features(X_train, fit=True)
        self.categorical_columns = prepared.select_dtypes(
            include=["object", "string", "category"]).columns.tolist()
        self.numeric_columns = [c for c in self.feature_columns if c not in self.categorical_columns]
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder.fit(prepared[self.categorical_columns])
        self.scaler = StandardScaler().fit(self._encode(prepared))
        return self

    def _encode(self, prepared: pd.DataFrame) -> np.ndarray:
        numeric = prepared[self.numeric_columns].astype(float).to_numpy()
        categorical = self.encoder.transform(prepared[self.categorical_columns])
        columns = []
        for col in self.feature_columns:
            if col in self.numeric_columns:
                columns.append(numeric[:, self.numeric_columns.index(col)])
            else:
                columns.append(categorical[:, self.categorical_columns.index(col)])
        return np.column_stack(columns).astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.scaler is None:
            raise RuntimeError("CanonicalPreprocessor must be fitted on training data first")
        if list(X.columns) != self.feature_columns:
            raise ValueError("Input columns do not match the canonical feature order")
        return self.scaler.transform(self._encode(self._prepare(X))).astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def metadata(self) -> dict:
        categories = self.encoder.categories_ if self.encoder is not None else []
        return {"feature_order": list(self.feature_columns),
                "categorical_columns": list(self.categorical_columns),
                "numeric_columns": list(self.numeric_columns),
                "encoder_categories": [list(x) for x in categories],
                "cleaning_metadata": dict(self.cleaning_metadata)}

    def save(self, path: str = SCALER_SAVE_PATH):
        if self.scaler is None:
            raise RuntimeError("Fit the preprocessor before saving it")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = SCALER_SAVE_PATH):
        return joblib.load(path)


def split_raw_dataframe(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> DataSplits:
    """Separate target, drop identifiers, and make the untouched three-way split."""
    from .splitter import split_three_way
    _validate_schema(df)
    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(np.int32)
    arrays = split_three_way(X, y, random_state=random_state)
    return DataSplits(*arrays[:3], *arrays[3:])


def prepare_training_data(df: pd.DataFrame, random_state: int = RANDOM_STATE,
                          smote_strategy: float = SMOTE_STRATEGY):
    """Split raw data, fit preprocessing on train, transform all, and SMOTE train only."""
    splits = split_raw_dataframe(df, random_state=random_state)
    preprocessor = CanonicalPreprocessor()
    X_train = preprocessor.fit_transform(splits.X_train)
    X_validation = preprocessor.transform(splits.X_validation)
    X_test = preprocessor.transform(splits.X_test)
    X_train_sm, y_train_sm = apply_smote(
        X_train, splits.y_train.to_numpy(), strategy=smote_strategy,
        random_state=random_state)
    return {"X_train": X_train_sm, "y_train": y_train_sm,
            "X_train_original": X_train, "y_train_original": splits.y_train.to_numpy(),
            "X_validation": X_validation, "y_validation": splits.y_validation.to_numpy(),
            "X_test": X_test, "y_test": splits.y_test.to_numpy(),
            "preprocessor": preprocessor, "feature_names": list(FEATURE_COLUMNS),
            "raw_splits": splits}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for legacy tests; canonical code uses train-fitted OrdinalEncoder."""
    result = df.copy()
    for col in result.select_dtypes(include=["object", "string", "category"]).columns:
        result[col] = pd.factorize(result[col].astype(str), sort=True)[0]
    return result


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, strategy: float = SMOTE_STRATEGY,
                random_state: int = RANDOM_STATE):
    """Apply SMOTE to a training partition only."""
    X_res, y_res = SMOTE(sampling_strategy=strategy, random_state=random_state).fit_resample(X_train, y_train)
    return X_res.astype(np.float32), y_res.astype(np.float32)


def preprocess(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None):
    """Legacy array API; use ``prepare_training_data`` for the canonical workflow."""
    X_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    X_df = X_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in X_df], errors="ignore")
    X_df = encode_categoricals(X_df)
    X = X_df.to_numpy(dtype=np.float32)
    if fit_scaler:
        scaler = StandardScaler().fit(X)
    elif scaler is None:
        scaler = joblib.load(SCALER_SAVE_PATH)
    return scaler.transform(X), df[TARGET_COLUMN].to_numpy(dtype=np.float32), scaler, list(X_df.columns)


def save_processed(X_train, X_test, y_train, y_test, X_validation=None, y_validation=None):
    os.makedirs(DATA_PROC_DIR, exist_ok=True)
    for name, value in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
        np.save(os.path.join(DATA_PROC_DIR, f"{name}.npy"), value)
    if X_validation is not None:
        np.save(os.path.join(DATA_PROC_DIR, "X_validation.npy"), X_validation)
    if y_validation is not None:
        np.save(os.path.join(DATA_PROC_DIR, "y_validation.npy"), y_validation)


def load_processed():
    return tuple(np.load(os.path.join(DATA_PROC_DIR, f"{name}.npy"))
                 for name in ["X_train", "X_test", "y_train", "y_test"])
