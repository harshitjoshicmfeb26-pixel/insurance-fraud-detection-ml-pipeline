"""Deployment copy of the artifact-compatible inference transform."""

from typing import Mapping

import numpy as np
import pandas as pd


def transform_with_bundle(frame: pd.DataFrame, bundle: Mapping) -> np.ndarray:
    feature_columns = list(bundle["feature_columns"])
    if list(frame.columns) != feature_columns:
        raise ValueError("Input columns do not match the canonical feature order")
    prepared = frame.loc[:, feature_columns].copy()
    for column in ("DayOfWeekClaimed", "MonthClaimed"):
        if column in prepared:
            prepared[column] = prepared[column].astype(str).replace("0", "Unknown")
    age = pd.to_numeric(prepared["Age"], errors="coerce")
    if age.isna().any():
        raise ValueError("Age must be numeric")
    prepared["Age"] = age.mask(age.eq(0), bundle["age_fill_value"])
    numeric_columns = list(bundle["numeric_columns"])
    categorical_columns = list(bundle["categorical_columns"])
    numeric = prepared[numeric_columns].apply(pd.to_numeric, errors="raise").to_numpy()
    categorical = bundle["encoder"].transform(prepared[categorical_columns])
    columns = []
    for column in feature_columns:
        if column in numeric_columns:
            columns.append(numeric[:, numeric_columns.index(column)])
        else:
            columns.append(categorical[:, categorical_columns.index(column)])
    encoded = np.column_stack(columns).astype(np.float32)
    return bundle["scaler"].transform(encoded).astype(np.float32)
