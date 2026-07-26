"""
data/loader.py — Load the raw Kaggle CSV and perform basic validation.

Dataset: Vehicle Insurance Claim Fraud Detection
URL: kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
File: fraud_oracle.csv  (15,421 rows × 33 columns)
"""

import os
import pandas as pd

from config import RAW_CSV, TARGET_COLUMN
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_raw_data(path: str = RAW_CSV) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.

    Returns
    -------
    pd.DataFrame with 15,421 rows and 33 columns.

    Raises
    ------
    FileNotFoundError  — if the CSV hasn't been downloaded yet.
    ValueError         — if the target column is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[!] Dataset not found at: {path}\n"
            "    Download it from Kaggle:\n"
            "    https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection\n"
            "    Then place fraud_oracle.csv in data/raw/"
        )

    df = pd.read_csv(path)
    log.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    fraud_rate = df[TARGET_COLUMN].mean() * 100
    log.info(f"Fraud rate in dataset: {fraud_rate:.1f}%  "
             f"({df[TARGET_COLUMN].sum()} fraudulent / {len(df)} total)")

    return df


def get_feature_types(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """
    Separate column names into categorical and numerical groups.

    Returns
    -------
    cat_cols : list of categorical column names
    num_cols : list of numerical column names
    """
    feature_df = df.drop(columns=[target])
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    log.info(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    log.info(f"Numerical features  ({len(num_cols)}): {num_cols}")

    return cat_cols, num_cols
