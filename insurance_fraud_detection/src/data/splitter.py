"""
data/splitter.py — Stratified train/test split for imbalanced classification.

Uses stratify=y to ensure both splits have the same fraud-to-legitimate ratio.
Without stratification, your test set might accidentally have no fraud cases.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_STATE
from src.utils.logger import get_logger

log = get_logger(__name__)


def split(X: np.ndarray, y: np.ndarray):
    """
    Stratified 80/20 train-test split.

    Parameters
    ----------
    X : feature matrix
    y : binary labels (0 = legitimate, 1 = fraud)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # keeps fraud % equal in both splits
    )

    log.info(f"Train: {X_train.shape[0]} samples  "
             f"(fraud: {int(y_train.sum())}, legit: {int((y_train == 0).sum())})")
    log.info(f"Test:  {X_test.shape[0]} samples  "
             f"(fraud: {int(y_test.sum())}, legit: {int((y_test == 0).sum())})")

    return X_train, X_test, y_train, y_test
