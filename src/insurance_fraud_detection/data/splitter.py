"""Reproducible stratified train/validation/test splitting."""

import numpy as np
from sklearn.model_selection import train_test_split

from ..config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from ..utils.logger import get_logger

log = get_logger(__name__)


def split_three_way(X: np.ndarray, y: np.ndarray,
                    validation_size: float = VALIDATION_SIZE,
                    test_size: float = TEST_SIZE,
                    random_state: int = RANDOM_STATE):
    """Return reproducible stratified 70/15/15 train, validation, and test arrays."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must leave a positive train split")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=validation_size + test_size,
        random_state=random_state, stratify=y)
    relative_test_size = test_size / (validation_size + test_size)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size,
        random_state=random_state, stratify=y_temp)

    for name, labels in [("Train", y_train), ("Validation", y_validation), ("Test", y_test)]:
        log.info(f"{name}: {len(labels)} samples (fraud: {int(labels.sum())}, "
                 f"legit: {int((labels == 0).sum())})")
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def split(X: np.ndarray, y: np.ndarray):
    """Canonical split entry point."""
    return split_three_way(X, y)
