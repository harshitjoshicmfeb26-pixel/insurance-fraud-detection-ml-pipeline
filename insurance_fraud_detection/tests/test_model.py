"""
tests/test_model.py — Unit tests for model architecture and output shape.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import tensorflow as tf

from src.models.ann_model import build_model
from src.models.baseline_model import build_baseline


@pytest.fixture
def dummy_data():
    np.random.seed(0)
    X = np.random.randn(100, 30).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    return X, y


def test_model_output_shape(dummy_data):
    """Model output must be shape (n_samples, 1) — one probability per claim."""
    X, _ = dummy_data
    model = build_model(input_dim=X.shape[1])
    preds = model.predict(X, verbose=0)
    assert preds.shape == (100, 1), f"Wrong output shape: {preds.shape}"


def test_model_output_range(dummy_data):
    """Sigmoid output must be in [0, 1]."""
    X, _ = dummy_data
    model = build_model(input_dim=X.shape[1])
    preds = model.predict(X, verbose=0).flatten()
    assert preds.min() >= 0.0, "Predictions below 0"
    assert preds.max() <= 1.0, "Predictions above 1"


def test_baseline_output_shape(dummy_data):
    X, _ = dummy_data
    model = build_baseline(input_dim=X.shape[1])
    preds = model.predict(X, verbose=0)
    assert preds.shape == (100, 1)


def test_model_trains_without_error(dummy_data):
    """Model must complete at least 2 epochs without raising an exception."""
    X, y = dummy_data
    model = build_model(input_dim=X.shape[1])
    history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 2


def test_model_loss_decreases(dummy_data):
    """Loss on epoch 5 must be less than epoch 1 (model is learning)."""
    X, y = dummy_data
    model = build_model(input_dim=X.shape[1])
    history = model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    loss_start = history.history["loss"][0]
    loss_end   = history.history["loss"][-1]
    assert loss_end < loss_start, \
        f"Model not learning: loss went {loss_start:.4f} → {loss_end:.4f}"
