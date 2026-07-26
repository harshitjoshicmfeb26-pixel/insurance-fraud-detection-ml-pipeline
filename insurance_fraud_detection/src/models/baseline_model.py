"""
models/baseline_model.py — Minimal 1-hidden-layer ANN baseline.

Used to show improvement from adding BatchNorm + Dropout + L2 (Sessions 7–11).
Always build the baseline first — it anchors your comparison table.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

from src.utils.logger import get_logger

log = get_logger(__name__)


def build_baseline(input_dim: int) -> tf.keras.Model:
    """
    Shallow ANN with 1 hidden layer, no regularization, SGD optimizer.
    This is the 'worst case' — shows what the model achieves without any
    of the improvements from Sessions 7–12.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="sigmoid"),   # old-style: sigmoid hidden layer
        Dense(1, activation="sigmoid"),
    ], name="Baseline_ANN")

    model.compile(
        optimizer=SGD(learning_rate=0.01),   # basic gradient descent
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")]
    )

    log.info(f"Baseline model: {model.count_params():,} parameters")
    return model
