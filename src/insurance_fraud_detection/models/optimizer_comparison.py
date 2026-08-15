"""
models/optimizer_comparison.py — Compare SGD, RMSProp, Momentum, ADAM.

Syllabus: Session 12 (ADAM, Mini-batch GD) + Session 13 (RMSProp, Momentum).

Running this file trains the same architecture with 4 different optimizers
and prints a comparison table. This is a great addition to your project report.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping

from ..config import PLOTS_DIR, RANDOM_STATE
from ..utils.logger import get_logger
from ..utils.seed import set_all_seeds

import os

log = get_logger(__name__)


OPTIMIZERS = {
    "SGD (vanilla)": SGD(learning_rate=0.01),
    "SGD + Momentum": SGD(learning_rate=0.01, momentum=0.9),
    "RMSProp":        RMSprop(learning_rate=0.001),
    "ADAM":           Adam(learning_rate=0.001),
}


def _build_fixed_model(input_dim: int, optimizer) -> tf.keras.Model:
    """Same architecture for all optimizers — only optimizer differs."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def compare_optimizers(X_train, y_train, X_test, y_test,
                       epochs: int = 40, batch_size: int = 32):
    """
    Train with each optimizer and compare val_loss curves.

    Returns
    -------
    results : dict {optimizer_name: history}
    """
    input_dim = X_train.shape[1]
    results = {}
    es = EarlyStopping(monitor="val_loss", patience=8,
                       restore_best_weights=True)

    for name, opt in OPTIMIZERS.items():
        log.info(f"\nTraining with optimizer: {name}")
        set_all_seeds(RANDOM_STATE)

        model = _build_fixed_model(input_dim, opt)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )
        results[name] = hist
        final_auc = max(hist.history["val_auc"])
        log.info(f"  → {name}: Best val_AUC = {final_auc:.4f}")

    _plot_optimizer_comparison(results)
    return results


def _plot_optimizer_comparison(results: dict):
    """Plot val_loss curves for all optimizers side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Optimizer Comparison — Validation Metrics", fontsize=13)

    colors = ["#E24B4A", "#378ADD", "#1D9E75", "#BA7517"]

    for ax, metric in zip(axes, ["val_loss", "val_auc"]):
        for (name, hist), color in zip(results.items(), colors):
            vals = hist.history.get(metric, [])
            ax.plot(vals, label=name, color=color, linewidth=1.5)
        ax.set_title(metric.replace("val_", "Validation ").title())
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "optimizer_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.info(f"Optimizer comparison plot saved → {path}")
    plt.show()
