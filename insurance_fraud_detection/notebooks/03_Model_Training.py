"""
notebooks/03_Model_Training.py — Training experiments and ablation study.

Runs 4 experiments to show the impact of each technique:
  Experiment 1: Baseline (no regularization)
  Experiment 2: + L1/L2 Regularization
  Experiment 3: + BatchNorm + Dropout
  Experiment 4: + ADAM + SMOTE + Class Weights (full stack)

This is your ablation study — shows each component's contribution.
"""

import sys, os
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping

from src.data.preprocessor import load_processed
from src.evaluation.metrics import evaluate

# Load preprocessed data
X_train, X_test, y_train, y_test = load_processed()
input_dim = X_train.shape[1]
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

es = EarlyStopping(monitor="val_loss", patience=10,
                   restore_best_weights=True, verbose=0)

RESULTS = {}

# ─── Experiment 1: Pure Baseline ─────────────────────────────────────────────
print("\n── Experiment 1: Baseline (SGD, sigmoid hidden, no regularization)")

m1 = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation="sigmoid"),
    Dense(1, activation="sigmoid"),
], name="E1_Baseline")
m1.compile(optimizer=SGD(0.01), loss="binary_crossentropy",
           metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.Recall(name="recall")])
h1 = m1.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=50, batch_size=32, callbacks=[es], verbose=0)
RESULTS["E1 Baseline"] = evaluate(m1, X_test, y_test, threshold=0.40)

# ─── Experiment 2: + L2 Regularization ───────────────────────────────────────
print("\n── Experiment 2: + L2 Regularization")

m2 = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(1, activation="sigmoid"),
], name="E2_L2_Reg")
m2.compile(optimizer=SGD(0.01), loss="binary_crossentropy",
           metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.Recall(name="recall")])
h2 = m2.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=50, batch_size=32, callbacks=[es], verbose=0)
RESULTS["E2 +L2"] = evaluate(m2, X_test, y_test, threshold=0.40)

# ─── Experiment 3: + BatchNorm + Dropout ─────────────────────────────────────
print("\n── Experiment 3: + BatchNorm + Dropout")

m3 = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
], name="E3_BN_Dropout")
m3.compile(optimizer=SGD(0.01), loss="binary_crossentropy",
           metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.Recall(name="recall")])
h3 = m3.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=50, batch_size=32, callbacks=[es], verbose=0)
RESULTS["E3 +BN+Drop"] = evaluate(m3, X_test, y_test, threshold=0.40)

# ─── Experiment 4: Full Stack (ADAM + class weights) ─────────────────────────
print("\n── Experiment 4: Full Stack — ADAM + BN + Dropout + L2 + class_weight")

m4 = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid"),
], name="E4_Full_Stack")
m4.compile(optimizer=Adam(0.001), loss="binary_crossentropy",
           metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.Recall(name="recall")])
h4 = m4.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=100, batch_size=32, callbacks=[es],
            class_weight={0: 1.0, 1: 3.0}, verbose=0)
RESULTS["E4 Full Stack"] = evaluate(m4, X_test, y_test, threshold=0.40)

# ─── Comparison table ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<18} {'Accuracy':>10} {'Recall':>10} {'AUC':>10} {'F1':>10}")
print("-"*65)
for name, m in RESULTS.items():
    print(f"{name:<18} {m['accuracy']*100:>9.1f}% "
          f"{m['fraud_recall']*100:>9.1f}% "
          f"{m['roc_auc']:>10.4f} "
          f"{m['fraud_f1']*100:>9.1f}%")
print("="*65)

# ─── Learning curves comparison ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
histories = {"E1": h1, "E2": h2, "E3": h3, "E4": h4}
colors = ["#E24B4A", "#378ADD", "#1D9E75", "#BA7517"]

for ax, metric in zip(axes, ["val_loss", "val_auc"]):
    for (name, h), color in zip(histories.items(), colors):
        ax.plot(h.history[metric], label=name, color=color, lw=1.5)
    ax.set_title(metric.replace("val_", "Validation "))
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Ablation Study — Impact of Each Component", fontsize=12)
plt.tight_layout()
plt.savefig("../outputs/plots/ablation_study.png", dpi=150, bbox_inches="tight")
plt.show()
