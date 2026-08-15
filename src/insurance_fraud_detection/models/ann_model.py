"""
models/ann_model.py — Main Deep ANN for insurance fraud detection.

Architecture:
  Input(n_features)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.3)   [Session 6, 9, 11]
    → Dense(64,  ReLU) + BatchNorm + Dropout(0.3)
    → Dense(32,  ReLU) + BatchNorm + Dropout(0.2)
    → Dense(1,   Sigmoid)                             [Session 5]

Regularization:
  - L2 weight regularization on Dense layers         [Session 7–8]
  - Dropout                                           [Session 9]
  - Batch Normalization                               [Session 11]
  - Early Stopping + Class Weights                    [Session 9]
  - ADAM optimizer                                    [Session 12]
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ..config import (HIDDEN_LAYERS, DROPOUT_RATE, L2_LAMBDA, LEARNING_RATE,
                      BATCH_SIZE, EPOCHS, PATIENCE, CLASS_WEIGHT,
                      USE_BATCH_NORM, MODEL_SAVE_PATH, MODELS_DIR)
from ..utils.logger import get_logger

log = get_logger(__name__)


def build_model(input_dim: int,
                hidden_layers: list = HIDDEN_LAYERS,
                dropout_rate: float = DROPOUT_RATE,
                l2_lambda: float = L2_LAMBDA,
                learning_rate: float = LEARNING_RATE,
                use_batch_norm: bool = USE_BATCH_NORM) -> tf.keras.Model:
    """
    Build and compile the fraud detection ANN.

    Parameters
    ----------
    input_dim     : number of input features (auto-detected from data)
    hidden_layers : list of neuron counts per hidden layer e.g. [128, 64, 32]
    dropout_rate  : dropout probability after each hidden layer
    l2_lambda     : L2 regularization coefficient
    learning_rate : ADAM learning rate
    use_batch_norm: whether to add BatchNormalization layers

    Returns
    -------
    Compiled tf.keras.Model ready for .fit()
    """
    model = Sequential(name="FraudDetector_ANN")
    model.add(Input(shape=(input_dim,), name="input"))

    for i, units in enumerate(hidden_layers):
        # ── Dense layer with L2 regularization ───────────────────────────────
        # L2 adds λ * sum(w²) to the loss — penalises large weights,
        # preventing overfitting. (Session 7–8)
        model.add(Dense(
            units,
            activation="relu",       # ReLU prevents vanishing gradient (Session 10)
            kernel_regularizer=l2(l2_lambda),
            name=f"dense_{i+1}"
        ))

        # ── Batch Normalization ───────────────────────────────────────────────
        # Normalizes activations across the mini-batch, stabilizing training.
        # Reduces sensitivity to weight initialization. (Session 11)
        if use_batch_norm:
            model.add(BatchNormalization(name=f"bn_{i+1}"))

        # ── Dropout ──────────────────────────────────────────────────────────
        # Randomly zeros out dropout_rate fraction of neurons each step.
        # Acts as ensemble learning — network can't rely on any single neuron.
        # Use smaller dropout on last hidden layer. (Session 9)
        dr = dropout_rate if i < len(hidden_layers) - 1 else dropout_rate * 0.7
        model.add(Dropout(dr, name=f"dropout_{i+1}"))

    # ── Output layer ─────────────────────────────────────────────────────────
    # Sigmoid squashes output to [0, 1] = probability of fraud. (Session 5)
    model.add(Dense(1, activation="sigmoid", name="output"))

    # ── Compile ──────────────────────────────────────────────────────────────
    # ADAM = Adaptive Moment Estimation (Session 12)
    # Combines Momentum (remembers past gradients) + RMSProp (adapts LR per param)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",     # correct loss for binary classification
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    log.info(f"Model built: {model.count_params():,} total parameters")
    model.summary(print_fn=log.info)

    return model


def get_callbacks(save_path: str = MODEL_SAVE_PATH, monitor: str = "val_loss",
                  patience: int = PATIENCE, include_reduce_lr: bool = True) -> list:
    """
    Build list of Keras callbacks for training.

    EarlyStopping   — stop when val_loss stops improving (Session 9)
    ReduceLROnPlateau — halve LR when stuck (helps escape plateaus)
    ModelCheckpoint — save the best model automatically
    """
    os.makedirs(os.path.dirname(save_path) or MODELS_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            mode="max" if monitor == "val_auc" else "min",
            patience=patience,
            restore_best_weights=True,   # revert to best epoch on stop
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,          # halve the learning rate
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
    ]
    if not include_reduce_lr:
        callbacks = [callbacks[0], callbacks[2]]
    return callbacks


def train(model: tf.keras.Model,
          X_train, y_train,
          X_val=None, y_val=None,
          batch_size: int = BATCH_SIZE,
          epochs: int = EPOCHS,
          class_weight: dict = CLASS_WEIGHT,
          checkpoint_path: str = MODEL_SAVE_PATH,
          monitor: str = "val_loss", patience: int = PATIENCE,
          include_reduce_lr: bool = True, verbose: int = 1):
    """
    Train the model.

    class_weight penalises misclassifying fraud more than legitimate claims.
    {0: 1.0, 1: 3.0} means a missed fraud (false negative) counts 3× in loss.

    Parameters
    ----------
    X_train, y_train  : SMOTE-balanced training data
    X_val, y_val      : validation data (or use validation_split if None)
    class_weight      : dict penalising minority class errors

    Returns
    -------
    history : Keras History object (loss/metric curves)
    """
    log.info(f"Training for up to {epochs} epochs | batch_size={batch_size}")
    log.info(f"Class weights: {class_weight}")

    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(checkpoint_path, monitor=monitor,
                                patience=patience,
                                include_reduce_lr=include_reduce_lr),
        verbose=verbose,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    if X_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.15

    history = model.fit(**fit_kwargs)

    log.info("Training complete.")
    log.info(f"Best val_auc: {max(history.history['val_auc']):.4f}")

    return history


def load_model(path: str = MODEL_SAVE_PATH) -> tf.keras.Model:
    """Load a previously saved model."""
    model = tf.keras.models.load_model(path)
    log.info(f"Model loaded from {path}")
    return model
