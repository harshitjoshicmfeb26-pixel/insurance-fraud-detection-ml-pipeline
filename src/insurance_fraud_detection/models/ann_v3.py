"""Faithful ANN v3 reconstruction and controlled optimization configurations."""

from dataclasses import asdict, dataclass

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

V3_HIDDEN_LAYERS = [64, 32]
V3_L2_LAMBDA = 0.01
V3_DROPOUT_RATES = [0.4, 0.3]
V3_LEARNING_RATE = 0.001
V3_BATCH_SIZE = 64
V3_EPOCHS = 200
V3_PATIENCE = 20
V3_SMOTE_STRATEGY = 0.15
V3_CLASS_WEIGHT = {0: 1.0, 1: 5.0}


@dataclass(frozen=True)
class ANNOptimizationConfig:
    """One candidate configuration for validation-only ANN selection."""

    candidate_id: str
    hidden_layers: tuple
    l2_lambda: float
    dropout_rates: tuple
    learning_rate: float
    batch_size: int
    smote_ratio: float
    fraud_class_weight: float

    def to_dict(self):
        values = asdict(self)
        values["hidden_layers"] = list(self.hidden_layers)
        values["dropout_rates"] = list(self.dropout_rates)
        return values


def build_optimized_ann(input_dim: int, config: ANNOptimizationConfig) -> tf.keras.Model:
    """Build a candidate model without changing the deployed or baseline models."""
    if len(config.hidden_layers) != len(config.dropout_rates):
        raise ValueError("Each hidden layer must have one dropout rate")
    model = Sequential(name=f"ANN_optimization_{config.candidate_id}")
    model.add(Input(shape=(input_dim,), name="input"))
    for index, (units, dropout_rate) in enumerate(
            zip(config.hidden_layers, config.dropout_rates), start=1):
        model.add(Dense(units, activation="relu",
                        kernel_regularizer=l2(config.l2_lambda),
                        name=f"dense_{index}"))
        model.add(BatchNormalization(name=f"bn_{index}"))
        model.add(Dropout(dropout_rate, name=f"dropout_{index}"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_v3(input_dim: int) -> tf.keras.Model:
    model = Sequential(name="Tuned_ANN_v3")
    model.add(Input(shape=(input_dim,), name="input"))
    for index, (units, dropout_rate) in enumerate(
            zip(V3_HIDDEN_LAYERS, V3_DROPOUT_RATES), start=1):
        model.add(Dense(units, activation="relu",
                        kernel_regularizer=l2(V3_L2_LAMBDA),
                        name=f"dense_{index}"))
        model.add(BatchNormalization(name=f"bn_{index}"))
        model.add(Dropout(dropout_rate, name=f"dropout_{index}"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(
        optimizer=Adam(learning_rate=V3_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model
