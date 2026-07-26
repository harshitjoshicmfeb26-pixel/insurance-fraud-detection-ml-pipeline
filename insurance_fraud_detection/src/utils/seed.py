"""
utils/seed.py — Set all random seeds for reproducibility.
Call set_all_seeds() at the top of train.py and notebooks.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int = 42):
    """Fix random seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"[seed] All random seeds set to {seed}")
