"""Batch scoring and transparent investigation-queue ranking."""

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .decision import assess_probability
from .inference import transform_with_bundle


def score_claims(frame: pd.DataFrame, model, bundle: Mapping,
                 reference_column: str = "PolicyNumber") -> pd.DataFrame:
    """Score claims and rank them by model score, highest first.

    ``FraudFound_P`` is optional and is never passed to the model. Identifiers
    are retained only as display fields.
    """
    feature_columns = list(bundle["feature_columns"])
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required claim fields: {missing}")
    raw = frame.loc[:, feature_columns].copy()
    probabilities = np.asarray(model.predict_proba(
        transform_with_bundle(raw, bundle))[:, 1], dtype=float)
    rows = []
    for source_index, probability in zip(frame.index, probabilities):
        decision = assess_probability(probability)
        reference = frame.loc[source_index, reference_column] if reference_column in frame else None
        claim_id = str(reference) if reference is not None and str(reference).strip() else f"CLM-{len(rows) + 1:05d}"
        result = decision.to_dict()
        result.update({"claim_id": claim_id, "source_row": source_index,
                       "reference_id": reference})
        rows.append(result)
    scored = pd.DataFrame(rows)
    scored = scored.sort_values(
        ["fraud_probability", "claim_id", "source_row"],
        ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    scored.insert(0, "investigation_rank", np.arange(1, len(scored) + 1))
    return scored
