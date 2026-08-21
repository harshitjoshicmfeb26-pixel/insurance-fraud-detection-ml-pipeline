import numpy as np
import pandas as pd

from insurance_fraud_detection.business.batch import score_claims
from insurance_fraud_detection.business.decision import assess_probability
from insurance_fraud_detection.business.portfolio import (
    cumulative_gains, lift_by_decile, top_k_metrics)
from insurance_fraud_detection.business.risk_policy import OPERATING_THRESHOLD


def test_risk_policy_boundaries_and_actions():
    cases = [
        (0.00, "Lower Fraud Risk", "Standard Claim Processing"),
        (0.239999, "Lower Fraud Risk", "Standard Claim Processing"),
        (0.24, "Elevated Fraud Risk", "Additional Verification"),
        (0.399999, "Elevated Fraud Risk", "Additional Verification"),
        (0.40, "High Review Priority", "Fraud Analyst Review"),
        (0.499999, "High Review Priority", "Fraud Analyst Review"),
        (0.50, "Very High Review Priority", "Priority Fraud Investigation"),
        (1.00, "Very High Review Priority", "Priority Fraud Investigation"),
    ]
    for score, band, action in cases:
        decision = assess_probability(score)
        assert decision.risk_band == band
        assert decision.recommended_action == action
        assert decision.operating_threshold == OPERATING_THRESHOLD
    assert assess_probability(0.24).above_operating_threshold is True


def test_structured_decision_contains_business_fields():
    result = assess_probability(0.67).to_dict()
    assert set(result) >= {
        "fraud_probability", "operating_threshold",
        "above_operating_threshold", "risk_band", "review_priority",
        "recommended_action", "business_interpretation", "disclaimer",
    }
    assert result["fraud_probability"] == 0.67


class _DummyModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.received_shape = None

    def predict_proba(self, features):
        self.received_shape = features.shape
        return np.column_stack([1 - self.probabilities, self.probabilities])


def _dummy_bundle():
    class Encoder:
        def transform(self, frame):
            return np.zeros((len(frame), 1))

    class Scaler:
        def transform(self, array):
            return array

    return {
        "feature_columns": ["Age", "Category"],
        "numeric_columns": ["Age"],
        "categorical_columns": ["Category"],
        "encoder": Encoder(), "scaler": Scaler(), "age_fill_value": 35.0,
    }


def test_batch_scoring_preserves_rows_excludes_identifier_and_ranks():
    frame = pd.DataFrame({
        "PolicyNumber": ["B", "A", "C"],
        "Age": [30, 31, 32], "Category": ["x", "x", "x"],
        "FraudFound_P": [0, 1, 0],
    })
    model = _DummyModel([0.20, 0.80, 0.50])
    result = score_claims(frame, model, _dummy_bundle())
    assert len(result) == len(frame)
    assert list(result["claim_id"]) == ["A", "C", "B"]
    assert list(result["investigation_rank"]) == [1, 2, 3]
    assert model.received_shape == (3, 2)
    assert "PolicyNumber" not in _dummy_bundle()["feature_columns"]
    assert result["fraud_probability"].between(0, 1).all()


def test_top_k_metrics_hand_calculation_and_ceiling():
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    result = top_k_metrics(labels, scores, 20)
    assert result["claims_reviewed"] == 2
    assert result["fraud_selected"] == 1
    assert result["fraud_capture"] == 1 / 3
    assert result["precision"] == 0.5
    assert result["lift"] == (0.5 / 0.3)


def test_cumulative_gains_and_deciles_are_ranked():
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.arange(10, 0, -1)
    gains = cumulative_gains(labels, scores, percentages=(10, 100))
    assert list(gains["claims_reviewed"]) == [1, 10]
    assert list(gains["fraud_captured_percentage"]) == [1 / 3, 1.0]
    deciles = lift_by_decile(labels, scores)
    assert len(deciles) == 10
    assert deciles.iloc[0]["fraud_cases"] == 1
