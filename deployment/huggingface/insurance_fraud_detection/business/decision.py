"""Deployment copy of the canonical structured decision object."""

from dataclasses import asdict, dataclass

from .risk_policy import OPERATING_THRESHOLD, policy_for_probability


@dataclass(frozen=True)
class DecisionAssessment:
    fraud_probability: float
    operating_threshold: float
    above_operating_threshold: bool
    risk_band: str
    review_priority: str
    recommended_action: str
    business_interpretation: str
    disclaimer: str

    def to_dict(self) -> dict:
        return asdict(self)


def assess_probability(probability: float) -> DecisionAssessment:
    score = float(probability)
    policy = policy_for_probability(score)
    return DecisionAssessment(
        fraud_probability=score,
        operating_threshold=OPERATING_THRESHOLD,
        above_operating_threshold=score >= OPERATING_THRESHOLD,
        risk_band=policy.name,
        review_priority=policy.review_priority,
        recommended_action=policy.recommended_action,
        business_interpretation=policy.business_interpretation,
        disclaimer=("This model provides a risk signal for investigation support "
                     "and does not establish that a claim is fraudulent."),
    )
