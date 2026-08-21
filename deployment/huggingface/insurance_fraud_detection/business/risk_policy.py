"""Deployment copy of the canonical risk policy."""

from dataclasses import dataclass


OPERATING_THRESHOLD = 0.24


@dataclass(frozen=True)
class RiskBandPolicy:
    name: str
    review_priority: str
    recommended_action: str
    business_interpretation: str


LOWER = RiskBandPolicy(
    "Lower Fraud Risk", "Standard", "Standard Claim Processing",
    "The model score is below the validated fraud-review threshold. "
    "Continue normal verification and processing.")
ELEVATED = RiskBandPolicy(
    "Elevated Fraud Risk", "Elevated", "Additional Verification",
    "Perform enhanced standard checks, including supporting-document, "
    "policy-information, and claim-detail verification.")
HIGH = RiskBandPolicy(
    "High Review Priority", "High", "Fraud Analyst Review",
    "Route the claim to the fraud team's investigation queue for analyst review.")
VERY_HIGH = RiskBandPolicy(
    "Very High Review Priority", "Priority", "Priority Fraud Investigation",
    "Give the claim higher priority within the fraud-investigation queue.")


def policy_for_probability(probability: float) -> RiskBandPolicy:
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("fraud probability must be between 0 and 1")
    score = float(probability)
    if score < OPERATING_THRESHOLD:
        return LOWER
    if score < 0.40:
        return ELEVATED
    if score < 0.50:
        return HIGH
    return VERY_HIGH
