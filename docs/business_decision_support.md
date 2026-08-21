# Business Decision-Support Layer

This project uses the validated Gradient Boosting fraud model to prioritize vehicle-insurance claims for human investigation. It does not predict claim severity, claim amount, financial loss, or expected loss.

## Business workflow

```text
Claim
  → Existing preprocessing
  → Gradient Boosting score
  → Central risk policy
  → Explanation and contextual cues
  → Recommended action
  → Human review
```

For portfolios:

```text
Claims portfolio → Batch scoring → Probability ranking → Investigation queue
                → Investigator capacity → Fraud-capture analysis
```

## Intended users

- Claims Processing Team: ordinary claim processing and verification.
- Fraud Analysts: review claims flagged by the system.
- Investigation Managers: prioritize workload against team capacity.
- Model and Risk Analysts: monitor ranking effectiveness and threshold trade-offs.

## Existing ML contract

The business layer does not retrain or alter the validated model. It preserves the `FraudFound_P` target, 29-feature schema, identifier removal, train-fitted preprocessing, training-only SMOTE, canonical 70/15/15 split, validation-selected thresholding, and untouched final-test evaluation. The current deployed model is `GradientBoostingClassifier`; the threshold is `0.24`.

## Risk policy and actions

| Score band | Priority | Recommended action |
|---|---|---|
| `< 0.24` | Standard | Standard Claim Processing |
| `0.24–<0.40` | Elevated | Additional Verification |
| `0.40–<0.50` | High | Fraud Analyst Review |
| `>=0.50` | Priority | Priority Fraud Investigation |

`0.24` is the validation-selected operating threshold. The `0.40` and `0.50` boundaries are presentation/triage bands, not separately optimized model thresholds.

## Investigation queue

Batch claims are ranked by `fraud_probability` descending with stable deterministic tie handling. Identifiers such as `PolicyNumber` are display-only and are excluded from the 29-feature model matrix. The queue answers which claims should be reviewed first when investigators cannot inspect every claim.

## Portfolio metrics

Outcome-dependent metrics use labeled final-test data only:

- Fraud Capture @ K: fraud cases in the top K% divided by all fraud cases.
- Precision @ K: fraud cases in the top K% divided by claims reviewed in that segment.
- Lift @ K: Precision @ K divided by overall fraud prevalence.
- Cumulative gains: percentage of claims reviewed versus percentage of known fraud captured.
- Lift by decile: fraud rate and lift for each score-ranked decile.

When K% does not produce an integer number of claims, the implementation uses ceiling and records that rounding policy in the generated metadata. Metrics are not calculated on unlabeled uploaded claims.

## Human-in-the-loop policy

The model produces a risk signal for investigation support. It does not prove fraud, establish causality, automatically reject claims, or replace an investigator's decision. SHAP, if added later, will explain model behavior rather than prove why fraud occurred.

## Limitations and production considerations

The dataset is historical tabular data with a relatively small fraud class. Probability calibration is not implemented. Queue performance may change with population drift, operational changes, or different investigator capacity. Production use would require monitoring, governance, access controls, audit trails, and validation against current claims data.

## Sample batch data

`examples/sample_claims.csv` contains valid schema examples without `FraudFound_P`. In a real insurer, batch claim records would normally come from the claims-management system or an operational database. The sample contains no severity or financial-loss fields.
