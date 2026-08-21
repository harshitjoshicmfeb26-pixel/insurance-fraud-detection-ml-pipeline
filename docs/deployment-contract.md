# Hugging Face deployment contract

This document records the current operational deployment contract for the Space:

https://huggingface.co/spaces/harshitjoshiai/insurance-fraud-detector

The Space is deployed separately from GitHub. Updating this repository does not update or redeploy the live Space.

## Current deployed model

The current operational model is a scikit-learn `GradientBoostingClassifier`. It was selected for the comparative benchmark's highest F1 and PR-AUC; it is not claimed to be universally best.

The deployed bundle loads:

```text
artifacts/gradient_boosting_model.joblib
artifacts/preprocessor.joblib
artifacts/deployment_metadata.json
```

The deployment does not load `fraud_detector_final.keras`, `scaler.joblib`, or TensorFlow at inference time.

## Preprocessing and feature contract

The model expects exactly 29 features in this order:

1. Month
2. WeekOfMonth
3. DayOfWeek
4. Make
5. AccidentArea
6. DayOfWeekClaimed
7. MonthClaimed
8. WeekOfMonthClaimed
9. Sex
10. MaritalStatus
11. Age
12. Fault
13. PolicyType
14. VehicleCategory
15. VehiclePrice
16. Deductible
17. DriverRating
18. Days_Policy_Accident
19. Days_Policy_Claim
20. PastNumberOfClaims
21. AgeOfVehicle
22. AgeOfPolicyHolder
23. PoliceReportFiled
24. WitnessPresent
25. AgentType
26. NumberOfSuppliments
27. AddressChange_Claim
28. NumberOfCars
29. BasePolicy

`PolicyNumber`, `RepNumber`, and `Year` are identifiers and are excluded from model features. The saved preprocessor contains the training-fitted `OrdinalEncoder`, `StandardScaler`, feature order, numeric/categorical column lists, and the training-derived `Age` replacement value.

The current local deployment bundle includes a self-contained copy of the business package under `deployment/huggingface/insurance_fraud_detection/`. The app prefers this bundled package when run from the deployment directory; repository `src/` is only a local-development fallback. The bundled business files must be synchronized with `src/insurance_fraud_detection/business/` before a Space update.

## Operating threshold and presentation bands

The validation-selected binary operating threshold is:

```text
0.24
```

The current presentation policy is:

```text
score < 0.24         -> Lower Fraud Risk
0.24 <= score < 0.40 -> Elevated Fraud Risk
0.40 <= score < 0.50 -> High Review Priority
score >= 0.50       -> Very High Review Priority
```

Only `0.24` is the validated model operating threshold. The `0.40` and `0.50` boundaries are UX/triage interpretation bands and must not be described as separately optimized ML thresholds.

## ANN project history

TensorFlow/Keras, the canonical ANN, and ANN v3 remain part of the broader model-development and comparison project. ANN v3 is the strongest deep-learning model in the benchmark. These ANN artifacts are not the current operational Hugging Face model.

## Dependencies and packaging

The current deployment requires Gradio, pandas, NumPy, scikit-learn, and joblib. TensorFlow is not required for current inference. The deployment folder must be packaged with:

- `app.py`
- `insurance_fraud_detection/business/`
- `artifacts/gradient_boosting_model.joblib`
- `artifacts/preprocessor.joblib`
- `artifacts/deployment_metadata.json`
- `requirements.txt`

## Safety requirements

Do not change the model filename, preprocessor filename, feature order, categorical mappings, input field names, business policy, or operating threshold without updating and retesting the Space together with those changes. Model scores are risk signals for human review, not proof of fraud or automatic claim decisions.
