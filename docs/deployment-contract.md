# Hugging Face deployment contract

This document records the verified contract used by the live Space:

https://huggingface.co/spaces/harshitjoshiai/insurance-fraud-detector

The Space is deployed separately from GitHub. This repository snapshot does not update or redeploy it.

## Artifacts

The live application loads:

~~~text
fraud_detector_final.keras
scaler.joblib
~~~

The application does not load separate encoder files. It contains explicit categorical and ordinal mappings in app.py.

## Feature order

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

## Mappings

The live application uses these verified mapping groups:

- Alphabetical label-style maps for month, day, make, claimed day, accident area, sex, marital status, fault, policy type, vehicle category, base policy, and yes/no fields.
- Ordinal maps for vehicle price, policy/accident timing, policy/claim timing, past claims, vehicle age, policyholder age, supplements, address changes, and number of cars.

The exact executable mappings are preserved in app/app.py, which is a snapshot of the live source.

## Thresholds

~~~text
probability < 0.40  -> Low risk
0.40 to 0.59        -> Medium risk
probability >= 0.60 -> High risk
~~~

These values must not be changed without coordinating a deployment update.

## Dependencies

The live Space declares Gradio and uses TensorFlow, NumPy, scikit-learn, and joblib. Its verified requirements include:

~~~text
tensorflow==2.20.0
gradio
scikit-learn
joblib
numpy
~~~

## Verified edge-value differences

The local dataset contains values that are not represented identically by the live UI mappings:

- MonthClaimed == "0" occurs once in the dataset; the live MONTH_MAP does not include it.
- AgeOfPolicyHolder == "18 to 20" occurs 15 times; the live AGEPH_MAP does not include it.
- The dataset contains Make == "Porche" five times, while the live mapping uses "Porsche".

These differences are documented only; they are not changed in Phase 1.
