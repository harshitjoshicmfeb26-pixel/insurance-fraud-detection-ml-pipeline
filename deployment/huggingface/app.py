"""Local Gradio shell preserving the legacy grouped claim-form UX."""
import json
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    import gradio as gr
except ImportError:
    gr = None

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
_bundle = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
_model = joblib.load(ARTIFACT_DIR / "gradient_boosting_model.joblib")
with (ARTIFACT_DIR / "deployment_metadata.json").open(encoding="utf-8") as handle:
    _metadata = json.load(handle)

FEATURE_COLUMNS = list(_bundle["feature_columns"])
TARGET_COLUMN = _metadata["target_column"]
THRESHOLD = float(_metadata["selected_threshold"])
CATEGORICAL_CHOICES = {
    column: list(categories)
    for column, categories in zip(
        _bundle["categorical_columns"], _bundle["encoder"].categories_)
}

# Keep the legacy user-facing choices. The canonical encoder remains the source
# of truth for the backend; only compatibility translations happen internally.
_LEGACY_ORDER = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "MonthClaimed": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"],
    "DayOfWeekClaimed": ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"],
    "VehiclePrice": ["less than 20000", "20000 to 29000", "30000 to 39000",
                     "40000 to 59000", "60000 to 69000", "more than 69000"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years",
                      "6 years", "7 years", "more than 7"],
    "AgeOfPolicyHolder": ["16 to 17", "21 to 25", "26 to 30", "31 to 35",
                          "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year",
                             "2 to 3 years", "4 to 8 years"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}
VISIBLE_CHOICES = {}
for _column, _values in CATEGORICAL_CHOICES.items():
    _values = list(_LEGACY_ORDER.get(_column, _values))
    _values = [value for value in _values if value not in {"Unknown", "18 to 20"}]
    if _column == "Make":
        _values = ["Porsche" if value == "Porche" else value for value in _values]
    VISIBLE_CHOICES[_column] = _values

UI_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column != "AgeOfPolicyHolder"]
AGE_BAND_BY_AGE = {
    **{age: "18 to 20" for age in range(16, 18)},
    **{age: "21 to 25" for age in range(18, 21)},
    **{age: "26 to 30" for age in range(21, 26)},
    **{age: "31 to 35" for age in range(26, 36)},
    **{age: "36 to 40" for age in range(36, 46)},
    **{age: "41 to 50" for age in range(46, 56)},
    **{age: "51 to 65" for age in range(56, 66)},
    **{age: "over 65" for age in range(66, 81)},
}


def _derive_age_of_policy_holder(age):
    age = int(age)
    if age == 0:
        return "16 to 17"
    if age not in AGE_BAND_BY_AGE:
        raise ValueError("Age must be between 16 and 80, or 0 for canonical cleaning")
    return AGE_BAND_BY_AGE[age]


def _choices(column):
    return CATEGORICAL_CHOICES[column]


def _clean_and_transform(frame):
    """Apply canonical cleaning, encoding, feature order, and scaling."""
    prepared = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in ("DayOfWeekClaimed", "MonthClaimed"):
        prepared[column] = prepared[column].astype(str).replace("0", "Unknown")
    age = pd.to_numeric(prepared["Age"], errors="coerce")
    if age.isna().any():
        raise ValueError("Age must be numeric")
    prepared["Age"] = age.mask(age.eq(0), _bundle["age_fill_value"])
    numeric_columns = _bundle["numeric_columns"]
    categorical_columns = _bundle["categorical_columns"]
    numeric = prepared[numeric_columns].apply(pd.to_numeric, errors="raise").to_numpy()
    categorical = _bundle["encoder"].transform(prepared[categorical_columns])
    columns = []
    for column in FEATURE_COLUMNS:
        if column in numeric_columns:
            columns.append(numeric[:, numeric_columns.index(column)])
        else:
            columns.append(categorical[:, categorical_columns.index(column)])
    encoded = np.column_stack(columns).astype(np.float32)
    return _bundle["scaler"].transform(encoded).astype(np.float32)


def _frame_from_input(values: Mapping | Sequence):
    if isinstance(values, Mapping):
        missing = [column for column in UI_FEATURE_COLUMNS if column not in values]
        if missing:
            raise ValueError(f"Missing input fields: {missing}")
        row = {column: values[column] for column in UI_FEATURE_COLUMNS}
        if "AgeOfPolicyHolder" in values:
            row["AgeOfPolicyHolder"] = values["AgeOfPolicyHolder"]
        else:
            row["AgeOfPolicyHolder"] = _derive_age_of_policy_holder(row["Age"])
    else:
        if len(values) == len(UI_FEATURE_COLUMNS):
            row = dict(zip(UI_FEATURE_COLUMNS, values))
            row["AgeOfPolicyHolder"] = _derive_age_of_policy_holder(row["Age"])
        elif len(values) == len(FEATURE_COLUMNS):
            row = dict(zip(FEATURE_COLUMNS, values))
        else:
            raise ValueError(f"Expected {len(UI_FEATURE_COLUMNS)} visible claim fields")
    # The old UI uses the user-friendly spelling; the training CSV uses Porche.
    if row.get("Make") == "Porsche":
        row["Make"] = "Porche"
    for column, value in row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{column} is required")
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _risk_interpretation(probability):
    if probability < THRESHOLD:
        return "Lower Fraud Risk", "Proceed with standard processing"
    if probability < 0.50:
        return "Elevated Fraud Risk", "Review claim details and supporting information"
    return "Higher Fraud Risk - Review Recommended", "Prioritize the claim for investigation"


def _context_cues(values):
    cues = []
    cues.append("Police report filed" if values["PoliceReportFiled"] == "Yes"
                else "No police report filed")
    cues.append("Witness present" if values["WitnessPresent"] == "Yes"
                else "No witness recorded")
    if values["Days_Policy_Accident"] == "more than 30":
        cues.append("Long policy-to-accident interval")
    if values["PastNumberOfClaims"] in {"2 to 4", "more than 4"}:
        cues.append("Multiple past claims")
    if values["NumberOfSuppliments"] in {"3 to 5", "more than 5"}:
        cues.append("Several supplements")
    if values["AddressChange_Claim"] in {"under 6 months", "1 year"}:
        cues.append("Recent address change")
    if values["DriverRating"] == 1:
        cues.append("Lowest driver rating")
    if values["Fault"] == "Third Party":
        cues.append("Third-party fault recorded")
    return cues


def predict_claim(values: Mapping | Sequence):
    frame = _frame_from_input(values)
    transformed = _clean_and_transform(frame)
    probability = float(_model.predict_proba(transformed)[0, 1])
    interpretation, action = _risk_interpretation(probability)
    return {
        "fraud_probability": probability,
        "threshold": THRESHOLD,
        "risk_flag": bool(probability >= THRESHOLD),
        "interpretation": interpretation,
        "action": action,
        "context_cues": _context_cues(frame.iloc[0].to_dict()),
        "disclaimer": (
            "This model provides a risk signal for investigation support and does "
            "not establish that a claim is fraudulent."
        ),
    }


def predict_fraud(
    month, week_of_month, day_of_week, make, accident_area,
    day_of_week_claimed, month_claimed, week_of_month_claimed,
    sex, marital_status, age, fault, policy_type, vehicle_category,
    vehicle_price, deductible, driver_rating, days_policy_accident,
    days_policy_claim, past_claims, age_of_vehicle,
    police_report, witness_present, agent_type, num_supplements,
    address_change, num_cars, base_policy):
    values = [
        month, week_of_month, day_of_week, make, accident_area,
        day_of_week_claimed, month_claimed, week_of_month_claimed,
        sex, marital_status, age, fault, policy_type, vehicle_category,
        vehicle_price, deductible, driver_rating, days_policy_accident,
        days_policy_claim, past_claims, age_of_vehicle,
        police_report, witness_present, agent_type, num_supplements,
        address_change, num_cars, base_policy]
    try:
        result = predict_claim(values)
    except (TypeError, ValueError, KeyError) as exc:
        return None, "Input error", str(exc)
    cues = "\n".join(f"- {cue}" for cue in result["context_cues"])
    summary = (
        f"Fraud Risk Probability : {result['fraud_probability']:.1%}\n"
        f"Risk interpretation    : {result['interpretation']}\n"
        f"Review action          : {result['action']}\n\n"
        "Input context (not model explanations):\n"
        f"{cues}\n\n"
        f"{result['disclaimer']}\n\n"
        "Deployment model: Gradient Boosting\n"
        "Selected for F1 and PR-AUC in the comparative benchmark.\n"
        "Model operating threshold: 0.24, selected on validation data; "
        "this is not a universal definition of fraud."
    )
    return result["fraud_probability"], result["interpretation"], summary


def _dropdown(column, value, label):
    return gr.Dropdown(VISIBLE_CHOICES[column], value=value, label=label)


def create_demo():
    if gr is None:
        raise RuntimeError("Gradio is required to launch the deployment app")
    with gr.Blocks(title="Insurance Fraud Detector") as demo:
        gr.Markdown(
            "# 🛡️ Insurance Fraud Detection\n"
            "Fill in all claim fields and click **Assess fraud risk**.\n"
            "This interactive demo currently uses Gradient Boosting, selected "
            "for its F1 and PR-AUC performance in the comparative benchmark.\n"
            "Predictions are risk signals for investigation support, not proof of fraud.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Claim details")
                with gr.Row():
                    month = _dropdown("Month", "Jan", "Month of accident")
                    month_claimed = _dropdown("MonthClaimed", "Jan", "Month claim filed")
                with gr.Row():
                    week_of_month = gr.Slider(1, 5, value=3, step=1,
                                              label="Week of month (accident)")
                    week_of_month_claimed = gr.Slider(1, 5, value=3, step=1,
                                                       label="Week of month (claim)")
                with gr.Row():
                    day_of_week = _dropdown("DayOfWeek", "Monday", "Day of accident")
                    day_of_week_claimed = _dropdown(
                        "DayOfWeekClaimed", "Monday", "Day claim filed")

                gr.Markdown("#### Vehicle")
                with gr.Row():
                    make = _dropdown("Make", "Honda", "Make")
                    vehicle_category = _dropdown("VehicleCategory", "Sport",
                                                 "Vehicle category")
                with gr.Row():
                    vehicle_price = _dropdown("VehiclePrice", "more than 69000",
                                              "Vehicle price")
                    age_of_vehicle = _dropdown("AgeOfVehicle", "3 years",
                                               "Age of vehicle")

                gr.Markdown("#### Policy & driver")
                with gr.Row():
                    policy_type = _dropdown("PolicyType", "Sport - Collision",
                                            "Policy type")
                    base_policy = _dropdown("BasePolicy", "Collision", "Base policy")
                with gr.Row():
                    fault = _dropdown("Fault", "Policy Holder", "Fault")
                    agent_type = _dropdown("AgentType", "External", "Agent type")
                with gr.Row():
                    deductible = gr.Slider(300, 700, value=400, step=100,
                                           label="Deductible ($)")
                    driver_rating = gr.Slider(
                        1, 4, value=2, step=1,
                        label="Driver rating (1=poor, 4=excellent)")

                gr.Markdown("#### Policyholder")
                with gr.Row():
                    sex = _dropdown("Sex", "Male", "Sex")
                    marital_status = _dropdown("MaritalStatus", "Single",
                                               "Marital status")
                with gr.Row():
                    age = gr.Slider(16, 80, value=35, step=1, label="Age")

                gr.Markdown("#### Incident details")
                with gr.Row():
                    accident_area = _dropdown("AccidentArea", "Urban",
                                               "Accident area")
                    num_cars = _dropdown("NumberOfCars", "1 vehicle",
                                         "Number of cars")
                with gr.Row():
                    police_report = _dropdown("PoliceReportFiled", "No",
                                              "Police report filed")
                    witness_present = _dropdown("WitnessPresent", "No",
                                                "Witness present")
                with gr.Row():
                    days_policy_accident = _dropdown(
                        "Days_Policy_Accident", "more than 30",
                        "Days: policy to accident")
                    days_policy_claim = _dropdown(
                        "Days_Policy_Claim", "more than 30",
                        "Days: policy to claim")
                with gr.Row():
                    past_claims = _dropdown("PastNumberOfClaims", "none",
                                            "Past claims")
                    num_supplements = _dropdown("NumberOfSuppliments", "none",
                                                "Supplements")
                address_change = _dropdown(
                    "AddressChange_Claim", "no change",
                    "Address change before claim")
                submit_btn = gr.Button("Assess fraud risk",
                                       variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Assessment result")
                prob_output = gr.Number(
                    label="Fraud probability (0.0 - 1.0)", precision=3)
                tier_output = gr.Textbox(label="Risk tier & decision", lines=1)
                detail_output = gr.Textbox(
                    label="Full assessment", lines=18)
                gr.Markdown(
                    "---\n"
                    "#### How to read this\n\n"
                    "| Probability | Presentation band | Meaning |\n"
                    "|---|---|---|\n"
                    "| `< 0.24` | Lower Fraud Risk | Below operating threshold |\n"
                    "| `0.24 - < 0.50` | Elevated Fraud Risk | Review details |\n"
                    "| `>= 0.50` | Higher Fraud Risk | Review recommended |\n\n"
                    "Model operating threshold: **0.24**, selected on validation "
                    "data. It controls the review flag and is not a universal "
                    "definition of fraud. The presentation bands are UX guidance.\n\n"
                    "This deployment uses Gradient Boosting for its F1 and PR-AUC "
                    "performance. Random Forest leads ROC-AUC, Decision Tree leads "
                    "recall, and ANN v3 is the strongest deep-learning model.\n\n"
                    "Predictions support investigation and do not establish fraud.")

        all_inputs = [
            month, week_of_month, day_of_week, make, accident_area,
            day_of_week_claimed, month_claimed, week_of_month_claimed,
            sex, marital_status, age, fault, policy_type, vehicle_category,
            vehicle_price, deductible, driver_rating, days_policy_accident,
            days_policy_claim, past_claims, age_of_vehicle,
            police_report, witness_present, agent_type, num_supplements,
            address_change, num_cars, base_policy]
        submit_btn.click(
            fn=predict_fraud, inputs=all_inputs,
            outputs=[prob_output, tier_output, detail_output])
    return demo


if __name__ == "__main__":
    create_demo().launch(theme=gr.themes.Soft())
