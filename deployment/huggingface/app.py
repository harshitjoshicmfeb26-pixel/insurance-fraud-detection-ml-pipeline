"""Gradio interface for insurance fraud investigation decision support."""
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
if (APP_DIR / "insurance_fraud_detection").exists():
    sys.path.insert(0, str(APP_DIR))
else:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insurance_fraud_detection.business.batch import score_claims
from insurance_fraud_detection.business.decision import assess_probability
from insurance_fraud_detection.business.inference import transform_with_bundle

try:
    import gradio as gr
except ImportError:
    gr = None

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
    return transform_with_bundle(frame, _bundle)


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
    decision = assess_probability(probability)
    return decision.risk_band, decision.recommended_action


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
    decision = assess_probability(probability)
    return {
        "fraud_probability": probability,
        "threshold": THRESHOLD,
        "operating_threshold": decision.operating_threshold,
        "risk_flag": decision.above_operating_threshold,
        "above_operating_threshold": decision.above_operating_threshold,
        "interpretation": decision.risk_band,
        "risk_band": decision.risk_band,
        "review_priority": decision.review_priority,
        "action": decision.recommended_action,
        "recommended_action": decision.recommended_action,
        "business_interpretation": decision.business_interpretation,
        "context_cues": _context_cues(frame.iloc[0].to_dict()),
        "disclaimer": decision.disclaimer,
    }


def score_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Score a claim portfolio with the same deployed model and bundle."""
    return score_claims(frame, _model, _bundle)


def score_uploaded_claims(file_path):
    """Gradio callback for a display-only investigation queue."""
    if not file_path:
        return pd.DataFrame(), "Upload a CSV containing the 29 claim fields."
    path = getattr(file_path, "name", file_path)
    try:
        scored = score_batch(pd.read_csv(path))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return pd.DataFrame(), f"Input error: {exc}"
    above = int(scored["above_operating_threshold"].sum())
    summary = (f"{len(scored)} claims scored. {above} claims are at or above "
               f"the validated operating threshold of {THRESHOLD:.2f}. "
               "Rows are ranked by model score descending; this is an advisory "
               "queue for human investigation.")
    return scored, summary


def _primary_queue_view(scored):
    """Return the concise operational view while retaining full scored output."""
    return scored.rename(columns={
        "investigation_rank": "Rank",
        "claim_id": "Claim Reference",
        "fraud_probability": "Fraud Probability",
        "risk_band": "Risk Band",
        "review_priority": "Review Priority",
        "recommended_action": "Recommended Action",
    })[[
        "Rank", "Claim Reference", "Fraud Probability", "Risk Band",
        "Review Priority", "Recommended Action",
    ]]


def score_uploaded_claims_with_detail(file_path):
    """Return the concise queue, full scored detail, and operational summary."""
    scored, summary = score_uploaded_claims(file_path)
    if scored.empty:
        return scored, scored, summary
    return _primary_queue_view(scored), scored, summary


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
    summary = _assessment_summary(result)
    return result["fraud_probability"], result["interpretation"], summary


def _assessment_summary(result):
    cues = "\n".join(f"- {cue}" for cue in result["context_cues"])
    return (
        f"Model Fraud Risk Score : {result['fraud_probability']:.1%}\n"
        f"Risk interpretation    : {result['risk_band']}\n"
        f"Review priority        : {result['review_priority']}\n"
        f"Recommended action     : {result['recommended_action']}\n"
        f"Above threshold (0.24): {result['above_operating_threshold']}\n\n"
        f"Business interpretation: {result['business_interpretation']}\n\n"
        "Contextual review cues:\n"
        f"{cues}\n\n"
        f"{result['disclaimer']}\n\n"
        "Deployed model: Gradient Boosting. Scores are interpreted relative "
        "to the validated operating threshold of 0.24."
    )


def predict_fraud_display(*values):
    """Presentation adapter exposing decision fields without changing scoring."""
    result = predict_claim(values)
    return (
        result["fraud_probability"],
        result["risk_band"],
        result["review_priority"],
        result["recommended_action"],
        _assessment_summary(result),
    )


def _dropdown(column, value, label):
    return gr.Dropdown(VISIBLE_CHOICES[column], value=value, label=label)


PRODUCT_TITLE = "Insurance Fraud Investigation Decision Support"
APP_CSS = """
.gradio-container { max-width: 1440px !important; }
#app-header { border-bottom: 1px solid #d9e2ec; margin-bottom: 1.25rem; }
#app-header h1 { margin-bottom: 0.35rem; color: #17324d; }
#app-header p { color: #526477; max-width: 930px; }
#decision-summary { border: 1px solid #d9e2ec; border-radius: 10px; padding: 1rem; }
#decision-summary textarea, #decision-summary input { font-weight: 600; }
#queue-table th { white-space: normal !important; }
"""


def create_demo():
    if gr is None:
        raise RuntimeError("Gradio is required to launch the deployment app")
    with gr.Blocks(title=PRODUCT_TITLE) as demo:
        gr.Markdown(
            f"# {PRODUCT_TITLE}\n"
            "Assess individual claims or prioritize a batch of claims for review. "
            "Outputs are decision-support signals for trained insurance teams and "
            "do not establish fraud.", elem_id="app-header")

        with gr.Tabs():
            with gr.Tab("Claim Assessment"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        gr.Markdown("### Claim information")
                        with gr.Row():
                            month = _dropdown("Month", "Jan", "Month of accident")
                            month_claimed = _dropdown(
                                "MonthClaimed", "Jan", "Month claim filed")
                        with gr.Row():
                            week_of_month = gr.Slider(
                                1, 5, value=3, step=1, label="Accident week (1–5)")
                            week_of_month_claimed = gr.Slider(
                                1, 5, value=3, step=1, label="Claim week (1–5)")
                        with gr.Row():
                            day_of_week = _dropdown(
                                "DayOfWeek", "Monday", "Day of accident")
                            day_of_week_claimed = _dropdown(
                                "DayOfWeekClaimed", "Monday", "Day claim filed")

                        gr.Markdown("#### Vehicle")
                        with gr.Row():
                            make = _dropdown("Make", "Honda", "Make")
                            vehicle_category = _dropdown(
                                "VehicleCategory", "Sport", "Vehicle category")
                        with gr.Row():
                            vehicle_price = _dropdown(
                                "VehiclePrice", "more than 69000", "Vehicle price")
                            age_of_vehicle = _dropdown(
                                "AgeOfVehicle", "3 years", "Age of vehicle")

                        gr.Markdown("#### Policy and driver")
                        with gr.Row():
                            policy_type = _dropdown(
                                "PolicyType", "Sport - Collision", "Policy type")
                            base_policy = _dropdown(
                                "BasePolicy", "Collision", "Base policy")
                        with gr.Row():
                            fault = _dropdown("Fault", "Policy Holder", "Fault")
                            agent_type = _dropdown(
                                "AgentType", "External", "Agent type")
                        with gr.Row():
                            deductible = gr.Slider(
                                300, 700, value=400, step=100, label="Deductible ($)")
                            driver_rating = gr.Slider(
                                1, 4, value=2, step=1, label="Driver rating (1–4)")

                        gr.Markdown("#### Policyholder")
                        with gr.Row():
                            sex = _dropdown("Sex", "Male", "Sex")
                            marital_status = _dropdown(
                                "MaritalStatus", "Single", "Marital status")
                        age = gr.Slider(16, 80, value=35, step=1, label="Age")

                        gr.Markdown("#### Incident details")
                        with gr.Row():
                            accident_area = _dropdown(
                                "AccidentArea", "Urban", "Accident area")
                            num_cars = _dropdown(
                                "NumberOfCars", "1 vehicle", "Number of cars")
                        with gr.Row():
                            police_report = _dropdown(
                                "PoliceReportFiled", "No", "Police report filed")
                            witness_present = _dropdown(
                                "WitnessPresent", "No", "Witness present")
                        with gr.Row():
                            days_policy_accident = _dropdown(
                                "Days_Policy_Accident", "more than 30",
                                "Policy-to-accident interval")
                            days_policy_claim = _dropdown(
                                "Days_Policy_Claim", "more than 30",
                                "Policy-to-claim interval")
                        with gr.Row():
                            past_claims = _dropdown(
                                "PastNumberOfClaims", "none", "Past claims")
                            num_supplements = _dropdown(
                                "NumberOfSuppliments", "none", "Supplements")
                        address_change = _dropdown(
                            "AddressChange_Claim", "no change",
                            "Address change before claim")
                        submit_btn = gr.Button(
                            "Assess claim", variant="primary", size="lg")

                    with gr.Column(scale=5, elem_id="decision-summary"):
                        gr.Markdown("### Decision summary")
                        gr.Markdown(
                            "Review the model signal alongside the policy context. "
                            "The validated operating threshold is **0.24**.")
                        prob_output = gr.Number(
                            label="Fraud probability", precision=3, interactive=False)
                        with gr.Row():
                            tier_output = gr.Textbox(
                                label="Risk band", lines=1, interactive=False)
                            priority_output = gr.Textbox(
                                label="Review priority", lines=1, interactive=False)
                        action_output = gr.Textbox(
                            label="Recommended action", lines=1, interactive=False)
                        detail_output = gr.Textbox(
                            label="Assessment details", lines=11, interactive=False)
                        gr.Markdown(
                            "**Decision-support notice**  \n"
                            "The model output is a risk estimation for investigation "
                            "support, not confirmation of fraud.")

                with gr.Accordion("How to interpret the score", open=False):
                    gr.Markdown(
                        "| Probability | Risk band | Operational meaning |\n"
                        "|---|---|---|\n"
                        "| `< 0.24` | Lower Fraud Risk | Standard processing |\n"
                        "| `0.24 - < 0.40` | Elevated Fraud Risk | Review details |\n"
                        "| `0.40 - < 0.50` | High Review Priority | Prioritize review |\n"
                        "| `>= 0.50` | Very High Review Priority | Prioritize investigation |\n\n"
                        "The operating threshold controls the review flag. Presentation "
                        "bands are guidance for prioritization and do not change the "
                        "underlying model score.")
                with gr.Accordion("About the model", open=False):
                    gr.Markdown(
                        "- Deployed model: **Gradient Boosting**\n"
                        f"- Model inputs: **{len(FEATURE_COLUMNS)}**\n"
                        f"- Validated operating threshold: **{THRESHOLD:.2f}**\n"
                        "- Output: model-estimated fraud risk for decision support, "
                        "not proof of fraud")

            with gr.Tab("Investigation Queue"):
                gr.Markdown("### Investigation Queue")
                gr.Markdown(
                    "Claims are ranked from highest to lowest model-estimated fraud "
                    "probability so investigators can review higher-risk claims first.")
                gr.Markdown(
                    "Upload multiple claims using the same predictor schema as the "
                    "model. Each row represents one claim. `FraudFound_P` is not "
                    "required for prediction. `PolicyNumber` may be included as a "
                    "display reference but is not used by the model.")
                with gr.Row():
                    batch_file = gr.File(
                        label="Claims CSV", file_types=[".csv"], scale=3)
                    batch_button = gr.Button(
                        "Score claims", variant="primary", scale=1)
                batch_summary = gr.Markdown()
                batch_table = gr.Dataframe(
                    label="Primary investigation queue", wrap=True, max_height=360,
                    elem_id="queue-table")
                with gr.Accordion("Detailed scored output", open=False):
                    gr.Markdown(
                        "The full scored result is retained here for operational "
                        "review and downstream export.")
                    batch_detail = gr.Dataframe(
                        label="Full scored results", wrap=True, max_height=360)
                batch_button.click(
                    fn=score_uploaded_claims_with_detail,
                    inputs=batch_file,
                    outputs=[batch_table, batch_detail, batch_summary])

        all_inputs = [
            month, week_of_month, day_of_week, make, accident_area,
            day_of_week_claimed, month_claimed, week_of_month_claimed,
            sex, marital_status, age, fault, policy_type, vehicle_category,
            vehicle_price, deductible, driver_rating, days_policy_accident,
            days_policy_claim, past_claims, age_of_vehicle,
            police_report, witness_present, agent_type, num_supplements,
            address_change, num_cars, base_policy]
        submit_btn.click(
            fn=predict_fraud_display,
            inputs=all_inputs,
            outputs=[prob_output, tier_output, priority_output,
                     action_output, detail_output])
    return demo


if __name__ == "__main__":
    create_demo().launch(
        theme=gr.themes.Soft(
            primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
        css=APP_CSS,
    )
