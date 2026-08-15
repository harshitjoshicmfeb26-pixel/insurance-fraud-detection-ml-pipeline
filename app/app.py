import gradio as gr
import numpy as np
import joblib
import tensorflow as tf


# ── Load model and scaler ─────────────────────────────────────
model = tf.keras.models.load_model(
    "fraud_detector_final.keras",
    compile=False
)
model.compile(optimizer="adam", loss="binary_crossentropy")
scaler = joblib.load("scaler.joblib")
print("Model loaded successfully")


# ── Encoding maps — verified against Cell 4 LabelEncoder ─────
# Rule: LabelEncoder sorts alphabetically then assigns 0,1,2...


MONTH_MAP = {
    "Apr":0,"Aug":1,"Dec":2,"Feb":3,"Jan":4,
    "Jul":5,"Jun":6,"Mar":7,"May":8,"Nov":9,"Oct":10,"Sep":11
}


DAY_MAP = {
    "Friday":0,"Monday":1,"Saturday":2,
    "Sunday":3,"Thursday":4,"Tuesday":5,"Wednesday":6
}


# 19 makes — sorted alphabetically by LabelEncoder
MAKE_MAP = {
    "Accura":0,"BMW":1,"Chevrolet":2,"Dodge":3,"Ferrari":4,
    "Ford":5,"Honda":6,"Jaguar":7,"Lexus":8,"Mazda":9,
    "Mecedes":10,"Mercury":11,"Nisson":12,"Pontiac":13,
    "Porsche":14,"Saab":15,"Saturn":16,"Toyota":17,"VW":18
}


# DayOfWeekClaimed has 8 values including Unknown (was '0' in data)
DOWC_MAP = {
    "Friday":0,"Monday":1,"Saturday":2,"Sunday":3,
    "Thursday":4,"Tuesday":5,"Unknown":6,"Wednesday":7
}


AREA_MAP    = {"Rural":0, "Urban":1}
SEX_MAP     = {"Female":0, "Male":1}
MARITAL_MAP = {"Divorced":0,"Married":1,"Single":2,"Widow":3}
FAULT_MAP   = {"Policy Holder":0,"Third Party":1}
YESNO_MAP   = {"No":0, "Yes":1}
AGENT_MAP   = {"External":0,"Internal":1}


# PolicyType — 9 values sorted alphabetically
POLICY_MAP = {
    "Sedan - All Perils":0,"Sedan - Collision":1,"Sedan - Liability":2,
    "Sport - All Perils":3,"Sport - Collision":4,"Sport - Liability":5,
    "Utility - All Perils":6,"Utility - Collision":7,"Utility - Liability":8
}


VCAT_MAP  = {"Sedan":0,"Sport":1,"Utility":2}
BASE_MAP  = {"All Perils":0,"Collision":1,"Liability":2}


# Ordinal maps — from Cell 4 ORDINAL_MAPS
VPRICE_MAP = {
    "less than 20000":1,"20000 to 29000":2,"30000 to 39000":3,
    "40000 to 59000":4,"60000 to 69000":5,"more than 69000":6
}
DPA_MAP = {
    "none":0,"1 to 7":1,"8 to 15":2,"15 to 30":3,"more than 30":4
}
DPC_MAP = {
    "none":0,"8 to 15":1,"15 to 30":2,"more than 30":3
}
PAST_MAP = {
    "none":0,"1":1,"2 to 4":2,"more than 4":3
}
AVEH_MAP = {
    "new":0,"2 years":2,"3 years":3,"4 years":4,
    "5 years":5,"6 years":6,"7 years":7,"more than 7":8
}
AGEPH_MAP = {
    "16 to 17":1,"21 to 25":3,"26 to 30":4,"31 to 35":5,
    "36 to 40":6,"41 to 50":7,"51 to 65":8,"over 65":9
}
SUPP_MAP = {
    "none":0,"1 to 2":1,"3 to 5":2,"more than 5":3
}
ADDR_MAP = {
    "no change":0,"under 6 months":1,"1 year":2,
    "2 to 3 years":3,"4 to 8 years":4
}
NCARS_MAP = {
    "1 vehicle":1,"2 vehicles":2,"3 to 4":3,
    "5 to 8":4,"more than 8":5
}




def predict_fraud(
    month, week_of_month, day_of_week, make, accident_area,
    day_of_week_claimed, month_claimed, week_of_month_claimed,
    sex, marital_status, age, fault, policy_type,
    vehicle_category, vehicle_price, deductible, driver_rating,
    days_policy_accident, days_policy_claim, past_claims,
    age_of_vehicle, age_of_policy_holder, police_report,
    witness_present, agent_type, num_supplements,
    address_change, num_cars, base_policy
):
    # ── Feature array — ORDER MUST MATCH Cell 5 FEATURE_NAMES ─
    # ['Month','WeekOfMonth','DayOfWeek','Make','AccidentArea',
    #  'DayOfWeekClaimed','MonthClaimed','WeekOfMonthClaimed',
    #  'Sex','MaritalStatus','Age','Fault','PolicyType',
    #  'VehicleCategory','VehiclePrice','Deductible','DriverRating',
    #  'Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims',
    #  'AgeOfVehicle','AgeOfPolicyHolder','PoliceReportFiled',
    #  'WitnessPresent','AgentType','NumberOfSuppliments',
    #  'AddressChange_Claim','NumberOfCars','BasePolicy']


    features = np.array([[
        MONTH_MAP[month],                      # Month
        int(week_of_month),                    # WeekOfMonth
        DAY_MAP[day_of_week],                  # DayOfWeek
        MAKE_MAP.get(make, 6),                 # Make
        AREA_MAP[accident_area],               # AccidentArea
        DOWC_MAP.get(day_of_week_claimed, 1),  # DayOfWeekClaimed
        MONTH_MAP[month_claimed],              # MonthClaimed
        int(week_of_month_claimed),            # WeekOfMonthClaimed
        SEX_MAP[sex],                          # Sex
        MARITAL_MAP[marital_status],           # MaritalStatus
        int(age),                              # Age
        FAULT_MAP[fault],                      # Fault
        POLICY_MAP[policy_type],               # PolicyType
        VCAT_MAP[vehicle_category],            # VehicleCategory
        VPRICE_MAP[vehicle_price],             # VehiclePrice
        int(deductible),                       # Deductible
        int(driver_rating),                    # DriverRating
        DPA_MAP[days_policy_accident],         # Days_Policy_Accident
        DPC_MAP[days_policy_claim],            # Days_Policy_Claim
        PAST_MAP[past_claims],                 # PastNumberOfClaims
        AVEH_MAP[age_of_vehicle],              # AgeOfVehicle
        AGEPH_MAP[age_of_policy_holder],       # AgeOfPolicyHolder
        YESNO_MAP[police_report],              # PoliceReportFiled
        YESNO_MAP[witness_present],            # WitnessPresent
        AGENT_MAP[agent_type],                 # AgentType
        SUPP_MAP[num_supplements],             # NumberOfSuppliments
        ADDR_MAP[address_change],              # AddressChange_Claim
        NCARS_MAP[num_cars],                   # NumberOfCars
        BASE_MAP[base_policy],                 # BasePolicy
    ]], dtype=np.float32)


    X_scaled = scaler.transform(features)
    prob     = float(model.predict(X_scaled, verbose=0)[0][0])
    pct      = round(prob * 100, 1)


    if prob >= 0.60:
        tier   = "🚨 HIGH RISK"
        action = "Flag for investigation immediately"
    elif prob >= 0.40:
        tier   = "⚠️ MEDIUM RISK"
        action = "Request supporting documents"
    else:
        tier   = "✅ LOW RISK"
        action = "Proceed with standard processing"


    factors = []
    factors.append(
        "✅ Police report filed (reduces risk)"
        if police_report == "Yes"
        else "❌ No police report filed (increases risk)"
    )
    factors.append(
        "✅ Witness present (reduces risk)"
        if witness_present == "Yes"
        else "❌ No witness present (increases risk)"
    )
    factors.append(
        "⚠️ Claim filed 30+ days after accident (increases risk)"
        if days_policy_accident == "more than 30"
        else "✅ Claim filed promptly (reduces risk)"
    )
    factors.append(
        "⚠️ Multiple past claims on record (increases risk)"
        if past_claims in ["2 to 4","more than 4"]
        else "✅ Clean claims history (reduces risk)"
    )
    factors.append(
        "⚠️ High number of supplements (increases risk)"
        if num_supplements in ["3 to 5","more than 5"]
        else "✅ Normal supplement count (reduces risk)"
    )
    factors.append(
        "⚠️ Recent address change before claim (increases risk)"
        if address_change in ["under 6 months","1 year"]
        else "✅ Stable address (reduces risk)"
    )
    factors.append(
        "⚠️ Poor driver rating (increases risk)"
        if int(driver_rating) == 1
        else "✅ Good driver rating (reduces risk)"
    )
    factors.append(
        "⚠️ Third party fault claim (check carefully)"
        if fault == "Third Party"
        else "ℹ️ Policy holder at fault"
    )


    summary = (
        f"Fraud Probability : {pct}%\n"
        f"Decision          : {tier}\n"
        f"Action            : {action}\n\n"
        f"Risk Factors:\n"
        + "\n".join(f"  {f}" for f in factors)
        + f"\n\n─────────────────────────────\n"
        f"Model  : Legacy notebook-derived ANN\n"
        f"Deployment: Separate Hugging Face demo\n"
        f"Risk tiers: <0.40 low, 0.40–0.59 medium, >=0.60 high"
    )


    return pct / 100, tier, summary




# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Insurance Fraud Detector",
               theme=gr.themes.Soft()) as demo:


    gr.Markdown("""
    # 🛡️ Insurance Fraud Detection
    **MiniProject-AI **

    Fill in all claim fields and click **Assess fraud risk**.
    The model outputs a fraud probability using the legacy notebook-derived ANN
    deployed separately on Hugging Face. This demo is not the local benchmark winner.
    """)


    with gr.Row():


        with gr.Column(scale=1):
            gr.Markdown("### Claim details")


            with gr.Row():
                month = gr.Dropdown(
                    sorted(MONTH_MAP.keys()),
                    value="Jan", label="Month of accident")
                month_claimed = gr.Dropdown(
                    sorted(MONTH_MAP.keys()),
                    value="Jan", label="Month claim filed")


            with gr.Row():
                week_of_month = gr.Slider(
                    1, 5, value=3, step=1,
                    label="Week of month (accident)")
                week_of_month_claimed = gr.Slider(
                    1, 5, value=3, step=1,
                    label="Week of month (claim)")


            with gr.Row():
                day_of_week = gr.Dropdown(
                    sorted(DAY_MAP.keys()),
                    value="Monday", label="Day of accident")
                day_of_week_claimed = gr.Dropdown(
                    sorted(DOWC_MAP.keys()),
                    value="Monday", label="Day claim filed")


            gr.Markdown("#### Vehicle")
            with gr.Row():
                make = gr.Dropdown(
                    sorted(MAKE_MAP.keys()),
                    value="Honda", label="Make")
                vehicle_category = gr.Dropdown(
                    sorted(VCAT_MAP.keys()),
                    value="Sport", label="Category")


            with gr.Row():
                vehicle_price = gr.Dropdown(
                    list(VPRICE_MAP.keys()),
                    value="more than 69000",
                    label="Vehicle price")
                age_of_vehicle = gr.Dropdown(
                    list(AVEH_MAP.keys()),
                    value="3 years", label="Age of vehicle")


            gr.Markdown("#### Policy & driver")
            with gr.Row():
                policy_type = gr.Dropdown(
                    sorted(POLICY_MAP.keys()),
                    value="Sport - Collision",
                    label="Policy type")
                base_policy = gr.Dropdown(
                    sorted(BASE_MAP.keys()),
                    value="Collision", label="Base policy")


            with gr.Row():
                fault = gr.Dropdown(
                    list(FAULT_MAP.keys()),
                    value="Policy Holder", label="Fault")
                agent_type = gr.Dropdown(
                    sorted(AGENT_MAP.keys()),
                    value="External", label="Agent type")


            with gr.Row():
                deductible = gr.Slider(
                    300, 700, value=400, step=100,
                    label="Deductible ($)")
                driver_rating = gr.Slider(
                    1, 4, value=2, step=1,
                    label="Driver rating (1=poor, 4=excellent)")


            gr.Markdown("#### Policyholder")
            with gr.Row():
                sex = gr.Dropdown(
                    sorted(SEX_MAP.keys()),
                    value="Male", label="Sex")
                marital_status = gr.Dropdown(
                    sorted(MARITAL_MAP.keys()),
                    value="Single", label="Marital status")


            with gr.Row():
                age = gr.Slider(
                    16, 80, value=35, step=1,
                    label="Age")
                age_of_policy_holder = gr.Dropdown(
                    list(AGEPH_MAP.keys()),
                    value="31 to 35", label="Age group")


            gr.Markdown("#### Incident details")
            with gr.Row():
                accident_area = gr.Dropdown(
                    sorted(AREA_MAP.keys()),
                    value="Urban", label="Accident area")
                num_cars = gr.Dropdown(
                    list(NCARS_MAP.keys()),
                    value="1 vehicle", label="Number of cars")


            with gr.Row():
                police_report = gr.Dropdown(
                    ["No", "Yes"],
                    value="No", label="Police report filed")
                witness_present = gr.Dropdown(
                    ["No", "Yes"],
                    value="No", label="Witness present")


            with gr.Row():
                days_policy_accident = gr.Dropdown(
                    list(DPA_MAP.keys()),
                    value="more than 30",
                    label="Days: policy to accident")
                days_policy_claim = gr.Dropdown(
                    list(DPC_MAP.keys()),
                    value="more than 30",
                    label="Days: policy to claim")


            with gr.Row():
                past_claims = gr.Dropdown(
                    list(PAST_MAP.keys()),
                    value="none", label="Past claims")
                num_supplements = gr.Dropdown(
                    list(SUPP_MAP.keys()),
                    value="none", label="Supplements")


            address_change = gr.Dropdown(
                list(ADDR_MAP.keys()),
                value="no change",
                label="Address change before claim")


            submit_btn = gr.Button(
                "🔍  Assess fraud risk",
                variant="primary", size="lg")


        with gr.Column(scale=1):
            gr.Markdown("### Assessment result")


            prob_output = gr.Number(
                label="Fraud probability (0.0 – 1.0)",
                precision=3)
            tier_output = gr.Textbox(
                label="Risk tier & decision", lines=1)
            detail_output = gr.Textbox(
                label="Full assessment", lines=18)


            gr.Markdown("""
---
#### How to read this
| Probability | Tier | Action |
|---|---|---|
| ≥ 0.60 | 🚨 High risk | Assign to investigator |
| 0.40 – 0.59 | ⚠️ Medium risk | Request documents |
| < 0.40 | ✅ Low risk | Standard processing |

#### Why ANN beats ML here
Your benchmark (Cell 14) showed:
- Gradient Boosting: **94% accuracy, 7 frauds caught**
- Decision Tree: **81% accuracy, 104 frauds caught**
- **This ANN: 68% accuracy, 148 frauds caught**

High accuracy ≠ good fraud detector.
The ANN was trained to catch fraud, not chase accuracy.
            """)


    all_inputs = [
        month, week_of_month, day_of_week, make, accident_area,
        day_of_week_claimed, month_claimed, week_of_month_claimed,
        sex, marital_status, age, fault, policy_type,
        vehicle_category, vehicle_price, deductible, driver_rating,
        days_policy_accident, days_policy_claim, past_claims,
        age_of_vehicle, age_of_policy_holder, police_report,
        witness_present, agent_type, num_supplements,
        address_change, num_cars, base_policy
    ]


    submit_btn.click(
        fn=predict_fraud,
        inputs=all_inputs,
        outputs=[prob_output, tier_output, detail_output]
    )


if __name__ == "__main__":
    demo.launch()
