"""Smoke tests for the proposed local Hugging Face deployment bundle."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deployment", "huggingface"))

import app
from insurance_fraud_detection.data.preprocessor import FEATURE_COLUMNS as CANONICAL_FEATURE_COLUMNS


def _synthetic_claim():
    values = {}
    for column in app.FEATURE_COLUMNS:
        if column in app._bundle["numeric_columns"]:
            values[column] = 35 if column == "Age" else 1
        else:
            values[column] = "unseen-category"
    values["DayOfWeekClaimed"] = "0"
    values["MonthClaimed"] = "0"
    return values


def test_deployment_artifacts_and_schema_load():
    assert len(app.FEATURE_COLUMNS) == 29
    assert app.FEATURE_COLUMNS == CANONICAL_FEATURE_COLUMNS
    assert app.TARGET_COLUMN == "FraudFound_P"
    assert app.THRESHOLD == 0.24
    assert app._model.__class__.__name__ == "GradientBoostingClassifier"
    assert "Porsche" in app.VISIBLE_CHOICES["Make"]
    assert "Porche" not in app.VISIBLE_CHOICES["Make"]
    assert "18 to 20" not in app.VISIBLE_CHOICES["AgeOfPolicyHolder"]
    assert "Unknown" not in app.VISIBLE_CHOICES["DayOfWeekClaimed"]
    assert "Unknown" not in app.VISIBLE_CHOICES["MonthClaimed"]
    assert app.VISIBLE_CHOICES["Month"] == [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert app.VISIBLE_CHOICES["DayOfWeek"] == [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]


def test_prediction_handles_unknown_categories_without_dataset_dependency():
    result = app.predict_claim(_synthetic_claim())
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_flag"] == (result["fraud_probability"] >= 0.24)
    assert "does not establish that a claim is fraudulent" in result["disclaimer"]


def test_age_derives_canonical_policyholder_band_at_boundaries():
    expected = {
        16: "18 to 20", 17: "18 to 20", 18: "21 to 25",
        20: "21 to 25", 21: "26 to 30", 25: "26 to 30",
        26: "31 to 35", 35: "31 to 35", 36: "36 to 40",
        45: "36 to 40", 46: "41 to 50", 55: "41 to 50",
        56: "51 to 65", 65: "51 to 65", 66: "over 65", 80: "over 65",
    }
    for age, category in expected.items():
        assert app._derive_age_of_policy_holder(age) == category
    assert app._derive_age_of_policy_holder(0) == "16 to 17"


def test_age_only_visible_input_still_populates_all_canonical_features():
    values = _synthetic_claim()
    values.pop("AgeOfPolicyHolder")
    frame = app._frame_from_input(values)
    assert list(frame.columns) == app.FEATURE_COLUMNS
    assert frame.loc[0, "AgeOfPolicyHolder"] == "31 to 35"


def test_prediction_rejects_invalid_numeric_input():
    values = _synthetic_claim()
    values["Age"] = "not-a-number"
    with pytest.raises(ValueError, match="Age must be numeric"):
        app.predict_claim(values)
