"""Tests for the underwriting pipeline."""

import pytest
from app.api.schemas import FeatureContribution, LoanApplication, RiskAssessment
from app.services.predictor import Predictor
from app.services.report_generator import _format_applicant_data, _format_shap

MODEL_PATH = "models/model.pkl"


def test_predictor_loads():
    """Predictor should load without errors."""
    predictor = Predictor(MODEL_PATH)
    assert predictor.features == [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_AMT1",
        "BILL_AMT1",
    ]


def test_low_risk_prediction():
    """On-time payer with high credit limit should be low risk."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=200000,
        age=35,
        pay_0=-1,
        pay_2=-1,
        pay_3=-1,
        pay_amt1=5000,
        bill_amt1=30000,
    )
    result = predictor.predict(app)
    assert result.risk == "Low"
    assert result.default_probability < 0.5


def test_high_risk_prediction():
    """Delayed payer with low credit should be high risk."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=50000,
        age=25,
        pay_0=3,
        pay_2=2,
        pay_3=2,
        pay_amt1=500,
        bill_amt1=50000,
    )
    result = predictor.predict(app)
    assert result.risk == "High"
    assert result.default_probability > 0.5


def test_shap_explanations_are_present():
    """RiskAssessment should include SHAP explanations for all features."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=100000,
        age=40,
        pay_0=-1,
        pay_2=-1,
        pay_3=-1,
        pay_amt1=3000,
        bill_amt1=20000,
    )
    result = predictor.predict(app)
    assert len(result.shap_explanations) == 7


def test_shap_explanations_sorted_by_importance():
    """SHAP explanations should be sorted by absolute value descending."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=100000,
        age=40,
        pay_0=-1,
        pay_2=-1,
        pay_3=-1,
        pay_amt1=3000,
        bill_amt1=20000,
    )
    result = predictor.predict(app)
    values = [abs(e.shap_value) for e in result.shap_explanations]
    assert values == sorted(values, reverse=True)


def test_shap_impact_direction():
    """SHAP impact should correctly reflect positive/negative values."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=100000,
        age=40,
        pay_0=-1,
        pay_2=-1,
        pay_3=-1,
        pay_amt1=3000,
        bill_amt1=20000,
    )
    result = predictor.predict(app)
    for e in result.shap_explanations:
        if e.shap_value >= 0:
            assert e.impact == "increases_risk"
        else:
            assert e.impact == "decreases_risk"


def test_risk_assessment_has_all_fields():
    """RiskAssessment response should include all required fields."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=100000,
        age=30,
        pay_0=-1,
        pay_2=-1,
        pay_3=-1,
        pay_amt1=2000,
        bill_amt1=15000,
    )
    result = predictor.predict(app)
    assert result.risk in ("Low", "High")
    assert 0.0 <= result.default_probability <= 1.0
    assert 0.0 <= result.baseline_probability <= 1.0
    assert len(result.features_used) == 7
    assert result.risk_report == ""  # LLM not available in tests


def test_shap_explanations_are_human_readable():
    """SHAP explanations should include human-readable labels."""
    predictor = Predictor(MODEL_PATH)
    app = LoanApplication(
        limit_bal=100000,
        age=40,
        pay_0=-1,
        pay_2=-1,
        pay_3=-1,
        pay_amt1=3000,
        bill_amt1=20000,
    )
    result = predictor.predict(app)
    by_name = {e.feature_name: e for e in result.shap_explanations}

    pay_0 = by_name["PAY_0"]
    assert pay_0.feature_label == "Repayment Status (Last Month)"
    assert pay_0.value_label == "Paid in full"

    limit = by_name["LIMIT_BAL"]
    assert limit.feature_label == "Credit Limit"
    assert limit.value_label == "100,000"

    age = by_name["AGE"]
    assert age.value_label == "40 years"

    for e in result.shap_explanations:
        assert e.feature_label
        assert e.value_label
        assert "risk" in e.magnitude


def test_pay_exceeds_bill_rejected():
    """Pay amount exceeding bill amount should raise validation error."""
    with pytest.raises(ValueError, match="pay_amt1.*exceeds bill amount"):
        LoanApplication(
            limit_bal=100000, age=30, pay_0=-1, pay_2=-1, pay_3=-1,
            pay_amt1=50000, bill_amt1=30000,
        )


def test_bill_exceeds_limit_rejected():
    """Bill amount exceeding credit limit should raise validation error."""
    with pytest.raises(ValueError, match="bill_amt1.*exceeds credit limit"):
        LoanApplication(
            limit_bal=10000, age=30, pay_0=-1, pay_2=-1, pay_3=-1,
            pay_amt1=500, bill_amt1=50000,
        )


_RAW_FEATURE_CODES = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "LIMIT_BAL",
    "PAY_AMT1",
    "BILL_AMT1",
]


def test_applicant_data_is_human_readable():
    """Applicant data fed to the LLM must use labels, never raw codes."""
    app = LoanApplication(
        limit_bal=200000,
        age=35,
        pay_0=-1,
        pay_2=-1,
        pay_3=0,
        pay_amt1=5000,
        bill_amt1=30000,
    )
    text = _format_applicant_data(app)

    assert "Repayment Status (Last Month): Paid in full" in text
    assert "Credit Limit: 200,000" in text
    assert "Age: 35 years" in text
    for code in _RAW_FEATURE_CODES:
        assert code not in text


def test_shap_prompt_is_human_readable():
    """SHAP text fed to the LLM must use labels, never raw codes."""
    explanations = [
        FeatureContribution(
            feature_name="PAY_0",
            feature_label="Repayment Status (Last Month)",
            feature_value=-1,
            value_label="Paid in full",
            shap_value=-0.4123,
            impact="decreases_risk",
            magnitude="Very strongly decreases risk",
        ),
        FeatureContribution(
            feature_name="LIMIT_BAL",
            feature_label="Credit Limit",
            feature_value=200000,
            value_label="200,000",
            shap_value=0.15,
            impact="increases_risk",
            magnitude="Moderately increases risk",
        ),
    ]
    text = _format_shap(explanations)

    assert "Repayment Status (Last Month) (Paid in full)" in text
    assert "SHAP -0.4123 (Very strongly decreases risk)" in text
    assert "Credit Limit (200,000)" in text
    for code in _RAW_FEATURE_CODES:
        assert code not in text