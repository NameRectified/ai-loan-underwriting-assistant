"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, model_validator


class LoanApplication(BaseModel):
    """7 features the XGBoost model was trained on."""

    limit_bal: float = Field(..., description="Total credit limit")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    # Repayment status: -2 = no usage, -1 = paid duly, 0 = revolving, 1+ = months delayed
    pay_0: int = Field(..., ge=-2, le=8, description="Repayment status last month")
    pay_2: int = Field(..., ge=-2, le=8, description="Repayment status 2 months ago")
    pay_3: int = Field(..., ge=-2, le=8, description="Repayment status 3 months ago")
    pay_amt1: float = Field(..., ge=0, description="Amount paid last month")
    bill_amt1: float = Field(..., ge=0, description="Bill amount last month")

    @model_validator(mode="after")
    def check_credit_limits(self):
        errs = []
        if self.bill_amt1 > self.limit_bal:
            errs.append(f"bill_amt1 ({self.bill_amt1}) exceeds credit limit ({self.limit_bal})")
        if self.pay_amt1 > self.bill_amt1:
            errs.append(f"pay_amt1 ({self.pay_amt1}) exceeds bill amount ({self.bill_amt1})")
        if errs:
            raise ValueError("; ".join(errs))
        return self


class FeatureContribution(BaseModel):
    """Per-feature SHAP explanation (log-odds units vs baseline)."""

    feature_name: str
    feature_label: str
    feature_value: float
    value_label: str
    shap_value: float
    impact: str  # increases_risk | decreases_risk
    magnitude: str  # e.g. "Strongly decreases risk"


class RiskAssessment(BaseModel):
    """Model prediction + SHAP explanations + optional LLM report."""

    risk: str  # Low | High
    default_probability: float = Field(..., ge=0.0, le=1.0)
    baseline_probability: float = Field(..., ge=0.0, le=1.0)
    features_used: list[str]
    shap_explanations: list[FeatureContribution]
    risk_report: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
