"""Generates narrative risk reports using an LLM."""

from loguru import logger

from app.api.schemas import FeatureContribution, LoanApplication, RiskAssessment
from app.services.labels import FEATURE_META, humanize_value
from app.services.llm_client import LLMClient

SYSTEM_PROMPT = """You are a professional loan underwriting assistant. Your role is to
generate clear, concise risk assessment reports for loan officers.

Rules:
- Base your report ONLY on the data provided below. Do not invent information.
- Do NOT make the final lending decision. That is the loan officer's job.
- Be specific: mention exact probability, feature values, and SHAP contributions.
- Use professional financial language but keep it readable.
- Highlight the top 3 risk factors (both positive and negative).
- End with a recommendation for the loan officer to review.
- Use only the plain-language factor names and values provided below.
  Never output internal codes such as PAY_0, PAY_2, LIMIT_BAL, or BILL_AMT1."""

USER_TEMPLATE = """LOAN APPLICATION RISK ASSESSMENT
================================

Prediction: {risk} Risk
Default Probability: {probability:.1%}

APPLICANT DATA
--------------
{applicant_data}

FEATURE CONTRIBUTIONS (SHAP)
----------------------------
{shap_explanations}

INSTRUCTIONS
------------
Generate a professional risk assessment report for the loan officer.
Structure it as:
1. Summary — one sentence stating the risk level and probability
2. Key Risk Factors — the top factors driving this prediction (refer to SHAP values)
3. Positive Factors — any factors that reduce risk
4. Recommendation — what the loan officer should review before making a decision"""


# Maps Pydantic request fields to the model feature names, so applicant data
# can be formatted with the same human-readable labels as SHAP explanations.
_APPLICATION_TO_FEATURE = {
    "limit_bal": "LIMIT_BAL",
    "age": "AGE",
    "pay_0": "PAY_0",
    "pay_2": "PAY_2",
    "pay_3": "PAY_3",
    "pay_amt1": "PAY_AMT1",
    "bill_amt1": "BILL_AMT1",
}


def _format_applicant_data(application: LoanApplication) -> str:
    """Format applicant data as a readable table for the LLM."""
    lines = []
    for field, feature in _APPLICATION_TO_FEATURE.items():
        value = getattr(application, field)
        label = FEATURE_META[feature]["label"]
        lines.append(f"  {label}: {humanize_value(feature, value)}")
    return "\n".join(lines)


def _format_shap(
    explanations: list[FeatureContribution],
) -> str:
    """Format SHAP explanations as a readable table for the LLM."""
    lines = []
    for e in explanations:
        sign = "+" if e.shap_value >= 0 else ""
        lines.append(
            f"  {e.feature_label} ({e.value_label}): "
            f"SHAP {sign}{e.shap_value:.4f} ({e.magnitude})"
        )
    return "\n".join(lines)


class ReportGenerator:
    """Generates narrative risk reports from model predictions."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system_prompt = SYSTEM_PROMPT
        self._user_template = USER_TEMPLATE
        logger.info("Report generator initialized")

    def generate(
        self,
        application: LoanApplication,
        assessment: RiskAssessment,
    ) -> str:
        """Generate a narrative risk report for a loan officer.

        Args:
            application: The original applicant data.
            assessment: The model's prediction and SHAP explanations.

        Returns:
            A narrative risk report in plain text.

        Raises:
            RuntimeError: If no LLM provider is available.
        """
        applicant_data = _format_applicant_data(application)
        shap_text = _format_shap(assessment.shap_explanations)

        user_prompt = self._user_template.format(
            risk=assessment.risk,
            probability=assessment.default_probability,
            applicant_data=applicant_data,
            shap_explanations=shap_text,
        )

        return self._llm.generate(self._system_prompt, user_prompt)
