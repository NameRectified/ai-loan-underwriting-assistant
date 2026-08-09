"""Generates narrative risk reports using an LLM."""

from loguru import logger
import yaml

from app.api.schemas import FeatureContribution, LoanApplication, RiskAssessment
from app.services.feature_meta import FEATURE_META, humanize_value
from app.services.llm_client import LLMClient

PROMPT_PATH = "prompts/report.yaml"


def _load_prompt(path: str) -> tuple[str, str]:
    """Load the system prompt and user prompt template from a YAML file.

    Args:
        path: Path to the YAML prompt file.

    Returns:
        (system_prompt, user_prompt_template)
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    prompts = data["report_generation"]
    return prompts["system_prompt"], prompts["user_prompt_template"]


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

    def __init__(
        self, llm_client: LLMClient, prompt_path: str = PROMPT_PATH
    ) -> None:
        self._llm = llm_client
        self._system_prompt, self._user_template = _load_prompt(prompt_path)
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