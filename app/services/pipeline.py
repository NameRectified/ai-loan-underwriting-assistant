"""Orchestrates the full underwriting pipeline."""

from typing import Optional

from loguru import logger

from app.api.schemas import LoanApplication, RiskAssessment
from app.database.repository import save_application
from app.services.predictor import Predictor
from app.services.report_generator import ReportGenerator


class UnderwritingPipeline:
    """Coordinates the end-to-end underwriting flow.

    For a single loan application, this pipeline:
    1. Runs the ML model to get a risk prediction + SHAP explanations.
    2. Generates a narrative risk report via LLM (if configured).
    3. Persists the application and result for audit trail.
    """

    def __init__(
        self,
        predictor: Predictor,
        report_generator: Optional[ReportGenerator] = None,
    ) -> None:
        self._predictor = predictor
        self._report_generator = report_generator

    def process(self, application: LoanApplication) -> RiskAssessment:
        """Run the full underwriting pipeline on a single application.

        Args:
            application: Validated applicant data.

        Returns:
            Complete risk assessment with prediction, SHAP explanations,
            and optionally an LLM-generated risk report.
        """
        assessment = self._predictor.predict(application)

        if self._report_generator is not None:
            try:
                report = self._report_generator.generate(application, assessment)
                assessment.risk_report = report
            except Exception as exc:
                logger.warning(f"Report generation failed: {exc}")
                assessment.risk_report = ""

        try:
            save_application(
                application=application.model_dump(),
                assessment=assessment.model_dump(),
            )
        except Exception as exc:
            logger.warning(f"Persistence failed: {exc}")

        return assessment