"""Model loading and prediction service."""

import joblib
import pandas as pd
import shap
from loguru import logger
from xgboost import XGBClassifier

from app.api.schemas import FeatureContribution, LoanApplication, RiskAssessment


class Predictor:
    """Wraps a trained XGBoost model for loan default prediction.

    Loads the model artifact once at startup and provides a
    ``predict`` method that converts a ``LoanApplication``
    into a ``RiskAssessment`` with SHAP explanations.
    """

    def __init__(self, model_path: str) -> None:
        """Load the model artifact and initialize the SHAP explainer.

        Args:
            model_path: Path to the ``.pkl`` file produced by
                ``training/train.py``.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        saved = joblib.load(model_path)
        self._model: XGBClassifier = saved["model"]
        self._features: list[str] = saved["features"]
        self._threshold: float = saved["threshold"]

        # TreeExplainer uses the model's internal tree structure to compute
        # SHAP values efficiently (no background dataset required).
        self._explainer = shap.TreeExplainer(self._model)

        logger.info(
            f"Loaded model with {len(self._features)} features, "
            f"threshold={self._threshold:.2f}"
        )

    @property
    def features(self) -> list[str]:
        """The feature names the model expects, in order."""
        return list(self._features)

    def predict(self, application: LoanApplication) -> RiskAssessment:
        """Run the model on a single application and explain the prediction.

        Args:
            application: Validated applicant data.

        Returns:
            RiskAssessment with risk label, probability, SHAP explanations.
        """
        # Map Pydantic field names to the column names the model was trained on.
        # The model expects these 7 features in a specific order (saved in
        # self._features), so we reindex the DataFrame to guarantee that order.
        row = {
            "LIMIT_BAL": application.limit_bal,
            "AGE": application.age,
            "PAY_0": application.pay_0,
            "PAY_2": application.pay_2,
            "PAY_3": application.pay_3,
            "PAY_AMT1": application.pay_amt1,
            "BILL_AMT1": application.bill_amt1,
        }
        # Wrap the single-row dict in a list so DataFrame treats it as one row,
        # then select columns in the exact training order via self._features.
        features_df = pd.DataFrame([row])[self._features]

        proba = float(self._model.predict_proba(features_df)[0, 1])
        risk = "High" if proba > self._threshold else "Low"

        # Compute SHAP values: each feature's contribution to pushing this
        # prediction above or below the baseline (average default rate).
        # shap_values shape is (1, n_features) for a single prediction.
        shap_values = self._explainer.shap_values(features_df)
        shap_row = shap_values[0]

        explanations = [
            FeatureContribution(
                feature_name=feat,
                feature_value=row[feat],
                shap_value=round(float(shap_row[i]), 4),
                impact="increases_risk" if shap_row[i] >= 0 else "decreases_risk",
            )
            for i, feat in enumerate(self._features)
        ]

        # Sort by absolute SHAP value so the most influential features appear first
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

        return RiskAssessment(
            risk=risk,
            default_probability=round(proba, 4),
            features_used=self._features,
            shap_explanations=explanations,
        )