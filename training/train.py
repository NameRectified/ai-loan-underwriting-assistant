"""Train an XGBoost model for credit default risk prediction.

Downloads the Default of Credit Card Clients dataset,
trains an XGBoost classifier on 7 selected features,
tunes the decision threshold for best F1 score,
and saves the model + metadata to models/model.pkl.
"""

import os

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_URL = "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/refs/heads/main/dataset/default_of_credit_card_clients.csv"

SELECTED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_AMT1",
    "BILL_AMT1",
]

MODEL_PATH = "models/model.pkl"


def load_data(url: str) -> pd.DataFrame:
    """Download and load the credit card default dataset.

    Args:
        url: Raw CSV URL of the dataset.

    Returns:
        DataFrame with target column renamed to 'target'.
    """
    df = pd.read_csv(url)
    df.rename(columns={"default payment next month": "target"}, inplace=True)
    return df


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float
) -> XGBClassifier:
    """Train an XGBoost classifier with class imbalance handling.

    Args:
        X_train: Training features.
        y_train: Training labels.
        scale_pos_weight: Weight for the positive class (neg/pos ratio).

    Returns:
        Trained XGBClassifier.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=(-1, 0, 1, 1, 1, -1, 1),
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(
    model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    """Search for the decision threshold that maximizes F1 score.

    Args:
        model: Trained classifier.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Best threshold value (between 0.30 and 0.70).
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.30, 0.71, 0.05):
        preds = (probs > threshold).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    logger.info(f"Best threshold {best_threshold:.2f} achieved F1 {best_f1:.4f}")
    return best_threshold


def main() -> None:
    """Run the full training pipeline."""
    logger.info("Loading data...")
    df = load_data(DATA_URL)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

    X = df[SELECTED_FEATURES]
    y = df["target"]

    # Split: 70% train, 15% validation (for threshold tuning), 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"Class ratio (neg/pos): {scale_pos_weight:.2f}")

    logger.info("Training XGBoost...")
    model = train_model(X_train, y_train, scale_pos_weight)

    threshold = find_best_threshold(model, X_val, y_val)

    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs > threshold).astype(int)

    logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logger.info(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    importance = dict(zip(SELECTED_FEATURES, model.feature_importances_))
    importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )
    logger.info("Feature importance:")
    for k, v in importance.items():
        logger.info(f"  {k}: {v:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    baseline_default_rate = float(y_train.mean())
    joblib.dump(
        {
            "model": model,
            "features": SELECTED_FEATURES,
            "threshold": threshold,
            "baseline_default_rate": baseline_default_rate,
        },
        MODEL_PATH,
    )
    logger.success(f"Model saved to {MODEL_PATH} with baseline default rate {baseline_default_rate:.4f}")


if __name__ == "__main__":
    main()
