import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

# =========================
# Load data
# =========================
df = pd.read_csv(
    "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/refs/heads/main/dataset/default_of_credit_card_clients.csv"
)

df.rename(columns={"default payment next month": "target"}, inplace=True)

# =========================
# Feature selection
# =========================
selected_features = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "AGE",
    "PAY_AMT1"
]

X = df[selected_features]
y = df["target"]

print("Selected Features:", selected_features)

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train
# =========================
model = LogisticRegression(
    C=0.615848211066026,
    solver="liblinear",
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluate with custom threshold
# =========================
threshold = 0.5
probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > threshold).astype(int)
# for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
#     y_pred = (probs > threshold).astype(int)
#     f1 = f1_score(y_test, y_pred)
#     print(f"Threshold: {threshold} | F1: {f1:.4f}")

print(f"\nUsing threshold = {threshold}")
print("\nF1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# Feature importance
# =========================
importance = dict(zip(selected_features, model.coef_[0]))
importance = {k: abs(v) for k, v in importance.items()}

print("\nFeature Importance:")
for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v}")

# =========================
# Save model + config
# =========================
joblib.dump(
    {
        "model": model,
        "features": selected_features,
        "threshold": threshold   # important: save threshold too
    },
    "model.pkl"
)

print("\nModel saved as model.pkl")