from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

saved = joblib.load("model.pkl")

model = saved["model"]
FEATURE_ORDER = saved["features"]
THRESHOLD = saved.get("threshold", 0.5)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form   # <-- you missed this

    features = np.array([[float(data[f]) for f in FEATURE_ORDER]])

    prob = model.predict_proba(features)[0][1]
    prediction = 1 if prob > THRESHOLD else 0


    pay_0 = float(data["PAY_0"])
    pay_2 = float(data["PAY_2"])
    pay_3 = float(data["PAY_3"])
    amount = float(data["PAY_AMT1"])

    reasons = []

    if pay_0 >= 2:
        reasons.append("Recent payments are significantly delayed")
    elif pay_0 == 1:
        reasons.append("Recent payment was slightly delayed")

    if pay_2 >= 2 or pay_3 >= 2:
        reasons.append("Past payment history shows repeated delays")

    if amount < 1000:
        reasons.append("Low recent payment amount")

    if pay_0 <= 0 and pay_2 <= 0 and pay_3 <= 0:
        reasons.append("Consistent on-time payments")

    if amount > 5000:
        reasons.append("Strong recent payment amount")

    if not reasons:
        reasons.append("No major risk indicators detected")

    result = "High Risk" if prediction == 1 else "Low Risk"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(prob, 2),
        reasons=reasons
    )

if __name__ == "__main__":
    app.run(debug=True)