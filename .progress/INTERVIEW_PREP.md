# Interview Prep — AI Loan Underwriting Assistant

## Product Principle: Human-Readable Outputs

**Q: How do you present ML output to non-technical stakeholders?**

Every ML output exposed to users must be self-explanatory. Raw artifacts — feature codes (`PAY_0`), log-odds SHAP values, `-1`/`0`/`1` enums — are *internal representations*. The API and UI translate them into domain language:

- `PAY_0` → "Repayment Status (Last Month)"
- `-1` → "Paid in full"
- SHAP `-0.4988` → "Very strongly decreases risk" (raw value in a tooltip)

I implement this at the API layer (backend returns `feature_label`, `value_label`, `magnitude`), not just the frontend, so *any* consumer — a loan officer dashboard, a mobile app, an API integration — gets the same clarity. This matters because outputs are consumed by non-technical stakeholders (loan officers, regulators) and decisions must be auditable and explainable.

The LLM reports follow the same rule: the prompt is built from the human-readable labels (`feature_label`, `value_label`, `magnitude`), and the system prompt forbids internal codes like `PAY_0`. So even the free-form narrative never leaks raw feature codes to the reader.

## Monotonic Constraints

**Q: Why did you add monotonic constraints?**

The model showed counterintuitive behavior — increasing `PAY_2` from -1 (paid in full) to 1 (1 month delay) actually *decreased* the predicted risk. This is unacceptable in credit scoring where worse payment history must never suggest lower risk.

**Q: How do monotonic constraints work?**

XGBoost's `monotone_constraints` parameter restricts the decision tree splits so that the prediction direction stays consistent across a feature's range. Internally, during tree construction, XGBoost rejects any split that would violate the constraint — the gradient of the prediction w.r.t. the constrained feature never changes sign.

**Q: What values did you use and why?**

| Feature | Constraint | Rationale |
|---|---|---|
| LIMIT_BAL | -1 (decreasing) | Higher credit limit → lower default risk |
| AGE | 0 (none) | No strong monotonic assumption for age |
| PAY_0/2/3 | 1 (increasing) | Longer payment delays → higher risk |
| PAY_AMT1 | -1 (decreasing) | Higher payments → lower risk |
| BILL_AMT1 | 1 (increasing) | Higher bills → higher risk |

**Q: Did the constraints hurt model performance?**

No — F1 remained ~0.53 and ROC AUC ~0.77. The constraints only removed spurious relationships; the model's predictive power came from the real patterns.

## Architecture Decisions

**Q: Why use a provider-agnostic LLM layer?**

To avoid vendor lock-in. The `LLMClient` tries Groq first, falls back to Gemini, then OpenRouter. Each provider implements the same interface, so adding a new provider is just another class — no pipeline changes needed.

**Q: Why FastAPI over Flask?**

FastAPI has native async support, automatic OpenAPI docs, and Pydantic v2 integration — meaning request/response validation is declared once in schemas and enforced automatically. For an ML API, this eliminates boilerplate.

**Q: How did you select the 7 features?**

From 23 original features in the UCI dataset, I used XGBoost feature importance and domain knowledge. PAY_0 dominated (58% importance), followed by PAY_2 (24%) and PAY_3 (9%). The billing/payment amounts contributed less but were kept for the SHAP explanations to give loan officers actionable detail.

**Q: Why does every assessment get persisted to a JSON file?**

Every prediction is saved as an audit trail — UUID + timestamp + full assessment. Lending decisions must be traceable and explainable after the fact (regulatory/compliance requirement in real underwriting systems). I'd frame this as an **audit log**, not durable storage — in production I'd swap the JSON file for Postgres/SQLite.

## Performance Numbers

**Q: What are the concrete numbers?**

- Dataset: 30,000 samples, 23 original features → 7 selected
- Split: 21,000 train / 4,500 validation / 4,500 test
- ROC AUC 0.77 (was 0.72 with LogisticRegression), F1 0.53, accuracy 80%
- Decision threshold 0.60, tuned over 0.30–0.70 for best F1
- 12 pytest tests, all passing, ~3s run time
- Inference: prediction + SHAP in milliseconds; + LLM report via Groq ~2s
- 5 LLM provider entries in fallback order: Groq → Gemini → OpenRouter (3 models)

## Model Limitations

**Q: What are the model's weaknesses?**

- F1 of 0.53 means many false positives — the model flags low-risk applicants as high-risk
- Trained on 2005 Taiwan data — may not generalize to other populations or time periods
- Only 7 features miss important signals like debt-to-income ratio, employment history, or loan purpose