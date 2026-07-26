# Design Decisions

## Data Dictionary

The model uses 7 features from the [Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI / Kaggle). Below is the complete value reference for each feature the API accepts.

### Feature Reference

#### LIMIT_BAL
- **Description**: Total credit limit (individual + supplementary)
- **Type**: Float (currency units)
- **Range**: 10,000 — 1,000,000
- **Example**: 200000

#### AGE
- **Description**: Applicant's age in years
- **Type**: Integer
- **Range**: 18 — 120
- **Example**: 35

#### PAY_0, PAY_2, PAY_3
- **Description**: Repayment status in the last 3 months (PAY_0 = most recent month)
- **Type**: Integer
- **Range**: -2 to 8
- **Value meaning**:
  | Value | Meaning |
  |-------|---------|
  | -2 | No consumption that month (card existed but not used) |
  | -1 | Pay duly (paid in full on time) |
  | 0 | No delay (revolving credit used, payment received) |
  | 1 | Payment delay of 1 month |
  | 2 | Payment delay of 2 months |
  | 3 | Payment delay of 3 months |
  | 4–8 | Payment delay of 4–8 months |
- **Example**: -1 (paid on time)

#### PAY_AMT1
- **Description**: Amount paid in the most recent month
- **Type**: Float (currency units)
- **Range**: 0 — 873,552
- **Example**: 5000

#### BILL_AMT1
- **Description**: Bill statement amount for the most recent month
- **Type**: Float (can exceed credit limit)
- **Range**: 0 — 964,663
- **Example**: 30000

### Target Variable
- **Description**: Whether the client defaulted the following month
- **Values**: 0 = No default, 1 = Default
- **Distribution**: 78% non-default, 22% default (imbalanced)

## Why 7 Features Instead of 5 or 23?

### Background
The original dataset (Default of Credit Card Clients) contains 23 features spanning:
- Demographics (AGE, SEX, EDUCATION, MARRIAGE)
- Credit limit (LIMIT_BAL)
- 6 months of repayment status (PAY_0 through PAY_6)
- 6 months of bill amounts (BILL_AMT1 through BILL_AMT6)
- 6 months of payment amounts (PAY_AMT1 through PAY_AMT6)

### The Problem with 23 Features
A loan officer cannot reasonably fill 23 fields for each applicant. The form becomes unusable, and the demo loses its practical value. Not all 23 features carry independent signal — many are highly correlated.

### The Problem with 5 Features
The original model used only PAY_0, PAY_2, PAY_3, AGE, and PAY_AMT1. This misses an important dimension: the applicant's *financial capacity* (credit limit, bill size). A person with a high credit limit and timely payments is very different from someone with a low credit limit and timely payments.

### The 7-Feature Solution
We capture the *same information* as the full 23 features by selecting one representative from each correlated group:

| Feature | Represents | Why |
|---------|------------|-----|
| LIMIT_BAL | Financial capacity | Higher credit limit = more trust from other banks |
| AGE | Demographics | Younger applicants statistically default more |
| PAY_0 | Current repayment behavior | Most recent payment status is the strongest predictor |
| PAY_2 | Recent history (2 months ago) | Shows if current delay is a pattern |
| PAY_3 | Recent history (3 months ago) | Provides trend direction |
| PAY_AMT1 | Payment amount | How much they actually paid last month |
| BILL_AMT1 | Current outstanding bill | Debt-to-payment ratio signal |

### Features Excluded

| Excluded Features | Reason |
|-------------------|--------|
| PAY_4, PAY_5, PAY_6 | Highly correlated with PAY_0–PAY_3. If someone was late 3 months ago, they're likely still late. |
| BILL_AMT2–BILL_AMT6 | Highly correlated with BILL_AMT1. Bills don't change drastically month-to-month. |
| PAY_AMT2–PAY_AMT6 | Highly correlated with PAY_AMT1. Payment behavior is consistent. |
| SEX | Ethical concern + weak predictive signal. Excluding it avoids bias in lending decisions. |
| EDUCATION | Weak signal, encoding issues in dataset (values 5, 6 are "unknown"). |
| MARRIAGE | Weak predictive signal. |

## Technology Choices

### FastAPI over Flask
- Built-in request validation (Pydantic) — no manual field checking
- Automatic OpenAPI docs at `/docs` — great for API exploration
- Async support — important for LLM calls (don't block the server)
- Industry trajectory: new ML/AI projects are increasingly using FastAPI

### XGBoost over Logistic Regression
- Higher accuracy on tabular data (handles non-linear relationships)
- Built-in handling of missing values
- Feature importance scores (useful for SHAP analysis)
- The standard for Kaggle-style tabular problems

### SHAP for Explainability
- Model-agnostic — works with XGBoost, Logistic Regression, any model
- Provides per-feature contribution (not just global importance)
- Industry standard for ML interpretability
- Direct answer to interview question: "How do you explain your model's predictions?"

### LLM for Report Generation (not for Decision)
The LLM generates a *narrative report* summarizing risk factors. It does NOT:
- Make the accept/reject decision
- Assign the probability score
- Determine which features matter

This separation is intentional: the ML model handles quantification (deterministic, verifiable), and the LLM handles communication (fluent, adaptive).

### JSON File Storage (not a Database)
For this project, a database adds setup complexity without proportional benefit. JSON storage provides:
- Zero configuration
- Human-readable audit trail
- Easy to upgrade to SQLite/PostgreSQL later

A database is the right choice at scale. JSON is the right choice for a demo-able, deployable project.

### Adapter Pattern for LLM Providers
Instead of hardcoding one LLM provider, we use an abstract interface:
- Swap providers by changing config, not code
- Automatic fallback if one provider is rate-limited
- Demonstrates understanding of dependency inversion principle
