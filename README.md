# Credit Risk Prediction App

A machine learning web application that predicts whether a user is likely to default on credit payments based on recent repayment behavior.

This project demonstrates an end-to-end ML workflow:
- Data understanding and feature selection
- Model training and evaluation
- Backend API using Flask
- User-friendly frontend interface
- Deployment-ready setup

---

## Problem Statement

Given a user's financial and repayment history, predict whether they are at **high risk** of defaulting on credit payments.

---

## Dataset

- Default of Credit Card Clients Dataset (UCI / Kaggle)
- Includes demographic data, billing amounts, and repayment history

---

## Approach

### Feature Selection

The initial model used 20+ features.  
This was reduced to **5 key features** to balance performance and usability:

- PAY_0 → last month repayment status
- PAY_2 → repayment status 2 months ago
- PAY_3 → repayment status 3 months ago
- AGE → user age
- PAY_AMT1 → last payment amount

This reduction:
- simplifies user input
- improves interpretability
- enables a clean UI

---

### Model

- Algorithm: Logistic Regression  
- Class imbalance handled using: `class_weight='balanced'`

---

### Evaluation

Final model performance:

- F1 Score: ~0.52  
- Accuracy: ~0.77  
- Precision (default class): 0.48  
- Recall (default class): 0.56  

Threshold tuning was performed:
- 0.5 chosen for best F1 balance  
- higher thresholds improve precision but reduce recall  

---

### Key Insight

Repayment behavior is the strongest predictor:

- PAY_0 has the highest impact  
- recent delays strongly indicate default risk  

---

## Features

- Simple and user-friendly web form  
- Dropdown-based repayment selection  
- Default values for quick testing  
- Risk prediction (High / Low)  
- Probability score output  
- Explanation of prediction (rule-based reasoning)  

---

## Tech Stack

- Python  
- Flask  
- Scikit-learn  
- Pandas  
- NumPy  
- HTML + Bootstrap  

---

## Project Structure

    credit-risk-prediction-app/
    │
    ├── app.py
    ├── model.pkl
    ├── requirements.txt
    ├── templates/
    │   └── index.html

---

## How to Run Locally

### 1. Clone the repository

    git clone <your-repo-url>
    cd credit-risk-prediction-app

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Run the application

    python app.py

### 4. Open in browser

    http://127.0.0.1:5000

---

## Example Input

- Age: 25  
- Payment status: Paid on time  
- Payment amount: 2000  

---

## Example Output

- Prediction: Low Risk  
- Probability: 0.32  
- Explanation:
  - Consistent on-time payments  
  - Strong recent payment amount  

---

## Future Improvements

- Add model-based explanations (SHAP/LIME)  
- Improve UI/UX design  
- Containerize with Docker  
- Add authentication and logging  
- Experiment with advanced models (XGBoost, LightGBM)  

---

## Author

Balaji Mahendra