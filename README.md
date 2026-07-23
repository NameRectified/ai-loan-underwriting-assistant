# AI Loan Underwriting Assistant

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered loan underwriting system that predicts default risk using XGBoost, explains decisions with SHAP values, and generates natural-language risk reports via LLMs.

## Features

- **Risk Prediction** — XGBoost classifier trained on 7 features from the UCI Credit Card Default dataset (F1 ~0.53, ROC AUC ~0.76)
- **Explainable AI** — SHAP `TreeExplainer` shows which features drove each decision, sorted by impact
- **LLM Risk Reports** — Generates human-readable assessment reports via Groq, Gemini, or OpenRouter (auto fallback)
- **Production API** — FastAPI with Pydantic v2 validation, async lifespan, OpenAPI docs at `/docs`
- **Minimal Frontend** — Pico CSS single-page app at `GET /`
- **Docker Ready** — Multi-platform container with `--env-file` for secrets
- **Tested** — pytest suite with 7 tests for predictor, SHAP, and risk labels

## Quick Start

### Prerequisites

- Python 3.9+
- LLM API key (Groq recommended — free tier available)

### Local Setup

```bash
# Clone and enter the project
git clone https://github.com/NameRectified/ai-loan-underwriting-assistant.git
cd ai-loan-underwriting-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (see below)

# Run the server
./run.sh
```

Open http://127.0.0.1:8000 in your browser.

### Environment Variables

```env
GROQ_API_KEY=gsk_...          # Primary (recommended)
GEMINI_API_KEY=AIza...        # Fallback 1
OPENROUTER_API_KEY=sk-or-...  # Fallback 2
```

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/schemas.py       # Pydantic request/response models
│   ├── config/settings.py   # Environment config via pydantic-settings
│   ├── services/
│   │   ├── predictor.py     # XGBoost + SHAP prediction
│   │   ├── llm_client.py    # Provider-agnostic LLM client
│   │   ├── report_generator.py  # Builds prompts, calls LLM
│   │   └── pipeline.py      # Orchestrates predict → report → persist
│   ├── database/repository.py   # JSON file persistence
│   └── static/index.html    # Frontend
├── models/model.pkl         # Trained XGBoost model
├── prompts/report.yaml      # LLM prompt templates
├── training/train.py        # Model training script
├── tests/test_pipeline.py   # pytest suite
├── Dockerfile               # Container build
├── .dockerignore
├── run.sh                   # Cross-platform launcher
└── requirements.txt
```

## API

### `POST /api/v1/predict`

```json
{
  "limit_bal": 200000,
  "age": 35,
  "pay_0": -1,
  "pay_2": -1,
  "pay_3": -1,
  "pay_amt1": 5000,
  "bill_amt1": 30000
}
```

Returns risk category, probability, 7 SHAP explanations, and an LLM-generated report.

Interactive docs at http://127.0.0.1:8000/docs.

## Docker

```bash
docker build -t loan-underwriter .
docker run --env-file .env -p 8000:8000 loan-underwriter
```

## Tests

```bash
pytest tests/ -v
```

## Design Decisions

See [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) for feature selection rationale, model evaluation, and architecture choices.

## License

MIT