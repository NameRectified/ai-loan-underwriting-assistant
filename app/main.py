"""FastAPI application for the AI Loan Underwriting Assistant."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.schemas import ErrorResponse, LoanApplication, RiskAssessment
from app.config.settings import settings
from app.services.llm_client import LLMClient
from app.services.predictor import Predictor
from app.services.report_generator import ReportGenerator

# Module-level state: loaded once at startup via lifespan.
predictor: Optional[Predictor] = None
report_generator: Optional[ReportGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize clients on startup. Clean up on shutdown.

    Everything before ``yield`` runs once when the server starts.
    Everything after ``yield`` runs once when the server stops.
    """
    global predictor, report_generator

    logger.info(f"Loading model from {settings.model_path}")
    predictor = Predictor(settings.model_path)

    llm_client = LLMClient()
    if llm_client.available:
        report_generator = ReportGenerator(llm_client)
        logger.info("LLM report generator initialized")
    else:
        report_generator = None
        logger.warning("No LLM providers configured — reports will be skipped")

    logger.info("Server initialized: predict -> explain -> report")

    yield

    predictor = None
    report_generator = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="AI Loan Underwriting Assistant",
    description=(
        "Predicts default risk, explains predictions via SHAP, "
        "and generates human-readable risk reports."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the minimal frontend."""
    path = Path("app/static/index.html")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post(
    "/api/v1/predict",
    response_model=RiskAssessment,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(application: LoanApplication):
    """Submit a loan application for underwriting review.

    Runs prediction → SHAP explanation → LLM report and returns the
    complete risk assessment.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    assessment = predictor.predict(application)

    if report_generator is not None:
        try:
            report = report_generator.generate(application, assessment)
            assessment.risk_report = report
        except Exception as exc:
            logger.warning(f"Report generation failed: {exc}")
            assessment.risk_report = ""

    return assessment
