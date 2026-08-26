"""
FastAPI entry point for the EMI prediction service.
"""

from __future__ import annotations
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from src.api.schemas import EMIRequest, EMIResponse
from src.api.exceptions import inference_exception_handler, unexpected_exception_handler
from src.inference.predictor import EMIPredictor

predictor: EMIPredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = EMIPredictor()
    yield
    predictor = None

app = FastAPI(
    title="EMI Predictor API",
    description="API for classifying EMI eligibility and predicting maximum monthly EMI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_exception_handler(ValueError, inference_exception_handler)
app.add_exception_handler(TypeError, inference_exception_handler)
app.add_exception_handler(Exception, unexpected_exception_handler)

@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "EMI Predictor API is running. Visit /docs for documentation."}

@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=EMIResponse)
def predict(request: EMIRequest) -> EMIResponse:
    """Generate classification and regression predictions."""
    if predictor is None:
        raise RuntimeError("Predictor not initialized.")

    # Convert request payload to DataFrame
    data = pd.DataFrame([request.model_dump()])
    
    # Run predictions
    results = predictor.predict(data)
    
    return EMIResponse(
        emi_eligibility=results.iloc[0]["emi_eligibility"],
        max_monthly_emi=float(results.iloc[0]["max_monthly_emi"]),
    )
