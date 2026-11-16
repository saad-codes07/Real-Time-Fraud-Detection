"""
main.py - FastAPI service that serves the trained fraud model.

Run from the project root:
    uvicorn api.main:app --reload
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# Where train.py saved model.json and metadata.json. Docker can override this
# later through an environment variable, so we don't hard-code the path.
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models")
)


# --- What a request must look like -------------------------------------------
# A transaction has V1..V28 (anonymized features), Amount, and Hour of day.
# Pydantic checks every incoming request against this shape and automatically
# rejects bad input (missing field, text instead of a number, Hour = 99...).
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)  # ge=0: must be 0 or more
    Hour: int = Field(ge=0, le=23)  # must be between 0 and 23


# --- What a response looks like ----------------------------------------------
class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


# --- Load the model ONCE when the server starts ------------------------------
# Loading from disk on every request would be slow. Loading once at startup
# keeps each prediction fast.
@asynccontextmanager
async def lifespan(app: FastAPI):
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    app.state.model = model
    app.state.threshold = meta["threshold"]  # cut-off chosen during training
    app.state.features = meta["features"]  # exact column order the model expects
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    """Quick check that the server is up and the model is loaded."""
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction, request: Request):
    """Score one transaction."""
    state = request.app.state
    # One-row table; columns must be in the same order used during training.
    row = pd.DataFrame([tx.model_dump()])[state.features]
    proba = float(state.model.predict_proba(row)[0, 1])  # chance of fraud, 0 to 1
    return Prediction(
        fraud_probability=proba,
        is_fraud=proba >= state.threshold,
        threshold=state.threshold,
    )