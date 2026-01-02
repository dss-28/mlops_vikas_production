# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import run_inference
import logging

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="ML Inference API", version="1.0")

# ----------------------------
# Logger
# ----------------------------
logger = logging.getLogger("uvicorn.error")

# ----------------------------
# Request/Response Schemas
# ----------------------------
class PredictRequest(BaseModel):
    data: dict  # input features

class PredictResponse(BaseModel):
    prediction: str  # keep string to match your model labels (e.g., <=50K, >50K)

# ----------------------------
# Health endpoint
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Convert JSON input to DataFrame
        df = pd.DataFrame([request.data])

        # Run inference using your pipeline
        preds = run_inference(df)

        # Return first prediction (as string)
        return {"prediction": preds[0]}

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))
