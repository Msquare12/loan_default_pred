from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from typing import List, Dict, Any
from pathlib import Path
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/latest/model.joblib")

app = FastAPI(title="Loan Default Model API", version="0.1.0")

class Record(RootModel[Dict[str, Any]]):
    """A single record: arbitrary feature dict as the root value."""


class Batch(BaseModel):
    records: List[Record]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def load_model():
    global ARTIFACT
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at: {model_path.resolve()}")
    ARTIFACT = joblib.load(model_path)


@app.post("/predict")
def predict(batch: Batch):
    try:
        pipe = ARTIFACT["pipeline"]
        th = ARTIFACT["threshold"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model artifact not loaded or invalid") from e

    import pandas as pd
    df = pd.DataFrame([r.root for r in batch.records])

    # Ensure missing columns are present (keeps column order stable)
    pre = pipe.named_steps["pre"]
    required_cols = list(pre.feature_names_in_)
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Predict
    probs = pipe.predict_proba(df[required_cols])[:, 1]
    preds = (probs >= th).astype(int).tolist()
    return {
        "probabilities": probs.tolist(),
        "predictions": preds,
        "threshold": float(th),
    }
