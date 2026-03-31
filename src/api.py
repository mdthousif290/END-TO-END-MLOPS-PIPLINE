import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()
models_dir = Path(__file__).resolve().parents[1] / "models"
model_path = models_dir / "model.pkl"
preprocessor_path = models_dir / "preprocessor.pkl"

if not model_path.exists() or not preprocessor_path.exists():
    raise FileNotFoundError(
        "Model or preprocessor not found. Run src/preprocess.py and src/train.py first."
    )

with open(preprocessor_path, "rb") as handle:
    preprocessor = pickle.load(handle)
with open(model_path, "rb") as handle:
    model = pickle.load(handle)


@app.get("/")
def root() -> dict:
    return {
        "status": "ready",
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
    }


@app.post("/predict")
def predict(data: dict[str, Any]) -> dict:
    if not data:
        raise HTTPException(status_code=400, detail="Request body must be a JSON object with model features.")

    df = pd.DataFrame([data])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0.0)

    try:
        features = preprocessor.transform(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not transform input data: {exc}")

    prediction = int(model.predict(features)[0])
    return {"prediction": prediction}
