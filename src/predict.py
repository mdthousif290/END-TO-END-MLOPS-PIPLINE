import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="Churn Prediction API", version="1.0.0")

try:
    model = mlflow.sklearn.load_model("models:/ChurnModel/1")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    model = None


class CustomerData(BaseModel):
    tenure: float = 12
    MonthlyCharges: float = 70.35
    TotalCharges: float = 845.5
    gender: str = "Male"
    SeniorCitizen: str = "0"
    Partner: str = "Yes"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.5,
                "gender": "Male",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_label": "Yes - Will Churn" if prediction == 1 else "No - Will Stay",
        "churn_probability": round(float(probability), 4)
    }
