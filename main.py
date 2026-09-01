from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Telco Churn Prediction API")

# Load once at startup, not per-request
pipeline = joblib.load("churn_model_pipeline.pkl")
threshold = joblib.load("churn_model_threshold.pkl")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])
    prob = pipeline.predict_proba(input_df)[0, 1]
    prediction = int(prob >= threshold)

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": prediction,
        "threshold_used": round(float(threshold), 4),   # <-- cast this too
    }