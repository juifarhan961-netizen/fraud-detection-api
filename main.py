
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Fraud Detection API")

model = joblib.load("random_forest_fraud_model.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


class Transaction(BaseModel):
    account_age_days: int
    total_transactions_user: int
    avg_amount_user: float
    amount: float
    country: str
    bin_country: str
    channel: str
    merchant_category: str
    promo_used: int
    avs_match: int
    cvv_result: int
    three_ds_flag: int
    shipping_distance_km: float
    transaction_hour: int
    transaction_dayofweek: int
    transaction_month: int
    is_weekend: int


@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([transaction.model_dump()])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    result = "Fraud" if prediction == 1 else "Not Fraud"

    return {
        "prediction": int(prediction),
        "result": result,
        "risk_probability": round(float(probability), 4)
    }
