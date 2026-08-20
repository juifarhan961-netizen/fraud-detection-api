from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

app = FastAPI(title="Fraud Detection API")

# Download model from private Hugging Face repository
model_path = hf_hub_download(
    repo_id="juifarhan961/fraud-detection-random-forest",
    filename="random_forest_fraud_model.pkl",
    token=os.getenv("HF_TOKEN")
)

# Load model
model = joblib.load(model_path)

print("✅ Fraud Detection Model Loaded")


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
