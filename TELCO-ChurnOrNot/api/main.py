from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # + all dummy variables

@app.post("/predict")
def predict_churn(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]
    return {"churn_probability": float(prob)}
