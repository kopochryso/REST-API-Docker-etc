import joblib
import pandas as pd

MODEL_PATH = "../model/churn_model.pkl"
SCALER_PATH = "../model/scaler.pkl"


def predict(input_data: dict):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)

    prob = model.predict_proba(df_scaled)[0][1]
    return prob


if __name__ == "__main__":
    example = {
        "tenure": 12,
        "MonthlyCharges": 89.3,
        "TotalCharges": 1020,
        # + ALL dummy columns
    }

    print("Churn probability:", predict(example))
