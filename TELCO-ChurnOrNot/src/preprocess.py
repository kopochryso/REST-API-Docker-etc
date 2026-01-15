import pandas as pd

def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing
    df = df.dropna()

    # Drop ID
    df = df.drop(columns=['customerID'])

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    preprocess(
        "../data/raw/telco_churn.csv",
        "../data/processed/churn_clean.csv"
    )
