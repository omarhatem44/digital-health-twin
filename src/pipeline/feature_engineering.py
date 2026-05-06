import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

FEATURES = [
    'age','bmi','systolic_bp','diastolic_bp','heart_rate',
    'glucose','cholesterol','gender_enc','activity_enc',
    'smoking','diabetes',
    'pulse_pressure','bmi_age_ratio','hypertensive'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['bmi_age_ratio']  = df['bmi'] * df['age'] / 100
    df['hypertensive']   = (
        (df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)
    ).astype(int)
    return df

def scale_features(df, fit=True,
                   scaler_path="data/processed/scaler.pkl"):
    df = engineer_features(df)
    X  = df[FEATURES].copy()

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs("data/processed", exist_ok=True)
        with open(scaler_path,'wb') as f: pickle.dump(scaler, f)
    else:
        with open(scaler_path,'rb') as f: scaler = pickle.load(f)
        X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=FEATURES), df['high_risk']

if __name__ == "__main__":
    df = pd.read_csv("data/processed/patients_processed.csv")
    X, y = scale_features(df, fit=True)
    print("Feature matrix:", X.shape)
    print(X.describe().round(2))