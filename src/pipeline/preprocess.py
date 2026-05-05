import pandas as pd
import numpy as np
import os

def preprocess(input_path="data/raw/patients.csv",
               output_path="data/processed/patients_processed.csv"):
    df = pd.read_csv(input_path)

    # Fill numeric nulls with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode categoricals
    df['gender_enc']   = (df['gender'] == 'M').astype(int)
    df['activity_enc'] = df['activity_level'].map(
        {'low':0,'moderate':1,'high':2})

    # Clip physiological outliers
    df['bmi']         = df['bmi'].clip(15, 50)
    df['systolic_bp'] = df['systolic_bp'].clip(80, 200)
    df['glucose']     = df['glucose'].clip(60, 400)
    df['cholesterol'] = df['cholesterol'].clip(100, 400)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed → {output_path}  shape={df.shape}")
    return df

if __name__ == "__main__":
    preprocess()    