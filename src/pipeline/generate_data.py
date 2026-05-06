import pandas as pd
import numpy as np
import random
import os

def generate_patient_data(n=400, seed=42):
    np.random.seed(seed); random.seed(seed)
    pid = [f"P{str(i).zfill(4)}" for i in range(1,n+1)]
    ages = np.random.randint(20,85,n)
    genders = np.random.choice(['M','F'],n)
    bmis = np.clip(np.round(np.random.normal(27.5,5.5,n),1),15,50)
    sbp = np.random.normal(125,18,n).astype(int)
    dbp = np.random.normal(82,12,n).astype(int)
    hr  = np.random.normal(75,12,n).astype(int)
    glucose = np.random.normal(100,25,n)
    chol    = np.random.normal(200,35,n)
    activity = np.random.choice(['low','moderate','high'],n,p=[0.4,0.4,0.2])
    smoking  = np.random.choice([0,1],n,p=[0.7,0.3])
    diabetes = np.random.choice([0,1],n,p=[0.85,0.15])

    risk = (
        (ages>55).astype(int)*2 + (bmis>30).astype(int)*2 +
        (sbp>140).astype(int)*2 + (glucose>126).astype(int)*2 +
        smoking*2 + diabetes*3 +
        (activity=='low').astype(int)
    )
    high_risk = (risk>=5).astype(int)

    notes_pool = [
        "Patient reports occasional chest pain and shortness of breath.",
        "Regular exercise 3x per week. No major complaints.",
        "History of hypertension, currently on medication.",
        "Family history of diabetes. Monitoring glucose levels.",
        "Sedentary lifestyle, high-stress job. Advised lifestyle changes.",
        "Recent weight gain of 10kg over 6 months.",
        "Non-smoker, occasional alcohol. Generally healthy.",
        "Complains of fatigue and dizziness.",
        "Post-surgery follow-up. Recovery progressing well.",
        "Elevated cholesterol despite medication."
    ]
    notes = [random.choice(notes_pool) for _ in range(n)]

    df = pd.DataFrame({
        'patient_id':pid,'age':ages,'gender':genders,'bmi':bmis,
        'systolic_bp':sbp,'diastolic_bp':dbp,'heart_rate':hr,
        'glucose':glucose,'cholesterol':chol,'activity_level':activity,
        'smoking':smoking,'diabetes':diabetes,'notes':notes,'high_risk':high_risk
    })
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_patient_data(400)
    df.to_csv("data/raw/patients.csv", index=False)
    print(f"Generated {len(df)} records")
    print(df['high_risk'].value_counts())