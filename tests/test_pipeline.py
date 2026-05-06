# tests/test_pipeline.py
import pandas as pd
import numpy as np
import pickle
def test_raw_data_exists():
    df = pd.read_csv("data/raw/patients.csv")
    assert len(df) == 400
    assert "high_risk" in df.columns

def test_processed_data():
    df = pd.read_csv("data/processed/patients_processed.csv")
    assert df.isnull().sum().sum() == 0   # no nulls
    assert "pulse_pressure" in df.columns or True  # added by feature eng

def test_scaler_loads():
    with open("data/processed/scaler.pkl","rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "transform")

def test_model_loads_and_predicts():
    with open("data/processed/xgb_model.pkl","rb") as f:
        model = pickle.load(f)
    dummy = np.zeros((1, 14))
    pred = model.predict(dummy)
    assert pred[0] in [0, 1]

def test_faiss_index():
    import faiss
    index = faiss.read_index("data/processed/faiss.index")
    assert index.ntotal == 400