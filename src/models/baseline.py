import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('.')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, roc_auc_score)
from src.pipeline.feature_engineering import scale_features

def train_baseline(df):
    X, y = scale_features(df, fit=True)
    X_tr,X_te,y_tr,y_te = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100,random_state=42)
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp   = m.predict(X_te)
        prob = m.predict_proba(X_te)[:,1]
        results[name] = {
            'model':m,
            'accuracy':  accuracy_score(y_te,yp),
            'precision': precision_score(y_te,yp),
            'recall':    recall_score(y_te,yp),
            'roc_auc':   roc_auc_score(y_te,prob)
        }
        print(f"\n{name}")
        for k,v in results[name].items():
            if k!='model': print(f"  {k:<12} {v:.4f}")

    # Save best model (RF usually wins on tabular data)
    os.makedirs("data/processed",exist_ok=True)
    with open("data/processed/rf_model.pkl",'wb') as f:
        pickle.dump(results['Random Forest']['model'],f)
    return results

if __name__=="__main__":
    df = pd.read_csv("data/processed/patients_processed.csv")
    train_baseline(df)