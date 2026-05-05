import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('.')
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, roc_auc_score)
from src.pipeline.feature_engineering import scale_features

def train_xgboost(df):
    X, y = scale_features(df, fit=False)
    X_tr,X_te,y_tr,y_te = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y)

    # Handle class imbalance
    spw = (y_tr==0).sum()/(y_tr==1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='auc', random_state=42, verbosity=0,
        early_stopping_rounds=20
    )
    model.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=False)

    yp   = model.predict(X_te)
    prob = model.predict_proba(X_te)[:,1]

    metrics = {
        'accuracy':  float(accuracy_score(y_te,yp)),
        'precision': float(precision_score(y_te,yp)),
        'recall':    float(recall_score(y_te,yp)),
        'roc_auc':   float(roc_auc_score(y_te,prob))
    }
    print("\nXGBoost Results")
    for k,v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    # Feature importance
    fi = pd.DataFrame({'feature':X.columns,
                       'importance':model.feature_importances_})
    fi = fi.sort_values('importance',ascending=False)
    print("\nTop 5 Features:\n", fi.head().to_string(index=False))

    # Save
    os.makedirs("data/processed",exist_ok=True)
    with open("data/processed/xgb_model.pkl",'wb') as f:
        pickle.dump(model,f)
    with open("data/processed/metrics.json",'w') as f:
        json.dump(metrics,f,indent=2)
    return model, metrics

if __name__=="__main__":
    df = pd.read_csv("data/processed/patients_processed.csv")
    train_xgboost(df)