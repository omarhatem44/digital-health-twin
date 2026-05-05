import mlflow, mlflow.xgboost
import pandas as pd, sys
sys.path.append('.')
from src.pipeline.feature_engineering import scale_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("digital-health-twin")

PARAMS = {
    "n_estimators":200,"max_depth":5,"learning_rate":0.1,
    "subsample":0.8,"colsample_bytree":0.8
}

def run():
    df = pd.read_csv("data/processed/patients_processed.csv")
    X, y = scale_features(df, fit=False)
    X_tr,X_te,y_tr,y_te = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y)

    spw = (y_tr==0).sum()/(y_tr==1).sum()

    with mlflow.start_run(run_name="xgb_experiment"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("scale_pos_weight", round(float(spw),2))

        model = xgb.XGBClassifier(
            **PARAMS, scale_pos_weight=spw,
            eval_metric='auc', random_state=42,
            verbosity=0, early_stopping_rounds=20)
        model.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=False)

        yp   = model.predict(X_te)
        prob = model.predict_proba(X_te)[:,1]

        from sklearn.metrics import (accuracy_score,precision_score,
                                     recall_score,roc_auc_score)
        metrics = {
            "accuracy":  accuracy_score(y_te,yp),
            "precision": precision_score(y_te,yp),
            "recall":    recall_score(y_te,yp),
            "roc_auc":   roc_auc_score(y_te,prob)
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model,"xgb_model",
            registered_model_name="HealthRiskXGB")

        print("MLflow run complete. Metrics:", metrics)

if __name__=="__main__":
    run()