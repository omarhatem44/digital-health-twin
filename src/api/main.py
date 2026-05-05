from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('.')
from src.pipeline.feature_engineering import engineer_features, FEATURES
from src.rag.retriever import HealthRAGRetriever
from src.insights.explainer import InsightGenerator

app = FastAPI(
    title="Digital Health Twin API",
    description="AI-powered health risk prediction + clinical reasoning",
    version="1.0.0"
)

# ── Load artifacts once at startup ────────────────────────────
with open("data/processed/xgb_model.pkl","rb") as f:
    MODEL = pickle.load(f)
with open("data/processed/scaler.pkl","rb") as f:
    SCALER = pickle.load(f)

RETRIEVER = HealthRAGRetriever()
INSIGHT   = InsightGenerator()

# ── Schemas ───────────────────────────────────────────────────
class Patient(BaseModel):
    patient_id:    str
    age:           int
    gender:        str        # "M" | "F"
    bmi:           float
    systolic_bp:   int
    diastolic_bp:  int
    heart_rate:    int
    glucose:       float
    cholesterol:   float
    activity_level:str        # "low"|"moderate"|"high"
    smoking:       int        # 0|1
    diabetes:      int        # 0|1
    notes:         Optional[str] = ""

class AskRequest(BaseModel):
    patient_id:   str
    question:     str
    patient_data: Patient

# ── Helpers ───────────────────────────────────────────────────
def patient_to_array(p: Patient):
    row = {
        'age':p.age,'bmi':p.bmi,
        'systolic_bp':p.systolic_bp,'diastolic_bp':p.diastolic_bp,
        'heart_rate':p.heart_rate,'glucose':p.glucose,
        'cholesterol':p.cholesterol,
        'gender_enc':1 if p.gender.upper()=="M" else 0,
        'activity_enc':{'low':0,'moderate':1,'high':2}.get(
                        p.activity_level.lower(),1),
        'smoking':p.smoking,'diabetes':p.diabetes
    }
    df   = engineer_features(pd.DataFrame([row]))
    return SCALER.transform(df[FEATURES])

# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status":"healthy","service":"Digital Health Twin API"}

@app.post("/predict")
def predict(patient: Patient):
    try:
        X    = patient_to_array(patient)
        pred = int(MODEL.predict(X)[0])
        conf = float(MODEL.predict_proba(X)[0][pred])

        query    = (f"patient age {patient.age} bmi {patient.bmi} "
                    f"bp {patient.systolic_bp}/{patient.diastolic_bp}")
        ctx      = RETRIEVER.retrieve(query, top_k=3)
        insight  = INSIGHT.generate_insight(
            patient.model_dump(), {'prediction':pred,'confidence':conf}, ctx)

        return {
            "patient_id":  patient.patient_id,
            "prediction":  pred,
            "risk_level":  "HIGH" if pred==1 else "LOW",
            "confidence":  round(conf,4),
            "explanation": insight,
            "retrieved_context": [
                {"patient_id":r["patient_id"],
                 "similarity":round(r["score"],4),
                 "summary":r["text"][:200]}
                for r in ctx
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(req: AskRequest):
    try:
        ctx    = RETRIEVER.retrieve(req.question, top_k=3)
        answer = INSIGHT.answer_question(
            req.question, req.patient_data.model_dump(), ctx)
        return {
            "patient_id": req.patient_id,
            "question":   req.question,
            "answer":     answer,
            "retrieved_context": [
                {"patient_id":r["patient_id"],
                 "similarity":round(r["score"],4),
                 "summary":r["text"][:200]}
                for r in ctx
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)