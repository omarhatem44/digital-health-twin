from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from typing import Optional

import pandas as pd
import pickle
import sys
from pathlib import Path


# =========================================================
# PATHS
# src/api/main.py → parent = src/api → parent = src → parent = project root
# Docker WORKDIR /app → BASE_DIR = /app  ✅
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(BASE_DIR))
# noqa: E402

from src.pipeline.feature_engineering import engineer_features, FEATURES
from src.rag.retriever import HealthRAGRetriever
from src.insights.explainer import InsightGenerator


MODEL_PATH    = BASE_DIR / "model" / "xgb_model.pkl"
SCALER_PATH   = BASE_DIR / "model" / "scaler.pkl"
TEMPLATES_DIR = BASE_DIR / "templates"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Digital Health Twin API",
    description="AI-powered health risk prediction + clinical reasoning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================================================
# LOAD ARTIFACTS AT STARTUP
# =========================================================

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)

RETRIEVER = HealthRAGRetriever()
INSIGHT   = InsightGenerator()


# =========================================================
# SCHEMAS
# =========================================================

class Patient(BaseModel):
    patient_id:     str
    age:            int
    gender:         str
    bmi:            float
    systolic_bp:    int
    diastolic_bp:   int
    heart_rate:     int
    glucose:        float
    cholesterol:    float
    activity_level: str
    smoking:        int
    diabetes:       int
    notes:          Optional[str] = ""


class AskRequest(BaseModel):
    patient_id:   str
    question:     str
    patient_data: Patient


# =========================================================
# HELPERS
# =========================================================

def patient_to_array(p: Patient):
    row = {
        "age":          p.age,
        "bmi":          p.bmi,
        "systolic_bp":  p.systolic_bp,
        "diastolic_bp": p.diastolic_bp,
        "heart_rate":   p.heart_rate,
        "glucose":      p.glucose,
        "cholesterol":  p.cholesterol,
        "gender_enc":   1 if p.gender.upper() == "M" else 0,
        "activity_enc": {"low": 0, "moderate": 1, "high": 2}.get(
            p.activity_level.lower(), 1
        ),
        "smoking":  p.smoking,
        "diabetes": p.diabetes,
    }

    df = engineer_features(pd.DataFrame([row]))
    X  = SCALER.transform(df[FEATURES])
    return X


# =========================================================
# ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "healthy", "service": "Digital Health Twin API"}


@app.post("/predict")
def predict(patient: Patient):
    try:
        X    = patient_to_array(patient)
        pred = int(MODEL.predict(X)[0])
        conf = float(MODEL.predict_proba(X)[0][pred])

        query = (
            f"patient age {patient.age} "
            f"bmi {patient.bmi} "
            f"bp {patient.systolic_bp}/{patient.diastolic_bp}"
        )

        ctx     = RETRIEVER.retrieve(query, top_k=3)
        insight = INSIGHT.generate_insight(
            patient.model_dump(),
            {"prediction": pred, "confidence": conf},
            ctx,
        )

        return {
            "patient_id": patient.patient_id,
            "prediction": pred,
            "risk_level": "HIGH" if pred == 1 else "LOW",
            "confidence": round(conf, 4),
            "explanation": insight,
            "retrieved_context": [
                {
                    "patient_id": r["patient_id"],
                    "similarity": round(r["score"], 4),
                    "summary":    r["text"][:200],
                }
                for r in ctx
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        ctx    = RETRIEVER.retrieve(req.question, top_k=3)
        answer = INSIGHT.answer_question(
            req.question,
            req.patient_data.model_dump(),
            ctx,
        )

        return {
            "patient_id": req.patient_id,
            "question":   req.question,
            "answer":     answer,
            "retrieved_context": [
                {
                    "patient_id": r["patient_id"],
                    "similarity": round(r["score"], 4),
                    "summary":    r["text"][:200],
                }
                for r in ctx
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# LOCAL DEV ONLY — Docker uses CMD in Dockerfile
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )