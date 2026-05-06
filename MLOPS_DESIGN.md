# MLOps Design — Digital Health Twin

> This document covers the MLOps strategy for the Digital Health Twin project across four areas:
> data & model versioning, pipeline orchestration, deployment, and monitoring & retraining.
>
> **Legend:**
> - ✅ Implemented — exists in the codebase
> - 📄 Designed — architecture decision documented; not deployed (as per task scope)

---

## 1. Data & Model Versioning

### 1.1 Data Versioning — DVC ✅

All data artifacts are tracked with DVC and defined as a reproducible DAG in `dvc.yaml`.

```
dvc.yaml stages:
  generate    → data/raw/patients.csv
  preprocess  → data/processed/patients_processed.csv
  featurize   → data/processed/scaler.pkl
  train       → data/processed/xgb_model.pkl + metrics.json
```

```bash
dvc repro          # reproduce only changed stages
dvc dag            # visualise the pipeline graph
dvc push           # push artifacts to remote storage   [📄 S3 remote — designed]
dvc pull           # restore artifacts from remote
```

Rolling back data to any previous version:
```bash
git checkout <commit-hash> data/raw/patients.csv.dvc
dvc pull
```

### 1.2 Model Versioning — MLflow ✅

Every training run in `train_mlflow.py` logs the following to MLflow:

| Category | What is logged |
|----------|---------------|
| Parameters | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` |
| Metrics | accuracy, precision, recall, roc_auc |
| Artifacts | `xgb_model.pkl`, `scaler.pkl`, confusion matrix |
| Registry | model registered as `HealthRiskXGB` → Staging → Production |

```bash
mlflow ui --host 0.0.0.0 --port 5000    # browse experiment runs
```

Model promotion workflow:
```
New training run
    │
    ▼
Compare AUC vs current champion
    │
    ├── AUC better → promote to "Production" in MLflow Registry
    │
    └── AUC worse  → keep in "Staging", alert team
```

---

## 2. Pipeline Orchestration

### 2.1 Current — DVC ✅

The pipeline is currently orchestrated through DVC stages. Running `dvc repro` executes only the stages whose dependencies have changed, making local development and CI fast.

```bash
dvc repro        # smart re-execution of changed stages only
```

### 2.2 Production Design — Prefect 3 📄

For production, the pipeline would be managed by **Prefect 3** (Python-native, lightweight, easy to self-host).

```python
# Designed weekly retrain flow

@flow(name="weekly-retrain")
def retrain_flow():
    pull_new_data()          # ingest from EHR / FHIR API
    validate_schema()        # Great Expectations checks
    preprocess()             # src/pipeline/preprocess.py
    feature_engineer()       # src/pipeline/feature_engineering.py
    train_model()            # XGBoost + MLflow logging
    evaluate_model()         # compare AUC vs champion
    promote_if_better()      # MLflow Model Registry update
    rebuild_rag_index()      # re-embed + rebuild FAISS index
    restart_api()            # rolling ECS service update
```

Trigger conditions:
- Scheduled: every Monday at 02:00 — `cron("0 2 * * 1")`
- Data threshold: more than 500 new records accumulated
- Drift alert: Evidently AI triggers retraining automatically

Alternative for enterprise scale: **Apache Airflow on AWS MWAA**.

---

## 3. Deployment Strategy

### 3.1 Local Development ✅

```bash
# API with hot-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# MLflow tracking UI
mlflow ui --host 0.0.0.0 --port 5000
```

### 3.2 Docker ✅

The project is fully containerised. The `Dockerfile` uses a slim Python base and the `.dockerignore` excludes all data and model artifacts, reducing the image from ~4GB to ~1.8GB. Artifacts are mounted as volumes at runtime.

```bash
# Build and run with docker-compose (API + MLflow server)
docker compose up --build

# Verify
curl http://localhost:8000/health
```

Services defined in `docker-compose.yml`:

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI application |
| `mlflow` | 5000 | MLflow tracking server |

### 3.3 CI/CD — GitHub Actions ✅

Defined in `.github/workflows/ci.yml`. On every push to `main`:

```
Push to main
    │
    ├── Install dependencies
    ├── Run dvc repro (pipeline check)
    ├── Run tests (tests/)
    ├── Build Docker image
    └── [📄 Push to ECR → deploy to ECS — designed]
```

### 3.4 Cloud Architecture — AWS 📄

```
Route 53 (DNS)
    │
    ▼
ALB (Application Load Balancer)
    │
    ▼
ECS Fargate (serverless — no EC2 to manage)
    ├── Task: health-twin-api  (2 vCPU, 4 GB RAM)
    │   └── pulls from ECR on each deployment
    └── Auto Scaling: CPU > 70% → scale out (max 5 tasks)

ECR  →  image registry (health-twin:latest, :v1.0, :v1.1 ...)
S3   →  data & model artifact storage
         ├── s3://health-twin-data/raw/
         ├── s3://health-twin-data/processed/
         └── s3://health-twin-models/registry/

Secrets Manager → GROQ_API_KEY injected at ECS task startup
```

Force a new deployment after pushing a new image:
```bash
aws ecs update-service \
  --cluster health-twin-cluster \
  --service health-twin-api \
  --force-new-deployment
```

### 3.5 Blue/Green Deployment Strategy 📄

```
Blue  (v1.1 — current live, 100% traffic)
Green (v1.2 — new build,    0%   traffic)

1. Deploy v1.2 to Green ECS tasks
2. Run smoke tests against Green target group
3. Pass → shift 100% traffic to Green
4. Fail → Green stays offline, Blue unchanged (zero downtime)
```

---

## 4. Monitoring & Retraining

### 4.1 Monitoring Stack 📄

| Signal | Tool | Alert Threshold |
|--------|------|----------------|
| Data drift (feature distribution) | Evidently AI | PSI > 0.2 on any feature |
| Prediction drift (output shift) | Evidently AI | KL-divergence > 0.1 |
| API latency | CloudWatch | p99 > 500ms over 5 min |
| Error rate | CloudWatch | 5xx rate > 1% over 5 min |
| Model accuracy decay | MLflow + cron | ROC-AUC drop > 3% vs champion |
| RAG retrieval quality | Custom metric | Mean cosine similarity < 0.70 |

Weekly Evidently drift report:
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=production_df)
report.save_html("drift_report.html")
```

### 4.2 Retraining Strategy 📄

```
Weekly cron fires
    │
    ▼
Pull last 7 days of production requests
    │
    ▼
Run Evidently drift check
    │
    ├── No drift → log "healthy", stop
    │
    └── Drift detected → trigger Prefect retrain flow
                │
                ▼
        Train new XGBoost on accumulated data
                │
                ▼
        Evaluate new AUC vs champion
                │
                ├── Worse  → keep champion, alert team
                │
                └── Better → promote to Production
                        │
                        ▼
                Shadow mode (7 days):
                both models run in parallel,
                champion serves users,
                challenger predictions logged only
                        │
                        ▼
                Full cutover after shadow validation ✅
```

---

## Summary

| Area | Tool | Status |
|------|------|--------|
| Data versioning | DVC | ✅ Implemented |
| Model versioning | MLflow | ✅ Implemented |
| Experiment tracking | MLflow UI | ✅ Implemented |
| Containerisation | Docker + Compose | ✅ Implemented |
| CI/CD | GitHub Actions | ✅ Implemented |
| Pipeline orchestration | Prefect 3 | 📄 Designed |
| Cloud deployment | AWS ECS Fargate + ECR + S3 | 📄 Designed |
| Data drift monitoring | Evidently AI | 📄 Designed |
| API monitoring | CloudWatch | 📄 Designed |
| Automated retraining | Prefect + Evidently | 📄 Designed |