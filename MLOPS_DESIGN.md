# MLOps Design — Digital Health Twin

## 1. Data & Model Versioning

| Layer        | Tool      | Approach                                      |
|--------------|-----------|-----------------------------------------------|
| Raw data     | DVC       | dvc add data/raw/patients.csv                 |
| Processed    | DVC       | Pipeline stages tracked in dvc.yaml           |
| Models       | MLflow    | mlflow.xgboost.log_model() per run            |
| Experiments  | MLflow UI | Parameters, metrics, artifacts per run        |

```bash
# Developer workflow
dvc repro          # reproduce entire pipeline
dvc push           # push data to S3 remote
mlflow ui          # browse experiment registry
```

## 2. Pipeline Orchestration

Preferred: **Prefect 3** (lightweight, Python-native)
Alternative: Apache Airflow for enterprise scale

```
DAG: weekly-retrain
  ├─ ingest_new_data        (pull from EHR / FHIR API)
  ├─ validate_schema        (Great Expectations)
  ├─ preprocess + featurize (src/pipeline/*)
  ├─ train_xgboost          (src/models/advanced.py)
  ├─ evaluate_metrics       (compare vs champion)
  └─ promote_if_better      (MLflow Model Registry → Production)
```

Trigger: cron weekly  OR  data-volume threshold exceeded.

## 3. Deployment Strategy

### Container
```dockerfile
# Multi-stage build (Dockerfile)
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]
```

### Cloud (AWS)
```
ECR (image registry)
  └─ ECS Fargate (serverless containers — no EC2 to manage)
       ├─ ALB (Application Load Balancer)
       └─ Auto-scaling: CPU > 70% → scale out
```

Alternative: AWS Lambda + API Gateway for low-traffic / cold start.

### Config
- Secrets via AWS Secrets Manager (GROQ_API_KEY, DB creds)
- Environment variables injected at task-definition level
- Blue/green deployments via ECS rolling update

## 4. Monitoring & Retraining

| Signal              | Tool              | Threshold             |
|---------------------|-------------------|-----------------------|
| Data drift          | Evidently AI      | PSI > 0.2 any feature |
| Prediction drift    | Evidently AI      | Output dist shift     |
| API latency         | CloudWatch        | p99 > 500ms           |
| Error rate          | CloudWatch        | 5xx rate > 1%         |
| Model accuracy      | MLflow + cron job | AUC drop > 3%         |

**Retraining trigger** → Prefect flow automatically queued.
**Shadow mode** → new model runs in parallel for 1 week before promotion.