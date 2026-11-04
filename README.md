# Breast Cancer Triage Prediction Service

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/data-DVC-9cf.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg)](https://www.docker.com/)

Production-grade ML prediction service developed for **Meridian Diagnostics Lab**. The service screens fine-needle aspirate cell-nucleus measurements, classifying samples as either **malignant** (positive class) or **benign** (negative class) to assist pathologists in prioritizing urgent reviews.

---

## Repository Structure

```
breast-cancer-triage-service/
├── .dvc/                       # DVC internal configuration
├── app/                        # FastAPI application code
│   ├── __init__.py
│   ├── main.py                 # API service implementation & routes
│   └── schemas.py              # Pydantic schemas for strict validation
├── training/                   # Model training and data preparation
│   ├── __init__.py
│   ├── prepare_data.py         # Dataset export & validation
│   └── train.py                # Reproducible training & model selection
├── data/
│   └── raw/
│       └── breast_cancer.csv.dvc # DVC data pointer
├── dvc-remote/                 # Committed local DVC storage remote
├── dvc.yaml                    # DVC pipeline declaration
├── dvc.lock                    # DVC pipeline lockfile
├── params.yaml                 # Pinned hyperparameters & seeds
├── Dockerfile                  # Production multi-stage Dockerfile
├── Dockerfile.naive            # Baseline single-stage Dockerfile
├── .dockerignore               # Context exclusion rules
├── docker-compose.yml          # Multi-container orchestration (API + Redis)
├── docker-compose.hub.yml      # Clean-room verification compose stack
├── .env.example                # Documented configuration template
├── .gitignore                  # Git untracked patterns
├── .dvcignore                  # DVC exclusion rules
├── requirements-train.txt      # Exact pinned training dependencies
├── requirements-serve.txt      # Exact pinned serving dependencies
├── requirements-dev.txt        # Local developer tooling (DVC, pytest, httpx)
└── README.md                   # Operations and run documentation
```

---

## 1. Quickstart & Local Environment Setup

### Prerequisites
* Python 3.12+
* Git & Docker Desktop (WSL2 backend on Windows or Docker Engine on Linux)

### Setup Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-train.txt -r requirements-serve.txt -r requirements-dev.txt
```

---

## 2. Restore Dataset with DVC

The dataset is versioned using DVC and backed up in the committed `dvc-remote/` folder:

```bash
dvc pull
```

To verify the dataset row count (Version 2 uses 539 samples):
```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/breast_cancer.csv'); print('Row count:', len(df))"
```

---

## 3. Run the Reproducible Training Pipeline

Execute the DVC pipeline:
```bash
dvc repro
```

This runs `training/train.py`, performs a stratified 80/20 train/test split (seed 42), performs 5-fold cross-validation across `LogisticRegression` and `RandomForestClassifier`, enforces a test malignant recall gate ($\ge 0.95$), and produces:
* `artifacts/model.joblib`
* `artifacts/model_card.json`
* `metrics.json`

---

## 4. Run API Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 5. Build Docker Images

### Baseline Image (Naive)
```bash
docker build -t naive -f Dockerfile.naive .
```

### Production Multi-Stage Image
```bash
docker build \
  --build-arg MODEL_VERSION=1.0.0 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t <your-dockerhub-username>/breast-cancer-triage-api:1.0.0 \
  -t <your-dockerhub-username>/breast-cancer-triage-api:latest \
  -f Dockerfile .
```

---

## 6. Multi-Container Stack (Docker Compose)

Configure `.env`:
```bash
cp .env.example .env
```

Start the multi-container stack (API + Redis):
```bash
docker compose up -d --build
```

Check health status:
```bash
docker compose ps
```

Query endpoints:
* **Health Check:**
  ```bash
  curl http://localhost:8000/health
  ```
* **Model Info:**
  ```bash
  curl http://localhost:8000/model/info
  ```
* **Stats Counter:**
  ```bash
  curl http://localhost:8000/stats
  ```
* **Sample Prediction:**
  ```bash
  python -c "from sklearn.datasets import load_breast_cancer as l; import json, urllib.request; d=l(); payload=json.dumps(dict(zip([n.replace(' ','_') for n in d.feature_names], map(float, d.data[0])))).encode('utf-8'); req=urllib.request.Request('http://localhost:8000/predict', data=payload, headers={'Content-Type': 'application/json'}); print(urllib.request.urlopen(req).read().decode('utf-8'))"
  ```

Stop the stack (preserving Redis volume):
```bash
docker compose down
```

Stop and destroy volumes:
```bash
docker compose down -v
```

---

## 7. Push to Docker Hub

```bash
docker login
docker push <your-dockerhub-username>/breast-cancer-triage-api:1.0.0
docker push <your-dockerhub-username>/breast-cancer-triage-api:latest
```

---

## 8. Clean-Room Verification

In a standalone directory containing only `docker-compose.hub.yml` and `.env`:
```bash
docker compose -f docker-compose.hub.yml up -d
```
