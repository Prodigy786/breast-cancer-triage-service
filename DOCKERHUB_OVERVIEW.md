# Breast Cancer Screening Triage Service API

A production-ready machine learning prediction microservice that classifies cell-nucleus measurements from fine-needle aspirate (FNA) biopsy samples as either **malignant** (positive class) or **benign** (negative class). Developed for **Meridian Diagnostics Lab** to prioritize high-risk pathology review queues.

---

## Available Image Tags

* `1.0.0` - Production release tagged to Git commit revision
* `latest` - Latest stable production release (identical digest to `1.0.0`)

---

## Quickstart

### Run with Docker CLI
```bash
docker run -d \
  --name breast-cancer-api \
  -p 8000:8000 \
  -e DECISION_THRESHOLD=0.5 \
  -e LOG_LEVEL=INFO \
  <your-dockerhub-username>/breast-cancer-triage-api:latest
```

### Run with Docker Compose
```yaml
services:
  redis:
    image: redis:7.2-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  api:
    image: <your-dockerhub-username>/breast-cancer-triage-api:1.0.0
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: /app/artifacts/model.joblib
      DECISION_THRESHOLD: 0.5
      LOG_LEVEL: INFO
      REDIS_HOST: redis
      REDIS_PORT: 6379
    depends_on:
      redis:
        condition: service_healthy

volumes:
  redis_data:
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `/app/artifacts/model.joblib` | Filesystem location of serialized pipeline inside container |
| `DECISION_THRESHOLD` | `0.5` | Malignant probability threshold for positive classification |
| `LOG_LEVEL` | `INFO` | Standard logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `REDIS_HOST` | `""` (empty) | Hostname for Redis hit counter; feature disabled if omitted |
| `REDIS_PORT` | `6379` | Port for Redis connection |
| `MODEL_VERSION` | `1.0.0` | Injected model version tag |
| `GIT_SHA` | `unknown` | Commit revision identifier |

---

## Exposed Ports & Endpoints

Exposed Container Port: `8000`

* `GET /health` - Health check status (`200 OK`)
* `GET /model/info` - Model family, scikit-learn version, git commit, threshold, test-set metrics
* `GET /stats` - Total count of predictions served (tracked via Redis)
* `POST /predict` - Accepts 30 non-negative features (`mean_radius` to `worst_fractal_dimension`), returns malignant probability and class label.
