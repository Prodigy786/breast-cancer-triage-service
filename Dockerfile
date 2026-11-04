# ==========================================================
# Stage 1: Trainer - Trains the model using DVC data
# ==========================================================
FROM python:3.12-slim AS trainer

WORKDIR /build

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY training/ training/
COPY data/ data/
COPY params.yaml .

ARG MODEL_VERSION=1.0.0
RUN python training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version ${MODEL_VERSION}

# ==========================================================
# Stage 2: Runtime - Minimal, secure serving environment
# ==========================================================
FROM python:3.12-slim AS runtime

ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

LABEL org.opencontainers.image.title="breast-cancer-triage-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      maintainer="Meridian Diagnostics Lab"

# Sensible environment variable defaults
ENV MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA} \
    MODEL_PATH=/app/artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    REDIS_HOST="" \
    REDIS_PORT=6379 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cache layer: Install serving dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Non-root security user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Copy application source code
COPY app/ /app/app/

# Copy only model artifacts from builder stage (no dataset, no training packages)
COPY --from=trainer /build/artifacts/ /app/artifacts/

# Fix ownership
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
