import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
import redis

from app.schemas import (
    PredictionInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    StatsResponse,
)

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "development")

# Configure logging to standard output only
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("breast-cancer-triage")

# Global variables for loaded model and metadata
model = None
model_card = {}
redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    global redis_client
    if not REDIS_HOST:
        return None
    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            redis_client.ping()
            logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            logger.warning("Redis connection failed (%s). Continuing without Redis.", e)
            redis_client = None
    return redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_card
    logger.info("Initializing Breast Cancer Triage Service...")
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Decision threshold: %s", DECISION_THRESHOLD)
    logger.info("Log level: %s", LOG_LEVEL)

    # Fail fast if model artifact is missing
    if not os.path.exists(MODEL_PATH):
        logger.error("CRITICAL: Model artifact not found at '%s'. Failing fast.", MODEL_PATH)
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Successfully loaded model artifact from %s", MODEL_PATH)
    except Exception as e:
        logger.error("CRITICAL: Failed to deserialize model artifact: %s", e)
        raise RuntimeError(f"Failed to load model: {e}")

    # Load model card if available
    card_path = os.path.join(os.path.dirname(MODEL_PATH), "model_card.json")
    if os.path.exists(card_path):
        try:
            with open(card_path, "r") as f:
                model_card = json.load(f)
            logger.info("Loaded model card from %s", card_path)
        except Exception as e:
            logger.warning("Could not load model card from %s: %s", card_path, e)
    else:
        logger.warning("No model card found at %s. Using default metadata.", card_path)

    # Test Redis if configured
    if REDIS_HOST:
        get_redis_client()

    yield

    logger.info("Shutting down Breast Cancer Triage Service...")


app = FastAPI(
    title="Breast Cancer Screening Triage Service",
    description="Classifies cell-nucleus measurements into malignant or benign for pathologist prioritization.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint. Returns 200 independently of Redis."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: PredictionInput):
    """
    Accepts 30 feature measurements (>= 0), returns malignant probability and classification.
    Increments Redis prediction counter if Redis is configured.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    # Feature input dictionary
    data_dict = payload.model_dump()
    ordered_features = model_card.get("ordered_features", list(data_dict.keys()))

    try:
        df_input = pd.DataFrame([data_dict])[ordered_features]
        # In our trained model, class 1 is malignant (positive class)
        probs = model.predict_proba(df_input)
        malignant_prob = float(probs[0, 1])
    except Exception as e:
        logger.error("Inference execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    is_malignant = malignant_prob >= DECISION_THRESHOLD
    label = "malignant" if is_malignant else "benign"

    # Increment Redis counter if enabled (fail-safe)
    if REDIS_HOST:
        try:
            r = get_redis_client()
            if r is not None:
                r.incr("predictions_served")
        except Exception as e:
            logger.warning("Failed to increment Redis hit counter: %s", e)

    return PredictionResponse(
        label=label,
        malignant_probability=round(malignant_prob, 4),
        threshold=DECISION_THRESHOLD,
        model_version=MODEL_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Metadata"])
def model_info():
    """Returns model metadata, versions, threshold, and training metrics."""
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_family=model_card.get("model_family", "LogisticRegression"),
        scikit_learn_version=model_card.get("scikit_learn_version", "1.5.2"),
        git_commit=GIT_SHA,
        decision_threshold=DECISION_THRESHOLD,
        feature_count=model_card.get("feature_count", 30),
        test_set_metrics=model_card.get("test_metrics", {}),
    )


@app.get("/stats", response_model=StatsResponse, tags=["Metrics"])
def get_stats():
    """Returns total predictions served from Redis, or 0 if Redis is disabled/unreachable."""
    count = 0
    if REDIS_HOST:
        try:
            r = get_redis_client()
            if r is not None:
                val = r.get("predictions_served")
                count = int(val) if val else 0
        except Exception as e:
            logger.warning("Failed to retrieve Redis counter: %s", e)
            count = 0
    return StatsResponse(predictions_served=count)
