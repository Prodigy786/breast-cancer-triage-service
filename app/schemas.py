from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class PredictionInput(BaseModel):
    mean_radius: float = Field(..., ge=0.0)
    mean_texture: float = Field(..., ge=0.0)
    mean_perimeter: float = Field(..., ge=0.0)
    mean_area: float = Field(..., ge=0.0)
    mean_smoothness: float = Field(..., ge=0.0)
    mean_compactness: float = Field(..., ge=0.0)
    mean_concavity: float = Field(..., ge=0.0)
    mean_concave_points: float = Field(..., ge=0.0)
    mean_symmetry: float = Field(..., ge=0.0)
    mean_fractal_dimension: float = Field(..., ge=0.0)
    radius_error: float = Field(..., ge=0.0)
    texture_error: float = Field(..., ge=0.0)
    perimeter_error: float = Field(..., ge=0.0)
    area_error: float = Field(..., ge=0.0)
    smoothness_error: float = Field(..., ge=0.0)
    compactness_error: float = Field(..., ge=0.0)
    concavity_error: float = Field(..., ge=0.0)
    concave_points_error: float = Field(..., ge=0.0)
    symmetry_error: float = Field(..., ge=0.0)
    fractal_dimension_error: float = Field(..., ge=0.0)
    worst_radius: float = Field(..., ge=0.0)
    worst_texture: float = Field(..., ge=0.0)
    worst_perimeter: float = Field(..., ge=0.0)
    worst_area: float = Field(..., ge=0.0)
    worst_smoothness: float = Field(..., ge=0.0)
    worst_compactness: float = Field(..., ge=0.0)
    worst_concavity: float = Field(..., ge=0.0)
    worst_concave_points: float = Field(..., ge=0.0)
    worst_symmetry: float = Field(..., ge=0.0)
    worst_fractal_dimension: float = Field(..., ge=0.0)


class PredictionResponse(BaseModel):
    label: str
    malignant_probability: float
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_family: str
    scikit_learn_version: str
    git_commit: str
    decision_threshold: float
    feature_count: int
    test_set_metrics: Dict[str, Any]


class StatsResponse(BaseModel):
    predictions_served: int
