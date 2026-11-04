import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert "model_family" in data
    assert data["feature_count"] == 30
    assert "test_set_metrics" in data


def test_stats_without_redis(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["predictions_served"] == 0


def test_predict_valid_sample(client):
    # Sample from dataset
    sample = {
        "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0,
        "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471, "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589, "area_error": 153.4,
        "smoothness_error": 0.006399, "compactness_error": 0.04904, "concavity_error": 0.05373,
        "concave_points_error": 0.01587, "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
        "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    res = response.json()
    assert res["label"] in ["malignant", "benign"]
    assert 0.0 <= res["malignant_probability"] <= 1.0
    assert res["threshold"] == 0.5
    assert res["model_version"] == "1.0.0"


def test_predict_validation_errors(client):
    # Missing fields should return 422
    response = client.post("/predict", json={"mean_radius": 17.99})
    assert response.status_code == 422

    # Negative values should return 422
    sample = {
        "mean_radius": -5.0, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0,
        "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471, "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589, "area_error": 153.4,
        "smoothness_error": 0.006399, "compactness_error": 0.04904, "concavity_error": 0.05373,
        "concave_points_error": 0.01587, "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
        "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
    }
    response2 = client.post("/predict", json=sample)
    assert response2.status_code == 422
