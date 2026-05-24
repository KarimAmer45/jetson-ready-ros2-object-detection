from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_reports_loaded_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "baseline-linear-v1"}


def test_predict_returns_expected_shape_and_risk() -> None:
    response = client.post(
        "/predict",
        json={
            "spindle_speed": 8000,
            "feed_rate": 500,
            "depth_of_cut": 1.5,
            "vibration_features": [0.16, 0.19, 0.24, 0.21, 0.18, 0.27],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["surface_quality_risk"] == "medium"
    assert 0.14 <= payload["predicted_tool_wear"] <= 0.22
    assert 0.7 <= payload["confidence"] <= 0.95


def test_predict_rejects_empty_vibration_features() -> None:
    response = client.post(
        "/predict",
        json={
            "spindle_speed": 8000,
            "feed_rate": 500,
            "depth_of_cut": 1.5,
            "vibration_features": [],
        },
    )

    assert response.status_code == 422
