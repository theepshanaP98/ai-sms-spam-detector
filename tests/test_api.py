from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "AI SMS Spam Detector"


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI SMS Spam Detector" in response.text


def test_empty_prediction_rejected_by_validation():
    response = client.post("/api/predict", json={"message": ""})
    assert response.status_code == 422
