from fastapi.testclient import TestClient
import sys
import os

# Ajoute backend au path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_path)

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "OK"}

def test_predict():
    sample = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    res = client.post("/predict", json=sample)
    assert res.status_code == 200
    assert "prediction" in res.json()
    assert "probability" in res.json()
