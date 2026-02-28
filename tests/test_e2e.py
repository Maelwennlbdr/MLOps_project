from fastapi.testclient import TestClient
import sys
import os

# Ajoute backend au path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_path)

from backend.app.main import app

client = TestClient(app)


def test_e2e_predict():
    """
    Test end-to-end de la prédiction :
    - appelle /predict
    - vérifie le status code
    - vérifie la présence de prediction et probability
    - vérifie les types
    """
    sample = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data

    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)

    # optionnel : probabilité entre 0 et 1
    assert 0.0 <= data["probability"] <= 1.0
