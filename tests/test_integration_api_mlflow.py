from fastapi.testclient import TestClient
import numpy as np
import sys
import os

# Ajoute backend au path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_path)

from app.main import app
from app.predict import predict_diabetes

client = TestClient(app)


def test_api_matches_model():
    """
    Integration test :
    - appel API
    - résultat cohérent avec le modèle direct
    """
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

    # appel API
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    api_data = response.json()

    # appel direct du modèle (integration)
    data_np = np.array([[
        sample["Pregnancies"],
        sample["Glucose"],
        sample["BloodPressure"],
        sample["SkinThickness"],
        sample["Insulin"],
        sample["BMI"],
        sample["DiabetesPedigreeFunction"],
        sample["Age"]
    ]])

    model_pred, model_prob = predict_diabetes(data_np)

    assert api_data["prediction"] == model_pred
    assert abs(api_data["probability"] - model_prob) < 1e-6